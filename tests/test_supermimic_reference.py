"""Contracts for the separate paper-alignment experiment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holomotion.scripts.training.prepare_supermimic_lafan1 import resample_csv
from holomotion.src.algo.supermimic.buffer import ReplayBuffer
from holomotion.src.algo.supermimic.diagnostics import dynamics_diagnostics
from holomotion.src.algo.supermimic.learner import ReferenceLearner
from holomotion.src.algo.supermimic.networks import Critic, Policy, WorldModel
from holomotion.src.env.reference_tracking.mdp import MDP, flatten_state
from holomotion.src.env.reference_tracking.motion import MotionLib
from holomotion.src.evaluation.reference_tracking.evaluation import (
    full_clip_scores,
)


class FixtureKinematics:
    q_lo = torch.full((29,), -2.0)
    q_hi = torch.full((29,), 2.0)

    def forward(self, p, rotation, velocity, angular, q, qd):
        # Fourteen smooth link quantities exercise the reference loss without
        # requiring a simulator or privately hydrated robot meshes.
        count = 14
        local = q[..., :count, None] * torch.tensor([0.1, 0.02, 0.05])
        offset = (rotation[..., None, :, :] @ local.unsqueeze(-1)).squeeze(-1)
        positions = p[..., None, :] + offset
        body_rotation = rotation[..., None, :, :].expand(
            *q.shape[:-1], count, 3, 3
        )
        local_velocity = qd[..., :count, None] * torch.tensor(
            [0.1, 0.02, 0.05]
        )
        body_velocity = velocity[..., None, :] + (
            rotation[..., None, :, :] @ local_velocity.unsqueeze(-1)
        ).squeeze(-1)
        body_velocity = body_velocity + torch.cross(
            angular[..., None, :].expand_as(offset), offset, dim=-1
        )
        body_angular = angular[..., None, :].expand_as(positions)
        return positions, body_rotation, body_velocity, body_angular


def fixture_motion(frames=64):
    kin = FixtureKinematics()
    time = torch.arange(frames) * 0.02
    p = torch.stack((0.1 * time, time * 0, time * 0 + 0.8), -1)
    quat = torch.zeros(frames, 4)
    quat[:, 0] = 1.0
    q = 0.02 * torch.sin(time[:, None] + torch.arange(29)[None, :])
    motion = MotionLib.__new__(MotionLib)
    motion.device, motion.names = "cpu", ["fixture"]
    motion._initialize([p], [quat], [q], [frames], kin, "cpu")
    return kin, motion


def test_reset_bins_never_start_at_a_clip_terminal_frame():
    _, motion = fixture_motion(frames=51)
    assert motion.num_bins == 1
    assert torch.all(motion.bin_start < motion.clip_end[motion.bin_clip])
    assert motion.frame_bin[-1] == motion.frame_bin[-2]


def test_csv_conversion_preserves_duration_and_shortest_quaternion_arc():
    # Endpoints describe identity and 90 degrees about Z. A negative endpoint
    # quaternion must choose the same physical short arc.
    raw = np.zeros((3, 36))
    raw[:, 0] = [0.0, 1.0, 2.0]
    raw[:, 3:7] = [
        [0, 0, 0, 1],
        [0, 0, -np.sin(np.pi / 8), -np.cos(np.pi / 8)],
        [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
    ]
    raw[:, 7:] = np.arange(3)[:, None]
    converted = resample_csv(raw, source_fps=2.0, target_fps=4.0)
    assert converted.shape == (5, 36)
    np.testing.assert_allclose(converted[:, 0], np.arange(5) * 0.5)
    np.testing.assert_allclose(converted[:, 7], np.arange(5) * 0.5)
    np.testing.assert_allclose(
        np.abs(converted[:, 6]), np.cos(np.arange(5) * np.pi / 16), atol=1e-6
    )
    with pytest.raises(ValueError, match="zero norm"):
        raw[1, 3:7] = 0
        resample_csv(raw)


def test_paper_observation_has_no_added_reference_orientation():
    kin, motion = fixture_motion()
    frame = torch.tensor([0, 1])
    state = motion.ref_state(frame)
    xi, previous = torch.zeros(2, 33), torch.zeros(2, 29)
    paper = MDP(kin, motion, torch.zeros(29))
    augmented = MDP(kin, motion, torch.zeros(29), reference_orientation=True)
    first = paper.obs(state, previous, xi, frame)
    second = augmented.obs(state, previous, xi, frame)
    assert first.shape == (2, 93)
    assert second.shape == (2, 99)
    torch.testing.assert_close(first, second[:, :93])
    motion.root_R[:] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(first, paper.obs(state, previous, xi, frame))
    assert not torch.equal(
        second[:, 93:], augmented.obs(state, previous, xi, frame)[:, 93:]
    )


def test_reference_architecture_and_zero_dynamics_initialization():
    policy = Policy(93, 730)
    assert policy.enc[0].in_features == 730
    assert [
        layer.out_features
        for layer in policy.enc
        if isinstance(layer, torch.nn.Linear)
    ] == [2048, 1024, 512, 256, 32]
    assert [
        layer.out_features
        for layer in policy.dec
        if isinstance(layer, torch.nn.Linear)
    ] == [2048, 1024, 512, 256, 29]
    critic = Critic(914)
    assert [
        layer.out_features
        for layer in critic.net
        if isinstance(layer, torch.nn.Linear)
    ] == [256, 128, 1]
    model = WorldModel()
    assert torch.count_nonzero(model.net[-1].weight) == 0
    assert torch.count_nonzero(model.net[-1].bias) == 0


def test_early_failure_keeps_the_unexecuted_tail_in_score_denominator():
    # Both clips have ten planned transitions. One fails after a single
    # perfect-reward transition, so its score is 0.1 rather than 1.0.
    scores = full_clip_scores(torch.tensor([0.1, 1.0]), torch.tensor([10, 10]))
    torch.testing.assert_close(scores, torch.tensor([0.1, 1.0]))
    with pytest.raises(ValueError, match="positive planned"):
        full_clip_scores(torch.tensor([0.1]), torch.tensor([0]))


def test_reference_fit_then_policy_update_keeps_physics_model_frozen():
    torch.manual_seed(4)
    kin, motion = fixture_motion()
    task = MDP(kin, motion, torch.zeros(29))
    replay = ReplayBuffer(16, 2, "cpu")
    for index in range(12):
        frame = torch.tensor([index, index + 1])
        state, next_state = (
            motion.ref_state(frame),
            motion.ref_state(frame + 1),
        )
        replay.add(
            dict(
                s=flatten_state(state),
                s1=flatten_state(next_state),
                a=torch.full((2, 29), 0.01),
                a_prev=torch.zeros(2, 29),
                xi=torch.zeros(2, 33),
                r=torch.ones(2) * 0.1,
                f=frame,
                ep_step=torch.full((2,), index),
                done=torch.zeros(2, dtype=torch.bool),
            )
        )
    args = SimpleNamespace(
        device="cpu",
        goal_frames=2,
        hist=3,
        horizon=2,
        lr=2e-4,
        compile=False,
        wm_window=2,
        wm_batch=2,
        pi_batch=2,
        gamma=0.99,
        lam=0.95,
        no_critic=False,
        critic_minibatches=2,
        critic_epochs=1,
    )
    learner = ReferenceLearner(args, motion, task, replay)
    learner.pi.obs_norm.update(torch.randn(16, 93) * 0.1)
    learner.pi.goal_norm.update(motion.goal(torch.arange(12), 2))
    loss, error = learner.wm_update()
    assert torch.isfinite(loss) and torch.isfinite(error)
    rng = torch.random.get_rng_state().clone()
    diagnostic = dynamics_diagnostics(
        learner, seed=17, batch_size=3, horizon=2
    )
    torch.testing.assert_close(
        torch.random.get_rng_state(), rng, rtol=0, atol=0
    )
    assert (
        diagnostic["available"] and not diagnostic["unseen_physical_holdout"]
    )
    assert diagnostic["steps"][-1]["joint_position_component_rmse_rad"] >= 0
    world_before = {
        name: value.detach().clone()
        for name, value in learner.wm.state_dict().items()
    }
    actor_before = learner.pi.dec[-1].weight.detach().clone()
    values = learner.policy_update()
    assert all(torch.isfinite(value) for value in values)
    assert values[3] > 0
    assert not torch.equal(actor_before, learner.pi.dec[-1].weight)
    for name, value in learner.wm.state_dict().items():
        torch.testing.assert_close(value, world_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("no_critic", [False, True])
def test_imagination_microbatches_keep_noise_gradients_and_one_logical_update(
    no_critic,
):
    import copy

    torch.manual_seed(121)
    kin, motion = fixture_motion()
    task = MDP(kin, motion, torch.zeros(29))
    replay = ReplayBuffer(16, 2, "cpu")
    for index in range(12):
        frame = torch.tensor([index, index + 1])
        replay.add(
            dict(
                s=flatten_state(motion.ref_state(frame)),
                s1=flatten_state(motion.ref_state(frame + 1)),
                a=torch.full((2, 29), 0.01),
                a_prev=torch.zeros(2, 29),
                xi=torch.zeros(2, 33),
                r=torch.ones(2) * 0.1,
                f=frame,
                ep_step=torch.full((2,), index),
                done=torch.zeros(2, dtype=torch.bool),
            )
        )
    args = SimpleNamespace(
        device="cpu",
        goal_frames=2,
        hist=3,
        horizon=2,
        lr=2e-4,
        compile=False,
        wm_window=2,
        wm_batch=2,
        pi_batch=5,
        pi_microbatch=5,
        gamma=0.99,
        lam=0.95,
        no_critic=no_critic,
        critic_minibatches=2,
        critic_epochs=1,
    )
    full = ReferenceLearner(args, motion, task, replay)
    with torch.no_grad():
        full.wm.net[-1].weight.normal_(0, 0.001)
        full.wm.out_norm.var.fill_(0.01)
    # SGD makes this a direct independent check of the accumulated gradients.
    full.opt_pi = torch.optim.SGD(full.pi.parameters(), lr=0.01)
    full.opt_v = torch.optim.SGD(full.critic.parameters(), lr=0.01)
    split = copy.deepcopy(full)
    split.args.pi_microbatch = (
        2  # Uneven final chunk, not a batch-size change.
    )
    before_world = {k: v.clone() for k, v in split.wm.state_dict().items()}
    torch.manual_seed(345)
    expected = full.policy_update()
    rng = torch.random.get_rng_state()
    torch.manual_seed(345)
    actual = split.policy_update()
    assert torch.equal(torch.random.get_rng_state(), rng)
    for a, b in zip(expected, actual, strict=False):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    for left, right in zip(
        full.pi.parameters(), split.pi.parameters(), strict=False
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-4)
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-5)
    for key, value in full.critic.state_dict().items():
        torch.testing.assert_close(
            value, split.critic.state_dict()[key], atol=2e-6, rtol=2e-4
        )
    for key, value in before_world.items():
        assert torch.equal(split.wm.state_dict()[key], value)
    assert all(p.requires_grad for p in split.wm.parameters())
    assert all(p.requires_grad for p in split.critic.parameters())
