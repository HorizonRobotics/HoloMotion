"""Native opponents keep their input contract and independent causal state."""

import pytest
import torch
from torch import nn

from holoboxing.training.native_robostriker import (
    NativeRoboController,
    NativeRoboStriker,
    native_task_observation,
)
from holoboxing.training.tasks.combat.config.sandbag import BODY_NAMES


def native_checkpoint():
    torch.manual_seed(13)
    state = {}
    for prefix, inputs, outputs in (
        ("taskpolicy_ego", 192, 32),
        ("prior", 900, 64),
        ("decoder", 122, 29),
    ):
        network = nn.Sequential(
            nn.Linear(inputs, 8), nn.ELU(), nn.Linear(8, outputs)
        )
        state.update(
            {
                prefix + "." + key: value
                for key, value in network.state_dict().items()
            }
        )
    checkpoint = dict(model_state_dict=state)
    for name, width in (
        ("taskpolicy_obs_normalizer_ego", 192),
        ("prior_obs_normalizer", 900),
        ("decoder_obs_normalizer", 90),
    ):
        checkpoint[name + "_state_dict"] = {
            "_mean": torch.randn(1, width),
            "_std": torch.rand(1, width),
        }
    return checkpoint


def paired_state():
    position = torch.zeros(1, 2, 30, 3)
    position[:, 0, :, 0] = -0.5
    position[:, 1, :, 0] = 0.5
    position[..., 2] = 0.8
    quaternion = torch.zeros(1, 2, 30, 4)
    quaternion[:, 0, :, 0] = 1
    quaternion[:, 1, :, 3] = 1  # Facing the other fighter: yaw pi.
    return (
        position,
        quaternion,
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 29),
        torch.zeros(1, 2, 29),
    )


def test_native_observation_has_role_canonical_fist_targets():
    position, quaternion, *_ = paired_state()
    observation = native_task_observation(
        position, quaternion, torch.zeros(1, 2, 90)
    )
    assert observation.shape == (1, 2, 192)
    torch.testing.assert_close(observation[0, 0], observation[0, 1])
    # Both own fists are behind the opponent torso, in each fighter's frame.
    torch.testing.assert_close(
        observation[0, :, -12:],
        torch.tensor([[-1.0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0]]).expand(2, -1),
    )
    position[:, 1, BODY_NAMES.index("left_wrist_yaw_link"), 1] += 0.2
    changed = native_task_observation(
        position, quaternion, torch.zeros(1, 2, 90)
    )
    assert changed[0, 0, -5] == pytest.approx(0.2)


def test_native_history_last_action_and_round_reset_are_independent():
    policy = NativeRoboStriker(native_checkpoint())
    controllers = [NativeRoboController(policy) for _ in range(2)]
    state = paired_state()
    first = [
        controller.act(*state, side)[0].clone()
        for side, controller in enumerate(controllers)
    ]
    for side, controller in enumerate(controllers):
        torch.testing.assert_close(
            controller.history, controller.history[:, :1].expand(-1, 10, -1)
        )
        assert torch.count_nonzero(controller.last_inputs[-1][:, -29:]) == 0
        assert not any(
            parameter.requires_grad for parameter in policy.parameters()
        )
        assert torch.isfinite(first[side]).all()
    saved_opponent = controllers[1].history.clone()
    controllers[0].act(*state, 0)
    torch.testing.assert_close(
        controllers[0].last_inputs[-1][:, -29:], first[0]
    )
    assert torch.equal(saved_opponent, controllers[1].history)
    controllers[0].reset()
    action, _ = controllers[0].act(*state, 0)
    torch.testing.assert_close(action, first[0])


def test_native_normalization_keeps_epsilon_outside_standard_deviation():
    checkpoint = native_checkpoint()
    checkpoint["decoder_obs_normalizer_state_dict"]["_std"].fill_(0.1)
    policy = NativeRoboStriker(checkpoint)
    mean = policy.decoder_norm.mean
    torch.testing.assert_close(
        policy.decoder_norm(mean + 1), torch.full_like(mean, 1 / 0.11)
    )
