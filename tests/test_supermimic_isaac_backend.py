"""Isaac boundary tests independent of simulator installation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from holomotion.src.env.reference_tracking.isaac_backend import (
    IsaacTrackingBackend,
)
from holomotion.src.evaluation.reference_tracking.evaluation import (
    FullClipEvaluator,
)
from holomotion.src.evaluation.reference_tracking.isaac_evaluation import (
    IsaacFullClipEvaluator,
)


def test_isaac_link_state_round_trip_and_joint_order():
    env = object.__new__(IsaacTrackingBackend)
    env.N = 2
    env.device = "cpu"
    env.qpos = torch.zeros(2, 36)
    env.qvel = torch.zeros(2, 35)
    env.qpos[:, :3] = torch.tensor([2.0, 3.0, 4.0])
    env.qpos[:, 3:7] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    env.qpos[:, 7:] = torch.arange(29)
    env.qvel[:, :3] = torch.tensor([5.0, 6.0, 7.0])
    env.qvel[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])
    env.qvel[:, 6:] = torch.arange(29) + 100.0
    env.origins = torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0]])
    env.joint_ids = list(range(28, -1, -1))
    captured = {}

    class Articulation:
        def write_root_link_state_to_sim(self, root, env_ids):
            captured["root"] = root.clone()

        def write_joint_state_to_sim(self, pos, vel, joint_ids, env_ids):
            captured.update(
                pos=pos.clone(), vel=vel.clone(), joint_ids=joint_ids
            )

        def reset(self, ids):
            captured["reset_ids"] = ids.clone()

    env.robot = Articulation()
    env._write_physical_state(torch.tensor([1]))
    torch.testing.assert_close(
        captured["root"][0, :3], torch.tensor([32.0, 43.0, 4.0])
    )
    torch.testing.assert_close(
        captured["root"][0, 7:10], torch.tensor([5.0, 6.0, 7.0])
    )
    torch.testing.assert_close(
        captured["root"][0, 10:],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=2e-7,
        rtol=0,
    )
    assert captured["joint_ids"] == list(range(28, -1, -1))
    expected_qpos, expected_qvel = env.qpos.clone(), env.qvel.clone()
    env.robot.data = SimpleNamespace(
        root_link_pos_w=env.qpos[:, :3] + env.origins,
        root_link_quat_w=env.qpos[:, 3:7].clone(),
        root_link_lin_vel_w=env.qvel[:, :3].clone(),
        root_link_ang_vel_w=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        joint_pos=env.qpos[:, 7:].flip(-1).clone(),
        joint_vel=env.qvel[:, 6:].flip(-1).clone(),
    )
    env._read_physical_state()
    torch.testing.assert_close(env.qpos, expected_qpos)
    torch.testing.assert_close(env.qvel, expected_qvel, atol=2e-7, rtol=0)


def test_common_evaluator_and_frame_sampler_are_shared():
    assert IsaacFullClipEvaluator.evaluate is FullClipEvaluator.evaluate
    directory = (
        Path(__file__).resolve().parents[1]
        / "holomotion/src/env/reference_tracking"
    )

    def method(filename, name):
        tree = ast.parse((directory / filename).read_text())
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    assert ast.dump(method("isaac_backend.py", "sample_frames")) == ast.dump(
        method("warp_backend.py", "sample_frames")
    )


def test_isaac_complete_clip_emits_native_protocol_without_warp_backend():
    motion = SimpleNamespace(
        names=["one_step"],
        clip_start=torch.tensor([0]),
        clip_len=torch.tensor([2]),
        root_p=torch.zeros(2, 3),
        body_p=torch.zeros(2, 1, 3),
        goal=lambda frames, count: torch.zeros(1, count),
    )
    mdp = SimpleNamespace(
        obs=lambda *args: torch.zeros(1, 2),
        kin=SimpleNamespace(forward=lambda *args: (torch.zeros(1, 1, 3),)),
    )
    env = SimpleNamespace(
        N=1,
        device="cpu",
        done_eval=torch.zeros(1, dtype=torch.bool),
        failed_eval=torch.zeros(1, dtype=torch.bool),
        f=torch.tensor([0]),
        a_prev=torch.zeros(1, 1),
        xi=torch.zeros(1, 1),
        reset=lambda *args: None,
        state=lambda: (torch.zeros(1, 3),),
        status_summary=lambda: {"finite": True},
    )

    def step(action):
        env.f.fill_(1)
        env.done_eval.fill_(True)
        return dict(
            s1=torch.zeros(1, 71),
            a=action,
            r=torch.tensor([0.1]),
            term=torch.zeros(1, dtype=torch.bool),
            done=env.done_eval.clone(),
        )

    env.step = step
    evaluator = IsaacFullClipEvaluator(None, motion, mdp, device="cpu")
    evaluator._env = env
    result = evaluator.evaluate(lambda *args: torch.zeros(1, 1))
    assert result["evaluation_mode"] == "native"
    assert result["evaluation_protocol"] == {"mode": "native"}
    assert result["new_evaluation_protocol"] is False
    assert result["baseline_rebuild_required"] is False
    assert result["evaluated_clips"] == 1
    assert result["normalized_return"] == 1.0
    assert len(evaluator.last_trace["next_state"]) == 1


def test_finite_solver_explosion_is_not_an_accepted_transition():
    state = (
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.zeros(1, 29),
        torch.full((1, 29), 70.0),
    )
    assert IsaacTrackingBackend._physical_state_valid(state)
    # Observed solver failure was finite, with a joint near 3e15 radians.
    state[4][0, 0] = 3e15
    assert not IsaacTrackingBackend._physical_state_valid(state)
    state[4][0, 0] = 0.0
    state[5][0, 0] = float("nan")
    assert not IsaacTrackingBackend._physical_state_valid(state)
