from collections import deque

import numpy as np
from omegaconf import OmegaConf

import holomotion.src.evaluation.eval_mujoco_sim2sim as eval_mujoco_sim2sim


def test_action_delay_cfg_defaults_to_disabled_episode():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create({})

    max_delay_step, delay_type = evaluator._get_action_delay_cfg()

    assert max_delay_step == 0
    assert delay_type == "episode"


def test_action_delay_cfg_rejects_invalid_delay_type():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create(
        {
            "policy_action_delay_step": 2,
            "action_delay_type": "frame",
        }
    )

    try:
        evaluator._get_action_delay_cfg()
    except ValueError as exc:
        assert "action_delay_type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid delay type.")


def test_apply_action_delay_passthrough_when_disabled():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.policy_action_delay_step = 0
    evaluator.action_delay_type = "episode"
    evaluator._policy_action_delay_buffer = deque(maxlen=1)
    evaluator._current_policy_action_delay_step = 0

    delayed = evaluator._apply_action_delay(
        np.array([1.0, -1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        delayed, np.array([1.0, -1.0], dtype=np.float32)
    )


def test_apply_action_delay_episode_reuses_single_sample(monkeypatch):
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.policy_action_delay_step = 1
    evaluator.action_delay_type = "episode"

    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 1

    monkeypatch.setattr(eval_mujoco_sim2sim.np.random, "randint", fake_randint)

    evaluator._reset_action_delay_randomization()

    first = evaluator._apply_action_delay(np.array([1.0], dtype=np.float32))
    second = evaluator._apply_action_delay(np.array([2.0], dtype=np.float32))
    third = evaluator._apply_action_delay(np.array([3.0], dtype=np.float32))

    assert calls == [(0, 2)]
    assert evaluator._current_policy_action_delay_step == 1
    np.testing.assert_allclose(first, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(third, np.array([2.0], dtype=np.float32))


def test_apply_action_delay_step_resamples_each_policy_step(monkeypatch):
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.policy_action_delay_step = 2
    evaluator.action_delay_type = "step"
    evaluator._policy_action_delay_buffer = deque(maxlen=3)
    evaluator._current_policy_action_delay_step = 0

    sampled_delays = iter([2, 0, 1])
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return next(sampled_delays)

    monkeypatch.setattr(eval_mujoco_sim2sim.np.random, "randint", fake_randint)

    first = evaluator._apply_action_delay(np.array([1.0], dtype=np.float32))
    second = evaluator._apply_action_delay(np.array([2.0], dtype=np.float32))
    third = evaluator._apply_action_delay(np.array([3.0], dtype=np.float32))

    assert calls == [(0, 3), (0, 3), (0, 3)]
    assert evaluator._current_policy_action_delay_step == 1
    np.testing.assert_allclose(first, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([2.0], dtype=np.float32))
    np.testing.assert_allclose(third, np.array([2.0], dtype=np.float32))
