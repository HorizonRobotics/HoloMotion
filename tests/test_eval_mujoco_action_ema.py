import numpy as np
from omegaconf import OmegaConf

import holomotion.src.evaluation.eval_mujoco_sim2sim as eval_mujoco_sim2sim


def test_action_ema_filter_cfg_reads_erfi_settings():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create(
        {
            "robot": {
                "actuators": {
                    "actuator_type": "unitree_erfi",
                    "ema_filter_enabled": True,
                    "ema_filter_alpha": 0.37,
                }
            },
        }
    )

    enabled, alpha = evaluator._get_action_ema_filter_cfg()

    assert enabled is True
    assert alpha == 0.37


def test_action_ema_filter_defaults_to_disabled_for_non_erfi():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create(
        {
            "robot": {
                "actuators": {
                    "actuator_type": "unitree",
                    "ema_filter_enabled": True,
                    "ema_filter_alpha": 0.37,
                }
            },
        }
    )

    enabled, alpha = evaluator._get_action_ema_filter_cfg()

    assert enabled is False
    assert alpha == 1.0


def test_apply_action_ema_filter_uses_previous_filtered_action():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.action_ema_filter_enabled = True
    evaluator.action_ema_filter_alpha = 0.25
    evaluator._filtered_actions_onnx = None

    first = evaluator._apply_action_ema_filter(
        np.array([1.0, -1.0], dtype=np.float32)
    )
    second = evaluator._apply_action_ema_filter(
        np.array([3.0, 1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(first, np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([1.5, -0.5], dtype=np.float32))


def test_reset_action_ema_filter_clears_state():
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator._filtered_actions_onnx = np.array([1.0], dtype=np.float32)

    evaluator._reset_action_ema_filter()

    assert evaluator._filtered_actions_onnx is None
