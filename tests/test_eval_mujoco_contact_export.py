import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import holomotion.src.evaluation.eval_mujoco_sim2sim as eval_mujoco_sim2sim


def _build_export_evaluator(tmp_path: Path):
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.simulation_dt = 0.005
    evaluator._get_stacked_moe_routing_tensors = lambda: (None, None)
    evaluator._robot_dof_pos_seq = [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.5, 1.5], dtype=np.float32),
    ]
    evaluator._robot_dof_vel_seq = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.3, 0.4], dtype=np.float32),
    ]
    evaluator._robot_dof_acc_seq = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    evaluator._robot_dof_torque_seq = [
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
    ]
    evaluator._robot_low_level_dof_torque_seq = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
    ]
    evaluator._robot_low_level_foot_contact_seq = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ]
    evaluator._robot_low_level_foot_normal_force_seq = [
        np.array([50.0, 0.0], dtype=np.float32),
        np.array([60.0, 55.0], dtype=np.float32),
        np.array([0.0, 45.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ]
    evaluator._robot_low_level_foot_tangent_speed_seq = [
        np.array([0.02, 0.0], dtype=np.float32),
        np.array([0.03, 0.04], dtype=np.float32),
        np.array([0.0, 0.05], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ]
    evaluator._robot_action_rate_seq = [
        np.float32(0.0),
        np.float32(1.0),
    ]
    evaluator._robot_actions_seq = [
        np.array([0.11, 0.22], dtype=np.float32),
        np.array([0.33, 0.44], dtype=np.float32),
    ]
    evaluator._robot_global_translation_seq = [
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ]
    evaluator._robot_global_rotation_quat_seq = [
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
    ]
    evaluator._robot_global_velocity_seq = [
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ]
    evaluator._robot_global_angular_velocity_seq = [
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ]
    evaluator.ref_dof_pos = np.zeros((2, 2), dtype=np.float32)
    evaluator.ref_dof_vel = np.zeros((2, 2), dtype=np.float32)
    evaluator.ref_global_translation = np.zeros((2, 2, 3), dtype=np.float32)
    evaluator.ref_global_rotation_quat_xyzw = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 2, 1)
    )
    evaluator.ref_global_velocity = np.zeros((2, 2, 3), dtype=np.float32)
    evaluator.ref_global_angular_velocity = np.zeros(
        (2, 2, 3), dtype=np.float32
    )

    motion_npz_path = tmp_path / "motion.npz"
    np.savez_compressed(
        motion_npz_path,
        metadata=np.array(json.dumps({"clip_length": 2}), dtype=np.str_),
    )
    evaluator.config = OmegaConf.create(
        {
            "motion_npz_path": str(motion_npz_path),
            "ckpt_onnx_path": str(tmp_path / "model.onnx"),
        }
    )
    return evaluator


def test_save_batch_result_exports_low_level_contact_traces(tmp_path: Path):
    evaluator = _build_export_evaluator(tmp_path)
    output_path = tmp_path / "batch_result.npz"

    evaluator.save_batch_result(str(output_path), {"clip_length": 2})

    with np.load(output_path, allow_pickle=True) as data:
        assert "robot_actions" in data.files
        assert "robot_low_level_foot_contact" in data.files
        assert "robot_low_level_foot_normal_force" in data.files
        assert "robot_low_level_foot_tangent_speed" in data.files
        assert "robot_low_level_contact_dt" in data.files
        np.testing.assert_allclose(
            data["robot_actions"],
            np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32),
        )
        assert data["robot_low_level_foot_contact"].shape == (4, 2)
        np.testing.assert_allclose(
            data["robot_low_level_contact_dt"], np.array(0.005, np.float32)
        )


def test_dump_robot_augmented_npz_exports_low_level_contact_traces(
    tmp_path: Path,
):
    evaluator = _build_export_evaluator(tmp_path)

    evaluator._dump_robot_augmented_npz()

    output_path = (
        tmp_path
        / "mujoco_output_model"
        / f"{Path(evaluator.config.motion_npz_path).stem}_robot.npz"
    )
    with np.load(output_path, allow_pickle=True) as data:
        assert "robot_actions" in data.files
        assert "robot_low_level_foot_contact" in data.files
        assert "robot_low_level_foot_normal_force" in data.files
        assert "robot_low_level_foot_tangent_speed" in data.files
        assert "robot_low_level_contact_dt" in data.files
        np.testing.assert_allclose(
            data["robot_actions"],
            np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32),
        )
        assert data["robot_low_level_foot_normal_force"].shape == (4, 2)


def test_init_low_level_foot_contact_logging_falls_back_to_ankle_roll_bodies(
    monkeypatch,
):
    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create({"robot": {}})
    evaluator.m = type(
        "FakeModel",
        (),
        {
            "geom_bodyid": np.array([5, 6, 6, 9, 10], dtype=np.int32),
            "geom_contype": np.array([0, 1, 1, 0, 1], dtype=np.int32),
            "geom_conaffinity": np.array([0, 1, 1, 0, 1], dtype=np.int32),
        },
    )()

    def fake_name2id(model, obj_type, name):
        if obj_type == eval_mujoco_sim2sim.mujoco.mjtObj.mjOBJ_GEOM:
            return -1
        if obj_type == eval_mujoco_sim2sim.mujoco.mjtObj.mjOBJ_BODY:
            return {
                "left_ankle_roll_link": 6,
                "right_ankle_roll_link": 10,
            }.get(name, -1)
        return -1

    monkeypatch.setattr(eval_mujoco_sim2sim.mujoco, "mj_name2id", fake_name2id)

    evaluator._init_low_level_foot_contact_logging()

    assert evaluator._foot_contact_logging_enabled is True
    assert evaluator._foot_geom_id_groups == [[1, 2], [4]]
    assert evaluator._foot_geom_id_to_side == {1: 0, 2: 0, 4: 1}
