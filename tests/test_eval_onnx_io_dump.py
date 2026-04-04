import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.evaluation.eval_mujoco_sim2sim import (
    MujocoEvaluator,
    write_onnx_io_dump_readme,
)
from holomotion.src.evaluation.ray_evaluator_actor import RayEvaluatorActor


class _Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_save_onnx_io_dump_stacks_per_frame_inputs_and_outputs(tmp_path):
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator._reset_onnx_io_dump_buffers()

    evaluator._record_onnx_io_frame(
        input_feed={
            "obs": np.array([[1.0, 2.0]], dtype=np.float32),
            "step": np.array([0], dtype=np.int64),
        },
        output_names=["action", "kv_cache"],
        onnx_output=[
            np.array([[0.1, 0.2]], dtype=np.float32),
            np.array([[[3.0, 4.0]]], dtype=np.float32),
        ],
    )
    evaluator._record_onnx_io_frame(
        input_feed={
            "obs": np.array([[5.0, 6.0]], dtype=np.float32),
            "step": np.array([1], dtype=np.int64),
        },
        output_names=["action", "kv_cache"],
        onnx_output=[
            np.array([[0.3, 0.4]], dtype=np.float32),
            np.array([[[7.0, 8.0]]], dtype=np.float32),
        ],
    )

    output_path = tmp_path / "clip_onnx_io.npy"
    evaluator.save_onnx_io_dump(
        output_path,
        {
            "source_npz": "clip.npz",
            "onnx_model": "model.onnx",
        },
    )

    payload = np.load(output_path, allow_pickle=True).item()

    assert payload["input_names"] == ["obs", "step"]
    assert payload["output_names"] == ["action", "kv_cache"]
    np.testing.assert_allclose(
        payload["inputs"]["obs"],
        np.array([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        payload["inputs"]["step"],
        np.array([[0], [1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        payload["outputs"]["action"],
        np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        payload["outputs"]["kv_cache"],
        np.array([[[[3.0, 4.0]]], [[[7.0, 8.0]]]], dtype=np.float32),
    )
    assert payload["source_npz"] == "clip.npz"
    assert payload["onnx_model"] == "model.onnx"


def test_write_onnx_io_dump_readme_creates_chinese_loading_instructions(
    tmp_path,
):
    readme_path = write_onnx_io_dump_readme(tmp_path)

    assert readme_path == tmp_path / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "每个动作片段会生成一个 `.npy` 文件" in content
    assert "allow_pickle=True" in content
    assert "np.load(npy_path, allow_pickle=True).item()" in content


def test_save_batch_result_persists_low_level_torque_dump_and_dt(tmp_path):
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.policy_dt = 0.02
    evaluator.simulation_dt = 0.005

    evaluator._robot_dof_pos_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_vel_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_acc_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_torque_seq = [np.ones(2, dtype=np.float32)]
    evaluator._robot_low_level_dof_torque_seq = [
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float32),
    ]
    evaluator._robot_action_rate_seq = [np.float32(0.0)]
    evaluator._robot_global_translation_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator._robot_global_rotation_quat_seq = [
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    ]
    evaluator._robot_global_velocity_seq = [np.zeros((1, 3), dtype=np.float32)]
    evaluator._robot_global_angular_velocity_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator.ref_dof_pos = np.zeros((1, 2), dtype=np.float32)
    evaluator.ref_dof_vel = np.zeros((1, 2), dtype=np.float32)
    evaluator.ref_global_translation = np.zeros((1, 1, 3), dtype=np.float32)
    evaluator.ref_global_rotation_quat_xyzw = np.array(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32
    )
    evaluator.ref_global_velocity = np.zeros((1, 1, 3), dtype=np.float32)
    evaluator.ref_global_angular_velocity = np.zeros(
        (1, 1, 3), dtype=np.float32
    )

    output_path = tmp_path / "demo_eval.npz"
    evaluator.save_batch_result(
        output_path, {"source_file": "clip.npz", "clip_length": 1}
    )

    with np.load(output_path, allow_pickle=True) as payload:
        metadata = json.loads(payload["metadata"].item())
        np.testing.assert_allclose(
            payload["robot_low_level_dof_torque"],
            np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32),
        )

    assert metadata["source_file"] == "clip.npz"
    assert metadata["clip_length"] == 1
    assert metadata["robot_low_level_torque_dt"] == 0.005


def test_save_batch_result_persists_moe_routing_tensors(tmp_path):
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.policy_dt = 0.02
    evaluator.simulation_dt = 0.005

    evaluator._robot_dof_pos_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_vel_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_acc_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_torque_seq = [np.ones(2, dtype=np.float32)]
    evaluator._robot_low_level_dof_torque_seq = [
        np.array([0.5, -0.5], dtype=np.float32)
    ]
    evaluator._robot_action_rate_seq = [np.float32(0.0)]
    evaluator._robot_global_translation_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator._robot_global_rotation_quat_seq = [
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    ]
    evaluator._robot_global_velocity_seq = [np.zeros((1, 3), dtype=np.float32)]
    evaluator._robot_global_angular_velocity_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator._robot_moe_expert_indices_seq = [
        np.array([[1, 3], [0, 2]], dtype=np.int64)
    ]
    evaluator._robot_moe_expert_logits_seq = [
        np.array(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]],
            dtype=np.float32,
        )
    ]
    evaluator.ref_dof_pos = np.zeros((1, 2), dtype=np.float32)
    evaluator.ref_dof_vel = np.zeros((1, 2), dtype=np.float32)
    evaluator.ref_global_translation = np.zeros((1, 1, 3), dtype=np.float32)
    evaluator.ref_global_rotation_quat_xyzw = np.array(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32
    )
    evaluator.ref_global_velocity = np.zeros((1, 1, 3), dtype=np.float32)
    evaluator.ref_global_angular_velocity = np.zeros(
        (1, 1, 3), dtype=np.float32
    )

    output_path = tmp_path / "demo_eval_moe.npz"
    evaluator.save_batch_result(output_path, {"source_file": "clip.npz"})

    with np.load(output_path, allow_pickle=True) as payload:
        np.testing.assert_array_equal(
            payload["robot_moe_expert_indices"],
            np.array([[[1, 3], [0, 2]]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            payload["robot_moe_expert_logits"],
            np.array(
                [[[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]]],
                dtype=np.float32,
            ),
        )


def test_dump_robot_augmented_npz_persists_moe_routing_tensors(tmp_path):
    source_npz = tmp_path / "clip.npz"
    np.savez(source_npz, ref=np.array([1], dtype=np.int32))

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")

    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.simulation_dt = 0.005
    evaluator.config = _Config(
        motion_npz_path=str(source_npz),
        ckpt_onnx_path=str(onnx_path),
    )
    evaluator._robot_dof_pos_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_vel_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_acc_seq = [np.zeros(2, dtype=np.float32)]
    evaluator._robot_dof_torque_seq = [np.ones(2, dtype=np.float32)]
    evaluator._robot_low_level_dof_torque_seq = [
        np.array([0.5, -0.5], dtype=np.float32)
    ]
    evaluator._robot_action_rate_seq = [np.float32(0.0)]
    evaluator._robot_global_translation_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator._robot_global_rotation_quat_seq = [
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    ]
    evaluator._robot_global_velocity_seq = [np.zeros((1, 3), dtype=np.float32)]
    evaluator._robot_global_angular_velocity_seq = [
        np.zeros((1, 3), dtype=np.float32)
    ]
    evaluator._robot_moe_expert_indices_seq = [
        np.array([[1, 3], [0, 2]], dtype=np.int64)
    ]
    evaluator._robot_moe_expert_logits_seq = [
        np.array(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]],
            dtype=np.float32,
        )
    ]

    evaluator._dump_robot_augmented_npz()

    out_path = tmp_path / "mujoco_output_model" / "clip_robot.npz"
    with np.load(out_path, allow_pickle=True) as payload:
        np.testing.assert_array_equal(
            payload["robot_moe_expert_indices"],
            np.array([[[1, 3], [0, 2]]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            payload["robot_moe_expert_logits"],
            np.array(
                [[[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]]],
                dtype=np.float32,
            ),
        )


def test_ray_actor_run_clip_overwrites_existing_outputs_and_sidecar(tmp_path):
    class _FakeEvaluator:
        def __init__(self):
            self.n_motion_frames = 2
            self.calls = []
            self.counter = 0

        def load_specific_motion(self, file_path):
            self.calls.append(("load", file_path))

        def reset_state_teleport(self):
            self.calls.append(("reset",))

        def _update_policy(self):
            self.calls.append(("update",))

        def _apply_control(self, sleep=False):
            self.calls.append(("apply", sleep))

        def save_batch_result(self, output_path, meta_info):
            self.calls.append(("save_batch", output_path, meta_info))
            Path(output_path).write_text("fresh-npz", encoding="utf-8")

        def save_onnx_io_dump(self, output_path, meta_info):
            self.calls.append(("save_onnx", output_path, meta_info))
            np.save(
                output_path,
                {"source_npz": meta_info["source_file"]},
                allow_pickle=True,
            )

    actor = RayEvaluatorActor.__new__(RayEvaluatorActor)
    actor.output_dir = str(tmp_path)
    actor.config_dict = {
        "ckpt_onnx_path": "model.onnx",
        "dump_onnx_io_npy": True,
    }
    actor.evaluator = _FakeEvaluator()

    clip_path = tmp_path / "demo_clip.npz"
    np.savez(clip_path, dummy=np.array([1], dtype=np.int32))

    existing_npz = tmp_path / "demo_clip_eval.npz"
    existing_npz.write_text("stale", encoding="utf-8")
    onnx_dir = tmp_path / "onnx_io_npy"
    onnx_dir.mkdir()

    status = actor.run_clip(str(clip_path))

    assert status == "success"
    assert existing_npz.read_text(encoding="utf-8") == "fresh-npz"
    onnx_dump_path = onnx_dir / "demo_clip_onnx_io.npy"
    assert onnx_dump_path.is_file()
    payload = np.load(onnx_dump_path, allow_pickle=True).item()
    assert payload["source_npz"] == "demo_clip.npz"
    assert ("load", str(clip_path)) in actor.evaluator.calls
    assert ("reset",) in actor.evaluator.calls
    assert actor.evaluator.calls.count(("update",)) == 2


def test_ray_actor_skips_sidecar_for_non_default_model_type(tmp_path):
    class _FakeEvaluator:
        def __init__(self):
            self.n_motion_frames = 1
            self.calls = []
            self.counter = 0

        def load_specific_motion(self, file_path):
            self.calls.append(("load", file_path))

        def reset_state_teleport(self):
            self.calls.append(("reset",))

        def _update_policy(self):
            self.calls.append(("update",))

        def _apply_control(self, sleep=False):
            self.calls.append(("apply", sleep))

        def save_batch_result(self, output_path, meta_info):
            self.calls.append(("save_batch", output_path, meta_info))
            Path(output_path).write_text("fresh-npz", encoding="utf-8")

        def save_onnx_io_dump(self, output_path, meta_info):
            self.calls.append(("save_onnx", output_path, meta_info))

    actor = RayEvaluatorActor.__new__(RayEvaluatorActor)
    actor.output_dir = str(tmp_path)
    actor.config_dict = {
        "ckpt_onnx_path": "model.onnx",
        "dump_onnx_io_npy": True,
        "model_type": "gmt",
    }
    actor.evaluator = _FakeEvaluator()

    clip_path = tmp_path / "demo_clip.npz"
    np.savez(clip_path, dummy=np.array([1], dtype=np.int32))

    status = actor.run_clip(str(clip_path))

    assert status == "success"
    assert not any(call[0] == "save_onnx" for call in actor.evaluator.calls)


def test_ray_actor_treats_empty_model_type_as_default_holomotion(tmp_path):
    class _FakeEvaluator:
        def __init__(self):
            self.n_motion_frames = 1
            self.calls = []
            self.counter = 0

        def load_specific_motion(self, file_path):
            self.calls.append(("load", file_path))

        def reset_state_teleport(self):
            self.calls.append(("reset",))

        def _update_policy(self):
            self.calls.append(("update",))

        def _apply_control(self, sleep=False):
            self.calls.append(("apply", sleep))

        def save_batch_result(self, output_path, meta_info):
            self.calls.append(("save_batch", output_path, meta_info))
            Path(output_path).write_text("fresh-npz", encoding="utf-8")

        def save_onnx_io_dump(self, output_path, meta_info):
            self.calls.append(("save_onnx", output_path, meta_info))
            np.save(output_path, {"source_npz": meta_info["source_file"]})

    actor = RayEvaluatorActor.__new__(RayEvaluatorActor)
    actor.output_dir = str(tmp_path)
    actor.config_dict = {
        "ckpt_onnx_path": "model.onnx",
        "dump_onnx_io_npy": True,
        "model_type": "",
    }
    actor.evaluator = _FakeEvaluator()

    clip_path = tmp_path / "demo_clip.npz"
    np.savez(clip_path, dummy=np.array([1], dtype=np.int32))

    status = actor.run_clip(str(clip_path))

    assert status == "success"
    assert any(call[0] == "save_onnx" for call in actor.evaluator.calls)


def test_ray_actor_init_uses_configured_evaluator_module(
    monkeypatch, tmp_path
):
    class _FakeEvaluator:
        def __init__(self):
            self.setup_called = False

        def setup(self):
            self.setup_called = True

    captured = {}
    fake_evaluator = _FakeEvaluator()

    def _unexpected_default_factory(*args, **kwargs):
        raise AssertionError("default evaluator factory should not be used")

    def _fake_override_factory(config_dict, model_type):
        captured["config_dict"] = config_dict
        captured["model_type"] = model_type
        return fake_evaluator

    monkeypatch.setattr(
        "holomotion.src.evaluation.eval_mujoco_sim2sim._create_ray_evaluator",
        _unexpected_default_factory,
    )
    sys.modules["holomotion.src.evaluation.fake_eval_module"] = (
        types.SimpleNamespace(_create_ray_evaluator=_fake_override_factory)
    )

    actor = RayEvaluatorActor(
        {
            "ckpt_onnx_path": "model.onnx",
            "model_type": "holomotion",
            "ray_evaluator_module": "holomotion.src.evaluation.fake_eval_module",
        },
        str(tmp_path),
    )

    assert actor.evaluator is fake_evaluator
    assert fake_evaluator.setup_called is True
    assert captured["model_type"] == "holomotion"
    assert (
        captured["config_dict"]["ray_evaluator_module"]
        == "holomotion.src.evaluation.fake_eval_module"
    )
