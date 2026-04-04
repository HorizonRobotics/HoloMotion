import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holomotion.src.evaluation.eval_mujoco_sim2sim_s100 as eval_mujoco_sim2sim_s100


class _FakeIoNode:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def _make_value_info(name, shape):
    dims = [SimpleNamespace(dim_value=dim) for dim in shape]
    tensor_shape = SimpleNamespace(dim=dims)
    tensor_type = SimpleNamespace(shape=tensor_shape)
    return SimpleNamespace(
        name=name, type=SimpleNamespace(tensor_type=tensor_type)
    )


def _make_fake_onnx_model():
    return SimpleNamespace(
        graph=SimpleNamespace(
            input=[
                _make_value_info("obs", [1, 16]),
                _make_value_info("past_key_values", [1, 2, 3, 4]),
                _make_value_info("step_idx", [1]),
            ],
            output=[
                _make_value_info("action", [1, 12]),
                _make_value_info("present_key_values", [1, 2, 3, 4]),
            ],
        )
    )


def _make_evaluator(model_path: Path, bc_path: Path | None = None):
    config_dict = {
        "ckpt_onnx_path": str(model_path),
        "use_gpu": False,
        "gpu_id": 0,
    }
    if bc_path is not None:
        config_dict["bc_path"] = str(bc_path)

    evaluator = eval_mujoco_sim2sim_s100.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim_s100.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create(config_dict)
    evaluator.max_context_len = 0
    evaluator._discover_policy_moe_outputs = lambda: None
    return evaluator


def test_load_policy_falls_back_to_horizon_quantized_bc_for_ptq_onnx(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "demo_ptq_model.onnx"
    model_path.write_bytes(b"onnx")
    quantized_path = tmp_path / "demo_quantized_model.bc"
    quantized_path.write_bytes(b"bc")
    captured = {}

    class _FakeHBRuntime:
        def __init__(self, model_path):
            captured["hb_model_path"] = model_path
            self.input_names = ["obs", "past_key_values", "step_idx"]
            self.output_names = ["action", "present_key_values"]

        def run(self, output_names, input_feed):
            raise AssertionError("run should not be called in this test")

    def _raise_hz_calibration(*args, **kwargs):
        raise RuntimeError("Failed to load custom op HzCalibration")

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "InferenceSession",
        _raise_hz_calibration,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100,
        "HBRuntime",
        _FakeHBRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnx,
        "load",
        lambda _: _make_fake_onnx_model(),
    )

    evaluator = _make_evaluator(model_path)

    evaluator.load_policy()

    assert captured["hb_model_path"] == str(quantized_path)
    assert evaluator.policy_input_name == "obs"
    assert evaluator.policy_kv_input_name == "past_key_values"
    assert evaluator.policy_step_input_name == "step_idx"
    assert evaluator.policy_output_name == "action"
    assert evaluator.policy_kv_output_name == "present_key_values"
    assert evaluator.policy_model_context_len == 4


@pytest.mark.parametrize(
    "runtime_name",
    [
        "demo_model_16000_ptq_model.bc",
        "demo_model_16000_ptq_model.hbm",
        "demo_model_16000_quantized_model.hbm",
    ],
)
def test_load_policy_resolves_common_horizon_runtime_artifact_names(
    monkeypatch, tmp_path, runtime_name
):
    model_path = tmp_path / "demo_model_16000_ptq_model.onnx"
    model_path.write_bytes(b"onnx")
    runtime_path = tmp_path / runtime_name
    runtime_path.write_bytes(b"runtime")
    captured = {}

    class _FakeHBRuntime:
        def __init__(self, model_path):
            captured["hb_model_path"] = model_path
            self.input_names = ["obs", "past_key_values", "step_idx"]
            self.output_names = ["action", "present_key_values"]

        def run(self, output_names, input_feed):
            raise AssertionError("run should not be called in this test")

    def _raise_hz_calibration(*args, **kwargs):
        raise RuntimeError("Failed to load custom op HzCalibration")

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "InferenceSession",
        _raise_hz_calibration,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100,
        "HBRuntime",
        _FakeHBRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnx,
        "load",
        lambda _: _make_fake_onnx_model(),
    )

    evaluator = _make_evaluator(model_path)

    evaluator.load_policy()

    assert captured["hb_model_path"] == str(runtime_path)
    assert evaluator.policy_model_context_len == 4


def test_load_policy_raises_original_error_when_ptq_fallback_bc_missing(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "demo_ptq_model.onnx"
    model_path.write_bytes(b"onnx")

    def _raise_hz_calibration(*args, **kwargs):
        raise RuntimeError("Failed to load custom op HzCalibration")

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "InferenceSession",
        _raise_hz_calibration,
    )

    evaluator = _make_evaluator(model_path)

    with pytest.raises(RuntimeError, match="HzCalibration"):
        evaluator.load_policy()


def test_load_policy_keeps_standard_onnxruntime_path_for_regular_onnx(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "demo_model.onnx"
    model_path.write_bytes(b"onnx")
    captured = {}

    class _FakeInferenceSession:
        def __init__(self, model_path, sess_options, providers):
            captured["model_path"] = model_path
            captured["providers"] = providers

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                _FakeIoNode("obs", [1, 16]),
                _FakeIoNode("past_key_values", [1, 2, 3, 4]),
                _FakeIoNode("step_idx", [1]),
            ]

        def get_outputs(self):
            return [
                _FakeIoNode("action", [1, 12]),
                _FakeIoNode("present_key_values", [1, 2, 3, 4]),
            ]

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "InferenceSession",
        _FakeInferenceSession,
    )

    evaluator = _make_evaluator(model_path)

    evaluator.load_policy()

    assert captured["model_path"] == str(model_path)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert evaluator.policy_model_context_len == 4


def test_load_policy_prefers_explicit_bc_path_for_inference_and_onnx_for_metadata(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "demo_model.onnx"
    model_path.write_bytes(b"onnx")
    runtime_path = tmp_path / "demo_quantized_model.bc"
    runtime_path.write_bytes(b"bc")
    captured = {}

    class _FakeHBRuntime:
        def __init__(self, model_path):
            captured["hb_model_path"] = model_path
            self.input_names = ["obs", "past_key_values", "step_idx"]
            self.output_names = ["action", "present_key_values"]

        def run(self, output_names, input_feed):
            raise AssertionError("run should not be called in this test")

    def _unexpected_ort_session(*args, **kwargs):
        raise AssertionError(
            "InferenceSession should not be created when bc_path is set"
        )

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnxruntime,
        "InferenceSession",
        _unexpected_ort_session,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100,
        "HBRuntime",
        _FakeHBRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100.onnx,
        "load",
        lambda _: _make_fake_onnx_model(),
    )

    evaluator = _make_evaluator(model_path, bc_path=runtime_path)

    evaluator.load_policy()

    assert captured["hb_model_path"] == str(runtime_path)
    assert evaluator.policy_input_name == "obs"
    assert evaluator.policy_kv_input_name == "past_key_values"
    assert evaluator.policy_step_input_name == "step_idx"
    assert evaluator.policy_output_name == "action"
    assert evaluator.policy_kv_output_name == "present_key_values"
    assert evaluator.policy_model_context_len == 4


def test_bc_runtime_run_normalizes_inputs_for_hbruntime(monkeypatch, tmp_path):
    runtime_path = tmp_path / "demo_quantized_model.bc"
    runtime_path.write_bytes(b"bc")
    captured = {}

    class _FakeHBRuntime:
        def __init__(self, model_path):
            captured["model_path"] = model_path
            self.input_names = ["obs", "past_key_values", "step_idx"]
            self.output_names = ["action", "present_key_values"]

        def run(self, output_names, input_feed):
            captured["output_names"] = list(output_names)
            captured["input_feed"] = input_feed
            return ["ok"]

    monkeypatch.setattr(
        eval_mujoco_sim2sim_s100,
        "HBRuntime",
        _FakeHBRuntime,
        raising=False,
    )

    wrapper = eval_mujoco_sim2sim_s100._HbSessionWrapper(runtime_path)
    obs = np.arange(6, dtype=np.float64).reshape(2, 3).T
    past_key_values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    step_idx = np.array([7], dtype=np.int32)

    outputs = wrapper.run(
        ["action"],
        {
            "obs": obs,
            "past_key_values": past_key_values,
            "step_idx": step_idx,
        },
    )

    assert outputs == ["ok"]
    assert captured["model_path"] == str(runtime_path)
    assert captured["output_names"] == ["action"]
    assert captured["input_feed"]["obs"].dtype == np.float32
    assert captured["input_feed"]["obs"].flags["C_CONTIGUOUS"]
    assert captured["input_feed"]["past_key_values"].dtype == np.float32
    assert captured["input_feed"]["past_key_values"].flags["C_CONTIGUOUS"]
    assert captured["input_feed"]["step_idx"].dtype == np.int64
    assert captured["input_feed"]["step_idx"].flags["C_CONTIGUOUS"]


def test_update_policy_raises_clear_error_before_runtime_on_obs_dim_mismatch():
    evaluator = eval_mujoco_sim2sim_s100.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim_s100.MujocoEvaluator
    )
    evaluator._record_robot_states = lambda: None
    evaluator.obs_builder = SimpleNamespace(
        build_policy_obs=lambda: np.zeros(425, dtype=np.float32)
    )
    evaluator.policy_input_name = "obs"
    evaluator.policy_output_name = "action"
    evaluator.policy_obs_expected_dim = 786
    evaluator.use_kv_cache = False
    evaluator.policy_step_input_name = None
    evaluator.policy_kv_output_name = None
    evaluator.policy_moe_layer_output_names = []
    evaluator.dump_onnx_io_npy = False
    evaluator.counter = 0
    evaluator.command_mode = "velocity_tracking"
    evaluator.config = OmegaConf.create(
        {"motion_npz_dir": "", "motion_npz_path": ""}
    )
    evaluator.policy_session = SimpleNamespace(
        run=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime should not be called on shape mismatch")
        )
    )

    with pytest.raises(
        ValueError, match="expects 786 features but evaluator built 425"
    ):
        evaluator._update_policy()
