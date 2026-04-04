import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holomotion.src.evaluation.eval_mujoco_sim2sim as eval_mujoco_sim2sim


class _FakeIoNode:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def test_load_policy_treats_false_string_use_gpu_as_cpu(monkeypatch):
    captured = {}

    class _FakeInferenceSession:
        def __init__(self, model_path, sess_options, providers):
            captured["model_path"] = model_path
            captured["providers"] = providers

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [_FakeIoNode("obs", [1, 16])]

        def get_outputs(self):
            return [_FakeIoNode("action", [1, 12])]

    monkeypatch.setattr(
        eval_mujoco_sim2sim.onnxruntime,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim.onnxruntime,
        "InferenceSession",
        _FakeInferenceSession,
    )

    evaluator = eval_mujoco_sim2sim.MujocoEvaluator.__new__(
        eval_mujoco_sim2sim.MujocoEvaluator
    )
    evaluator.config = OmegaConf.create(
        {
            "ckpt_onnx_path": "model.onnx",
            "use_gpu": "false",
            "gpu_id": 3,
        }
    )
    evaluator.max_context_len = 0

    evaluator.load_policy()

    assert captured["model_path"] == "model.onnx"
    assert captured["providers"] == ["CPUExecutionProvider"]


def test_create_ray_evaluator_preserves_use_gpu_false(monkeypatch):
    captured = {}

    class _FakeEvaluator:
        def __init__(self, config):
            captured["use_gpu"] = config.use_gpu
            captured["gpu_id"] = config.gpu_id

    monkeypatch.setattr(eval_mujoco_sim2sim, "MujocoEvaluator", _FakeEvaluator)

    eval_mujoco_sim2sim._create_ray_evaluator(
        {"use_gpu": False, "gpu_id": 5}, "holomotion"
    )

    assert captured["use_gpu"] is False
    assert captured["gpu_id"] == 5


def test_run_mujoco_sim2sim_eval_preserves_use_gpu_false(
    monkeypatch, tmp_path
):
    captured = {}

    class _FakeEvaluator:
        def __init__(self, config):
            captured["use_gpu"] = config.use_gpu

        def setup(self):
            captured["setup"] = True

        def run_simulation(self):
            captured["run_simulation"] = True

    monkeypatch.setattr(
        eval_mujoco_sim2sim.hydra.utils,
        "get_original_cwd",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        eval_mujoco_sim2sim,
        "process_config",
        lambda _: OmegaConf.create(
            {
                "use_gpu": False,
                "model_type": "holomotion",
            }
        ),
    )
    monkeypatch.setattr(eval_mujoco_sim2sim, "MujocoEvaluator", _FakeEvaluator)

    eval_mujoco_sim2sim.run_mujoco_sim2sim_eval(OmegaConf.create({}))

    assert captured["use_gpu"] is False
    assert captured["setup"] is True
    assert captured["run_simulation"] is True
