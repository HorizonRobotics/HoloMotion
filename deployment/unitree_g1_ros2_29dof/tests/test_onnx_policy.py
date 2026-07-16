import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from humanoid_policy import onnx_policy


class FakeOnnxNode:
    def __init__(self, name, shape, type_):
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class FakeIoBinding:
    def __init__(self):
        self.input_bindings = []
        self.cpu_inputs = {}
        self.output_bindings = []
        self.outputs = [np.asarray([[1.0, 2.0]], dtype=np.float32)]

    def bind_input(self, **kwargs):
        self.input_bindings.append(kwargs)

    def bind_cpu_input(self, name, value):
        self.cpu_inputs[name] = value

    def bind_output(self, name, **kwargs):
        self.output_bindings.append((name, kwargs))

    def copy_outputs_to_cpu(self):
        return self.outputs


class FakeBindingSession:
    def __init__(self):
        self.binding = FakeIoBinding()
        self.ran_with = None

    def io_binding(self):
        return self.binding

    def run_with_iobinding(self, binding):
        self.ran_with = binding


class OnnxPolicyTest(unittest.TestCase):
    def test_warp_cuda_observation_uses_io_binding(self):
        class WarpObservation:
            is_cuda = True
            shape = (1, 12)
            buffer_ptr = 123456

            def __init__(self):
                self.synchronized = False

            def synchronize(self):
                self.synchronized = True

        session = FakeBindingSession()
        observation = WarpObservation()

        outputs = onnx_policy.run_with_cuda_observation(
            session,
            input_name="obs",
            observation=observation,
            output_names=["actions"],
        )

        self.assertTrue(observation.synchronized)
        binding = session.binding.input_bindings[0]
        self.assertEqual(binding["shape"], (1, 12))
        self.assertEqual(binding["buffer_ptr"], 123456)
        self.assertIs(outputs, session.binding.outputs)

    def test_cuda_observation_uses_io_binding(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        session = FakeBindingSession()
        observation = torch.arange(
            12, device="cuda:0", dtype=torch.float32
        ).reshape(1, 12)
        kv_cache = np.zeros((1, 2, 3), dtype=np.float16)

        outputs = onnx_policy.run_with_cuda_observation(
            session,
            input_name="obs",
            observation=observation,
            output_names=["actions"],
            cpu_inputs={"kv": kv_cache},
        )

        self.assertIs(session.ran_with, session.binding)
        self.assertEqual(len(session.binding.input_bindings), 1)
        binding = session.binding.input_bindings[0]
        self.assertEqual(binding["name"], "obs")
        self.assertEqual(binding["device_type"], "cuda")
        self.assertEqual(binding["device_id"], 0)
        self.assertEqual(binding["element_type"], np.float32)
        self.assertEqual(binding["shape"], (1, 12))
        self.assertEqual(binding["buffer_ptr"], observation.data_ptr())
        self.assertIs(session.binding.cpu_inputs["kv"], kv_cache)
        self.assertEqual(
            session.binding.output_bindings,
            [("actions", {"device_type": "cpu"})],
        )
        self.assertIs(outputs, session.binding.outputs)

    def test_build_inference_providers_defaults_to_current_onnx_cuda_path(self):
        providers = onnx_policy.build_inference_providers(
            inference_backend="onnx",
            device_id=2,
        )

        self.assertEqual(
            providers,
            [
                ("CUDAExecutionProvider", {"device_id": 2}),
                "CPUExecutionProvider",
            ],
        )

    def test_build_inference_providers_uses_tensorrt_only_with_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "trt_cache")
            providers = onnx_policy.build_inference_providers(
                inference_backend="tensorrt",
                device_id=1,
                tensorrt_fp16_enable=True,
                tensorrt_engine_cache_enable=True,
                tensorrt_engine_cache_path=cache_path,
                available_providers=[
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            self.assertTrue(Path(cache_path).is_dir())

        self.assertEqual(len(providers), 1)
        provider_name, provider_options = providers[0]
        self.assertEqual(provider_name, "TensorrtExecutionProvider")
        self.assertEqual(provider_options["device_id"], "1")
        self.assertEqual(provider_options["trt_fp16_enable"], "True")
        self.assertEqual(provider_options["trt_engine_cache_enable"], "True")
        self.assertEqual(provider_options["trt_engine_cache_path"], cache_path)

    def test_tensorrt_backend_defaults_to_fp32(self):
        providers = onnx_policy.build_inference_providers(
            inference_backend="tensorrt",
            available_providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        provider_options = providers[0][1]
        self.assertEqual(provider_options["trt_fp16_enable"], "False")

    def test_tensorrt_backend_requires_tensorrt_provider(self):
        with self.assertRaisesRegex(RuntimeError, "TensorRTExecutionProvider"):
            onnx_policy.build_inference_providers(
                inference_backend="tensorrt",
                available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def test_tensorrt_shared_library_preflight_reports_missing_library(self):
        with patch.object(onnx_policy.ctypes, "CDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "libnvinfer.so.8"):
                onnx_policy.check_tensorrt_shared_libraries()

    def test_infers_dtype_and_dynamic_dummy_shape(self):
        node = FakeOnnxNode("obs", [1, "obs_dim"], "tensor(float)")

        dummy = onnx_policy.build_dummy_input_from_onnx_node(
            node,
            fallback_last_dim=522,
        )

        self.assertEqual(dummy.shape, (1, 522))
        self.assertEqual(dummy.dtype, np.float32)
        self.assertIs(
            onnx_policy.infer_numpy_dtype_from_onnx_type("tensor(float16)"),
            np.float16,
        )
        self.assertIs(
            onnx_policy.infer_numpy_dtype_from_onnx_type("tensor(int64)"),
            np.int64,
        )
        self.assertIs(
            onnx_policy.infer_numpy_dtype_from_onnx_type("tensor(bool)"),
            np.bool_,
        )

    def test_inspects_motion_io_and_falls_back_to_non_kv_output(self):
        inputs = [
            FakeOnnxNode("obs", [1, 522], "tensor(float)"),
            FakeOnnxNode(
                "past_key_values",
                [3, 2, 1, 32, 4, 64],
                "tensor(float16)",
            ),
            FakeOnnxNode("step_idx", [1], "tensor(int64)"),
        ]
        outputs = [
            FakeOnnxNode(
                "present_key_values",
                [3, 2, 1, 32, 4, 64],
                "tensor(float16)",
            ),
            FakeOnnxNode("policy_output", [1, 29], "tensor(float)"),
        ]

        io = onnx_policy.inspect_motion_policy_io(
            inputs,
            outputs,
            default_input_name="input0",
        )

        self.assertEqual(io.input_name, "obs")
        self.assertEqual(io.output_name, "policy_output")
        self.assertEqual(io.kv_input_name, "past_key_values")
        self.assertEqual(io.kv_output_name, "present_key_values")
        self.assertEqual(io.step_idx_input_name, "step_idx")
        self.assertIs(io.kv_dtype, np.float16)

    def test_inspects_motion_io_prefers_actions_output(self):
        outputs = [
            FakeOnnxNode("other_output", [1, 29], "tensor(float)"),
            FakeOnnxNode("actions", [1, 29], "tensor(float)"),
            FakeOnnxNode("present_key_values", [1], "tensor(float)"),
        ]

        io = onnx_policy.inspect_motion_policy_io(
            [FakeOnnxNode("input0", [1, 522], "tensor(float)")],
            outputs,
            default_input_name="input0",
        )

        self.assertEqual(io.output_name, "actions")
        self.assertEqual(io.kv_output_name, "present_key_values")

    def test_creates_motion_kv_cache_with_effective_context_len(self):
        cache = onnx_policy.create_motion_kv_cache(
            kv_input_name="past_key_values",
            kv_shape=[3, 2, 1, 32, 4, 64],
            kv_dtype=np.float32,
            max_context_len=16,
        )

        self.assertTrue(cache.enabled)
        self.assertEqual(cache.cache.shape, (3, 2, 1, 32, 4, 64))
        self.assertEqual(cache.cache.dtype, np.float32)
        self.assertEqual(cache.model_context_len, 32)
        self.assertEqual(cache.effective_context_len, 16)

    def test_warmup_motion_policy_uses_local_kv_cache_without_mutating_source(self):
        inputs = [
            FakeOnnxNode("obs", [1, "obs_dim"], "tensor(float)"),
            FakeOnnxNode("past_key_values", [1, 1, 1, 4], "tensor(float)"),
            FakeOnnxNode("step_idx", [1], "tensor(int64)"),
        ]
        kv_cache = np.full((1, 1, 1, 4), 7.0, dtype=np.float32)

        class WarmupSession:
            def __init__(self):
                self.calls = []

            def get_inputs(self):
                return inputs

            def run(self, output_names, input_feed):
                self.calls.append(
                    {
                        "output_names": list(output_names),
                        "obs": input_feed["obs"].copy(),
                        "kv": input_feed["past_key_values"].copy(),
                        "step_idx": input_feed["step_idx"].copy(),
                    }
                )
                return [
                    np.zeros((1, 29), dtype=np.float32),
                    np.ones((1, 1, 1, 4), dtype=np.float32),
                ]

        session = WarmupSession()

        iterations = onnx_policy.warmup_motion_policy(
            session=session,
            input_name="obs",
            output_name="actions",
            kv_input_name="past_key_values",
            kv_output_name="present_key_values",
            step_idx_input_name="step_idx",
            use_kv_cache=True,
            kv_cache=kv_cache,
            kv_shape=[1, 1, 1, 4],
            kv_dtype=np.float32,
            obs_dim=522,
            num_iters=2,
        )

        self.assertEqual(iterations, 2)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            session.calls[0]["output_names"],
            ["actions", "present_key_values"],
        )
        self.assertEqual(session.calls[0]["obs"].shape, (1, 522))
        self.assertTrue(np.all(session.calls[0]["kv"] == 0.0))
        self.assertTrue(np.all(session.calls[1]["kv"] == 1.0))
        self.assertEqual(session.calls[0]["step_idx"].dtype, np.int64)
        self.assertEqual(session.calls[0]["step_idx"].tolist(), [0])
        self.assertEqual(session.calls[1]["step_idx"].tolist(), [1])
        self.assertTrue(np.all(kv_cache == 7.0))

    def test_warmup_policy_session_runs_fixed_number_of_dummy_inferences(self):
        inputs = [FakeOnnxNode("velocity_obs", [1, "obs_dim"], "tensor(float)")]

        class WarmupSession:
            def __init__(self):
                self.calls = []

            def get_inputs(self):
                return inputs

            def run(self, output_names, input_feed):
                self.calls.append(
                    {
                        "output_names": list(output_names),
                        "obs": input_feed["velocity_obs"].copy(),
                    }
                )
                return [np.zeros((1, 29), dtype=np.float32)]

        session = WarmupSession()

        iterations = onnx_policy.warmup_policy_session(
            session=session,
            input_name="velocity_obs",
            output_name="velocity_actions",
            obs_dim=123,
            num_iters=3,
        )

        self.assertEqual(iterations, 3)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.calls[0]["output_names"], ["velocity_actions"])
        self.assertEqual(session.calls[0]["obs"].shape, (1, 123))
        self.assertEqual(session.calls[0]["obs"].dtype, np.float32)

    def test_load_dual_policy_bundle_wires_sessions_and_kv(self):
        velocity_session = FakeSession(
            inputs=[FakeOnnxNode("velocity_obs", [1, 100], "tensor(float)")],
            outputs=[FakeOnnxNode("velocity_actions", [1, 29], "tensor(float)")],
        )
        motion_session = FakeSession(
            inputs=[
                FakeOnnxNode("obs", [1, 522], "tensor(float)"),
                FakeOnnxNode(
                    "past_key_values",
                    [3, 2, 1, 32, 4, 64],
                    "tensor(float)",
                ),
                FakeOnnxNode("step_idx", [1], "tensor(int64)"),
            ],
            outputs=[
                FakeOnnxNode("actions", [1, 29], "tensor(float)"),
                FakeOnnxNode(
                    "present_key_values",
                    [3, 2, 1, 32, 4, 64],
                    "tensor(float)",
                ),
            ],
        )
        sessions_by_path = {
            "/pkg/models/velocity/exported/model.onnx": velocity_session,
            "/pkg/models/motion/exported/model.onnx": motion_session,
        }
        created_sessions = []

        def fake_resolve_policy_onnx_path(package_share_dir, model_folder):
            return f"{package_share_dir}/models/{model_folder}/exported/model.onnx"

        def fake_create_onnx_session(
            onnx_path,
            sess_options,
            providers,
            disable_session_fallback=False,
        ):
            created_sessions.append(
                (onnx_path, sess_options, providers, disable_session_fallback)
            )
            return sessions_by_path[onnx_path]

        with patch.object(
            onnx_policy,
            "resolve_policy_onnx_path",
            side_effect=fake_resolve_policy_onnx_path,
        ), patch.object(
            onnx_policy,
            "create_onnx_session_options",
            return_value={"intra_op_threads": 3},
        ), patch.object(
            onnx_policy,
            "create_onnx_session",
            side_effect=fake_create_onnx_session,
        ):
            bundle = onnx_policy.load_dual_policy_bundle(
                package_share_dir="/pkg",
                velocity_model_folder="velocity",
                motion_model_folder="motion",
                intra_op_threads=3,
                motion_max_context_len=16,
                providers=["CPUExecutionProvider"],
            )

        self.assertIs(bundle.velocity_session, velocity_session)
        self.assertIs(bundle.motion_session, motion_session)
        self.assertEqual(bundle.velocity_input_name, "velocity_obs")
        self.assertEqual(bundle.velocity_output_name, "velocity_actions")
        self.assertEqual(bundle.motion_io.input_name, "obs")
        self.assertEqual(bundle.motion_io.output_name, "actions")
        self.assertTrue(bundle.motion_kv_cache.enabled)
        self.assertEqual(bundle.motion_kv_cache.effective_context_len, 16)
        self.assertEqual(bundle.inference_backend, "onnx")
        self.assertEqual(
            created_sessions,
            [
                (
                    "/pkg/models/velocity/exported/model.onnx",
                    {"intra_op_threads": 3},
                    ["CPUExecutionProvider"],
                    False,
                ),
                (
                    "/pkg/models/motion/exported/model.onnx",
                    {"intra_op_threads": 3},
                    ["CPUExecutionProvider"],
                    False,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
