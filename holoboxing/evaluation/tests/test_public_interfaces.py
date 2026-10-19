"""Public boundaries tested without a GPU or private services."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

from holoboxing.evaluation.qualification import qualification, hardware_matches
from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.runtime.model_contract import (
    check_model,
    check_contract,
)
from holoboxing.evaluation.runtime.rules import MatchLogger
from holoboxing.evaluation.validation import validate_model


class PublicInterfacesTest(unittest.TestCase):
    def test_local_gpu_results_cannot_qualify_other_hardware(self):
        report = qualification("NVIDIA GeForce RTX 4090", 2.0)
        self.assertTrue(report["local_mean_within_budget"])
        self.assertFalse(report["target_hardware_match"])
        self.assertEqual(report["official_status"], "NOT_EVALUATED")
        self.assertEqual(
            qualification("NVIDIA GeForce RTX 5090", 5.0)["official_status"],
            "PASS",
        )
        self.assertEqual(
            qualification("NVIDIA GeForce RTX 5090", 5.001)["official_status"],
            "FAIL",
        )
        self.assertFalse(hardware_matches("RTX 50900"))
        for latency in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                qualification("RTX 5090", latency)

    def test_resource_hash_and_relative_closure_fail_closed(self):
        from holoboxing.evaluation.tests.resource_fixtures import (
            make_resources,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = make_resources(directory)
            tree = ResourceTree(root)
            self.assertEqual(tree.verify()["relative_references"], 18)
            with self.assertRaises(ValueError):
                tree.path("../escape")
            (root / "left_boxing_glove.STL").write_bytes(b"FAIL")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                tree.verify()

    @staticmethod
    def model(path, *, width=225, external=False):
        initializer = numpy_helper.from_array(
            np.zeros((1, width), dtype=np.float32), "pose"
        )
        if external:
            initializer.data_location = TensorProto.EXTERNAL
            entry = initializer.external_data.add()
            entry.key, entry.value = "location", "untrusted.bin"
            initializer.ClearField("raw_data")
        graph = helper.make_graph(
            [helper.make_node("Identity", ["pose"], ["target_pose"])],
            "fixture",
            [
                helper.make_tensor_value_info(
                    "obs", TensorProto.FLOAT, [1, 1097]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "target_pose", TensorProto.FLOAT, [1, width]
                )
            ],
            [initializer],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
        )
        path.write_bytes(model.SerializeToString())

    def test_invalid_model_report_distinguishes_format_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"invalid")
            report = validate_model(path, directory, directory)
            self.assertEqual(report["error_code"], "FORMAT_CHECK_FAILED")
            self.model(path, width=224)
            report = validate_model(path, directory, directory)
            self.assertEqual(report["error_code"], "CONTRACT_CHECK_FAILED")
            self.model(path, external=True)
            with self.assertRaisesRegex(ValueError, "self-contained"):
                check_model(path)

    def test_gpu_absence_is_platform_failure_not_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self.model(path)
            graph, _ = check_model(path)
            check_contract(graph)
            with patch(
                "holoboxing.evaluation.runtime.onnx_device.create_device_session",
                side_effect=RuntimeError("No CUDA GPU"),
            ):
                report = validate_model(path, directory, directory)
            self.assertEqual(
                report["error_code"], "PLATFORM_CHAIN_VALIDATION_FAILED"
            )
            self.assertEqual(report["format_check"]["status"], "PASS")

    def test_administrative_size_limit_can_only_be_stricter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self.model(path)
            report = validate_model(
                path, directory, directory, max_model_bytes=1
            )
            self.assertEqual(report["error_code"], "MODEL_TOO_LARGE")
            self.assertEqual(report["size_check"]["limit_bytes_exclusive"], 1)
            self.assertEqual(report["format_check"]["status"], "NOT_RUN")
            for limit in (0, -1, True, 50_000_001):
                with self.assertRaises(ValueError):
                    validate_model(
                        path, directory, directory, max_model_bytes=limit
                    )

    def test_sparse_external_weights_are_not_a_self_contained_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self.model(path)
            model = onnx.load(path)
            values = numpy_helper.from_array(
                np.array([1.0], dtype=np.float32), "pose"
            )
            values.ClearField("raw_data")
            values.data_location = TensorProto.EXTERNAL
            entry = values.external_data.add()
            entry.key, entry.value = "location", "outside.bin"
            indices = numpy_helper.from_array(
                np.array([[0, 0]], dtype=np.int64), "indices"
            )
            model.graph.ClearField("initializer")
            model.graph.sparse_initializer.append(
                helper.make_sparse_tensor(values, indices, [1, 225])
            )
            path.write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(ValueError, "self-contained"):
                check_model(path)

    def test_match_summary_has_only_local_result_fields(self):
        logger = MatchLogger(model_id=["a", "b"], match_id="test")
        logger.record_frame(
            root_pos=[[0, 0, 1]] * 2,
            root_quat=[[1, 0, 0, 0]] * 2,
            dof_pos=np.zeros((2, 29)),
            action_raw=np.zeros((2, 29)),
            policy_start_ns=[1, 1],
            policy_end_ns=[2, 2],
            score=[1, 0],
            events=[dict(type="ROUND_END", reason="TIMEOUT")],
        )
        summary = logger.summary([1, 0])
        self.assertEqual(summary["end_reason"], "TIMEOUT")
        self.assertEqual(summary["winner"], 0)
        self.assertEqual(
            set(summary),
            {
                "model_id",
                "final_score",
                "winner",
                "end_reason",
                "hit_count",
                "fall_count",
                "knockdown_count",
                "ko_count",
                "avg_fps",
                "avg_latency_ms",
            },
        )
