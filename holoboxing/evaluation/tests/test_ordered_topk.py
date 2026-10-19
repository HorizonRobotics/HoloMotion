"""Hierarchical selection matches stable ONNX ordering, including exact ties."""

import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

from holoboxing.evaluation.runtime.ordered_topk import ordered_model


def fixture(width, k=2, largest=1):
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "TopK",
                    ["x", "k"],
                    ["values", "indices"],
                    axis=-1,
                    largest=largest,
                    sorted=1,
                )
            ],
            "selection",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [2, 1, width]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "values", TensorProto.FLOAT, [2, 1, k]
                ),
                helper.make_tensor_value_info(
                    "indices", TensorProto.INT64, [2, 1, k]
                ),
            ],
            [numpy_helper.from_array(np.array([k], np.int64), "k")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


class OrderedTopKTest(unittest.TestCase):
    def test_ties_padding_and_recursive_selection_preserve_values_and_indices(
        self,
    ):
        rng = np.random.default_rng(1097)
        for width, k in (
            (257, 2),
            (1000, 2),
            (1024, 7),
            (32769, 100),
            (257, 128),
        ):
            for largest in (0, 1):
                with self.subTest(width=width, k=k, largest=largest):
                    model = ordered_model(fixture(width, k, largest))
                    onnx.checker.check_model(model)
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    session = ort.InferenceSession(
                        model.SerializeToString(),
                        options,
                        providers=["CPUExecutionProvider"],
                    )
                    samples = [
                        np.zeros((2, 1, width), np.float32),
                        rng.integers(-10, 10, size=(2, 1, width)).astype(
                            np.float32
                        ),
                        np.full(
                            (2, 1, width),
                            -np.inf if largest else np.inf,
                            np.float32,
                        ),
                    ]
                    for x in samples:
                        values, indices = session.run(None, {"x": x})
                        expected = np.argsort(
                            -x if largest else x, axis=-1, kind="stable"
                        )[..., :k]
                        np.testing.assert_array_equal(indices, expected)
                        np.testing.assert_array_equal(
                            values, np.take_along_axis(x, expected, axis=-1)
                        )

    def test_narrow_selection_needs_no_rewrite(self):
        self.assertIsNone(ordered_model(fixture(256)))

    def test_unsupported_wide_k_fails_instead_of_looping_or_running_unordered(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "K <= 128"):
            ordered_model(fixture(1024, 255))


if __name__ == "__main__":
    unittest.main()
