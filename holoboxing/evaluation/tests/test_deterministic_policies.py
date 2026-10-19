"""Reject stochastic policies even in unexecuted branches and local functions."""

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from holoboxing.evaluation.runtime.model_contract import check_contract


def model(nodes, initializers=(), functions=()):
    return helper.make_model(
        helper.make_graph(
            nodes,
            "policy",
            [
                helper.make_tensor_value_info(
                    "obs", TensorProto.FLOAT, [1, 1097]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "target_pose", TensorProto.FLOAT, [1, 225]
                )
            ],
            list(initializers),
        ),
        functions=list(functions),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


class DeterministicPolicyTest(unittest.TestCase):
    def test_random_branch_is_rejected_without_execution(self):
        random = helper.make_graph(
            [helper.make_node("RandomNormal", [], ["sample"], shape=[1, 225])],
            "hidden_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "sample", TensorProto.FLOAT, [1, 225]
                )
            ],
        )
        graph = model(
            [
                helper.make_node(
                    "If",
                    ["condition"],
                    ["target_pose"],
                    then_branch=random,
                    else_branch=random,
                )
            ]
        )
        with self.assertRaisesRegex(
            ValueError, "random operator RandomNormal"
        ):
            check_contract(graph)

    def test_random_local_function_is_rejected(self):
        function = helper.make_function(
            "policy",
            "Sample",
            [],
            ["value"],
            [helper.make_node("RandomUniform", [], ["value"], shape=[1, 225])],
            [helper.make_opsetid("", 17)],
        )
        graph = model(
            [helper.make_node("Sample", [], ["target_pose"], domain="policy")],
            functions=[function],
        )
        with self.assertRaisesRegex(
            ValueError, "random operator RandomUniform"
        ):
            check_contract(graph)

    def test_inference_dropout_is_accepted_and_training_rejected(self):
        for training in (False, True):
            graph = model(
                [
                    helper.make_node(
                        "Dropout", ["pose", "", "training"], ["target_pose"]
                    )
                ],
                [
                    numpy_helper.from_array(
                        np.zeros((1, 225), np.float32), "pose"
                    ),
                    numpy_helper.from_array(np.array(training), "training"),
                ],
            )
            if training:
                with self.assertRaisesRegex(ValueError, "training_mode"):
                    check_contract(graph)
            else:
                check_contract(graph)


if __name__ == "__main__":
    unittest.main()
