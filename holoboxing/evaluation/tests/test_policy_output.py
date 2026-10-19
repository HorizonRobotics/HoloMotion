import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from holoboxing.evaluation.runtime.policy_output import (
    sanitize_target_pose,
    run_competitor,
)


def target():
    value = np.zeros((1, 225), dtype=np.float32)
    value.reshape(5, 5, 9)[..., 3] = 1
    value.reshape(5, 5, 9)[..., 8] = 1
    return value


class PolicyOutputTest(unittest.TestCase):
    def test_nonfinite_numbers_and_only_corrupt_orientations_repaired(self):
        value = target()
        points = value.reshape(5, 5, 9)
        points[0, 0, 0] = np.nan
        points[1, 2, 4] = np.inf
        points[3, 4, 2] = -np.inf
        original = value.copy()
        result, stats = sanitize_target_pose(value)
        self.assertEqual(
            stats, dict(nonfinite_components=3, orientation_fallback_slots=1)
        )
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(
            result.reshape(5, 5, 9)[1, 2, 3:], [1, 0, 0, 0, 0, 1]
        )
        np.testing.assert_array_equal(value, original)

    def test_valid_output_is_preserved_and_no_alias(self):
        value = target()
        result, stats = sanitize_target_pose(value)
        np.testing.assert_array_equal(result, value)
        self.assertFalse(np.shares_memory(result, value))
        self.assertEqual(stats["nonfinite_components"], 0)

    def test_malformed_dtype_shape_and_finite_invalid_axes_fail(self):
        for value in [
            target().astype(np.float64),
            target().reshape(225),
            np.zeros((1, 225), np.float32),
        ]:
            with (
                self.subTest(shape=value.shape, dtype=value.dtype),
                self.assertRaises(ValueError),
            ):
                sanitize_target_pose(value)

    def test_torch_and_numpy_match_without_mutating_source(self):
        import torch

        value = target()
        value[0, [0, 7, 10]] = [np.nan, np.inf, -np.inf]
        expected, statistics = sanitize_target_pose(value)
        tensor = torch.from_numpy(value.copy())
        actual, tensor_statistics = sanitize_target_pose(tensor)
        np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual(tensor_statistics, statistics)
        self.assertEqual(int((~torch.isfinite(tensor)).sum()), 3)

    def test_measured_input_rejected_without_sanitizing_or_running(self):
        calls = []
        session = types.SimpleNamespace(run=lambda *args: calls.append(args))
        device = types.SimpleNamespace(
            verify_cuda_execution=lambda session: None
        )
        observation = np.zeros((1, 1097), np.float32)
        observation[0, 0] = np.nan
        with (
            patch.dict(
                sys.modules,
                {"holoboxing.evaluation.runtime.onnx_device": device},
            ),
            self.assertRaises(ValueError),
        ):
            run_competitor(session, observation)
        self.assertTrue(np.isnan(observation[0, 0]))
        self.assertEqual(calls, [])

    def test_nonfinite_output_reaches_sanitizer_before_any_finite_assertion(
        self,
    ):
        value = target()
        value[0, 0] = np.nan
        session = types.SimpleNamespace(run=lambda *args: [value])
        checked = []
        device = types.SimpleNamespace(
            verify_cuda_execution=lambda session: checked.append(session)
        )
        with patch.dict(
            sys.modules, {"holoboxing.evaluation.runtime.onnx_device": device}
        ):
            raw = run_competitor(session, np.zeros((1, 1097), np.float32))
        result, stats = sanitize_target_pose(raw)
        self.assertEqual(stats["nonfinite_components"], 1)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(checked, [session])


if __name__ == "__main__":
    unittest.main()
