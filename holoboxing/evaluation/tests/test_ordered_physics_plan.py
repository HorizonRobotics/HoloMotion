"""Sparse scheduling preserves contribution order and rejects dependencies."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from threading import Thread
import numpy as np

from holoboxing.evaluation.runtime.ordered_physics_linear import build_plans
from holoboxing.evaluation.runtime import deterministic_physics as ordering


class OrderedSparsePlanTest(unittest.TestCase):
    def setUp(self):
        self.addresses = np.array([0, 1, 3, 6])
        self.counts = np.array([1, 2, 3, 4])

    def test_each_destination_keeps_original_contribution_order(self):
        updates = np.array([[0, 3, 6], [0, 1, 1], [0, 2, 3]])
        matrix, up, down = build_plans(self.addresses, self.counts, [updates])[
            0
        ]
        self.assertEqual(matrix, ([0], [0, 3], [6, 1, 3], [0, 1, 2]))
        self.assertEqual(up, ([0], [0, 3], [3, 1, 2], [6, 1, 3]))
        self.assertEqual(down[0], [3, 1, 2])
        # These contributions cannot be sorted without changing float32 output.
        values = np.array([1e20, -1e20, 3], dtype=np.float32)
        self.assertNotEqual(
            np.float32(np.float32(values[0] + values[1]) + values[2]),
            np.float32(values[0] + np.float32(values[1] + values[2])),
        )

    def test_same_group_dependency_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dependency"):
            build_plans(
                self.addresses, self.counts, [np.array([[0, 1, 1], [1, 2, 4]])]
            )

    def test_duplicate_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_plans(
                self.addresses, self.counts, [np.array([[0, 1, 1], [0, 1, 1]])]
            )

    def test_diagonal_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout"):
            build_plans(self.addresses, self.counts, [np.array([[0, 1, 2]])])

    def test_scoped_adapter_does_not_escape_or_cross_threads(self):
        baseline = Mock(return_value="baseline")
        fast = Mock(return_value="arena")
        arena = SimpleNamespace(linear=SimpleNamespace(factor=fast))
        observed = []
        with (
            patch.object(ordering, "_installed", True),
            patch.object(ordering, "_factor", baseline),
        ):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with ordering.ordered_physics(arena):
                    self.assertEqual(
                        ordering._ordered_factor(1, 2, 3, 4, 5), "arena"
                    )
                    thread = Thread(
                        target=lambda: observed.append(
                            ordering._ordered_factor(1, 2, 3, 4, 5)
                        )
                    )
                    thread.start()
                    thread.join()
                    raise RuntimeError("probe")
            self.assertEqual(
                ordering._ordered_factor(1, 2, 3, 4, 5), "baseline"
            )
        self.assertEqual(observed, ["baseline"])
        fast.assert_called_once()


if __name__ == "__main__":
    unittest.main()
