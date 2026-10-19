"""Acceptance comparison must detect state, event and artifact differences."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from holoboxing.evaluation.repeatability import compare


class RepeatabilityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.first = Path(self.temporary.name) / "first"
        self.second = Path(self.temporary.name) / "second"
        for i, root in enumerate((self.first, self.second)):
            root.mkdir()
            payload = {
                "summary": {"final_score": [0, 0], "avg_fps": [10 + i, 20]},
                "metadata": {
                    "match_completed": True,
                    "frames_recorded": 1,
                    "match_id": str(i),
                    "physics_ordering": "fixture",
                },
                "frames": [
                    {
                        "root_pos": [[0.0, 0.0, 1.0]],
                        "events": [],
                        "policy_start_ns": i,
                    }
                ],
            }
            (root / "match.json").write_text(json.dumps(payload))
            with h5py.File(root / "replay.hdf5", "w") as replay:
                replay["state/qpos"] = np.zeros((1, 72), np.float32)

    def change_json(self, change):
        path = self.second / "match.json"
        payload = json.loads(path.read_text())
        change(payload)
        path.write_text(json.dumps(payload))

    def test_timing_and_identity_do_not_mask_identical_physics(self):
        self.assertEqual(compare(self.first, self.second)["status"], "PASS")

    def test_one_state_component_is_a_failure(self):
        with h5py.File(self.second / "replay.hdf5", "a") as replay:
            replay["state/qpos"][0, 2] = 1e-10
        report = compare(self.first, self.second)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(
            report["datasets"]["state/qpos"]["unequal_elements"], 1
        )

    def test_different_event_is_a_failure_even_with_equal_score(self):
        self.change_json(
            lambda p: p["frames"][0]["events"].append({"type": "FALL"})
        )
        self.assertFalse(compare(self.first, self.second)["frames_exact"])

    def test_asymmetric_dataset_is_a_failure(self):
        with h5py.File(self.second / "replay.hdf5", "a") as replay:
            replay["score/extra"] = [1]
        self.assertEqual(compare(self.first, self.second)["status"], "FAIL")

    def test_incomplete_match_is_rejected(self):
        self.change_json(lambda p: p["metadata"].update(match_completed=False))
        with self.assertRaisesRegex(ValueError, "Complete"):
            compare(self.first, self.second)


if __name__ == "__main__":
    unittest.main()
