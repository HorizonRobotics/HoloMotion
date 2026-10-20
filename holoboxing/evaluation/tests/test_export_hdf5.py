import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from holoboxing.evaluation.export_hdf5 import export
from holoboxing.evaluation.runtime.hand_hit_rules import (
    SCORING_VERSION,
    scoring_config,
)
from holoboxing.evaluation.replay_schema import read_replay


class ExportHDF5Test(unittest.TestCase):
    def payload(self):
        metadata = dict(
            match_completed=True,
            control_hz=50,
            physics_hz=200,
            match_id="test",
            dof_names=[f"joint{i}" for i in range(29)],
            retargeted_sha256="a" * 64,
            scoring_version=SCORING_VERSION,
            scoring_config=scoring_config(),
            arena_version="smplolympics-boxing-ring-v2",
            physics_backend="mujoco-cpu-test",
            holomotion_version="2.0_link_10500",
            num_rounds=1,
            round_duration_s=60,
            policy_sha256=["b" * 64, "c" * 64],
            competitor_sanitization_total=[3, 0],
            control_loop_benchmark=dict(status="INSUFFICIENT_SAMPLES"),
        )
        frames = [
            dict(
                dof_pos=np.zeros((2, 29)).tolist(),
                action_raw=np.zeros((2, 29)).tolist(),
                root_pos=[[0, 0, 0.8], [1, 0, 0.8]],
                root_quat=[[1, 0, 0, 0]] * 2,
                events=[],
                score=[0, 0],
                control_step_start_ns=100 + i * 100,
                control_step_end_ns=150 + i * 100,
                competitor_sanitization=[
                    dict(nonfinite_components=3, orientation_fallback_slots=1),
                    dict(nonfinite_components=0, orientation_fallback_slots=0),
                ],
                competitor_sanitization_ns=[10, 11],
                contacts=[],
            )
            for i in range(2)
        ]
        return dict(
            metadata=metadata,
            frames=frames,
            summary=dict(
                final_score=[0, 0], winner=None, model_id=["red", "blue"]
            ),
        )

    def test_v6_replay_preserves_rules_timing_and_sanitization_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = (
                Path(directory) / "match.json",
                Path(directory) / "replay.hdf5",
            )
            source.write_text(json.dumps(self.payload()))
            result = export(source, destination)
            self.assertEqual(result["frames"], 2)
            replay = read_replay(destination)
            self.assertEqual(replay.meta["scoring_config"], scoring_config())
            self.assertNotIn(
                "rearm_no_contact_ms", replay.meta["scoring_config"]
            )
            self.assertEqual(
                replay.meta["source"]["match_metadata"][
                    "competitor_sanitization_total"
                ],
                [3, 0],
            )
            self.assertEqual(
                replay.meta["environment"]["environment_version"],
                "boxing-track2-20261020-v6",
            )
            with h5py.File(destination) as file:
                np.testing.assert_array_equal(
                    file["policy/control_step_end_ns"][:], [150, 250]
                )
                statistics = json.loads(
                    file["policy/competitor_sanitization_json"][0]
                )
                self.assertEqual(statistics[0]["nonfinite_components"], 3)

    def test_historical_v4_v5_rules_are_preserved(self):
        for version, threshold, environment in (
            ("hands_only_normal_impulse100ms_5Ns_cooldown100ms_fall030_ko1000_v4", 5, "boxing-track2-20261014-v4"),
            ("hands_only_normal_impulse100ms_10Ns_cooldown100ms_fall030_ko1000_v5", 10, "boxing-track2-20261015-v5"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "match.json"
                destination = Path(directory) / "replay.hdf5"
                payload = self.payload()
                payload["metadata"]["scoring_version"] = version
                recorded = payload["metadata"]["scoring_config"]
                recorded.update(
                    normal_impulse_threshold_ns=threshold,
                    window_ms=100, per_hand_cooldown_ms=100,
                )
                source.write_text(json.dumps(payload))
                export(source, destination)
                replay = read_replay(destination)
                self.assertEqual(replay.meta["scoring_config"], recorded)
                self.assertEqual(replay.meta["environment"]["environment_version"], environment)

    def test_v6_missing_rule_metadata_is_rejected_without_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = (
                Path(directory) / "match.json",
                Path(directory) / "replay.hdf5",
            )
            value = self.payload()
            del value["metadata"]["scoring_config"]
            source.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                export(source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
