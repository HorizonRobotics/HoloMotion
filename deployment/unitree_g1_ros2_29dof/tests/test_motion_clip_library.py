import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_policy.motion_clip_library import list_motion_clip_files
from humanoid_policy.motion_clip_library import load_motion_clip
from humanoid_policy.motion_clip_library import load_motion_clips
from humanoid_policy.motion_clip_library import select_motion_clip_index
from humanoid_policy.motion_clip_library import validate_loaded_motion_clip


def _write_clip(path: Path, frames: int, dofs: int = 2, bodies: int = 1, offset: float = 0.0):
    dof_pos = np.arange(frames * dofs, dtype=np.float32).reshape(frames, dofs) + offset
    np.savez(
        path,
        ref_dof_pos=dof_pos,
        ref_dof_vel=dof_pos + 10.0,
        ref_global_translation=np.full((frames, bodies, 3), offset + 20.0, dtype=np.float32),
        ref_global_rotation_quat=np.full((frames, bodies, 4), offset + 30.0, dtype=np.float32),
        ref_global_velocity=np.full((frames, bodies, 3), offset + 40.0, dtype=np.float32),
        ref_global_angular_velocity=np.full((frames, bodies, 3), offset + 50.0, dtype=np.float32),
    )


class MotionClipLibraryTest(unittest.TestCase):
    def test_lists_only_npz_files_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_clip(tmp_path / "b.npz", frames=2)
            _write_clip(tmp_path / "a.npz", frames=2)
            (tmp_path / "ignore.txt").write_text("ignore")

            self.assertEqual(list_motion_clip_files(tmp), ["a.npz", "b.npz"])

    def test_loads_legacy_motion_dict_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.npz"
            _write_clip(clip_path, frames=3, dofs=2, bodies=1, offset=5.0)

            clip = load_motion_clip(str(clip_path))

        self.assertEqual(
            sorted(clip.keys()),
            [
                "dof_pos",
                "dof_vel",
                "global_angular_velocity",
                "global_rotation_quat",
                "global_translation",
                "global_velocity",
                "n_frames",
            ],
        )
        self.assertEqual(clip["n_frames"], 3)
        self.assertEqual(clip["dof_pos"].shape, (3, 2))
        self.assertEqual(clip["global_translation"].shape, (3, 1, 3))
        np.testing.assert_allclose(clip["dof_vel"], clip["dof_pos"] + 10.0)

    def test_loads_files_in_caller_provided_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_clip(tmp_path / "b.npz", frames=2, offset=20.0)
            _write_clip(tmp_path / "a.npz", frames=2, offset=10.0)

            clips = load_motion_clips(tmp, ["a.npz", "b.npz"])

        self.assertEqual(len(clips), 2)
        self.assertEqual(float(clips[0]["dof_pos"][0, 0]), 10.0)
        self.assertEqual(float(clips[1]["dof_pos"][0, 0]), 20.0)

    def test_validates_loaded_clip_without_copying_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.npz"
            _write_clip(clip_path, frames=3, dofs=2, bodies=4)
            clip = load_motion_clip(str(clip_path))

        loaded = validate_loaded_motion_clip(
            clip,
            expected_dof_count=2,
            expected_body_count=4,
        )

        self.assertIs(loaded.dof_pos, clip["dof_pos"])
        self.assertIs(loaded.global_translation, clip["global_translation"])
        self.assertEqual(loaded.n_frames, 3)

    def test_rejects_loaded_clip_dof_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.npz"
            _write_clip(clip_path, frames=3, dofs=2, bodies=4)
            clip = load_motion_clip(str(clip_path))

        with self.assertRaisesRegex(ValueError, "ref_dof_pos DOF dimension mismatch"):
            validate_loaded_motion_clip(
                clip,
                expected_dof_count=29,
                expected_body_count=4,
            )

    def test_rejects_loaded_clip_body_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.npz"
            _write_clip(clip_path, frames=3, dofs=2, bodies=4)
            clip = load_motion_clip(str(clip_path))

        with self.assertRaisesRegex(
            ValueError,
            "ref_global_translation body dimension mismatch",
        ):
            validate_loaded_motion_clip(
                clip,
                expected_dof_count=2,
                expected_body_count=6,
            )

    def test_selects_motion_clip_indices_with_phase_3b_semantics(self):
        self.assertEqual(select_motion_clip_index(0, 3, "previous"), 2)
        self.assertEqual(select_motion_clip_index(2, 3, "next"), 0)
        self.assertEqual(select_motion_clip_index(2, 3, "first"), 0)
        self.assertEqual(select_motion_clip_index(0, 3, "last"), 2)
        self.assertEqual(select_motion_clip_index(5, 0, "next"), 5)

    def test_rejects_unknown_motion_clip_selection_command(self):
        with self.assertRaisesRegex(ValueError, "Unsupported motion clip selection command"):
            select_motion_clip_index(0, 3, "middle")


if __name__ == "__main__":
    unittest.main()
