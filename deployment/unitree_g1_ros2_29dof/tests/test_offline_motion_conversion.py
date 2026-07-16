import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from humanoid_policy.offline_motion_conversion import convert_legacy_offline_npz


def _legacy_arrays(frames: int = 4, dofs: int = 29, bodies: int = 30):
    rotation = np.zeros((frames, bodies, 4), dtype=np.float64)
    rotation[..., 3] = 2.0
    return {
        "dof_pos": np.zeros((frames, dofs), dtype=np.float64),
        "dof_vels": np.ones((frames, dofs), dtype=np.float64),
        "global_translation": np.zeros((frames, bodies, 3), dtype=np.float64),
        "global_rotation_quat": rotation,
        "global_velocity": np.zeros((frames, bodies, 3), dtype=np.float64),
        "global_angular_velocity": np.zeros((frames, bodies, 3), dtype=np.float64),
        "frame_flag": np.ones(frames, dtype=np.int32),
    }


def test_converts_prefixless_legacy_clip_to_v1_4_schema():
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "old.npz"
        output = Path(temp_dir) / "new.npz"
        metadata = {
            "motion_key": "legacy_clip",
            "motion_fps": 25.0,
            "original_num_frames": 3,
        }
        np.savez(
            source,
            metadata=np.asarray(json.dumps(metadata)),
            **_legacy_arrays(),
        )

        result = convert_legacy_offline_npz(source, output)

        assert result["format_version"] == "1.4.0"
        assert result["num_frames"] == 4
        assert result["motion_fps"] == 25.0
        assert result["clip_length"] == 3
        with np.load(output) as converted:
            assert set(converted.files) == {
                "metadata",
                "ref_dof_pos",
                "ref_dof_vel",
                "ref_global_translation",
                "ref_global_rotation_quat",
                "ref_global_velocity",
                "ref_global_angular_velocity",
            }
            assert converted["ref_dof_pos"].dtype == np.float32
            np.testing.assert_allclose(
                np.linalg.norm(converted["ref_global_rotation_quat"], axis=-1),
                1.0,
            )


def test_accepts_existing_ref_keys_and_drops_non_deployment_arrays():
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "old_ref.npz"
        output = Path(temp_dir) / "new_ref.npz"
        legacy = _legacy_arrays()
        arrays = {
            "ref_dof_pos": legacy["dof_pos"],
            "ref_dof_vel": legacy["dof_vels"],
            "ref_global_translation": legacy["global_translation"],
            "ref_global_rotation_quat": legacy["global_rotation_quat"],
            "ref_global_velocity": legacy["global_velocity"],
            "ref_global_angular_velocity": legacy["global_angular_velocity"],
        }
        arrays["ft_ref_dof_pos"] = np.zeros((4, 29), dtype=np.float32)
        np.savez(source, **arrays)

        convert_legacy_offline_npz(source, output)

        with np.load(output) as converted:
            assert "ft_ref_dof_pos" not in converted.files
            assert converted["ref_dof_vel"].shape == (4, 29)


def test_rejects_wrong_dof_count():
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "wrong.npz"
        output = Path(temp_dir) / "new.npz"
        np.savez(source, **_legacy_arrays(dofs=28))

        with pytest.raises(ValueError, match="ref_dof_pos must have shape"):
            convert_legacy_offline_npz(source, output)
