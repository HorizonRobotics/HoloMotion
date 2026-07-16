"""Conversion helpers for deployment offline-motion NPZ files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


OFFLINE_ARRAY_ALIASES = {
    "ref_dof_pos": ("ref_dof_pos", "dof_pos"),
    "ref_dof_vel": ("ref_dof_vel", "dof_vel", "dof_vels"),
    "ref_global_translation": ("ref_global_translation", "global_translation"),
    "ref_global_rotation_quat": (
        "ref_global_rotation_quat",
        "global_rotation_quat",
    ),
    "ref_global_velocity": ("ref_global_velocity", "global_velocity"),
    "ref_global_angular_velocity": (
        "ref_global_angular_velocity",
        "global_angular_velocity",
    ),
}


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    value = np.asarray(raw)
    if value.size == 0:
        return {}
    item = value.reshape(-1)[0]
    if isinstance(item, dict):
        return dict(item)
    try:
        parsed = json.loads(str(item))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _pick_array(archive: Any, target_key: str) -> np.ndarray:
    for source_key in OFFLINE_ARRAY_ALIASES[target_key]:
        if source_key in archive:
            return np.asarray(archive[source_key], dtype=np.float32)
    aliases = ", ".join(OFFLINE_ARRAY_ALIASES[target_key])
    raise ValueError(f"Missing {target_key}; accepted legacy keys: {aliases}")


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_dof_count: int,
    expected_body_count: int,
) -> int:
    dof_pos = arrays["ref_dof_pos"]
    dof_vel = arrays["ref_dof_vel"]
    translation = arrays["ref_global_translation"]
    rotation = arrays["ref_global_rotation_quat"]
    velocity = arrays["ref_global_velocity"]
    angular_velocity = arrays["ref_global_angular_velocity"]

    if dof_pos.ndim != 2:
        raise ValueError(f"ref_dof_pos must have shape [T, D], got {dof_pos.shape}")
    frame_count = int(dof_pos.shape[0])
    expected_shapes = {
        "ref_dof_pos": (frame_count, expected_dof_count),
        "ref_dof_vel": (frame_count, expected_dof_count),
        "ref_global_translation": (frame_count, expected_body_count, 3),
        "ref_global_rotation_quat": (frame_count, expected_body_count, 4),
        "ref_global_velocity": (frame_count, expected_body_count, 3),
        "ref_global_angular_velocity": (frame_count, expected_body_count, 3),
    }
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(
                f"{key} must have shape {expected_shape}, got {arrays[key].shape}"
            )
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains NaN or Inf")

    if frame_count <= 0:
        raise ValueError("Offline motion must contain at least one frame")

    quat_norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError("ref_global_rotation_quat contains a zero quaternion")
    rotation /= quat_norm
    return frame_count


def convert_legacy_offline_npz(
    source_path: str | Path,
    output_path: str | Path,
    *,
    expected_dof_count: int = 29,
    expected_body_count: int = 30,
    fps: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a legacy G1 offline clip to the v1.4 deployment NPZ schema."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy offline NPZ not found: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    with np.load(source, allow_pickle=True) as archive:
        old_metadata = _parse_metadata(archive.get("metadata"))
        arrays = {
            target_key: _pick_array(archive, target_key).copy()
            for target_key in OFFLINE_ARRAY_ALIASES
        }

    frame_count = _validate_arrays(
        arrays,
        expected_dof_count=int(expected_dof_count),
        expected_body_count=int(expected_body_count),
    )
    resolved_fps = float(
        fps if fps is not None else old_metadata.get("motion_fps", 50.0)
    )
    if not np.isfinite(resolved_fps) or resolved_fps <= 0.0:
        raise ValueError(f"motion_fps must be positive, got {resolved_fps}")

    motion_key = str(old_metadata.get("motion_key", source.stem))
    metadata = {
        "format_version": "1.4.0",
        "motion_key": motion_key,
        "raw_motion_key": str(old_metadata.get("raw_motion_key", motion_key)),
        "motion_fps": resolved_fps,
        "num_frames": frame_count,
        "wallclock_len": (frame_count - 1) / resolved_fps,
        "num_dofs": int(expected_dof_count),
        "num_bodies": int(expected_body_count),
        "clip_length": int(
            old_metadata.get(
                "clip_length",
                old_metadata.get("original_num_frames", frame_count),
            )
        ),
        "valid_prefix_len": frame_count,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            np.savez_compressed(
                temp_file,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                **arrays,
            )
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return metadata
