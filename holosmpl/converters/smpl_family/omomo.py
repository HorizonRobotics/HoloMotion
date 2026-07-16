from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)


OMOMO_SOURCE_COORDINATE_FRAME = "omomo_smpl_family_default_z_up_meter"
OMOMO_COORDINATE_TRANSFORM = "identity"
OMOMO_FPS_FIELDS = ("mocap_frame_rate", "mocap_framerate", "fps")
OMOMO_MOTION_FIELD_GROUPS = (
    ("poses",),
    ("trans",),
    OMOMO_FPS_FIELDS,
    ("betas",),
)
OMOMO_BODY_POSE_SLICE = (3, 66)
OMOMO_IGNORED_POSE_TAIL_SLICE = (66, None)


def classify_omomo_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one OMOMO NPZ is convertible."""

    import numpy as np

    source_path = Path(source_path)
    with np.load(source_path, allow_pickle=True) as data:
        source_fields = sorted(data.files)
        poses_shape = tuple(data["poses"].shape) if "poses" in data.files else None

    missing_groups = [
        list(group)
        for group in OMOMO_MOTION_FIELD_GROUPS
        if not any(name in source_fields for name in group)
    ]
    if missing_groups:
        return {
            "status": "unsupported_missing_motion_fields",
            "reason": "missing_required_motion_fields",
            "source_fields": source_fields,
            "missing_field_groups": missing_groups,
            "poses_shape": poses_shape,
        }
    if poses_shape is None or len(poses_shape) != 2 or poses_shape[1] < 66:
        return {
            "status": "unsupported_pose_shape",
            "reason": "poses_must_be_T_by_at_least_66",
            "source_fields": source_fields,
            "missing_field_groups": [],
            "poses_shape": poses_shape,
        }
    return {
        "status": "convertible_motion_npz",
        "reason": None,
        "source_fields": source_fields,
        "missing_field_groups": [],
        "poses_shape": poses_shape,
    }


def convert_omomo_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Read one OMOMO raw NPZ and return standardized canonical arrays.

    OMOMO stores SMPL-family pose vectors as [T,165]. The canonical human side
    only uses root + body pose, so poses[:, 66:] is preserved in metadata as an
    ignored tail rather than silently treated as body joints.
    """

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    with np.load(source_path, allow_pickle=True) as data:
        source_fields = sorted(data.files)
        poses = _required_array(data, "poses")
        root_orient = _load_root_orient_from_poses(poses)
        pose_body = _load_pose_body_from_poses(poses)
        trans = _required_array(data, "trans")
        betas, betas_policy = _load_betas(data)
        gender = _load_gender(data)
        source_fps = _load_source_fps(data)
        poses_shape = tuple(poses.shape)

    root_orient = np.asarray(root_orient, dtype=np.float32)
    pose_body = np.asarray(pose_body, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)

    _validate_raw_arrays(root_orient=root_orient, pose_body=pose_body, trans=trans, betas=betas)
    pose_body_dim = int(pose_body.shape[1])
    pose_body_layout = _pose_body_layout(pose_body_dim)

    root_orient, pose_body, trans, resample_metadata = resample_motion_to_fps(
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans,
        source_fps=source_fps,
        target_fps=target_fps,
    )

    try:
        source_relative_path = source_path.relative_to(input_root).as_posix()
    except ValueError:
        source_relative_path = source_path.as_posix()

    metadata = {
        "schema_version": "canonical_smpl_v1",
        "dataset": "OMOMO",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_fields": source_fields,
        "source_poses_shape": list(poses_shape),
        "root_orient_source": "poses[:, :3]",
        "pose_body_source": "poses[:, 3:66]",
        "source_pose_field": "poses",
        "source_translation_field": "trans",
        "ignored_pose_tail_policy": "ignore_poses_66_to_end_for_body_only_human_side",
        "ignored_pose_tail_slice": [66, int(poses_shape[1])],
        "betas_policy": betas_policy,
        "gender": gender,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "pose_body_dim": pose_body_dim,
        "pose_body_layout": pose_body_layout,
        "source_coordinate_system": OMOMO_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": OMOMO_COORDINATE_TRANSFORM,
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "root_frame_semantics": "standard_smpl_family_root_local_frame",
        "root_orient_policy": "copy_source_smpl_family_global_root_orient",
        "root_frame_certified": True,
        **resample_metadata,
    }

    return {
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": gender,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "metadata": metadata,
    }


def _required_array(data: Any, name: str) -> Any:
    if name not in data.files:
        raise KeyError(f"missing required OMOMO field: {name}")
    return data[name]


def _load_root_orient_from_poses(poses: Any) -> Any:
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"poses must be [T,D>=3], got {poses.shape}")
    return poses[:, :3]


def _load_pose_body_from_poses(poses: Any) -> Any:
    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(f"poses must be [T,D>=66] to derive OMOMO body pose, got {poses.shape}")
    return poses[:, OMOMO_BODY_POSE_SLICE[0] : OMOMO_BODY_POSE_SLICE[1]]


def _load_betas(data: Any) -> tuple[Any, str]:
    import numpy as np

    betas = _required_array(data, "betas")
    betas = np.asarray(betas)
    if betas.ndim == 1:
        return betas, "as_source_vector"
    if betas.ndim == 2:
        if len(betas) > 1 and not np.allclose(betas, betas[:1], atol=1e-6, rtol=1e-6):
            raise ValueError("per-frame betas are not constant; cannot canonicalize to [B]")
        return betas[0], "constant_per_frame_take_first"
    raise ValueError(f"betas must be [B] or [T,B], got {betas.shape}")


def _load_gender(data: Any) -> str:
    if "gender" not in data.files:
        return "unknown"
    value = data["gender"]
    if getattr(value, "shape", None) == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip().lower()
    return text if text in {"male", "female", "neutral", "unknown"} else "unknown"


def _load_source_fps(data: Any) -> float:
    for name in OMOMO_FPS_FIELDS:
        if name in data.files:
            value = data[name]
            if getattr(value, "shape", None) == ():
                value = value.item()
            fps = float(value)
            if fps <= 0:
                raise ValueError(f"{name} must be positive, got {fps}")
            return fps
    raise KeyError("missing source FPS field: expected mocap_frame_rate/mocap_framerate/fps")


def _validate_raw_arrays(*, root_orient: Any, pose_body: Any, trans: Any, betas: Any) -> None:
    import numpy as np

    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError(f"root_orient must be [T,3], got {root_orient.shape}")
    if pose_body.ndim != 2 or pose_body.shape[1] != 63:
        raise ValueError(f"pose_body must be [T,63], got {pose_body.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans.shape}")
    if not (len(root_orient) == len(pose_body) == len(trans)):
        raise ValueError(
            "root_orient, pose_body, and trans frame counts differ: "
            f"{len(root_orient)}, {len(pose_body)}, {len(trans)}"
        )
    if betas.ndim != 1:
        raise ValueError(f"betas must be [B], got {betas.shape}")
    for name, array in {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
        "betas": betas,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _pose_body_layout(dim: int) -> str:
    if dim == 63:
        return "smplx_21_body"
    raise ValueError(f"unsupported OMOMO pose_body dim: {dim}")


def metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)
