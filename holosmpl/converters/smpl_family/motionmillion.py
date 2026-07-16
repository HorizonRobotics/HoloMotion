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


MOTIONMILLION_SOURCE_COORDINATE_FRAME = "motionmillion_smplx_default_z_up_meter"
MOTIONMILLION_COORDINATE_TRANSFORM = "identity"
MOTIONMILLION_FPS_FIELDS = ("mocap_frame_rate", "mocap_framerate", "fps")
MOTIONMILLION_MOTION_FIELD_GROUPS = (
    ("trans",),
    MOTIONMILLION_FPS_FIELDS,
    ("root_orient",),
    ("pose_body",),
    ("betas",),
)


def classify_motionmillion_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one MotionMillion NPZ is convertible."""

    import numpy as np

    source_path = Path(source_path)
    with np.load(source_path, allow_pickle=True) as data:
        source_fields = sorted(data.files)

    missing_groups = [
        list(group)
        for group in MOTIONMILLION_MOTION_FIELD_GROUPS
        if not any(name in source_fields for name in group)
    ]
    if not missing_groups:
        return {
            "status": "convertible_motion_npz",
            "reason": None,
            "source_fields": source_fields,
            "missing_field_groups": [],
        }

    return {
        "status": "unsupported_missing_motion_fields",
        "reason": "missing_required_motion_fields",
        "source_fields": source_fields,
        "missing_field_groups": missing_groups,
    }


def convert_motionmillion_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Read one MotionMillion raw NPZ and return standardized canonical arrays."""

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    with np.load(source_path, allow_pickle=True) as data:
        source_fields = sorted(data.files)
        root_orient = _required_array(data, "root_orient")
        pose_body = _required_array(data, "pose_body")
        trans = _required_array(data, "trans")
        betas, betas_policy = _load_betas(data)
        gender = _load_gender(data)
        source_fps = _load_source_fps(data)

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
        "dataset": "MotionMillion",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_subset": _source_subset(source_relative_path),
        "source_fields": source_fields,
        "root_orient_source": "root_orient",
        "pose_body_source": "pose_body",
        "betas_policy": betas_policy,
        "gender": gender,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "pose_body_dim": pose_body_dim,
        "pose_body_layout": pose_body_layout,
        "source_quality_gate": "not_applied",
        "source_quality_gate_default": "disabled",
        "source_coordinate_system": MOTIONMILLION_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": MOTIONMILLION_COORDINATE_TRANSFORM,
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
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
        raise KeyError(f"missing required MotionMillion field: {name}")
    return data[name]


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
    for name in MOTIONMILLION_FPS_FIELDS:
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
    if pose_body.ndim != 2 or pose_body.shape[1] not in (63, 69):
        raise ValueError(f"pose_body must be [T,63] or [T,69], got {pose_body.shape}")
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
    if dim == 69:
        return "smpl_23_body"
    raise ValueError(f"unsupported pose_body dim: {dim}")


def _source_subset(source_relative_path: str) -> str:
    parts = Path(source_relative_path).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return "unknown"


def metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)
