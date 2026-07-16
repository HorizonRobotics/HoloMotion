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


GVHMR_SOURCE_COORDINATE_FRAME = "gvhmr_smplx_default_z_up_meter"
GVHMR_COORDINATE_TRANSFORM = "identity"
GVHMR_FPS_FIELDS = ("mocap_frame_rate", "mocap_framerate", "fps")
GVHMR_REQUIRED_FIELDS = (
    "root_orient",
    "pose_body",
    "trans",
    "betas",
    "mocap_frame_rate",
)


def classify_gvhmr_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one GVHMR NPZ is convertible."""

    import numpy as np

    source_path = Path(source_path)
    with np.load(source_path, allow_pickle=True) as data:
        source_fields = sorted(data.files)
        root_shape = tuple(data["root_orient"].shape) if "root_orient" in data.files else None
        pose_shape = tuple(data["pose_body"].shape) if "pose_body" in data.files else None
        trans_shape = tuple(data["trans"].shape) if "trans" in data.files else None
        betas_shape = tuple(data["betas"].shape) if "betas" in data.files else None

    missing = [[name] for name in GVHMR_REQUIRED_FIELDS if name not in source_fields]
    if missing:
        return {
            "status": "unsupported_missing_motion_fields",
            "reason": "missing_required_gvhmr_fields",
            "source_fields": source_fields,
            "missing_field_groups": missing,
            "root_orient_shape": root_shape,
            "pose_body_shape": pose_shape,
            "trans_shape": trans_shape,
            "betas_shape": betas_shape,
        }
    if root_shape is None or len(root_shape) != 2 or root_shape[1] != 3:
        return {
            "status": "unsupported_root_orient_shape",
            "reason": "root_orient_must_be_T_by_3",
            "source_fields": source_fields,
            "missing_field_groups": [],
            "root_orient_shape": root_shape,
            "pose_body_shape": pose_shape,
            "trans_shape": trans_shape,
            "betas_shape": betas_shape,
        }
    if pose_shape is None or len(pose_shape) != 2 or pose_shape[1] != 63:
        return {
            "status": "unsupported_pose_body_shape",
            "reason": "pose_body_must_be_T_by_63",
            "source_fields": source_fields,
            "missing_field_groups": [],
            "root_orient_shape": root_shape,
            "pose_body_shape": pose_shape,
            "trans_shape": trans_shape,
            "betas_shape": betas_shape,
        }
    if trans_shape is None or len(trans_shape) != 2 or trans_shape[1] != 3:
        return {
            "status": "unsupported_trans_shape",
            "reason": "trans_must_be_T_by_3",
            "source_fields": source_fields,
            "missing_field_groups": [],
            "root_orient_shape": root_shape,
            "pose_body_shape": pose_shape,
            "trans_shape": trans_shape,
            "betas_shape": betas_shape,
        }
    return {
        "status": "convertible_motion_npz",
        "reason": None,
        "source_fields": source_fields,
        "missing_field_groups": [],
        "root_orient_shape": root_shape,
        "pose_body_shape": pose_shape,
        "trans_shape": trans_shape,
        "betas_shape": betas_shape,
    }


def convert_gvhmr_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Read one GVHMR raw NPZ and return standardized canonical arrays.

    GVHMR stores SMPL-family root/body axis-angle poses directly.  Its body
    pose has 21 SMPL-X body joints, so canonical preserves [T,63] and formal_npz
    later pads the two missing SMPL body hand joints with identity rotations.
    """

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
        source_shapes = {
            "root_orient": list(root_orient.shape),
            "pose_body": list(pose_body.shape),
            "trans": list(trans.shape),
            "betas": list(np.asarray(betas).shape),
        }

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
        "dataset": "GVHMR",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_fields": source_fields,
        "source_shapes": source_shapes,
        "root_orient_source": "root_orient",
        "pose_body_source": "pose_body",
        "source_translation_field": "trans",
        "source_fps_field": "mocap_frame_rate",
        "betas_policy": betas_policy,
        "gender": gender,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "pose_body_dim": pose_body_dim,
        "pose_body_layout": pose_body_layout,
        "source_coordinate_system": GVHMR_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": GVHMR_COORDINATE_TRANSFORM,
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
        raise KeyError(f"missing required GVHMR field: {name}")
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
    for name in GVHMR_FPS_FIELDS:
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
    raise ValueError(f"unsupported GVHMR pose_body dim: {dim}")


def metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)
