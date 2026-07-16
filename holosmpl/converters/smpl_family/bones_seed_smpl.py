from __future__ import annotations

from pathlib import Path
from typing import Any

from holosmpl.core.processing.coordinate import yup_trans_to_canonical_zup
from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)


BONES_SEED_SMPL_SOURCE_COORDINATE_FRAME = "bones_seed_smpl_pose_y_up_translation_y_up_joints_z_up_m"
BONES_SEED_SMPL_CANONICAL_TRANSFORM = "transl_yup_to_zup_and_root_orient_premultiply_yup_to_zup"
BONES_SEED_SMPL_REQUIRED_FIELDS = ("fps", "pose_aa", "transl", "smpl_joints")
BONES_SEED_SMPL_OPTIONAL_FIELDS = ("original_fps", "original_pose_aa")
BONES_SEED_SMPL_ZERO_BETA_DIM = 16
BONES_SEED_SMPL_YUP_TO_ZUP_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def classify_bones_seed_smpl_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one BonesSeedSMPL PKL is convertible before full conversion."""

    obj = _load_joblib_dict(source_path)
    source_fields = sorted(obj.keys())
    missing = [name for name in BONES_SEED_SMPL_REQUIRED_FIELDS if name not in obj]
    if not missing:
        return {
            "status": "convertible_bones_seed_smpl_pkl",
            "reason": None,
            "source_fields": source_fields,
            "missing_field_groups": [],
        }
    return {
        "status": "unsupported_missing_bones_seed_smpl_fields",
        "reason": "missing_required_bones_seed_smpl_fields",
        "source_fields": source_fields,
        "missing_field_groups": [[name] for name in missing],
    }


def convert_bones_seed_smpl_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Read one BonesSeedSMPL smpl_filtered PKL and return standardized canonical arrays."""

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    obj = _load_joblib_dict(source_path)
    source_fields = sorted(obj.keys())

    pose_aa = np.asarray(_required_field(obj, "pose_aa"), dtype=np.float32)
    transl = np.asarray(_required_field(obj, "transl"), dtype=np.float32)
    smpl_joints = np.asarray(_required_field(obj, "smpl_joints"), dtype=np.float32)
    source_fps = _load_source_fps(obj)
    original_fps = _load_optional_float(obj, "original_fps")
    original_pose_shape = _shape_or_none(obj.get("original_pose_aa"))

    _validate_raw_arrays(pose_aa=pose_aa, transl=transl, smpl_joints=smpl_joints)
    root_orient = _premultiply_root_yup_to_zup(pose_aa[:, :3])
    pose_body = pose_aa[:, 3:72]
    trans = yup_trans_to_canonical_zup(transl)
    betas = np.zeros((BONES_SEED_SMPL_ZERO_BETA_DIM,), dtype=np.float32)

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

    axis_audit = _axis_audit(transl=transl, smpl_joints=smpl_joints)
    metadata = {
        "schema_version": "canonical_smpl_v1",
        "dataset": "BonesSeedSMPL",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_fields": source_fields,
        "source_pose_field": "pose_aa",
        "source_translation_field": "transl",
        "source_fps_field": "fps",
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "original_fps": original_fps,
        "original_pose_aa_shape": original_pose_shape,
        "original_pose_aa_policy": "ignored_already_resampled_to_pose_aa_fps",
        "pose_body_dim": 69,
        "pose_body_layout": "smpl_23_body",
        "betas_policy": "zero_unknown",
        "source_native_beta_present": False,
        "gender": "unknown",
        "source_coordinate_system": BONES_SEED_SMPL_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": BONES_SEED_SMPL_CANONICAL_TRANSFORM,
        "coordinate_transform_matrix": [list(row) for row in BONES_SEED_SMPL_YUP_TO_ZUP_MATRIX],
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "bones_seed_smpl_transl_axis_policy": "y_up_to_canonical_z_up",
        "bones_seed_smpl_joints_axis_policy": "joints_observed_z_up_audit_only",
        "bones_seed_smpl_axis_consistency_check": axis_audit,
        "bones_seed_smpl_root_trans_source": "transl",
        "root_frame_semantics": "canonical_smpl_root_frame",
        "root_orient_policy": "premultiply_yup_to_zup_world_rotation",
        "root_frame_certified": True,
        **resample_metadata,
    }

    return {
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": "unknown",
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "metadata": metadata,
    }


def _load_joblib_dict(source_path: str | Path) -> dict[str, Any]:
    import joblib

    source_path = Path(source_path)
    obj = joblib.load(source_path)
    if not isinstance(obj, dict):
        raise TypeError(f"BonesSeedSMPL source must load as dict, got {type(obj).__name__}: {source_path}")
    return obj


def _required_field(obj: dict[str, Any], name: str) -> Any:
    if name not in obj:
        raise KeyError(f"missing required BonesSeedSMPL field: {name}")
    return obj[name]


def _load_source_fps(obj: dict[str, Any]) -> float:
    fps = float(_required_field(obj, "fps"))
    if fps <= 0:
        raise ValueError(f"BonesSeedSMPL fps must be positive, got {fps}")
    return fps


def _load_optional_float(obj: dict[str, Any], name: str) -> float | None:
    if name not in obj:
        return None
    value = float(obj[name])
    return value if value > 0 else None


def _shape_or_none(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def _premultiply_root_yup_to_zup(root_orient: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    root = np.asarray(root_orient, dtype=np.float32)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_orient must be [T,3], got {root.shape}")
    world_rot = Rotation.from_matrix(np.asarray(BONES_SEED_SMPL_YUP_TO_ZUP_MATRIX, dtype=np.float64))
    return (world_rot * Rotation.from_rotvec(root)).as_rotvec().astype(np.float32)


def _validate_raw_arrays(*, pose_aa: Any, transl: Any, smpl_joints: Any) -> None:
    import numpy as np

    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"pose_aa must be [T,72], got {pose_aa.shape}")
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"transl must be [T,3], got {transl.shape}")
    if smpl_joints.ndim != 3 or smpl_joints.shape[2] != 3:
        raise ValueError(f"smpl_joints must be [T,J,3], got {smpl_joints.shape}")
    if not (len(pose_aa) == len(transl) == len(smpl_joints)):
        raise ValueError(
            "pose_aa, transl, and smpl_joints frame counts differ: "
            f"{len(pose_aa)}, {len(transl)}, {len(smpl_joints)}"
        )
    for name, array in {
        "pose_aa": pose_aa,
        "transl": transl,
        "smpl_joints": smpl_joints,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _axis_audit(*, transl: Any, smpl_joints: Any) -> dict[str, Any]:
    import numpy as np

    transl = np.asarray(transl, dtype=np.float32)
    smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
    joint_extent = np.median(smpl_joints.max(axis=1) - smpl_joints.min(axis=1), axis=0)
    transl_median = np.median(transl, axis=0)
    trans_canonical = yup_trans_to_canonical_zup(transl)
    return {
        "smpl_joints_median_extent_xyz": [float(x) for x in joint_extent],
        "smpl_joints_height_axis_guess": int(np.argmax(joint_extent)),
        "transl_median_xyz": [float(x) for x in transl_median],
        "transl_height_axis_guess": int(np.argmax(transl_median)),
        "canonical_trans_median_xyz": [float(x) for x in np.median(trans_canonical, axis=0)],
        "axis_consistency_status": "known_mixed_axes_transl_y_up_joints_z_up",
    }
