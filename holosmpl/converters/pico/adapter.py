from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)
from holosmpl.core.schema.joints import SMPL_24_JOINT_NAMES


PICO_TIMING_COLS = (
    "timestamp_realtime",
    "timestamp_monotonic",
    "timestamp_ns",
    "pico_dt",
    "pico_fps",
)
PICO_BODY_COLS = tuple(
    f"body_pose_{joint_idx}_{component_idx}"
    for joint_idx in range(24)
    for component_idx in range(7)
)
PICO_EXPECTED_COLS = PICO_TIMING_COLS + PICO_BODY_COLS
PICO_SOURCE_COORDINATE_FRAME = "pico4_ultra_enterprise_global_body_pose_xyz_quat_xyzw"
PICO_CANONICAL_TRANSFORM = (
    "pico_global_quat_xyzw_apply_y180_global_to_local_smpl24_root_rx90_trans_rx90"
)
PICO_ZERO_BETA_DIM = 10
PICO_SOURCE_FORMAT = "csv_body_pose_24x7_global_xyz_quat_xyzw"
PICO_PARENT_TREE_SOURCE = "assumed_smpl_24_parent_tree"
PICO_SMPL_PARENTS_24 = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
PICO_RX90_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def classify_pico_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one Pico CSV is convertible."""

    source_path = Path(source_path)
    try:
        with source_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except OSError as exc:
        return {
            "status": "unsupported_unreadable_pico_csv",
            "reason": f"{type(exc).__name__}: {exc}",
            "source_fields": [],
            "missing_field_groups": [["csv_header"]],
        }

    source_fields = list(header or [])
    if tuple(source_fields) == PICO_EXPECTED_COLS:
        return {
            "status": "convertible_pico_csv",
            "reason": None,
            "source_fields": source_fields,
            "missing_field_groups": [],
        }

    missing = [[name] for name in PICO_EXPECTED_COLS if name not in source_fields]
    return {
        "status": "unsupported_unexpected_pico_csv_header",
        "reason": "unexpected_or_missing_pico_columns",
        "source_fields": source_fields,
        "missing_field_groups": missing,
    }


def convert_pico_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Convert one Pico4UltraEnterprise CSV into canonical SMPL-like arrays.

    Pico CSV stores per-joint global positions and global quaternions. It does
    not contain a source skeleton, so this converter assumes the 24 Pico joint
    indices follow SMPL-like order and converts global rotations to local SMPL
    rotations with the standard SMPL 24 parent tree.
    """

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    body, csv_metadata = read_pico_csv(source_path)
    source_fps = pico_timestamp_effective_fps(csv_metadata)
    pose_aa, trans = body_poses_to_smpl_pose_trans(body)
    root_orient = pose_aa[:, :3]
    pose_body = pose_aa[:, 3:72]
    betas = np.zeros((PICO_ZERO_BETA_DIM,), dtype=np.float32)

    _validate_canonical_arrays(root_orient=root_orient, pose_body=pose_body, trans=trans)
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
        "dataset": "Pico4UltraEnterprise",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_subset": _source_subset(source_relative_path),
        "source_format": PICO_SOURCE_FORMAT,
        "source_fields": list(PICO_EXPECTED_COLS),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "pose_body_dim": 69,
        "pose_body_layout": "smpl_23_body",
        "smpl_source_type": "pico_smpl_like_approx",
        "human_representation": "pico_smpl_like_approx",
        "is_strict_smpl": False,
        "approximation_warning": (
            "Pico source has global tracker joint poses but no explicit SMPL "
            "skeleton or fitted betas; converted by assuming SMPL-like joint "
            "order and SMPL 24 parent tree."
        ),
        "skeleton_in_source": False,
        "parent_tree_source": PICO_PARENT_TREE_SOURCE,
        "smpl_24_joint_names": list(SMPL_24_JOINT_NAMES),
        "smpl_24_parents": list(PICO_SMPL_PARENTS_24),
        "betas_policy": "zero_unknown",
        "source_native_beta_present": False,
        "gender": "unknown",
        "source_coordinate_system": PICO_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": PICO_CANONICAL_TRANSFORM,
        "coordinate_transform_matrix_root_trans_rx90": [list(row) for row in PICO_RX90_MATRIX],
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "source_fps_policy": "timestamp_ns_effective_fps",
        "pico_fps_policy": "diagnostic_only_not_used_for_resampling",
        "pico_timing_metadata": csv_metadata,
        "root_frame_semantics": "canonical_smpl_root_frame_approx_from_pico_tracker_root",
        "root_orient_policy": "global_pico_root_quat_y180_then_root_rx90",
        "root_trans_policy": "pico_joint0_xyz_then_root_rx90",
        "body_pose_policy": "global_pico_quat_y180_to_local_smpl24_rotvec",
        "root_frame_certified": False,
        **resample_metadata,
    }

    return {
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas,
        "gender": "unknown",
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "metadata": metadata,
    }


def read_pico_csv(source_path: str | Path) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    source_path = Path(source_path)
    body_rows: list[list[float]] = []
    pico_fps_values: list[float] = []
    pico_dt_values: list[float] = []
    timestamp_ns: list[int] = []
    timestamp_realtime: list[float] = []
    timestamp_monotonic: list[float] = []

    with source_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_pico_header(reader.fieldnames)
        for row in reader:
            body_rows.append([float(row[name]) for name in PICO_BODY_COLS])
            pico_fps_values.append(float(row["pico_fps"]))
            pico_dt_values.append(float(row["pico_dt"]))
            timestamp_ns.append(int(float(row["timestamp_ns"])))
            timestamp_realtime.append(float(row["timestamp_realtime"]))
            timestamp_monotonic.append(float(row["timestamp_monotonic"]))

    if len(body_rows) < 2:
        raise ValueError(f"Pico CSV must contain at least 2 frames, got {len(body_rows)}")

    body = np.asarray(body_rows, dtype=np.float32).reshape(-1, 24, 7)
    _validate_pico_body(body)
    timestamp_effective_fps = _timestamp_effective_fps(timestamp_ns, len(body_rows))
    monotonic_ok = bool(np.all(np.diff(np.asarray(timestamp_ns, dtype=np.int64)) > 0))
    metadata = {
        "source_rows": int(len(body_rows)),
        "pico_fps_median": _finite_median(pico_fps_values),
        "pico_fps_min": _finite_min(pico_fps_values),
        "pico_fps_max": _finite_max(pico_fps_values),
        "pico_dt_median": _finite_median(pico_dt_values),
        "timestamp_ns_start": int(timestamp_ns[0]),
        "timestamp_ns_end": int(timestamp_ns[-1]),
        "timestamp_ns_monotonic_strict": monotonic_ok,
        "timestamp_realtime_start": float(timestamp_realtime[0]),
        "timestamp_realtime_end": float(timestamp_realtime[-1]),
        "timestamp_monotonic_start": float(timestamp_monotonic[0]),
        "timestamp_monotonic_end": float(timestamp_monotonic[-1]),
        "timestamp_effective_fps": float(timestamp_effective_fps),
        "timestamp_duration_sec": float((timestamp_ns[-1] - timestamp_ns[0]) * 1.0e-9),
    }
    if not monotonic_ok:
        raise ValueError("Pico timestamp_ns must be strictly increasing")
    return body, metadata


def pico_timestamp_effective_fps(csv_metadata: dict[str, Any]) -> float:
    fps = float(csv_metadata.get("timestamp_effective_fps") or 0.0)
    if fps > 0:
        return fps
    fallback = float(csv_metadata.get("pico_fps_median") or 0.0)
    if fallback <= 0:
        raise ValueError("Pico source fps could not be inferred")
    return fallback


def body_poses_to_smpl_pose_trans(body: Any) -> tuple[Any, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    body_arr = np.asarray(body, dtype=np.float32)
    if body_arr.ndim != 3 or body_arr.shape[1:] != (24, 7):
        raise ValueError(f"Pico body must be [T,24,7], got {body_arr.shape}")

    positions = body_arr[:, :, :3]
    quat_xyzw = _normalize_quaternion_xyzw(body_arr[:, :, 3:7])
    frame_count = int(body_arr.shape[0])

    raw_mats = Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(
        frame_count, 24, 3, 3
    )
    y180 = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
    rx90 = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    global_mats = raw_mats @ y180

    local_mats = np.empty_like(global_mats)
    for joint_idx, parent_idx in enumerate(PICO_SMPL_PARENTS_24):
        if parent_idx < 0:
            local_mats[:, joint_idx] = global_mats[:, joint_idx]
        else:
            parent_inv = np.swapaxes(global_mats[:, parent_idx], -1, -2)
            local_mats[:, joint_idx] = parent_inv @ global_mats[:, joint_idx]

    local_mats[:, 0] = rx90 @ local_mats[:, 0]
    pose_aa = Rotation.from_matrix(local_mats.reshape(-1, 3, 3)).as_rotvec().astype(
        np.float32
    )
    pose_aa = pose_aa.reshape(frame_count, 24, 3)
    trans = (rx90 @ positions[:, 0, :, None]).reshape(frame_count, 3).astype(np.float32)
    return pose_aa.reshape(frame_count, 72).astype(np.float32), trans


def _validate_pico_header(fieldnames: Any) -> None:
    if tuple(fieldnames or ()) != PICO_EXPECTED_COLS:
        raise ValueError("Unexpected Pico CSV header; expected body_pose_<joint>_<component> order")


def _validate_pico_body(body: Any) -> None:
    import numpy as np

    body_arr = np.asarray(body, dtype=np.float32)
    if body_arr.ndim != 3 or body_arr.shape[1:] != (24, 7):
        raise ValueError(f"Pico body must be [T,24,7], got {body_arr.shape}")
    if not np.isfinite(body_arr).all():
        raise ValueError("Pico body contains NaN or Inf")
    quat = body_arr[:, :, 3:7]
    norms = np.linalg.norm(quat, axis=-1)
    bad = (~np.isfinite(norms)) | (norms <= 1.0e-8)
    if bool(np.any(bad)):
        raise ValueError("Pico quaternion contains invalid zero/nonfinite rows")


def _validate_canonical_arrays(*, root_orient: Any, pose_body: Any, trans: Any) -> None:
    import numpy as np

    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError(f"root_orient must be [T,3], got {root_orient.shape}")
    if pose_body.ndim != 2 or pose_body.shape[1] != 69:
        raise ValueError(f"pose_body must be [T,69], got {pose_body.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans.shape}")
    if not (len(root_orient) == len(pose_body) == len(trans)):
        raise ValueError(
            "root_orient, pose_body, and trans frame counts differ: "
            f"{len(root_orient)}, {len(pose_body)}, {len(trans)}"
        )
    for name, array in {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _normalize_quaternion_xyzw(quat_xyzw: Any) -> Any:
    import numpy as np

    quat = np.asarray(quat_xyzw, dtype=np.float32)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = np.zeros_like(quat, dtype=np.float32)
    identity[..., 3] = 1.0
    return np.where(norms > 1.0e-8, quat / norms, identity).astype(np.float32)


def _timestamp_effective_fps(timestamp_ns: list[int], row_count: int) -> float:
    if row_count <= 1:
        raise ValueError(f"row_count must be > 1 to infer timestamp fps, got {row_count}")
    duration = (float(timestamp_ns[-1]) - float(timestamp_ns[0])) * 1.0e-9
    if duration <= 1.0e-6:
        raise ValueError(f"invalid Pico timestamp duration: {duration}")
    return float((row_count - 1) / duration)


def _finite_median(values: list[float]) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else 0.0


def _finite_min(values: list[float]) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else 0.0


def _finite_max(values: list[float]) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else 0.0


def _source_subset(source_relative_path: str) -> str:
    parts = Path(source_relative_path).parts
    if len(parts) >= 2 and parts[0] == "pico_raw0402":
        return parts[1]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"
