from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)


CUSTOM_SMPLX_DATASET_NAME = "CustomSMPLX"
CUSTOM_SMPLX_SOURCE_COORDINATE_FRAME = "custom_smplx_z_up_meter"
CUSTOM_SMPLX_COORDINATE_TRANSFORM = "identity"
CUSTOM_SMPLX_PACKED_LIST_V1_SOURCE_FPS = 50.0
CUSTOM_SMPLX_LEGACY_SOURCE_FPS = 30.0
CUSTOM_SMPLX_TRUE_BETAS = (
    -1.7961855,
    0.2894760,
    -0.2109168,
    -0.00637845,
    0.02236953,
    0.04821591,
    -0.04049987,
    0.01280841,
    -0.05742017,
    -0.01974897,
)


def classify_custom_smplx_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one custom SMPL-X pkl file is convertible."""

    import joblib

    source_path = Path(source_path)
    try:
        payload = joblib.load(source_path)
    except Exception as exc:
        return {
            "status": "unsupported_unreadable_pkl",
            "reason": f"{type(exc).__name__}: {exc}",
            "source_fields": [],
            "missing_field_groups": [["motion"], ["poses"], ["trans"], ["joints"]],
        }

    try:
        source_format, sequence_count, first_shapes, source_fields = _classify_payload(payload)
    except Exception as exc:
        return {
            "status": "unsupported_invalid_custom_smplx_payload",
            "reason": f"{type(exc).__name__}: {exc}",
            "source_fields": _payload_fields(payload),
            "missing_field_groups": [],
        }

    return {
        "status": "convertible_custom_smplx_pkl",
        "reason": None,
        "source_fields": source_fields,
        "missing_field_groups": [],
        "source_format": source_format,
        "sequence_count": sequence_count,
        "first_sequence_shapes": first_shapes,
    }


def iter_convert_custom_smplx_source(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
    sequence_shard_index: int = 0,
    sequence_shard_count: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield standardized canonical clips from packed custom SMPL-X data.

    The current packed_list_v1 format is a list of split motion samples at
    50 Hz.  Each sample already stores SMPL-X body axis-angle poses:
    poses[:, :3] is root orientation and poses[:, 3:66] is the 21-joint SMPL-X
    body pose.  The source betas are zero in this file, so this adapter applies
    the known subject betas requested for this dataset and records the override
    in metadata.
    """

    import joblib

    source_path = Path(source_path)
    input_root = Path(input_root)
    sequence_shard_index = int(sequence_shard_index)
    sequence_shard_count = max(1, int(sequence_shard_count))
    if sequence_shard_index < 0 or sequence_shard_index >= sequence_shard_count:
        raise ValueError(
            "sequence_shard_index must be in "
            f"[0, {sequence_shard_count}), got {sequence_shard_index}"
        )
    payload = joblib.load(source_path)
    try:
        source_relative_base = source_path.relative_to(input_root).as_posix()
    except ValueError:
        source_relative_base = source_path.as_posix()

    if isinstance(payload, list):
        yield from _iter_packed_list_v1_payload(
            payload,
            source_path=source_path,
            source_relative_base=source_relative_base,
            target_fps=target_fps,
            sequence_shard_index=sequence_shard_index,
            sequence_shard_count=sequence_shard_count,
        )
        return

    if isinstance(payload, dict):
        yield from _iter_legacy_dict_payload(
            payload,
            source_path=source_path,
            source_relative_base=source_relative_base,
            target_fps=target_fps,
            sequence_shard_index=sequence_shard_index,
            sequence_shard_count=sequence_shard_count,
        )
        return

    raise ValueError(f"unsupported custom SMPL-X payload type: {type(payload)}")


def _iter_packed_list_v1_payload(
    payload: list[Any],
    *,
    source_path: Path,
    source_relative_base: str,
    target_fps: float,
    sequence_shard_index: int,
    sequence_shard_count: int,
) -> Iterator[dict[str, Any]]:
    import numpy as np

    true_betas = np.asarray(CUSTOM_SMPLX_TRUE_BETAS, dtype=np.float32)
    sequence_count = len(payload)
    for sequence_index, sample in enumerate(payload):
        if sequence_index % sequence_shard_count != sequence_shard_index:
            continue
        if not isinstance(sample, dict) or "motion" not in sample:
            raise ValueError(f"packed sample {sequence_index} missing motion dict")
        motion = sample["motion"]
        poses = np.asarray(motion["poses"], dtype=np.float32)
        trans = np.asarray(motion["trans"], dtype=np.float32)
        joints = np.asarray(motion["joints"], dtype=np.float32)
        source_betas = np.asarray(motion.get("betas", []), dtype=np.float32)
        gender = _normalize_gender(motion.get("gender", "neutral"))
        _validate_sequence_arrays(poses=poses, trans=trans, betas=true_betas)
        if joints.ndim != 3 or joints.shape[1:] != (22, 3) or joints.shape[0] != poses.shape[0]:
            raise ValueError(f"joints must be [T,22,3] and match poses, got {joints.shape}")

        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
        source_fps = CUSTOM_SMPLX_PACKED_LIST_V1_SOURCE_FPS
        root_orient, pose_body, trans, resample_metadata = resample_motion_to_fps(
            root_orient=root_orient,
            pose_body=pose_body,
            trans=trans,
            source_fps=source_fps,
            target_fps=target_fps,
        )

        seq_name = str(sample.get("seq_name") or f"packed_seq_{sequence_index}")
        source_relative_path = f"{source_relative_base}::{seq_name}"
        frame_labels = sample.get("frame_labels") or []
        metadata = _base_metadata(
            source_path=source_path,
            source_relative_path=source_relative_path,
            source_file_relative_path=source_relative_base,
            sequence_index=sequence_index,
            sequence_count=sequence_count,
            source_format="packed_custom_smplx_v1_list",
            source_fields=["list[sample.motion]"],
            source_fps=source_fps,
            target_fps=target_fps,
            gender=gender,
            betas_policy="override_source_zero_betas_with_known_subject_betas",
            source_betas_shape=list(source_betas.shape),
            source_betas_all_zero=bool(source_betas.size == 0 or np.allclose(source_betas, 0.0)),
            source_betas_max_abs=float(np.max(np.abs(source_betas))) if source_betas.size else 0.0,
            extra={
                "seq_name": seq_name,
                "feat_p": sample.get("feat_p"),
                "data_source": sample.get("data_source"),
                "source_seq_idx": sample.get("source_seq_idx"),
                "source_start_frame": sample.get("source_start_frame"),
                "source_end_frame": sample.get("source_end_frame"),
                "source_shapes": {
                    "poses": list(poses.shape),
                    "trans": list(trans.shape),
                    "joints": list(joints.shape),
                    "betas": list(source_betas.shape),
                    "frame_labels": [len(frame_labels)],
                },
                "source_frame_label_policy": "per_frame_0.02s_labels_confirm_50hz",
                "source_frame_label_count": len(frame_labels),
                "crop_stage": "source_pre_split",
                "crop_policy": "already_split_by_source",
                "source_start_frame_dropped": 0,
                "target_equivalent_start_frame_dropped": 0,
                "source_start_time_dropped_sec": 0.0,
            },
            resample_metadata=resample_metadata,
        )
        yield {
            "root_orient": root_orient.astype(np.float32),
            "pose_body": pose_body.astype(np.float32),
            "trans": trans.astype(np.float32),
            "betas": true_betas.astype(np.float32),
            "gender": gender,
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "metadata": metadata,
        }


def _iter_legacy_dict_payload(
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_relative_base: str,
    target_fps: float,
    sequence_shard_index: int,
    sequence_shard_count: int,
) -> Iterator[dict[str, Any]]:
    import numpy as np

    sequence_count = _validate_legacy_top_level_lengths(payload)
    true_betas = np.asarray(CUSTOM_SMPLX_TRUE_BETAS, dtype=np.float32)
    for sequence_index in range(sequence_count):
        if sequence_index % sequence_shard_count != sequence_shard_index:
            continue
        poses = np.asarray(payload["poses"][sequence_index], dtype=np.float32)
        trans = np.asarray(payload["trans"][sequence_index], dtype=np.float32)
        joints = np.asarray(payload["joints"][sequence_index], dtype=np.float32)
        source_betas = np.asarray(payload.get("betas", [])[sequence_index], dtype=np.float32)
        _validate_sequence_arrays(poses=poses, trans=trans, betas=true_betas)
        if joints.ndim != 3 or joints.shape[1:] != (22, 3) or joints.shape[0] != poses.shape[0]:
            raise ValueError(f"joints must be [T,22,3] and match poses, got {joints.shape}")

        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
        source_fps = CUSTOM_SMPLX_LEGACY_SOURCE_FPS
        root_orient, pose_body, trans, resample_metadata = resample_motion_to_fps(
            root_orient=root_orient,
            pose_body=pose_body,
            trans=trans,
            source_fps=source_fps,
            target_fps=target_fps,
        )
        source_relative_path = f"{source_relative_base}::legacy_sequence_{sequence_index:06d}"
        metadata = _base_metadata(
            source_path=source_path,
            source_relative_path=source_relative_path,
            source_file_relative_path=source_relative_base,
            sequence_index=sequence_index,
            sequence_count=sequence_count,
            source_format="legacy_packed_dict",
            source_fields=sorted(str(key) for key in payload.keys()),
            source_fps=source_fps,
            target_fps=target_fps,
            gender="neutral",
            betas_policy="override_source_betas_with_known_subject_betas",
            source_betas_shape=list(source_betas.shape),
            source_betas_all_zero=bool(source_betas.size == 0 or np.allclose(source_betas, 0.0)),
            source_betas_max_abs=float(np.max(np.abs(source_betas))) if source_betas.size else 0.0,
            extra={
                "source_shapes": {
                    "poses": list(poses.shape),
                    "trans": list(trans.shape),
                    "joints": list(joints.shape),
                    "betas": list(source_betas.shape),
                    "ball_pos": list(np.asarray(payload["ball_pos"][sequence_index]).shape)
                    if "ball_pos" in payload
                    else None,
                },
                "crop_stage": "not_applied",
                "crop_policy": "legacy_source_not_used_for_current_run",
            },
            resample_metadata=resample_metadata,
        )
        yield {
            "root_orient": root_orient.astype(np.float32),
            "pose_body": pose_body.astype(np.float32),
            "trans": trans.astype(np.float32),
            "betas": true_betas.astype(np.float32),
            "gender": "neutral",
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "metadata": metadata,
        }


def _base_metadata(
    *,
    source_path: Path,
    source_relative_path: str,
    source_file_relative_path: str,
    sequence_index: int,
    sequence_count: int,
    source_format: str,
    source_fields: list[str],
    source_fps: float,
    target_fps: float,
    gender: str,
    betas_policy: str,
    source_betas_shape: list[int],
    source_betas_all_zero: bool,
    source_betas_max_abs: float,
    extra: dict[str, Any],
    resample_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema_version": "canonical_smpl_v1",
        "dataset": CUSTOM_SMPLX_DATASET_NAME,
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_file_relative_path": source_file_relative_path,
        "source_sequence_index": int(sequence_index),
        "source_sequence_count": int(sequence_count),
        "source_format": source_format,
        "source_fields": source_fields,
        "root_orient_source": "motion.poses[:, :3]",
        "pose_body_source": "motion.poses[:, 3:66]",
        "source_pose_field": "motion.poses",
        "source_translation_field": "motion.trans",
        "source_fps": float(source_fps),
        "source_fps_field": None,
        "source_fps_policy": "frame_labels_0.02s_and_user_confirmed_50hz"
        if abs(float(source_fps) - 50.0) < 1e-6
        else "legacy_external_provenance_30hz",
        "target_fps": float(target_fps),
        "pose_body_dim": 63,
        "pose_body_layout": "smplx_21_body",
        "betas_policy": betas_policy,
        "source_native_beta_present": True,
        "source_betas_shape": source_betas_shape,
        "source_betas_all_zero": source_betas_all_zero,
        "source_betas_max_abs": source_betas_max_abs,
        "known_subject_betas": list(CUSTOM_SMPLX_TRUE_BETAS),
        "known_subject_betas_source": "user_provided_custom_smplx_subject_betas",
        "gender": gender,
        "source_coordinate_system": CUSTOM_SMPLX_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": CUSTOM_SMPLX_COORDINATE_TRANSFORM,
        "coordinate_system_evidence": (
            "source joints/trans statistics show z is height axis; arrays are meter-scale SMPL-X outputs"
        ),
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "root_frame_semantics": "standard_smplx_body_root_local_frame",
        "root_orient_policy": "copy_source_smplx_global_root_orient",
        "root_frame_certified": True,
        "auxiliary_fields_present": ["joints"],
        "dropped_auxiliary_fields": ["joints"],
        "auxiliary_field_policy": "canonical/formal human SMPL fields only; joints remain in raw source",
    }
    metadata.update(extra)
    metadata.update(resample_metadata)
    return metadata


def _classify_payload(payload: Any) -> tuple[str, int, dict[str, list[int]], list[str]]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("packed list is empty")
        sample = payload[0]
        if not isinstance(sample, dict) or "motion" not in sample:
            raise ValueError("packed sample must contain motion")
        motion = sample["motion"]
        for name in ("poses", "trans", "joints", "betas"):
            if name not in motion:
                raise ValueError(f"packed motion missing {name}")
        return (
            "packed_custom_smplx_v1_list",
            len(payload),
            _packed_list_first_shapes(sample),
            sorted(str(key) for key in sample.keys()),
        )
    if isinstance(payload, dict):
        count = _validate_legacy_top_level_lengths(payload)
        return (
            "legacy_packed_dict",
            count,
            _legacy_first_shapes(payload),
            sorted(str(key) for key in payload.keys()),
        )
    raise ValueError(f"unsupported payload type: {type(payload)}")


def _payload_fields(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        return sorted(str(key) for key in payload.keys())
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return sorted(str(key) for key in payload[0].keys())
    return []


def _packed_list_first_shapes(sample: dict[str, Any]) -> dict[str, list[int]]:
    import numpy as np

    motion = sample["motion"]
    return {
        "poses": list(np.asarray(motion["poses"]).shape),
        "trans": list(np.asarray(motion["trans"]).shape),
        "joints": list(np.asarray(motion["joints"]).shape),
        "betas": list(np.asarray(motion["betas"]).shape),
    }


def _legacy_first_shapes(payload: dict[str, Any]) -> dict[str, list[int]]:
    import numpy as np

    return {
        "poses": list(np.asarray(payload["poses"][0]).shape),
        "trans": list(np.asarray(payload["trans"][0]).shape),
        "joints": list(np.asarray(payload["joints"][0]).shape),
        "betas": list(np.asarray(payload["betas"][0]).shape),
        "ball_pos": list(np.asarray(payload["ball_pos"][0]).shape)
        if "ball_pos" in payload
        else [],
    }


def _validate_legacy_top_level_lengths(payload: dict[str, Any]) -> int:
    required = ("poses", "trans", "joints", "betas")
    lengths = {}
    for name in required:
        if name not in payload:
            raise ValueError(f"legacy payload missing {name}")
        lengths[name] = len(payload[name])
    if len(set(lengths.values())) != 1:
        raise ValueError(f"top-level sequence counts differ: {lengths}")
    count = next(iter(lengths.values()))
    if count <= 0:
        raise ValueError("source contains no motion sequences")
    return int(count)


def _validate_sequence_arrays(*, poses: Any, trans: Any, betas: Any) -> None:
    import numpy as np

    if poses.ndim != 2 or poses.shape[1] != 66:
        raise ValueError(f"poses must be [T,66], got {poses.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans.shape}")
    if len(poses) != len(trans):
        raise ValueError(f"poses/trans frame counts differ: {len(poses)} vs {len(trans)}")
    if betas.ndim != 1:
        raise ValueError(f"betas must be [B], got {betas.shape}")
    for name, array in {"poses": poses, "trans": trans, "betas": betas}.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _normalize_gender(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text in {"male", "female", "neutral", "unknown"} else "neutral"
