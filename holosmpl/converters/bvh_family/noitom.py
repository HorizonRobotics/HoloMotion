from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)


NOITOM_SOURCE_COORDINATE_FRAME = "noitom_bvh_y_up_xyz_cm"
NOITOM_CANONICAL_TRANSFORM = (
    "bvh_yup_cm_to_canonical_zup_m_root_orient_premultiply_and_trans_transform"
)
NOITOM_UNIT_SCALE = 0.01
NOITOM_DROP_FIRST_FRAME = True
NOITOM_ZERO_BETA_DIM = 10
NOITOM_YUP_TO_ZUP_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)

SMPL_24_JOINTS = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

BVH_TO_SMPL_INDEX = {
    "Hips": 0,
    "LeftUpLeg": 1,
    "RightUpLeg": 2,
    "Spine": 3,
    "LeftLeg": 4,
    "RightLeg": 5,
    "Spine1": 6,
    "LeftFoot": 7,
    "RightFoot": 8,
    "Spine2": 9,
    "Neck": 12,
    "LeftShoulder": 13,
    "RightShoulder": 14,
    "Head": 15,
    "LeftArm": 16,
    "RightArm": 17,
    "LeftForeArm": 18,
    "RightForeArm": 19,
    "LeftHand": 20,
    "RightHand": 21,
}
IDENTITY_FILL_SMPL_INDICES = (10, 11, 22, 23)


@dataclass(frozen=True)
class BvhJoint:
    name: str
    parent: int
    offset: tuple[float, float, float]
    channels: tuple[str, ...]


@dataclass(frozen=True)
class BvhMotion:
    joints: tuple[BvhJoint, ...]
    frame_time: float
    values: Any


def classify_noitom_bvh_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one Noitom BVH is convertible."""

    source_path = Path(source_path)
    try:
        header = source_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {
            "status": "unsupported_unreadable_bvh",
            "reason": f"{type(exc).__name__}: {exc}",
            "source_fields": [],
            "missing_field_groups": [["HIERARCHY"], ["MOTION"], ["Frames"], ["Frame Time"]],
        }

    source_fields = _detect_bvh_fields(header)
    missing = [
        [name]
        for name in ("HIERARCHY", "MOTION", "Frames", "Frame Time")
        if name not in source_fields
    ]
    if missing:
        return {
            "status": "unsupported_malformed_bvh",
            "reason": "missing_required_bvh_sections",
            "source_fields": source_fields,
            "missing_field_groups": missing,
        }

    return {
        "status": "convertible_bvh",
        "reason": None,
        "source_fields": source_fields,
        "missing_field_groups": [],
    }


def convert_noitom_bvh_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Convert one Noitom BVH into canonical SMPL-like arrays.

    This is an approximate BVH-to-SMPL mapping. It does not fit SMPL mesh
    parameters and always records the approximation policy in metadata.
    """

    return convert_bvh_family_sample(
        source_path,
        input_root=input_root,
        target_fps=target_fps,
        dataset_name="NoitomBVH",
        source_coordinate_frame=NOITOM_SOURCE_COORDINATE_FRAME,
    )


def convert_bvh_family_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
    dataset_name: str,
    source_coordinate_frame: str,
    unit_scale: float = NOITOM_UNIT_SCALE,
    drop_first_frame: bool = NOITOM_DROP_FIRST_FRAME,
    zero_beta_dim: int = NOITOM_ZERO_BETA_DIM,
    bvh_to_smpl_index: dict[str, int] | None = None,
    identity_fill_smpl_indices: tuple[int, ...] = IDENTITY_FILL_SMPL_INDICES,
    canonical_transform: str = NOITOM_CANONICAL_TRANSFORM,
    repair_root_translation_spikes: bool = False,
) -> dict[str, Any]:
    """Convert one BVH source into canonical SMPL-like arrays.

    This helper is shared by Noitom-family BVH datasets.  It is intentionally
    approximate: BVH joint local rotations are copied by semantic joint name
    into an SMPL-24 pose vector, and missing SMPL joints are identity-filled.
    """

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    motion = load_bvh_motion(source_path)
    source_fps = 1.0 / motion.frame_time
    source_frame_count_raw = int(motion.values.shape[0])
    root_orient, pose_body, trans, mapping_metadata = _build_smpl_like_arrays(
        motion,
        unit_scale=unit_scale,
        bvh_to_smpl_index=bvh_to_smpl_index or BVH_TO_SMPL_INDEX,
        identity_fill_smpl_indices=identity_fill_smpl_indices,
    )

    if drop_first_frame:
        root_orient = root_orient[1:]
        pose_body = pose_body[1:]
        trans = trans[1:]

    root_repair_metadata: dict[str, Any] = {
        "root_translation_spike_repair_policy": "disabled",
        "root_translation_spike_repaired_frame_indices": [],
    }
    if repair_root_translation_spikes:
        trans, repaired_indices = _repair_isolated_root_translation_spikes(trans)
        root_repair_metadata = {
            "root_translation_spike_repair_policy": (
                "repair_isolated_translation_blocks_with_linear_interpolation"
            ),
            "root_translation_spike_repaired_frame_indices": repaired_indices,
            "root_translation_spike_repaired_frame_count": len(repaired_indices),
        }

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

    betas = np.zeros((zero_beta_dim,), dtype=np.float32)
    source_subset = _source_subset(source_relative_path)
    metadata = {
        "schema_version": "canonical_smpl_v1",
        "dataset": dataset_name,
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_subset": source_subset,
        "source_format": "bvh",
        "source_fields": ["bvh_hierarchy", "bvh_motion"],
        "source_fps": float(source_fps),
        "source_frame_time": float(motion.frame_time),
        "source_frame_count_raw": source_frame_count_raw,
        "target_fps": float(target_fps),
        "pose_body_dim": 69,
        "pose_body_layout": "smpl_23_body",
        "smpl_source_type": "bvh_smpl_like_approx",
        "human_representation": "bvh_smpl_like_approx",
        "is_strict_smpl": False,
        "approximation_warning": "Approximate BVH-to-SMPL-like mapping only; not fitted/strict SMPL.",
        "betas_policy": "zero_unknown",
        "source_native_beta_present": False,
        "gender": "unknown",
        "drop_first_frame": bool(drop_first_frame),
        "unit_scale": float(unit_scale),
        "source_coordinate_system": source_coordinate_frame,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": canonical_transform,
        "coordinate_transform_matrix": [list(row) for row in NOITOM_YUP_TO_ZUP_MATRIX],
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "root_frame_semantics": "canonical_smpl_root_frame_approx_from_bvh_root",
        "root_orient_policy": "premultiply_yup_to_zup_world_rotation",
        "root_trans_policy": "bvh_root_position_yup_cm_to_canonical_zup_m",
        "body_pose_policy": "mapped_bvh_local_rotations_identity_fill_missing_smpl_joints",
        "root_frame_certified": False,
        "source_joint_count": len(motion.joints),
        "source_channel_count": int(motion.values.shape[1]),
        "source_root_joint": motion.joints[0].name if motion.joints else None,
        "source_joint_names": [joint.name for joint in motion.joints],
        "source_joint_channel_counts": {
            joint.name: len(joint.channels) for joint in motion.joints
        },
        "source_root_channels": list(motion.joints[0].channels) if motion.joints else [],
        **mapping_metadata,
        **root_repair_metadata,
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


def inspect_noitom_bvh_source(source_path: str | Path) -> dict[str, Any]:
    motion = load_bvh_motion(source_path)
    channels = [channel for joint in motion.joints for channel in joint.channels]
    return {
        "source_path": str(source_path),
        "joint_count": len(motion.joints),
        "channel_count": len(channels),
        "frame_count": int(motion.values.shape[0]),
        "frame_time": float(motion.frame_time),
        "fps": float(1.0 / motion.frame_time),
        "root_joint": motion.joints[0].name if motion.joints else None,
        "joint_names": [joint.name for joint in motion.joints],
        "channel_names": channels,
    }


def load_bvh_motion(source_path: str | Path) -> BvhMotion:
    import numpy as np

    source_path = Path(source_path)
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_index = _find_line_index(lines, "MOTION")
    if motion_index is None:
        raise ValueError("BVH missing MOTION section")

    joints = _parse_hierarchy(lines[:motion_index])
    if not joints:
        raise ValueError("BVH hierarchy has no joints")

    frames_line_index = _find_prefixed_line_index(lines, "Frames:", start=motion_index + 1)
    frame_time_line_index = _find_prefixed_line_index(lines, "Frame Time:", start=motion_index + 1)
    if frames_line_index is None or frame_time_line_index is None:
        raise ValueError("BVH motion section missing Frames or Frame Time")
    frame_count = int(lines[frames_line_index].split(":", 1)[1].strip())
    frame_time = float(lines[frame_time_line_index].split(":", 1)[1].strip())
    if frame_count <= 0 or frame_time <= 0:
        raise ValueError(f"invalid BVH frame metadata: frames={frame_count}, frame_time={frame_time}")

    expected_channels = sum(len(joint.channels) for joint in joints)
    value_rows = []
    for line in lines[frame_time_line_index + 1 :]:
        text = line.strip()
        if not text:
            continue
        row = np.fromstring(text, sep=" ", dtype=np.float32)
        if row.size:
            value_rows.append(row)
    if len(value_rows) != frame_count:
        raise ValueError(f"BVH frame count mismatch: header={frame_count}, rows={len(value_rows)}")
    values = np.vstack(value_rows).astype(np.float32)
    if values.shape[1] != expected_channels:
        raise ValueError(
            f"BVH channel count mismatch: hierarchy={expected_channels}, motion={values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("BVH motion values contain NaN or Inf")
    return BvhMotion(joints=tuple(joints), frame_time=frame_time, values=values)


def _parse_hierarchy(lines: list[str]) -> list[BvhJoint]:
    joints: list[BvhJoint] = []
    stack: list[int | None] = []
    pending_joint: int | None = None
    pending_end_site = False

    for raw in lines:
        line = raw.strip()
        if not line or line == "HIERARCHY":
            continue
        match = re.match(r"^(ROOT|JOINT)\s+(\S+)", line)
        if match:
            parent = next((idx for idx in reversed(stack) if idx is not None), -1)
            joints.append(
                BvhJoint(name=match.group(2), parent=parent, offset=(0.0, 0.0, 0.0), channels=())
            )
            pending_joint = len(joints) - 1
            pending_end_site = False
            continue
        if line.startswith("End Site"):
            pending_joint = None
            pending_end_site = True
            continue
        if line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site:
                stack.append(None)
                pending_end_site = False
            continue
        if line == "}":
            if stack:
                stack.pop()
            continue

        current = stack[-1] if stack else None
        if current is None:
            continue
        if line.startswith("OFFSET"):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"invalid OFFSET line: {line}")
            joint = joints[current]
            joints[current] = BvhJoint(
                name=joint.name,
                parent=joint.parent,
                offset=(float(parts[1]), float(parts[2]), float(parts[3])),
                channels=joint.channels,
            )
        elif line.startswith("CHANNELS"):
            parts = line.split()
            count = int(parts[1])
            channels = tuple(parts[2:])
            if len(channels) != count:
                raise ValueError(f"invalid CHANNELS line: {line}")
            joint = joints[current]
            joints[current] = BvhJoint(
                name=joint.name,
                parent=joint.parent,
                offset=joint.offset,
                channels=channels,
            )

    return joints


def _build_smpl_like_arrays(
    motion: BvhMotion,
    *,
    unit_scale: float,
    bvh_to_smpl_index: dict[str, int],
    identity_fill_smpl_indices: tuple[int, ...],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    frame_count = int(motion.values.shape[0])
    world_rot = Rotation.from_matrix(np.asarray(NOITOM_YUP_TO_ZUP_MATRIX, dtype=np.float64))
    root_orient = np.zeros((frame_count, 3), dtype=np.float32)
    trans = np.zeros((frame_count, 3), dtype=np.float32)
    pose24 = np.zeros((frame_count, 24, 3), dtype=np.float32)
    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}

    local_rotvec: dict[str, Any] = {}
    cursor = 0
    for joint in motion.joints:
        values = motion.values[:, cursor : cursor + len(joint.channels)]
        channel_to_col = {name: idx for idx, name in enumerate(joint.channels)}
        rotations = _rotation_from_channels(values, joint.channels)
        if rotations is None:
            rotvec = np.zeros((frame_count, 3), dtype=np.float32)
        else:
            rotvec = rotations.as_rotvec().astype(np.float32)
        local_rotvec[joint.name] = rotvec

        if joint.parent == -1:
            if all(name in channel_to_col for name in ("Xposition", "Yposition", "Zposition")):
                root_pos = np.stack(
                    [
                        values[:, channel_to_col["Xposition"]],
                        values[:, channel_to_col["Yposition"]],
                        values[:, channel_to_col["Zposition"]],
                    ],
                    axis=1,
                )
            else:
                root_pos = np.broadcast_to(np.asarray(joint.offset, dtype=np.float32), (frame_count, 3))
            trans[:, 0] = root_pos[:, 0] * unit_scale
            trans[:, 1] = -root_pos[:, 2] * unit_scale
            trans[:, 2] = root_pos[:, 1] * unit_scale
            source_root_rot = (
                Rotation.from_rotvec(rotvec.astype(np.float64))
                if rotations is not None
                else Rotation.identity(frame_count)
            )
            root_orient = (world_rot * source_root_rot).as_rotvec().astype(np.float32)
            pose24[:, 0, :] = root_orient
        cursor += len(joint.channels)

    mapped: dict[str, Any] = {}
    missing: dict[str, Any] = {}
    for bvh_name, smpl_idx in bvh_to_smpl_index.items():
        smpl_name = SMPL_24_JOINTS[smpl_idx]
        if bvh_name == "Hips":
            mapped[smpl_name] = {
                "smpl_index": smpl_idx,
                "bvh_joint": bvh_name,
                "rotation": "transformed_global_root",
            }
            continue
        if bvh_name not in name_to_index:
            missing[smpl_name] = {"smpl_index": smpl_idx, "expected_bvh_joint": bvh_name}
            continue
        pose24[:, smpl_idx, :] = local_rotvec[bvh_name]
        mapped[smpl_name] = {
            "smpl_index": smpl_idx,
            "bvh_joint": bvh_name,
            "rotation": "raw_bvh_local_rotation",
        }

    identity = {
        SMPL_24_JOINTS[idx]: {
            "smpl_index": idx,
            "reason": "No reliable BVH/SMPL counterpart; identity fill",
        }
        for idx in identity_fill_smpl_indices
    }
    for smpl_name, info in identity.items():
        mapped[smpl_name] = {
            "smpl_index": info["smpl_index"],
            "bvh_joint": None,
            "rotation": "identity_fill",
        }

    mapping_metadata = {
        "smpl_24_joints": list(SMPL_24_JOINTS),
        "bvh_to_smpl_mapping": mapped,
        "missing_or_identity_joints": {**missing, **identity},
        "identity_fill_smpl_indices": list(identity_fill_smpl_indices),
    }
    return root_orient, pose24[:, 1:, :].reshape(frame_count, 69), trans, mapping_metadata


def _repair_isolated_root_translation_spikes(trans: Any) -> tuple[Any, list[int]]:
    import numpy as np

    trans_arr = np.asarray(trans, dtype=np.float32)
    if trans_arr.ndim != 2 or trans_arr.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans_arr.shape}")
    if trans_arr.shape[0] < 3:
        return trans_arr.copy(), []

    repaired = trans_arr.copy()
    repaired_indices: list[int] = []
    jump = np.linalg.norm(np.diff(trans_arr, axis=0), axis=1) > 0.5
    idx = 0
    while idx < jump.shape[0]:
        if not jump[idx]:
            idx += 1
            continue
        block_start = idx + 1
        block_end_jump = idx + 1
        while block_end_jump < jump.shape[0] and not jump[block_end_jump]:
            block_end_jump += 1
        if block_end_jump < jump.shape[0]:
            block_end = block_end_jump
            left = block_start - 1
            right = block_end + 1
            block_len = block_end - block_start + 1
            neighbor_span = float(np.linalg.norm(trans_arr[right] - trans_arr[left]))
            if 1 <= block_len <= 12 and neighbor_span < 0.25:
                for frame_idx in range(block_start, block_end + 1):
                    alpha = (frame_idx - left) / float(right - left)
                    repaired[frame_idx] = (
                        (1.0 - alpha) * trans_arr[left] + alpha * trans_arr[right]
                    )
                    repaired_indices.append(frame_idx)
                idx = block_end_jump + 1
                continue
        idx += 1

    repaired_set = set(repaired_indices)
    for idx in range(1, trans_arr.shape[0] - 1):
        if idx in repaired_set:
            continue
        prev_jump = float(np.linalg.norm(trans_arr[idx] - trans_arr[idx - 1]))
        next_jump = float(np.linalg.norm(trans_arr[idx + 1] - trans_arr[idx]))
        neighbor_span = float(np.linalg.norm(trans_arr[idx + 1] - trans_arr[idx - 1]))
        if prev_jump > 0.5 and next_jump > 0.5 and neighbor_span < 0.25:
            repaired[idx] = 0.5 * (trans_arr[idx - 1] + trans_arr[idx + 1])
            repaired_indices.append(idx)

    return repaired.astype(np.float32), sorted(set(repaired_indices))


def _rotation_from_channels(values: Any, channels: tuple[str, ...]) -> Any | None:
    from scipy.spatial.transform import Rotation

    rot_channel_items = [
        (idx, channel[0].upper())
        for idx, channel in enumerate(channels)
        if channel.endswith("rotation")
    ]
    if not rot_channel_items:
        return None
    order = "".join(axis for _, axis in rot_channel_items)
    angles = values[:, [idx for idx, _ in rot_channel_items]]
    return Rotation.from_euler(order, angles, degrees=True)


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


def _detect_bvh_fields(header: str) -> list[str]:
    fields = []
    for name in ("HIERARCHY", "MOTION", "Frames", "Frame Time"):
        if name in header:
            fields.append(name)
    return fields


def _find_line_index(lines: list[str], expected: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == expected:
            return index
    return None


def _find_prefixed_line_index(lines: list[str], prefix: str, *, start: int = 0) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip().startswith(prefix):
            return index
    return None


def _source_subset(source_relative_path: str) -> str:
    parts = Path(source_relative_path).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return "unknown"
