from __future__ import annotations

from pathlib import Path
from typing import Any

from holosmpl.converters.bvh_family.noitom import (
    classify_noitom_bvh_source,
    load_bvh_motion,
    _rotation_from_channels,
)
from holosmpl.core.processing.resample import resample_motion_to_fps
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)


LAFAN1_SOURCE_COORDINATE_FRAME = "lafan1_bvh_y_up_xyz_cm"
LAFAN1_CANONICAL_TRANSFORM = (
    "lafan1_fk_yup_cm_to_canonical_zup_m_then_lafan_to_smplx_frame_offsets"
)
LAFAN1_UNIT_SCALE = 0.01
LAFAN1_DROP_FIRST_FRAME = False
LAFAN1_BETA = (
    1.4775,
    0.6674,
    -1.1742,
    0.4731,
    1.2984,
    -0.2159,
    1.5276,
    -0.3152,
    -0.6441,
    -0.2986,
    0.5089,
    -0.6354,
    0.3321,
    -0.1099,
    -0.3060,
    -0.7330,
)
LAFAN1_YUP_TO_ZUP_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)

# The frame-offset mapping and beta values are adapted from:
#   https://github.com/jaraujo98/lafan_to_smplx
#   commit d465aa7202ddc94f2cd557b682854437573e0dca
# Copyright (c) 2025 Joao Pedro Araujo, MIT License.
# See LICENSE_lafan_to_smplx.txt in this package.
SMPLX_21_BODY_TO_LAFAN = (
    ("left_hip", "LeftUpLeg"),
    ("right_hip", "RightUpLeg"),
    ("spine1", "Spine"),
    ("left_knee", "LeftLeg"),
    ("right_knee", "RightLeg"),
    ("spine2", "Spine1"),
    ("left_ankle", "LeftFoot"),
    ("right_ankle", "RightFoot"),
    ("spine3", "Spine2"),
    ("left_foot", "LeftToe"),
    ("right_foot", "RightToe"),
    ("neck", "Neck"),
    ("left_collar", "LeftShoulder"),
    ("right_collar", "RightShoulder"),
    ("head", "Head"),
    ("left_shoulder", "LeftArm"),
    ("right_shoulder", "RightArm"),
    ("left_elbow", "LeftForeArm"),
    ("right_elbow", "RightForeArm"),
    ("left_wrist", "LeftHand"),
    ("right_wrist", "RightHand"),
)


def classify_lafan1_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one LaFAN1 BVH is convertible."""

    return classify_noitom_bvh_source(source_path)


def convert_lafan1_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Convert one LaFAN1 BVH into canonical SMPL-X-like arrays.

    This adapter follows the lafan_to_smplx frame-offset mapping instead of
    directly copying BVH local rotations. LaFAN1 and SMPL-X have comparable
    joint semantics, but their joint local frames differ substantially.
    """

    import numpy as np

    source_path = Path(source_path)
    input_root = Path(input_root)
    motion = load_bvh_motion(source_path)
    source_fps = 1.0 / motion.frame_time
    source_frame_count_raw = int(motion.values.shape[0])
    root_orient, pose_body, trans, mapping_metadata = _build_smplx_like_arrays(motion)

    if LAFAN1_DROP_FIRST_FRAME:
        root_orient = root_orient[1:]
        pose_body = pose_body[1:]
        trans = trans[1:]

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

    betas = np.asarray(LAFAN1_BETA, dtype=np.float32)
    metadata = {
        "schema_version": "canonical_smpl_v1",
        "dataset": "LaFAN1",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_subset": _source_subset(source_relative_path),
        "source_format": "bvh",
        "source_fields": ["bvh_hierarchy", "bvh_motion"],
        "source_fps": float(source_fps),
        "source_frame_time": float(motion.frame_time),
        "source_frame_count_raw": source_frame_count_raw,
        "target_fps": float(target_fps),
        "pose_body_dim": 63,
        "pose_body_layout": "smplx_21_body",
        "smpl_source_type": "bvh_to_smplx_frame_offset_approx",
        "human_representation": "smplx_21_body_from_lafan1_bvh_frame_offsets",
        "is_strict_smpl": False,
        "approximation_warning": (
            "Approximate LaFAN1 BVH-to-SMPL-X mapping using fixed joint frame "
            "offsets and precomputed SMPL-X beta; not optimized per-frame fitting."
        ),
        "betas_policy": "lafan_to_smplx_precomputed_beta",
        "source_native_beta_present": False,
        "gender": "neutral",
        "drop_first_frame": bool(LAFAN1_DROP_FIRST_FRAME),
        "unit_scale": float(LAFAN1_UNIT_SCALE),
        "source_coordinate_system": LAFAN1_SOURCE_COORDINATE_FRAME,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "coordinate_transform": LAFAN1_CANONICAL_TRANSFORM,
        "coordinate_transform_matrix": [list(row) for row in LAFAN1_YUP_TO_ZUP_MATRIX],
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "root_frame_semantics": "smplx_root_frame_from_lafan_to_smplx_offsets",
        "root_orient_policy": "canonical_lafan_global_hips_rotation_then_lafan_to_smplx_hips_offset",
        "root_trans_policy": "canonical_lafan_global_hips_position_m",
        "body_pose_policy": "lafan_to_smplx_global_frame_offset_to_local_smplx_body_pose",
        "root_frame_certified": False,
        "source_joint_count": len(motion.joints),
        "source_channel_count": int(motion.values.shape[1]),
        "source_root_joint": motion.joints[0].name if motion.joints else None,
        "source_joint_names": [joint.name for joint in motion.joints],
        "source_joint_channel_counts": {
            joint.name: len(joint.channels) for joint in motion.joints
        },
        "source_root_channels": list(motion.joints[0].channels) if motion.joints else [],
        "third_party_algorithm": {
            "name": "lafan_to_smplx",
            "repository": "https://github.com/jaraujo98/lafan_to_smplx",
            "commit": "d465aa7202ddc94f2cd557b682854437573e0dca",
            "license": "MIT",
        },
        **mapping_metadata,
        **resample_metadata,
    }

    return {
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas,
        "gender": "neutral",
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "metadata": metadata,
    }


def _build_smplx_like_arrays(motion: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    frame_count = int(motion.values.shape[0])
    source_to_canonical = Rotation.from_matrix(
        np.asarray(LAFAN1_YUP_TO_ZUP_MATRIX, dtype=np.float64)
    )
    local_rot, local_pos = _local_rotations_and_positions(motion)
    global_rot, global_pos = _forward_kinematics(motion, local_rot, local_pos)
    global_rot = {
        name: source_to_canonical * rot for name, rot in global_rot.items()
    }
    global_pos = {
        name: source_to_canonical.apply(pos) * LAFAN1_UNIT_SCALE
        for name, pos in global_pos.items()
    }

    frame_offsets = _lafan_frame_offsets()
    root_orient = (global_rot["Hips"] * frame_offsets["Hips"]).as_rotvec().astype(np.float32)
    pose_body = np.zeros((frame_count, 21, 3), dtype=np.float32)
    for smplx_index, (smplx_joint, lafan_joint) in enumerate(SMPLX_21_BODY_TO_LAFAN):
        parent_name = motion.joints[_joint_index(motion, lafan_joint)].parent
        if parent_name < 0:
            raise ValueError(f"LaFAN body joint unexpectedly has no parent: {lafan_joint}")
        lafan_parent = motion.joints[parent_name].name
        parent_frame = global_rot[lafan_parent] * frame_offsets[lafan_parent]
        joint_frame = global_rot[lafan_joint] * frame_offsets[lafan_joint]
        pose_body[:, smplx_index, :] = (parent_frame.inv() * joint_frame).as_rotvec().astype(
            np.float32
        )

    metadata = {
        "smplx_21_body_to_lafan_mapping": [
            {
                "smplx_body_index": index,
                "smplx_joint": smplx_joint,
                "lafan_joint": lafan_joint,
            }
            for index, (smplx_joint, lafan_joint) in enumerate(SMPLX_21_BODY_TO_LAFAN)
        ],
        "lafan_frame_offset_joint_names": sorted(frame_offsets),
        "identity_fill_smpl_pose72_indices": [66, 69, 69, 72],
        "pose_body_layout_source": "SMPL-X NUM_BODY_JOINTS=21 order",
    }
    return root_orient, pose_body.reshape(frame_count, 63), global_pos["Hips"], metadata


def _local_rotations_and_positions(motion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    frame_count = int(motion.values.shape[0])
    local_rot: dict[str, Any] = {}
    local_pos: dict[str, Any] = {}
    cursor = 0
    for joint in motion.joints:
        values = motion.values[:, cursor : cursor + len(joint.channels)]
        channel_to_col = {name: idx for idx, name in enumerate(joint.channels)}
        rotations = _rotation_from_channels(values, joint.channels)
        local_rot[joint.name] = rotations if rotations is not None else Rotation.identity(frame_count)
        if joint.parent == -1 and all(
            name in channel_to_col for name in ("Xposition", "Yposition", "Zposition")
        ):
            local_pos[joint.name] = np.stack(
                [
                    values[:, channel_to_col["Xposition"]],
                    values[:, channel_to_col["Yposition"]],
                    values[:, channel_to_col["Zposition"]],
                ],
                axis=1,
            ).astype(np.float32)
        else:
            local_pos[joint.name] = np.broadcast_to(
                np.asarray(joint.offset, dtype=np.float32), (frame_count, 3)
            ).copy()
        cursor += len(joint.channels)
    return local_rot, local_pos


def _forward_kinematics(
    motion: Any,
    local_rot: dict[str, Any],
    local_pos: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    frame_count = int(motion.values.shape[0])
    global_rot: dict[str, Any] = {}
    global_pos: dict[str, Any] = {}
    for joint in motion.joints:
        if joint.parent == -1:
            global_rot[joint.name] = local_rot[joint.name]
            global_pos[joint.name] = local_pos[joint.name]
            continue
        parent = motion.joints[joint.parent].name
        global_rot[joint.name] = global_rot[parent] * local_rot[joint.name]
        global_pos[joint.name] = (
            global_rot[parent].apply(local_pos[joint.name])
            + global_pos[parent]
        ).astype(np.float32)
    if not global_rot:
        raise ValueError("LaFAN1 BVH has no joints")
    if any(len(rot) != frame_count for rot in global_rot.values()):
        raise ValueError("LaFAN1 FK produced inconsistent rotation lengths")
    return global_rot, global_pos


def _lafan_frame_offsets() -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    return {
        "world": Rotation.from_euler("x", 0.0),
        "Hips": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "LeftUpLeg": Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2),
        "LeftLeg": Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2),
        "LeftFoot": Rotation.from_euler("z", 0.37117860986509)
        * Rotation.from_euler("y", np.pi / 2),
        "LeftToe": Rotation.from_euler("y", np.pi / 2),
        "RightUpLeg": Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2),
        "RightLeg": Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2),
        "RightFoot": Rotation.from_euler("z", 0.37117860986509)
        * Rotation.from_euler("y", np.pi / 2),
        "RightToe": Rotation.from_euler("y", np.pi / 2),
        "Spine": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "Spine1": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "Spine2": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "Neck": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "Head": Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2),
        "LeftShoulder": Rotation.from_euler("x", np.pi / 2),
        "LeftArm": Rotation.from_euler("x", np.pi / 2),
        "LeftForeArm": Rotation.from_euler("x", np.pi / 2),
        "LeftHand": Rotation.from_euler("x", np.pi / 2),
        "RightShoulder": Rotation.from_euler("z", np.pi) * Rotation.from_euler("x", -np.pi / 2),
        "RightArm": Rotation.from_euler("z", np.pi) * Rotation.from_euler("x", -np.pi / 2),
        "RightForeArm": Rotation.from_euler("z", np.pi) * Rotation.from_euler("x", -np.pi / 2),
        "RightHand": Rotation.from_euler("z", np.pi) * Rotation.from_euler("x", -np.pi / 2),
    }


def _joint_index(motion: Any, name: str) -> int:
    for index, joint in enumerate(motion.joints):
        if joint.name == name:
            return index
    raise ValueError(f"LaFAN1 BVH missing required joint: {name}")


def _validate_canonical_arrays(*, root_orient: Any, pose_body: Any, trans: Any) -> None:
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
    for name, array in {
        "root_orient": root_orient,
        "pose_body": pose_body,
        "trans": trans,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _source_subset(source_relative_path: str) -> str:
    parts = Path(source_relative_path).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return "unknown"
