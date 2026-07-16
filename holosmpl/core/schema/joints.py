from __future__ import annotations


SMPL_24_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPLX_21_BODY_JOINT_NAMES = SMPL_24_JOINT_NAMES[1:22]
SMPL_23_BODY_JOINT_NAMES = SMPL_24_JOINT_NAMES[1:24]
SMPLX_21_TO_SMPL_23_PADDED_JOINTS = ("left_hand", "right_hand")


def smpl_joint_summary() -> dict[str, object]:
    return {
        "smpl_24_joint_names": list(SMPL_24_JOINT_NAMES),
        "smplx_21_body_joint_names": list(SMPLX_21_BODY_JOINT_NAMES),
        "smpl_23_body_joint_names": list(SMPL_23_BODY_JOINT_NAMES),
        "smplx_21_to_smpl_23_padded_joints": list(SMPLX_21_TO_SMPL_23_PADDED_JOINTS),
    }
