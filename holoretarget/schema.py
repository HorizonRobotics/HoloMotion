"""Public HoloRetarget output schema."""

from __future__ import annotations


ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
DOF_POS_DIM = 29
QPOS_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM + DOF_POS_DIM

UNITREE_G1_29DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

if len(UNITREE_G1_29DOF_NAMES) != DOF_POS_DIM:
    raise RuntimeError(
        "HoloRetarget DoF schema must contain exactly 29 joints"
    )

__all__ = [
    "DOF_POS_DIM",
    "QPOS_DIM",
    "ROOT_POS_DIM",
    "ROOT_QUAT_DIM",
    "UNITREE_G1_29DOF_NAMES",
]
