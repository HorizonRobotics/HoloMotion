from __future__ import annotations

from typing import Any

import numpy as np

from holosmpl.core.schema.canonical import ALLOWED_BODY_POSE_LAYOUTS


def canonical_pose_to_human_pose_aa(
    root_orient_aa: np.ndarray,
    body_pose_aa: np.ndarray,
    body_pose_layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = np.asarray(root_orient_aa, dtype=np.float32)
    body = np.asarray(body_pose_aa, dtype=np.float32)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_orient_aa must be [T,3], got {root.shape}")
    if body.ndim != 2:
        raise ValueError(f"body_pose_aa must be [T,D], got {body.shape}")
    if body.shape[0] != root.shape[0]:
        raise ValueError(f"root/body length mismatch: {root.shape[0]} vs {body.shape[0]}")
    expected_dim = ALLOWED_BODY_POSE_LAYOUTS.get(body_pose_layout)
    if expected_dim is None:
        raise ValueError(f"Unsupported body_pose_layout: {body_pose_layout}")
    if body.shape[1] != expected_dim:
        raise ValueError(
            f"{body_pose_layout} expects D={expected_dim}, got D={body.shape[1]}"
        )

    t = int(root.shape[0])
    pose72 = np.zeros((t, 72), dtype=np.float32)
    pose72[:, :3] = root
    if body_pose_layout == "smpl_23_body":
        pose72[:, 3:72] = body
        metadata = {
            "formal_pose_72_policy": "concat_root_and_smpl_23_body",
            "formal_pose_72_padded_joints": [],
            "formal_pose_72_padded_dims": [],
        }
    elif body_pose_layout == "smplx_21_body":
        pose72[:, 3:66] = body
        metadata = {
            "formal_pose_72_policy": "pad_smpl_left_hand_right_hand_with_identity",
            "formal_pose_72_padded_joints": ["left_hand", "right_hand"],
            "formal_pose_72_padded_dims": [66, 72],
        }
    else:  # Defensive guard for future layout additions.
        raise ValueError(f"Unsupported body_pose_layout: {body_pose_layout}")
    return pose72, metadata
