from __future__ import annotations

import numpy as np


WORLD_GRAVITY = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def root_height_from_trans(human_root_trans: np.ndarray) -> np.ndarray:
    human_root_trans = np.asarray(human_root_trans, dtype=np.float32)
    if human_root_trans.ndim != 2 or human_root_trans.shape[1] != 3:
        raise ValueError(f"human_root_trans must be [T,3], got {human_root_trans.shape}")
    return human_root_trans[:, 2:3].astype(np.float32, copy=True)


def gravity_projection_from_root_orient(root_orient_aa: np.ndarray) -> np.ndarray:
    """Project world gravity into the root frame.

    This function imports scipy lazily so schema/CLI commands remain lightweight.
    """

    from scipy.spatial.transform import Rotation as R

    root = np.asarray(root_orient_aa, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_orient_aa must be [T,3], got {root.shape}")
    gravity = np.repeat(WORLD_GRAVITY.astype(np.float64)[None, :], root.shape[0], axis=0)
    return R.from_rotvec(root).inv().apply(gravity).astype(np.float32)


def derive_formal_fields(human_pose_aa: np.ndarray, human_root_trans: np.ndarray) -> dict[str, np.ndarray]:
    human_pose_aa = np.asarray(human_pose_aa, dtype=np.float32)
    if human_pose_aa.ndim != 2 or human_pose_aa.shape[1] != 72:
        raise ValueError(f"human_pose_aa must be [T,72], got {human_pose_aa.shape}")
    if human_root_trans.shape[0] != human_pose_aa.shape[0]:
        raise ValueError(
            f"pose/root_trans length mismatch: {human_pose_aa.shape[0]} vs {human_root_trans.shape[0]}"
        )
    return {
        "human_root_height": root_height_from_trans(human_root_trans),
        "human_gravity_projection": gravity_projection_from_root_orient(human_pose_aa[:, :3]),
    }
