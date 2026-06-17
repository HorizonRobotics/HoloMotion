# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# representation: 272 dim
# :2 local xz velocities of root, no heading, can recover translation
# 2:8  heading angular velocities, 6d rotation, can recover heading
# 8:8+3*njoint local position, no heading, all at xz origin
# 8+3*njoint:8+6*njoint local velocities, no heading, all at xz origin, can recover local postion
# 8+6*njoint:8+12*njoint local rotations, 6d rotation, no heading, all frames z+

import argparse
import os
from glob import glob
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def _skew(v):
    x, y, z = v
    return np.array(
        [[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64
    )


def expmap_to_R(r: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix (3, 3)."""
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = r / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def R_to_expmap(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    # Numerically robust handling
    R = R.astype(np.float64)
    # Rotation angle
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float32)

    # Standard formula: k = (1/(2 sinθ)) * [R32-R23, R13-R31, R21-R12]
    denom = 2 * np.sin(theta)
    k = np.array(
        [
            (R[2, 1] - R[1, 2]) / denom,
            (R[0, 2] - R[2, 0]) / denom,
            (R[1, 0] - R[0, 1]) / denom,
        ],
        dtype=np.float64,
    )

    # Normalize and return axis-angle
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    k = k / k_norm
    return (k * theta).astype(np.float32)


def euler_to_R_xyz(
    euler_deg: Tuple[float, float, float], order: str = "xyz"
) -> np.ndarray:
    """Euler angles (degrees) -> rotation matrix; order e.g. 'xyz', 'zyx' (extrinsic, right-multiply in sequence)."""
    ax = np.deg2rad(euler_deg[0])
    ay = np.deg2rad(euler_deg[1])
    az = np.deg2rad(euler_deg[2])

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    for ch in order.lower():
        if ch == "x":
            R = R @ Rx(ax)
        elif ch == "y":
            R = R @ Ry(ay)
        elif ch == "z":
            R = R @ Rz(az)
        else:
            raise ValueError(f"Unsupported Euler order char: {ch}")
    return R


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d: (..., 6)
        Encodes rotation using the FIRST TWO ROWS of a rotation matrix,
        flattened row-major (consistent with rotations_matrix[..., :2, :]).
    return: (..., 3, 3) rotation matrix
    """
    # first two rows
    r1 = rot6d[..., 0:3]
    r2 = rot6d[..., 3:6]

    # normalize r1
    r1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)

    # make r2 orthogonal to r1
    proj = np.sum(r1 * r2, axis=-1, keepdims=True) * r1
    r2 = r2 - proj
    r2 = r2 / (np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-8)

    # third row by right-hand rule
    r3 = np.cross(r1, r2, axis=-1)

    # stack as ROWS
    R = np.stack([r1, r2, r3], axis=-2)  # (..., 3, 3)
    return R


def rotate_npz_upright(
    trans_world: np.ndarray,  # (T,3)
    root_orient_aa: np.ndarray,  # (T,3)
    euler_deg: Tuple[float, float, float] = (90.0, 0.0, 0.0),
    order: str = "xyz",
    up_axis: str = "z",  # 'z', 'y', or 'x'; defines the up axis
):
    N = trans_world.shape[0]

    # World-frame rotation matrix
    R_fix = euler_to_R_xyz(euler_deg, order=order)

    axis_map = {"x": 0, "y": 1, "z": 2}
    if up_axis.lower() not in axis_map:
        raise ValueError("up_axis must be one of 'x','y','z'")

    # Left-multiply root orientation by world-frame rotation
    root_orient_new = np.empty_like(root_orient_aa)
    for i in range(N):
        R_old = expmap_to_R(root_orient_aa[i])
        R_new = R_fix @ R_old
        root_orient_new[i] = R_to_expmap(R_new)

    root_trans_world_rot = np.stack(
        [R_fix @ trans_world[i] for i in range(trans_world.shape[0])],
        axis=0,
    )

    return root_trans_world_rot, root_orient_new


def convert_272_to_smpl_like(np_file, out_file):
    x = np.load(np_file)  # (T, 272)
    T = x.shape[0]

    njoint = 22
    pos_start = 8
    pos_end = pos_start + 3 * njoint  # 8 : 74

    joint_rot_start = 8 + 6 * njoint
    joint_rot_end = 8 + 12 * njoint  # 140:272

    # joint local positions
    joint_pos = x[:, pos_start:pos_end].reshape(T, njoint, 3)

    # Recover root rotation
    # Δyaw 6D -> ΔR_yaw
    root_rot_delta_yaw_6D = x[:, 2:8]  # (T, 6)
    root_rot_delta_yaw_R = rot6d_to_matrix(root_rot_delta_yaw_6D)  # (T, 3, 3)

    yaw0 = 0.0
    R0 = R.from_euler("y", yaw0, degrees=False).as_matrix()  # (3, 3)

    R_yaw_abs = np.zeros_like(root_rot_delta_yaw_R)
    R_yaw_abs[0] = R0
    for t in range(1, T):
        R_yaw_abs[t] = root_rot_delta_yaw_R[t] @ R_yaw_abs[t - 1]  # Integrate to obtain world-frame yaw

    # Root rotation without heading
    joint_rot_6D = x[:, joint_rot_start:joint_rot_end].reshape(T, njoint, 6)

    # remaining 21 joints → flattened pose_body (T, 63)
    pose_body = joint_rot_6D[:, 1:, :].reshape(T, -1)
    R_body_mat = rot6d_to_matrix(joint_rot_6D[:, 1:, :])  # (T,21,3,3)
    pose_body_aa = (
        R.from_matrix(R_body_mat.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(T, 21, 3)
    )  # (T,21,3)
    pose_body = pose_body_aa.reshape(T, 63).astype(np.float32)

    root_rot_noheading_6D = joint_rot_6D[:, 0, :]  # (T, 6)
    root_rot_noheading_R = rot6d_to_matrix(root_rot_noheading_6D)

    # Apply yaw: R_world_root = R_yaw^{-1} @ root_rot_noheading_R
    R_world_root = np.matmul(
        np.transpose(R_yaw_abs, (0, 2, 1)), root_rot_noheading_R
    )  # (T,3,3)

    # axis-angle:
    root_orient_aa = R.from_matrix(R_world_root).as_rotvec().astype(np.float32)  # (T,3)

    # pelvis / root joint → trans
    v_xz_nohead = x[:, 0:2].astype(np.float64)  # (T, 2)   local xz velocities of root, no heading
    v_xyz = np.zeros((T, 3), dtype=np.float64)  # (T, 3)
    v_xyz[:, 0] = v_xz_nohead[:, 0]
    v_xyz[:, 2] = v_xz_nohead[:, 1]

    v_world = np.einsum(
        "tij,tj->ti", np.transpose(R_yaw_abs, (0, 2, 1)), v_xyz
    )  # (T,3)  v_world[t] = A[t] @ v3[t]

    x_world = np.zeros((T,), dtype=np.float64)
    z_world = np.zeros((T,), dtype=np.float64)
    x_world[0] = 0.0
    z_world[0] = 0.0
    for t in range(1, T):
        x_world[t] = x_world[t - 1] + v_world[t, 0]
        z_world[t] = z_world[t - 1] + v_world[t, 2]

    trans = joint_pos[:, 0, :]  # (T, 3)

    trans_world = trans.astype(np.float64).copy()
    trans_world[:, 0] = x_world
    trans_world[:, 2] = z_world
    trans_world = trans_world.astype(np.float32)

    # Y-up -> Z-up conversion (your matrix)
    trans_world, root_orient_aa = rotate_npz_upright(
        trans_world=trans_world,
        root_orient_aa=root_orient_aa,
        euler_deg=(90.0, 0.0, 0.0),
        order="xyz",
        up_axis="z",
    )

    # betas: 16-dim all zeros
    betas = np.zeros((16,), dtype=np.float32)

    np.savez_compressed(
        out_file,
        betas=betas.astype(np.float32),
        trans=trans_world.astype(np.float32),  # (T, 3)
        pose_body=pose_body.astype(np.float32),  # (T, 63)
        root_orient=root_orient_aa.astype(np.float32),
        mocap_frame_rate=30,
        gender="neutral",
    )


def motionmillion_to_amass(src_root, dst_root):
    """Convert MotionMillion 272-dim .npy files to AMASS-compatible .npz."""
    os.makedirs(dst_root, exist_ok=True)
    npy_files = glob(os.path.join(src_root, "**/*.npy"), recursive=True)
    for f in tqdm(npy_files, desc="MotionMillion to AMASS", unit="file"):
        rel = os.path.relpath(f, src_root)
        rel_dir = os.path.dirname(rel)
        base = os.path.basename(rel)
        stem, _ = os.path.splitext(base)

        prefix = "MotionMillion_"
        new_name = prefix + stem + ".npz"

        out = os.path.join(dst_root, rel_dir, new_name)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        convert_272_to_smpl_like(f, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        type=str,
        required=True,
        help="source .npy file or directory",
    )
    parser.add_argument(
        "--tgt_folder",
        type=str,
        required=True,
        help="target .npz file or directory",
    )
    args = parser.parse_args()

    # Single file
    if os.path.isfile(args.src_folder):
        os.makedirs(os.path.dirname(args.tgt_folder), exist_ok=True)
        convert_272_to_smpl_like(args.src_folder, args.tgt_folder)

    # Batch directory processing
    else:
        motionmillion_to_amass(args.src_folder, args.tgt_folder)
