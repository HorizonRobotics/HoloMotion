# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import os

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def zju_single_to_amass(
    param_dir, out_path, gender="neutral", fps=30, rotate=False
):
    """Convert .npy files into a single AMASS-style .npz file.

    Args:
        param_dir: Folder containing 0.npy, 1.npy, ....
        out_path: Output .npz path.
        gender: Gender to assign ('neutral', 'male', 'female').
        fps: Mocap frame rate.
        rotate: whether or not rotate the body

    """
    pose_list = []
    trans_list = []
    shape_list = []

    # Get sorted list of npy files
    files = sorted(
        [f for f in os.listdir(param_dir) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    if rotate:
        ry_minus_180 = Rotation.from_euler("y", -180, degrees=True).as_matrix()
    else:
        ry_minus_180 = Rotation.from_euler("y", 0, degrees=True).as_matrix()
    for fname in tqdm(files, desc="Processing frames"):
        fpath = os.path.join(param_dir, fname)
        data = np.load(fpath, allow_pickle=True).item()

        poses = data["poses"]  # (1, 72)
        global_orient = data["Rh"]

        root_orient = global_orient
        root_mat = Rotation.from_rotvec(root_orient).as_matrix()
        align_r = ry_minus_180 @ root_mat
        align_axis_angle = Rotation.from_matrix(align_r).as_rotvec()
        global_orient = align_axis_angle
        body_pose = poses[:, 3:66]
        hand_pose = poses[:, 66:72]

        full_pose = np.concatenate(
            [global_orient, body_pose, hand_pose], axis=1
        )  # (1, 165)

        pose_list.append(full_pose[0])  # shape: (165,)
        trans_list.append(data["Th"][0])  # shape: (3,)
        shape_list.append(data["shapes"][0])  # shape: (10,)

    poses = np.stack(pose_list, axis=0).astype(np.float32)  # (N, 165)
    trans = np.stack(trans_list, axis=0).astype(np.float32)  # (N, 3)
    trans_rotated = ry_minus_180 @ (trans.T)
    trans_rotated = trans_rotated.T
    betas = shape_list[0].astype(np.float32)  # (10,) same for all frames

    # Save as AMASS-style npz
    np.savez_compressed(
        out_path,
        poses=poses,
        trans=trans_rotated,
        betas=betas,
        gender=gender,
        mocap_frame_rate=fps,
    )

    print(f"Saved AMASS-style file to: {out_path}")
    print(f"Total frames: {poses.shape[0]}")


def zju_to_amass(input_dir, output_dir):
    """Convert multiple ZJU-formatted folders to AMASS-style .npz files.

    Args:
        input_dir: Path to ZJU dataset root folder.
        output_dir: Path to save AMASS-format .npz files.

    """
    os.makedirs(output_dir, exist_ok=True)

    subjects = sorted(
        [
            d
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]
    )

    for subject in subjects:
        subject_dir = os.path.join(input_dir, subject)

        new_params_dir = os.path.join(subject_dir, "new_params")
        params_dir = os.path.join(subject_dir, "params")

        if os.path.isdir(new_params_dir):
            param_dir = new_params_dir
            print(f"[{subject}] Using new_params")
        elif os.path.isdir(params_dir):
            param_dir = params_dir
            print(f"[{subject}] Using params")
        else:
            print(f"[{subject}] No params found, skipping")
            continue

        out_path = os.path.join(output_dir, f"{subject}.npz")
        zju_single_to_amass(param_dir, out_path)

    print(f"All subjects processed. Output saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    zju_to_amass(
        param_dir="",
        out_path="",
    )
