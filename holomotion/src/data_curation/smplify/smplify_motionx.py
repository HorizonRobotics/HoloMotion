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


def motionx_to_amass(src_root, dst_root):
    """Convert MotionX format motion data to AMASS format.

    Args:
        src_root (str): Source directory containing MotionX .npy files
        dst_root (str): Destination directory for processed AMASS .npz files

    Side effects:
        Creates directory structure mirroring src_root under dst_root
        Generates compressed .npz files in destination directory
        Prints file paths of processed files

    Processed data contains:
        poses: [T, 165] float array of joint rotations (root first)
        trans: [T, 3] float array of root translations
        betas: [10] float array of shape parameters
        gender: str (always "neutral")
        mocap_frame_rate: int (always 30)

    """
    os.makedirs(dst_root, exist_ok=True)
    for root, _, files in os.walk(src_root):
        # print(files)
        for file in files:
            src_file_path = os.path.join(root, file)
            motion = np.load(src_file_path)
            poses = motion[:, :156]  # 最终 shape: (T, 156)
            num_frames = poses.shape[0]
            sl = poses.shape[1]

            pad = np.zeros((num_frames, 165 - sl), dtype=poses.dtype)  # (T, 9)
            poses = np.concatenate([poses, pad], axis=1)  # (T, 165)
            align_axis_angle = poses[:, :3]
            root_orient = poses[:, :3]
            root_mat = Rotation.from_rotvec(root_orient).as_matrix()
            rotate_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            align_r = rotate_matrix @ root_mat
            align_axis_angle = Rotation.from_matrix(align_r).as_rotvec()
            poses[:, :3] = align_axis_angle

            trans = motion[:, 309:312]  # (T, 3)
            trans[:, 2] = trans[:, 2] * (-1)
            trans_matrix = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            )
            trans = np.dot(trans, trans_matrix)
            trans = rotate_matrix @ (trans.T)
            trans = trans.T
            betas = motion[0, 312:]
            amass_data = {
                "poses": poses,
                "trans": trans,
                "betas": betas,
                "gender": "neutral",
                "mocap_frame_rate": 30,
            }
            relative_path = src_file_path.replace(src_root, "")
            file_name = dst_root + relative_path
            save_dir = file_name.split("/")[-1]
            save_dir = file_name.replace(save_dir, "")
            os.makedirs(save_dir, exist_ok=True)
            file_name = file_name.replace(".npy", ".npz")
            print(file_name)
            np.savez_compressed(file_name, **amass_data)


if __name__ == "__main__":
    src_root = "./data/smplx_322"
    dst_root = "./data/smplx_data/MotionX"
    motionx_to_amass(src_root, dst_root)
