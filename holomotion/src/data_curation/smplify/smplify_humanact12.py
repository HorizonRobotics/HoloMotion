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
import random

import h5py
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from thirdparties.joints2smpl.src import config
from thirdparties.joints2smpl.src.smplify import SMPLify3D

SMPL_MODEL_DIR = "./assets/smpl/"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

num_joints = 22
joint_category = "AMASS"
num_smplify_iters = 150
fix_foot = False


def joints2smpl(file_name, data_dir, save_dir):
    """Convert 3D joint positions to SMPL-X parameters.

    Args:
        file_name (str): Name of the input .npy joint file
        data_dir (str): Directory containing input joint files
        save_dir (str): Directory to save processed output files

    """
    # print(file_name)
    input_joints = np.load(os.path.join(data_dir, file_name))

    input_joints = input_joints[:, :, [0, 1, 2]]  # amass stands on x, y

    """XY at origin"""
    input_joints[..., [0, 1]] -= input_joints[0, 0, [0, 1]]

    """Put on Floor"""
    floor_height = input_joints[:, :, 2].min()
    input_joints[:, :, 2] -= floor_height

    batch_size = input_joints.shape[0]

    smplmodel = smplx.create(
        SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="npz",
        batch_size=batch_size,
    ).to(device)

    # ## --- load the mean pose as original ----
    smpl_mean_file = config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, "r")
    init_mean_pose = (
        torch.from_numpy(file["pose"][:])
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .float()
        .to(device)
    )
    init_mean_shape = (
        torch.from_numpy(file["shape"][:])
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .float()
        .to(device)
    )
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

    # # #-------------initialize SMPLify
    smplify = SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=batch_size,
        joints_category=joint_category,
        num_iters=num_smplify_iters,
        device=device,
    )

    keypoints_3d = torch.Tensor(input_joints).to(device).float()

    pred_betas = init_mean_shape
    pred_pose = init_mean_pose
    pred_cam_t = cam_trans_zero

    if joint_category == "AMASS":
        confidence_input = torch.ones(num_joints)
        # make sure the foot and ankle
        if fix_foot:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
    else:
        print("Such category not settle down!")

    (
        new_opt_vertices,
        new_opt_joints,
        new_opt_pose,
        new_opt_betas,
        new_opt_cam_t,
        new_opt_joint_loss,
    ) = smplify(
        pred_pose.detach(),
        pred_betas.detach(),
        pred_cam_t.detach(),
        keypoints_3d,
        conf_3d=confidence_input.to(device),
        # seq_ind=idx
    )

    poses = new_opt_pose.detach().cpu().numpy()
    betas = new_opt_betas.mean(axis=0).detach().cpu().numpy()
    trans = keypoints_3d[:, 0].detach().cpu().numpy()

    target_dim = 165
    current_dim = poses.shape[-1]
    pad_dim = target_dim - current_dim

    if pad_dim > 0:
        pad_array = np.zeros((*poses.shape[:-1], pad_dim), dtype=poses.dtype)
        poses = np.concatenate([poses, pad_array], axis=-1)

    root_orient = poses[:, :3]
    root_mat = Rotation.from_rotvec(root_orient).as_matrix()
    rx_minus_100 = Rotation.from_euler("x", -100, degrees=True).as_matrix()
    align_r = rx_minus_100 @ root_mat
    align_axis_angle = Rotation.from_matrix(align_r).as_rotvec()
    poses[:, :3] = align_axis_angle
    trans_rotated = rx_minus_100 @ (trans.T)
    trans_rotated = trans_rotated.T
    input_joints = input_joints[:, :, [0, 2, 1]]  # jts stands on x, z
    input_joints[..., 0] *= -1
    param = {
        "poses": poses,
        "trans": trans_rotated,
        "betas": betas,
        "gender": "neutral",
        "jtr": input_joints,
        "mocap_frame_rate": 30,
    }
    file_name = file_name.split(".")[0] + ".npz"
    print(file_name)
    np.savez_compressed(os.path.join(save_dir, file_name), **param)


def humanact12_to_amass(data_dir, save_dir):
    """Convert HumanAct12 dataset to AMASS-compatible format.

    Args:
        data_dir (str): Directory containing HumanAct12 .npy joint files
        save_dir (str): Directory to save processed AMASS .npz files

    """
    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(data_dir)
    random.shuffle(file_list)
    for file_name in tqdm(file_list):
        if os.path.exists(os.path.join(save_dir, file_name)):
            print(f"{os.path.join(save_dir, file_name)} already exists")
            continue
        joints2smpl(file_name, data_dir, save_dir)


if __name__ == "__main__":
    data_dir = ""
    save_dir = ""
