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
#
# -----------------------------------------------------------------------------
# Portions of this file are derived from omomo_release (https://github.com/lijiaman/omomo_release).
# The original omomo_release code is licensed under the MIT license.
# -----------------------------------------------------------------------------

import os

import numpy as np
import pytorch3d.transforms as transforms
import torch
from torch.utils import data

from thirdparties.omomo_release.manip.data.hand_foot_dataset import (
    HandFootManipDataset,
    quat_ik_torch,
)


class MyHandFootManipDataset(HandFootManipDataset):
    """Modified dataset class for hand-foot manipulation tasks.

    This class overrides the __getitem__ method.

    """

    def __init__(self, *args, **kwargs):
        """Initialize the dataset instance by forwarding all arguments.

        This constructor ensures proper initialization
        of the HandFootManipDataset parent class.
        All parameters and keyword arguments are passed through unchanged.

        Args:
            *args: Variable length argument list for parent class
            **kwargs: Arbitrary keyword arguments for parent class

        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Retrieve and process a data sample by index.

        Try not to padding when retrieve motion data.

        Args:
            index (int): Index of the sample to retrieve

        Reference:
            https://github.com/lijiaman/omomo_release/blob/main/manip/data/hand_foot_dataset.py

        """
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        object_name = seq_name.split("_")[1]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        if self.use_object_splits:
            ori_w_idx = self.window_data_dict[index]["ori_w_idx"]
            obj_bps_npy_path = os.path.join(
                self.dest_obj_bps_npy_folder,
                seq_name + "_" + str(ori_w_idx) + ".npy",
            )
        else:
            obj_bps_npy_path = os.path.join(
                self.dest_obj_bps_npy_folder,
                seq_name + "_" + str(index) + ".npy",
            )
        obj_bps_data = np.load(obj_bps_npy_path)  # T X N X 3
        obj_bps_data = torch.from_numpy(obj_bps_data)

        num_joints = 24

        normalized_jpos = self.normalize_jpos_min_max(
            data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)
        )  # T X 22 X 3

        global_joint_rot = data_input[:, 2 * num_joints * 3 :]  # T X (22*6)

        new_data_input = torch.cat(
            (normalized_jpos.reshape(-1, num_joints * 3), global_joint_rot),
            dim=1,
        )
        ori_data_input = torch.cat(
            (data_input[:, : num_joints * 3], global_joint_rot), dim=1
        )

        # Add padding.
        actual_steps = new_data_input.shape[0]
        # pass
        paded_new_data_input = new_data_input
        paded_ori_data_input = ori_data_input

        paded_obj_bps = obj_bps_data.reshape(actual_steps, -1)
        paded_obj_com_pos = torch.from_numpy(
            self.window_data_dict[index]["window_obj_com_pos"]
        ).float()

        paded_obj_rot_mat = torch.from_numpy(
            self.window_data_dict[index]["obj_rot_mat"]
        ).float()
        paded_obj_scale = torch.from_numpy(
            self.window_data_dict[index]["obj_scale"]
        ).float()
        paded_obj_trans = torch.from_numpy(
            self.window_data_dict[index]["obj_trans"]
        ).float()

        if object_name in ["mop", "vacuum"]:
            paded_obj_bottom_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_bottom_rot_mat"]
            ).float()
            paded_obj_bottom_scale = torch.from_numpy(
                self.window_data_dict[index]["obj_bottom_scale"]
            ).float()
            paded_obj_bottom_trans = (
                torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_trans"]
                )
                .float()
                .squeeze(-1)
            )
        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["obj_bps"] = paded_obj_bps
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        if object_name in ["mop", "vacuum"]:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_bottom_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_bottom_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_bottom_trans
        else:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_trans

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        return data_input_dict


def run_smplx_model(root_trans, aa_rot_rep, betas, gender, fname):
    """Prepare and save SMPL-X motion data in AMASS npz format.

    Processes input motion parameters into SMPL-X compatible format and saves
    as a compressed npz file.

    Args:
        root_trans (torch.Tensor): Root translations [BS, T, 3]
        aa_rot_rep (torch.Tensor): Axis-angle joint rotations
        [BS, T, num_joints, 3]
            where num_joints can be either 22 (body only) or 52 (body+hands)
        betas (torch.Tensor): Shape parameters [BS, 16]
        gender (list): Gender strings for each sample in batch [BS]
        fname (str): Output filename/path for saving .npz file

    Output npz file contains:
        poses: [BS*T, 165] float array of pose parameters
        trans: [BS*T, 3] float array of translations
        betas: [16] float array of shape parameters (from first sample)
        gender: str (always "neutral")
        mocap_frame_rate: int (always 30)

    """
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(
            aa_rot_rep.device
        )  # BS X T X 30 X 3
        aa_rot_rep = torch.cat(
            (aa_rot_rep, padding_zeros_hand), dim=2
        )  # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(
        bs * num_steps, -1, 3
    )  # (BS*T) X n_joints X 3
    betas = (
        betas[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)
    )  # (BS*T) X 16
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
    # print(smpl_root_orient.shape)
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90
    poses = torch.cat(
        [smpl_root_orient, smpl_pose_body, smpl_pose_hand], dim=-1
    )
    target_dim = 165
    current_dim = poses.shape[-1]
    pad_dim = target_dim - current_dim

    if pad_dim > 0:
        pad_tensor = torch.zeros(
            *poses.shape[:-1], pad_dim, device=poses.device, dtype=poses.dtype
        )
        poses_padded = torch.cat([poses, pad_tensor], dim=-1)
    else:
        poses_padded = poses  # already 165 or more

    amass_data = {
        "poses": poses_padded.detach().cpu().numpy(),
        "trans": smpl_trans.detach().cpu().numpy(),
        "betas": betas[0].detach().cpu().numpy(),
        "gender": "neutral",
        "mocap_frame_rate": 30,
    }
    np.savez_compressed(fname, **amass_data)


def process_dataset(dl, dataset, target_folder, split_name: str):
    """Process a motion dataset batch and convert sequences to SMPL-X format.

    Args:
        dl (DataLoader): PyTorch DataLoader providing batched data
        dataset (Dataset): Source dataset object (for denormalization)
        target_folder (str): target folder for data saving
        split_name (str): Name of data split being processed

    Output files:
        Saved as: {target_folder}/{split_name}_{object_name}_{index}.npz
        Where:
            target_folder: (implied from external context)
            object_name: Extracted from sequence name
            index: Incremental sequence counter

    """
    index = 0
    for data_dict in dl:
        val_data = data_dict["motion"].cuda()
        for_vis_gt_data = val_data[:]
        all_res_list = for_vis_gt_data

        num_seq = all_res_list.shape[0]
        print(f"Processing {split_name}, num_seq: {num_seq}")
        num_joints = 24

        normalized_global_jpos = all_res_list[:, :, : num_joints * 3].reshape(
            num_seq, -1, num_joints, 3
        )
        global_jpos = dataset.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)
        global_root_jpos = global_jpos[:, :, 0, :].clone()
        global_rot_6d = all_res_list[:, :, -22 * 6 :].reshape(
            num_seq, -1, 22, 6
        )
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d)

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)
        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx]
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )

            curr_global_root_jpos = global_root_jpos[idx]
            curr_trans2joint = trans2joint[idx : idx + 1].clone()
            root_trans = curr_global_root_jpos + curr_trans2joint

            betas = data_dict["betas"][idx]
            gender = data_dict["gender"][idx]
            curr_seq_name = data_dict["seq_name"][idx]
            object_name = curr_seq_name.split("_")[1]

            fname = os.path.join(
                target_folder, f"{split_name}_{object_name}_{index}.npz"
            )
            print(fname)

            run_smplx_model(
                root_trans[None].cuda(),
                curr_local_rot_aa_rep[None].cuda(),
                betas.cuda(),
                [gender],
                fname,
            )
            index += 1


def omomo_to_amass(data_root_folder, target_folder):
    """Convert Omomo dataset to AMASS-compatible SMPL-X format.

    Args:
        data_root_folder (str): Path to the root directory of Omomo dataset
        target_folder (str): Output directory for processed AMASS files

    """
    use_object_split = True
    window_size = 120

    train_dataset = MyHandFootManipDataset(
        train=True,
        data_root_folder=data_root_folder,
        window=window_size,
        use_object_splits=use_object_split,
    )
    val_dataset = MyHandFootManipDataset(
        train=False,
        data_root_folder=data_root_folder,
        window=window_size,
        use_object_splits=use_object_split,
    )

    val_ds = val_dataset
    train_ds = train_dataset
    val_dl = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0
    )
    train_dl = data.DataLoader(
        train_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0
    )

    process_dataset(train_dl, train_dataset, target_folder, "train")
    process_dataset(val_dl, val_dataset, target_folder, "val")


if __name__ == "__main__":
    data_root_folder = ""
    target_folder = ""
    omomo_to_amass(data_root_folder, target_folder)
