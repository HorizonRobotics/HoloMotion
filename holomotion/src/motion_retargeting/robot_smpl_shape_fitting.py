# Project RoboOrchard
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
# This file was originally copied from the [PHC] repository:
# https://github.com/ZhengyiLuo/PHC
# Modifications have been made to fit the needs of this project.
#

import os
import os.path as osp
import sys

sys.path.append(os.getcwd())


import hydra
import joblib
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from torch.autograd import Variable
from tqdm import tqdm
from utils.torch_humanoid_batch import HumanoidBatch


@hydra.main(
    version_base=None,
    config_path="../../config/motion_retargeting",
    config_name="shape_fitting_config",
)
def main(cfg: DictConfig) -> None:
    """Main function for robot SMPL shape fitting optimization.

    This function performs shape parameter optimization to match a robot's
    joint configuration with SMPL humanoid model. The process involves:
    1. Loading robot model and defining joint correspondences
    2. Initializing shape parameters and scale for optimization
    3. Performing gradient-based optimization to minimize joint position
    differences
    4. Optionally visualizing the fitting results
    5. Saving the optimized shape parameters for later use

    The optimization aims to find optimal SMPL shape parameters (betas) and
    scale factor that minimize the distance between corresponding robot and
    SMPL joint positions in a neutral standing pose.

    Args:
        cfg: Configuration dictionary containing:
            - robot: Robot configuration with joint matches and pose modifiers
            - robot.fitted_shape_dump_path:
                Path to save fitted shape parameters

    Returns:
        None: Optimized shape parameters are saved to the specified dump path

    Reference:
        https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/
        fit_smpl_shape.py

    """
    humanoid_fk = HumanoidBatch(cfg.robot)  # load forward kinematics model

    #### Define corresonpdances between robot and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [
        robot_joint_names_augment.index(j) for j in robot_joint_pick
    ]
    smpl_joint_pick_idx = [
        SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick
    ]

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(
        np.repeat(
            sRot.identity().as_rotvec()[
                None,
                None,
                None,
            ],
            humanoid_fk.num_bodies,
            axis=2,
        ),
        1,
        axis=1,
    )
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    ###### prepare SMPL default pause for robot
    pose_aa_stand = np.zeros((1, 72))
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)

    for modifiers in cfg.robot.smpl_pose_modifier:
        modifier_key = list(modifiers.keys())[0]
        modifier_value = list(modifiers.values())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(modifier_key)] = (
            sRot.from_euler(
                "xyz", eval(modifier_value), degrees=False
            ).as_rotvec()
        )

    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
    smpl_parser_n = SMPL_Parser(model_path="assets/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(
        pose_aa_robot[None,], root_trans_offset[None, 0:1]
    )

    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

    train_iterations = 1000
    print("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_stand, shape_new, trans[0:1]
        )  # fitted smpl shape
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        if len(cfg.robot.extend_config) > 0:
            diff = (
                fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )
        else:
            diff = (
                fk_return.global_translation[:, :, robot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )

        # loss_g = diff.norm(dim = -1).mean()
        loss_g = diff.norm(dim=-1).square().sum()

        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # print the fitted shape and scale parameters
    print("shape:", shape_new.detach())
    print("scale:", scale)

    if cfg.get("vis", False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

        j3d = (
            fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :]
            .detach()
            .numpy()
        )
        j3d = j3d - j3d[:, 0:1]
        j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_joints = j3d_joints - j3d_joints[:, 0:1]
        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(90, 0)
        ax.scatter(
            j3d[idx, :, 0],
            j3d[idx, :, 1],
            j3d[idx, :, 2],
            label="Humanoid Shape",
            c="blue",
        )
        ax.scatter(
            j3d_joints[idx, :, 0],
            j3d_joints[idx, :, 1],
            j3d_joints[idx, :, 2],
            label="Fitted Shape",
            c="red",
        )

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()

    os.makedirs(osp.dirname(cfg.robot.fitted_shape_dump_path), exist_ok=True)
    joblib.dump((shape_new.detach(), scale), cfg.robot.fitted_shape_dump_path)
    logger.info(f"Fitted shape dumped to {cfg.robot.fitted_shape_dump_path}")


if __name__ == "__main__":
    main()
