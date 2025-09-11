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

import glob
import os
import sys

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU visibility

sys.path.append(os.getcwd())

import random
import time
import traceback

import hydra
import joblib
import numpy as np
import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
)
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
)
from smpl_sim.utils import torch_utils
from smpl_sim.utils.smoothing_utils import (
    gaussian_filter_1d_batch,
)
from torch.autograd import Variable
from tqdm import tqdm
from utils.torch_humanoid_batch import HumanoidBatch

torch.set_num_threads(1)


def load_amass_data(data_path):
    """Load and preprocess AMASS motion data from .npz files.

    Args:
        data_path (str): Path to the .npz file containing AMASS motion data

    Returns:
        dict or None: Dictionary containing processed motion data with keys:
            - pose_aa: Pose parameters in axis-angle format (N, 72)
            - gender: Character gender ('male', 'female', or 'neutral')
            - trans: Root translations (N, 3)
            - betas: Shape parameters (10,) or None
            - fps: Frame rate of the motion
        Returns None if data loading fails or framerate is missing
    """
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if (
        "mocap_framerate" not in entry_data
        and "mocap_frame_rate" not in entry_data
    ):
        return None

    framerate = (
        entry_data["mocap_framerate"]
        if "mocap_framerate" in entry_data
        else entry_data["mocap_frame_rate"]
    )

    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))],
        axis=-1,
    )
    betas = entry_data.get("betas", None)
    gender = entry_data.get("gender", "neutral")
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def process_single_motion_phc(data_key, key_name_to_pkls, cfg):
    """Process motion file with PHC (Physics-based Humanoid Control) fitting.

    Args:
        data_key (str): Unique identifier for the motion data
        key_name_to_pkls (dict): Mapping from data keys to file paths
        cfg (DictConfig): Configuration object containing robot parameters,
                         optimization settings, and file paths

    Returns:
        tuple: (data_dump, message) where:
            - data_dump (dict or None): Processed motion data containing:
                - root_trans_offset: Root translations with offset (N, 3)
                - pose_aa: Pose parameters in axis-angle format (N, J*3)
                - dof: Joint degrees of freedom (N, num_dof)
                - root_rot: Root rotation in quaternion format (N, 4)
                - smpl_joints: Original SMPL joint positions (N, J, 3)
                - fps: Frame rate (30)
            - message (str): Success/error message describing the processing
                result

    Reference:
        https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/
        convert_amass_data.py
    """
    try:
        # Skip if file doesn't exist
        amass_data_path = key_name_to_pkls[data_key]
        if not os.path.exists(amass_data_path):
            return None, f"File not found: {amass_data_path}"

        amass_data = load_amass_data(amass_data_path)
        if amass_data is None:
            return None, f"Failed to load AMASS data: {amass_data_path}"

        skip = int(amass_data["fps"] // 30)
        if skip == 0:
            skip = 1
        trans = torch.from_numpy(amass_data["trans"][::skip])
        num = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data["pose_aa"][::skip]).float()

        if num < 10:
            return None, "Motion too short (< 10 frames)"

        device = torch.device("cpu")
        humanoid_fk = HumanoidBatch(cfg.robot)  # load forward kinematics model
        num_augment_joint = len(cfg.robot.extend_config)

        #### Define corresonpdances between h1 and smpl joints
        robot_joint_names_augment = humanoid_fk.body_names_augment
        robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
        smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
        robot_joint_pick_idx = [
            robot_joint_names_augment.index(j) for j in robot_joint_pick
        ]
        smpl_joint_pick_idx = [
            SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick
        ]

        smpl_parser_n = SMPL_Parser(model_path="assets/smpl", gender="neutral")
        shape_new, scale = joblib.load(cfg.robot.fitted_shape_dump_path)

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new, trans
            )
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_rot_quat = torch.from_numpy(
            (
                sRot.from_rotvec(pose_aa_walk[:, :3])
                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            ).as_quat()
        ).float()  # can't directly use this
        gt_root_rot = torch.from_numpy(
            sRot.from_quat(
                torch_utils.calc_heading_quat(gt_root_rot_quat)
            ).as_rotvec()
        ).float()  # so only use the heading.

        # def dof_to_pose_aa(dof_pos):
        dof_pos = torch.zeros((1, num, humanoid_fk.num_dof, 1))

        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        # optimizer_pose = torch.optim.Adam([dof_pos_new],lr=0.01)
        # optimizer_root =
        # torch.optim.Adam([root_rot_new, root_pos_offset],lr=0.01)
        optimizer = torch.optim.Adam(
            [dof_pos_new, root_rot_new, root_pos_offset], lr=0.02
        )

        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        # B, T, J, D = dof_pos_new.shape

        patience_threshold = 50
        best_loss = float("inf")
        start_time = time.time()

        for _iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_h1_new = torch.cat(
                [
                    root_rot_new[None, :, None],
                    humanoid_fk.dof_axis * dof_pos_new,
                    torch.zeros((1, num, num_augment_joint, 3)).to(device),
                ],
                axis=2,
            )
            fk_return = humanoid_fk.fk_batch(
                pose_aa_h1_new, root_trans_offset[None,] + root_pos_offset
            )

            diff = (
                fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )

            loss_g = diff.norm(dim=-1).mean() + 0.01 * torch.mean(
                torch.square(dof_pos_new)
            )
            loss = loss_g
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_threshold:
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dof_pos_new.data.clamp_(
                humanoid_fk.joints_range[:, 0, None],
                humanoid_fk.joints_range[:, 1, None],
            )

            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None,],
                kernel_size,
                sigma,
            ).transpose(2, 1)[..., None]

        dof_pos_new.data.clamp_(
            humanoid_fk.joints_range[:, 0, None],
            humanoid_fk.joints_range[:, 1, None],
        )
        pose_aa_h1_new = torch.cat(
            [
                root_rot_new[None, :, None],
                humanoid_fk.dof_axis * dof_pos_new,
                torch.zeros((1, num, num_augment_joint, 3)).to(device),
            ],
            axis=2,
        )

        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()
        combined_mesh = humanoid_fk.mesh_fk(
            pose_aa_h1_new[:, :1].detach(),
            root_trans_offset_dump[None, :1].detach(),
        )
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()

        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze()
            .detach()
            .numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().numpy(),
            "root_rot": sRot.from_rotvec(
                root_rot_new.detach().numpy()
            ).as_quat(),
            "smpl_joints": joints_dump,
            "fps": 30,
        }

        processing_time = time.time() - start_time
        return data_dump, f"Successfully processed in {processing_time:.1f}s"

    except Exception as e:
        error_msg = (
            f"Error processing {data_key}: {str(e)}\n{traceback.format_exc()}"
        )
        return None, error_msg


@ray.remote
def process_single_motion_ray(
    data_key: str, amass_data_path: str, cfg_dict: dict, dump_dir: str
):
    """Ray remote function to process a single motion file with PHC fitting.

    Args:
        data_key (str): Unique identifier for the motion data
        amass_data_path (str): Path to the AMASS .npz file
        cfg_dict (dict): Configuration dictionary (converted from DictConfig)
        dump_dir (str): Directory to save processed motion files

    Returns:
        tuple: (data_key, message) where:
            - data_key (str or None): The processed data key if successful,
                None if failed
            - message (str): Success/error message describing the processing
                result
    """
    import os
    import time
    import traceback

    from omegaconf import DictConfig

    # Convert dict back to DictConfig
    cfg = DictConfig(cfg_dict)

    try:
        # Skip if file doesn't exist
        if not os.path.exists(amass_data_path):
            return None, f"File not found: {amass_data_path}"

        # Create a temporary key_name_to_pkls dict for compatibility
        key_name_to_pkls = {data_key: amass_data_path}

        # Process the motion file
        start_time = time.time()
        data_dump, message = process_single_motion_phc(
            data_key, key_name_to_pkls, cfg
        )

        if data_dump is not None:
            processing_time = time.time() - start_time

            # Save individual motion file immediately
            motion_file = f"{dump_dir}/{data_key}.pkl"
            joblib.dump({data_key: data_dump}, motion_file)

            return (
                data_key,
                f"Successfully processed and saved to {motion_file} in "
                f"{processing_time:.1f}s",
            )
        else:
            return None, message

    except Exception as e:
        error_msg = (
            f"Error processing {data_key}: {str(e)}\n{traceback.format_exc()}"
        )
        return None, error_msg


def process_motion_with_ray(key_names, key_name_to_pkls, cfg):
    """Process motion files using Ray for parallel distributed computing.

    Args:
        key_names (list): List of motion data keys to process
        key_name_to_pkls (dict): Mapping from data keys to file paths
        cfg (DictConfig): Configuration object containing processing
            parameters, Ray settings, and output directory

    Returns:
        int: Number of successfully processed motion files
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            num_cpus=cfg.get("num_jobs", 16),
            object_store_memory=cfg.get(
                "ray_object_store_memory", 2000000000
            ),  # 2GB
            ignore_reinit_error=True,
            log_to_driver=False,  # Reduce logging noise
        )
        logger.info(f"Ray initialized with {ray.cluster_resources()}")

    # Convert DictConfig to regular dict for Ray serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create remote tasks
    logger.info(f"Creating {len(key_names)} Ray tasks...")
    futures = []

    # Create dump directory
    os.makedirs(cfg.dump_dir, exist_ok=True)

    for data_key in key_names:
        amass_data_path = key_name_to_pkls[data_key]
        future = process_single_motion_ray.remote(
            data_key, amass_data_path, cfg_dict, cfg.dump_dir
        )
        futures.append((data_key, future))

    # Process results with progress tracking
    failed_count = 0
    success_count = 0

    logger.info("Processing motion files with Ray...")
    pbar = tqdm(total=len(futures), desc="Ray Processing")

    # Use Ray's built-in progress tracking
    remaining_futures = {future: data_key for data_key, future in futures}

    while remaining_futures:
        # Get completed futures
        ready_futures, remaining_future_list = ray.wait(
            list(remaining_futures.keys()),
            num_returns=min(
                10, len(remaining_futures)
            ),  # Process up to 10 at a time
            timeout=1.0,  # 1 second timeout
        )

        # Process completed futures
        for future in ready_futures:
            data_key = remaining_futures.pop(future)
            try:
                result, message = ray.get(future)
                if result is not None:
                    success_count += 1
                    # result is now just the data_key
                    logger.debug(f"✓ {data_key}: {message}")
                else:
                    failed_count += 1
                    if (
                        "too short" not in message
                        and "File not found" not in message
                    ):
                        logger.warning(f"✗ {data_key}: {message}")

                pbar.update(1)
                pbar.set_description(
                    f"Ray Processing (✓{success_count} ✗{failed_count})"
                )

            except Exception as e:
                failed_count += 1
                logger.error(f"✗ {data_key}: Ray task failed: {str(e)}")
                pbar.update(1)
                pbar.set_description(
                    f"Ray Processing (✓{success_count} ✗{failed_count})"
                )

        # Update remaining futures list
        remaining_futures = {
            f: remaining_futures[f] for f in remaining_future_list
        }

    pbar.close()

    logger.info(
        f"Ray processing completed: {success_count} successful, "
        f"{failed_count} failed/skipped"
    )
    logger.info(f"Individual motion files saved to: {cfg.dump_dir}")

    # Optionally shutdown Ray
    # (comment out if you want to reuse the Ray cluster)
    if cfg.get("ray_shutdown_after_processing", False):
        ray.shutdown()
        logger.info("Ray cluster shutdown")

    return success_count  # Return count instead of data


@hydra.main(
    version_base=None,
    config_path="../../config/motion_retargeting",
    config_name="phc_config",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for PHC motion fitting pipeline.

    This function orchestrates the entire motion processing pipeline:
    1. Discovers AMASS motion files from the specified root directory
    2. Creates a mapping between motion keys and file paths
    3. Initializes parallel processing with Ray
    4. Processes all motion files and saves results

    Args:
        cfg (DictConfig): Hydra configuration object containing:
            - amass_root: Root directory containing AMASS .npz files
            - dump_dir: Output directory for processed motion files
            - num_jobs: Number of parallel workers
            - Other processing parameters

    Returns:
        None: Results are saved to the specified dump directory
    """
    if "amass_root" in cfg:
        amass_root = os.path.normpath(cfg.amass_root)
    else:
        raise ValueError("amass_root is not specified in the config")
    all_pkls = glob.glob(
        os.path.join(amass_root, "**", "*.npz"), recursive=True
    )
    key_name_to_pkls = {
        "0-"
        + os.path.relpath(data_path, amass_root)
        .replace(os.sep, "_")
        .replace(".npz", ""): data_path
        for data_path in all_pkls
    }
    key_names = list(key_name_to_pkls.keys())

    if not key_names:
        logger.warning("No motions found to process")
        return

    # Set up processing
    random.shuffle(key_names)  # shuffle for easy load balance
    num_jobs = cfg.get("num_jobs", 16)
    logger.info(f"Processing {len(key_names)} motions with {num_jobs} workers")
    logger.info(f"Running with Ray ({num_jobs} workers)")
    success_count = process_motion_with_ray(key_names, key_name_to_pkls, cfg)

    # Report results
    if success_count > 0:
        logger.info(
            f"Processing completed: {success_count} motions successfully\
             processed "
            f"and saved to {cfg.dump_dir}"
        )
    else:
        logger.info("No motions were successfully processed")


if __name__ == "__main__":
    main()
