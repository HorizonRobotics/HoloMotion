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
# This file was originally copied from the [ProtoMotions] repository:
# https://github.com/NVLabs/ProtoMotions/
# Modifications have been made to fit the needs of this project.
#

import glob
import os
import random
import time
import traceback
import uuid
from typing import Sequence

import hydra
import joblib
import mink
import mujoco
import mujoco.viewer
import numpy as np
import ray
import torch
from dm_control import mjcf
from loguru import logger
from loop_rate_limiters import RateLimiter
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.poselib.skeleton.skeleton3d import (
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
)
from smpl_sim.smpllib.smpl_joint_names import (
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from tqdm import tqdm
from utils.torch_humanoid_batch import HumanoidBatch

# Global caches to avoid recreating heavy objects
_MODEL_CACHE = {}
_SMPL_PARSER_CACHE = {}
_SKELETON_TREE_CACHE = {}
_HUMANOID_BATCH_CACHE = {}
_SMPL_ROBOT_CACHE = {}


def get_or_create_model(
    robot_mjcf_path: str,
    keypoint_names: Sequence[str],
    hand_names: Sequence[str] = None,
):
    """Get or create a cached MuJoCo model to avoid memory leaks.

    Args:
        robot_mjcf_path: Path to the robot MJCF model file
        keypoint_names: Sequence of keypoint names for the model
        hand_names: Optional sequence of hand joint names for special handling

    Returns:
        Cached MuJoCo model instance

    """
    # Create a cache key based on the model parameters
    cache_key = (
        robot_mjcf_path,
        tuple(keypoint_names),
        tuple(hand_names or []),
    )

    if cache_key not in _MODEL_CACHE:
        # Only create the model if it's not in cache
        _MODEL_CACHE[cache_key] = construct_model(
            robot_mjcf_path, keypoint_names, hand_names
        )

    return _MODEL_CACHE[cache_key]


def get_or_create_smpl_parser(model_path: str, gender: str = "neutral"):
    """Get or create a cached SMPL parser.

    Args:
        model_path: Path to the SMPL model files
        gender: Gender of the SMPL model ("neutral", "male", "female")

    Returns:
        Cached SMPL_Parser instance.

    """
    cache_key = (model_path, gender)

    if cache_key not in _SMPL_PARSER_CACHE:
        _SMPL_PARSER_CACHE[cache_key] = SMPL_Parser(
            model_path=model_path,
            gender=gender,
        )

    return _SMPL_PARSER_CACHE[cache_key]


def get_or_create_humanoid_batch(cfg):
    """Get or create a cached Humanoid_Batch instance.

    Args:
        cfg: Configuration dictionary containing asset settings

    Returns:
        Cached Humanoid_Batch instance

    """
    # Create a cache key from the config
    cache_key = (cfg.asset.assetRoot, cfg.asset.assetFileName)

    if cache_key not in _HUMANOID_BATCH_CACHE:
        _HUMANOID_BATCH_CACHE[cache_key] = HumanoidBatch(cfg)

    return _HUMANOID_BATCH_CACHE[cache_key]


def construct_model(
    humanoid_mjcf_path: str,
    keypoint_names: Sequence[str],
    hand_names: Sequence[str] = None,
):
    """Construct a complete MuJoCo model with humanoid and keypoint.

    Args:
        humanoid_mjcf_path: Path to the humanoid MJCF model file
        keypoint_names: Sequence of keypoint names for visualization
        hand_names: Optional sequence of hand joint names for special sizing

    Returns:
        Complete MuJoCo model with all components integrated

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    root = mjcf.RootElement()

    getattr(root.visual, "global").offwidth = "1920"
    getattr(root.visual, "global").offheight = "1080"

    root.asset.add(
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="800",
        height="800",
    )
    root.asset.add(
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0 0 0",
        rgb2="0 0 0",
        width="300",
        height="300",
        mark="edge",
        markrgb=".2 .3 .4",
    )
    root.asset.add(
        "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )
    root.worldbody.add(
        "geom", name="ground", type="plane", size="0 0 .01", material="grid"
    )

    for keypoint_name in keypoint_names:
        if hand_names and any(
            hand_name in keypoint_name for hand_name in hand_names
        ):
            size = 0.01
        else:
            size = 0.02
        body = root.worldbody.add(
            "body", name=f"keypoint_{keypoint_name}", mocap="true"
        )
        rgb = np.random.rand(3)
        body.add(
            "site",
            name=f"site_{keypoint_name}",
            type="sphere",
            size=f"{size}",
            rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
        )
        if keypoint_name == "Pelvis":
            body.add("light", pos="0 0 2", directional="false")
            root.worldbody.add(
                "camera",
                name="tracking01",
                pos=[2.972, -0.134, 1.303],
                xyaxes="0.294 0.956 0.000 -0.201 0.062 0.978",
                mode="trackcom",
            )
            root.worldbody.add(
                "camera",
                name="tracking02",
                pos="4.137 2.642 1.553",
                xyaxes="-0.511 0.859 0.000 -0.123 -0.073 0.990",
                mode="trackcom",
            )

    humanoid_mjcf = mjcf.from_path(humanoid_mjcf_path)

    humanoid_mjcf.worldbody.add(
        "camera",
        name="front_track",
        pos="-0.120 3.232 1.064",
        xyaxes="-1.000 -0.002 -0.000 0.000 -0.103 0.995",
        mode="trackcom",
    )
    root.include_copy(humanoid_mjcf)

    root_str = to_string(root, pretty=True)
    assets = get_assets(root)
    return mujoco.MjModel.from_xml_string(root_str, assets)


def to_string(
    root: mjcf.RootElement,
    precision: float = 17,
    zero_threshold: float = 0.0,
    pretty: bool = False,
) -> str:
    """Convert MJCF root element to XML string with optional formatting.

    Args:
        root: MJCF root element to convert
        precision: Number of decimal places for floating point values
        zero_threshold: Threshold below which values are considered zero
        pretty: Whether to format the XML with proper indentation

    Returns:
        Formatted XML string representation of the MJCF model

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    from lxml import etree

    xml_string = root.to_xml_string(
        precision=precision, zero_threshold=zero_threshold
    )
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    # Remove hashes from asset filenames.
    tags = ["mesh", "texture"]
    for tag in tags:
        assets = [
            asset
            for asset in root.find("asset").iter()
            if asset.tag == tag and "file" in asset.attrib
        ]
        for asset in assets:
            name, extension = asset.get("file").split(".")
            asset.set("file", ".".join((name[:-41], extension)))

    if not pretty:
        return etree.tostring(root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in root.iter():
        for key in elem.keys():
            if key == "name" and "unnamed" in elem.get(key):
                elem.attrib.pop(key)

    # Get string from lxml.
    xml_string = etree.tostring(root, pretty_print=True)

    # Remove redundant attributes.
    xml_string = xml_string.replace(b' gravcomp="0"', b"")

    # Insert spaces between top-level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b"  <"):
            if line.startswith(b"  </") or line.endswith(b"/>"):
                newlines.append(b"")
    newlines.append(b"")
    xml_string = b"\n".join(newlines)

    return xml_string.decode()


def get_assets(root: mjcf.RootElement):
    """Process assets from MJCF root element.

    Args:
        root: MJCF root element containing assets

    Returns:
        Dictionary mapping processed asset names to their payload data

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    assets = {}
    for file, payload in root.get_assets().items():
        name, extension = file.split(".")
        assets[".".join((name[:-41], extension))] = payload
    return assets


def create_robot_motion(
    poses: np.ndarray,
    trans: np.ndarray,
    orig_global_trans: np.ndarray,
    mocap_fr: float,
    cfg: DictConfig,
) -> SkeletonMotion:
    """Create a SkeletonMotion for robot from poses and translations.

    Args:
        poses: Joint angles from mujoco [N, num_dof] in proper ordering -
            groups of 3 hinge joints per joint
        trans: Root transform [N, 7] (pos + quat)
        orig_global_trans: Original global translations [N, num_joints, 3]
        mocap_fr: Motion capture framerate

    Returns:
        SkeletonMotion: Motion data in proper format for robot

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    # cfg.asset_file = cfg.robot.asset.assetFileName
    humanoid_batch = get_or_create_humanoid_batch(cfg)
    b, seq_len = 1, poses.shape[0]
    poses_tensor = torch.from_numpy(poses).float().reshape(b, seq_len, -1, 1)
    root_rot = sRot.from_quat(np.roll(trans[:, 3:7], -1)).as_rotvec()
    root_rot_tensor = (
        torch.from_numpy(root_rot).float().reshape(b, seq_len, 1, 3)
    )
    poses_tensor = torch.cat(
        [
            root_rot_tensor,
            humanoid_batch.dof_axis * poses_tensor,
            torch.zeros((1, seq_len, len(cfg.extend_config), 3)),
        ],
        axis=2,
    )
    trans_tensor = (
        torch.from_numpy(trans[:, :3]).float().reshape(b, seq_len, 3)
    )
    motion_data = humanoid_batch.fk_batch(
        poses_tensor, trans_tensor, return_full=True, dt=1.0 / mocap_fr
    )
    fk_return_proper = humanoid_batch.convert_to_proper_kinematic(motion_data)
    orig_lowest_heights = torch.from_numpy(
        orig_global_trans[..., 2].min(axis=1)
    )
    retarget_lowest_heights = (
        fk_return_proper.global_translation[..., 2].min(dim=-1).values
    )
    height_offset = (retarget_lowest_heights - orig_lowest_heights).unsqueeze(
        -1
    )
    fk_return_proper.global_translation[..., 2] -= height_offset

    curr_motion = {
        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
        for k, v in fk_return_proper.items()
    }

    if cfg.get("phc_compatible", True):
        phc_motion = {
            "pose_aa": poses_tensor.squeeze().detach().cpu().numpy(),
            "dof": curr_motion["dof_pos"].squeeze().detach().cpu().numpy(),
            "root_rot": curr_motion["global_rotation"]
            .squeeze()[:, 0, :]
            .detach()
            .cpu()
            .numpy(),
            "root_trans_offset": curr_motion["global_translation"]
            .squeeze()[:, 0, :]
            .detach()
            .cpu()
            .numpy(),
            "fps": curr_motion["fps"],
        }
        return phc_motion

    return curr_motion


def retarget_motion(
    motion: SkeletonMotion,
    robot_type: str,
    robot_mjcf_path: str,
    cfg: DictConfig,
    orig_global_trans: np.ndarray = None,
):
    """Retarget motion from SMPL format to robot format using Mink.

    Args:
        motion: Source motion in SkeletonMotion format
        robot_type: Type of robot for retargeting
        robot_mjcf_path: Path to robot MJCF model file
        cfg: Configuration dictionary containing retargeting parameters
        orig_global_trans: Original global translations for height adjustment

    Returns:
        Retargeted motion data in robot-compatible format

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    global_translations = motion.global_translation.numpy()
    pose_quat_global = motion.global_rotation.numpy()
    # pose_quat = motion.local_rotation.numpy()
    timeseries_length = global_translations.shape[0]
    fps = motion.fps

    smplx_mujoco_joint_names = SMPLH_MUJOCO_NAMES
    hand_names = cfg.robot.get("hand_names", [])

    # Use cached model to avoid memory leaks from repeated model creation
    model = get_or_create_model(
        robot_mjcf_path, smplx_mujoco_joint_names, hand_names
    )

    # Create fresh configuration and data for this motion
    configuration = mink.Configuration(model)
    data = configuration.data

    tasks = []
    frame_tasks = {}
    posture_task = None
    vel = None
    rate = None

    try:
        for (
            _joint_name,
            retarget_info,
        ) in cfg.robot.mink_keypoint_to_joint.items():
            # orientation_base_cost = 1e-4
            orientation_base_cost = 0.0
            if "wrist" in retarget_info["name"]:
                orientation_base_cost = 1e-3
            # if "ankle" in retarget_info["name"]:
            #     orientation_base_cost = 1.0
            task = mink.FrameTask(
                frame_name=retarget_info["name"],
                frame_type="body",
                position_cost=1.0 * retarget_info["weight"],
                orientation_cost=orientation_base_cost
                * retarget_info["weight"],
                lm_damping=1.0,
            )
            frame_tasks[retarget_info["name"]] = task
        tasks.extend(frame_tasks.values())

        posture_task = mink.PostureTask(
            model, cost=1e-4
        )  # Lower cost for more natural poses
        tasks.append(posture_task)

        # viewer_context = nullcontext()

        retargeted_poses = []
        retargeted_trans = []

        # with viewer_context as viewer:
        data.qpos[:] = 0
        data.qpos[0:3] = global_translations[0, 0]
        data.qpos[3:7] = pose_quat_global[0, 0]

        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_step(model, data)

        optimization_steps_per_frame = 2
        rate = RateLimiter(frequency=fps * optimization_steps_per_frame)
        solver = "quadprog"

        t: int = int(np.ceil(-100.0 * fps / 30))
        vel = None

        while t < timeseries_length:
            # Set targets for current frame
            for i, (joint_name, _) in enumerate(
                cfg.robot.mink_keypoint_to_joint.items()
            ):
                body_idx = smplx_mujoco_joint_names.index(joint_name)
                target_pos = global_translations[max(0, t), body_idx, :].copy()
                target_rot = pose_quat_global[max(0, t), body_idx].copy()
                rot_matrix = sRot.from_quat(target_rot).as_matrix()
                rot = mink.SO3.from_matrix(rot_matrix)
                tasks[i].set_target(
                    mink.SE3.from_rotation_and_translation(rot, target_pos)
                )

            # Update keypoint positions.
            keypoint_pos = {}
            for keypoint_name, keypoint in zip(
                smplx_mujoco_joint_names,
                global_translations[max(0, t)],
            ):
                mid = model.body(f"keypoint_{keypoint_name}").mocapid[0]
                data.mocap_pos[mid] = keypoint
                keypoint_pos[keypoint_name] = keypoint

            # Perform multiple optimization steps
            for _ in range(optimization_steps_per_frame):
                limits = [
                    mink.ConfigurationLimit(model),
                    mink.VelocityLimit(model),
                ]

                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    1e-1,
                    limits=limits,
                )
                configuration.integrate_inplace(vel, rate.dt)

            # Store poses and translations if we're past initialization
            if t >= 0:
                retargeted_poses.append(data.qpos[7:].copy())
                retargeted_trans.append(data.qpos[:7].copy())

            t += 1

        retargeted_poses = np.stack(retargeted_poses)
        retargeted_trans = np.stack(retargeted_trans)

        # Apply temporal smoothing to improve naturalness
        if len(retargeted_poses) > 1:
            # Smooth positions
            kernel_size = 5
            sigma = 0.75
            retargeted_poses_tensor = torch.from_numpy(
                retargeted_poses
            ).float()
            smoothed_poses = (
                gaussian_filter_1d_batch(
                    retargeted_poses_tensor.transpose(1, 0)[None,],
                    kernel_size,
                    sigma,
                )
                .transpose(2, 1)
                .squeeze(0)
                .numpy()
            )
            retargeted_poses = smoothed_poses

            # Smooth orientations
            root_orientations = retargeted_trans[:, 3:7]
            smoothed_orientations = (
                gaussian_filter_1d_batch(
                    torch.from_numpy(root_orientations)
                    .float()
                    .transpose(1, 0)[None,],
                    kernel_size,
                    sigma,
                )
                .transpose(2, 1)
                .squeeze(0)
                .numpy()
            )

            # Normalize quaternions after smoothing
            for i in range(smoothed_orientations.shape[0]):
                smoothed_orientations[i] = sRot.from_quat(
                    smoothed_orientations[i]
                ).as_quat()

            retargeted_trans[:, 3:7] = smoothed_orientations

        orig_trans_for_height = (
            orig_global_trans
            if orig_global_trans is not None
            else global_translations
        )

        result = create_robot_motion(
            retargeted_poses,
            retargeted_trans,
            orig_trans_for_height,
            fps,
            cfg.robot,
        )

        return result

    finally:
        # Lightweight cleanup - only clean up per-motion objects,
        # not the cached model
        try:
            # Clear task references
            if tasks:
                for task in tasks:
                    try:
                        del task
                    except Exception:
                        pass
                del tasks
            if frame_tasks:
                for task in frame_tasks.values():
                    try:
                        del task
                    except Exception:
                        pass
                del frame_tasks
            if posture_task is not None:
                del posture_task

            # Only delete the data, not the model (model is cached)
            if data is not None:
                try:
                    mujoco.mj_deleteData(data)
                except Exception:
                    pass

            # Clear configuration (this doesn't delete the underlying model)
            if configuration is not None:
                del configuration

            if vel is not None:
                del vel
            if rate is not None:
                del rate

            # Light garbage collection
            import gc

            gc.collect()

        except Exception as cleanup_error:
            # Don't let cleanup errors break the function
            logger.warning(f"Cleanup warning: {cleanup_error}")
            pass


def manually_retarget_motion_phc_style(
    amass_data_path: str,
    output_path: str,
    robot_type: str,
    cfg: DictConfig = None,
    process_id: int = None,
):
    """Retarget motion using PHC-style optimized shape parameters approach.

    Args:
        amass_data_path: Path to AMASS motion data file (.npz)
        output_path: Path to save the retargeted motion data
        robot_type: Type of robot for retargeting
        cfg: Configuration dictionary containing retargeting parameters
        process_id: Process ID for unique file naming (optional)

    Returns:
        None: Saves retargeted motion to output_path

    Reference:
        https://github.com/NVlabs/ProtoMotions/blob/main/data/scripts/
        retargeting/mink_retarget.py

    """
    motion_data = dict(np.load(open(amass_data_path, "rb"), allow_pickle=True))

    if (
        "mocap_framerate" not in motion_data
        and "mocap_frame_rate" not in motion_data
    ):
        logger.error(f"Failed to load AMASS data: {amass_data_path}")
        return None

    framerate = (
        motion_data["mocap_framerate"]
        if "mocap_framerate" in motion_data
        else motion_data["mocap_frame_rate"]
    )

    # Downsample to 30 FPS
    skip = max(1, int(framerate // 30))
    pose_aa_raw = torch.from_numpy(motion_data["poses"][::skip]).float()
    trans = torch.from_numpy(motion_data["trans"][::skip]).float()
    num_t = pose_aa_raw.shape[0]

    # Create pose_aa in the same format as load_amass_data for SMPL processing
    pose_aa = torch.cat([pose_aa_raw[:, :66], torch.zeros(num_t, 6)], dim=-1)

    if num_t < 10:
        return None

    shape_params = joblib.load(cfg.robot.fitted_shape_dump_path)
    shape_new, scale = shape_params

    # Initialize SMPL parser
    smpl_parser = get_or_create_smpl_parser(
        cfg.robot.asset.smpl_dir, "neutral"
    )

    # Track objects for cleanup
    cleanup_objects = [motion_data, pose_aa_raw, trans, pose_aa, smpl_parser]

    try:
        with torch.no_grad():
            verts, joints = smpl_parser.get_joints_verts(
                pose_aa, shape_new, trans
            )
            joints_original = joints.clone()
            root_pos = joints[:, 0:1]  # Keep original root position
            joints_scaled = (
                joints - joints[:, 0:1]
            ) * scale.detach() + root_pos
            trans_scaled = joints_scaled[:, 0]  # Use scaled root positions

        cleanup_objects.extend(
            [
                verts,
                joints,
                joints_original,
                root_pos,
                joints_scaled,
                trans_scaled,
            ]
        )

        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
        smpl_2_mujoco = [
            joint_names.index(q)
            for q in mujoco_joint_names
            if q in joint_names
        ]
        pose_aa_processed = np.concatenate(
            [
                pose_aa_raw.numpy()[:, :66],
                pose_aa_raw.numpy()[
                    :, 75:
                ],  # This gets the hand poses from raw data
            ],
            axis=-1,
        )
        pose_aa_mj = pose_aa_processed.reshape(num_t, 52, 3)[:, smpl_2_mujoco]
        pose_quat = (
            sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
            .as_quat()
            .reshape(num_t, 52, 4)
        )

        cleanup_objects.extend([pose_aa_processed, pose_aa_mj, pose_quat])

        robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": True,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True,
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False,
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smplx",
            "sim": "isaacgym",
        }

        smpl_local_robot = get_or_create_smpl_robot(
            robot_cfg,
            data_dir=cfg.robot.asset.smpl_dir,
        )

        cleanup_objects.append(smpl_local_robot)

        if shape_new.dim() == 1:
            shape_new_2d = shape_new[None, :]  # [10] -> [1, 10]
        else:
            shape_new_2d = shape_new.squeeze()  # Remove extra dimensions
            if shape_new_2d.dim() == 1:
                shape_new_2d = shape_new_2d[None, :]  # [10] -> [1, 10]

        smpl_local_robot.load_from_skeleton(
            betas=shape_new_2d, gender=[0], objs_info=None
        )

        # Create a cache key based on shape parameters instead of temp filename
        shape_cache_key = tuple(shape_new_2d.flatten().tolist())

        if shape_cache_key in _SKELETON_TREE_CACHE:
            # Use cached skeleton tree
            skeleton_tree = _SKELETON_TREE_CACHE[shape_cache_key]
        else:
            # Create unique temporary directory for this process to avoid
            # conflicts
            tmp_smpl_dir = "/tmp/smpl"
            os.makedirs(tmp_smpl_dir, exist_ok=True)

            # Use process ID and timestamp for unique filenames
            if process_id is None:
                process_id = os.getpid()

            uuid_str = f"{process_id}_{int(time.time() * 1000)}_{uuid.uuid4()}"
            xml_path = f"{tmp_smpl_dir}/smpl_humanoid_{uuid_str}.xml"

            try:
                smpl_local_robot.write_xml(xml_path)
                skeleton_tree = SkeletonTree.from_mjcf(xml_path)
                # Cache by shape parameters, not filename
                _SKELETON_TREE_CACHE[shape_cache_key] = skeleton_tree
            except Exception as e:
                logger.error(f"Failed to create SMPL skeleton: {e}")
                raise
            finally:
                # Clean up temporary XML file
                if os.path.exists(xml_path):
                    try:
                        os.remove(xml_path)
                    except Exception:
                        pass  # Ignore cleanup errors

        cleanup_objects.append(skeleton_tree)

        root_trans_offset = trans_scaled + skeleton_tree.local_translation[0]

        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )

        cleanup_objects.extend([root_trans_offset, sk_state])

        pose_quat_global = (
            (
                sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy())
                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            )
            .as_quat()
            .reshape(num_t, -1, 4)
        )

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False,
        )
        sk_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

        cleanup_objects.extend([pose_quat_global, new_sk_state, sk_motion])

        retargeted_motion = retarget_motion(
            sk_motion,
            robot_type,
            robot_mjcf_path=cfg.robot.asset.training_mjcfName,
            cfg=cfg,
            orig_global_trans=joints_original.numpy(),  # Pass original joints
            # for proper height matching
        )

        # cfg.asset_file = cfg.robot.asset.assetFileName
        humanoid_batch = get_or_create_humanoid_batch(cfg.robot)
        cleanup_objects.append(humanoid_batch)

        # Convert retargeted motion back to tensor format for forward
        # kinematics
        pose_aa_tensor = (
            torch.from_numpy(retargeted_motion["pose_aa"]).float().unsqueeze(0)
        )
        root_trans_tensor = (
            torch.from_numpy(retargeted_motion["root_trans_offset"])
            .float()
            .unsqueeze(0)
        )

        cleanup_objects.extend([pose_aa_tensor, root_trans_tensor])

        # Perform forward kinematics to get all joint positions
        motion_data = humanoid_batch.fk_batch(
            pose_aa_tensor, root_trans_tensor, return_full=True, dt=1.0 / 30.0
        )
        fk_return_proper = humanoid_batch.convert_to_proper_kinematic(
            motion_data
        )

        cleanup_objects.extend([motion_data, fk_return_proper])

        # Find the minimum z-value across all joints and all frames
        all_joint_positions = (
            fk_return_proper.global_translation.squeeze()
        )  # [T, J, 3]
        min_z_value = all_joint_positions[..., 2].min().item()

        # If there's ground penetration (min_z < 0), shift everything up
        if min_z_value < 0:
            ground_clearance = 0.001  # Small clearance above ground
            z_adjustment = -min_z_value + ground_clearance
            retargeted_motion["root_trans_offset"][:, 2] += z_adjustment

        return retargeted_motion

    finally:
        # Explicit cleanup of all tracked objects
        for obj in cleanup_objects:
            try:
                # Check if it's a SMPL_Robot instance that might have MuJoCo
                # models
                if (
                    hasattr(obj, "_mjcf_root")
                    or hasattr(obj, "model")
                    or hasattr(obj, "data")
                ):
                    # If it has MuJoCo data/model, try to clean them up
                    # properly
                    if hasattr(obj, "data") and obj.data is not None:
                        try:
                            mujoco.mj_deleteData(obj.data)
                        except Exception:
                            pass
                    if hasattr(obj, "model") and obj.model is not None:
                        try:
                            mujoco.mj_deleteModel(obj.model)
                        except Exception:
                            pass
                # Regular object deletion
                del obj
            except Exception:
                pass

        # Force multiple garbage collection passes
        import gc

        for _ in range(3):
            gc.collect()

        # Additional torch cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def get_or_create_smpl_robot(robot_cfg, data_dir):
    """Get or create a cached SMPL_Robot instance.

    Args:
        robot_cfg: Robot configuration dictionary
        data_dir: Directory containing SMPL model data

    Returns:
        Cached SMPL_Robot instance

    """
    # Create a cache key from the config
    cache_key = (str(robot_cfg), data_dir)

    if cache_key not in _SMPL_ROBOT_CACHE:
        _SMPL_ROBOT_CACHE[cache_key] = SMPL_Robot(robot_cfg, data_dir=data_dir)

    return _SMPL_ROBOT_CACHE[cache_key]


def clear_model_cache():
    """Clear all global caches to free memory."""
    global \
        _MODEL_CACHE, \
        _SMPL_PARSER_CACHE, \
        _SKELETON_TREE_CACHE, \
        _HUMANOID_BATCH_CACHE, \
        _SMPL_ROBOT_CACHE

    # Clear MuJoCo models
    for model in _MODEL_CACHE.values():
        try:
            mujoco.mj_deleteModel(model)
        except Exception:
            pass
    _MODEL_CACHE.clear()

    # Clear SMPL parsers
    _SMPL_PARSER_CACHE.clear()

    # Clear skeleton trees
    _SKELETON_TREE_CACHE.clear()

    # Clear humanoid batches
    _HUMANOID_BATCH_CACHE.clear()

    # Clear SMPL robots
    _SMPL_ROBOT_CACHE.clear()

    # Force garbage collection
    import gc

    gc.collect()


@ray.remote(
    num_cpus=1, memory=1000 * 1024 * 1024
)  # Limit each task to 1GB memory
def process_single_motion_task(
    data_key: str, amass_data_path: str, dump_dir: str, cfg_dict: dict
):
    """Process a single motion file as a Ray task.

    Args:
        data_key: Unique identifier for the motion data
        amass_data_path: Path to AMASS motion data file
        dump_dir: Directory to save processed motion files
        cfg_dict: Configuration dictionary (will be converted to DictConfig)

    Returns:
        Tuple of (data_key, status_message) or (None, error_message)

    """
    import gc

    # Convert dict back to DictConfig
    cfg = DictConfig(cfg_dict)

    try:
        # Skip if file doesn't exist
        if not os.path.exists(amass_data_path):
            return None, f"File not found: {amass_data_path}"

        # Process the motion file through the full retargeting pipeline
        start_time = time.time()

        try:
            # Run the full retargeting pipeline
            retargeted_motion = manually_retarget_motion_phc_style(
                amass_data_path=amass_data_path,
                output_path=None,  # Don't save to file
                robot_type=cfg.robot.humanoid_type,
                cfg=cfg,
                process_id=os.getpid(),
            )

            # Check if retargeting succeeded
            if retargeted_motion is None:
                return (
                    None,
                    f"Failed to retarget motion (missing framerate or other "
                    f"issues): {amass_data_path}",
                )

            # Save individual motion file immediately
            motion_file = f"{dump_dir}/{data_key}.pkl"
            joblib.dump({data_key: retargeted_motion}, motion_file)

            processing_time = time.time() - start_time

            # Explicitly delete the motion data to free memory
            del retargeted_motion

            return (
                data_key,
                f"Successfully processed and saved to {motion_file} in "
                f"{processing_time:.1f}s",
            )

        except Exception as e:
            return None, f"Error processing motion: {str(e)}"

    except Exception as e:
        error_msg = (
            f"Error processing {data_key}: {str(e)}\n{traceback.format_exc()}"
        )
        return None, error_msg
    finally:
        # Clear model cache periodically to prevent memory buildup
        # Only clear every 10th task to balance memory vs performance
        # if hash(data_key) % 10 == 0:
        #     clear_model_cache()

        # Force garbage collection before task ends
        gc.collect()


def process_single_motion_sequential(
    data_key: str, amass_data_path: str, dump_dir: str, cfg: DictConfig
):
    """Process a single motion file sequentially.

    Non-Ray version for debugging.

    Args:
        data_key: Unique identifier for the motion data
        amass_data_path: Path to AMASS motion data file
        dump_dir: Directory to save processed motion files
        cfg: Configuration dictionary

    Returns:
        Tuple of (data_key, status_message) or (None, error_message)

    """
    import gc

    # Skip if file doesn't exist
    if not os.path.exists(amass_data_path):
        return None, f"File not found: {amass_data_path}"

    # Process the motion file through the full retargeting pipeline
    start_time = time.time()

    # Run the full retargeting pipeline
    retargeted_motion = manually_retarget_motion_phc_style(
        amass_data_path=amass_data_path,
        output_path=None,  # Don't save to file
        robot_type=cfg.robot.humanoid_type,
        cfg=cfg,
        process_id=os.getpid(),
    )

    # Check if retargeting succeeded
    if retargeted_motion is None:
        return (
            None,
            f"Failed to retarget motion (missing framerate or other "
            f"issues): {amass_data_path}",
        )

    # Save individual motion file immediately
    motion_file = f"{dump_dir}/{data_key}.pkl"
    joblib.dump({data_key: retargeted_motion}, motion_file)

    processing_time = time.time() - start_time

    # Explicitly delete the motion data to free memory
    del retargeted_motion

    # Clear model cache periodically to prevent memory buildup
    # Only clear every 10th task to balance memory vs performance
    # if hash(data_key) % 10 == 0:
    #     clear_model_cache()

    # Force garbage collection before task ends
    gc.collect()

    return (
        data_key,
        f"Successfully processed and saved to {motion_file} in "
        f"{processing_time:.1f}s",
    )


def process_motion_sequentially(key_names, key_name_to_pkls, cfg):
    """Process motion files using plain Python for loop (for debugging).

    Args:
        key_names: List of motion data keys to process
        key_name_to_pkls: Dictionary mapping keys to AMASS data file paths
        cfg: Configuration dictionary

    Returns:
        Number of successfully processed motions

    """
    # Create job list
    jobs = [(key, key_name_to_pkls[key]) for key in key_names]

    logger.info(f"Processing {len(jobs)} motions sequentially (no Ray)")

    # Set up tracking
    os.makedirs(cfg.dump_dir, exist_ok=True)
    failed_count = 0
    success_count = 0

    logger.info("Starting sequential processing...")
    pbar = tqdm(total=len(jobs), desc="Sequential Processing")

    # Process jobs one by one
    for data_key, amass_data_path in jobs:
        result, message = process_single_motion_sequential(
            data_key, amass_data_path, cfg.dump_dir, cfg
        )

        # Process result
        if result is not None:
            success_count += 1
            logger.debug(f"✓ {message}")
        else:
            failed_count += 1
            if "too short" not in message and "File not found" not in message:
                logger.warning(f"✗ {message}")

        # Update progress
        pbar.update(1)
        pbar.set_description(
            f"Sequential Processing (✓{success_count} ✗{failed_count})"
        )

    pbar.close()

    logger.info(
        f"Sequential processing completed: {success_count} successful, "
        f"{failed_count} failed/skipped"
    )
    logger.info(f"Individual motion files saved to: {cfg.dump_dir}")

    # Clear model cache to free memory
    clear_model_cache()
    logger.info("Model cache cleared")

    return success_count


def process_motion_with_ray(key_names, key_name_to_pkls, cfg):
    """Process motion files using dynamic Ray tasks.

    Args:
        key_names: List of motion data keys to process
        key_name_to_pkls: Dictionary mapping keys to AMASS data file paths
        cfg: Configuration dictionary containing Ray settings

    Returns:
        Number of successfully processed motions

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

    # Create job list
    jobs = [(key, key_name_to_pkls[key]) for key in key_names]
    max_concurrent = cfg.get("num_jobs", 16)

    logger.info(
        f"Processing {len(jobs)} motions with max {max_concurrent} concurrent "
        f"tasks"
    )

    # Set up tracking
    os.makedirs(cfg.dump_dir, exist_ok=True)
    failed_count = 0
    success_count = 0

    logger.info("Starting Ray task processing...")
    pbar = tqdm(total=len(jobs), desc="Ray Task Processing")

    # Process jobs in batches to limit concurrency
    job_idx = 0
    active_futures = {}  # future -> data_key

    # Start initial batch of tasks
    while job_idx < len(jobs) and len(active_futures) < max_concurrent:
        data_key, amass_data_path = jobs[job_idx]
        future = process_single_motion_task.remote(
            data_key, amass_data_path, cfg.dump_dir, cfg_dict
        )
        active_futures[future] = data_key
        job_idx += 1

    # Process results as they complete and spawn new tasks
    while active_futures:
        # Wait for at least one task to complete
        ready_futures, _ = ray.wait(
            list(active_futures.keys()), num_returns=1, timeout=1.0
        )

        for future in ready_futures:
            data_key = active_futures.pop(future)

            try:
                result, message = ray.get(future)

                # Immediately delete the future reference to free Ray object
                # store memory
                del future

                # Process result
                if result is not None:
                    success_count += 1
                    logger.debug(f"✓ {message}")
                else:
                    failed_count += 1
                    if (
                        "too short" not in message
                        and "File not found" not in message
                    ):
                        logger.warning(f"✗ {message}")

                # Update progress
                pbar.update(1)
                pbar.set_description(
                    f"Ray Task Processing (✓{success_count} ✗{failed_count})"
                )

            except Exception as e:
                # Handle task failure
                failed_count += 1
                logger.error(f"✗ Task failed processing {data_key}: {str(e)}")
                pbar.update(1)
                pbar.set_description(
                    f"Ray Task Processing (✓{success_count} ✗{failed_count})"
                )

                # Delete the failed future reference
                del future

            # Start next task if available
            if job_idx < len(jobs):
                next_data_key, next_amass_data_path = jobs[job_idx]
                next_future = process_single_motion_task.remote(
                    next_data_key, next_amass_data_path, cfg.dump_dir, cfg_dict
                )
                active_futures[next_future] = next_data_key
                job_idx += 1

    pbar.close()

    logger.info(
        f"Ray task processing completed: {success_count} successful, "
        f"{failed_count} failed/skipped"
    )
    logger.info(f"Individual motion files saved to: {cfg.dump_dir}")

    # Always shutdown Ray cluster after processing
    ray.shutdown()
    logger.info("Ray cluster shutdown")

    # Clear model cache to free memory
    clear_model_cache()
    logger.info("Model cache cleared")

    return success_count  # Return count instead of data


@hydra.main(
    version_base=None,
    config_path="../../config/motion_retargeting",
    config_name="mink_config",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for motion retargeting using Mink optimization.

    This function orchestrates the entire motion retargeting pipeline:
    1. Discovers AMASS motion files from the specified root directory
    2. Processes motions either sequentially or in parallel using Ray
    3. Saves retargeted motions to the specified output directory

    Args:
        cfg: Configuration dictionary containing all retargeting parameters

    Returns:
        None: Results are saved to files in the configured dump directory

    """
    if hasattr(cfg, "amass_root") and cfg.amass_root:
        amass_root = cfg.amass_root
        all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
        # random shuffle
        random.shuffle(all_pkls)
        # Extract relative path components after amass_root
        key_name_to_pkls = {}
        for data_path in all_pkls:
            # Get relative path from amass_root
            rel_path = os.path.relpath(data_path, amass_root)
            # Remove .npz extension and split into components
            path_components = rel_path.replace(".npz", "").split(os.sep)
            # Join components with underscore for key
            key = "0-" + "_".join(path_components)
            key_name_to_pkls[key] = data_path
        key_names = list(key_name_to_pkls.keys())

        if cfg.get("filter_regex", None):
            import re

            regex_pattern = re.compile(cfg.filter_regex, re.IGNORECASE)
            key_names = [key for key in key_names if regex_pattern.search(key)]
            logger.info(
                f"Filtered to {len(key_names)} motions matching regex "
                f"'{cfg.filter_regex}'"
            )

        if not key_names:
            logger.warning("No motions found to process")
            return

        # Set up processing
        random.shuffle(key_names)  # Shuffle for load balancing
        num_jobs = cfg.get("num_jobs", 16)

        # Choose processing method based on num_jobs
        if num_jobs == 0:
            logger.info(
                f"Processing {len(key_names)} motions sequentially "
                f"(debugging mode)"
            )
            success_count = process_motion_sequentially(
                key_names, key_name_to_pkls, cfg
            )
        else:
            logger.info(
                f"Processing {len(key_names)} motions with Ray "
                f"({num_jobs} actors)"
            )
            success_count = process_motion_with_ray(
                key_names, key_name_to_pkls, cfg
            )

        logger.info(
            f"Processing completed: {success_count} motions successfully "
            f"processed and saved to {cfg.dump_dir}"
        )
    else:
        logger.info("No motions were successfully processed")


if __name__ == "__main__":
    main()
