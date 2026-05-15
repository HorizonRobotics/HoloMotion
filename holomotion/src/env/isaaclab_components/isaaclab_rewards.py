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


import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
import isaaclab.utils.math as isaaclab_math

from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
from holomotion.src.utils.frame_utils import (
    positions_world_to_env_frame,
    root_relative_positions_from_env_frame,
    root_relative_positions_from_mixed_position_frames,
)
import isaaclab.envs.mdp as isaaclab_mdp
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from loguru import logger
from holomotion.src.env.isaaclab_components.isaaclab_utils import (
    _get_body_indices,
    resolve_holo_config,
    _get_dof_indices,
)


def _joint_ids_to_tensor(
    joint_ids: slice | list[int] | tuple[int, ...] | torch.Tensor | None,
    num_joints: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert joint indices to a dense tensor in articulation order."""
    if joint_ids is None:
        return torch.arange(num_joints, device=device, dtype=torch.long)
    if isinstance(joint_ids, slice):
        if joint_ids == slice(None):
            return torch.arange(num_joints, device=device, dtype=torch.long)
        return torch.arange(num_joints, device=device, dtype=torch.long)[
            joint_ids
        ]
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.to(device=device, dtype=torch.long).flatten()
    return torch.tensor(joint_ids, device=device, dtype=torch.long)


def _select_effort_limit_vector(
    asset: Articulation,
    selected_joint_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a per-joint effort-limit vector from instantiated actuators."""
    num_joints = asset.data.applied_torque.shape[1]
    device = asset.data.applied_torque.device
    dtype = asset.data.applied_torque.dtype

    effort_limit_vec = torch.zeros(num_joints, device=device, dtype=dtype)
    is_filled = torch.zeros(num_joints, device=device, dtype=torch.bool)

    for actuator in asset.actuators.values():
        actuator_joint_ids = _joint_ids_to_tensor(
            actuator.joint_indices, num_joints=num_joints, device=device
        )
        actuator_effort_limit = torch.as_tensor(
            actuator.effort_limit, device=device, dtype=dtype
        )
        if actuator_effort_limit.ndim == 0:
            actuator_effort_limit = actuator_effort_limit.expand(
                actuator_joint_ids.numel()
            )
        elif actuator_effort_limit.ndim == 2:
            if actuator_effort_limit.shape[0] > 1:
                reference = actuator_effort_limit[0].unsqueeze(0)
                if not torch.allclose(
                    actuator_effort_limit,
                    reference.expand_as(actuator_effort_limit),
                ):
                    raise ValueError(
                        "normed_torque_rate requires actuator effort limits to be static across envs."
                    )
            actuator_effort_limit = actuator_effort_limit[0]
        elif actuator_effort_limit.ndim != 1:
            raise ValueError(
                "normed_torque_rate expects actuator effort limits to be scalar, 1-D, or 2-D tensors."
            )

        if actuator_effort_limit.numel() != actuator_joint_ids.numel():
            raise ValueError(
                "normed_torque_rate found mismatched actuator joint indices and effort limits."
            )

        effort_limit_vec[actuator_joint_ids] = actuator_effort_limit
        is_filled[actuator_joint_ids] = True

    if not torch.all(is_filled[selected_joint_ids]):
        missing_joint_ids = selected_joint_ids[~is_filled[selected_joint_ids]]
        raise ValueError(
            "normed_torque_rate could not resolve actuator effort limits for "
            f"joint ids {missing_joint_ids.tolist()}."
        )

    selected_effort_limits = effort_limit_vec[selected_joint_ids]
    if not torch.all(torch.isfinite(selected_effort_limits)):
        raise ValueError(
            "normed_torque_rate requires finite actuator effort limits for all selected joints."
        )
    if not torch.all(selected_effort_limits > 0.0):
        raise ValueError(
            "normed_torque_rate requires strictly positive actuator effort limits for all selected joints."
        )

    return selected_effort_limits


def _body_local_offsets_tensor(
    offsets: list[list[float]] | tuple[tuple[float, float, float], ...],
    *,
    num_bodies: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    offset_tensor = torch.as_tensor(offsets, device=device, dtype=dtype)
    if offset_tensor.shape != (num_bodies, 3):
        raise ValueError(
            "reward_point_body_offset must have shape "
            f"({num_bodies}, 3), got {tuple(offset_tensor.shape)}."
        )
    return offset_tensor


def _body_error_weights_tensor(
    weights: list[float] | tuple[float, ...] | None,
    *,
    num_bodies: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if weights is None:
        return None
    weight_tensor = torch.as_tensor(weights, device=device, dtype=dtype)
    if weight_tensor.shape != (num_bodies,):
        raise ValueError(
            "point_weights must have shape "
            f"({num_bodies},), got {tuple(weight_tensor.shape)}."
        )
    if torch.any(weight_tensor < 0):
        raise ValueError("point_weights must be non-negative.")
    if torch.sum(weight_tensor) <= 0:
        raise ValueError("point_weights must contain a positive weight.")
    return weight_tensor


def _apply_body_local_offsets(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return body_pos_w + isaaclab_math.quat_apply(
        body_quat_w, offsets[None, :, :].expand_as(body_pos_w)
    )


def _weighted_body_error_mean(
    error: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return error.mean(-1)
    return torch.sum(error * weights[None, :], dim=-1) / torch.sum(weights)


def key_dof_position_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    key_dofs: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keydof_idxs = _get_dof_indices(command.robot, key_dofs)
    ref_dof_pos = command.get_ref_motion_dof_pos_immediate_next(
        prefix=ref_prefix
    )
    error = torch.sum(
        torch.square(
            command.robot.data.joint_pos[:, keydof_idxs]
            - ref_dof_pos[:, keydof_idxs]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def key_dof_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    key_dofs: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keydof_idxs = _get_dof_indices(command.robot, key_dofs)
    ref_dof_vel = command.get_ref_motion_dof_vel_immediate_next(
        prefix=ref_prefix
    )
    error = torch.sum(
        torch.square(
            command.robot.data.joint_vel[:, keydof_idxs]
            - ref_dof_vel[:, keydof_idxs]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    ref_motion_command: RefMotionCommand = env.command_manager.get_term(
        command_name
    )
    ref_anchor_pos = ref_motion_command.get_ref_motion_anchor_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )
    robot_anchor_pos = ref_motion_command.global_robot_anchor_pos_cur
    error = torch.sum(
        torch.square(ref_anchor_pos - robot_anchor_pos),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_anchor_quat = (
        command.get_ref_motion_anchor_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )
    error = (
        isaaclab_math.quat_error_magnitude(
            ref_anchor_quat,
            command.robot.data.body_quat_w[:, command.anchor_bodylink_idx],
        )
        ** 2
    )
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor positions/orientations
    ref_anchor_pos = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )  # [B, 3]
    ref_anchor_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4] (w,x,y,z)
    robot_anchor_pos = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body positions in global frame
    ref_body_pos_global = (
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, num_bodies, 3]

    # Transform reference body positions to be relative to robot's current anchor
    # This follows the same logic as the whole_body_tracking implementation

    # Select relevant body indices first
    ref_body_pos_selected = ref_body_pos_global[
        :, keybody_idxs
    ]  # [B, selected_bodies, 3]

    # Expand anchor positions/orientations to match number of selected bodies
    num_bodies = len(keybody_idxs)
    ref_anchor_pos_exp = ref_anchor_pos[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 3]
    ref_anchor_quat_exp = ref_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]
    robot_anchor_pos_exp = robot_anchor_pos[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 3]
    robot_anchor_quat_exp = robot_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]

    # Create delta transformation (preserving z from reference, aligning xy to robot)
    delta_pos = robot_anchor_pos_exp.clone()
    delta_pos[..., 2] = ref_anchor_pos_exp[..., 2]  # Keep reference Z height

    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_anchor_quat_exp,
            isaaclab_math.quat_inv(ref_anchor_quat_exp),
        )
    )

    # Transform reference body positions to relative frame
    ref_body_pos_relative = delta_pos + isaaclab_math.quat_apply(
        delta_ori, ref_body_pos_selected - ref_anchor_pos_exp
    )

    # Get robot body positions
    robot_body_pos = command.robot.data.body_pos_w[:, keybody_idxs]

    # Compute error
    error = torch.sum(
        torch.square(ref_body_pos_relative - robot_body_pos),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def root_rel_keybodylink_pos_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root-relative keybody positions using environment-frame positions.

    IsaacLab MDP root position helpers are expressed in the environment frame
    (simulation-world position minus `env.scene.env_origins`). This reward
    converts body positions into the same environment frame before computing
    root-relative vectors.
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot root positions/orientations
    ref_root_pos_env = positions_world_to_env_frame(
        command.get_ref_motion_root_global_pos_immediate_next(
            prefix=ref_prefix
        ),
        env.scene.env_origins,
    )  # [B, 3]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4] (w,x,y,z)
    robot_root_pos_env = isaaclab_mdp.root_pos_w(env)  # [B, 3]
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4] (w,x,y,z)

    # Select relevant body indices first
    ref_body_pos_env = positions_world_to_env_frame(
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs],
        env.scene.env_origins,
    )
    robot_body_pos_root_rel = (
        root_relative_positions_from_mixed_position_frames(
            body_pos_w=command.robot.data.body_pos_w[:, keybody_idxs],
            root_pos_env=robot_root_pos_env,
            root_quat_w=robot_root_quat_w,
            env_origins=env.scene.env_origins,
        )
    )
    ref_body_pos_root_rel = root_relative_positions_from_env_frame(
        body_pos_env=ref_body_pos_env,
        root_pos_env=ref_root_pos_env,
        root_quat_w=ref_root_quat_w,
    )

    # Compute error
    error = torch.sum(
        torch.square(ref_body_pos_root_rel - robot_body_pos_root_rel),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def local_reward_point_body_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    reward_point_body: list[str],
    reward_point_body_offset: list[list[float]],
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
    point_weights: list[float] | None = None,
) -> torch.Tensor:
    """Track body-local reward points in each entity's root frame.

    `reward_point_body_offset` is expressed in each named body's local frame
    and is rotated by the corresponding body orientation for both reference
    and robot states.
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    body_idxs = _get_body_indices(command.robot, reward_point_body)
    num_bodies = len(body_idxs)
    if len(reward_point_body_offset) != num_bodies:
        raise ValueError(
            "reward_point_body and reward_point_body_offset must have the "
            f"same length, got {num_bodies} and "
            f"{len(reward_point_body_offset)}."
        )

    robot_body_pos_w = command.robot.data.body_pos_w[:, body_idxs]
    robot_body_quat_w = command.robot.data.body_quat_w[:, body_idxs]
    device = robot_body_pos_w.device
    dtype = robot_body_pos_w.dtype
    offsets = _body_local_offsets_tensor(
        reward_point_body_offset,
        num_bodies=num_bodies,
        device=device,
        dtype=dtype,
    )
    weights = _body_error_weights_tensor(
        point_weights,
        num_bodies=num_bodies,
        device=device,
        dtype=dtype,
    )

    ref_body_pos_w = (
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )[:, body_idxs]
    )
    ref_body_quat_w = (
        command.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )[:, body_idxs]
    )

    ref_point_w = _apply_body_local_offsets(
        ref_body_pos_w, ref_body_quat_w, offsets
    )
    robot_point_w = _apply_body_local_offsets(
        robot_body_pos_w, robot_body_quat_w, offsets
    )

    ref_root_pos_w = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )
    robot_root_pos_w = command.robot.data.root_pos_w
    robot_root_quat_w = command.robot.data.root_quat_w

    ref_root_quat_inv = isaaclab_math.quat_inv(ref_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)
    robot_root_quat_inv = isaaclab_math.quat_inv(robot_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)
    ref_point_root_local = isaaclab_math.quat_apply(
        ref_root_quat_inv, ref_point_w - ref_root_pos_w[:, None, :]
    )
    robot_point_root_local = isaaclab_math.quat_apply(
        robot_root_quat_inv, robot_point_w - robot_root_pos_w[:, None, :]
    )

    error = torch.sum(
        torch.square(robot_point_root_local - ref_point_root_local),
        dim=-1,
    )
    return torch.exp(-_weighted_body_error_mean(error, weights) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor orientations
    ref_anchor_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4] (w,x,y,z)
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body orientations in global frame
    ref_body_quat_global = (
        command.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, num_bodies, 4]

    # Select relevant body indices
    ref_body_quat_selected = ref_body_quat_global[
        :, keybody_idxs
    ]  # [B, selected_bodies, 4]

    # Expand anchor orientations to match number of selected bodies
    num_bodies = len(keybody_idxs)
    ref_anchor_quat_exp = ref_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]
    robot_anchor_quat_exp = robot_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]

    # Compute relative orientation transformation (only yaw component)
    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_anchor_quat_exp,
            isaaclab_math.quat_inv(ref_anchor_quat_exp),
        )
    )

    # Transform reference body orientations to relative frame
    ref_body_quat_relative = isaaclab_math.quat_mul(
        delta_ori, ref_body_quat_selected
    )

    # Get robot body orientations
    robot_body_quat = command.robot.data.body_quat_w[:, keybody_idxs]

    # Compute error
    error = (
        isaaclab_math.quat_error_magnitude(
            ref_body_quat_relative, robot_body_quat
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global velocities (no coordinate transformation needed)
    ref_lin_vel = (
        command.get_ref_motion_bodylink_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )
    robot_lin_vel = command.robot.data.body_lin_vel_w[:, keybody_idxs]
    error = torch.sum(torch.square(ref_lin_vel - robot_lin_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global angular velocities (no coordinate transformation needed)
    ref_ang_vel = (
        command.get_ref_motion_bodylink_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )
    robot_ang_vel = command.robot.data.body_ang_vel_w[:, keybody_idxs]
    error = torch.sum(torch.square(ref_ang_vel - robot_ang_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def root_pos_xy_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_pos = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )
    error = torch.sum(
        torch.square(
            ref_root_pos[:, :2] - command.robot.data.root_pos_w[:, :2]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def root_rot_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )
    error = (
        isaaclab_math.quat_error_magnitude(
            ref_root_quat,
            isaaclab_mdp.root_quat_w(env),
        )
        ** 2
    )
    return torch.exp(-error / std**2)


def root_pos_rel_z_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    robot_root_z = command.robot.data.root_pos_w[:, 2]
    ref_root_z = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )[:, 2]
    dz_rel = robot_root_z - ref_root_z
    error = torch.square(dz_rel)
    return torch.exp(-error / std**2)


def _vector_relative_error_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Relative vector error ratio ||pred - target|| / (||target|| + eps)."""
    diff_norm = torch.linalg.vector_norm(pred - target, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    return diff_norm / (target_norm + eps)


def root_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root linear velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_lin_vel_w = isaaclab_mdp.root_lin_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_lin_vel_w = (
        command.get_ref_motion_root_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )
    )
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )

    # Project to respective root frames
    robot_root_lin_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(robot_root_quat_w),
        robot_root_lin_vel_w,
    )  # [B, 3]
    ref_root_lin_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(ref_root_quat_w),
        ref_root_lin_vel_w,
    )  # [B, 3]

    error = torch.sum(
        torch.square(ref_root_lin_vel - robot_root_lin_vel), dim=-1
    )
    return torch.exp(-error / std**2)


def root_lin_vel_tracking_rel_ratio_exp(
    env: ManagerBasedRLEnv,
    std: float,
    eps: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root linear velocity using relative error ratio in each root frame."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    robot_root_lin_vel_w = isaaclab_mdp.root_lin_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_lin_vel_w = (
        command.get_ref_motion_root_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )
    )
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )

    robot_root_lin_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(robot_root_quat_w),
        robot_root_lin_vel_w,
    )
    ref_root_lin_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(ref_root_quat_w),
        ref_root_lin_vel_w,
    )

    error_ratio = _vector_relative_error_ratio(
        robot_root_lin_vel,
        ref_root_lin_vel,
        eps=eps,
    )
    return torch.exp(-torch.square(error_ratio) / std**2)


def root_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root angular velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_ang_vel_w = (
        command.get_ref_motion_root_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )
    )
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )

    # Project to respective root frames
    robot_root_ang_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(robot_root_quat_w),
        robot_root_ang_vel_w,
    )  # [B, 3]
    ref_root_ang_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(ref_root_quat_w),
        ref_root_ang_vel_w,
    )  # [B, 3]

    error = torch.sum(
        torch.square(ref_root_ang_vel - robot_root_ang_vel), dim=-1
    )
    return torch.exp(-error / std**2)


def root_ang_vel_tracking_rel_ratio_exp(
    env: ManagerBasedRLEnv,
    std: float,
    eps: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root angular velocity using relative error ratio in each root frame."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_ang_vel_w = (
        command.get_ref_motion_root_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )
    )
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )

    robot_root_ang_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(robot_root_quat_w),
        robot_root_ang_vel_w,
    )
    ref_root_ang_vel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(ref_root_quat_w),
        ref_root_ang_vel_w,
    )

    error_ratio = _vector_relative_error_ratio(
        robot_root_ang_vel,
        ref_root_ang_vel,
        eps=eps,
    )
    return torch.exp(-torch.square(error_ratio) / std**2)


def root_rel_keybodylink_pos_tracking_l2_exp_bydmmc_style(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track keybody positions using per-entity heading-aligned frames.

    For each of robot and reference:
    - subtract own root position (root-relative in world)
    - rotate by own yaw-only inverse (heading-aligned frame)
    Then compare these root-relative, heading-aligned positions.

    All positions are first converted into IsaacLab's environment frame
    (simulation world minus `env.scene.env_origins`) so robot root and body
    positions use the same translation convention.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root states in environment frame
    ref_root_pos = positions_world_to_env_frame(
        command.get_ref_motion_root_global_pos_immediate_next(
            prefix=ref_prefix
        ),
        env.scene.env_origins,
    )  # [B, 3]
    ref_root_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]
    robot_root_pos = isaaclab_mdp.root_pos_w(env)  # [B, 3]
    robot_root_quat = isaaclab_mdp.root_quat_w(env)  # [B, 4]

    # Body positions in environment frame
    robot_body_pos = positions_world_to_env_frame(
        command.robot.data.body_pos_w[:, keybody_idxs],
        env.scene.env_origins,
    )  # [B, N, 3]
    ref_body_pos = positions_world_to_env_frame(
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs],
        env.scene.env_origins,
    )  # [B, N, 3]

    # Expand for broadcasting
    num_bodies = len(keybody_idxs)
    ref_root_pos_exp = ref_root_pos[:, None, :].expand(-1, num_bodies, -1)
    ref_root_quat_exp = ref_root_quat[:, None, :].expand(-1, num_bodies, -1)
    robot_root_pos_exp = robot_root_pos[:, None, :].expand(-1, num_bodies, -1)
    robot_root_quat_exp = robot_root_quat[:, None, :].expand(
        -1, num_bodies, -1
    )

    # Yaw-only delta orientation (root frames)
    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_root_quat_exp, isaaclab_math.quat_inv(ref_root_quat_exp)
        )
    )  # [B, N, 4]

    # Keep origin at root: compare root-relative vectors after yaw alignment
    robot_rel = robot_body_pos - robot_root_pos_exp  # [B, N, 3]
    ref_rel = ref_body_pos - ref_root_pos_exp  # [B, N, 3]
    ref_rel_aligned = isaaclab_math.quat_apply(delta_ori, ref_rel)  # [B, N, 3]

    # Compare in world (root-relative)
    error = torch.sum(
        torch.square(ref_rel_aligned - robot_rel), dim=-1
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def root_rel_keybodylink_rot_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root-relative keybody rotations in each entity's root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root orientations
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]

    # Body orientations (world)
    robot_body_quat_w = command.robot.data.body_quat_w[
        :, keybody_idxs
    ]  # [B, N, 4]
    ref_body_quat_w = (
        command.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 4]

    # Relative (q_rel = q_root^{-1} * q_body)
    num_bodies = len(keybody_idxs)
    robot_root_quat_inv_exp = isaaclab_math.quat_inv(robot_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)
    ref_root_quat_inv_exp = isaaclab_math.quat_inv(ref_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)

    robot_rel_quat = isaaclab_math.quat_mul(
        robot_root_quat_inv_exp,
        robot_body_quat_w,
    )  # [B, N, 4]
    ref_rel_quat = isaaclab_math.quat_mul(
        ref_root_quat_inv_exp,
        ref_body_quat_w,
    )  # [B, N, 4]

    error = (
        isaaclab_math.quat_error_magnitude(ref_rel_quat, robot_rel_quat) ** 2
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def root_rel_keybodylink_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track keybody linear velocities with motion_relative frame alignment.

    Compute rigid-body-relative velocities for both entities w.r.t. their
    roots, yaw-align reference to robot using root quats, then compare in
    world space.

    Root/body positions used for rigid-body radius vectors are first converted
    into IsaacLab's environment frame (simulation world minus
    `env.scene.env_origins`) so the translation convention matches
    `isaaclab_mdp.root_pos_w(env)`.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root states
    robot_root_pos_w = isaaclab_mdp.root_pos_w(env)  # [B, 3]
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    robot_root_lin_vel_w = isaaclab_mdp.root_lin_vel_w(env)  # [B, 3]
    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)  # [B, 3]

    ref_root_pos_w = positions_world_to_env_frame(
        command.get_ref_motion_root_global_pos_immediate_next(
            prefix=ref_prefix
        ),
        env.scene.env_origins,
    )  # [B, 3]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]
    ref_root_lin_vel_w = (
        command.get_ref_motion_root_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 3]
    ref_root_ang_vel_w = (
        command.get_ref_motion_root_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 3]

    # Body states (world)
    robot_body_pos_w = positions_world_to_env_frame(
        command.robot.data.body_pos_w[:, keybody_idxs],
        env.scene.env_origins,
    )  # [B, N, 3]
    robot_body_lin_vel_w = command.robot.data.body_lin_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_pos_w = positions_world_to_env_frame(
        command.get_ref_motion_bodylink_global_pos_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs],
        env.scene.env_origins,
    )  # [B, N, 3]
    ref_body_lin_vel_w = (
        command.get_ref_motion_bodylink_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]

    # Rigid-body relative (world)
    robot_r_w = robot_body_pos_w - robot_root_pos_w[:, None, :]
    ref_r_w = ref_body_pos_w - ref_root_pos_w[:, None, :]

    robot_cross = torch.cross(
        robot_root_ang_vel_w[:, None, :], robot_r_w, dim=-1
    )  # [B, N, 3]
    ref_cross = torch.cross(
        ref_root_ang_vel_w[:, None, :], ref_r_w, dim=-1
    )  # [B, N, 3]

    robot_v_rel_w = (
        robot_body_lin_vel_w - robot_root_lin_vel_w[:, None, :] - robot_cross
    )  # [B, N, 3]
    ref_v_rel_w = (
        ref_body_lin_vel_w - ref_root_lin_vel_w[:, None, :] - ref_cross
    )  # [B, N, 3]
    # Yaw-only delta orientation from root quats; rotate reference velocities
    num_bodies = len(keybody_idxs)
    robot_root_quat_exp = robot_root_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, N, 4]
    ref_root_quat_exp = ref_root_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, N, 4]
    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_root_quat_exp, isaaclab_math.quat_inv(ref_root_quat_exp)
        )
    )  # [B, N, 4]

    ref_v_rel_aligned_w = isaaclab_math.quat_apply(delta_ori, ref_v_rel_w)

    error = torch.sum(
        torch.square(ref_v_rel_aligned_w - robot_v_rel_w), dim=-1
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def root_rel_keybodylink_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track root-relative keybody angular velocities in root frames.

    Uses w_rel_w = w_body - w_root, then rotates into each entity's root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root orientations and angular velocities
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)  # [B, 3]
    ref_root_quat_w = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]
    ref_root_ang_vel_w = (
        command.get_ref_motion_root_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 3]

    # Body angular velocities (world)
    robot_body_ang_vel_w = command.robot.data.body_ang_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_ang_vel_w = (
        command.get_ref_motion_bodylink_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]

    # Relative (world), then rotate
    robot_w_rel_w = robot_body_ang_vel_w - robot_root_ang_vel_w[:, None, :]
    ref_w_rel_w = ref_body_ang_vel_w - ref_root_ang_vel_w[:, None, :]

    num_bodies = len(keybody_idxs)
    robot_root_quat_inv_exp = isaaclab_math.quat_inv(robot_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)
    ref_root_quat_inv_exp = isaaclab_math.quat_inv(ref_root_quat_w)[
        :, None, :
    ].expand(-1, num_bodies, -1)

    robot_w_rel = isaaclab_math.quat_apply(
        robot_root_quat_inv_exp,
        robot_w_rel_w,
    )  # [B, N, 3]
    ref_w_rel = isaaclab_math.quat_apply(
        ref_root_quat_inv_exp,
        ref_w_rel_w,
    )  # [B, N, 3]

    error = torch.sum(torch.square(ref_w_rel - robot_w_rel), dim=-1)  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def global_keybodylink_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track global keybody linear velocities."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    ref_global_keybody_lin_vel = (
        command.get_ref_motion_bodylink_global_lin_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]
    robot_keybody_lin_vel = command.robot.data.body_lin_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]

    error = torch.sum(
        torch.square(ref_global_keybody_lin_vel - robot_keybody_lin_vel),
        dim=-1,
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


def global_keybodylink_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Track global keybody angular velocities."""
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    ref_global_keybody_ang_vel = (
        command.get_ref_motion_bodylink_global_ang_vel_immediate_next(
            prefix=ref_prefix
        )[:, keybody_idxs]
    )  # [B, N, 3]
    robot_keybody_ang_vel = command.robot.data.body_ang_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]

    error = torch.sum(
        torch.square(ref_global_keybody_ang_vel - robot_keybody_ang_vel),
        dim=-1,
    )  # [B, N]
    return torch.exp(-error.mean(-1) / std**2)


#  @torch.compile
def feet_contact_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[
        :, sensor_cfg.body_ids
    ]
    last_contact_time = contact_sensor.data.last_contact_time[
        :, sensor_cfg.body_ids
    ]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track linear velocity (xy) in the gravity-aligned yaw frame using exponential kernel.

    This mirrors the implementation in IsaacLab locomotion velocity MDP.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = isaaclab_math.quat_apply_inverse(
        isaaclab_math.yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3],
    )
    # vel_yaw = isaaclab_math.quat_rotate_inverse(
    #     isaaclab_math.yaw_quat(asset.data.root_quat_w),
    #     asset.data.root_lin_vel_w[:, :3],
    # )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / (std**2))


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding when in contact using contact forces and foot linear velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_slide_ang_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding when in contact using contact forces and foot linear velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2:3]
    reward = torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward swinging feet clearing a target height with velocity-shaped kernel.

    Only rewards feet that are swinging (not in contact) and are close to the target height.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [B, N]

    delta_z = target_height - foot_z
    delta_z = torch.clamp(delta_z, min=0.0)  # only penalze if below target

    foot_z_error = torch.square(delta_z)  # [B, N]

    # Only reward swinging feet (not in contact)
    is_swinging = torch.ones_like(foot_z_error, dtype=torch.bool)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    )  # [B, N]
    is_swinging = ~is_contact

    # Gate reward by horizontal velocity to ensure feet are actually moving
    foot_horizontal_vel = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
    )  # [B, N]
    velocity_gate = torch.tanh(tanh_mult * foot_horizontal_vel)  # [B, N]

    # Reward: high when error is low (at target height) and foot is swinging
    reward_per_foot = (
        torch.exp(-foot_z_error / std**2) * velocity_gate * is_swinging.float()
    )
    return torch.sum(reward_per_foot, dim=1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    )

    global_phase = (
        (env.episode_length_buf * env.step_dt) % period / period
    ).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(
            env.command_manager.get_command(command_name), dim=1
        )
        reward *= cmd_norm > 0.1
    return reward


joint_deviation_l1_arms = isaaclab_mdp.joint_deviation_l1
joint_deviation_l1_arms_roll = isaaclab_mdp.joint_deviation_l1

joint_deviation_l1_waists = isaaclab_mdp.joint_deviation_l1

joint_deviation_l1_legs = isaaclab_mdp.joint_deviation_l1
joint_deviation_l1_legs_yaw = isaaclab_mdp.joint_deviation_l1

joint_deviation_l1_stand_still = isaaclab_mdp.joint_deviation_l1


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(angle), dim=1)


joint_deviation_l2_arms_roll = joint_deviation_l2
joint_deviation_l2_arms = joint_deviation_l2
joint_deviation_l2_waists = joint_deviation_l2
joint_deviation_l2_legs = joint_deviation_l2
joint_deviation_l2_shoulder_roll = joint_deviation_l2
joint_deviation_l2_hip_roll = joint_deviation_l2


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def positive_work(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only the positive mechanical work (energy injected) by the joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]

    # Calculate raw mechanical power (positive = motoring, negative = braking)
    power = qfrc * qvel

    # Only keep positive values, zero out negative (braking) work
    positive_power = torch.relu(power)

    return torch.sum(positive_power, dim=-1)


class normed_positive_work(ManagerTermBase):
    """Penalize positive joint work normalized by effort and velocity limits."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_name: str | None = None
        self._joint_ids: torch.Tensor | None = None
        self._inv_effort_limit: torch.Tensor | None = None

    def _maybe_build_cache(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> Articulation:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = _joint_ids_to_tensor(
            getattr(asset_cfg, "joint_ids", None),
            num_joints=asset.data.applied_torque.shape[1],
            device=asset.data.applied_torque.device,
        )
        cache_needs_refresh = (
            self._asset_name != asset_cfg.name
            or self._joint_ids is None
            or not torch.equal(self._joint_ids, joint_ids)
            or self._inv_effort_limit is None
            or self._inv_effort_limit.shape != (joint_ids.numel(),)
            or self._inv_effort_limit.device
            != asset.data.applied_torque.device
            or self._inv_effort_limit.dtype != asset.data.applied_torque.dtype
        )
        if not cache_needs_refresh:
            return asset

        effort_limit = _select_effort_limit_vector(asset, joint_ids)
        self._asset_name = asset_cfg.name
        self._joint_ids = joint_ids
        self._inv_effort_limit = effort_limit.reciprocal()
        return asset

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = self._maybe_build_cache(env, asset_cfg)
        joint_ids = self._joint_ids
        inv_effort_limit = self._inv_effort_limit
        assert joint_ids is not None
        assert inv_effort_limit is not None

        current_torque = asset.data.applied_torque[:, joint_ids]
        current_joint_vel = asset.data.joint_vel[:, joint_ids]
        joint_vel_limits = asset.data.joint_vel_limits[:, joint_ids]

        if not torch.all(torch.isfinite(joint_vel_limits)) or not torch.all(
            joint_vel_limits > 0.0
        ):
            raise ValueError(
                "normed_positive_work requires finite, strictly positive "
                "joint velocity limits for all selected joints."
            )

        normalized_power = (current_torque * inv_effort_limit) * (
            current_joint_vel / joint_vel_limits
        )
        return torch.sum(torch.relu(normalized_power), dim=-1)


class normed_torque_rate(ManagerTermBase):
    """Penalize joint torque-rate changes normalized by actuator effort limits."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset_name: str | None = None
        self._joint_ids: torch.Tensor | None = None
        self._inv_effort_limit: torch.Tensor | None = None
        self._prev_applied_torque: torch.Tensor | None = None
        self._needs_reseed = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._needs_reseed[:] = True
            return
        if isinstance(env_ids, slice):
            self._needs_reseed[env_ids] = True
            return
        env_ids_tensor = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        self._needs_reseed[env_ids_tensor] = True

    def _maybe_build_cache(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> Articulation:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = _joint_ids_to_tensor(
            getattr(asset_cfg, "joint_ids", None),
            num_joints=asset.data.applied_torque.shape[1],
            device=asset.data.applied_torque.device,
        )
        cache_needs_refresh = (
            self._asset_name != asset_cfg.name
            or self._joint_ids is None
            or not torch.equal(self._joint_ids, joint_ids)
            or self._prev_applied_torque is None
            or self._prev_applied_torque.shape
            != (env.num_envs, joint_ids.numel())
            or self._prev_applied_torque.device
            != asset.data.applied_torque.device
            or self._prev_applied_torque.dtype
            != asset.data.applied_torque.dtype
        )
        if not cache_needs_refresh:
            return asset

        effort_limit = _select_effort_limit_vector(asset, joint_ids)
        self._asset_name = asset_cfg.name
        self._joint_ids = joint_ids
        self._inv_effort_limit = effort_limit.reciprocal()
        self._prev_applied_torque = torch.zeros(
            env.num_envs,
            joint_ids.numel(),
            device=asset.data.applied_torque.device,
            dtype=asset.data.applied_torque.dtype,
        )
        self._needs_reseed = torch.ones(
            env.num_envs,
            device=asset.data.applied_torque.device,
            dtype=torch.bool,
        )
        return asset

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = self._maybe_build_cache(env, asset_cfg)
        joint_ids = self._joint_ids
        inv_effort_limit = self._inv_effort_limit
        prev_applied_torque = self._prev_applied_torque
        assert joint_ids is not None
        assert inv_effort_limit is not None
        assert prev_applied_torque is not None

        current_torque = asset.data.applied_torque[:, joint_ids]
        reward = torch.zeros(
            env.num_envs,
            device=current_torque.device,
            dtype=current_torque.dtype,
        )

        reseed_mask = self._needs_reseed.clone()
        if hasattr(env, "episode_length_buf"):
            reseed_mask |= env.episode_length_buf == 0

        active_mask = ~reseed_mask
        if torch.any(active_mask):
            delta = (
                current_torque[active_mask] - prev_applied_torque[active_mask]
            ) * inv_effort_limit
            reward[active_mask] = torch.sum(delta.square(), dim=1)

        prev_applied_torque.copy_(current_torque)
        self._needs_reseed[reseed_mask] = False

        return reward


def track_stand_still_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track stand still joint position using exponential kernel when command velocity is low.

    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    error = torch.sum(
        torch.square(asset.data.joint_pos - asset.data.default_joint_pos),
        dim=1,
    )
    # Use generated velocity commands (vx, vy, yaw_rate). Some command terms may
    # expose additional channels (e.g., heading) via get_command().
    cmd = isaaclab_mdp.generated_commands(env, command_name=command_name)
    if cmd.shape[-1] > 3:
        cmd = cmd[..., :3]
    cmd_norm = torch.norm(cmd, dim=1)
    return torch.exp(-error / std**2) * (cmd_norm < 0.1)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize L1 joint deviation from default pose when command velocity is low.

    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # L1 error: sum(|q - q_default|)
    error = torch.sum(
        torch.abs(asset.data.joint_pos - asset.data.default_joint_pos),
        dim=1,
    )

    cmd = isaaclab_mdp.generated_commands(env, command_name=command_name)
    if cmd.shape[-1] > 3:
        cmd = cmd[..., :3]
    cmd_norm = torch.norm(cmd, dim=1)

    # Return error (to be penalized with negative weight) only when standing still
    return error * (cmd_norm < 0.1)


def stand_still_action_rate(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    cmd = isaaclab_mdp.generated_commands(env, command_name=command_name)
    if cmd.shape[-1] > 3:
        cmd = cmd[..., :3]
    stand_still = torch.norm(cmd, dim=1) < 0.1
    return (
        torch.sum(
            torch.square(
                env.action_manager.action - env.action_manager.prev_action
            ),
            dim=1,
        )
        * stand_still
    )


def stand_still_dof_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    cmd = isaaclab_mdp.generated_commands(env, command_name=command_name)
    if cmd.shape[-1] > 3:
        cmd = cmd[..., :3]
    stand_still = torch.norm(cmd, dim=1) < 0.1
    return (
        torch.sum(
            torch.square(env.scene[asset_cfg.name].data.joint_vel),
            dim=1,
        )
        * stand_still
    )


class action_acc(ManagerTermBase):
    """Penalize the change in action-rate using a stateful second difference."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_action: torch.Tensor | None = None
        self._prev_action_rate: torch.Tensor | None = None
        self._needs_reseed = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._needs_prev_rate = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._needs_reseed[:] = True
            self._needs_prev_rate[:] = True
            return
        if isinstance(env_ids, slice):
            self._needs_reseed[env_ids] = True
            self._needs_prev_rate[env_ids] = True
            return
        env_ids_tensor = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        self._needs_reseed[env_ids_tensor] = True
        self._needs_prev_rate[env_ids_tensor] = True

    def _maybe_build_cache(
        self, env: ManagerBasedRLEnv
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_action = env.action_manager.action
        cache_needs_refresh = (
            self._prev_action is None
            or self._prev_action_rate is None
            or self._prev_action.shape != current_action.shape
            or self._prev_action.device != current_action.device
            or self._prev_action.dtype != current_action.dtype
            or self._prev_action_rate.shape != current_action.shape
            or self._prev_action_rate.device != current_action.device
            or self._prev_action_rate.dtype != current_action.dtype
        )
        if cache_needs_refresh:
            self._prev_action = torch.zeros_like(current_action)
            self._prev_action_rate = torch.zeros_like(current_action)
            self._needs_reseed = torch.ones(
                env.num_envs,
                device=current_action.device,
                dtype=torch.bool,
            )
            self._needs_prev_rate = torch.ones(
                env.num_envs,
                device=current_action.device,
                dtype=torch.bool,
            )

        assert self._prev_action is not None
        assert self._prev_action_rate is not None
        return self._prev_action, self._prev_action_rate

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_action = env.action_manager.action
        prev_action, prev_action_rate = self._maybe_build_cache(env)
        reward = torch.zeros(
            env.num_envs,
            device=current_action.device,
            dtype=current_action.dtype,
        )

        reseed_mask = self._needs_reseed.clone()
        if hasattr(env, "episode_length_buf"):
            reseed_mask |= env.episode_length_buf == 0

        if torch.any(reseed_mask):
            prev_action[reseed_mask] = current_action[reseed_mask]
            prev_action_rate[reseed_mask].zero_()
            self._needs_prev_rate[reseed_mask] = True

        active_mask = ~reseed_mask
        if torch.any(active_mask):
            current_action_rate = (
                current_action[active_mask] - prev_action[active_mask]
            )
            ready_mask = ~self._needs_prev_rate[active_mask]
            if torch.any(ready_mask):
                action_acc_value = (
                    current_action_rate[ready_mask]
                    - prev_action_rate[active_mask][ready_mask]
                )
                reward[
                    active_mask.nonzero(as_tuple=False).flatten()[ready_mask]
                ] = torch.sum(action_acc_value.square(), dim=1)

            prev_action[active_mask] = current_action[active_mask]
            prev_action_rate[active_mask] = current_action_rate
            self._needs_prev_rate[active_mask] = False

        self._needs_reseed[reseed_mask] = False
        return reward


action_acc_l2 = action_acc


def feet_stumble(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    )
    forces_xy = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2
    )
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_too_near(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    )

    cmd = isaaclab_mdp.generated_commands(env, command_name=command_name)
    if cmd.shape[-1] > 3:
        cmd = cmd[..., :3]
    command_norm = torch.norm(cmd, dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def torso_xy_ang_vel_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")

    # World-frame torso angular velocity: [B, 3]
    torso_ang_vel_w: torch.Tensor = robot_ptr.data.body_ang_vel_w[
        :, torso_idx, :
    ]

    # Heading-aligned frame: z-up, x-forward, y-left, defined by robot yaw heading.
    # Build yaw-only quaternion from stored heading_w (shape [B]).
    heading_yaw: torch.Tensor = robot_ptr.data.heading_w  # [B]
    zero = torch.zeros_like(heading_yaw, device=env.device)
    heading_quat_wxyz: torch.Tensor = isaaclab_math.quat_from_euler_xyz(
        roll=zero,
        pitch=zero,
        yaw=heading_yaw,
    )  # [B, 4]

    # Re-express torso angular velocity in heading-aligned frame.
    heading_inv_wxyz: torch.Tensor = isaaclab_math.quat_inv(heading_quat_wxyz)
    torso_ang_vel_h: torch.Tensor = isaaclab_math.quat_apply(
        heading_inv_wxyz,
        torso_ang_vel_w,
    )  # [B, 3]

    # Penalize lateral components (x, y) with squared magnitude.
    penalty: torch.Tensor = torch.sum(
        torch.square(torso_ang_vel_h[:, :2]),
        dim=-1,
    )  # [B]
    return penalty


def torso_upright_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_rot_quat_w = robot_ptr.data.body_quat_w[:, torso_idx, :]

    # Heading-aligned frame: z-up, x-forward, y-left, defined by robot yaw heading.
    # Build yaw-only quaternion from stored heading_w (shape [B]).
    heading_yaw: torch.Tensor = robot_ptr.data.heading_w  # [B]
    zero = torch.zeros_like(heading_yaw, device=env.device)
    heading_quat_wxyz: torch.Tensor = isaaclab_math.quat_from_euler_xyz(
        roll=zero,
        pitch=zero,
        yaw=heading_yaw,
    )  # [B, 4]

    # Re-express torso angular velocity in heading-aligned frame.
    heading_inv_wxyz: torch.Tensor = isaaclab_math.quat_inv(heading_quat_wxyz)
    torso_rot_quat_h: torch.Tensor = isaaclab_math.quat_mul(
        heading_inv_wxyz,
        torso_rot_quat_w,
    )  # [B, 3]

    # get the roll and pitch
    roll, pitch, _ = isaaclab_math.euler_xyz_from_quat(torso_rot_quat_h)
    pitch *= pitch > 0.0
    rollpitch = torch.stack([roll * 2.0, pitch], dim=-1)

    # Penalize lateral components (x, y) with squared magnitude.
    penalty: torch.Tensor = torch.sum(
        torch.square(rollpitch),
        dim=-1,
    )  # [B]
    return penalty


def torso_upright_l2_penalty_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_pitch: float = 0.0,
    roll_scale: float = 2.0,
    pitch_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize torso roll/pitch deviation in a heading-aligned frame (symmetric).

    Compared to `torso_upright_l2_penalty`, this version penalizes *both* forward
    and backward pitch w.r.t. `target_pitch`.

    Returns: [B]
    """
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_rot_quat_w: torch.Tensor = robot_ptr.data.body_quat_w[
        :, torso_idx, :
    ]  # [B, 4]

    heading_yaw: torch.Tensor = robot_ptr.data.heading_w  # [B]
    zero = torch.zeros_like(heading_yaw, device=env.device)
    heading_quat_wxyz: torch.Tensor = isaaclab_math.quat_from_euler_xyz(
        roll=zero,
        pitch=zero,
        yaw=heading_yaw,
    )  # [B, 4]

    heading_inv_wxyz: torch.Tensor = isaaclab_math.quat_inv(heading_quat_wxyz)
    torso_rot_quat_h: torch.Tensor = isaaclab_math.quat_mul(
        heading_inv_wxyz,
        torso_rot_quat_w,
    )  # [B, 4]

    roll, pitch, _ = isaaclab_math.euler_xyz_from_quat(torso_rot_quat_h)  # [B]
    roll_err: torch.Tensor = roll_scale * roll
    pitch_err: torch.Tensor = pitch_scale * (pitch - target_pitch)
    roll_pitch = torch.stack([roll_err, pitch_err], dim=-1)  # [B, 2]

    penalty: torch.Tensor = torch.sum(torch.square(roll_pitch), dim=-1)  # [B]
    return penalty


def stand_still_torso_upright_exp_v2(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
    target_pitch: float = 0.0,
    roll_scale: float = 2.0,
    pitch_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward torso uprightness under stand-still commands using an exp kernel.

    Reward:
        exp(-penalty / std^2)  if ||cmd|| <= cmd_threshold else 0
    where penalty is computed by `torso_upright_l2_penalty_v2`.

    Returns: [B]
    """
    command = env.command_manager.get_command(command_name)
    stand_still_flag: torch.Tensor = (
        torch.norm(command, dim=1) <= cmd_threshold
    )

    penalty = torso_upright_l2_penalty_v2(
        env,
        asset_cfg=asset_cfg,
        target_pitch=target_pitch,
        roll_scale=roll_scale,
        pitch_scale=pitch_scale,
    )  # [B]
    reward = torch.exp(-penalty / std**2)  # [B]
    return reward * stand_still_flag.to(dtype=reward.dtype)


def torso_linacc_xy_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")

    # World-frame torso angular velocity: [B, 3]

    torso_linacc_w = robot_ptr.data.body_lin_acc_w[:, torso_idx, :]

    # Heading-aligned frame: z-up, x-forward, y-left, defined by robot yaw heading.
    # Build yaw-only quaternion from stored heading_w (shape [B]).
    heading_yaw: torch.Tensor = robot_ptr.data.heading_w  # [B]
    zero = torch.zeros_like(heading_yaw, device=env.device)
    heading_quat_wxyz: torch.Tensor = isaaclab_math.quat_from_euler_xyz(
        roll=zero,
        pitch=zero,
        yaw=heading_yaw,
    )  # [B, 4]

    # Re-express torso angular velocity in heading-aligned frame.
    heading_inv_wxyz: torch.Tensor = isaaclab_math.quat_inv(heading_quat_wxyz)
    torso_linacc_h: torch.Tensor = isaaclab_math.quat_apply(
        heading_inv_wxyz,
        torso_linacc_w,
    )  # [B, 3]

    # Penalize lateral components (x, y) with squared magnitude.
    penalty: torch.Tensor = torch.sum(
        torch.square(torso_linacc_h),
        dim=-1,
    )  # [B]
    return penalty


def track_lin_vel_xy_heading_aligned_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Track linear velocity (xy) in the heading-aligned frame using exponential kernel.
    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = isaaclab_math.quat_apply_inverse(
        isaaclab_math.yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3],
    )
    command = env.command_manager.get_command(command_name)
    stand_still_envs = torch.norm(command, dim=1) <= 0.1

    # treat yaw-only envs as zero-translation targets too
    # (vx, vy are approx 0 by definition)
    zero_lin_vel_envs = stand_still_envs
    tracking_targets = torch.where(
        zero_lin_vel_envs[:, None], 0.0, command[:, :2]
    )
    lin_vel_error = torch.sum(
        torch.square(tracking_targets - vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_heading_aligned_frame_exp_v2(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = isaaclab_math.quat_apply_inverse(
        isaaclab_math.yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3],
    )
    command = env.command_manager.get_command(command_name)

    yaw_envs = (torch.norm(command[:, :2], dim=1) < 0.1) & (
        torch.abs(command[:, 2]) > 0.1
    )
    stand_still_envs = torch.norm(command, dim=1) <= 0.1

    # treat yaw-only envs as zero-translation targets too
    # (vx, vy are approx 0 by definition)
    zero_lin_vel_envs = stand_still_envs | yaw_envs
    tracking_targets = torch.where(
        zero_lin_vel_envs[:, None], 0.0, command[:, :2]
    )
    lin_vel_error = torch.sum(
        torch.square(tracking_targets - vel_yaw[:, :2]),
        dim=1,
    )

    # encourage zero linear velocity for stand still environments, and encourage yaw-only environments to have more
    # precise zero linear velocity tracking too
    reward_weights = torch.where(yaw_envs, 2.0, 1.0) + torch.where(
        stand_still_envs, 10.0, 0.0
    )

    return reward_weights * torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_heading_aligned_frame_exp_v2(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Track angular velocity (z) in the heading-aligned frame using exponential kernel.
    Note that the angular velocity in the world frame is the same as the angular velocity in the heading-aligned frame.
    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    yaw_envs = (torch.norm(command[:, :2], dim=1) < 0.1) & (
        torch.abs(command[:, 2]) > 0.1
    )
    stand_still_envs = torch.norm(command, dim=1) <= 0.1

    # set the tracking targets to 0.0 for stand still environments
    tracking_targets = torch.where(stand_still_envs, 0.0, command[:, 2])

    ang_vel_error = torch.square(
        tracking_targets - asset.data.root_ang_vel_w[:, 2]
    )

    # encourage zero angular velocity for stand still environments, and encourage yaw-only environments to have more
    # precise angular velocity tracking
    reward_weights = torch.where(yaw_envs, 2.0, 1.0) + torch.where(
        stand_still_envs, 10.0, 0.0
    )
    return reward_weights * torch.exp(-ang_vel_error / std**2)


def track_ang_vel_z_heading_aligned_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Track angular velocity (z) in the heading-aligned frame using exponential kernel.
    Note that the angular velocity in the world frame is the same as the angular velocity in the heading-aligned frame.
    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    stand_still_envs = torch.norm(command, dim=1) <= 0.1
    tracking_targets = torch.where(stand_still_envs, 0.0, command[:, 2])
    ang_vel_error = torch.square(
        tracking_targets - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def smoothed_track_ang_vel_z_heading_aligned_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Track angular velocity (z) in the heading-aligned frame using exponential kernel.
    Note that the angular velocity in the world frame is the same as the angular velocity in the heading-aligned frame.
    Returns: [B]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    hist_robot_heading_aligned_ang_vel_z = env.observation_manager.compute()[
        "unified"
    ]["rew_heading_aligned_root_ang_vel"][..., 2]
    ep_len = env.episode_length_buf
    obs_window_len = hist_robot_heading_aligned_ang_vel_z.shape[1]
    smooth_window = torch.minimum(
        torch.full_like(ep_len, obs_window_len), ep_len
    )
    smoothed_robot_heading_aligned_ang_vel_z = (
        hist_robot_heading_aligned_ang_vel_z.sum(dim=1) / smooth_window
    )
    ang_vel_error = torch.square(
        command[:, 2] - smoothed_robot_heading_aligned_ang_vel_z
    )
    return torch.exp(-ang_vel_error / std**2)


def feet_air_time(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command = env.command_manager.get_command(command_name)
    reward *= (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) > 0.1
    return reward


def feet_air_time_v2(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command = env.command_manager.get_command(command_name)
    stand_still_envs_flag = torch.norm(command, dim=1) <= 0.1
    ang_z_only_mask = (torch.norm(command[:, :2], dim=1) <= 0.1) & (
        torch.abs(command[:, 2]) > 0.1
    )
    # Stand still: 0.0, yaw-only: 10.0, other: 1.0
    reward_weights = torch.where(
        stand_still_envs_flag, 0.0, 1.0
    ) + torch.where(ang_z_only_mask, 5.0, 0.0)
    return reward * reward_weights


def feet_air_time_v3(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for stand still commands, larger reward for yaw-only commands
    commands = env.command_manager.get_command(command_name)
    stand_still_envs = torch.norm(commands, dim=1) <= 0.1
    yaw_only_envs = (torch.norm(commands[:, :2], dim=1) <= 0.1) & (
        torch.abs(commands[:, 2]) > 0.1
    )
    reward_weights = torch.where(stand_still_envs, 0.0, 1.0) + torch.where(
        yaw_only_envs, 4.0, 0.0
    )

    return reward * reward_weights


def feet_air_time_v4(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for stand still commands, larger reward for yaw-only commands
    commands = env.command_manager.get_command(command_name)
    stand_still_envs = torch.norm(commands, dim=1) <= 0.1
    reward_weights = torch.where(stand_still_envs, 0.0, 1.0)

    return reward * reward_weights


def yaw_rate_only_movement_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize world-frame root XY translation during yaw-only commands.

    When vx, vy are small commands, penalize
    the squared magnitude of root linear velocity (vx, vy) in world frame.

    Returns: [B]
    """
    # Velocity command: [B, 3] (vx, vy, yaw_rate). Some command terms may
    # expose extra channels (e.g., heading) via generated_commands().
    command = env.command_manager.get_command(command_name)

    # Gate only yaw-rate-only envs: vx=vy=0 and v_yaw > 0.0.
    yaw_only_mask: torch.Tensor = (
        torch.norm(command[:, :2], dim=1) <= 0.1
    )  # [B]

    # Penalize global (world-frame) root linear velocity in x/y.
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_w: torch.Tensor = asset.data.root_lin_vel_w  # [B, 3]
    penalty: torch.Tensor = torch.sum(
        torch.square(root_lin_vel_w[:, :2]),
        dim=1,
    )  # [B]
    return penalty * yaw_only_mask.to(dtype=penalty.dtype)


def fly(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > threshold
    )
    return torch.sum(is_contact, dim=-1) < 0.5


def stand_still_torso_lin_vel_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_lin_vel_w = robot_ptr.data.body_lin_vel_w[:, torso_idx, :]
    penalty = torch.sum(torch.square(torso_lin_vel_w), dim=-1)
    command = env.command_manager.get_command(command_name)
    stand_still_flag = torch.norm(command, dim=1) <= 0.1
    return penalty * stand_still_flag.to(dtype=penalty.dtype)


def stand_still_torso_ang_vel_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_ang_vel_w = robot_ptr.data.body_ang_vel_w[:, torso_idx, :]
    penalty = torch.sum(torch.square(torso_ang_vel_w), dim=-1)
    command = env.command_manager.get_command(command_name)
    stand_still_flag = torch.norm(command, dim=1) <= 0.1
    return penalty * stand_still_flag.to(dtype=penalty.dtype)


def stand_still_torso_lin_vel_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward staying still (zero torso linear velocity) when commanded to stand.

    Uses exponential kernel: exp(-||v||^2 / std^2)

    Args:
        env: Environment instance
        std: Standard deviation for exponential kernel
        asset_cfg: Robot asset configuration
        command_name: Name of velocity command

    Returns:
        Reward tensor of shape [B], active only when stand still commanded
    """
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_lin_vel_w = robot_ptr.data.body_lin_vel_w[:, torso_idx, :]
    error = torch.sum(torch.square(torso_lin_vel_w), dim=-1)
    reward = torch.exp(-error / std**2)
    command = env.command_manager.get_command(command_name)
    stand_still_flag = torch.norm(command, dim=1) <= 0.1
    return reward * stand_still_flag.to(dtype=reward.dtype)


def stand_still_torso_ang_vel_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward staying still (zero torso angular velocity) when commanded to stand.

    Uses exponential kernel: exp(-||omega||^2 / std^2)

    Args:
        env: Environment instance
        std: Standard deviation for exponential kernel
        asset_cfg: Robot asset configuration
        command_name: Name of velocity command

    Returns:
        Reward tensor of shape [B], active only when stand still commanded
    """
    robot_ptr = env.scene[asset_cfg.name]
    torso_idx = robot_ptr.body_names.index("torso_link")
    torso_ang_vel_w = robot_ptr.data.body_ang_vel_w[:, torso_idx, :]
    error = torch.sum(torch.square(torso_ang_vel_w), dim=-1)
    reward = torch.exp(-error / std**2)
    command = env.command_manager.get_command(command_name)
    stand_still_flag = torch.norm(command, dim=1) <= 0.1
    return reward * stand_still_flag.to(dtype=reward.dtype)


def yaw_rate_only_movement_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward minimal XY translation during yaw-only commands.

    When vx, vy commands are small, reward staying in place using exponential kernel.
    Uses exponential kernel: exp(-||v_xy||^2 / std^2)

    Args:
        env: Environment instance
        std: Standard deviation for exponential kernel
        asset_cfg: Robot asset configuration
        command_name: Name of velocity command

    Returns:
        Reward tensor of shape [B], active only during yaw-only commands
    """
    command = env.command_manager.get_command(command_name)
    yaw_only_mask: torch.Tensor = torch.norm(command[:, :2], dim=1) <= 0.1

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_w: torch.Tensor = asset.data.root_lin_vel_w
    error: torch.Tensor = torch.sum(torch.square(root_lin_vel_w[:, :2]), dim=1)
    reward: torch.Tensor = torch.exp(-error / std**2)
    return reward * yaw_only_mask.to(dtype=reward.dtype)


def yaw_rate_only_hip_yaw_usage_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    hip_yaw_dofs: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_threshold: float = 0.1,
    yaw_threshold: float = 0.1,
    command_tanh_mult: float = 1.0,
) -> torch.Tensor:
    """Encourage using hip_yaw joint(s) during yaw-rate-only commands.

    Active only when commanded to rotate in place (vx, vy small and |yaw_rate| large).
    Rewards hip_yaw joint velocity magnitude using a saturating exponential kernel:
        r = (1 - exp(-mean(qd_hip_yaw^2) / std^2)) * tanh(command_tanh_mult * |cmd_yaw|)

    Shapes:
    - command: [B, 3] (vx, vy, yaw_rate)
    - asset.data.joint_vel: [B, num_dofs]
    - return: [B]
    """
    command: torch.Tensor = env.command_manager.get_command(command_name)
    yaw_only_mask: torch.Tensor = (
        torch.norm(command[:, :2], dim=1) <= lin_threshold
    ) & (torch.abs(command[:, 2]) > yaw_threshold)  # [B]

    asset: Articulation = env.scene[asset_cfg.name]
    if hip_yaw_dofs is None:
        raise ValueError(
            "yaw_rate_only_hip_yaw_usage_exp requires hip_yaw_dofs (joint names in "
            f"robot.joint_names). Got hip_yaw_dofs=None. robot.joint_names={asset.joint_names}"
        )
    hip_yaw_joint_ids: list[int] = _get_dof_indices(asset, hip_yaw_dofs)

    hip_yaw_vel: torch.Tensor = asset.data.joint_vel[
        :, hip_yaw_joint_ids
    ]  # [B, N]
    activity_sq: torch.Tensor = torch.mean(
        torch.square(hip_yaw_vel), dim=-1
    )  # [B]
    usage_reward: torch.Tensor = 1.0 - torch.exp(-activity_sq / std**2)  # [B]

    cmd_yaw_abs: torch.Tensor = torch.abs(command[:, 2])  # [B]
    cmd_weight: torch.Tensor = torch.tanh(
        command_tanh_mult * cmd_yaw_abs
    )  # [B]

    reward: torch.Tensor = usage_reward * cmd_weight
    return reward * yaw_only_mask.to(dtype=reward.dtype)


@configclass
class RewardsCfg:
    pass


class TaskGatedReward:
    """Callable wrapper to gate reward terms by task_id."""

    def __init__(self, func, task_name: str):
        self.func = func
        self.task_name = task_name
        self.__name__ = f"TaskGatedReward[{task_name}]"

    def __call__(self, env: ManagerBasedRLEnv, *args, **kwargs):
        task_ids = getattr(env, "holo_task_ids", None)
        mapping = getattr(env, "holo_task_name_to_id", None)
        if task_ids is None or mapping is None:
            return torch.zeros(env.num_envs, device=env.device)
        target = mapping.get(self.task_name, None)
        if target is None:
            return torch.zeros(env.num_envs, device=env.device)
        mask = task_ids == target
        if not torch.any(mask):
            return torch.zeros(env.num_envs, device=env.device)

        inner_args = kwargs.pop("args", None)
        inner_kwargs = kwargs.pop("kwargs", None)
        call_args = args if inner_args is None else (*args, *inner_args)
        call_kwargs = (
            kwargs if inner_kwargs is None else {**kwargs, **inner_kwargs}
        )

        reward = self.func(env, *call_args, **call_kwargs)
        mask = mask.to(device=reward.device, dtype=reward.dtype)
        return reward * mask


def build_rewards_config(reward_config_dict: dict):
    if isinstance(reward_config_dict, (DictConfig, ListConfig)):
        reward_config_dict = OmegaConf.to_container(
            reward_config_dict, resolve=True
        )

    rewards_cfg = RewardsCfg()

    # Detect grouped (multi-task) vs flat (legacy) layout
    def _is_grouped(cfg: dict) -> bool:
        for k, v in cfg.items():
            if k == "_config":
                continue
            if isinstance(v, dict) and "weight" in v:
                return False
            return True
        return False

    is_grouped = _is_grouped(reward_config_dict)

    if not is_grouped:
        for reward_name, reward_cfg in reward_config_dict.items():
            if reward_name == "_config":
                continue
            reward_cfg = resolve_holo_config(reward_cfg)
            base_params = resolve_holo_config(reward_cfg["params"])
            method_name = f"{reward_name}"
            func = globals().get(method_name, None)
            if func is None:
                func = getattr(isaaclab_mdp, reward_name, None)
            if func is None:
                raise ValueError(f"Unknown reward function: {reward_name}")
            params = dict(base_params)
            setattr(
                rewards_cfg,
                reward_name,
                RewardTermCfg(
                    func=func,
                    weight=reward_cfg["weight"],
                    params=params,
                ),
            )
        return rewards_cfg

    # Grouped: rewards: {task_name: {term: ...}}
    for task_name, task_group in reward_config_dict.items():
        if task_name.startswith("_"):
            continue
        if not isinstance(task_group, dict):
            raise ValueError(f"Expected dict for task group {task_name}")
        for reward_name, reward_cfg in task_group.items():
            reward_cfg = resolve_holo_config(reward_cfg)
            base_params = resolve_holo_config(reward_cfg["params"])
            method_name = f"{reward_name}"
            func = globals().get(method_name, None)
            if func is None:
                func = getattr(isaaclab_mdp, reward_name, None)
            if func is None:
                raise ValueError(f"Unknown reward function: {reward_name}")
            if task_name != "common":
                func = TaskGatedReward(func, task_name)
                params = {"args": [], "kwargs": base_params}
            else:
                params = base_params
            flat_name = f"{task_name}.{reward_name}"
            setattr(
                rewards_cfg,
                flat_name,
                RewardTermCfg(
                    func=func,
                    weight=reward_cfg["weight"],
                    params=params,
                ),
            )

    return rewards_cfg
