import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
import isaaclab.utils.math as isaaclab_math

from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
import isaaclab.envs.mdp as isaaclab_mdp
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from loguru import logger
from holomotion.src.env.isaaclab_components.isaaclab_utils import (
    _get_body_indices,
    resolve_holo_config,
)


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    ref_motion_command: RefMotionCommand = env.command_manager.get_term(
        command_name
    )
    error = torch.sum(
        torch.square(
            ref_motion_command.ref_motion_anchor_bodylink_global_pos_cur
            - ref_motion_command.global_robot_anchor_pos_cur
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    error = (
        isaaclab_math.quat_error_magnitude(
            command.ref_motion_anchor_bodylink_global_rot_cur_wxyz,
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
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor positions/orientations
    ref_anchor_pos = command.ref_motion_root_global_pos_cur  # [B, 3]
    ref_anchor_quat = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4] (w,x,y,z)
    robot_anchor_pos = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body positions in global frame
    ref_body_pos_global = (
        command.ref_motion_bodylink_global_pos_cur
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
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor positions/orientations
    ref_root_pos_w = command.ref_motion_root_global_pos_cur  # [B, 3]
    ref_root_quat_w = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4] (w,x,y,z)
    robot_root_pos_w = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_root_quat_w = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body positions in global frame
    ref_body_pos_w = (
        command.ref_motion_bodylink_global_pos_cur
    )  # [B, num_bodies, 3]

    # Transform reference body positions to be relative to robot's current anchor
    # This follows the same logic as the whole_body_tracking implementation

    # Select relevant body indices first
    ref_body_pos_selected_w = ref_body_pos_w[
        :, keybody_idxs
    ]  # [B, selected_bodies, 3]
    robot_body_pos_selected_w = command.robot.data.body_pos_w[:, keybody_idxs]

    # Expand anchor positions/orientations to match number of selected bodies
    num_bodies = len(keybody_idxs)
    ref_root_pos_expanded_w = ref_root_pos_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 3]
    ref_root_quat_expaned_w = ref_root_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]
    robot_root_pos_expaned_w = robot_root_pos_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 3]
    robot_root_quat_expanded_w = robot_root_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, num_bodies, 4]

    # Transform reference body positions to relative frame
    ref_body_pos_root_rel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(ref_root_quat_expaned_w),
        ref_body_pos_selected_w - ref_root_pos_expanded_w,
    )

    # Get robot body positions
    robot_body_pos_root_rel = isaaclab_math.quat_apply(
        isaaclab_math.quat_inv(robot_root_quat_expanded_w),
        robot_body_pos_selected_w - robot_root_pos_expaned_w,
    )

    # Compute error
    error = torch.sum(
        torch.square(ref_body_pos_root_rel - robot_body_pos_root_rel),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor orientations
    ref_anchor_quat = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4] (w,x,y,z)
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body orientations in global frame
    ref_body_quat_global = (
        command.ref_motion_bodylink_global_rot_wxyz_cur
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
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global velocities (no coordinate transformation needed)
    error = torch.sum(
        torch.square(
            command.ref_motion_bodylink_global_lin_vel_cur[:, keybody_idxs]
            - command.robot.data.body_lin_vel_w[:, keybody_idxs]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global angular velocities (no coordinate transformation needed)
    error = torch.sum(
        torch.square(
            command.ref_motion_bodylink_global_ang_vel_cur[:, keybody_idxs]
            - command.robot.data.body_ang_vel_w[:, keybody_idxs]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def root_pos_xy_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.ref_motion_root_global_pos_cur[:, :2]
            - command.robot.data.root_pos_w[:, :2]
        ),
        dim=-1,
    )
    return torch.exp(-error / std**2)


def root_rot_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    error = (
        isaaclab_math.quat_error_magnitude(
            command.ref_motion_root_global_rot_quat_wxyz_cur,
            isaaclab_mdp.root_quat_w(env),
        )
        ** 2
    )
    return torch.exp(-error / std**2)


def root_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    """Track root linear velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_lin_vel_w = isaaclab_mdp.root_lin_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_lin_vel_w = command.ref_motion_root_global_lin_vel_cur
    ref_root_quat_w = command.ref_motion_root_global_rot_quat_wxyz_cur

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


def root_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    """Track root angular velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_ang_vel_w = command.ref_motion_root_global_ang_vel_cur
    ref_root_quat_w = command.ref_motion_root_global_rot_quat_wxyz_cur

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


def root_rel_keybodylink_pos_tracking_l2_exp_bydmmc_style(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
) -> torch.Tensor:
    """Track keybody positions using per-entity heading-aligned frames.

    For each of robot and reference:
    - subtract own root position (root-relative in world)
    - rotate by own yaw-only inverse (heading-aligned frame)
    Then compare these root-relative, heading-aligned positions.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Anchor and root states
    ref_anchor_pos = command.ref_motion_root_global_pos_cur  # [B, 3]
    ref_anchor_quat = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4]
    robot_anchor_pos = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4]

    # Body positions (world)
    robot_body_pos_w = command.robot.data.body_pos_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_pos_w = command.ref_motion_bodylink_global_pos_cur[
        :, keybody_idxs
    ]  # [B, N, 3]

    # Expand for broadcasting
    num_bodies = len(keybody_idxs)
    ref_anchor_pos_exp = ref_anchor_pos[:, None, :].expand(-1, num_bodies, -1)
    ref_anchor_quat_exp = ref_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )
    robot_anchor_pos_exp = robot_anchor_pos[:, None, :].expand(
        -1, num_bodies, -1
    )
    robot_anchor_quat_exp = robot_anchor_quat[:, None, :].expand(
        -1, num_bodies, -1
    )

    # Yaw-only delta orientation (anchor frames)
    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_anchor_quat_exp, isaaclab_math.quat_inv(ref_anchor_quat_exp)
        )
    )  # [B, N, 4]

    # Keep origin at root: compare root-relative vectors after yaw alignment
    robot_rel = robot_body_pos_w - robot_anchor_pos_exp  # [B, N, 3]
    ref_rel = ref_body_pos_w - ref_anchor_pos_exp  # [B, N, 3]
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
) -> torch.Tensor:
    """Track root-relative keybody rotations in each entity's root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root orientations
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    ref_root_quat_w = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4]

    # Body orientations (world)
    robot_body_quat_w = command.robot.data.body_quat_w[
        :, keybody_idxs
    ]  # [B, N, 4]
    ref_body_quat_w = command.ref_motion_bodylink_global_rot_wxyz_cur[
        :, keybody_idxs
    ]  # [B, N, 4]

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
) -> torch.Tensor:
    """Track keybody linear velocities with motion_relative frame alignment.

    Compute rigid-body-relative velocities for both entities w.r.t. their
    anchors, yaw-align reference to robot using anchor quats, then compare in
    world space.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Anchor states (robot uses anchor link; reference uses root)
    robot_anchor_pos_w = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_quat_w = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4]
    robot_anchor_lin_vel_w = command.robot.data.body_lin_vel_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_ang_vel_w = command.robot.data.body_ang_vel_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]

    ref_anchor_pos_w = command.ref_motion_root_global_pos_cur  # [B, 3]
    ref_anchor_quat_w = (
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4]
    ref_anchor_lin_vel_w = command.ref_motion_root_global_lin_vel_cur  # [B, 3]
    ref_anchor_ang_vel_w = command.ref_motion_root_global_ang_vel_cur  # [B, 3]

    # Body states (world)
    robot_body_pos_w = command.robot.data.body_pos_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    robot_body_lin_vel_w = command.robot.data.body_lin_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_pos_w = command.ref_motion_bodylink_global_pos_cur[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_lin_vel_w = command.ref_motion_bodylink_global_lin_vel_cur[
        :, keybody_idxs
    ]  # [B, N, 3]

    # Rigid-body relative (world)
    robot_r_w = robot_body_pos_w - robot_anchor_pos_w[:, None, :]
    ref_r_w = ref_body_pos_w - ref_anchor_pos_w[:, None, :]

    robot_cross = torch.cross(
        robot_anchor_ang_vel_w[:, None, :], robot_r_w, dim=-1
    )  # [B, N, 3]
    ref_cross = torch.cross(
        ref_anchor_ang_vel_w[:, None, :], ref_r_w, dim=-1
    )  # [B, N, 3]

    robot_v_rel_w = (
        robot_body_lin_vel_w - robot_anchor_lin_vel_w[:, None, :] - robot_cross
    )  # [B, N, 3]
    ref_v_rel_w = (
        ref_body_lin_vel_w - ref_anchor_lin_vel_w[:, None, :] - ref_cross
    )  # [B, N, 3]
    # Yaw-only delta orientation from anchor quats; rotate reference velocities
    num_bodies = len(keybody_idxs)
    robot_anchor_quat_exp = robot_anchor_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, N, 4]
    ref_anchor_quat_exp = ref_anchor_quat_w[:, None, :].expand(
        -1, num_bodies, -1
    )  # [B, N, 4]
    delta_ori = isaaclab_math.yaw_quat(
        isaaclab_math.quat_mul(
            robot_anchor_quat_exp, isaaclab_math.quat_inv(ref_anchor_quat_exp)
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
        command.ref_motion_root_global_rot_quat_wxyz_cur
    )  # [B, 4]
    ref_root_ang_vel_w = command.ref_motion_root_global_ang_vel_cur  # [B, 3]

    # Body angular velocities (world)
    robot_body_ang_vel_w = command.robot.data.body_ang_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_ang_vel_w = command.ref_motion_bodylink_global_ang_vel_cur[
        :, keybody_idxs
    ]  # [B, N, 3]

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


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet clearing a target height with velocity-shaped kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
        )
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


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

joint_deviation_l1_waists = isaaclab_mdp.joint_deviation_l1

joint_deviation_l1_legs = isaaclab_mdp.joint_deviation_l1


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


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
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return torch.exp(-error / std**2) * (cmd_norm < 0.1)


def stand_still_action_rate(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    stand_still = (
        torch.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    )
    return (
        torch.sum(
            torch.square(
                env.action_manager.action - env.action_manager.prev_action
            ),
            dim=1,
        )
        * stand_still
    )


def orientation_l2(
    env: ManagerBasedRLEnv,
    desired_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(
        asset.data.projected_gravity_b * desired_gravity, dim=-1
    )  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(
        env.command_manager.get_command("base_velocity"), dim=1
    )
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )


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


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
        )
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


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

    command_norm = torch.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[
        :, sensor_cfg.body_ids
    ]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "joint_mirror_joints_cache")
        or env.joint_mirror_joints_cache is None
    ):
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


@configclass
class RewardsCfg:
    pass


def build_rewards_config(reward_config_dict: dict):
    if isinstance(reward_config_dict, (DictConfig, ListConfig)):
        reward_config_dict = OmegaConf.to_container(
            reward_config_dict, resolve=True
        )

    rewards_cfg = RewardsCfg()

    for reward_name, reward_cfg in reward_config_dict.items():
        reward_cfg = resolve_holo_config(reward_cfg)
        params_in = resolve_holo_config(reward_cfg["params"])

        func = None
        method_name = f"{reward_name}"
        func = globals().get(method_name, None)
        if func is None:
            func = getattr(isaaclab_mdp, reward_name, None)
        if func is None:
            raise ValueError(f"Unknown reward function: {reward_name}")
        params = params_in

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
