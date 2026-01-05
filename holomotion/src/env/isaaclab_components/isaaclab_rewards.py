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
    _get_dof_indices,
)


def key_dof_position_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    key_dofs: list[str] | None = None,
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keydof_idxs = _get_dof_indices(command.robot, key_dofs)
    ref_dof_pos = command.get_ref_motion_dof_pos_cur(prefix=ref_prefix)
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keydof_idxs = _get_dof_indices(command.robot, key_dofs)
    ref_dof_vel = command.get_ref_motion_dof_vel_cur(prefix=ref_prefix)
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
    ref_prefix: str = "",
) -> torch.Tensor:
    ref_motion_command: RefMotionCommand = env.command_manager.get_term(
        command_name
    )
    ref_anchor_pos = (
        ref_motion_command.get_ref_motion_anchor_bodylink_global_pos_cur(
            prefix=ref_prefix
        )
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_anchor_quat = (
        command.get_ref_motion_anchor_bodylink_global_rot_wxyz_cur(
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor positions/orientations
    ref_anchor_pos = command.get_ref_motion_root_global_pos_cur(
        prefix=ref_prefix
    )  # [B, 3]
    ref_anchor_quat = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4] (w,x,y,z)
    robot_anchor_pos = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body positions in global frame
    ref_body_pos_global = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor positions/orientations
    ref_root_pos_w = command.get_ref_motion_root_global_pos_cur(
        prefix=ref_prefix
    )  # [B, 3]
    ref_root_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4] (w,x,y,z)
    robot_root_pos_w = command.robot.data.body_pos_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 3]
    robot_root_quat_w = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body positions in global frame
    ref_body_pos_w = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Get reference and robot anchor orientations
    ref_anchor_quat = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4] (w,x,y,z)
    robot_anchor_quat = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4] (w,x,y,z)

    # Get reference body orientations in global frame
    ref_body_quat_global = command.get_ref_motion_bodylink_global_rot_wxyz_cur(
        prefix=ref_prefix
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global velocities (no coordinate transformation needed)
    ref_lin_vel = command.get_ref_motion_bodylink_global_lin_vel_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]
    robot_lin_vel = command.robot.data.body_lin_vel_w[:, keybody_idxs]
    error = torch.sum(torch.square(ref_lin_vel - robot_lin_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    # Get body indexes based on body names (similar to whole_body_tracking implementation)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Direct comparison of global angular velocities (no coordinate transformation needed)
    ref_ang_vel = command.get_ref_motion_bodylink_global_ang_vel_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]
    robot_ang_vel = command.robot.data.body_ang_vel_w[:, keybody_idxs]
    error = torch.sum(torch.square(ref_ang_vel - robot_ang_vel), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def root_pos_xy_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_pos = command.get_ref_motion_root_global_pos_cur(
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
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
    ref_prefix: str = "",
) -> torch.Tensor:
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    robot_root_z = command.robot.data.root_pos_w[:, 2]
    ref_root_z = command.get_ref_motion_root_global_pos_cur(prefix=ref_prefix)[
        :, 2
    ]
    dz_rel = (robot_root_z - ref_root_z) - command.root_z_delta0
    error = torch.square(dz_rel)
    return torch.exp(-error / std**2)


def root_lin_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "",
) -> torch.Tensor:
    """Track root linear velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_lin_vel_w = isaaclab_mdp.root_lin_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_lin_vel_w = command.get_ref_motion_root_global_lin_vel_cur(
        prefix=ref_prefix
    )
    ref_root_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
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


def root_ang_vel_tracking_l2_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "",
) -> torch.Tensor:
    """Track root angular velocity in each entity's own root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)

    # [B, 3], [B, 4]
    robot_root_ang_vel_w = isaaclab_mdp.root_ang_vel_w(env)
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)
    ref_root_ang_vel_w = command.get_ref_motion_root_global_ang_vel_cur(
        prefix=ref_prefix
    )
    ref_root_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
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


def root_rel_keybodylink_pos_tracking_l2_exp_bydmmc_style(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "",
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
    ref_anchor_pos = command.get_ref_motion_root_global_pos_cur(
        prefix=ref_prefix
    )  # [B, 3]
    ref_anchor_quat = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
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
    ref_body_pos_w = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]  # [B, N, 3]

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
    ref_prefix: str = "",
) -> torch.Tensor:
    """Track root-relative keybody rotations in each entity's root frame.

    Returns: [B]
    """
    command: RefMotionCommand = env.command_manager.get_term(command_name)
    keybody_idxs = _get_body_indices(command.robot, keybody_names)

    # Root orientations
    robot_root_quat_w = isaaclab_mdp.root_quat_w(env)  # [B, 4]
    ref_root_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4]

    # Body orientations (world)
    robot_body_quat_w = command.robot.data.body_quat_w[
        :, keybody_idxs
    ]  # [B, N, 4]
    ref_body_quat_w = command.get_ref_motion_bodylink_global_rot_wxyz_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]  # [B, N, 4]

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
    ref_prefix: str = "",
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

    ref_anchor_pos_w = command.get_ref_motion_root_global_pos_cur(
        prefix=ref_prefix
    )  # [B, 3]
    ref_anchor_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4]
    ref_anchor_lin_vel_w = command.get_ref_motion_root_global_lin_vel_cur(
        prefix=ref_prefix
    )  # [B, 3]
    ref_anchor_ang_vel_w = command.get_ref_motion_root_global_ang_vel_cur(
        prefix=ref_prefix
    )  # [B, 3]

    # Body states (world)
    robot_body_pos_w = command.robot.data.body_pos_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    robot_body_lin_vel_w = command.robot.data.body_lin_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_pos_w = command.get_ref_motion_bodylink_global_pos_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]  # [B, N, 3]
    ref_body_lin_vel_w = command.get_ref_motion_bodylink_global_lin_vel_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]  # [B, N, 3]

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
    ref_prefix: str = "",
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
    ref_root_quat_w = command.get_ref_motion_root_global_rot_quat_wxyz_cur(
        prefix=ref_prefix
    )  # [B, 4]
    ref_root_ang_vel_w = command.get_ref_motion_root_global_ang_vel_cur(
        prefix=ref_prefix
    )  # [B, 3]

    # Body angular velocities (world)
    robot_body_ang_vel_w = command.robot.data.body_ang_vel_w[
        :, keybody_idxs
    ]  # [B, N, 3]
    ref_body_ang_vel_w = command.get_ref_motion_bodylink_global_ang_vel_cur(
        prefix=ref_prefix
    )[:, keybody_idxs]  # [B, N, 3]

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


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the L2-squared norm of the action second finite difference.

    Approximates per-dimension action acceleration as
    a''_t ≈ (a_t - 2 * a_{t-1} + a_{t-2}) / step_dt^2 and returns
    sum(a''_t^2) for each env. Shapes:
    - env.action_manager.action: [B, A]
    - return: [B]
    """
    # current and previous actions from IsaacLab action manager
    curr: torch.Tensor = env.action_manager.action  # [B, A]

    # lazily initialize history buffers on env
    if not hasattr(env, "_action_acc_prev_action") or not hasattr(
        env, "_action_acc_prev_prev_action"
    ):
        prev: torch.Tensor = env.action_manager.prev_action  # [B, A]
        env._action_acc_prev_action = prev.clone()
        env._action_acc_prev_prev_action = prev.clone()

    # handle potential shape changes (e.g. different env configs)
    if env._action_acc_prev_action.shape != curr.shape:
        prev = env.action_manager.prev_action  # [B, A]
        env._action_acc_prev_action = prev.clone()
        env._action_acc_prev_prev_action = prev.clone()

    # reset history at the beginning of new episodes so that acceleration
    # does not couple across episode boundaries
    step_count: torch.Tensor = env.episode_length_buf  # [B]
    first_step: torch.Tensor = step_count == 0  # [B]
    if torch.any(first_step):
        env._action_acc_prev_action[first_step] = curr[first_step]
        env._action_acc_prev_prev_action[first_step] = curr[first_step]

    prev1: torch.Tensor = env._action_acc_prev_action  # [B, A], a_{t-1}
    prev2: torch.Tensor = env._action_acc_prev_prev_action  # [B, A], a_{t-2}

    acc = curr - 2.0 * prev1 + prev2  # [B, A]
    reward = torch.sum(acc * acc, dim=1)  # [B]

    # update history for next step
    env._action_acc_prev_prev_action = prev1.clone()
    env._action_acc_prev_action = curr.clone()

    return reward


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
    rollpitch = torch.stack([roll, pitch], dim=-1)

    # Penalize lateral components (x, y) with squared magnitude.
    penalty: torch.Tensor = torch.sum(
        torch.square(rollpitch),
        dim=-1,
    )  # [B]
    return penalty


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
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


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
    ang_vel_error = torch.square(
        command[:, 2] - asset.data.root_ang_vel_w[:, 2]
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
