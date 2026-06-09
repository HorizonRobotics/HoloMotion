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


import inspect

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.utils.math as isaaclab_math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from holomotion.src.env.isaaclab_components import (
    isaaclab_motion_tracking_command as motion_tracking_command,
    isaaclab_utils,
)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a WXYZ quaternion tensor."""
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _list_supported_terminations() -> list[str]:
    custom_terminations = {
        name
        for name, obj in globals().items()
        if (
            inspect.isfunction(obj)
            and obj.__module__ == __name__
            and not name.startswith("_")
        )
    }
    native_terminations = {
        name
        for name in dir(isaaclab_mdp.terminations)
        if (
            not name.startswith("_")
            and callable(getattr(isaaclab_mdp.terminations, name))
        )
    }
    return sorted(custom_terminations | native_terminations)


def _resolve_termination_func(name: str):
    func = globals().get(name)
    if inspect.isfunction(func) and func.__module__ == __name__:
        return func

    func = getattr(isaaclab_mdp.terminations, name, None)
    if callable(func):
        return func

    supported = _list_supported_terminations()
    raise ValueError(
        f"Unknown termination function: {name}. Supported: {supported}"
    )


def global_bodylink_pos_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Any body link position deviates more than threshold (world frame)."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )  # [B, Nb, 3]
    robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

    keybody_idxs = isaaclab_utils._get_body_indices(
        command.robot, keybody_names
    )

    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_pos_w = ref_pos_w[:, idxs]
        robot_pos_w = robot_pos_w[:, idxs]

    error = torch.norm(ref_pos_w - robot_pos_w, dim=-1)  # [B, Nb]
    return torch.any(error > threshold, dim=-1)  # [B]


def anchor_ref_z_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Anchor link z difference exceeds threshold (world frame)."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_z = command.get_ref_motion_anchor_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )[:, -1]
    robot_z = command.global_robot_anchor_pos_cur[:, -1]
    return (ref_z - robot_z).abs() > threshold


def ref_gravity_projection_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_name: str = "robot",
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Difference in projected gravity z-component exceeds threshold.

    Project world gravity into the anchor body frames using inverse
    quaternion rotation and compare z-components.
    """
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    g_w = env.scene[asset_name].data.GRAVITY_VEC_W  # [B, 3]

    # Reference anchor orientation (xyzw) from motion cache
    ref_anchor_quat_xyzw = (
        command.get_ref_motion_anchor_bodylink_global_rot_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )  # [B, 4]

    motion_projected_gravity_b = isaaclab_math.quat_apply_inverse(
        ref_anchor_quat_xyzw, g_w
    )  # [B, 3]

    # motion_projected_gravity_b = isaaclab_math.quat_rotate_inverse(
    #     ref_anchor_quat_xyzw, g_w
    # )  # [B, 3]

    # Robot anchor orientation (xyzw) from sim
    robot_anchor_quat_wxyz = command.robot.data.body_quat_w[
        :, command.anchor_bodylink_idx
    ]  # [B, 4]

    robot_projected_gravity_b = isaaclab_math.quat_apply_inverse(
        robot_anchor_quat_wxyz, g_w
    )  # [B, 3]

    # robot_projected_gravity_b = isaaclab_math.quat_rotate_inverse(
    #     robot_anchor_quat_wxyz, g_w
    # )  # [B, 3]

    return (
        motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def root_yaw_ref_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Terminate when root yaw deviates from the reference beyond threshold."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_root_quat = (
        command.get_ref_motion_root_global_rot_quat_wxyz_immediate_next(
            prefix=ref_prefix
        )
    )
    robot_root_quat = command.robot.data.root_quat_w
    yaw_error = _wrap_to_pi(
        _yaw_from_quat_wxyz(ref_root_quat)
        - _yaw_from_quat_wxyz(robot_root_quat)
    )
    return yaw_error.abs() > threshold


def keybody_ref_pos_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Any key body link z difference exceeds threshold (world frame)."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )  # [B, Nb, 3]
    robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

    keybody_idxs = isaaclab_utils._get_body_indices(
        command.robot, keybody_names
    )

    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_pos_w = ref_pos_w[:, idxs]
        robot_pos_w = robot_pos_w[:, idxs]

    error = torch.norm(ref_pos_w - robot_pos_w, dim=-1)  # [B, Nb]
    return torch.any(error > threshold, dim=-1)  # [B]


def keybody_ref_z_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Any key body link z difference exceeds threshold (world frame)."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )  # [B, Nb, 3]
    robot_pos_w = command.robot.data.body_pos_w  # [B, Nb, 3]

    keybody_idxs = isaaclab_utils._get_body_indices(
        command.robot, keybody_names
    )

    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_pos_w = ref_pos_w[:, idxs]
        robot_pos_w = robot_pos_w[:, idxs]

    error_z = (ref_pos_w[..., 2] - robot_pos_w[..., 2]).abs()  # [B, Nb]
    return torch.any(error_z > threshold, dim=-1)  # [B]


def root_delta_z_ref_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    grace_steps: int = 10,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Terminate on root z displacement mismatch relative to a grace-step baseline."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    robot_z = command.robot.data.root_pos_w[:, 2]
    ref_z = command.get_ref_motion_root_global_pos_immediate_next(
        prefix=ref_prefix
    )[:, 2]

    num_envs = int(robot_z.shape[0])
    device = robot_z.device
    state = getattr(command, "_root_delta_z_ref_far_state", None)
    if state is None or int(state["valid"].shape[0]) != num_envs:
        state = {
            "valid": torch.zeros(num_envs, device=device, dtype=torch.bool),
            "clip": torch.full(
                (num_envs,), -1, device=device, dtype=torch.long
            ),
            "start": torch.full(
                (num_envs,), -1, device=device, dtype=torch.long
            ),
            "robot_base_z": torch.zeros(
                num_envs, device=device, dtype=robot_z.dtype
            ),
            "ref_base_z": torch.zeros(
                num_envs, device=device, dtype=ref_z.dtype
            ),
        }
        setattr(command, "_root_delta_z_ref_far_state", state)

    clip_indices = getattr(command, "_clip_indices")
    frame_indices = getattr(command, "_frame_indices")
    start_frame_indices = getattr(command, "_start_frame_indices")
    segment_steps = (frame_indices - start_frame_indices).clamp(min=0)
    within_grace = segment_steps < max(0, int(grace_steps))

    valid = state["valid"]
    valid[within_grace] = False
    segment_changed = (
        valid
        & (
            (state["clip"] != clip_indices)
            | (state["start"] != start_frame_indices)
        )
    )
    need_baseline = (~valid | segment_changed) & ~within_grace
    fresh_baseline = torch.zeros(num_envs, device=device, dtype=torch.bool)
    if torch.any(need_baseline):
        state["robot_base_z"][need_baseline] = robot_z[need_baseline]
        state["ref_base_z"][need_baseline] = ref_z[need_baseline]
        state["clip"][need_baseline] = clip_indices[need_baseline]
        state["start"][need_baseline] = start_frame_indices[need_baseline]
        valid[need_baseline] = True
        fresh_baseline[need_baseline] = True

    robot_delta_z = robot_z - state["robot_base_z"]
    ref_delta_z = ref_z - state["ref_base_z"]
    error = (robot_delta_z - ref_delta_z).abs()
    check_mask = valid & ~within_grace & ~fresh_baseline
    return check_mask & (error > threshold)


def keybody_delta_z_ref_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    grace_steps: int = 10,
    command_name: str = "ref_motion",
    keybody_names: list[str] | None = None,
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Terminate on keybody z displacement mismatch relative to a grace-step baseline."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_pos_w = command.get_ref_motion_bodylink_global_pos_immediate_next(
        prefix=ref_prefix
    )
    robot_pos_w = command.robot.data.body_pos_w

    keybody_idxs = isaaclab_utils._get_body_indices(
        command.robot, keybody_names
    )
    if keybody_idxs is not None and len(keybody_idxs) > 0:
        idxs = torch.as_tensor(
            keybody_idxs,
            device=ref_pos_w.device,
            dtype=torch.long,
        )
        ref_z = ref_pos_w[:, idxs, 2]
        robot_z = robot_pos_w[:, idxs, 2]
    else:
        ref_z = ref_pos_w[..., 2]
        robot_z = robot_pos_w[..., 2]

    num_envs, num_bodies = int(robot_z.shape[0]), int(robot_z.shape[1])
    device = robot_z.device
    state = getattr(command, "_keybody_delta_z_ref_far_state", None)
    if (
        state is None
        or int(state["valid"].shape[0]) != num_envs
        or int(state["robot_base_z"].shape[1]) != num_bodies
    ):
        state = {
            "valid": torch.zeros(num_envs, device=device, dtype=torch.bool),
            "clip": torch.full(
                (num_envs,), -1, device=device, dtype=torch.long
            ),
            "start": torch.full(
                (num_envs,), -1, device=device, dtype=torch.long
            ),
            "robot_base_z": torch.zeros_like(robot_z),
            "ref_base_z": torch.zeros_like(ref_z),
        }
        setattr(command, "_keybody_delta_z_ref_far_state", state)

    clip_indices = getattr(command, "_clip_indices")
    frame_indices = getattr(command, "_frame_indices")
    start_frame_indices = getattr(command, "_start_frame_indices")
    segment_steps = (frame_indices - start_frame_indices).clamp(min=0)
    within_grace = segment_steps < max(0, int(grace_steps))

    valid = state["valid"]
    valid[within_grace] = False
    segment_changed = (
        valid
        & (
            (state["clip"] != clip_indices)
            | (state["start"] != start_frame_indices)
        )
    )
    need_baseline = (~valid | segment_changed) & ~within_grace
    fresh_baseline = torch.zeros(num_envs, device=device, dtype=torch.bool)
    if torch.any(need_baseline):
        state["robot_base_z"][need_baseline] = robot_z[need_baseline]
        state["ref_base_z"][need_baseline] = ref_z[need_baseline]
        state["clip"][need_baseline] = clip_indices[need_baseline]
        state["start"][need_baseline] = start_frame_indices[need_baseline]
        valid[need_baseline] = True
        fresh_baseline[need_baseline] = True

    robot_delta_z = robot_z - state["robot_base_z"]
    ref_delta_z = ref_z - state["ref_base_z"]
    error = (robot_delta_z - ref_delta_z).abs()
    check_mask = valid & ~within_grace & ~fresh_baseline
    return check_mask & torch.any(error > threshold, dim=-1)


def wholebody_mpjpe_far(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "ref_motion",
    ref_prefix: str = "ref_",
) -> torch.Tensor:
    """Mean whole-body DOF position error exceeds threshold."""
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    ref_dof_pos = command.get_ref_motion_dof_pos_immediate_next(
        prefix=ref_prefix
    )
    robot_dof_pos = command.robot.data.joint_pos
    mean_dof_error = torch.mean(torch.abs(robot_dof_pos - ref_dof_pos), dim=-1)
    return mean_dof_error > threshold


def motion_end(
    env: ManagerBasedRLEnv,
    command_name: str = "ref_motion",
) -> torch.Tensor:
    """Terminate when reference motion frames exceed their end frames.

    Returns a boolean mask of shape [num_envs].
    """
    command: motion_tracking_command.RefMotionCommand = (
        env.command_manager.get_term(command_name)
    )
    result = command.motion_end_mask.clone().bool()
    return result


@configclass
class TerminationsCfg:
    pass


def build_terminations_config(
    termination_config_dict: dict,
) -> TerminationsCfg:
    terminations_cfg = TerminationsCfg()

    for termination_name, termination_cfg in termination_config_dict.items():
        termination_cfg = isaaclab_utils.resolve_holo_config(termination_cfg)
        func = _resolve_termination_func(termination_name)
        params = isaaclab_utils.resolve_holo_config(
            termination_cfg.get("params", {})
        )

        term_cfg = TerminationTermCfg(
            func=func,
            params=params,
            time_out=(termination_name == "time_out")
            or termination_cfg.get("time_out", False),
        )
        setattr(terminations_cfg, termination_name, term_cfg)

    return terminations_cfg
