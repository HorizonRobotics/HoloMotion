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

"""World-frame tracking rewards using clean immediate-next motion references.

The reference getters already add environment origins, reorder URDF bodies to
simulator order, and convert quaternions from XYZW to WXYZ. These rewards do
not read actor observations or their optional feature noise.
"""

from __future__ import annotations
import math
from collections.abc import Sequence

import torch


def _check_std(std: float) -> float:
    value = float(std)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("std must be finite and positive")
    return value


def height_tracking_exp_from_positions(
    reference_pos_w: torch.Tensor,
    robot_pos_w: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Return exp(-squared Z error / std squared) for matching [B, 3]."""
    std = _check_std(std)
    if (
        reference_pos_w.shape != robot_pos_w.shape
        or reference_pos_w.ndim != 2
        or reference_pos_w.shape[-1] != 3
    ):
        raise ValueError(
            "Reference and robot positions must have matching [B,3] shapes"
        )
    error_squared = (reference_pos_w[:, 2] - robot_pos_w[:, 2]).square()
    return torch.exp(-error_squared / std**2)


def quaternion_angle_error_wxyz(
    reference_quat_wxyz: torch.Tensor,
    robot_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return the shortest full SO(3) angle, invariant to quaternion sign.

    Both inputs must have matching [..., 4] shapes. Invalid zero or nonfinite
    quaternions yield nonfinite angles instead of an undeserved high reward.
    """
    if (
        reference_quat_wxyz.shape != robot_quat_wxyz.shape
        or reference_quat_wxyz.shape[-1] != 4
    ):
        raise ValueError(
            "Reference and robot quaternions must have matching [...,4] shapes"
        )
    reference = reference_quat_wxyz / torch.linalg.vector_norm(
        reference_quat_wxyz, dim=-1, keepdim=True
    )
    robot = robot_quat_wxyz / torch.linalg.vector_norm(
        robot_quat_wxyz, dim=-1, keepdim=True
    )
    # Relative quaternion is reference * conjugate(robot), in WXYZ order.
    scalar = (reference * robot).sum(dim=-1)
    vector = (
        robot[..., :1] * reference[..., 1:]
        - reference[..., :1] * robot[..., 1:]
    )
    vector = vector - torch.linalg.cross(
        reference[..., 1:], robot[..., 1:], dim=-1
    )
    return 2.0 * torch.atan2(
        torch.linalg.vector_norm(vector, dim=-1), scalar.abs()
    )


def body_rotation_tracking_exp_from_quaternions(
    reference_quat_wxyz: torch.Tensor,
    robot_quat_wxyz: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Exponentiate mean squared full-body angular error, yielding [B]."""
    std = _check_std(std)
    if reference_quat_wxyz.ndim != 3 or reference_quat_wxyz.shape[1] == 0:
        raise ValueError("Body quaternions must have nonempty [B,L,4] shape")
    angles = quaternion_angle_error_wxyz(reference_quat_wxyz, robot_quat_wxyz)
    return torch.exp(-angles.square().mean(dim=-1) / std**2)


def _body_indices(command, body_names: Sequence[str]) -> list[int]:
    if isinstance(body_names, str):
        raise ValueError(
            "body_names must be a sequence of names, not one string"
        )
    requested = tuple(body_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("body_names must be nonempty and unique")
    simulator_names = tuple(command.robot.body_names)
    reference_names = tuple(command.urdf_body_names)
    missing = [
        name
        for name in requested
        if name not in simulator_names or name not in reference_names
    ]
    if missing:
        raise ValueError(f"Body names missing from robot/reference: {missing}")
    return [simulator_names.index(name) for name in requested]


def anchor_height_tracking_exp(
    env,
    std: float,
    ref_motion_command_name: str = "ref_motion",
    anchor_body_name: str = "pelvis",
) -> torch.Tensor:
    """Track one named body's world height against the clean reference."""
    command = env.command_manager.get_term(ref_motion_command_name)
    index = _body_indices(command, [anchor_body_name])[0]
    reference_w = command.get_ref_motion_bodylink_global_pos_immediate_next(
        prefix="ref_"
    )[:, index]
    actual_w = command.robot.data.body_pos_w[:, index]
    return height_tracking_exp_from_positions(reference_w, actual_w, std)


def global_body_rotation_tracking_exp(
    env,
    std: float,
    body_names: Sequence[str],
    ref_motion_command_name: str = "ref_motion",
) -> torch.Tensor:
    """Track selected bodies' world SO(3) rotations, including heading."""
    command = env.command_manager.get_term(ref_motion_command_name)
    indices = _body_indices(command, body_names)
    reference_wxyz = (
        command.get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            prefix="ref_"
        )[:, indices]
    )
    actual_wxyz = command.robot.data.body_quat_w[:, indices]
    return body_rotation_tracking_exp_from_quaternions(
        reference_wxyz, actual_wxyz, std
    )
