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



import isaaclab.utils.math as isaaclab_math
import torch


def positions_world_to_env_frame(
    positions_w: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Convert simulator-world positions to IsaacLab environment frame.

    IsaacLab's MDP root position helpers return positions in the environment
    frame, i.e. simulation-world coordinates with per-environment
    `env_origins` subtracted. This helper applies the same
    translation removal to arbitrary position tensors so position arithmetic
    stays frame-consistent.
    """
    if positions_w.ndim < 2 or positions_w.shape[-1] != 3:
        raise ValueError(
            "positions_w must have shape [B, ..., 3], "
            f"got {tuple(positions_w.shape)}."
        )
    if env_origins.ndim != 2 or env_origins.shape[-1] != 3:
        raise ValueError(
            "env_origins must have shape [B, 3], "
            f"got {tuple(env_origins.shape)}."
        )
    if positions_w.shape[0] != env_origins.shape[0]:
        raise ValueError(
            "Batch size mismatch between positions_w and env_origins: "
            f"{positions_w.shape[0]} vs {env_origins.shape[0]}."
        )
    origin_view = env_origins.view(
        env_origins.shape[0],
        *([1] * (positions_w.ndim - 2)),
        3,
    )
    return positions_w - origin_view


def root_relative_positions_from_env_frame(
    body_pos_env: torch.Tensor,
    root_pos_env: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Convert environment-frame body positions into the root frame.

    The input positions must already be in IsaacLab's environment frame rather
    than raw simulator-world coordinates. Orientation is unaffected by
    `env_origins`, so the articulation root quaternion is reused directly.
    """
    if body_pos_env.ndim < 3 or body_pos_env.shape[-1] != 3:
        raise ValueError(
            "body_pos_env must have shape [B, ..., 3], "
            f"got {tuple(body_pos_env.shape)}."
        )
    if root_pos_env.ndim != 2 or root_pos_env.shape[-1] != 3:
        raise ValueError(
            "root_pos_env must have shape [B, 3], "
            f"got {tuple(root_pos_env.shape)}."
        )
    if root_quat_w.ndim != 2 or root_quat_w.shape[-1] != 4:
        raise ValueError(
            "root_quat_w must have shape [B, 4], "
            f"got {tuple(root_quat_w.shape)}."
        )
    if body_pos_env.shape[0] != root_pos_env.shape[0]:
        raise ValueError(
            "Batch size mismatch between body_pos_env and root_pos_env: "
            f"{body_pos_env.shape[0]} vs {root_pos_env.shape[0]}."
        )
    if body_pos_env.shape[0] != root_quat_w.shape[0]:
        raise ValueError(
            "Batch size mismatch between body_pos_env and root_quat_w: "
            f"{body_pos_env.shape[0]} vs {root_quat_w.shape[0]}."
        )
    root_pos_view = root_pos_env.view(
        root_pos_env.shape[0],
        *([1] * (body_pos_env.ndim - 2)),
        3,
    )
    root_quat_view = root_quat_w.view(
        root_quat_w.shape[0],
        *([1] * (body_pos_env.ndim - 2)),
        4,
    ).expand(*body_pos_env.shape[:-1], 4)
    rel_pos_env = body_pos_env - root_pos_view
    return isaaclab_math.quat_apply_inverse(root_quat_view, rel_pos_env)


def root_relative_positions_from_mixed_position_frames(
    body_pos_w: torch.Tensor,
    root_pos_env: torch.Tensor,
    root_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Build root-relative positions from world-frame bodies.

    This is the safe adapter for common IsaacLab code paths where body poses
    are read from `robot.data.body_pos_w` in simulator world coordinates while
    `isaaclab_mdp.root_pos_w(env)` is already expressed in the environment
    frame.
    """
    body_pos_env = positions_world_to_env_frame(body_pos_w, env_origins)
    return root_relative_positions_from_env_frame(
        body_pos_env=body_pos_env,
        root_pos_env=root_pos_env,
        root_quat_w=root_quat_w,
    )
