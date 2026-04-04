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


from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)

from typing import Sequence


class HoloMotionUniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: HoloMotionUniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(
        self, cfg: HoloMotionUniformVelocityCommandCfg, env: ManagerBasedEnv
    ):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.rel_yaw_envs > 0.0:
            yaw_min, yaw_max = self.cfg.ranges.ang_vel_z

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_yaw_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_vel_yaw"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "HoloMotionUniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tYaw-only probability: {self.cfg.rel_yaw_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2]
                - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(
                self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
            )
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_yaw_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_yaw_envs
        )
        if self.cfg.heading_command:
            # yaw-only envs should follow directly sampled yaw commands (not heading control)
            self.is_heading_env[env_ids] &= ~self.is_yaw_env[env_ids]
        # update standing envs
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids]
                - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        yaw_env_ids = self.is_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[yaw_env_ids, :2] = 0.0

        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(
            as_tuple=False
        ).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = (
            self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(
            zeros, zeros, heading_angle
        )
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class HoloMotionUniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = HoloMotionUniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_yaw_envs: float = 0.0
    """The sampled probability of environments that should receive yaw-only commands. Defaults to 0.0.

    For yaw-only environments, the command is post-processed to:
    - enforce vx=vy=0

    This is sampled independently from :attr:`rel_standing_envs`. If an environment is both yaw-only
    and standing, standing still overrides to zero command.
    """

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~HoloMotionUniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_goal"
        )
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_current"
        )
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class VelTrack_CommandsCfg:
    pass


def _convert_ranges_dict_to_object(
    ranges_dict: dict,
) -> HoloMotionUniformVelocityCommandCfg.Ranges:
    """Convert a dict of ranges to a proper Ranges object with tuples."""
    ranges_kwargs = {}
    for key, value in ranges_dict.items():
        if value is None:
            ranges_kwargs[key] = None
        elif isinstance(value, (list, tuple)):
            ranges_kwargs[key] = tuple(value)
        else:
            ranges_kwargs[key] = value
    return HoloMotionUniformVelocityCommandCfg.Ranges(**ranges_kwargs)


def build_velocity_commands_config(
    command_config_dict: dict,
) -> VelTrack_CommandsCfg:
    """Build a CommandsCfg that supports velocity commands via IsaacLab isaaclab_mdp.

    Expected format:
    {
      "base_velocity": {
        "type": "VelocityCommandCfg" | "HoloMotionUniformVelocityCommandCfg" | "UniformLevelVelocityCommandCfg",
        "params": { ... }  # args compatible with mdp command cfgs
      }
    }

    For ranges and limit_ranges, pass them as dicts with keys like lin_vel_x, lin_vel_y, ang_vel_z, heading.
    """
    commands_cfg = VelTrack_CommandsCfg()

    for name, cfg in command_config_dict.items():
        command_type = cfg.get("type", "VelocityCommandCfg")
        params = cfg.get("params", {}).copy()

        if "ranges" in params and isinstance(params["ranges"], dict):
            params["ranges"] = _convert_ranges_dict_to_object(params["ranges"])

        if "limit_ranges" in params and isinstance(
            params["limit_ranges"], dict
        ):
            params["limit_ranges"] = _convert_ranges_dict_to_object(
                params["limit_ranges"]
            )

        if command_type == "HoloMotionUniformVelocityCommandCfg":
            term_cfg = HoloMotionUniformVelocityCommandCfg(**params)
        else:
            raise ValueError(f"Unknown velocity command type: {command_type}")

        setattr(commands_cfg, name, term_cfg)

    return commands_cfg
