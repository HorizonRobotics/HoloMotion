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



# This file is modified from the unitree_rl_lab repository:
# https://github.com/unitreerobotics/unitree_rl_lab

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Sequence

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

class UnitreeActuator(DelayedPDActuator):
    """Unitree actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    - X2: No-Load Speed Test

    - Fs: Static friction coefficient
    - Fd: Dynamic friction coefficient
    - Va: Velocity at which the friction is fully activated
    """

    cfg: UnitreeActuatorCfg

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints).
        armature = J2 + J1 * i2 ^ 2 + Jr * (i1 * i2) ^ 2
    """

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        self.applied_effort -= (
            self._friction_static
            * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(
            same_direction, self._effort_y1, self._effort_y2
        )
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


class UnitreeErfiActuator(UnitreeActuator):
    """Unitree actuator with per-env ERFI-50 torque perturbations.

    On environment reset, each env is assigned either step-wise random force
    injection (RFI) or episode-level random actuation offset (RAO). During
    rollout, only the selected mode is applied for that env.
    """

    cfg: UnitreeErfiActuatorCfg

    def __init__(self, cfg: UnitreeErfiActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._mode_is_rfi = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._rfi_lim_scale = torch.ones_like(self.computed_effort)
        self._rao_scale = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int] | slice | None):
        super().reset(env_ids)
        env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return
        if not self.cfg.erfi_enabled:
            self._mode_is_rfi[env_ids_tensor] = False
            self._rfi_lim_scale[env_ids_tensor] = 1.0
            self._rao_scale[env_ids_tensor] = 0.0
            return

        sampled_is_rfi = (
            torch.rand(env_ids_tensor.numel(), device=self._device)
            < self.cfg.rfi_probability
        )
        self._mode_is_rfi[env_ids_tensor] = sampled_is_rfi

        if self.cfg.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids_tensor] = self._sample_uniform(
                self.cfg.rfi_lim_range[0],
                self.cfg.rfi_lim_range[1],
                (env_ids_tensor.numel(), self.num_joints),
            )
        else:
            self._rfi_lim_scale[env_ids_tensor] = 1.0

        self._rao_scale[env_ids_tensor] = self._sample_uniform(
            -self.cfg.rao_lim,
            self.cfg.rao_lim,
            (env_ids_tensor.numel(), self.num_joints),
        )

        rfi_env_ids = env_ids_tensor[sampled_is_rfi]
        if rfi_env_ids.numel() > 0:
            self._rao_scale[rfi_env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if not self.cfg.erfi_enabled:
            return super().compute(control_action, joint_pos, joint_vel)

        if control_action.joint_efforts is None:
            base_joint_efforts = torch.zeros_like(joint_pos)
        else:
            base_joint_efforts = control_action.joint_efforts.clone()

        effort_limit = self.effort_limit.to(base_joint_efforts)
        rfi_noise = self._sample_uniform(-1.0, 1.0, base_joint_efforts.shape)
        rfi_term = (
            rfi_noise * self.cfg.rfi_lim * self._rfi_lim_scale * effort_limit
        )
        rao_term = self._rao_scale * effort_limit
        mode_is_rfi = self._mode_is_rfi.unsqueeze(-1)
        control_action_with_erfi = ArticulationActions(
            joint_positions=control_action.joint_positions,
            joint_velocities=control_action.joint_velocities,
            joint_efforts=base_joint_efforts
            + torch.where(mode_is_rfi, rfi_term, rao_term),
            joint_indices=control_action.joint_indices,
        )

        return super().compute(control_action_with_erfi, joint_pos, joint_vel)

    def _env_ids_to_tensor(
        self, env_ids: Sequence[int] | slice | None
    ) -> torch.Tensor:
        if env_ids is None or env_ids == slice(None):
            return torch.arange(self._num_envs, device=self._device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long).flatten()
        return torch.tensor(env_ids, device=self._device, dtype=torch.long)

    def _sample_uniform(
        self, low: float, high: float, shape: tuple[int, ...]
    ) -> torch.Tensor:
        return torch.empty(shape, device=self._device).uniform_(low, high)


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """
    Configuration for Unitree actuators.
    """

    class_type: type = UnitreeActuator

    X1: float = 1e9
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = 1e9
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float | None = None
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""

    Fs: float = 0.0
    """ Static friction coefficient """

    Fd: float = 0.0
    """ Dynamic friction coefficient """

    Va: float = 0.01
    """ Velocity at which the friction is fully activated """


@configclass
class UnitreeErfiActuatorCfg(UnitreeActuatorCfg):
    """Configuration for Unitree actuators with ERFI-50 perturbations."""

    class_type: type = UnitreeErfiActuator

    erfi_enabled: bool = False
    """Whether ERFI perturbations are enabled for this actuator."""

    rfi_probability: float = 0.5
    """Probability of assigning RFI to an environment on reset."""

    rfi_lim: float = 0.1
    """Base RFI limit, expressed as a ratio of joint effort limits."""

    randomize_rfi_lim: bool = True
    """Whether to randomize the per-episode RFI limit scale."""

    rfi_lim_range: tuple[float, float] = (0.5, 1.5)
    """Multiplicative range for per-episode RFI scaling."""

    rao_lim: float = 0.1
    """RAO limit, expressed as a ratio of joint effort limits."""


@configclass
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    X1 = 14.0
    X2 = 25.6
    Y1 = 150.0
    Y2 = 182.8

    armature = 0.063259741


@configclass
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    X1 = 8.8
    X2 = 16
    Y1 = 240
    Y2 = 292.5

    armature = 0.160478022


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    X1 = 13.5
    X2 = 30
    Y1 = 20.2
    Y2 = 23.4


@configclass
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1 = 22.63
    X2 = 35.52
    Y1 = 71
    Y2 = 83.3

    Fs = 1.6
    Fd = 0.16

    """
    | rotor  | 0.489e-4 kg·m²
    | gear_1 | 0.098e-4 kg·m² | ratio | 4.5
    | gear_2 | 0.533e-4 kg·m² | ratio | 48/22+1
    """
    armature = 0.01017752


@configclass
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1 = 14.5
    X2 = 22.7
    Y1 = 111.0
    Y2 = 131.0

    Fs = 2.4
    Fd = 0.24

    """
    | rotor  | 0.489e-4 kg·m²
    | gear_1 | 0.109e-4 kg·m² | ratio | 4.5
    | gear_2 | 0.738e-4 kg·m² | ratio | 5.0
    """
    armature = 0.025101925


@configclass
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    X1 = 27.0
    X2 = 41.5
    Y1 = 9.5
    Y2 = 17.0

    """
    | rotor  | 0.084e-4 kg·m²
    | gear_1 | 0.015e-4 kg·m² | ratio | 4
    | gear_2 | 0.068e-4 kg·m² | ratio | 4
    """
    armature = 0.0021812


@configclass
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    X1 = 30.86
    X2 = 40.13
    Y1 = 24.8
    Y2 = 31.9

    Fs = 0.6
    Fd = 0.06

    """
    | rotor  | 0.139e-4 kg·m²
    | gear_1 | 0.017e-4 kg·m² | ratio | 46/18+1
    | gear_2 | 0.169e-4 kg·m² | ratio | 56/16+1
    """
    armature = 0.003609725


@configclass
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    X1 = 15.3
    X2 = 24.76
    Y1 = 4.8
    Y2 = 8.6

    Fs = 0.6
    Fd = 0.06

    """
    | rotor  | 0.068e-4 kg·m²
    | gear_1 |                | ratio | 5
    | gear_2 |                | ratio | 5
    """
    armature = 0.00425
