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
import time
import os
import yaml
from collections import deque
from functools import wraps
from easydict import EasyDict
import random
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_yaml
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.env.isaaclab_components import (
    ActionsCfg,
    VelTrack_CommandsCfg,
    MoTrack_CommandsCfg,
    EventsCfg,
    MotionTrackingSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
    build_actions_config,
    build_motion_tracking_commands_config,
    build_velocity_commands_config,
    build_domain_rand_config,
    build_curriculum_config,
    build_observations_config,
    build_rewards_config,
    build_scene_config,
    build_terminations_config,
)
from holomotion.src.env.isaaclab_components.isaaclab_observation import (
    ObservationFunctions,
)
from holomotion.src.env.isaaclab_components.isaaclab_utils import (
    resolve_holo_config,
)

# from holomotion.src.modules.agent_modules import ObsSeqSerializer
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.utils import configclass


from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers import EventTermCfg as EventTerm


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING, Literal


def _joint_ids_to_tensor(
    joint_ids: slice | list[int] | tuple[int, ...] | torch.Tensor | None,
    num_joints: int,
    device: torch.device | str,
) -> torch.Tensor:
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


class MotionTrackingEnv:
    """IsaacLab-based Motion Tracking Environment.

    This environment integrates motion tracking capabilities with IsaacLab's
    manager-based architecture, supporting curriculum learning, domain randomization,
    and various termination conditions.

    This is a wrapper class that handles Isaac Sim initialization and delegates
    to an internal ManagerBasedRLEnv instance.
    """

    def __init__(
        self,
        config,
        device: torch.device = None,
        log_dir: str = None,
        render_mode: str | None = None,
        headless: bool = True,
        accelerator=None,
    ):
        """Initialize the Motion Tracking Environment.

        Args:
            config: Configuration for the environment
            device: Device for tensor operations
            log_dir: Logging directory
            render_mode: Render mode for the environment
            headless: Whether to run in headless mode
            accelerator: Accelerator instance for distributed training (optional)
        """
        self.config = config
        self._device = device
        self.accelerator = accelerator

        self.log_dir = log_dir
        self.headless = headless
        self.init_done = False
        self.is_evaluating = False
        self.render_mode = render_mode

        # self._init_motion_tracking_components()
        self._init_isaaclab_env()
        # self._init_serializers()
        self._completion_total_queue = deque(maxlen=1000)
        self._completion_success_queue = deque(maxlen=1000)
        self.metrics = {}
        self._robot_prev_joint_vel = None
        self._robot_prev_applied_torque = None
        self._robot_torque_rate_inv_effort_limit = None
        self._robot_torque_rate_needs_reseed = None

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _init_isaaclab_env(self):
        _device = self._device

        curriculum = CurriculumCfg()

        # Determine per-process seed if provided; else create a deterministic per-rank default
        seed_val = getattr(self.config, "seed", None)
        if seed_val is None:
            if self.accelerator is not None:
                pid = self.accelerator.process_index
            else:
                pid = int(self.config.get("process_id", 0))
            seed_val = int(time.time()) + pid

        _robot_config_dict = EasyDict(
            OmegaConf.to_container(self.config.robot, resolve=True)
        )
        _terrain_config_dict = EasyDict(
            OmegaConf.to_container(self.config.terrain, resolve=True)
        )
        _obs_config_dict = EasyDict(
            OmegaConf.to_container(self.config.obs, resolve=True)
        )
        _rewards_config_dict = EasyDict(
            OmegaConf.to_container(self.config.rewards, resolve=True)
        )
        _domain_rand_config_dict = (
            EasyDict(
                OmegaConf.to_container(
                    self.config.domain_rand,
                    resolve=True,
                )
            )
            if self.config.domain_rand is not None
            else {}
        )
        _terminations_config_dict = (
            EasyDict(
                OmegaConf.to_container(
                    self.config.terminations,
                    resolve=True,
                )
            )
            if self.config.terminations is not None
            else {}
        )
        _scene_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.scene,
                resolve=True,
            )
        )
        _commands_config_dict = OmegaConf.to_container(
            self.config.commands,
            resolve=True,
        )

        _simulation_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.simulation,
                resolve=True,
            )
        )
        _actions_config_dict = EasyDict(
            OmegaConf.to_container(
                self.config.actions,
                resolve=True,
            )
        )
        if getattr(self.config, "curriculum", None) is not None:
            _curriculum_config_dict = EasyDict(
                OmegaConf.to_container(self.config.curriculum, resolve=True)
            )
        else:
            _curriculum_config_dict = {}

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            seed: int = seed_val
            scene_config_dict = {
                "num_envs": self.config.num_envs,
                "env_spacing": self.config.env_spacing,
                "replicate_physics": self.config.replicate_physics,
                "robot": _robot_config_dict,
                "terrain": _terrain_config_dict,
                "domain_rand": _domain_rand_config_dict,
                "lighting": _scene_config_dict.lighting,
                "contact_sensor": _scene_config_dict.contact_sensor,
            }

            decimation: int = _simulation_config_dict.control_decimation
            episode_length_s: int = _simulation_config_dict.episode_length_s
            sim_freq = _simulation_config_dict.sim_freq
            dt = 1.0 / sim_freq
            physx = PhysxCfg(
                bounce_threshold_velocity=_simulation_config_dict.physx.bounce_threshold_velocity,
                gpu_max_rigid_patch_count=_simulation_config_dict.physx.gpu_max_rigid_patch_count,
                enable_stabilization=True,
            )

            if self.accelerator is not None:
                main_process = self.accelerator.is_main_process
                process_id = self.accelerator.process_index
                num_processes = self.accelerator.num_processes
            else:
                main_process = self.config.get("main_process", True)
                process_id = self.config.get("process_id", 0)
                num_processes = self.config.get("num_processes", 1)
            scene: MotionTrackingSceneCfg = build_scene_config(
                scene_config_dict,
                main_process=main_process,
                process_id=process_id,
                num_processes=num_processes,
            )

            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
                enable_scene_query_support=True,
            )
            sim.physics_material = scene.terrain.physics_material

            viewer: ViewerCfg = ViewerCfg(origin_type="world")

            motion_cmds = {}
            vel_cmds = {}
            for k, v in _commands_config_dict.items():
                if (
                    isinstance(v, dict)
                    and v.get("type", "") == "MotionCommandCfg"
                ):
                    motion_cmds[k] = v
                else:
                    vel_cmds[k] = v

            # Populate RefMotionCommand distributed params when present.
            if "ref_motion" in motion_cmds:
                if self.accelerator is not None:
                    cmd_process_id = self.accelerator.process_index
                    cmd_num_processes = self.accelerator.num_processes
                else:
                    cmd_process_id = getattr(self.config, "process_id", 0)
                    cmd_num_processes = getattr(
                        self.config, "num_processes", 1
                    )
                motion_cmds["ref_motion"]["params"].update(
                    {
                        "seed": int(seed_val),
                        "process_id": cmd_process_id,
                        "num_processes": cmd_num_processes,
                        "is_evaluating": self.is_evaluating,
                    }
                )

            # Build a unified commands cfg that may contain both motion and velocity terms.
            if motion_cmds:
                commands: MoTrack_CommandsCfg = (
                    build_motion_tracking_commands_config(motion_cmds)
                )
            else:
                commands: MoTrack_CommandsCfg = MoTrack_CommandsCfg()
            if vel_cmds:
                vel_commands: VelTrack_CommandsCfg = (
                    build_velocity_commands_config(vel_cmds)
                )
                for name in vel_cmds.keys():
                    setattr(commands, name, getattr(vel_commands, name))
            observations: ObservationsCfg = build_observations_config(
                _obs_config_dict.obs_groups
            )
            rewards: RewardsCfg = build_rewards_config(_rewards_config_dict)

            if _terminations_config_dict:
                terminations: TerminationsCfg = build_terminations_config(
                    _terminations_config_dict
                )
            else:
                terminations: TerminationsCfg = TerminationsCfg()

            if _domain_rand_config_dict:
                events: EventsCfg = build_domain_rand_config(
                    _domain_rand_config_dict
                )
            else:
                events: EventsCfg = EventsCfg()

            if "base_velocity" in vel_cmds:
                events.reset_base = EventTerm(
                    func=isaaclab_mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                            "yaw": (-3.14, 3.14),
                        },
                        "velocity_range": {
                            "x": (0.0, 0.0),
                            "y": (0.0, 0.0),
                            "z": (0.0, 0.0),
                            "roll": (0.0, 0.0),
                            "pitch": (0.0, 0.0),
                            "yaw": (0.0, 0.0),
                        },
                    },
                )
                events.reset_robot_joints = EventTerm(
                    func=isaaclab_mdp.reset_joints_by_scale,
                    mode="reset",
                    params={
                        "position_range": (1.0, 1.0),
                        "velocity_range": (-1.0, 1.0),
                    },
                )

            curriculum: CurriculumCfg = build_curriculum_config(
                _curriculum_config_dict
            )
            actions: ActionsCfg = build_actions_config(_actions_config_dict)
            sim: SimulationCfg = SimulationCfg(
                dt=dt,
                render_interval=decimation,
                physx=physx,
                device=_device,
                enable_scene_query_support=True,
            )
            sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
            sim.physx.enable_stabilization = True
            sim.physics_material = scene.terrain.physics_material

        isaaclab_env_cfg = MotionTrackingEnvCfg()

        isaaclab_envconfig_dump_path = os.path.join(
            self.log_dir, "isaaclab_env_cfg.yaml"
        )
        dump_yaml(isaaclab_envconfig_dump_path, isaaclab_env_cfg)

        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env

    def _init_motion_tracking_components(self):
        self.n_fut_frames = self.config.commands.ref_motion.params.n_fut_frames
        self.target_fps = self.config.commands.ref_motion.params.target_fps
        self._init_serializers()

    def step(self, actor_state: dict):
        obs_dict, rewards, terminated, time_outs, infos = self._env.step(
            actor_state
        )
        # IsaacLab separates terminated vs time_outs, combine them for consistency
        dones = terminated | time_outs
        self._update_completion_rate_stats(terminated, time_outs, infos)
        self._update_robot_metrics(infos)
        return obs_dict, rewards, dones, time_outs, infos

    def _update_robot_metrics(self, infos: dict) -> None:
        """Log robot low-level metrics (scalar means) for TensorBoard/console."""
        if ("log" not in infos) or (not isinstance(infos["log"], dict)):
            infos["log"] = {}

        dt = float(self._env.step_dt)
        action = self._env.action_manager.action  # [B, A]
        prev_action = self._env.action_manager.prev_action  # [B, A]
        action_rate = torch.norm(action - prev_action, dim=-1) / dt  # [B]

        robot = self._env.scene["robot"]
        dof_vel = robot.data.joint_vel  # [B, Nd]
        dof_torque = robot.data.applied_torque  # [B, Nd]

        if self._robot_prev_joint_vel is None or (
            self._robot_prev_joint_vel.shape != dof_vel.shape
        ):
            self._robot_prev_joint_vel = dof_vel.clone()

        dof_acc = (dof_vel - self._robot_prev_joint_vel) / dt  # [B, Nd]
        self._robot_prev_joint_vel = dof_vel.clone()

        if self._robot_prev_applied_torque is None or (
            self._robot_prev_applied_torque.shape != dof_torque.shape
        ):
            joint_ids = torch.arange(
                dof_torque.shape[1], device=dof_torque.device, dtype=torch.long
            )
            effort_limit = _select_effort_limit_vector(robot, joint_ids)
            self._robot_torque_rate_inv_effort_limit = (
                effort_limit.reciprocal()
            )
            self._robot_prev_applied_torque = torch.zeros_like(dof_torque)
            self._robot_torque_rate_needs_reseed = torch.ones(
                dof_torque.shape[0], device=dof_torque.device, dtype=torch.bool
            )

        normed_torque_rate = torch.zeros(
            dof_torque.shape[0],
            device=dof_torque.device,
            dtype=dof_torque.dtype,
        )
        reseed_mask = self._robot_torque_rate_needs_reseed.clone()
        if hasattr(self._env, "episode_length_buf"):
            reseed_mask |= self._env.episode_length_buf == 0

        active_mask = ~reseed_mask
        if torch.any(active_mask):
            delta = (
                dof_torque[active_mask]
                - self._robot_prev_applied_torque[active_mask]
            ) * self._robot_torque_rate_inv_effort_limit
            normed_torque_rate[active_mask] = torch.sum(delta.square(), dim=1)

        self._robot_prev_applied_torque.copy_(dof_torque)
        self._robot_torque_rate_needs_reseed[reseed_mask] = False

        dof_acc_norm = torch.norm(dof_acc, dim=-1)  # [B]
        dof_torque_norm = torch.norm(dof_torque, dim=-1)  # [B]
        energy = torch.sum(
            torch.abs(dof_vel) * torch.abs(dof_torque), dim=-1
        )  # [B]

        self.metrics["Robot/Action_Rate"] = action_rate.mean()
        self.metrics["Robot/DOF_Acc"] = dof_acc_norm.mean()
        self.metrics["Robot/DOF_Torque"] = dof_torque_norm.mean()
        self.metrics["Robot/Energy"] = energy.mean()
        self.metrics["Robot/Normed_Torque_Rate"] = normed_torque_rate.mean()

        infos["log"]["Metrics/Robot/Action_Rate"] = self.metrics[
            "Robot/Action_Rate"
        ]
        infos["log"]["Metrics/Robot/DOF_Acc"] = self.metrics["Robot/DOF_Acc"]
        infos["log"]["Metrics/Robot/DOF_Torque"] = self.metrics[
            "Robot/DOF_Torque"
        ]
        infos["log"]["Metrics/Robot/Energy"] = self.metrics["Robot/Energy"]
        infos["log"]["Metrics/Robot/Normed_Torque_Rate"] = self.metrics[
            "Robot/Normed_Torque_Rate"
        ]

    def _update_completion_rate_stats(
        self,
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        infos: dict,
    ) -> None:
        """Log completion rate over recent done batches.

        Definition:
        - Completed: time_outs==True and terminated==False.
        - Failed: terminated==True.
        The rolling window stores per-step done counts (only when any done occurs).
        """
        done_mask = (terminated | time_outs).reshape(-1).bool()
        if torch.any(done_mask):
            done_count = int(done_mask.sum().item())
            completed_mask = (
                time_outs.reshape(-1).bool()
                & ~terminated.reshape(-1).bool()
                & done_mask
            )
            completed_count = int(completed_mask.sum().item())
            self._completion_total_queue.append(done_count)
            self._completion_success_queue.append(completed_count)

        denom = sum(self._completion_total_queue)
        completion_rate = (
            float(sum(self._completion_success_queue)) / float(denom)
            if denom > 0
            else 0.0
        )
        if ("log" not in infos) or (not isinstance(infos["log"], dict)):
            infos["log"] = {}
        infos["log"]["Metrics/ref_motion/Task/Completion_Rate"] = torch.tensor(
            completion_rate, device=self.device, dtype=torch.float32
        )
        self.metrics["Metrics/ref_motion/Task/Completion_Rate"] = (
            completion_rate
        )

    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        out = self._env.reset(env_ids=env_ids)
        return out

    def set_is_evaluating(self):
        logger.info("Setting environment to evaluation mode")
        self.is_evaluating = True

    def seed(self, seed: int):
        self._env.seed(seed)
