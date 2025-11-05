# Project HoloMotion
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

import torch
import time
import yaml
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
from holomotion.src.modules.agent_modules import ObsSeqSerializer
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
        self._init_serializers()

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def _init_isaaclab_env(self):
        _device = self._device

        # curriculum = CurriculumCfg()

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

        @configclass
        class MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
            seed: int = seed_val
            scene_config_dict = {
                "num_envs": self.config.num_envs,
                "env_spacing": self.config.env_spacing,
                "replicate_physics": self.config.replicate_physics,
                "robot": _robot_config_dict,
                "terrain": _terrain_config_dict,
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

            command_name = list(_commands_config_dict.keys())[0]
            if command_name == "ref_motion":
                if self.accelerator is not None:
                    cmd_process_id = self.accelerator.process_index
                    cmd_num_processes = self.accelerator.num_processes
                else:
                    cmd_process_id = getattr(self.config, "process_id", 0)
                    cmd_num_processes = getattr(
                        self.config, "num_processes", 1
                    )
                _commands_config_dict[command_name]["params"].update(
                    {
                        "process_id": cmd_process_id,
                        "num_processes": cmd_num_processes,
                        "is_evaluating": self.is_evaluating,
                    }
                )
                commands: MoTrack_CommandsCfg = (
                    build_motion_tracking_commands_config(
                        _commands_config_dict
                    )
                )
            else:
                commands: VelTrack_CommandsCfg = (
                    build_velocity_commands_config(_commands_config_dict)
                )
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

            if command_name == "base_velocity":
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

            # curriculum: CurriculumCfg = build_curriculum_config(
            #     getattr(self.config, "curriculum", {})
            # )
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

        # dump_yaml("./isaaclab_env_cfg.yaml", isaaclab_env_cfg)

        self._env = ManagerBasedRLEnv(isaaclab_env_cfg, self.render_mode)

        logger.info("IsaacLab environment initialized !")
        return self._env

    def _init_motion_tracking_components(self):
        self.num_extend_bodies = len(
            getattr(self.config, "robot", {})
            .get("motion", {})
            .get("extend_config", [])
        )
        self.n_fut_frames = self.config.commands.ref_motion.params.n_fut_frames
        self.target_fps = self.config.commands.ref_motion.params.target_fps
        self._init_serializers()

    def step(self, actor_state: dict):
        obs_dict, rewards, terminated, time_outs, infos = self._env.step(
            actor_state
        )
        # IsaacLab separates terminated vs time_outs, combine them for consistency
        dones = terminated | time_outs
        return obs_dict, rewards, dones, time_outs, infos

    def reset_idx(self, env_ids: torch.Tensor):
        return self._env.reset(env_ids=env_ids)

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        out = self._env.reset(env_ids=env_ids)
        return out

    def set_is_evaluating(self):
        logger.info("Setting environment to evaluation mode")
        self.is_evaluating = True

    def _init_serializers(self):
        if hasattr(self.config, "obs"):
            obs_config = self.config.obs

            if obs_config.get("serialization_schema", None):
                self.obs_serializer = ObsSeqSerializer(
                    obs_config.serialization_schema
                )

            if obs_config.get("critic_serialization_schema", None):
                self.critic_obs_serializer = ObsSeqSerializer(
                    obs_config.critic_serialization_schema
                )

            if obs_config.get("teacher_serialization_schema", None):
                self.teacher_obs_serializer = ObsSeqSerializer(
                    obs_config.teacher_serialization_schema
                )

            if obs_config.get("command_serilization_schema", None):
                self.command_obs_serializer = ObsSeqSerializer(
                    obs_config.command_serilization_schema
                )

    def seed(self, seed: int):
        """Seed python, numpy and torch RNGs for this env wrapper and IsaacLab.

        Args:
            seed: Base seed value to use.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # If the underlying IsaacLab env exposes seed/reset seeding hooks, use them
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
