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
# This file was originally copied from the [ASAP] repository:
# https://github.com/LeCAR-Lab/ASAP
# Modifications have been made to fit the needs of this project.


from typing import Any, Dict, List

import numpy as np
import torch
from hydra.utils import get_class
from isaacgym import gymutil
from loguru import logger
from termcolor import colored
from torch import Tensor

from holomotion.src.simulator.base_simulator import BaseSimulator
from holomotion.src.utils.isaac_utils.rotations import (
    get_euler_xyz_in_tensor,
    wrap_to_pi,
)
from holomotion.src.utils.torch_utils import (
    calc_heading_quat_inv,
    get_axis_params,
    my_quat_rotate,
    quat_apply,
    quat_mul,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)


class BaseEnvironment:
    def __init__(self, config, device, log_dir=None):
        self.init_done = False
        self.config = config
        self.log_dir = log_dir
        
        logger.info(f"Log directory: {log_dir}")

        sim_cls = get_class(self.config.simulator._target_)
        self.simulator: BaseSimulator = sim_cls(
            config=self.config, device=device
        )

        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2

        self.dt = (
            self.config.simulator.config.sim.control_decimation * self.sim_dt
        )
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.num_envs = self.config.num_envs
        self.dim_obs = self.config.robot.policy_obs_dim
        self.dim_critic_obs = self.config.robot.critic_obs_dim
        self.dim_actions = self.config.robot.actions_dim

        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)

        self._load_assets()
        self._get_env_origins()
        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = (
            self.simulator.get_dof_limits_properties()
        )
        self._setup_robot_body_indices()
        self.simulator.prepare_sim()
        self.viewer = None
        if not self.headless:
            self.debug_viz = False
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer
        if self.config.get("disable_ref_viz", False):
            self.debug_viz = False
        self._init_buffers()

        # LeggedRobotBase specific initialization
        self._domain_rand_config()
        self._prepare_reward_function()
        self.history_handler = HistoryHandler(
            self.num_envs,
            config.obs.obs_auxiliary,
            config.obs.obs_dims,
            device,
        )
        if self.config.robot.control.control_type == "actuator_net":
            self.actuator_network = torch.jit.load(
                self.config.robot.control.actuator_network_path
            )
            self.actuator_network.eval()
            self.actuator_network.to(self.device)
            self.actuator_network.requires_grad = False
            torch.jit.optimize_for_inference(self.actuator_network)
        self.is_evaluating = False
        self.init_done = True

    def _init_buffers(self):
        self.obs_buf_dict = {}
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.falldown_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.extras = {}
        self.log_dict = {}
        self.log_dict_nonreduced = {}

        # LeggedRobotBase specific buffers
        self.base_quat = self.simulator.base_quat
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # initialize some data used later on
        self._init_counters()
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch(
            [1.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.max_torques = torch.zeros(
            self.num_dofs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions_after_delay = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(
            self.simulator.robot_root_states[:, 7:13]
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.joint_pos_err_last = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.joint_pos_err_last_last = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.joint_vel_last = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.joint_vel_last_last = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.max_torques[i] = (
                        self.config.robot.dof_effort_limit_list[i]
                    )
                    self.p_gains[i] = self.config.robot.control.stiffness[
                        dof_name
                    ]
                    self.d_gains[i] = self.config.robot.control.damping[
                        dof_name
                    ]
                    found = True
                    logger.debug(
                        f"PD gain of joint {name} were defined, setting them "
                        f"to {self.p_gains[i]} and {self.d_gains[i]}"
                    )
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(
                        f"PD gain of joint {name} were not defined, setting "
                        f"them to zero"
                    )
                    raise ValueError(
                        f"PD gain of joint {name} were not defined. Should be "
                        f"defined in the yaml file."
                    )
        if self.config.domain_rand.randomize_joint_default_pos:
            # Expand to per-environment and add random bias
            base_default_pos = self.default_dof_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
            bias = torch_rand_float(
                self.config.domain_rand.joint_default_pos_range[0],
                self.config.domain_rand.joint_default_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
            self.default_dof_pos = base_default_pos + bias
            logger.info(
                f"Randomized default joint positions with bias "
                f"range: {self.config.domain_rand.joint_default_pos_range}"
            )
        else:
            self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._init_domain_rand_buffers()

        # for reward penalty curriculum
        self.average_episode_length = (
            0.0  # num_compute_average_epl last termination episode length
        )
        self.last_episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self.need_to_refresh_envs = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.add_noise_currculum = self.config.obs.add_noise_currculum
        self.current_noise_curriculum_value = (
            self.config.obs.noise_initial_value
        )

    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            self.push_interval_s = torch.randint(
                self.config.domain_rand.push_interval_s[0],
                self.config.domain_rand.push_interval_s[1],
                (self.num_envs,),
                device=self.device,
            )

    def _init_counters(self):
        self.common_step_counter = 0
        self.push_robot_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int,
            device=self.device,
            requires_grad=False,
        )
        self.push_robot_plot_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int,
            device=self.device,
            requires_grad=False,
        )
        self.command_counter = torch.zeros(
            self.num_envs,
            dtype=torch.int,
            device=self.device,
            requires_grad=False,
        )

    def _update_counters_each_step(self):
        self.common_step_counter += 1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    def _init_domain_rand_buffers(self):
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs,
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.action_delay_idx = torch.randint(
                self.config.domain_rand.ctrl_delay_step_range[0],
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                (self.num_envs,),
                device=self.device,
                requires_grad=False,
            )

        self._kp_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._kd_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._rfi_lim_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.push_robot_vel_buf = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_contacts_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_max_height = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_joint_angle_bias = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _prepare_reward_function(self):
        # Create a new dictionary to avoid modifying the original OmegaConf config
        self.reward_scales = {}
        for key, scale in self.config.rewards.reward_scales.items():
            logger.info(f"Scale: {key} = {scale}")
            if scale != 0:
                self.reward_scales[key] = scale * self.dt
            else:
                logger.info(f"Skipping reward {key} as it has zero scale")

        self.use_reward_penalty_curriculum = (
            self.config.rewards.reward_penalty_curriculum
        )
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = (
                self.config.rewards.reward_initial_penalty_scale
            )

        logger.info(
            colored(
                f"Use Reward Penalty: {self.use_reward_penalty_curriculum}",
                "green",
            )
        )
        if self.use_reward_penalty_curriculum:
            logger.info(
                f"Penalty Reward Names: "
                f"{self.config.rewards.reward_penalty_reward_names}"
            )
            logger.info(
                f"Penalty Reward Initial Scale: "
                f"{self.config.rewards.reward_initial_penalty_scale}"
            )

        self.use_reward_limits_dof_pos_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum  # noqa: E501
        self.use_reward_limits_dof_vel_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum  # noqa: E501
        self.use_reward_limits_torque_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum  # noqa: E501

        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(
                f"Use Reward Limits DOF Curriculum: "
                f"{self.use_reward_limits_dof_pos_curriculum}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Initial Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Max Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}"
            )
            logger.info(
                f"Reward Limits DOF Curriculum Min Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}"
            )
            self.soft_dof_pos_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit  # noqa: E501

        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(
                f"Use Reward Limits DOF Vel Curriculum: "
                f"{self.use_reward_limits_dof_vel_curriculum}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Initial Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Max Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}"
            )
            logger.info(
                f"Reward Limits DOF Vel Curriculum Min Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}"
            )
            self.soft_dof_vel_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit  # noqa: E501

        if self.use_reward_limits_torque_curriculum:
            logger.info(
                f"Use Reward Limits Torque Curriculum: "
                f"{self.use_reward_limits_torque_curriculum}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Initial Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Max Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}"
            )
            logger.info(
                f"Reward Limits Torque Curriculum Min Limit: "
                f"{self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}"
            )
            self.soft_torque_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit  # noqa: E501

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, _ in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
            # reward episode sums
            self.episode_sums = {
                name: torch.zeros(
                    self.num_envs,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                for name in self.reward_scales.keys()
            }

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True

    def step(self, actor_state):
        actions = actor_state["actions"]
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(
            actions, -clip_action_limit, clip_action_limit
        ).to(self.device)

        self.log_dict["action_clip_frac"] = (
            self.actions.abs() == clip_action_limit
        ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[
                torch.arange(self.num_envs), self.action_delay_idx
            ].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    def _physics_step(self):
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(
            self.torques.shape
        )
        self.simulator.apply_torques_at_dof(self.torques)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(
            as_tuple=False
        ).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(
                refresh_env_ids, self.simulator.all_root_states
            )
            self.simulator.set_dof_state_tensor(
                refresh_env_ids, self.simulator.dof_state
            )
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations()
        self._post_compute_observations_callback()

        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(
                obs_val, -clip_obs, clip_obs
            )

        for key in self.history_handler.history.keys():
            if key.endswith("_valid_mask"):
                continue
            self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        self.extras["log_nonreduced"] = self.log_dict_nonreduced
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 7:10]
        )
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.simulator.robot_root_states[:, 10:13]
        )
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

    def _update_tasks_callback(self):
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (
                (
                    self.push_robot_counter
                    == (self.push_interval_s / self.dt).int()
                )
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(
                self.config.domain_rand.push_interval_s[0],
                self.config.domain_rand.push_interval_s[1],
                (len(push_robot_env_ids),),
                device=self.device,
                requires_grad=False,
            )
            self._push_robots(push_robot_env_ids)

    def _post_compute_observations_callback(self):
        self.last_last_actions[:] = self.last_actions[:].clone()
        self.last_actions[:] = self.actions[:].clone()
        self.last_dof_pos[:] = self.simulator.dof_pos[:].clone()
        self.last_dof_vel[:] = self.simulator.dof_vel[:].clone()
        self.last_root_vel[:] = self.simulator.robot_root_states[
            :, 7:13
        ].clone()

    def _check_termination(self):
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

        self.reset_buf |= self.time_out_buf

    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            self.reset_buf |= torch.any(
                torch.norm(
                    self.simulator.contact_forces[
                        :, self.termination_contact_indices, :
                    ],
                    dim=-1,
                )
                > 1.0,
                dim=1,
            )

        falldown_flag = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            falldown_flag |= torch.any(
                torch.abs(self.projected_gravity[:, 0:1])
                > self.config.termination_scales.termination_gravity_x,
                dim=1,
            )
            falldown_flag |= torch.any(
                torch.abs(self.projected_gravity[:, 1:2])
                > self.config.termination_scales.termination_gravity_y,
                dim=1,
            )
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            falldown_flag |= torch.any(
                self.simulator.robot_root_states[:, 2:3]
                < self.config.termination_scales.termination_min_base_height,
                dim=1,
            )
        self.falldown_buf = falldown_flag
        self.reset_buf |= self.falldown_buf

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(
                self.simulator.dof_pos
                - self.simulator.dof_pos_limits_termination[:, 0]
            ).clip(max=0.0)  # lower limit
            out_of_dof_pos_limits += (
                self.simulator.dof_pos
                - self.simulator.dof_pos_limits_termination[:, 1]
            ).clip(min=0.0)

            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_dof_pos_limit  # noqa: E501
            ):
                self.reset_buf |= out_of_dof_pos_limits > 0.0

        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits
                    * self.config.termination_scales.termination_close_to_dof_vel_limit  # noqa: E501
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_dof_vel_limit  # noqa: E501
            ):
                self.reset_buf |= out_of_dof_vel_limits > 0.0

        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum(
                (
                    torch.abs(self.torques)
                    - self.torque_limits
                    * self.config.termination_scales.termination_close_to_torque_limit  # noqa: E501
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

            if (
                torch.rand(1)
                < self.config.termination_probality.terminate_when_close_to_torque_limit  # noqa: E501
            ):
                self.reset_buf |= out_of_torque_limits > 0.0

    def _update_timeout_buf(self):
        self.time_out_buf |= (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(
            env_ids
        )  # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids]
                / (self.episode_length_buf[env_ids] + 1)
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_tasks_callback(self, env_ids):
        self._episodic_domain_randomization(env_ids)
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if (
            self.use_reward_limits_dof_pos_curriculum
            or self.use_reward_limits_dof_vel_curriculum
            or self.use_reward_limits_torque_curriculum
        ):
            self._update_reward_limits_curriculum()
        if self.add_noise_currculum:
            self._update_obs_noise_curriculum()

    def _update_obs_noise_curriculum(self):
        if (
            self.average_episode_length
            < self.config.obs.soft_dof_pos_curriculum_level_down_threshold
        ):
            self.current_noise_curriculum_value *= (
                1 - self.config.obs.soft_dof_pos_curriculum_degree
            )
        elif (
            self.average_episode_length
            > self.config.rewards.reward_penalty_level_up_threshold
        ):
            self.current_noise_curriculum_value *= (
                1 + self.config.obs.soft_dof_pos_curriculum_degree
            )

        self.current_noise_curriculum_value = np.clip(
            self.current_noise_curriculum_value,
            self.config.obs.noise_value_min,
            self.config.obs.noise_value_max,
        )

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(
                self.simulator.dof_pos.dtype
            )
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(
                self.simulator.dof_vel.dtype
            )
            self.base_quat[env_ids] = target_buf["base_quat"].to(
                self.base_quat.dtype
            )
            self.base_lin_vel[env_ids] = target_buf["base_lin_vel"].to(
                self.base_lin_vel.dtype
            )
            self.base_ang_vel[env_ids] = target_buf["base_ang_vel"].to(
                self.base_ang_vel.dtype
            )
            self.projected_gravity[env_ids] = target_buf[
                "projected_gravity"
            ].to(self.projected_gravity.dtype)
            self.torques[env_ids] = target_buf["torques"].to(
                self.torques.dtype
            )
            self.actions[env_ids] = target_buf["actions"].to(
                self.actions.dtype
            )
            self.last_actions[env_ids] = target_buf["last_actions"].to(
                self.last_actions.dtype
            )
            self.last_last_actions[env_ids] = target_buf[
                "last_last_actions"
            ].to(self.last_last_actions.dtype)
            self.last_dof_pos[env_ids] = target_buf["last_dof_pos"].to(
                self.last_dof_pos.dtype
            )
            self.last_dof_vel[env_ids] = target_buf["last_dof_vel"].to(
                self.last_dof_vel.dtype
            )
            self.episode_length_buf[env_ids] = target_buf[
                "episode_length_buf"
            ].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(
                self.reset_buf.dtype
            )
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(
                self.time_out_buf.dtype
            )
            self.feet_air_time[env_ids] = target_buf["feet_air_time"].to(
                self.feet_air_time.dtype
            )
            self.last_contacts[env_ids] = target_buf["last_contacts"].to(
                self.last_contacts.dtype
            )
            self.last_contacts_filt[env_ids] = target_buf[
                "last_contacts_filt"
            ].to(self.last_contacts_filt.dtype)
            self.feet_air_max_height[env_ids] = target_buf[
                "feet_air_max_height"
            ].to(self.feet_air_max_height.dtype)
            self.joint_pos_err_last_last[env_ids] = target_buf[
                "joint_pos_err_last_last"
            ].to(self.joint_pos_err_last_last.dtype)
            self.joint_pos_err_last[env_ids] = target_buf[
                "joint_pos_err_last"
            ].to(self.joint_pos_err_last.dtype)
            self.joint_vel_last_last[env_ids] = target_buf[
                "joint_vel_last_last"
            ].to(self.joint_vel_last_last.dtype)
            self.joint_vel_last[env_ids] = target_buf["joint_vel_last"].to(
                self.joint_vel_last.dtype
            )
        else:
            self.actions[env_ids] = 0.0
            self.last_actions[env_ids] = 0.0
            self.last_last_actions[env_ids] = 0.0
            self.actions_after_delay[env_ids] = 0.0
            self.last_dof_pos[env_ids] = 0.0
            self.last_dof_vel[env_ids] = 0.0
            self.joint_pos_err_last_last[env_ids] = 0.0
            self.joint_pos_err_last[env_ids] = 0.0
            self.joint_vel_last_last[env_ids] = 0.0
            self.joint_vel_last[env_ids] = 0.0
            self.feet_air_time[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            # self.reset_buf[env_ids] = 0
            # self.time_out_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._update_average_episode_length(env_ids)

            self.history_handler.reset(env_ids)

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            assert rew.shape[0] == self.num_envs
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = (
                self._reward_termination() * self.reward_scales["termination"]
            )
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(
                self.reward_penalty_scale, dtype=torch.float
            )
            self.log_dict["average_episode_length"] = (
                self.average_episode_length
            )

        if self.use_reward_limits_dof_pos_curriculum:
            self.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(
                self.soft_dof_pos_curriculum_value, dtype=torch.float
            )
        if self.use_reward_limits_dof_vel_curriculum:
            self.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(
                self.soft_dof_vel_curriculum_value, dtype=torch.float
            )
        if self.use_reward_limits_torque_curriculum:
            self.log_dict["soft_torque_curriculum_value"] = torch.tensor(
                self.soft_torque_curriculum_value, dtype=torch.float
            )

        if self.add_noise_currculum:
            self.log_dict["current_noise_curriculum_value"] = torch.tensor(
                self.current_noise_curriculum_value, dtype=torch.float
            )

    def _compute_observations(self):
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}

        if self.add_noise_currculum:
            noise_extra_scale = self.current_noise_curriculum_value
        else:
            noise_extra_scale = 1.0
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(
                self,
                obs_config,
                self.obs_buf_dict_raw[obs_key],
                self.config.obs.obs_scales,
                self.config.obs.noise_scales,
                noise_extra_scale,
            )

        history_obs_list = self.history_handler.history.keys()
        parse_observation(
            self,
            history_obs_list,
            self.hist_obs_dict,
            self.config.obs.obs_scales,
            self.config.obs.noise_scales,
            noise_extra_scale,
        )

        self._post_config_observation_callback()

    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()

        if self.config.obs.get("obs_type", "flattened") == "flattened":
            for obs_key, obs_config in self.config.obs.obs_dict.items():
                obs_keys = sorted(obs_config)
                self.obs_buf_dict[obs_key] = torch.cat(
                    [self.obs_buf_dict_raw[obs_key][key] for key in obs_keys],
                    dim=-1,
                )

    def _compute_torques(self, actions):
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if control_type == "actuator_net":
            self.joint_pos_target = actions_scaled + self.default_dof_pos
            self.joint_pos_err = self.simulator.dof_pos - self.joint_pos_target
            self.joint_vel = self.simulator.dof_vel
            actuator_inputs = torch.cat(
                [
                    self.joint_pos_err.unsqueeze(-1),  # Current error
                    self.joint_pos_err_last.unsqueeze(-1),  # Last error
                    self.joint_pos_err_last_last.unsqueeze(
                        -1
                    ),  # Last last error
                    self.joint_vel.unsqueeze(-1),  # Current velocity
                    self.joint_vel_last.unsqueeze(-1),  # Last velocity
                    self.joint_vel_last_last.unsqueeze(
                        -1
                    ),  # Last last velocity
                ],
                dim=-1,
            )

            # Update history
            self.joint_pos_err_last_last = self.joint_pos_err_last.clone()
            self.joint_pos_err_last = self.joint_pos_err.clone()
            self.joint_vel_last_last = self.joint_vel_last.clone()
            self.joint_vel_last = self.joint_vel.clone()

            # Get torques from actuator network
            with torch.inference_mode():
                torques = self.actuator_network(actuator_inputs)

        elif control_type == "P":
            torques = (
                self._kp_scale
                * self.p_gains
                * (
                    actions_scaled
                    + self.default_dof_pos
                    - self.simulator.dof_pos
                )
                - self._kd_scale * self.d_gains * self.simulator.dof_vel
            )
        elif control_type == "P_Normed":
            alpha = self.config.robot.control.action_scale * (
                self.max_torques / self.p_gains
            )
            target_actions = self.default_dof_pos + alpha * actions
            torques = (
                self._kp_scale
                * self.p_gains
                * (target_actions - self.simulator.dof_pos)
                - self._kd_scale * self.d_gains * self.simulator.dof_vel
            )
        elif control_type == "V":
            torques = (
                self._kp_scale
                * self.p_gains
                * (actions_scaled - self.simulator.dof_vel)
                - self._kd_scale
                * self.d_gains
                * (self.simulator.dof_vel - self.last_dof_vel)
                / self.sim_dt
            )
        elif control_type == "T":
            torques = actions_scaled
        elif control_type == "P_replay":
            torques = (
                self.p_gains * self.dif_joint_angles
                - self.d_gains * self.simulator.dof_vel
            )
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.config.domain_rand.randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0)
                * self.config.domain_rand.rfi_lim
                * self._rfi_lim_scale
                * self.torque_limits
            )

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)

        else:
            return torques

    def _create_terrain(self):
        super()._create_terrain()

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self._refresh_sim_tensors()

        draw_env_ids = (
            (self.push_robot_plot_counter < 10)
            .nonzero(as_tuple=False)
            .flatten()
        )
        not_draw_env_ids = (
            (self.push_robot_plot_counter >= 10)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.record_push_robot_vel_buf[not_draw_env_ids] *= 0
        self.push_robot_plot_counter[not_draw_env_ids] = 0

        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat(
                [push_vel, torch.zeros(1, device=self.device)]
            )
            push_pos = self.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for (
                push_vel,
                push_pos,
                push_mag,
                push_color,
                push_line_width,
            ) in zip(
                push_vel_list,
                push_pos_list,
                push_mag_list,
                push_color_schems,
                push_line_widths,
                strict=False,
            ):
                for _ in range(200):
                    gymutil.draw_line(
                        Point(
                            push_pos
                            + torch.rand(3, device=self.device)
                            * push_line_width
                        ),
                        Point(push_pos + push_vel * push_mag),
                        Point(push_color),
                        self.gym,
                        self.viewer,
                        self.envs[env_id],
                    )

    ################ Curriculum #################

    def _update_average_episode_length(self, env_ids):
        current_average_episode_length = torch.mean(
            self.last_episode_length_buf[env_ids], dtype=torch.float
        )
        ema_gamma = 0.8
        self.average_episode_length = (
            self.average_episode_length * ema_gamma
            + current_average_episode_length * (1 - ema_gamma)
        )

    def _update_reward_penalty_curriculum(self):
        if (
            self.average_episode_length
            < self.config.rewards.reward_penalty_level_down_threshold
        ):
            self.reward_penalty_scale *= (
                1 - self.config.rewards.reward_penalty_degree
            )
        elif (
            self.average_episode_length
            > self.config.rewards.reward_penalty_level_up_threshold
        ):
            self.reward_penalty_scale *= (
                1 + self.config.rewards.reward_penalty_degree
            )

        self.reward_penalty_scale = np.clip(
            self.reward_penalty_scale,
            self.config.rewards.reward_min_penalty_scale,
            self.config.rewards.reward_max_penalty_scale,
        )

    def _update_reward_limits_curriculum(self):
        if self.use_reward_limits_dof_pos_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold  # noqa: E501
            ):
                self.soft_dof_pos_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree  # noqa: E501
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold  # noqa: E501
            ):
                self.soft_dof_pos_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree  # noqa: E501
                )
            self.soft_dof_pos_curriculum_value = np.clip(
                self.soft_dof_pos_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit,
            )

        if self.use_reward_limits_dof_vel_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold  # noqa: E501
            ):
                self.soft_dof_vel_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree  # noqa: E501
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold  # noqa: E501
            ):
                self.soft_dof_vel_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree  # noqa: E501
                )
            self.soft_dof_vel_curriculum_value = np.clip(
                self.soft_dof_vel_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit,
            )

        if self.use_reward_limits_torque_curriculum:
            if (
                self.average_episode_length
                < self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold  # noqa: E501
            ):
                self.soft_torque_curriculum_value *= (
                    1
                    + self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree  # noqa: E501
                )
            elif (
                self.average_episode_length
                > self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold  # noqa: E501
            ):
                self.soft_torque_curriculum_value *= (
                    1
                    - self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree  # noqa: E501
                )
            self.soft_torque_curriculum_value = np.clip(
                self.soft_torque_curriculum_value,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit,
                self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit,
            )

    # ------------ reward functions----------------
    ########################### PENALTY REWARDS ###########################
    
    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_termination(self):
        # Terminal reward / penalty
        return (
            self.reset_buf * (~self.time_out_buf) * (~self.falldown_buf.bool())
        )

    def _reward_falldown(self):
        # Penalize falldown
        return self.falldown_buf

    def _reward_penalty_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)

    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square(
                (self.last_dof_vel - self.simulator.dof_vel) / self.dt
            ),
            dim=1,
        )

    def _reward_penalty_action_smooth(self):
        # Penalize action change 2nd derivative
        rew_action_smooth = torch.mean(
            torch.square(
                (self.actions - self.last_actions)
                - (self.last_actions - self.last_last_actions)
            ),
            dim=-1,
        )
        self.action_smooth = (self.actions - self.last_actions) - (
            self.last_actions - self.last_last_actions
        )
        return rew_action_smooth

    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        rew_action_rate = torch.mean(
            torch.square(self.last_actions - self.actions), dim=-1
        )
        self.action_rate = self.last_actions - self.actions
        return rew_action_rate

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_penalty_power(self):
        return torch.mean(
            torch.abs(self.torques * self.simulator.dof_vel), dim=-1
        )

    ######################## LIMITS REWARDS #########################

    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit

        if self.use_reward_limits_dof_pos_curriculum:
            m = (
                self.simulator.hard_dof_pos_limits[:, 0]
                + self.simulator.hard_dof_pos_limits[:, 1]
            ) / 2
            r = (
                self.simulator.hard_dof_pos_limits[:, 1]
                - self.simulator.hard_dof_pos_limits[:, 0]
            )
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
        else:
            lower_soft_limit = self.simulator.dof_pos_limits[:, 0]
            upper_soft_limit = self.simulator.dof_pos_limits[:, 1]
        out_of_limits = -(self.simulator.dof_pos - lower_soft_limit).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.simulator.dof_pos - upper_soft_limit).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if self.use_reward_limits_dof_vel_curriculum:
            return torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits * self.soft_dof_vel_curriculum_value
                ).clip(min=0.0, max=1.0),
                dim=1,
            )
        else:
            return torch.sum(
                (
                    torch.abs(self.simulator.dof_vel)
                    - self.dof_vel_limits
                    * self.config.rewards.reward_limit.soft_dof_vel_limit
                ).clip(min=0.0, max=1.0),
                dim=1,
            )

    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        if self.use_reward_limits_torque_curriculum:
            return torch.sum(
                (
                    torch.abs(self.torques)
                    - self.torque_limits * self.soft_torque_curriculum_value
                ).clip(min=0.0, max=1.0),
                dim=1,
            )
        else:
            return torch.sum(
                (
                    torch.abs(self.torques)
                    - self.torque_limits
                    * self.config.rewards.reward_limit.soft_torque_limit
                ).clip(min=0.0),
                dim=1,
            )

    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(
            torch.norm(foot_vel, dim=-1)
            * (
                torch.norm(
                    self.simulator.contact_forces[:, self.feet_indices, :],
                    dim=-1,
                )
                > 1.0
            ),
            dim=1,
        )

    def _reward_feet_max_height_for_this_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        from_air_to_contact = torch.logical_and(
            contact_filt, ~self.last_contacts_filt
        )
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt
        self.feet_air_max_height = torch.max(
            self.feet_air_max_height,
            self.simulator._rigid_body_pos[:, self.feet_indices, 2],
        )

        rew_feet_max_height = torch.sum(
            (
                torch.clamp_min(
                    self.config.rewards.desired_feet_max_height_for_this_air
                    - self.feet_air_max_height,
                    0,
                )
            )
            * from_air_to_contact,
            dim=1,
        )  # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(
            forward_left_feet[:, 1], forward_left_feet[:, 0]
        )
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(
            forward_right_feet[:, 1], forward_right_feet[:, 0]
        )

        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(
            wrap_to_pi(heading_left_feet - heading_root)
        )
        heading_diff_right = torch.abs(
            wrap_to_pi(heading_right_feet - heading_root)
        )

        return heading_diff_left + heading_diff_right

    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return (
            torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
            + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
        )

    def _episodic_domain_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.kp_range[0],
                self.config.domain_rand.kp_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )
            self._kd_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.kd_range[0],
                self.config.domain_rand.kd_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )

        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(
                self.config.domain_rand.rfi_lim_range[0],
                self.config.domain_rand.rfi_lim_range[1],
                (len(env_ids), self.num_dofs),
                device=self.device,
            )

        if self.config.domain_rand.randomize_ctrl_delay:
            # self.action_queue[env_ids] = 0.delay:
            self.action_queue[env_ids] *= 0.0
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(
                self.config.domain_rand.ctrl_delay_step_range[0],
                self.config.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )

    def _push_robots(self, env_ids):
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel = self.config.domain_rand.max_push_vel_xy
        self.push_robot_vel_buf[env_ids] = torch_rand_float(
            -max_vel, max_vel, (len(env_ids), 2), device=str(self.device)
        )  # lin vel x/y
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[
            env_ids
        ].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = (
            self.push_robot_vel_buf[env_ids]
        )

    def _reset_dofs(self, env_ids, target_state=None):
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = (
                self.default_dof_pos
                * torch_rand_float(
                    0.5,
                    1.5,
                    (len(env_ids), self.num_dof),
                    device=str(self.device),
                )
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[
                env_ids
            ]

        else:
            # base position
            if self.custom_origins:
                self.simulator.robot_root_states[env_ids] = (
                    self.base_init_state
                )
                self.simulator.robot_root_states[env_ids, :3] += (
                    self.env_origins[env_ids]
                )
                self.simulator.robot_root_states[env_ids, :2] += (
                    torch_rand_float(
                        -1.0, 1.0, (len(env_ids), 2), device=str(self.device)
                    )
                )  # xy position within 1m of the center
            else:
                self.simulator.robot_root_states[env_ids] = (
                    self.base_init_state
                )
                self.simulator.robot_root_states[env_ids, :3] += (
                    self.env_origins[env_ids]
                )
            # base velocities

            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel

    ######################### Observations #########################
    def _get_obs_base_pos_z(
        self,
    ):
        return self.simulator.robot_root_states[:, 2:3]

    def _get_obs_base_rot(
        self,
    ):
        return self.simulator.robot_root_states[:, 3:7]

    def _get_obs_feet_contact_force(
        self,
    ):
        return self.simulator.contact_forces[:, self.feet_indices, :].view(
            self.num_envs, -1
        )

    def _get_obs_base_lin_vel(
        self,
    ):
        return self.base_lin_vel

    def _get_obs_base_ang_vel(
        self,
    ):
        return self.base_ang_vel

    def _get_obs_projected_gravity(
        self,
    ):
        return self.projected_gravity

    def _get_obs_dof_pos(
        self,
    ):
        return self.simulator.dof_pos - self.default_dof_pos

    def _get_obs_dof_vel(
        self,
    ):
        return self.simulator.dof_vel

    def _get_obs_history(
        self,
    ):
        assert "history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_short_history(
        self,
    ):
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["short_history"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_long_history(
        self,
    ):
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["long_history"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_actor(self):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_actor"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_critic(self):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_critic"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[
                :, :history_length
            ]
            history_tensor = history_tensor.reshape(
                history_tensor.shape[0], -1
            )
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_domain_params(self):
        domain_params = self.config.get("domain_params", {})

        # base com
        if domain_params.get("randomize_base_com", False):
            com_vec = self.simulator._base_com_bias  # [num_envs, 3]
        else:
            com_vec = torch.zeros(
                self.num_envs, 3, device=self.device
            )  # [num_envs, 3]

        # link mass
        if domain_params.get("randomize_link_mass", False):
            link_mass_vec = (
                self.simulator._link_mass_scale
            )  # [num_envs, num_links]
        else:
            link_mass_vec = torch.ones(
                self.num_envs,
                len(self.config.robot.randomize_link_body_names),
                device=self.device,
            )  # [num_envs, num_links]

        # pd gain
        if domain_params.get("randomize_pd_gain", False):
            kp_vec = self._kp_scale  # [num_envs, num_dofs]
            kd_vec = self._kd_scale  # [num_envs, num_dofs]
            pd_vec = torch.cat(
                [kp_vec, kd_vec], dim=-1
            )  # [num_envs, 2*num_dofs]
        else:
            pd_vec = torch.ones(
                self.num_envs, 2 * self.num_dofs, device=self.device
            )  # [num_envs, 2*num_dofs]

        # friction
        if domain_params.get("randomize_friction", False):
            # directly use the environment friction
            friction_vec = self.simulator._friction_coeffs[
                :, None
            ]  # [num_envs, 1]
        else:
            friction_vec = torch.ones(
                self.num_envs, 1, device=self.device
            )  # [num_envs, 1]

        # base mass
        if domain_params.get("randomize_base_mass", False):
            base_mass_vec = self.simulator._base_mass_scale[
                :, None
            ]  # [num_envs, 1]
        else:
            base_mass_vec = torch.ones(
                self.num_envs, 1, device=self.device
            )  # [num_envs, 1]

        # rfi
        if domain_params.get("randomize_torque_rfi", False):
            rfi_lim_vec = self.config.domain_rand.rfi_lim * torch.ones(
                self.num_envs, 1, device=self.device
            )
            torque_rfi_lim_vec = self._rfi_lim_scale  # [num_envs, num_dofs]
            rfi_vec = torch.cat(
                [rfi_lim_vec, torque_rfi_lim_vec], dim=-1
            )  # [num_envs, 1 + num_dofs]
        else:
            rfi_lim_vec = torch.zeros(self.num_envs, 1, device=self.device)
            torque_rfi_lim_vec = torch.zeros(
                self.num_envs, self.num_dofs, device=self.device
            )
            rfi_vec = torch.cat(
                [rfi_lim_vec, torque_rfi_lim_vec], dim=-1
            )  # [num_envs, 1 + num_dofs]

        # combine all domain params
        # import ipdb; ipdb.set_trace()
        domain_params_vec = torch.cat(
            [
                com_vec.to(self.device),  # [num_envs, 3]
                link_mass_vec.to(self.device),  # [num_envs, num_links]
                pd_vec.to(self.device),  # [num_envs, 2*num_dofs]
                friction_vec.to(self.device),  # [num_envs, 1]
                base_mass_vec.to(self.device),  # [num_envs, 1]
                rfi_vec.to(self.device),  # [num_envs, 1 + num_dofs]
            ],
            dim=-1,
        )  # [num_envs, 3 + num_links + 2*num_dofs + 1 + num_dofs]
        return domain_params_vec

    def _get_obs_actions(self):
        return self.actions

    def _get_obs_base_height(self) -> torch.Tensor:
        return self.simulator.robot_root_states[:, 2:3]

    def _get_obs_local_body_pos(self) -> torch.Tensor:
        local_body_pos = (
            self.simulator._rigid_body_pos
            - self.simulator.robot_root_states[:, :3][:, None, :]
        )  # [num_envs, num_rigid_bodies, 3]
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7]
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_pos.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies, 4]
        local_body_pos = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4), local_body_pos.view(-1, 3)
        )  # [num_envs, num_rigid_bodies * 3]
        local_body_pos = local_body_pos.reshape(n_envs, n_bodies, 3)
        return local_body_pos

    def _get_obs_local_body_pos_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_pos().reshape(self.num_envs, -1)

    def _get_obs_local_body_rot_quat(self) -> torch.Tensor:
        global_body_rot_quat = self.simulator._rigid_body_rot
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7]
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = global_body_rot_quat.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies, 4]
        local_body_rot_quat = quat_mul(
            root_heading_quat_inv,
            global_body_rot_quat,
        )  # [num_envs, num_rigid_bodies * 4]
        local_body_rot_quat = local_body_rot_quat.reshape(n_envs, n_bodies, 4)
        return local_body_rot_quat

    def _get_obs_local_body_rot_quat_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_rot_quat().reshape(self.num_envs, -1)

    def _get_obs_local_body_vel(self) -> torch.Tensor:
        global_body_vel = (
            self.simulator._rigid_body_vel
        )  # [num_envs, num_rigid_bodies, 3]
        local_body_vel = global_body_vel - self.base_lin_vel[:, None, :]
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7]
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_vel.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies, 4]
        local_body_vel = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4),
            local_body_vel.view(-1, 3),
        )  # [num_envs, num_rigid_bodies, 3]
        local_body_vel = local_body_vel.reshape(n_envs, n_bodies, 3)
        return local_body_vel

    def _get_obs_local_body_vel_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_vel().reshape(self.num_envs, -1)

    def _get_obs_local_body_ang_vel(self) -> torch.Tensor:
        global_body_ang_vel = self.simulator._rigid_body_ang_vel
        local_body_ang_vel = (
            global_body_ang_vel - self.base_ang_vel[:, None, :]
        )
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7]
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_ang_vel.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies, 4]
        local_body_ang_vel = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4),
            local_body_ang_vel.view(-1, 3),
        )  # [num_envs, num_rigid_bodies, 3]
        local_body_ang_vel = local_body_ang_vel.reshape(n_envs, n_bodies, 3)
        return local_body_ang_vel

    def _get_obs_local_body_ang_vel_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_ang_vel().reshape(self.num_envs, -1)

    def _get_obs_limb_weight_params(self) -> torch.Tensor:
        env_limb_weights = self.simulator.limb_weights
        flat_limb_weights = env_limb_weights.view(self.num_envs, -1)
        return flat_limb_weights

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def reset_all(self):
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(
            torch.arange(self.num_envs, device=self.device),
            self.simulator.all_root_states,
        )
        self.simulator.set_dof_state_tensor(
            torch.arange(self.num_envs, device=self.device),
            self.simulator.dof_state,
        )
        actions = torch.zeros(
            self.num_envs,
            self.dim_actions,
            device=self.device,
            requires_grad=False,
        )
        actor_state = {}
        actor_state["actions"] = actions
        obs_dict, _, _, _ = self.step(actor_state)
        return obs_dict

    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    def _get_env_origins(self):
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            max_init_level = self.config.terrain.max_init_terrain_level
            if not self.config.terrain.curriculum:
                max_init_level = self.config.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.config.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.config.terrain.num_rows
            if isinstance(self.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = (
                    torch.from_numpy(self.simulator.terrain.env_origins)
                    .to(self.device)
                    .to(torch.float)
                )
            else:
                self.terrain_origins = self.simulator.terrain.env_origins.to(
                    self.device
                ).to(torch.float)
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
            # import ipdb; ipdb.set_trace()
            # print(self.terrain_origins.shape)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(
                torch.arange(num_rows), torch.arange(num_cols)
            )
            spacing = self.config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _load_assets(self):
        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
        )

        # check dimensions
        assert self.num_dof == self.dim_actions, (
            f"Number of DOFs must be equal to number of actions. "
            f"Got {self.num_dof} DOFs and {self.dim_actions} actions."
        )

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )

    def _create_envs(self):
        self.simulator.create_envs(
            self.num_envs, self.env_origins, self.base_init_state
        )

    def _setup_robot_body_indices(self):
        feet_names = [
            s for s in self.body_names if self.config.robot.foot_name in s
        ]
        knee_names = [
            s for s in self.body_names if self.config.robot.knee_name in s
        ]
        penalized_contact_names = []
        for name in self.config.robot.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in self.body_names if name in s]
            )
        termination_contact_names = []
        for name in self.config.robot.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in self.body_names if name in s]
            )

        self.feet_indices = torch.zeros(
            len(feet_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(
                feet_names[i]
            )

        self.knee_indices = torch.zeros(
            len(knee_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.simulator.find_rigid_body_indice(
                knee_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = (
                self.simulator.find_rigid_body_indice(
                    penalized_contact_names[i]
                )
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = (
                self.simulator.find_rigid_body_indice(
                    termination_contact_names[i]
                )
            )

        if self.config.robot.has_upper_body_dof:
            # maintain upper/lower dof idxs
            self.upper_dof_names = self.config.robot.upper_dof_names
            self.lower_dof_names = self.config.robot.lower_dof_names
            self.upper_dof_indices = [
                self.dof_names.index(dof) for dof in self.upper_dof_names
            ]
            self.lower_dof_indices = [
                self.dof_names.index(dof) for dof in self.lower_dof_names
            ]
            self.waist_dof_indices = [
                self.dof_names.index(dof)
                for dof in self.config.robot.waist_dof_names
            ]

        if self.config.robot.has_torso:
            self.torso_name = self.config.robot.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(
                self.torso_name
            )

        if self.config.robot.get("waist_roll_pitch_dof_names", []):
            self.waist_roll_pitch_dof_names = (
                self.config.robot.waist_roll_pitch_dof_names
            )
            self.waist_roll_pitch_dof_indices = [
                self.dof_names.index(dof)
                for dof in self.waist_roll_pitch_dof_names
            ]

        if self.config.robot.get("head_hand_bodies", []):
            self.head_hand_body_indices = [
                self.simulator.find_rigid_body_indice(name)
                for name in self.config.robot.head_hand_bodies
            ]

        if self.config.robot.get("foot_hand_bodies", []):
            self.foot_hand_body_indices = [
                self.simulator.find_rigid_body_indice(name)
                for name in self.config.robot.foot_hand_bodies
            ]

        if self.config.robot.get("reset_bodies", []):
            self.reset_body_indices = [
                self.simulator.find_rigid_body_indice(name)
                for name in self.config.robot.reset_bodies
            ]
        else:
            self.reset_body_indices = [
                self.simulator.find_rigid_body_indice(name)
                for name in self.body_names
            ]


class HistoryHandler:
    def __init__(self, num_envs, history_config, obs_dims, device):
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.config = history_config
        self.history = {}

        self.buffer_config = {}
        for _, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(
                        self.buffer_config[obs_key], obs_num
                    )
                else:
                    self.buffer_config[obs_key] = obs_num

        for key in self.buffer_config.keys():
            self.history[key] = torch.zeros(
                num_envs,
                self.buffer_config[key],
                obs_dims[key],
                device=self.device,
            )
            # Initialize valid mask for each history buffer
            self.history[key + "_valid_mask"] = torch.zeros(
                num_envs,
                self.buffer_config[key],
                dtype=torch.bool,
                device=self.device,
            )

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        # Also reset the valid masks for the specified environments
        for key in self.history.keys():
            if key.endswith("_valid_mask"):
                self.history[key][reset_ids] = False
            else:
                self.history[key][reset_ids] *= 0.0

    def add(self, key: str, value: Tensor):
        assert key in self.buffer_config.keys(), (
            f"Key {key} not found in history config"
        )
        mask_key = key + "_valid_mask"
        assert mask_key in self.history.keys(), (
            f"Mask key {mask_key} not found"
        )

        # Shift history data
        val = self.history[key].clone()
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value.clone()

        # Shift valid mask and set the new entry to True
        mask_val = self.history[mask_key].clone()
        self.history[mask_key][:, 1:] = mask_val[:, :-1]
        self.history[mask_key][:, 0] = True

    def query(self, key: str):
        assert key in self.buffer_config.keys(), (
            f"Key {key} not found in history config"
        )
        return self.history[key].clone()

    def query_valid_mask(self, key: str):
        """Returns the valid mask for the specified history key."""
        mask_key = key + "_valid_mask"
        assert mask_key in self.history.keys(), (
            f"Mask key {mask_key} not found"
        )
        return self.history[mask_key].clone()


class Point:
    def __init__(self, pt):
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]


def parse_observation(
    cls: Any,
    key_list: List,
    buf_dict: Dict,
    obs_scales: Dict,
    noise_scales: Dict,
    current_noise_curriculum_value: Any,
) -> None:
    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4]
            obs_noise = 0.0
        elif obs_key.endswith("_valid_mask"):
            continue
        else:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        buf_dict[obs_key] = (
            actor_obs + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise
        ) * obs_scale
