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


from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.transform import Rotation as sRot

from holomotion.src.env.env_base import BaseEnvironment
from holomotion.src.modules.agent_modules import ObsSeqSerializer
from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib
from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    my_quat_rotate,
    quat_inverse,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_matrix,
    wrap_to_pi,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holomotion.src.utils.torch_utils import quat_error_magnitude


class MotionTrackingEnvironment(BaseEnvironment):
    def __init__(self, config, device, log_dir=None):
        self.init_done = False
        self.debug_viz = True
        self.is_evaluating = False
        self.num_extend_bodies = len(config.robot.motion.get("extend_config", []))
        self.n_fut_frames = config.obs.get("n_fut_frames", 1)
        super().__init__(config, device, log_dir)
        self._init_motion_extend()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True

        self.log_dict_holomotion = {}
        self.log_dict_nonreduced_holomotion = {}
        if self.config.get("disable_ref_viz", False):
            self.debug_viz = False

        self.entropy_coef = self.config.get("init_entropy_coef", 0.01)

        if config.obs.get("serialization_schema", None):
            self.obs_serializer = ObsSeqSerializer(config.obs.serialization_schema)

        if config.obs.get("critic_serialization_schema", None):
            self.critic_obs_serializer = ObsSeqSerializer(
                config.obs.critic_serialization_schema
            )

        if config.obs.get("teacher_serialization_schema", None):
            self.teacher_obs_serializer = ObsSeqSerializer(
                config.obs.teacher_serialization_schema
            )

        if config.get("main_process", True):
            logger.info(f"Current termination strategy: {self.config.termination}")

        if (
            self.config.termination.terminate_when_motion_far
            and self.config.termination_curriculum.terminate_when_motion_far_curriculum  # noqa: E501
        ):
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold  # noqa: E501
            logger.info(
                f"Terminate when motion far threshold: "
                f"{self.terminate_when_motion_far_threshold}"
            )
        else:
            self.terminate_when_motion_far_threshold = (
                self.config.termination_scales.termination_motion_far_threshold
            )
            logger.info(
                f"Terminate when motion far threshold: "
                f"{self.terminate_when_motion_far_threshold}"
            )

        if (
            self.config.termination.get("terminate_when_joint_far", False)
            and self.config.termination_curriculum.terminate_when_joint_far_curriculum  # noqa: E501
        ):
            self.terminate_when_joint_far_threshold = self.config.termination_curriculum.terminate_when_joint_far_initial_threshold  # noqa: E501
            logger.info(
                f"Terminate when joint far threshold: "
                f"{self.terminate_when_joint_far_threshold}"
            )
        else:
            if self.config.termination_scales.get(
                "terminate_when_joint_far_threshold", False
            ):
                self.terminate_when_joint_far_threshold = (
                    self.config.termination_scales.terminate_when_joint_far_threshold
                )  # noqa: E501
                logger.info(
                    f"Terminate when joint far threshold: "
                    f"{self.terminate_when_joint_far_threshold}"
                )

        self.use_waist_dof_curriculum = self.config.rewards.get(
            "use_waist_dof_curriculum", False
        )
        if self.use_waist_dof_curriculum:
            logger.info(f"Use Waist DOF Curriculum: {self.use_waist_dof_curriculum}")
            self.waist_dof_penalty_scale = torch.tensor(1.0, device=self.device)
            logger.info(f"Waist DOF Penalty Scale: {self.waist_dof_penalty_scale}")

        # ---------------- Joint-far patience configuration -----------------
        # This setting controls how many consecutive steps an environment must
        # violate the joint-far threshold before it is reset. A value of 1
        # reproduces the original behaviour (immediate reset).
        if self.config.termination.get("terminate_when_joint_far", False):
            self.terminate_when_joint_far_patience_steps = self.config.termination.get(
                "terminate_when_joint_far_patience_steps", 1
            )
        else:
            # Fallback to 1 for compatibility when the mechanism is disabled
            self.terminate_when_joint_far_patience_steps = 1

        # ---------------- Motion-far patience configuration ----------------
        if self.config.termination.get("terminate_when_motion_far", False):
            self.terminate_when_motion_far_patience_steps = self.config.termination.get(
                "terminate_when_motion_far_patience_steps", 1
            )
        else:
            self.terminate_when_motion_far_patience_steps = 1

    def _init_motion_lib(self):
        # cache_device = torch.device("cpu")
        cache_device = self.device
        self._motion_lib = LmdbMotionLib(
            self.config.robot.motion,
            # self.device,
            cache_device,
            process_id=self.config.get("process_id", 0),
            num_processes=self.config.get("num_processes", 1),
        )
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [
                self.simulator._body_list.index(link)
                for link in self.config.robot.motion.motion_tracking_link
            ]
        if "hand_link" in self.config.robot.motion:
            self.hand_indices = [
                self.simulator._body_list.index(link)
                for link in self.config.robot.motion.hand_link
            ]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [
                self.simulator._body_list.index(link)
                for link in self.config.robot.motion.lower_body_link
            ]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [
                self.simulator._body_list.index(link)
                for link in self.config.robot.motion.upper_body_link
            ]
        if "lower_body_joint_ids" in self.config.robot.motion:
            self.lower_body_joint_ids = [
                self.simulator.dof_names.index(link)
                for link in self.config.robot.motion.lower_body_joint_ids
            ]
        if "upper_body_joint_ids" in self.config.robot.motion:
            self.upper_body_joint_ids = [
                self.simulator.dof_names.index(link)
                for link in self.config.robot.motion.upper_body_joint_ids
            ]
        if self.config.robot.get("has_key_bodies", False):
            self.key_body_names = self.config.robot.key_bodies
            self.key_body_indices = [
                self.simulator._body_list.index(link) for link in self.key_body_names
            ]
        if "ee_bodylink_ids" in self.config.robot.motion:
            self.ee_bodylink_ids = [
                self.simulator._body_list.index(link)
                for link in self.config.robot.motion.ee_bodylink_ids
            ]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(
                self.config.resample_time_interval_s / self.dt
            )
        self.eval_motion_far_threshold = self.config.termination_scales.get(
            "eval_motion_far_threshold", 0.25
        )

    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(
                    self.simulator._body_list.index(extend_config["parent_name"])
                )
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(
                extend_parent_ids, device=self.device, dtype=torch.long
            )
            self.extend_body_pos_in_parent = (
                torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            )
            self.extend_body_rot_in_parent_wxyz = (
                torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            )
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[
                :, :, [1, 2, 3, 0]
            ]

            self.marker_coords = torch.zeros(
                self.num_envs,
                self.num_bodies + self.num_extend_bodies,
                3,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )  # extend

            self.ref_body_pos_extend = torch.zeros(
                self.num_envs,
                self.num_bodies + self.num_extend_bodies,
                3,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dif_global_body_pos = torch.zeros(
                self.num_envs,
                self.num_bodies + self.num_extend_bodies,
                3,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0

    def _update_waist_dof_curriculum(self):
        if self.use_waist_dof_curriculum:
            if self.log_dict is not None and "mpjpe" in self.log_dict:
                if (
                    self.log_dict["mpjpe"]
                    < self.config.rewards.waist_dof_curriculum_level_down_mpjpe_threshold  # noqa: E501
                ):
                    self.waist_dof_penalty_scale *= (
                        1 - self.config.rewards.waist_dof_curriculum_degree
                    )
                elif (
                    self.log_dict["mpjpe"]
                    > self.config.rewards.waist_dof_curriculum_level_up_mpjpe_threshold  # noqa: E501
                ):
                    self.waist_dof_penalty_scale *= (
                        1 + self.config.rewards.waist_dof_curriculum_degree
                    ).clamp(max=10.0)

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
            if self.use_waist_dof_curriculum:
                if (
                    name in self.config.rewards.reward_waist_dof_penalty_reward_names  # noqa: E501
                ):
                    rew *= self.waist_dof_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(
                self.reward_penalty_scale, dtype=torch.float
            )
            self.log_dict["average_episode_length"] = self.average_episode_length

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

        if self.use_waist_dof_curriculum:
            self.log_dict["waist_dof_penalty_scale"] = torch.tensor(
                self.waist_dof_penalty_scale, dtype=torch.float
            )

    def _init_buffers(self):
        super()._init_buffers()
        self._init_motion_lib()
        self.vr_3point_marker_coords = torch.zeros(
            self.num_envs,
            3,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.realtime_vr_keypoints_pos = torch.zeros(
            3, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(
            3, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # hand, hand, head
        self.motion_ids = self._motion_lib.resample_new_motions(
            self.num_envs, eval=self.is_evaluating
        ).to(self.device)
        self.motion_global_start_frame_ids = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                torch.arange(self.num_envs),
                n_fut_frames=self.n_fut_frames,
                eval=self.is_evaluating,
            ).to(self.device)
        )
        self.motion_global_end_frame_ids = (
            self._motion_lib.cache.cached_motion_global_end_frames.to(self.device)
        )
        self.cur_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.reset_buf_motion_far = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.robot_initial_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.ref_initial_heading_inv_quat = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.joint_far_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.motion_far_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.pos_history_buffer = None
        self.rot_history_buffer = None
        self.ref_pos_history_buffer = None
        self.current_accel = None
        self.ref_body_accel = None
        self.current_ang_accel = None  # Placeholder for angular acceleration

    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        self._resample_motion_frame_ids(
            env_ids
        )  # need to resample before reset root states
        if (
            self.config.termination.terminate_when_motion_far
            and self.config.termination_curriculum.terminate_when_motion_far_curriculum  # noqa: E501
        ):
            self._update_terminate_when_motion_far_curriculum()
        if (
            self.config.termination.terminate_when_joint_far
            and self.config.termination_curriculum.terminate_when_joint_far_curriculum  # noqa: E501
        ):
            self._update_terminate_when_joint_far_curriculum()
        if self.config.get("entropy_curriculum", {}).get(
            "enable_entropy_curriculum", False
        ):
            self._update_entropy_curriculum()

        # Reset joint-far patience counter for the environments being reset
        if self.config.termination.get("terminate_when_joint_far", False):
            self.joint_far_counter[env_ids] = 0

        # Reset motion-far patience counter for the environments being reset
        if self.config.termination.get("terminate_when_motion_far", False):
            self.motion_far_counter[env_ids] = 0

        # Reset history buffers for the environments being reset
        if len(env_ids) > 0:
            # Only proceed if we have the necessary attributes initialized
            if hasattr(self, "_rigid_body_pos_extend") and hasattr(self, "simulator"):
                # Get current positions for the reset environments
                if hasattr(self.simulator, "num_bodies") and hasattr(
                    self, "_rigid_body_pos_extend"
                ):
                    try:
                        current_rigid_pos = torch.cat(
                            [
                                self.simulator._rigid_body_pos,
                                self._rigid_body_pos_extend[
                                    :, self.simulator.num_bodies :
                                ],
                            ],
                            dim=1,
                        )

                        # Initialize history buffers if they exist
                        if (
                            hasattr(self, "pos_history_buffer")
                            and self.pos_history_buffer is not None
                        ):
                            # Initialize all frames with current positions
                            self.pos_history_buffer[env_ids] = (
                                current_rigid_pos[env_ids]
                                .unsqueeze(1)
                                .repeat(1, 3, 1, 1)
                            )

                        if (
                            hasattr(self, "current_accel")
                            and self.current_accel is not None
                        ):
                            self.current_accel[env_ids] = torch.zeros_like(
                                current_rigid_pos[env_ids]
                            )
                    except Exception:
                        pass

            # Reset reference buffers if they exist
            if (
                hasattr(self, "ref_body_pos_t")
                and hasattr(self, "ref_pos_history_buffer")
                and self.ref_pos_history_buffer is not None
            ):
                try:
                    self.ref_pos_history_buffer[env_ids] = (
                        self.ref_body_pos_t[env_ids].unsqueeze(1).repeat(1, 3, 1, 1)
                    )

                    if (
                        hasattr(self, "ref_body_accel")
                        and self.ref_body_accel is not None
                    ):
                        self.ref_body_accel[env_ids] = torch.zeros_like(
                            self.ref_body_pos_t[env_ids]
                        )
                except Exception:
                    pass

    def _update_terminate_when_motion_far_curriculum(self):
        assert (
            self.config.termination.terminate_when_motion_far
            and self.config.termination_curriculum.terminate_when_motion_far_curriculum  # noqa: E501
        )
        if (
            self.average_episode_length
            < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold  # noqa: E501
        ):
            self.terminate_when_motion_far_threshold *= (
                1
                + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree  # noqa: E501
            )
        elif (
            self.average_episode_length
            > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold  # noqa: E501
        ):
            self.terminate_when_motion_far_threshold *= (
                1
                - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree  # noqa: E501
            )
        self.terminate_when_motion_far_threshold = np.clip(
            self.terminate_when_motion_far_threshold,
            self.config.termination_curriculum.terminate_when_motion_far_threshold_min,
            self.config.termination_curriculum.terminate_when_motion_far_threshold_max,
        )

    def _update_terminate_when_joint_far_curriculum(self):
        assert (
            self.config.termination.terminate_when_joint_far
            and self.config.termination_curriculum.terminate_when_joint_far_curriculum  # noqa: E501
        )
        if (
            self.average_episode_length
            < self.config.termination_curriculum.terminate_when_joint_far_curriculum_level_down_threshold  # noqa: E501
        ):
            self.terminate_when_joint_far_threshold *= (
                1
                + self.config.termination_curriculum.terminate_when_joint_far_curriculum_degree  # noqa: E501
            )
        elif (
            self.average_episode_length
            > self.config.termination_curriculum.terminate_when_joint_far_curriculum_level_up_threshold  # noqa: E501
        ):
            self.terminate_when_joint_far_threshold *= (
                1
                - self.config.termination_curriculum.terminate_when_joint_far_curriculum_degree  # noqa: E501
            )
        self.terminate_when_joint_far_threshold = np.clip(
            self.terminate_when_joint_far_threshold,
            self.config.termination_curriculum.terminate_when_joint_far_threshold_min,
            self.config.termination_curriculum.terminate_when_joint_far_threshold_max,
        )

    def _update_entropy_curriculum(self):
        if self.config.get("entropy_curriculum", {}):
            if (
                self.average_episode_length
                > self.config.entropy_curriculum.entropy_curriculum_threshold
            ):
                self.entropy_coef *= (
                    1 - self.config.entropy_curriculum.entropy_curriculum_degree
                )
            self.entropy_coef = np.clip(
                self.entropy_coef,
                self.config.entropy_curriculum.entropy_curriculum_threshold_min,
                self.config.entropy_curriculum.entropy_curriculum_threshold_max,
            )

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training and not self.is_evaluating:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _check_termination(self):
        super()._check_termination()
        # logger.info(f"{torch.abs(self.dif_joint_angles).mean()}")
        if self.config.termination.terminate_when_motion_far:
            self.motion_far_type = self.config.get("motion_far_type", "max")

            if not self.is_evaluating:
                if self.motion_far_type == "mean":
                    mean_dist = torch.mean(
                        torch.norm(
                            self.dif_global_body_pos[:, self.reset_body_indices],
                            dim=-1,
                        ),
                        dim=-1,
                    )
                    self.tmp_mean_dist = mean_dist
                    motion_far_now = (
                        mean_dist > self.terminate_when_motion_far_threshold
                    )
                elif self.motion_far_type == "max":
                    motion_far_now = torch.any(
                        torch.norm(
                            self.dif_global_body_pos[:, self.reset_body_indices],
                            dim=-1,
                        )
                        > self.terminate_when_motion_far_threshold,
                        dim=-1,
                    )
                else:
                    raise ValueError(f"Unknown motion far type: {self.motion_far_type}")
            else:
                mean_dist = torch.mean(
                    torch.norm(
                        self.dif_global_body_pos[:, self.reset_body_indices],
                        dim=-1,
                    ),
                    dim=-1,
                )
                self.tmp_mean_dist = mean_dist
                motion_far_now = mean_dist > self.eval_motion_far_threshold

            # Update patience counter for motion_far
            self.motion_far_counter = torch.where(
                motion_far_now,
                self.motion_far_counter + 1,
                torch.zeros_like(self.motion_far_counter),
            )

            reset_buf_motion_far = (
                self.motion_far_counter >= self.terminate_when_motion_far_patience_steps
            )

            self.reset_buf_motion_far = reset_buf_motion_far

            self.reset_buf |= reset_buf_motion_far * (self.episode_length_buf > 5)
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:  # noqa: E501
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(
                    self.terminate_when_motion_far_threshold,
                    dtype=torch.float,
                )
            self.extras["reset_buf_motion_far"] = reset_buf_motion_far
        if self.config.termination.get("terminate_when_joint_far", False):
            joint_far_now = (
                torch.abs(self.dif_joint_angles).mean(dim=-1)
                > self.terminate_when_joint_far_threshold
            )

            self.joint_far_counter = torch.where(
                joint_far_now,
                self.joint_far_counter + 1,
                torch.zeros_like(self.joint_far_counter),
            )

            # Environment terminates when counter ≥ patience steps
            reset_buf_joint_far = (
                self.joint_far_counter >= self.terminate_when_joint_far_patience_steps
            )

            self.reset_buf |= reset_buf_joint_far

            # log current joint far threshold and patience for monitoring
            if self.config.termination_curriculum.terminate_when_joint_far_curriculum:  # noqa: E501
                self.log_dict["terminate_when_joint_far_threshold"] = torch.tensor(
                    self.terminate_when_joint_far_threshold,
                    dtype=torch.float,
                )
        if self.config.termination.get("terminate_when_ee_z_far", False):
            # terminate the episode when the ee bodylinks' z error exceeds
            # the error threshold
            z_threshold = self.config.termination_scales.get(
                "terminate_when_ee_z_far_threshold", 0.25
            )
            assert self.ee_bodylink_ids is not None, (
                "ee_bodylink_ids not found in the config"
            )
            ee_z_error = torch.abs(
                self.dif_global_body_pos[:, self.ee_bodylink_ids, -1]
            )
            reset_buf_ee_z_far = torch.any(ee_z_error > z_threshold, dim=-1)
            self.reset_buf |= reset_buf_ee_z_far

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_global_frame_ids = (
                self.episode_length_buf + self.motion_global_start_frame_ids
            )
            self.time_out_buf |= current_global_frame_ids >= (
                self.motion_global_end_frame_ids - 1 - self.n_fut_frames
            )

    def _resample_motion_frame_ids(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_global_start_frame_ids[env_ids] = (
                torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
                + self.motion_global_start_frame_ids[env_ids]
            )
        else:
            self.motion_global_start_frame_ids[env_ids] = (
                self._motion_lib.cache.sample_cached_global_start_frames(
                    env_ids,
                    n_fut_frames=self.n_fut_frames,
                    eval=self.is_evaluating,
                ).to(self.device)
            )

    def resample_motion(self):
        self.motion_ids = (
            self._motion_lib.resample_new_motions(
                self.num_envs, eval=self.is_evaluating
            )
            .to(self.device)
            .clone()
        )
        self.motion_global_start_frame_ids = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                torch.arange(self.num_envs),
                n_fut_frames=self.n_fut_frames,
                eval=self.is_evaluating,
            ).to(self.device)
        )
        self.motion_global_end_frame_ids = (
            self._motion_lib.cache.cached_motion_global_end_frames.to(self.device)
        )
        self.reset_all()

    def resample_motion_eval(self):
        is_last_eval_batch = self._motion_lib.load_next_eval_batch(self.num_envs)
        self.motion_global_start_frame_ids = (
            self._motion_lib.cache.sample_cached_global_start_frames(
                torch.arange(self.num_envs),
                n_fut_frames=self.n_fut_frames,
                eval=self.is_evaluating,
            ).to(self.device)
        )
        self.motion_global_end_frame_ids = (
            self._motion_lib.cache.cached_motion_global_end_frames.to(self.device)
        )
        self.reset_all()
        return is_last_eval_batch

    def _log_motion_tracking_info(self):
        # Logging uses self.dif_... tensors (state_t vs ref_t)
        whole_body_diff = self.dif_global_body_pos
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        whole_body_joint_pos_diff = self.dif_joint_angles
        upper_body_joint_pos_diff = self.dif_joint_angles[:, self.upper_body_joint_ids]
        lower_body_joint_pos_diff = self.dif_joint_angles[:, self.lower_body_joint_ids]
        if self.waist_dof_indices is not None:
            waist_roll_pitch_diff = self.dif_joint_angles[:, self.waist_dof_indices]
        else:
            waist_roll_pitch_diff = torch.zeros_like(whole_body_joint_pos_diff)

        whole_body_diff_norm = whole_body_diff.norm(dim=-1)
        upper_body_diff_norm = upper_body_diff.norm(dim=-1)
        lower_body_diff_norm = lower_body_diff.norm(dim=-1)
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1)
        whole_body_joint_pos_diff = whole_body_joint_pos_diff.abs()
        upper_body_joint_pos_diff = upper_body_joint_pos_diff.abs()
        lower_body_joint_pos_diff = lower_body_joint_pos_diff.abs()
        waist_roll_pitch_diff = waist_roll_pitch_diff.abs()

        rel_root_lin_vel_mae = self.dif_local_root_lin_vel_t.abs().mean()
        rel_root_ang_vel_mae = self.dif_local_root_ang_vel_t.abs().mean()

        base_roll_mae = self.dif_base_roll_t.abs().mean()
        base_pitch_mae = self.dif_base_pitch_t.abs().mean()
        base_yaw_mae = self.dif_base_yaw_t.abs().mean()

        rel_body_pos_mae = self.dif_local_body_pos_t.abs().mean()
        rel_body_rot_tannorm_mae = self.dif_local_body_rot_tannorm.abs().mean()
        rel_body_vel_mae = self.dif_local_body_vel_t.abs().mean()
        rel_body_ang_vel_mae = self.dif_local_body_ang_vel_t.abs().mean()
        rel_body_vr3point_mae = (
            self.dif_local_body_pos_t[:, self.motion_tracking_id, :].abs().mean()
        )

        root_rel_body_pos_mae = self.dif_root_rel_body_pos_t.abs().mean()
        root_rel_body_rot_tannorm_mae = self.dif_root_rel_body_rot_tannorm.abs().mean()
        root_rel_body_vel_mae = self.dif_root_rel_body_vel_t.abs().mean()
        root_rel_body_ang_vel_mae = self.dif_root_rel_body_ang_vel_t.abs().mean()
        root_rel_vr3point_mae = (
            self.dif_root_rel_body_pos_t[:, self.motion_tracking_id, :].abs().mean()
        )

        self.log_dict_nonreduced["mpkpe"] = whole_body_diff_norm.mean(-1)
        self.log_dict_nonreduced["mpkpe_upper"] = upper_body_diff_norm.mean(-1)
        self.log_dict_nonreduced["mpkpe_lower"] = lower_body_diff_norm.mean(-1)
        self.log_dict_nonreduced["mpkpe_vr"] = vr_3point_diff_norm.mean(-1)
        self.log_dict_nonreduced["mpjpe"] = whole_body_joint_pos_diff.mean(-1)
        self.log_dict_nonreduced["mpjpe_upper"] = upper_body_joint_pos_diff.mean(-1)
        self.log_dict_nonreduced["mpjpe_lower"] = lower_body_joint_pos_diff.mean(-1)
        self.log_dict_nonreduced["mpjpe_waist_roll_pitch"] = waist_roll_pitch_diff.mean(
            -1
        )

        self.log_dict["mpkpe"] = whole_body_diff_norm.mean()
        self.log_dict["mpkpe_upper"] = upper_body_diff_norm.mean()
        self.log_dict["mpkpe_lower"] = lower_body_diff_norm.mean()
        self.log_dict["mpkpe_vr"] = vr_3point_diff_norm.mean()
        self.log_dict["mpjpe"] = whole_body_joint_pos_diff.mean()
        self.log_dict["mpjpe_upper"] = upper_body_joint_pos_diff.mean()
        self.log_dict["mpjpe_lower"] = lower_body_joint_pos_diff.mean()
        self.log_dict["mpjpe_waist_roll_pitch"] = waist_roll_pitch_diff.mean()

        self.log_dict["rel_root_lin_vel_mae"] = rel_root_lin_vel_mae
        self.log_dict["rel_root_ang_vel_mae"] = rel_root_ang_vel_mae

        self.log_dict["base_roll_mae"] = base_roll_mae
        self.log_dict["base_pitch_mae"] = base_pitch_mae
        self.log_dict["base_yaw_mae"] = base_yaw_mae

        self.log_dict["rel_body_pos_mae"] = rel_body_pos_mae
        self.log_dict["rel_body_rot_tannorm_mae"] = rel_body_rot_tannorm_mae
        self.log_dict["rel_body_vel_mae"] = rel_body_vel_mae
        self.log_dict["rel_body_ang_vel_mae"] = rel_body_ang_vel_mae
        self.log_dict["rel_body_vr3point_mae"] = rel_body_vr3point_mae

        self.log_dict["root_rel_body_pos_mae"] = root_rel_body_pos_mae
        self.log_dict["root_rel_body_rot_tannorm_mae"] = root_rel_body_rot_tannorm_mae
        self.log_dict["root_rel_body_vel_mae"] = root_rel_body_vel_mae
        self.log_dict["root_rel_body_ang_vel_mae"] = root_rel_body_ang_vel_mae
        self.log_dict["root_rel_vr3point_mae"] = root_rel_vr3point_mae

    def _log_motion_tracking_holomotion_metrics(self):
        current_pos = (
            self._rigid_body_pos_extend.detach().cpu().numpy()
        )  # [num_envs, num_bodies, 3]
        ref_pos = (
            self.ref_body_pos_t.detach().cpu().numpy()
        )  # [num_envs, num_bodies, 3]

        # Get current body rotations and reference rotations (if needed)
        current_rot = (
            self._rigid_body_rot_extend.detach().cpu().numpy()
        )  # [num_envs, num_bodies, 4]
        ref_rot = (
            self.ref_body_rot_t.detach().cpu().numpy()
        )  # [num_envs, num_bodies, 4]
        pred_pos_all = [current_pos[i : i + 1] for i in range(self.num_envs)]
        gt_pos_all = [ref_pos[i : i + 1] for i in range(self.num_envs)]
        pred_rot_all = [current_rot[i : i + 1] for i in range(self.num_envs)]
        gt_rot_all = [ref_rot[i : i + 1] for i in range(self.num_envs)]

        pred_vel = None
        gt_vel = None
        pred_accel = None
        gt_accel = None

        has_pos_history = (
            hasattr(self, "pos_history_buffer") and self.pos_history_buffer is not None
        )
        has_ref_pos_history = (
            hasattr(self, "ref_pos_history_buffer")
            and self.ref_pos_history_buffer is not None
        )
        has_current_accel_here = (
            hasattr(self, "current_accel") and self.current_accel is not None
        )
        has_ref_accel_here = (
            hasattr(self, "ref_body_accel") and self.ref_body_accel is not None
        )

        if (
            has_pos_history
            and has_ref_pos_history
            and has_current_accel_here
            and has_ref_accel_here
        ):
            # Check if we have enough history frames
            pos_frames = self.pos_history_buffer.shape[1] if has_pos_history else 0
            ref_frames = (
                self.ref_pos_history_buffer.shape[1] if has_ref_pos_history else 0
            )

            if pos_frames >= 3 and ref_frames >= 3:
                try:
                    # Extract relevant data from history buffers
                    # Note: We need to convert from torch to numpy

                    # Current positions (all frames)
                    pred_pos_history = self.pos_history_buffer.cpu().numpy()
                    gt_pos_history = self.ref_pos_history_buffer.cpu().numpy()

                    # Compute velocities from history (frames t-1 to t)
                    pred_vel = []
                    gt_vel = []
                    for i in range(self.num_envs):
                        pred_vel.append(
                            pred_pos_history[i, 1:] - pred_pos_history[i, :-1]
                        )
                        gt_vel.append(gt_pos_history[i, 1:] - gt_pos_history[i, :-1])

                    pred_accel = []
                    gt_accel = []
                    for i in range(self.num_envs):
                        pred_accel.append(
                            self.current_accel[i].unsqueeze(0).cpu().numpy()
                        )
                        gt_accel.append(
                            self.ref_body_accel[i].unsqueeze(0).cpu().numpy()
                        )
                except Exception:
                    pred_vel = None
                    gt_vel = None
                    pred_accel = None
                    gt_accel = None
            else:
                pred_vel = None
                gt_vel = None
                pred_accel = None
                gt_accel = None
        else:
            pred_vel = None
            gt_vel = None
            pred_accel = None
            gt_accel = None

        metrics = compute_metrics_lite(
            pred_pos_all,
            gt_pos_all,
            pred_rot_all,
            gt_rot_all,
            root_idx=0,  # Assuming root is at index 0
            use_tqdm=False,  # No need for progress bar
            concatenate=True,  # Concatenate results
            pred_vel=pred_vel,
            gt_vel=gt_vel,
            pred_accel=pred_accel,
            gt_accel=gt_accel,
        )

        def safe_mean(arr):
            if isinstance(arr, np.ndarray) and arr.size > 0:
                # Filter out nan values before computing mean
                clean_arr = arr[~np.isnan(arr)]
                if clean_arr.size > 0:
                    return float(np.mean(clean_arr))
            return 0.0

        # Process each metric with safe handling
        mpjpe_g = torch.tensor(
            safe_mean(metrics.get("mpjpe_g", np.array([]))), device=self.device
        )
        mpjpe_l = torch.tensor(
            safe_mean(metrics.get("mpjpe_l", np.array([]))), device=self.device
        )
        mpjpe_pa = torch.tensor(
            safe_mean(metrics.get("mpjpe_pa", np.array([]))),
            device=self.device,
        )
        vel_dist = torch.tensor(
            safe_mean(metrics.get("vel_dist", np.array([]))),
            device=self.device,
        )

        if (
            hasattr(self, "current_accel")
            and self.current_accel is not None
            and hasattr(self, "ref_body_accel")
            and self.ref_body_accel is not None
            and self.current_accel.shape == self.ref_body_accel.shape
        ):
            try:
                accel_diff = self.current_accel - self.ref_body_accel
                accel_dist = torch.norm(accel_diff, dim=-1).mean() * 1000
            except Exception:
                # Fall back to metrics if there's any error
                accel_dist = torch.tensor(
                    safe_mean(metrics.get("accel_dist", np.array([]))),
                    device=self.device,
                )
        else:
            # Fall back to metrics if our buffers aren't ready yet
            accel_dist = torch.tensor(
                safe_mean(metrics.get("accel_dist", np.array([]))),
                device=self.device,
            )

        # Calculate upper and lower body errors using the original method
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        upper_body_joints_dist = torch.norm(upper_body_diff, dim=-1).mean()
        lower_body_joints_dist = torch.norm(lower_body_diff, dim=-1).mean()

        # Calculate root errors using the original method
        root_height_error = torch.abs(self.dif_global_body_pos[:, 0, 2]).mean()
        root_vel_error = torch.norm(self.dif_local_root_lin_vel_t, dim=-1).mean()
        root_r_error = torch.abs(self.dif_base_roll_t).mean()
        root_p_error = torch.abs(self.dif_base_pitch_t).mean()
        root_y_error = torch.abs(self.dif_base_yaw_t).mean()

        default_values = torch.zeros(self.num_envs, device=self.device)

        # Add acceleration difference to per-environment metrics if available
        has_current_accel = (
            hasattr(self, "current_accel") and self.current_accel is not None
        )
        has_ref_accel = (
            hasattr(self, "ref_body_accel") and self.ref_body_accel is not None
        )
        shapes_match = False
        if has_current_accel and has_ref_accel:
            shapes_match = self.current_accel.shape == self.ref_body_accel.shape

        if has_current_accel and has_ref_accel and shapes_match:
            try:
                accel_diff = self.current_accel - self.ref_body_accel
                accel_norm = torch.norm(accel_diff, dim=-1).mean(-1) * 1000
                # Apply *1000 scaling to match compute_metrics_lite
                self.log_dict_nonreduced_holomotion["accel_dist"] = accel_norm
            except Exception:
                pass

        # Process mpjpe_g
        if "mpjpe_g" in metrics and metrics["mpjpe_g"].size > 0:
            try:
                # Mean over joints dimension (axis=1) for each environment
                if metrics["mpjpe_g"].ndim > 1:
                    mpjpe_g_per_env = np.mean(metrics["mpjpe_g"], axis=1)
                else:
                    # If it's 1D, make sure it's properly shaped
                    mpjpe_g_per_env = np.tile(metrics["mpjpe_g"], (self.num_envs,))[
                        : self.num_envs
                    ]

                # Ensure we have exactly num_envs values
                if len(mpjpe_g_per_env) == self.num_envs:
                    self.log_dict_nonreduced_holomotion["mpjpe_g"] = torch.from_numpy(
                        mpjpe_g_per_env
                    ).to(self.device)
                else:
                    # Pad or truncate to match num_envs
                    self.log_dict_nonreduced_holomotion["mpjpe_g"] = torch.from_numpy(
                        np.pad(
                            mpjpe_g_per_env,
                            (
                                0,
                                max(0, self.num_envs - len(mpjpe_g_per_env)),
                            ),
                        )
                    ).to(self.device)[: self.num_envs]
            except Exception:
                self.log_dict_nonreduced_holomotion["mpjpe_g"] = default_values
        else:
            # Calculate directly from available tensors
            self.log_dict_nonreduced_holomotion["mpjpe_g"] = torch.norm(
                self.dif_global_body_pos, dim=-1
            ).mean(-1)

        # Process mpjpe_l
        if "mpjpe_l" in metrics and metrics["mpjpe_l"].size > 0:
            try:
                if metrics["mpjpe_l"].ndim > 1:
                    mpjpe_l_per_env = np.mean(metrics["mpjpe_l"], axis=1)
                else:
                    mpjpe_l_per_env = np.tile(metrics["mpjpe_l"], (self.num_envs,))[
                        : self.num_envs
                    ]

                if len(mpjpe_l_per_env) == self.num_envs:
                    self.log_dict_nonreduced_holomotion["mpjpe_l"] = torch.from_numpy(
                        mpjpe_l_per_env
                    ).to(self.device)
                else:
                    self.log_dict_nonreduced_holomotion["mpjpe_l"] = torch.from_numpy(
                        np.pad(
                            mpjpe_l_per_env,
                            (
                                0,
                                max(0, self.num_envs - len(mpjpe_l_per_env)),
                            ),
                        )
                    ).to(self.device)[: self.num_envs]
            except Exception:
                self.log_dict_nonreduced_holomotion["mpjpe_l"] = torch.abs(
                    self.dif_joint_angles
                ).mean(-1)
        else:
            self.log_dict_nonreduced_holomotion["mpjpe_l"] = torch.abs(
                self.dif_joint_angles
            ).mean(-1)

        # Process mpjpe_pa
        if "mpjpe_pa" in metrics and metrics["mpjpe_pa"].size > 0:
            try:
                if metrics["mpjpe_pa"].ndim > 1:
                    mpjpe_pa_per_env = np.mean(metrics["mpjpe_pa"], axis=1)
                else:
                    mpjpe_pa_per_env = np.tile(metrics["mpjpe_pa"], (self.num_envs,))[
                        : self.num_envs
                    ]

                if len(mpjpe_pa_per_env) == self.num_envs:
                    self.log_dict_nonreduced_holomotion["mpjpe_pa"] = torch.from_numpy(
                        mpjpe_pa_per_env
                    ).to(self.device)
                else:
                    self.log_dict_nonreduced_holomotion["mpjpe_pa"] = torch.from_numpy(
                        np.pad(
                            mpjpe_pa_per_env,
                            (
                                0,
                                max(
                                    self.num_envs - len(mpjpe_pa_per_env),
                                ),
                            ),
                        )
                    ).to(self.device)[: self.num_envs]
            except Exception:
                self.log_dict_nonreduced_holomotion["mpjpe_pa"] = (
                    torch.norm(self.dif_global_body_pos, dim=-1).mean(-1) * 0.8
                )
        else:
            self.log_dict_nonreduced_holomotion["mpjpe_pa"] = (
                torch.norm(self.dif_global_body_pos, dim=-1).mean(-1) * 0.8
            )

        # Original metrics that are already calculated per environment
        self.log_dict_nonreduced_holomotion["upper_body_joints_dist"] = torch.norm(
            upper_body_diff, dim=-1
        ).mean(-1)
        self.log_dict_nonreduced_holomotion["lower_body_joints_dist"] = torch.norm(
            lower_body_diff, dim=-1
        ).mean(-1)
        self.log_dict_nonreduced_holomotion["root_height_error"] = torch.abs(
            self.dif_global_body_pos[:, 0, 2]
        )
        self.log_dict_nonreduced_holomotion["root_vel_error"] = torch.norm(
            self.dif_local_root_lin_vel_t, dim=-1
        )
        self.log_dict_nonreduced_holomotion["root_r_error"] = torch.abs(
            self.dif_base_roll_t
        )
        self.log_dict_nonreduced_holomotion["root_p_error"] = torch.abs(
            self.dif_base_pitch_t
        )
        self.log_dict_nonreduced_holomotion["root_y_error"] = torch.abs(
            self.dif_base_yaw_t
        )

        # Process vel_dist
        if "vel_dist" in metrics and metrics["vel_dist"].size > 0:
            try:
                if metrics["vel_dist"].ndim > 1:
                    vel_dist_per_env = np.mean(metrics["vel_dist"], axis=1)
                else:
                    vel_dist_per_env = np.tile(metrics["vel_dist"], (self.num_envs,))[
                        : self.num_envs
                    ]

                if len(vel_dist_per_env) == self.num_envs:
                    self.log_dict_nonreduced_holomotion["vel_dist"] = torch.from_numpy(
                        vel_dist_per_env
                    ).to(self.device)
                else:
                    self.log_dict_nonreduced_holomotion["vel_dist"] = torch.from_numpy(
                        np.pad(
                            vel_dist_per_env,
                            (
                                0,
                                max(
                                    self.num_envs - len(vel_dist_per_env),
                                ),
                            ),
                        )
                    ).to(self.device)[: self.num_envs]
            except Exception:
                vel_diff = self.dif_global_body_vel
                self.log_dict_nonreduced_holomotion["vel_dist"] = torch.norm(
                    vel_diff, dim=-1
                ).mean(-1)
        else:
            vel_diff = self.dif_global_body_vel
            self.log_dict_nonreduced_holomotion["vel_dist"] = torch.norm(
                vel_diff, dim=-1
            ).mean(-1)

        # Store aggregated metrics
        self.log_dict_holomotion["mpjpe_g"] = mpjpe_g
        self.log_dict_holomotion["mpjpe_l"] = mpjpe_l
        self.log_dict_holomotion["mpjpe_pa"] = mpjpe_pa
        self.log_dict_holomotion["accel_dist"] = accel_dist
        self.log_dict_holomotion["vel_dist"] = vel_dist

        self.log_dict_holomotion["upper_body_joints_dist"] = upper_body_joints_dist
        self.log_dict_holomotion["lower_body_joints_dist"] = lower_body_joints_dist
        self.log_dict_holomotion["root_r_error"] = root_r_error
        self.log_dict_holomotion["root_p_error"] = root_p_error
        self.log_dict_holomotion["root_y_error"] = root_y_error
        self.log_dict_holomotion["root_vel_error"] = root_vel_error
        self.log_dict_holomotion["root_height_error"] = root_height_error

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            for pos_id, pos_joint in enumerate(
                self.marker_coords[env_id]
            ):  # idx 0 torso (duplicate with 11)
                if self.config.robot.motion.visualization.customize_color:
                    color_inner = self.config.robot.motion.visualization.marker_joint_colors[  # noqa: E501
                        pos_id
                        % len(
                            self.config.robot.motion.visualization.marker_joint_colors
                        )
                    ]
                else:
                    color_inner = (0.3, 0.3, 0.3)
                color_inner = tuple(color_inner)
                self.simulator.draw_sphere(pos_joint, 0.03, color_inner, env_id, pos_id)

    def _reset_root_states(self, env_ids):
        offset = self.env_origins
        motion_frame_ids = self.episode_length_buf + self.motion_global_start_frame_ids
        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=offset
        )

        root_pos_noise = (
            self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
        )
        root_rot_noise = (
            self.config.init_noise_scale.root_rot * self.config.noise_to_initial_level
        )
        root_vel_noise = (
            self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
        )
        root_ang_vel_noise = (
            self.config.init_noise_scale.root_ang_vel
            * self.config.noise_to_initial_level
        )

        root_pos = motion_res["root_pos"][:, 0].to(self.device)[env_ids]
        root_rot = motion_res["root_rot"][:, 0].to(self.device)[env_ids]
        root_vel = motion_res["root_vel"][:, 0].to(self.device)[env_ids]
        root_ang_vel = motion_res["root_ang_vel"][:, 0].to(self.device)[env_ids]

        self.simulator.robot_root_states[env_ids, :3] = (
            root_pos + torch.randn_like(root_pos) * root_pos_noise
        )
        if self.config.simulator.config.name == "isaacgym":
            self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(
                self.small_random_quaternions(root_rot.shape[0], root_rot_noise),
                root_rot,
                w_last=True,
            )
        elif self.config.simulator.config.name == "isaacsim":
            self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(
                quat_mul(
                    self.small_random_quaternions(root_rot.shape[0], root_rot_noise),
                    root_rot,
                    w_last=True,
                )
            )
        elif self.config.simulator.config.name == "genesis":
            self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(
                self.small_random_quaternions(root_rot.shape[0], root_rot_noise),
                root_rot,
                w_last=True,
            )
        elif self.config.simulator.config.name == "mujoco":
            self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(
                self.small_random_quaternions(root_rot.shape[0], root_rot_noise),
                root_rot,
                w_last=True,
            )
        else:
            raise NotImplementedError
        self.simulator.robot_root_states[env_ids, 7:10] = (
            root_vel + torch.randn_like(root_vel) * root_vel_noise
        )
        self.simulator.robot_root_states[env_ids, 10:13] = (
            root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
        )

        robot_actual_root_rot_quat_raw = self.simulator.robot_root_states[
            env_ids, 3:7
        ].clone()
        if self.config.simulator.config.name == "isaacsim":
            robot_actual_root_rot_quat_xyzw = wxyz_to_xyzw(
                robot_actual_root_rot_quat_raw
            )
        else:
            robot_actual_root_rot_quat_xyzw = robot_actual_root_rot_quat_raw

        robot_actual_root_heading_quat_inv = calc_heading_quat_inv(
            robot_actual_root_rot_quat_xyzw, w_last=True
        )
        self.robot_initial_heading_inv_quat[env_ids] = (
            robot_actual_root_heading_quat_inv
        )

        # calculate and keep the ref_to_initial_heading_aligned_frame_quat here
        # root_rot is already xyzw from the motion library
        ref_root_rot_quat_xyzw = root_rot
        ref_root_heading_quat_inv = calc_heading_quat_inv(
            ref_root_rot_quat_xyzw, w_last=True
        )
        self.ref_initial_heading_inv_quat[env_ids] = ref_root_heading_quat_inv

    def small_random_quaternions(self, n, max_angle):
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
        angles = max_angle * torch.rand((n, 1), device=self.device)

        # Convert angle-axis to quaternion
        sin_half_angle = torch.sin(angles / 2)
        cos_half_angle = torch.cos(angles / 2)

        q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)
        return q

    def _reset_dofs(self, env_ids):
        offset = self.env_origins
        motion_frame_ids = self.episode_length_buf + self.motion_global_start_frame_ids
        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=offset
        )

        dof_pos_noise = (
            self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        )
        dof_vel_noise = (
            self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        )
        dof_pos = motion_res["dof_pos"][:, 0].to(self.device)[env_ids]
        dof_vel = motion_res["dof_vel"][:, 0].to(self.device)[env_ids]
        self.simulator.dof_pos[env_ids] = (
            dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        )
        self.simulator.dof_vel[env_ids] = (
            dof_vel + torch.randn_like(dof_vel) * dof_vel_noise
        )

        if self.config.get("direct_state_control", False):
            self.simulator.dof_pos[env_ids] = dof_pos
            self.simulator.dof_vel[env_ids] = dof_vel

    def _physics_step(self):
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            if self.config.get("direct_state_control", False):
                self._apply_direct_state_in_physics_step()
            else:
                self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_direct_state_in_physics_step(self):
        offset = self.env_origins
        motion_frame_ids = self.episode_length_buf + self.motion_global_start_frame_ids
        motion_res = self._motion_lib.cache.get_motion_state(
            motion_frame_ids, global_offset=offset
        )

        self.simulator.dof_state.view(self.num_envs, -1, 2)[:, :, 0] = motion_res[
            "dof_pos"
        ].to(self.device)
        self.simulator.dof_state.view(self.num_envs, -1, 2)[:, :, 1] = (
            motion_res["dof_vel"] * 0
        ).to(self.device)

        # Update simulator state for DOFs
        self.simulator.set_dof_state_tensor(
            torch.arange(self.num_envs, device=self.device),
            self.simulator.dof_state,
        )

        # Set root state to match reference motion
        root_states = self.simulator.robot_root_states.clone()

        # Set position from motion reference
        root_states[:, 0:3] = motion_res["root_pos"].to(self.device)

        # Set rotation from motion reference
        if self.config.simulator.config.name == "isaacgym":
            root_states[:, 3:7] = motion_res["root_rot"].to(self.device)  # xyzw format
        elif self.config.simulator.config.name == "isaacsim":
            # Convert from xyzw to wxyz if needed
            root_states[:, 3:7] = xyzw_to_wxyz(motion_res["root_rot"].to(self.device))
        else:
            root_states[:, 3:7] = motion_res["root_rot"].to(self.device)  # xyzw format

        # Set velocities from motion reference
        root_states[:, 7:10] = motion_res["root_vel"].to(self.device)
        root_states[:, 10:13] = motion_res["root_ang_vel"].to(self.device)

        # Update the root state in the simulator
        self.simulator.set_actor_root_state_tensor(
            torch.arange(self.num_envs, device=self.device), root_states
        )

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()
        if self.is_evaluating:
            self._pre_eval_compute_observations_callback()
        else:
            self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
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
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

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

    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(
                self.config.robot.motion.visualization.marker_joint_colors
            )
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(
                self.config.robot.motion.visualization.marker_joint_colors
            )
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass

    def _pre_compute_observations_callback(self):
        """Prepare shared variables for observations calculation."""
        super()._pre_compute_observations_callback()

        origin_global_offset = self.env_origins
        self.FT = self.config.obs.get("n_fut_frames", 0)
        self.target_fps = self.config.obs.get("target_fps", 50)
        assert self.FT >= 0, "Config obs.n_fut_frames cannot be negative."

        # Calculate heading rotation once based on current state t
        self.cur_heading_inv_quat = calc_heading_quat_inv(
            self.base_quat,
            w_last=True,
        )

        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(
                -1, 4
            ),
            self.extend_body_pos_in_parent.reshape(-1, 3),
        )
        extend_curr_pos = (
            rotated_pos_in_parent.view(self.num_envs, -1, 3)
            + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        )
        self._rigid_body_pos_extend = torch.cat(
            [self.simulator._rigid_body_pos, extend_curr_pos], dim=1
        )

        ################### EXTEND Rigid body Rotation #####################
        extend_curr_rot = quat_mul(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(
                -1, 4
            ),
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat(
            [self.simulator._rigid_body_rot, extend_curr_rot], dim=1
        )

        #### EXTEND Rigid Body Angular Velocity ############
        self._rigid_body_ang_vel_extend = torch.cat(
            [
                self.simulator._rigid_body_ang_vel,
                self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids],
            ],
            dim=1,
        )

        ###### EXTEND Rigid Body Linear Velocity #########
        parent_ang_vel_global = self.simulator._rigid_body_ang_vel[
            :, self.extend_body_parent_ids
        ]
        angular_velocity_contribution = torch.cross(
            parent_ang_vel_global,
            rotated_pos_in_parent.view(self.num_envs, -1, 3),
            dim=2,
        )
        extend_curr_vel = self.simulator._rigid_body_vel[
            :, self.extend_body_parent_ids
        ] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat(
            [self.simulator._rigid_body_vel, extend_curr_vel], dim=1
        )

        self.num_total_bodies = self._rigid_body_pos_extend.shape[1]
        self.num_dofs = self.simulator.num_dof

        # --- Get frame IDs ---
        self.motion_global_frame_ids_t = (
            self.episode_length_buf + self.motion_global_start_frame_ids
        )

        # --- Fetch reference motion sequence t, t+1, ..., t+n ---
        # Use n_fut_frames=n_fut_frames_config to get sequence of t and n
        # future frames
        # Returns shape [B, n_fut_frames_config + 1, ...]
        motion_res_seq = self._motion_lib.cache.get_motion_state(
            self.motion_global_frame_ids_t,
            global_offset=origin_global_offset,
            n_fut_frames=self.FT,  # Fetch frame t and n future frames
            target_fps=self.target_fps,
        )

        # --- Extract frame t data for Reward calculation ---
        # Slice time dimension at index 0 for current reference frame t
        self.ref_body_pos_t = motion_res_seq["rg_pos_t"][:, 0, ...].to(self.device)
        self.ref_body_rot_t = motion_res_seq["rg_rot_t"][:, 0, ...].to(self.device)
        self.ref_body_vel_t = motion_res_seq["body_vel_t"][:, 0, ...].to(self.device)
        self.ref_body_ang_vel_t = motion_res_seq["body_ang_vel_t"][:, 0, ...].to(
            self.device
        )
        self.ref_joint_pos_t = motion_res_seq["dof_pos"][:, 0, ...].to(self.device)
        self.ref_joint_vel_t = motion_res_seq["dof_vel"][:, 0, ...].to(self.device)

        self.ref_root_global_lin_vel_t = motion_res_seq["root_vel"][:, 0, ...].to(
            self.device
        )  # [B, 3]
        self.ref_root_global_ang_vel_t = motion_res_seq["root_ang_vel"][:, 0, ...].to(
            self.device
        )  # [B, 3]
        self.ref_root_global_pos_t = motion_res_seq["root_pos"][:, 0, ...].to(
            self.device
        )  # [B, 3]
        self.ref_root_global_rot_quat_t = motion_res_seq["root_rot"][:, 0, ...].to(
            self.device
        )  # [B, 4]

        self.ref_body_pos_fut = motion_res_seq["rg_pos_t"][:, 1:, ...].to(self.device)
        self.ref_body_rot_fut = motion_res_seq["rg_rot_t"][:, 1:, ...].to(self.device)
        self.ref_body_vel_fut = motion_res_seq["body_vel_t"][:, 1:, ...].to(self.device)
        self.ref_body_ang_vel_fut = motion_res_seq["body_ang_vel_t"][:, 1:, ...].to(
            self.device
        )
        self.ref_dof_pos_fut = motion_res_seq["dof_pos"][:, 1:, ...].to(self.device)
        self.ref_dof_vel_fut = motion_res_seq["dof_vel"][:, 1:, ...].to(self.device)
        self.ref_base_global_lin_vel_fut = motion_res_seq["root_vel"][:, 1:, ...].to(
            self.device
        )
        self.ref_base_global_ang_vel_fut = motion_res_seq["root_ang_vel"][
            :, 1:, ...
        ].to(self.device)
        self.ref_base_global_rot_quat_fut = motion_res_seq["root_rot"][:, 1:, ...].to(
            self.device
        )
        self.ref_base_global_pos_fut = motion_res_seq["root_pos"][:, 1:, ...].to(
            self.device
        )
        self.ref_fut_valid_mask = motion_res_seq["valid_frame_flag"][:, 1:, ...].to(
            self.device
        )

        # Calculate inverse heading quaternion for the reference motion at
        # time t
        # This will be used for transforming reference velocities to its
        # heading-aligned frame
        self.cur_ref_heading_inv_quat = calc_heading_quat_inv(
            self.ref_root_global_rot_quat_t, w_last=True
        )

        self.ref_base_rel_lin_vel_t = quat_rotate(
            self.cur_ref_heading_inv_quat,
            self.ref_root_global_lin_vel_t,
            w_last=True,
        )  # [B, 3], now in reference\'s heading-aligned frame
        self.ref_base_rel_ang_vel_t = quat_rotate(
            self.cur_ref_heading_inv_quat,
            self.ref_root_global_ang_vel_t,
            w_last=True,
        )  # [B, 3], now in reference\'s heading-aligned frame

        self.ref_base_roll_t, self.ref_base_pitch_t, _ = get_euler_xyz(
            self.ref_root_global_rot_quat_t,
            w_last=True,
        )  # [B, 3]
        _, _, self.ref_base_yaw_t = get_euler_xyz(
            quat_mul(
                self.ref_initial_heading_inv_quat,
                self.ref_root_global_rot_quat_t,
                w_last=True,
            ),
            w_last=True,
        )
        self.ref_base_roll_t = wrap_to_pi(self.ref_base_roll_t)
        self.ref_base_pitch_t = wrap_to_pi(self.ref_base_pitch_t)
        self.ref_base_yaw_t = wrap_to_pi(self.ref_base_yaw_t)

        n_bodies = self.ref_body_pos_t.shape[1]

        # calculate reference motion heading-aligned frame relative body pos,
        # rot, vel, ang vel
        self.cur_ref_heading_inv_quat_body_flat = self.cur_ref_heading_inv_quat.repeat(
            1, n_bodies, 1
        ).view(-1, 4)
        self.ref_rel_body_pos_t = quat_rotate(
            self.cur_ref_heading_inv_quat_body_flat,
            (self.ref_body_pos_t - self.ref_root_global_pos_t[:, None, :]).view(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]
        self.ref_rel_body_rot_quat_t = quat_mul(
            self.cur_ref_heading_inv_quat_body_flat,
            self.ref_body_rot_t.reshape(-1, 4),
            w_last=True,
        ).view(-1, n_bodies, 4)  # [B, N_total, 4]
        self.ref_rel_body_rot_tannorm_t = quat_to_tan_norm(
            self.ref_rel_body_rot_quat_t.reshape(-1, 4),
            w_last=True,
        ).view(-1, n_bodies, 6)
        self.ref_rel_body_vel_t = quat_rotate(
            self.cur_ref_heading_inv_quat_body_flat,
            self.ref_body_vel_t.reshape(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]
        self.ref_rel_body_ang_vel_t = quat_rotate(
            self.cur_ref_heading_inv_quat_body_flat,
            self.ref_body_ang_vel_t.reshape(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]

        # calculate reference motion root frame relative body pos, rot, vel,
        # ang vel
        self.ref_root_global_rot_quat_t_inv = quat_inverse(
            self.ref_root_global_rot_quat_t, w_last=True
        )
        self.ref_root_global_rot_quat_t_inv_body_flat = (
            self.ref_root_global_rot_quat_t_inv.repeat(1, n_bodies, 1).view(-1, 4)
        )
        self.ref_root_rel_body_pos_t = quat_rotate(
            self.ref_root_global_rot_quat_t_inv_body_flat,
            (self.ref_body_pos_t - self.ref_root_global_pos_t[:, None, :]).view(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]
        self.ref_root_rel_body_rot_quat_t = quat_mul(
            self.ref_root_global_rot_quat_t_inv_body_flat,
            self.ref_body_rot_t.reshape(-1, 4),
            w_last=True,
        ).view(-1, n_bodies, 4)  # [B, N_total, 4]
        self.ref_root_rel_body_rot_tannorm_t = quat_to_tan_norm(
            self.ref_root_rel_body_rot_quat_t.reshape(-1, 4),
            w_last=True,
        ).view(-1, n_bodies, 6)
        self.ref_root_rel_body_vel_t = quat_rotate(
            self.ref_root_global_rot_quat_t_inv_body_flat,
            self.ref_body_vel_t.reshape(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]
        self.ref_root_rel_body_ang_vel_t = quat_rotate(
            self.ref_root_global_rot_quat_t_inv_body_flat,
            self.ref_body_ang_vel_t.reshape(-1, 3),
            w_last=True,
        ).view(-1, n_bodies, 3)  # [B, N_total, 3]

        # Calculate robot RPY
        current_robot_world_quat = self.base_quat
        robot_quat_in_current_heading_frame = quat_mul(
            self.cur_heading_inv_quat,
            current_robot_world_quat,
            w_last=True,
        )
        self._robot_base_roll_t, self._robot_base_pitch_t, _ = get_euler_xyz(
            robot_quat_in_current_heading_frame, w_last=True
        )
        robot_quat_in_initial_heading_frame = quat_mul(
            self.robot_initial_heading_inv_quat,
            current_robot_world_quat,
            w_last=True,
        )
        _, _, self._robot_base_yaw_t = get_euler_xyz(
            robot_quat_in_initial_heading_frame, w_last=True
        )
        self._robot_base_roll_t = wrap_to_pi(self._robot_base_roll_t)
        self._robot_base_pitch_t = wrap_to_pi(self._robot_base_pitch_t)
        self._robot_base_yaw_t = wrap_to_pi(self._robot_base_yaw_t)

        # get the measured robot states at timestep t
        self._robot_base_rel_lin_vel_t = quat_rotate(
            self.cur_heading_inv_quat,
            self.simulator.robot_root_states[:, 7:10],
            w_last=True,
        )
        self._robot_base_rel_ang_vel_t = quat_rotate(
            self.cur_heading_inv_quat,
            self.simulator.robot_root_states[:, 10:13],
            w_last=True,
        )

        # calculate heading-aligned frame relative body pos, rot, vel, ang vel
        cur_heading_inv_quat_body_flat = self.cur_heading_inv_quat.repeat(
            1, n_bodies, 1
        ).view(-1, 4)
        self._robot_rel_body_pos_t = quat_rotate(
            cur_heading_inv_quat_body_flat,
            (
                self._rigid_body_pos_extend
                - self.simulator.robot_root_states[:, :3][:, None, :]
            ).reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)
        self._robot_rel_body_rot_quat_t = quat_mul(
            cur_heading_inv_quat_body_flat,
            self._rigid_body_rot_extend.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, n_bodies, 4)
        self._robot_rel_body_rot_tannorm_t = quat_to_tan_norm(
            self._robot_rel_body_rot_quat_t.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, n_bodies, 6)
        self._robot_rel_body_vel_t = quat_rotate(
            cur_heading_inv_quat_body_flat,
            self._rigid_body_vel_extend.reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)
        self._robot_rel_body_ang_vel_t = quat_rotate(
            cur_heading_inv_quat_body_flat,
            self._rigid_body_ang_vel_extend.reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)

        # calculate root frame relative body pos, rot, vel, ang vel
        current_robot_world_quat_inv = quat_inverse(
            current_robot_world_quat, w_last=True
        )
        current_robot_world_quat_inv_body_flat = current_robot_world_quat_inv.repeat(
            1, n_bodies, 1
        ).view(-1, 4)
        self._robot_root_rel_body_pos_t = quat_rotate(
            current_robot_world_quat_inv_body_flat,
            (
                self._rigid_body_pos_extend
                - self.simulator.robot_root_states[:, :3][:, None, :]
            ).reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)
        self._robot_root_rel_body_rot_quat_t = quat_mul(
            current_robot_world_quat_inv_body_flat,
            self._rigid_body_rot_extend.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, n_bodies, 4)
        self._robot_root_rel_body_rot_tannorm_t = quat_to_tan_norm(
            self._robot_root_rel_body_rot_quat_t.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, n_bodies, 6)
        self._robot_root_rel_body_rot_mat_t = quaternion_to_matrix(
            self._robot_root_rel_body_rot_quat_t.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3, 3)
        self._robot_root_rel_body_vel_t = quat_rotate(
            current_robot_world_quat_inv_body_flat,
            self._rigid_body_vel_extend.reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)
        self._robot_root_rel_body_ang_vel_t = quat_rotate(
            current_robot_world_quat_inv_body_flat,
            self._rigid_body_ang_vel_extend.reshape(-1, 3),
            w_last=True,
        ).view(self.num_envs, n_bodies, 3)

        # get the global root dif
        self.dif_global_root_pos_t = (
            self.simulator.robot_root_states[:, :3] - self.ref_root_global_pos_t
        )

        # get the diff between the measured and reference states at timestep t
        self.dif_local_root_lin_vel_t = (
            self._robot_base_rel_lin_vel_t - self.ref_base_rel_lin_vel_t
        )
        self.dif_local_root_ang_vel_t = (
            self._robot_base_rel_ang_vel_t - self.ref_base_rel_ang_vel_t
        )

        self.dif_local_root_roll_t = self._robot_base_roll_t - self.ref_base_roll_t
        self.dif_local_root_pitch_t = self._robot_base_pitch_t - self.ref_base_pitch_t
        self.dif_local_root_yaw_t = self._robot_base_yaw_t - self.ref_base_yaw_t

        self.dif_local_body_pos_t = self._robot_rel_body_pos_t - self.ref_rel_body_pos_t
        self.dif_local_body_vel_t = self._robot_rel_body_vel_t - self.ref_rel_body_vel_t
        self.dif_local_body_rot_tannorm = (
            self._robot_rel_body_rot_tannorm_t.view(-1, 6)
            - self.ref_rel_body_rot_tannorm_t.view(-1, 6)
        ).reshape(self.num_envs, n_bodies, 6)
        self.dif_local_body_ang_vel_t = (
            self._robot_rel_body_ang_vel_t - self.ref_rel_body_ang_vel_t
        )
        self.dif_root_rel_body_pos_t = (
            self._robot_root_rel_body_pos_t - self.ref_root_rel_body_pos_t
        )
        self.dif_root_rel_body_vel_t = (
            self._robot_root_rel_body_vel_t - self.ref_root_rel_body_vel_t
        )
        self.dif_root_rel_body_rot_tannorm = (
            self._robot_root_rel_body_rot_tannorm_t.view(-1, 6)
            - self.ref_root_rel_body_rot_tannorm_t.view(-1, 6)
        ).reshape(self.num_envs, n_bodies, 6)
        self.dif_root_rel_body_ang_vel_t = (
            self._robot_root_rel_body_ang_vel_t - self.ref_root_rel_body_ang_vel_t
        )

        robot_world_roll_t, robot_world_pitch_t, robot_world_yaw_t = get_euler_xyz(
            self.base_quat, w_last=True
        )
        robot_world_roll_t = wrap_to_pi(robot_world_roll_t)
        robot_world_pitch_t = wrap_to_pi(robot_world_pitch_t)
        robot_world_yaw_t = wrap_to_pi(robot_world_yaw_t)

        current_ref_world_quat = self.ref_root_global_rot_quat_t
        ref_world_roll_t, ref_world_pitch_t, ref_world_yaw_t = get_euler_xyz(
            current_ref_world_quat, w_last=True
        )
        ref_world_roll_t = wrap_to_pi(ref_world_roll_t)
        ref_world_pitch_t = wrap_to_pi(ref_world_pitch_t)
        ref_world_yaw_t = wrap_to_pi(ref_world_yaw_t)

        self.dif_base_roll_t = wrap_to_pi(robot_world_roll_t - ref_world_roll_t)
        self.dif_base_pitch_t = wrap_to_pi(robot_world_pitch_t - ref_world_pitch_t)
        self.dif_base_yaw_t = wrap_to_pi(robot_world_yaw_t - ref_world_yaw_t)

        self.dif_base_rpy_t = torch.stack(
            [
                self.dif_base_roll_t,
                self.dif_base_pitch_t,
                self.dif_base_yaw_t,
            ],
            dim=-1,
        )

        # Store reference state t for visualization/logging/extras
        self.ref_body_pos_extend = self.ref_body_pos_t  # Shape [B, N_total, 3]
        self.ref_body_rot_extend = self.ref_body_rot_t  # Shape [B, N_total, 4]

        # --- Compute differences (current_state_t vs ref_t) for Reward
        # Calculation ---
        self.dif_global_body_pos = self.ref_body_pos_t - self._rigid_body_pos_extend
        self.dif_global_body_rot = self.ref_body_rot_t - self._rigid_body_rot_extend
        self.dif_global_body_vel = self.ref_body_vel_t - self._rigid_body_vel_extend
        self.dif_global_body_ang_vel = (
            self.ref_body_ang_vel_t - self._rigid_body_ang_vel_extend
        )
        self.dif_joint_angles = self.ref_joint_pos_t - self.simulator.dof_pos
        self.dif_joint_velocities = self.ref_joint_vel_t - self.simulator.dof_vel

        # marker_coords for visualization (still uses ref_t)
        self.marker_coords[:] = self.ref_body_pos_t.reshape(
            self.num_envs,
            self.num_total_bodies,
            3,
        )
        if self.config.get("align_marker_to_root", False):
            self.marker_coords[:, :, :3] = (
                self.marker_coords[:, :, :3] - self.marker_coords[:, 0:1, :3]
            ) + self.simulator.robot_root_states[:, :3][:, None, :]

        # Log info based on state_t vs ref_t differences
        self._log_motion_tracking_info()

        # --- Extract future frames t+1...t+n for Observation calculation ---
        # Fetches frames t+1 ... t+n (if n_fut_config > 0)
        self.ref_body_pos_fut = motion_res_seq["rg_pos_t"][:, 1:, ...].to(self.device)
        self.ref_body_rot_fut = motion_res_seq["rg_rot_t"][:, 1:, ...].to(self.device)
        self.ref_body_vel_fut = motion_res_seq["body_vel_t"][:, 1:, ...].to(self.device)
        self.ref_body_ang_vel_fut = motion_res_seq["body_ang_vel_t"][:, 1:, ...].to(
            self.device
        )
        self.ref_base_height_fut = motion_res_seq["root_pos"][:, 1:, 2:3].to(
            self.device
        )
        self.ref_base_rot_fut = motion_res_seq["root_rot"][:, 1:, ...].to(self.device)
        self.ref_base_lin_vel_fut = motion_res_seq["root_vel"][:, 1:, ...].to(
            self.device
        )
        self.ref_base_ang_vel_fut = motion_res_seq["root_ang_vel"][:, 1:, ...].to(
            self.device
        )
        self.ref_dof_pos_fut = motion_res_seq["dof_pos"][:, 1:, ...].to(self.device)
        self.ref_dof_vel_fut = motion_res_seq["dof_vel"][:, 1:, ...].to(self.device)

    def _pre_eval_compute_observations_callback(self):
        """Evaluation observation computation callback function."""
        # 首先调用基础的观测计算函数
        self._pre_compute_observations_callback()

        # 评测特有的逻辑：初始化历史缓冲区和加速度计算
        # 这些在训练时不需要，只在评测时用于更详细的指标计算

        # 初始化位置历史缓冲区（用于加速度计算）
        if (
            self.pos_history_buffer is None
            or self.pos_history_buffer.shape[2] != self.num_total_bodies
        ):
            self.pos_history_buffer = torch.zeros(
                self.num_envs,
                3,  # Number of history frames
                self.num_total_bodies,  # Total number of bodies including
                # extended
                3,  # XYZ position
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # Initialize with current positions
            self.pos_history_buffer = self._rigid_body_pos_extend.unsqueeze(1).repeat(
                1, 3, 1, 1
            )

        # 初始化加速度缓冲区
        if (
            self.current_accel is None
            or self.current_accel.shape[1] != self.num_total_bodies
        ):
            self.current_accel = torch.zeros(
                self.num_envs,
                self.num_total_bodies,  # Total number of bodies including
                # extended
                3,  # XYZ acceleration
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

        # 更新位置历史缓冲区
        self.pos_history_buffer = torch.cat(
            [
                self.pos_history_buffer[:, 1:],
                self._rigid_body_pos_extend.unsqueeze(1),
            ],
            dim=1,
        )

        # 计算当前加速度
        self.current_accel = (
            self.pos_history_buffer[:, 2]
            - 2 * self.pos_history_buffer[:, 1]
            + self.pos_history_buffer[:, 0]
        )

        # 初始化旋转历史缓冲区（用于角加速度计算）
        if (
            self.rot_history_buffer is None
            or self.rot_history_buffer.shape[2] != self.num_total_bodies
        ):
            self.rot_history_buffer = torch.zeros(
                self.num_envs,
                3,  # Number of history frames
                self.num_total_bodies,  # Total number of bodies including
                # extended
                4,  # Quaternion
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # Set w component to 1 for valid quaternions
            self.rot_history_buffer[..., 3] = 1.0
            # Initialize with current rotations
            self.rot_history_buffer = self._rigid_body_rot_extend.unsqueeze(1).repeat(
                1, 3, 1, 1
            )

        # 更新旋转历史缓冲区
        self.rot_history_buffer = torch.cat(
            [
                self.rot_history_buffer[:, 1:],
                self._rigid_body_rot_extend.unsqueeze(1),
            ],
            dim=1,
        )

        # 初始化角加速度缓冲区
        if not hasattr(self, "current_ang_accel") or self.current_ang_accel is None:
            self.current_ang_accel = torch.zeros_like(self.current_accel)  # Placeholder

        # 初始化参考位置历史缓冲区（用于参考加速度计算）
        num_ref_bodies = self.ref_body_pos_t.shape[1]
        if (
            self.ref_pos_history_buffer is None
            or self.ref_pos_history_buffer.shape[2] != num_ref_bodies
        ):
            self.ref_pos_history_buffer = torch.zeros(
                self.num_envs,
                3,  # Number of history frames
                num_ref_bodies,  # Number of reference bodies
                3,  # XYZ position
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            # Initialize with current reference positions
            self.ref_pos_history_buffer = self.ref_body_pos_t.unsqueeze(1).repeat(
                1, 3, 1, 1
            )

        # 初始化参考加速度缓冲区
        if (
            not hasattr(self, "ref_body_accel")
            or self.ref_body_accel is None
            or self.ref_body_accel.shape[1] != num_ref_bodies
        ):
            self.ref_body_accel = torch.zeros(
                self.num_envs,
                num_ref_bodies,  # Number of reference bodies
                3,  # XYZ acceleration
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

        # 更新参考位置历史缓冲区
        self.ref_pos_history_buffer = torch.cat(
            [
                self.ref_pos_history_buffer[:, 1:],
                self.ref_body_pos_t.unsqueeze(1),
            ],
            dim=1,
        )

        # 计算参考加速度
        self.ref_body_accel = (
            self.ref_pos_history_buffer[:, 2]
            - 2 * self.ref_pos_history_buffer[:, 1]
            + self.ref_pos_history_buffer[:, 0]
        )

        # get the diff between the measured and reference states at timestep t
        self.dif_local_root_lin_vel_t = (
            self._robot_base_rel_lin_vel_t - self.ref_base_rel_lin_vel_t
        )
        self.dif_local_root_ang_vel_t = (
            self._robot_base_rel_ang_vel_t - self.ref_base_rel_ang_vel_t
        )

        self.dif_local_root_roll_t = self._robot_base_roll_t - self.ref_base_roll_t
        self.dif_local_root_pitch_t = self._robot_base_pitch_t - self.ref_base_pitch_t
        self.dif_local_root_yaw_t = self._robot_base_yaw_t - self.ref_base_yaw_t

        self.dif_local_body_pos_t = self._robot_rel_body_pos_t - self.ref_rel_body_pos_t
        self.dif_local_body_vel_t = self._robot_rel_body_vel_t - self.ref_rel_body_vel_t
        n_bodies = self.ref_body_pos_t.shape[1]
        self.dif_local_body_rot_tannorm = (
            self._robot_rel_body_rot_tannorm_t.view(-1, 6)
            - self.ref_rel_body_rot_tannorm_t.view(-1, 6)
        ).reshape(self.num_envs, n_bodies, 6)
        self.dif_local_body_ang_vel_t = (
            self._robot_rel_body_ang_vel_t - self.ref_rel_body_ang_vel_t
        )
        self.dif_root_rel_body_pos_t = (
            self._robot_root_rel_body_pos_t - self.ref_root_rel_body_pos_t
        )
        self.dif_root_rel_body_vel_t = (
            self._robot_root_rel_body_vel_t - self.ref_root_rel_body_vel_t
        )
        self.dif_root_rel_body_rot_tannorm = (
            self._robot_root_rel_body_rot_tannorm_t.view(-1, 6)
            - self.ref_root_rel_body_rot_tannorm_t.view(-1, 6)
        ).reshape(self.num_envs, n_bodies, 6)
        self.dif_root_rel_body_ang_vel_t = (
            self._robot_root_rel_body_ang_vel_t - self.ref_root_rel_body_ang_vel_t
        )

        robot_world_roll_t, robot_world_pitch_t, robot_world_yaw_t = get_euler_xyz(
            self.base_quat, w_last=True
        )
        robot_world_roll_t = wrap_to_pi(robot_world_roll_t)
        robot_world_pitch_t = wrap_to_pi(robot_world_pitch_t)
        robot_world_yaw_t = wrap_to_pi(robot_world_yaw_t)

        current_ref_world_quat = self.ref_root_global_rot_quat_t
        ref_world_roll_t, ref_world_pitch_t, ref_world_yaw_t = get_euler_xyz(
            current_ref_world_quat, w_last=True
        )
        ref_world_roll_t = wrap_to_pi(ref_world_roll_t)
        ref_world_pitch_t = wrap_to_pi(ref_world_pitch_t)
        ref_world_yaw_t = wrap_to_pi(ref_world_yaw_t)

        self.dif_base_roll_t = wrap_to_pi(robot_world_roll_t - ref_world_roll_t)
        self.dif_base_pitch_t = wrap_to_pi(robot_world_pitch_t - ref_world_pitch_t)
        self.dif_base_yaw_t = wrap_to_pi(robot_world_yaw_t - ref_world_yaw_t)

        self.dif_base_rpy_t = torch.stack(
            [
                self.dif_base_roll_t,
                self.dif_base_pitch_t,
                self.dif_base_yaw_t,
            ],
            dim=-1,
        )

        # Store reference state t for visualization/logging/extras
        self.ref_body_pos_extend = self.ref_body_pos_t  # Shape [B, N_total, 3]
        self.ref_body_rot_extend = self.ref_body_rot_t  # Shape [B, N_total, 4]

        self.dif_global_body_pos = self.ref_body_pos_t - self._rigid_body_pos_extend
        self.dif_global_body_rot = self.ref_body_rot_t - self._rigid_body_rot_extend
        self.dif_global_body_vel = self.ref_body_vel_t - self._rigid_body_vel_extend
        self.dif_global_body_ang_vel = (
            self.ref_body_ang_vel_t - self._rigid_body_ang_vel_extend
        )
        self.dif_joint_angles = self.ref_joint_pos_t - self.simulator.dof_pos
        self.dif_joint_velocities = self.ref_joint_vel_t - self.simulator.dof_vel

        # marker_coords for visualization (still uses ref_t)
        self.marker_coords[:] = self.ref_body_pos_t.reshape(
            self.num_envs,
            self.num_total_bodies,
            3,
        )
        if self.config.get("align_marker_to_root", False):
            self.marker_coords[:, :, :3] = (
                self.marker_coords[:, :, :3] - self.marker_coords[:, 0:1, :3]
            ) + self.simulator.robot_root_states[:, :3][:, None, :]

        # Log info based on state_t vs ref_t differences
        # self._log_motion_tracking_info()
        self._log_motion_tracking_holomotion_metrics()

        # 注意：future frames 的处理已经在
        # _pre_compute_observations_callback 中完成
        # 这里不需要重复处理，因为基础函数已经设置了所有必要的 future 数据

    def _get_obs_local_body_pos_extend(self) -> torch.Tensor:
        local_body_pos_extend = (
            self._rigid_body_pos_extend
            - self.simulator.robot_root_states[:, :3][:, None, :]
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 3]
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7], w_last=True
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_pos_extend.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 4]
        local_body_pos_extend = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4),
            local_body_pos_extend.view(-1, 3),
        )  # [num_envs, num_rigid_bodies + num_extended_bodies * 3]
        local_body_pos_extend = local_body_pos_extend.reshape(n_envs, n_bodies, 3)
        return local_body_pos_extend

    def _get_obs_local_body_pos_extend_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_pos_extend().reshape(self.num_envs, -1)

    def _get_obs_local_body_rot_quat_extend(self) -> torch.Tensor:
        global_body_rot_quat_extend = self._rigid_body_rot_extend
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7], w_last=True
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = global_body_rot_quat_extend.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 4]
        local_body_rot_quat_extend = quat_mul(
            root_heading_quat_inv,
            global_body_rot_quat_extend,
            w_last=True,
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 4]
        local_body_rot_quat_extend = local_body_rot_quat_extend.reshape(
            n_envs, n_bodies, 4
        )
        return local_body_rot_quat_extend

    def _get_obs_local_body_rot_quat_extend_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_rot_quat_extend().reshape(self.num_envs, -1)

    def _get_obs_local_body_vel_extend(self) -> torch.Tensor:
        global_body_vel_extend = (
            self._rigid_body_vel_extend
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 3]
        local_body_vel_extend = global_body_vel_extend - self.base_lin_vel[:, None, :]
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7], w_last=True
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_vel_extend.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 4]
        local_body_vel_extend = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4),
            local_body_vel_extend.view(-1, 3),
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 3]
        local_body_vel_extend = local_body_vel_extend.reshape(n_envs, n_bodies, 3)
        return local_body_vel_extend

    def _get_obs_local_body_vel_extend_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_vel_extend().reshape(self.num_envs, -1)

    def _get_obs_local_body_ang_vel_extend(self) -> torch.Tensor:
        global_body_ang_vel_extend = self._rigid_body_ang_vel_extend
        local_body_ang_vel_extend = (
            global_body_ang_vel_extend - self.base_ang_vel[:, None, :]
        )
        root_heading_quat_inv = calc_heading_quat_inv(
            self.simulator.robot_root_states[:, 3:7], w_last=True
        )[:, None, :]  # [num_envs, 1, 4]
        n_envs, n_bodies = local_body_ang_vel_extend.shape[:2]
        root_heading_quat_inv = root_heading_quat_inv.repeat(
            1, n_bodies, 1
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 4]
        local_body_ang_vel_extend = my_quat_rotate(
            root_heading_quat_inv.view(-1, 4),
            local_body_ang_vel_extend.view(-1, 3),
        )  # [num_envs, num_rigid_bodies + num_extended_bodies, 3]
        local_body_ang_vel_extend = local_body_ang_vel_extend.reshape(
            n_envs, n_bodies, 3
        )
        return local_body_ang_vel_extend

    def _get_obs_local_body_ang_vel_extend_flat(self) -> torch.Tensor:
        return self._get_obs_local_body_ang_vel_extend().reshape(self.num_envs, -1)

    def _get_obs_global_ref_rigid_body_pos(self):
        return self.ref_body_pos_fut

    def _get_obs_local_ref_rigid_body_pos(self):
        n_bodies = self.ref_body_pos_fut.shape[2]
        dif_global_body_pos = (
            self.ref_body_pos_fut - self._rigid_body_pos_extend[:, None, :, :]
        )  # [num_envs, FT, N_bodies, 3]
        local_ref_rigid_body_pos_fut = my_quat_rotate(
            self.cur_heading_inv_quat[:, None, None, :]
            .expand(-1, self.FT, n_bodies, -1)
            .reshape(-1, 4),
            dif_global_body_pos.reshape(-1, 3),
        )  # [num_envs * FT * N_bodies, 3]
        return local_ref_rigid_body_pos_fut.reshape(self.num_envs, self.FT, n_bodies, 3)

    def _get_obs_local_ref_rigid_body_pos_flat(self):
        return self._get_obs_local_ref_rigid_body_pos().reshape(self.num_envs, -1)

    def _get_obs_local_ref_rigid_body_rot(self):
        bs = self.num_envs
        fts = self.FT
        nbds = self.num_total_bodies

        q_heading_inv = self.cur_heading_inv_quat
        ref_rot_fut = self.ref_body_rot_fut  # Shape: [B, FT, N_bodies, 4]

        # Expand heading_inv to match ref_rot_fut dimensions
        q_heading_inv_expanded = q_heading_inv[:, None, None, :].expand(
            -1, fts, nbds, -1
        )

        # Flatten both tensors for quat_mul
        q_heading_inv_flat = q_heading_inv_expanded.reshape(-1, 4)
        ref_rot_fut_flat = ref_rot_fut.reshape(-1, 4)
        local_ref_rot_fut_flat = quat_mul(
            q_heading_inv_flat, ref_rot_fut_flat, w_last=True
        )

        # Reshape back to original batch dimensions
        local_ref_rot_fut = local_ref_rot_fut_flat.reshape(bs, fts, nbds, 4)

        return local_ref_rot_fut

    def _get_obs_local_ref_rigid_body_rot_flat(self):
        return self._get_obs_local_ref_rigid_body_rot().reshape(self.num_envs, -1)

    def _get_obs_local_ref_rigid_body_vel(self):
        n_bodies = self.ref_body_vel_fut.shape[2]
        dif_global_body_vel = (
            self.ref_body_vel_fut - self._rigid_body_vel_extend[:, None, :, :]
        )  # [num_envs, FT, N_bodies, 3]
        local_ref_rigid_body_vel_fut_flat = my_quat_rotate(
            self.cur_heading_inv_quat[:, None, None, :]
            .expand(-1, self.FT, n_bodies, -1)
            .reshape(-1, 4),
            dif_global_body_vel.reshape(-1, 3),
        )
        return local_ref_rigid_body_vel_fut_flat.reshape(
            self.num_envs, self.FT, n_bodies, 3
        )

    def _get_obs_local_ref_rigid_body_vel_flat(self):
        return self._get_obs_local_ref_rigid_body_vel().reshape(self.num_envs, -1)

    def _get_obs_local_ref_rigid_body_ang_vel(self):
        n_bodies = self.ref_body_ang_vel_fut.shape[2]
        dif_global_body_ang_vel = (
            self.ref_body_ang_vel_fut - self._rigid_body_ang_vel_extend[:, None, :, :]
        )
        local_ref_rigid_body_ang_vel_fut_flat = my_quat_rotate(
            self.cur_heading_inv_quat[:, None, None, :]
            .expand(-1, self.FT, n_bodies, -1)
            .reshape(-1, 4),
            dif_global_body_ang_vel.reshape(-1, 3),
        )
        return local_ref_rigid_body_ang_vel_fut_flat.reshape(
            self.num_envs, self.FT, n_bodies, 3
        )

    def _get_obs_local_ref_rigid_body_ang_vel_flat(self):
        return self._get_obs_local_ref_rigid_body_ang_vel().reshape(self.num_envs, -1)

    def _get_obs_ref_dof_pos(self):
        return self.ref_dof_pos_fut

    def _get_obs_ref_dof_pos_flat(self):
        return self.ref_dof_pos_fut.reshape(self.num_envs, -1)

    def _get_obs_ref_dof_vel(self):
        return self.ref_dof_vel_fut

    def _get_obs_ref_dof_vel_flat(self):
        return self.ref_dof_vel_fut.reshape(self.num_envs, -1)

    def _get_obs_ref_body_pos_extend(self):
        return self.ref_body_pos_fut

    def _get_obs_ref_body_pos_extend_flat(self):
        return self._get_obs_ref_body_pos_extend().reshape(self.num_envs, -1)

    def _get_obs_ref_body_vel_extend(self):
        return self.ref_body_vel_fut

    def _get_obs_ref_body_vel_extend_flat(self):
        return self._get_obs_ref_body_vel_extend().reshape(self.num_envs, -1)

    def _get_obs_ref_body_ang_vel_extend(self):
        return self.ref_body_ang_vel_fut

    def _get_obs_ref_body_ang_vel_extend_flat(self):
        return self._get_obs_ref_body_ang_vel_extend().reshape(self.num_envs, -1)

    def _get_obs_ref_body_rot_extend(self):
        return self.ref_body_rot_fut

    def _get_obs_ref_body_rot_extend_flat(self):
        return self._get_obs_ref_body_rot_extend().reshape(self.num_envs, -1)

    def _get_obs_ref_base_height(self):
        return self.ref_base_height_fut

    def _get_obs_ref_base_lin_vel(self):
        return self.ref_base_lin_vel_fut

    def _get_obs_ref_base_lin_vel_flat(self):
        return self._get_obs_ref_base_lin_vel().reshape(self.num_envs, -1)

    def _get_obs_ref_base_ang_vel(self):
        return self.ref_base_ang_vel_fut

    def _get_obs_ref_base_ang_vel_flat(self):
        return self._get_obs_ref_base_ang_vel().reshape(self.num_envs, -1)

    def _get_obs_ref_motion_state_flat(self):
        return torch.cat(
            [
                self._get_obs_ref_body_pos_extend_flat(),
                self._get_obs_ref_body_vel_extend_flat(),
                self._get_obs_ref_body_ang_vel_extend_flat(),
                self._get_obs_ref_body_rot_extend_flat(),
                self._get_obs_ref_dof_pos_flat(),
                self._get_obs_ref_dof_vel_flat(),
            ],
            dim=-1,
        )

    def _get_obs_dif_local_rigid_body_pos(self):
        bs = self.num_envs
        fts = self.FT  # Number of future frames
        nbds = self.num_total_bodies

        # Difference in global frame: [B, FT, N_bodies, 3]
        dif_global_body_pos_fut = (
            self.ref_body_pos_fut - self._rigid_body_pos_extend[:, None, :, :]
        )

        # Current heading inverse rotation: [B, 4]
        q = self.cur_heading_inv_quat

        # Expand q to match the dimensions of v for rotation
        # q: [B, 4] -> [B, 1, 1, 4] -> [B, FT, N_bodies, 4]
        q_expanded = q[:, None, None, :].expand(-1, fts, nbds, -1)

        q_flat = q_expanded.reshape(-1, 4)
        v_flat = dif_global_body_pos_fut.reshape(-1, 3)

        # Rotate difference into local heading frame
        rotated_v_flat = my_quat_rotate(q_flat, v_flat)

        # Reshape back to [B, FT, N_bodies, 3] and then flatten for observation
        dif_local_body_pos_fut = rotated_v_flat.reshape(bs, fts, nbds, 3)

        return dif_local_body_pos_fut

    def _get_obs_dif_local_rigid_body_pos_flat(self):
        return self._get_obs_dif_local_rigid_body_pos().reshape(self.num_envs, -1)

    def _get_obs_dif_local_rigid_body_rot(self):
        bs = self.num_envs
        fts = self.FT
        nbds = self.num_total_bodies

        q_heading_inv = self.cur_heading_inv_quat
        q_heading_inv_expanded = q_heading_inv[:, None, None, :].expand(
            -1, fts, nbds, -1
        )

        ref_rot_fut = self.ref_body_rot_fut  # Shape: [B, FT, N_bodies, 4]
        dif_ref_rot_fut = ref_rot_fut - self._rigid_body_rot_extend[:, None, :, :]

        q_heading_inv_flat = q_heading_inv_expanded.reshape(-1, 4)
        dif_ref_rot_fut_flat = dif_ref_rot_fut.reshape(-1, 4)

        local_ref_rot_fut_flat = quat_mul(
            q_heading_inv_flat,
            dif_ref_rot_fut_flat,
            w_last=True,
        )

        # Reshape back to original batch dimensions
        dif_local_body_rot_fut = local_ref_rot_fut_flat.reshape(bs, fts, nbds, 4)

        return dif_local_body_rot_fut

    def _get_obs_dif_local_rigid_body_rot_flat(self):
        return self._get_obs_dif_local_rigid_body_rot().reshape(self.num_envs, -1)

    def _get_obs_dif_local_rigid_body_vel(self):
        bs = self.num_envs
        fts = self.FT  # Number of future frames
        nbds = self.num_total_bodies

        # Difference in global frame: [B, FT, N_bodies, 3]
        dif_global_body_vel_fut = (
            self.ref_body_vel_fut - self._rigid_body_vel_extend[:, None, :, :]
        )

        # Current heading inverse rotation: [B, 4]
        q = self.cur_heading_inv_quat

        # Expand q to match the dimensions of v for rotation
        # q: [B, 4] -> [B, 1, 1, 4] -> [B, FT, N_bodies, 4]
        q_expanded = q[:, None, None, :].expand(-1, fts, nbds, -1)

        q_flat = q_expanded.reshape(-1, 4)
        v_flat = dif_global_body_vel_fut.reshape(-1, 3)

        # Rotate difference into local heading frame
        rotated_v_flat = my_quat_rotate(q_flat, v_flat)

        # Reshape back to [B, FT, N_bodies, 3] and then flatten for observation
        dif_local_body_vel_fut = rotated_v_flat.reshape(bs, fts, nbds, 3)

        return dif_local_body_vel_fut

    def _get_obs_dif_local_rigid_body_vel_flat(self):
        return self._get_obs_dif_local_rigid_body_vel().reshape(self.num_envs, -1)

    def _get_obs_dif_local_rigid_body_ang_vel(self):
        bs = self.num_envs
        fts = self.FT  # Number of future frames
        nbds = self.num_total_bodies

        # Difference in global frame: [B, FT, N_bodies, 3]
        dif_global_body_ang_vel_fut = (
            self.ref_body_ang_vel_fut - self._rigid_body_ang_vel_extend[:, None, :, :]
        )

        # Current heading inverse rotation: [B, 4]
        q = self.cur_heading_inv_quat

        # Expand q to match the dimensions of v for rotation
        # q: [B, 4] -> [B, 1, 1, 4] -> [B, FT, N_bodies, 4]
        q_expanded = q[:, None, None, :].expand(-1, fts, nbds, -1)

        q_flat = q_expanded.reshape(-1, 4)
        v_flat = dif_global_body_ang_vel_fut.reshape(-1, 3)

        # Rotate difference into local heading frame
        rotated_v_flat = my_quat_rotate(q_flat, v_flat)

        # Reshape back to [B, FT, N_bodies, 3] and then flatten for observation
        dif_local_body_ang_vel_fut = rotated_v_flat.reshape(bs, fts, nbds, 3)

        return dif_local_body_ang_vel_fut

    def _get_obs_dif_local_rigid_body_ang_vel_flat(self):
        return self._get_obs_dif_local_rigid_body_ang_vel().reshape(self.num_envs, -1)

    def _get_obs_dif_dof_pos(self):
        return self.ref_dof_pos_fut - self.simulator.dof_pos[:, None, :]

    def _get_obs_dif_dof_pos_flat(self):
        return self._get_obs_dif_dof_pos().reshape(self.num_envs, -1)

    def _get_obs_dif_dof_vel(self):
        return self.ref_dof_vel_fut - self.simulator.dof_vel[:, None, :]

    def _get_obs_dif_dof_vel_flat(self):
        return self._get_obs_dif_dof_vel().reshape(self.num_envs, -1)

    def _get_obs_ref_motion_phase(self):
        ref_motion_phase_t = (
            self.motion_global_frame_ids_t  # Use current frame ID [B]
            / self._motion_lib.cache.cached_motion_raw_num_frames.to(self.device)[
                :, None
            ]  # Expand denom [B, 1]
        )
        ref_motion_phase_t = torch.clamp(ref_motion_phase_t, 0.0, 1.0)
        return ref_motion_phase_t.unsqueeze(-1)

    def _get_obs_vr_3point_pos(self):
        return self._get_obs_local_ref_rigid_body_pos()[
            :, :, self.motion_tracking_id, :
        ]

    def _get_obs_vr_3point_pos_flat(self):
        return self._get_obs_vr_3point_pos().reshape(self.num_envs, -1)

    def _get_obs_ref_dof_pos_flat(self):
        return self.ref_dof_pos_fut.reshape(self.num_envs, -1)

    def _get_obs_ref_dof_vel_flat(self):
        return self.ref_dof_vel_fut.reshape(self.num_envs, -1)

    def _get_obs_history_actor(self):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_actor"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_critic(self):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary["history_critic"]
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_local_key_pos(self):
        # Shape: [B, num_key_bodies * 3]
        root_pos_t = self.simulator.robot_root_states[:, 0:3]
        # Use the pre-calculated value from _pre_compute_observations_callback
        heading_inv_rot_t = self.cur_heading_inv_quat

        # Get current global key body positions from the extended state
        global_key_body_pos_t = self._rigid_body_pos_extend[:, self.key_body_indices, :]

        # Calculate positions relative to current root
        relative_key_pos_t = global_key_body_pos_t - root_pos_t[:, None, :]

        # Rotate into local heading frame
        num_key_bodies = len(self.key_body_indices)
        # Expand heading quat: [B, 4] -> [B, 1, 4] -> [B, num_key_bodies, 4]
        heading_inv_rot_t_expand = heading_inv_rot_t[:, None, :].expand(
            -1, num_key_bodies, -1
        )
        cur_local_key_pos_flat = my_quat_rotate(
            heading_inv_rot_t_expand.reshape(-1, 4),
            relative_key_pos_t.reshape(-1, 3),
        )
        # Reshape back: [B, num_key_bodies * 3]
        cur_local_key_pos = cur_local_key_pos_flat.reshape(self.num_envs, -1)
        return cur_local_key_pos

    def _get_obs_amp_agent_seq(self):
        # --- Get Agent's CURRENT Local Key Poses (state_t) ---
        cur_local_key_pos = self._get_obs_local_key_pos()  # Use the dedicated method

        # --- Get history for AMP sequence --- (Uses single-frame history)
        hist_len = self.config.amp_context_length - 1

        agent_amp_projected_gravity_cur = self.projected_gravity
        agent_amp_projected_gravity_hist = self.history_handler.query(
            "projected_gravity"
        )[:, :hist_len]
        agent_amp_projected_gravity = torch.cat(
            [
                agent_amp_projected_gravity_cur[:, None, :],
                agent_amp_projected_gravity_hist,
            ],
            dim=1,
        )

        cur_root_vel = self.simulator.robot_root_states[:, 7:10]  # state_t
        hist_root_vel = self.history_handler.query("base_lin_vel")[:, :hist_len]
        agent_amp_root_vel = torch.cat([cur_root_vel[:, None, :], hist_root_vel], dim=1)

        cur_root_ang_vel = self.simulator.robot_root_states[:, 10:13]  # state_t
        hist_root_ang_vel = self.history_handler.query("base_ang_vel")[:, :hist_len]
        agent_amp_root_ang_vel = torch.cat(
            [cur_root_ang_vel[:, None, :], hist_root_ang_vel], dim=1
        )

        cur_dof_pos = self.simulator.dof_pos  # state_t
        hist_dof_pos = self.history_handler.query("dof_pos")[:, :hist_len]
        agent_amp_dof_pos = torch.cat([cur_dof_pos[:, None, :], hist_dof_pos], dim=1)

        cur_dof_vel = self.simulator.dof_vel  # state_t
        hist_dof_vel = self.history_handler.query("dof_vel")[:, :hist_len]
        agent_amp_dof_vel = torch.cat([cur_dof_vel[:, None, :], hist_dof_vel], dim=1)
        hist_local_key_pos = self.history_handler.query("local_key_pos")[:, :hist_len]

        agent_amp_local_key_pos = torch.cat(
            [
                cur_local_key_pos[
                    :, None, :
                ],  # Use agent's current local key pos computed above
                hist_local_key_pos,
            ],
            dim=1,
        )

        # Concatenate the full AMP agent sequence
        agent_amp_seq = torch.cat(
            [
                agent_amp_projected_gravity.reshape(self.num_envs, -1),
                agent_amp_root_vel.reshape(self.num_envs, -1),
                agent_amp_root_ang_vel.reshape(self.num_envs, -1),
                agent_amp_dof_pos.reshape(self.num_envs, -1),
                agent_amp_dof_vel.reshape(self.num_envs, -1),
                agent_amp_local_key_pos.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

        return agent_amp_seq

    def _get_obs_dof_seq(self):
        dof_pos = self._get_obs_dof_pos()[..., None]  # [num_envs, num_dofs, 1]
        dof_vel = self._get_obs_dof_vel()[..., None]  # [num_envs, num_dofs, 1]
        last_actions = self._get_obs_actions()[..., None]  # [num_envs, num_dofs, 1]
        ref_dof_pos = self._get_obs_ref_dof_pos().permute(
            0, 2, 1
        )  # [num_envs, num_dofs, FT]
        ref_dof_vel = self._get_obs_ref_dof_vel().permute(
            0, 2, 1
        )  # [num_envs, num_dofs, FT]

        dof_seq = torch.cat(
            [
                dof_pos,
                dof_vel,
                last_actions,
                ref_dof_pos,
                ref_dof_vel,
            ],
            dim=-1,
        )
        return dof_seq

    def _get_obs_body_seq(self):
        local_body_pos = (
            self._get_obs_local_body_pos_extend()
        )  # [num_envs, num_bodies, 3]
        local_body_rot = (
            self._get_obs_local_body_rot_quat_extend()
        )  # [num_envs, num_bodies, 4]
        local_body_vel = (
            self._get_obs_local_body_vel_extend()
        )  # [num_envs, num_bodies, 3]
        local_body_ang_vel = (
            self._get_obs_local_body_ang_vel_extend()
        )  # [num_envs, num_bodies, 3]

        nbds = local_body_pos.shape[1]
        dif_local_body_pos = (
            (self._get_obs_dif_local_rigid_body_pos())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 3 * FT]
        dif_local_body_rot = (
            (self._get_obs_dif_local_rigid_body_rot())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 4 * FT]
        dif_local_body_vel = (
            (self._get_obs_dif_local_rigid_body_vel())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 3 * FT]
        dif_local_body_ang_vel = (
            (self._get_obs_dif_local_rigid_body_ang_vel())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 3 * FT]
        local_ref_body_pos = (
            (self._get_obs_local_ref_rigid_body_pos())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 3 * FT]
        local_ref_body_rot = (
            (self._get_obs_local_ref_rigid_body_rot())
            .permute(0, 2, 1, 3)
            .reshape(self.num_envs, nbds, -1)
        )  # [num_envs, num_bodies, 4 * FT]

        body_seq = torch.cat(
            [
                local_body_pos,  # [num_envs, num_bodies, 3]
                local_body_rot,  # [num_envs, num_bodies, 4]
                local_body_vel,  # [num_envs, num_bodies, 3]
                local_body_ang_vel,  # [num_envs, num_bodies, 3]
                dif_local_body_pos,  # [num_envs, num_bodies, 3 * FT]
                dif_local_body_rot,  # [num_envs, num_bodies, 4 * FT]
                dif_local_body_vel,  # [num_envs, num_bodies, 3 * FT]
                dif_local_body_ang_vel,  # [num_envs, num_bodies, 3 * FT]
                local_ref_body_pos,  # [num_envs, num_bodies, 3 * FT]
                local_ref_body_rot,  # [num_envs, num_bodies, 4 * FT]
            ],
            dim=-1,
        )
        return body_seq

    def _get_obs_base_seq(self):
        base_height = self._get_obs_base_height()
        projected_gravity = self._get_obs_projected_gravity()
        base_lin_vel = self._get_obs_base_lin_vel()
        base_ang_vel = self._get_obs_base_ang_vel()
        base_seq = torch.cat(
            [
                base_height,  # [num_envs, 1]
                projected_gravity,  # [num_envs, 3]
                base_lin_vel,  # [num_envs, 3]
                base_ang_vel,  # [num_envs, 3]
            ],
            dim=-1,
        )[:, None, :]
        return base_seq

    def _get_obs_domain_seq(self):
        return self._get_obs_domain_params()[:, None, :]

    def _get_obs_privilege_body_states_seq(self):
        local_body_pos_flat = (
            self._get_obs_local_body_pos_extend_flat()
        )  # [num_envs, num_bodies, 3]
        local_body_rot_flat = (
            self._get_obs_local_body_rot_quat_extend_flat()
        )  # [num_envs, num_bodies, 4]
        local_body_vel_flat = (
            self._get_obs_local_body_vel_extend_flat()
        )  # [num_envs, num_bodies, 3]
        local_body_ang_vel_flat = (
            self._get_obs_local_body_ang_vel_extend_flat()
        )  # [num_envs, num_bodies, 3]

        body_seq = torch.cat(
            [
                local_body_pos_flat,  # [num_envs, num_bodies * 3]
                local_body_rot_flat,  # [num_envs, num_bodies * 4]
                local_body_vel_flat,  # [num_envs, num_bodies * 3]
                local_body_ang_vel_flat,  # [num_envs, num_bodies * 3]
            ],
            dim=-1,
        )
        return body_seq[:, None, :]  # [num_envs, 1, num_bodies * 13]

    def _get_obs_privilege_base_states_seq(self):
        base_height = self._get_obs_base_height()
        projected_gravity = self._get_obs_projected_gravity()
        base_lin_vel = self._get_obs_base_lin_vel()
        base_ang_vel = self._get_obs_base_ang_vel()

        return torch.cat(
            [base_height, projected_gravity, base_lin_vel, base_ang_vel],
            dim=-1,
        )[:, None, :]  # [num_envs, 1, 9]

    def _get_obs_privilege_domain_params_seq(self):
        return self._get_obs_domain_params()[
            :, None, :
        ]  # [num_envs, 1, 3 + num_links + 2*num_dofs + 1 + num_dofs]

    def _get_obs_rel_base_lin_vel(self):
        return self._robot_base_rel_lin_vel_t

    def _get_obs_rel_base_ang_vel(self):
        return self._robot_base_rel_ang_vel_t

    def _get_obs_base_rpy(self):
        return torch.stack(
            [
                self._robot_base_roll_t,
                self._robot_base_pitch_t,
                self._robot_base_yaw_t,
            ],
            dim=-1,
        )

    def _get_obs_rel_bodylink_pos(self):
        return self._robot_rel_body_pos_t

    def _get_obs_rel_bodylink_pos_flat(self):
        return self._robot_rel_body_pos_t.reshape(self.num_envs, -1)

    def _get_obs_rel_bodylink_rot_tannorm(self):
        return self._robot_rel_body_rot_tannorm_t

    def _get_obs_rel_bodylink_rot_tannorm_flat(self):
        return self._robot_rel_body_rot_tannorm_t.reshape(self.num_envs, -1)

    def _get_obs_rel_bodylink_vel(self):
        return self._robot_rel_body_vel_t

    def _get_obs_rel_bodylink_vel_flat(self):
        return self._robot_rel_body_vel_t.reshape(self.num_envs, -1)

    def _get_obs_rel_bodylink_ang_vel(self):
        return self._robot_rel_body_ang_vel_t

    def _get_obs_rel_bodylink_ang_vel_flat(self):
        return self._robot_rel_body_ang_vel_t.reshape(self.num_envs, -1)

    def _get_obs_root_rel_bodylink_pos(self):
        return self._robot_root_rel_body_pos_t

    def _get_obs_root_rel_bodylink_pos_flat(self):
        return self._robot_root_rel_body_pos_t.reshape(self.num_envs, -1)

    def _get_obs_root_rel_bodylink_rot_mat_flat(self):
        return self._robot_root_rel_body_rot_mat_t[:, :, :2].reshape(self.num_envs, -1)

    def _get_obs_root_rel_bodylink_rot_tannorm(self):
        return self._robot_root_rel_body_rot_tannorm_t

    def _get_obs_root_rel_bodylink_rot_tannorm_flat(self):
        return self._robot_root_rel_body_rot_tannorm_t.reshape(self.num_envs, -1)

    def _get_obs_root_rel_bodylink_vel(self):
        return self._robot_root_rel_body_vel_t

    def _get_obs_root_rel_bodylink_vel_flat(self):
        return self._robot_root_rel_body_vel_t.reshape(self.num_envs, -1)

    def _get_obs_root_rel_bodylink_ang_vel(self):
        return self._robot_root_rel_body_ang_vel_t

    def _get_obs_root_rel_bodylink_ang_vel_flat(self):
        return self._robot_root_rel_body_ang_vel_t.reshape(self.num_envs, -1)

    def _get_obs_rel_fut_ref_motion_state_flat(self):
        num_bodies_extend = self._robot_rel_body_pos_t.shape[1]
        num_fut_timesteps = self.ref_body_pos_fut.shape[1]

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4), w_last=True
        )  # [B*T, 4]
        ref_fut_heading_aligned_frame_quat = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_heading_aligned_frame_quat,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]

        ref_initial_heading_inv_quat_expanded = (
            self.ref_initial_heading_inv_quat[:, None, :]
            .repeat(1, num_fut_timesteps, 1)
            .reshape(-1, 4)
        )  # Shape [B*T, 4]

        q_ref_fut_in_initial_ref_heading = quat_mul(
            ref_initial_heading_inv_quat_expanded,
            self.ref_base_rot_fut.reshape(-1, 4),  # Use original future world rotation
            w_last=True,
        )  # Shape [B*T, 4]

        _, _, ref_fut_yaw_flat = (
            get_euler_xyz(  # ref_fut_yaw_flat was missing _flat suffix
                q_ref_fut_in_initial_ref_heading,
                w_last=True,
            )
        )
        ref_fut_yaw = wrap_to_pi(ref_fut_yaw_flat).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # Shape [B, T, 1]

        ref_fut_rpy = torch.cat(
            [ref_fut_roll, ref_fut_pitch, ref_fut_yaw], dim=-1
        )  # [B, T, 3]
        ref_fut_rpy_flat = ref_fut_rpy.reshape(self.num_envs, -1)  # [B, T * 3]
        # ---

        # --- get the current ref base rotation for later transformation ---
        cur_ref_base_pos = self.ref_root_global_pos_t  # [B, 3]
        cur_ref_base_rot_quat = self.ref_root_global_rot_quat_t  # [B, 4]

        cur_ref_base_rot_heading_quat_inv = calc_heading_quat_inv(
            cur_ref_base_rot_quat, w_last=True
        )  # [B, 4]
        cur_ref_base_rot_heading_quat_inv_body_flat = (
            cur_ref_base_rot_heading_quat_inv[
                :, None, None, :
            ]  # Corrected: Use the calculated inverse heading
            .repeat(1, num_fut_timesteps, num_bodies_extend, 1)
            .reshape(-1, 4)
        )
        fut_ref_bodylink_pos = self.ref_body_pos_fut
        fut_rel_ref_bodylink_pos_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            (fut_ref_bodylink_pos - cur_ref_base_pos[:, None, None, :]).view(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        fut_ref_bodylink_rot = self.ref_body_rot_fut
        fut_rel_ref_bodylink_rot_quat = quat_mul(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_rot.reshape(-1, 4),
            w_last=True,
        ).reshape(-1, 4)  # [B, num_fut_timesteps * num_bodies_extend * 4]
        fut_rel_ref_bodylink_rot_tannorm_flat = quat_to_tan_norm(
            fut_rel_ref_bodylink_rot_quat, w_last=True
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 6]
        fut_ref_bodylink_vel = self.ref_body_vel_fut
        fut_rel_ref_bodylink_vel_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        fut_ref_bodylink_ang_vel = self.ref_body_ang_vel_fut
        fut_rel_ref_bodylink_ang_vel_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        # ---

        fut_ref_rel_dof_pos_flat = self.ref_dof_pos_fut.reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)

        cur_ref_base_rot_heading_quat_inv_fut_flat = (
            cur_ref_base_rot_heading_quat_inv.reshape(-1, 4)
            .repeat(1, num_fut_timesteps, 1)
            .reshape(-1, 4)
        )

        fut_ref_rel_base_lin_vel = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_fut_flat,
            self.ref_base_lin_vel_fut.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]

        fut_ref_rel_base_ang_vel = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_fut_flat,
            self.ref_base_ang_vel_fut.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]

        rel_fut_ref_motion_state_flat = torch.cat(
            [
                ref_fut_rpy_flat,  # [B, T * 3]
                fut_ref_rel_base_lin_vel,  # [B, T * 3]
                fut_ref_rel_base_ang_vel,  # [B, T * 3]
                fut_ref_rel_dof_pos_flat,  # [B, T * num_dofs]
                fut_ref_rel_dof_vel_flat,  # [B, T * num_dofs]
                fut_rel_ref_bodylink_pos_flat,  # [B, T * num_bodies_extend * 3]  # noqa: E501
                fut_rel_ref_bodylink_rot_tannorm_flat,  # [B, T * num_bodies_extend * 6]  # noqa: E501
                fut_rel_ref_bodylink_vel_flat,  # [B, T * num_bodies_extend * 3]  # noqa: E501
                fut_rel_ref_bodylink_ang_vel_flat,  # [B, T * num_bodies_extend * 3]  # noqa: E501
            ],
            dim=-1,
        )

        return rel_fut_ref_motion_state_flat  # [B, (2*num_dofs + (3 + 6 + 3 + 3)*num_exntended_bodies)*T]  # noqa: E501

    def _get_obs_cur_priocep_v1(self):
        return torch.cat(
            [
                self._get_obs_base_rpy(),  # [B, 3]
                self._get_obs_rel_base_lin_vel(),  # [B, 3]
                self._get_obs_rel_base_ang_vel(),  # [B, 3]
                self._get_obs_dof_pos(),  # [B, num_dofs]
                self._get_obs_dof_vel(),  # [B, num_dofs]
                self._get_obs_actions(),  # [B, num_actions]
                self._get_obs_root_rel_bodylink_pos_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_rot_tannorm_flat(),  # [B, num_bodies_extend * 6]
                self._get_obs_root_rel_bodylink_vel_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_ang_vel_flat(),  # [B, num_bodies_extend * 3]
            ],
            dim=-1,
        )[:, None, :]

    def _get_obs_cur_priv_priocep_v1(self):
        return torch.cat(
            [
                self._get_obs_base_rpy(),  # [B, 3]
                self._get_obs_rel_base_lin_vel(),  # [B, 3]
                self._get_obs_rel_base_ang_vel(),  # [B, 3]
                self._get_obs_dof_pos(),  # [B, num_dofs]
                self._get_obs_dof_vel(),  # [B, num_dofs]
                self._get_obs_actions(),  # [B, num_actions]
                self._get_obs_root_rel_bodylink_pos_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_rot_mat_flat(),  # [B, num_bodies_extend * 6]
                self._get_obs_root_rel_bodylink_vel_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_ang_vel_flat(),  # [B, num_bodies_extend * 3]
            ],
            dim=-1,
        )[:, None, :]

    def _get_obs_cur_priocep_v1(self):
        return torch.cat(
            [
                self._get_obs_base_rpy(),  # [B, 3]
                self._get_obs_rel_base_lin_vel(),  # [B, 3]
                self._get_obs_rel_base_ang_vel(),  # [B, 3]
                self._get_obs_dof_pos(),  # [B, num_dofs]
                self._get_obs_dof_vel(),  # [B, num_dofs]
                self._get_obs_actions(),  # [B, num_actions]
                self._get_obs_root_rel_bodylink_pos_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_rot_tannorm_flat(),  # [B, num_bodies_extend * 6]
                self._get_obs_root_rel_bodylink_vel_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_ang_vel_flat(),  # [B, num_bodies_extend * 3]
            ],
            dim=-1,
        )[:, None, :]

    def _get_obs_cur_priv_priocep_v2(self):
        cur_root_global_lin_vel = self._get_obs_root_global_lin_vel()  # [B, 3]
        cur_root_global_ang_vel = self._get_obs_root_global_ang_vel()  # [B, 3]
        root_relative_reference_root_pos = (
            self._get_obs_root_relative_reference_root_pos()
        )
        root_relative_reference_root_rot = (
            self._get_obs_root_relative_reference_root_rot()
        )
        return torch.cat(
            [
                cur_root_global_lin_vel,  # [B, 3]
                cur_root_global_ang_vel,  # [B, 3]
                root_relative_reference_root_pos,  # [B, 3]
                root_relative_reference_root_rot,  # [B, 6]
                self._get_obs_base_rpy(),  # [B, 3]
                self._get_obs_rel_base_lin_vel(),  # [B, 3]
                self._get_obs_rel_base_ang_vel(),  # [B, 3]
                self._get_obs_dof_pos(),  # [B, num_dofs]
                self._get_obs_dof_vel(),  # [B, num_dofs]
                self._get_obs_actions(),  # [B, num_actions]
                self._get_obs_root_rel_bodylink_pos_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_rot_mat_flat(),  # [B, num_bodies_extend * 6]
            ],
            dim=-1,
        )[:, None, :]

    def _get_obs_cur_priv_priocep_v3(self):
        cur_root_global_lin_vel = self._get_obs_root_global_lin_vel()  # [B, 3]
        cur_root_global_ang_vel = self._get_obs_root_global_ang_vel()  # [B, 3]
        root_relative_reference_root_pos = (
            self._get_obs_root_relative_reference_root_pos()
        )
        root_relative_reference_root_rot = (
            self._get_obs_root_relative_reference_root_rot()
        )
        root_height = self._get_obs_base_height()
        return torch.cat(
            [
                cur_root_global_lin_vel,  # [B, 3]
                cur_root_global_ang_vel,  # [B, 3]
                root_relative_reference_root_pos,  # [B, 3]
                root_relative_reference_root_rot,  # [B, 6]
                root_height,  # [B, 1]
                self._get_obs_base_rpy(),  # [B, 3]
                self._get_obs_rel_base_lin_vel(),  # [B, 3]
                self._get_obs_rel_base_ang_vel(),  # [B, 3]
                self._get_obs_dof_pos(),  # [B, num_dofs]
                self._get_obs_dof_vel(),  # [B, num_dofs]
                self._get_obs_actions(),  # [B, num_actions]
                self._get_obs_root_rel_bodylink_pos_flat(),  # [B, num_bodies_extend * 3]
                self._get_obs_root_rel_bodylink_rot_mat_flat(),  # [B, num_bodies_extend * 6]
            ],
            dim=-1,
        )[:, None, :]

    def _get_obs_fut_ref_root_rel_teacher_v2(self):
        """
        This observation is used for obtaining the future reference motion state
        for training the teacher policy. Notice that the future bodylink properties
        are expressed in the **current** root reference frame.

        - Root roll and pitch: in future per-frame heading-aligned frame
        - Root linear and angular velocity: in the **current** root reference frame
        - DoF position and velocity: in the absolute frame
        - Bodylink position, rotation, linear and angular velocity: in the **current** root reference frame
        """
        NB = self._robot_rel_body_pos_t.shape[1]
        FT = self.ref_body_pos_fut.shape[1]

        cur_root_quat = self.base_quat
        cur_root_quat_inv = quat_inverse(cur_root_quat, w_last=True)
        cur_root_quat_inv_fut_flat = (
            cur_root_quat_inv[:, None, :].repeat(1, FT, 1).view(-1, 4)
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, FT, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, FT, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(self.num_envs, -1)  # [B, T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            cur_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_base_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            cur_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_base_ang_vel_fut.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_rel_dof_pos_flat = self.ref_dof_pos_fut.reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        # ---

        # --- calculate the relative bodylink pos in the current root reference frame ---
        ref_fut_cur_root_quat_inv_body_flat = (
            cur_root_quat_inv_fut_flat[:, None, :].repeat(1, NB, 1).view(-1, 4)
        )

        fut_root_rel_ref_bodylink_pos_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            (self.ref_body_pos_fut - self.ref_root_global_pos_t[:, None, None, :]).view(
                -1, 3
            ),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        fut_root_rel_ref_bodylink_rot_tannorm = quat_mul(
            ref_fut_cur_root_quat_inv_body_flat,
            self.ref_body_rot_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(-1, 4)  # [B*num_bodies_extend*T, 4]
        fut_root_rel_ref_bodylink_rot_tannorm_flat = quat_to_tan_norm(
            fut_root_rel_ref_bodylink_rot_tannorm,
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 6]
        fut_root_rel_ref_bodylink_vel_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            self.ref_body_vel_fut.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        fut_root_rel_ref_bodylink_ang_vel_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            self.ref_body_ang_vel_fut.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * num_bodies_extend * 3]
        # ---

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(self.num_envs, FT, -1),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, 3]
                fut_ref_rel_dof_pos_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_dofs]
                fut_ref_rel_dof_vel_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_dofs]
                fut_root_rel_ref_bodylink_pos_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_bodies_extend * 3]
                fut_root_rel_ref_bodylink_rot_tannorm_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_bodies_extend * 6]
                fut_root_rel_ref_bodylink_vel_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_bodies_extend * 3]
                fut_root_rel_ref_bodylink_ang_vel_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_bodies_extend * 3]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies_extend * (3 + 6 + 3 + 3)]
        # import ipdb

        # ipdb.set_trace()
        return rel_fut_ref_motion_state_seq

    def _get_obs_priocep_with_fut_ref_v7_moe(self):
        cur_priocep = self._get_obs_cur_priocep_v1().squeeze(1)
        fut_ref = self._get_obs_fut_ref_root_rel_teacher_v2()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()
        fut_ref = fut_ref.reshape(self.num_envs, -1)
        domain_params = self._get_obs_domain_params()
        full_obs = torch.cat(
            [
                cur_priocep,  # 3 + 3 + 3 + n_dof*3 + n_bodies*(3+6+3+3)
                fut_ref,  # FT*(2+3+3+n_bodies*(3+6+3+3)+n_dof*2)
                domain_params,  # 3 + len(${robot.randomize_link_body_names}) + 2 * ${robot.dof_obs_size} + 1 + 1 + 1 + ${robot.dof_obs_size}
            ],
            dim=-1,
        )
        return full_obs

    def _get_obs_noisy_dof_with_history_seq(self):
        student_obs_noise_scales_dict = self.config.obs.student_obs_noise_scales

        hist_len = self.config.obs.actor_context_length
        cur_dof_pos = self.simulator.dof_pos
        hist_dof_pos = self.history_handler.query("dof_pos")[:, :hist_len]
        dof_pos_seq = torch.cat(
            [cur_dof_pos[:, None, :], hist_dof_pos], dim=1
        )  # [num_envs, hist_len + 1, num_dofs]

        if student_obs_noise_scales_dict.dof_pos > 0.0:
            dof_pos_seq = (
                dof_pos_seq
                + torch.randn_like(dof_pos_seq, device=dof_pos_seq.device)
                * student_obs_noise_scales_dict.dof_pos
            )

        cur_dof_vel = self.simulator.dof_vel
        hist_dof_vel = self.history_handler.query("dof_vel")[:, :hist_len]
        dof_vel_seq = torch.cat(
            [cur_dof_vel[:, None, :], hist_dof_vel], dim=1
        )  # [num_envs, hist_len + 1, num_dofs]

        if student_obs_noise_scales_dict.dof_vel > 0.0:
            dof_vel_seq = (
                dof_vel_seq
                + torch.randn_like(dof_vel_seq, device=dof_vel_seq.device)
                * student_obs_noise_scales_dict.dof_vel
            )

        dof_seq = torch.cat(
            [dof_pos_seq, dof_vel_seq], dim=-1
        )  # [num_envs, hist_len + 1, 2 * num_dofs]
        return dof_seq  # [num_envs, hist_len + 1, 2 * num_dofs]

    def _get_obs_noisy_imu_with_history_seq(self):
        student_obs_noise_scales_dict = self.config.obs.student_obs_noise_scales
        hist_len = self.config.obs.actor_context_length
        current_base_ang_vel = self._get_obs_base_ang_vel()  # [num_envs, 3]
        current_base_proj_gravity = self._get_obs_projected_gravity()  # [num_envs, 3]
        hist_base_ang_vel = self.history_handler.query("base_ang_vel")[:, :hist_len]
        hist_base_proj_gravity = self.history_handler.query("projected_gravity")[
            :, :hist_len
        ]

        if student_obs_noise_scales_dict.base_ang_vel > 0.0:
            current_base_ang_vel = (
                current_base_ang_vel
                + torch.randn_like(
                    current_base_ang_vel,
                    device=current_base_ang_vel.device,
                )
                * student_obs_noise_scales_dict.base_ang_vel
            )

        if student_obs_noise_scales_dict.projected_gravity > 0.0:
            current_base_proj_gravity = (
                current_base_proj_gravity
                + torch.randn_like(
                    current_base_proj_gravity,
                    device=current_base_proj_gravity.device,
                )
                * student_obs_noise_scales_dict.projected_gravity
            )

        imu_seq = torch.cat(
            [
                torch.cat(
                    [
                        current_base_ang_vel[:, None, :],
                        hist_base_ang_vel,
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        current_base_proj_gravity[:, None, :],
                        hist_base_proj_gravity,
                    ],
                    dim=1,
                ),
            ],
            dim=-1,
        )  # [num_envs, hist_len + 1, 6]
        return imu_seq  # [num_envs, hist_len + 1, 6]

    def _get_obs_action_with_history_seq(self):
        hist_len = self.config.obs.actor_context_length
        last_actions = self._get_obs_actions()  # [num_envs, num_dofs]
        hist_actions = self.history_handler.query("actions")[
            :, :hist_len
        ]  # [num_envs, hist_len]
        action_seq = torch.cat(
            [last_actions[:, None, :], hist_actions], dim=1
        )  # [num_envs, hist_len + 1, num_dofs]
        return action_seq  # [num_envs, hist_len + 1, num_dofs]

    def _get_obs_priocep_with_fut_ref_v7_student(self):
        dof_seq = self._get_obs_noisy_dof_with_history_seq()  # [B, HT + 1, num_dofs]
        imu_seq = self._get_obs_noisy_imu_with_history_seq()  # [B, HT + 1, 6]
        action_seq = self._get_obs_action_with_history_seq()  # [B, HT + 1, num_actions]

        hist_seq = torch.cat([dof_seq, imu_seq, action_seq], dim=-1)

        fut_ref = self._get_obs_fut_ref_root_rel_teacher_v2()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()
        return self.obs_serializer.serialize(
            [
                hist_seq,
                fut_ref,
                fut_ref_valid_mask,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v8_student(self):
        dof_seq = self._get_obs_noisy_dof_with_history_seq()  # [B, HT + 1, num_dofs]
        imu_seq = self._get_obs_noisy_imu_with_history_seq()  # [B, HT + 1, 6]
        action_seq = self._get_obs_action_with_history_seq()  # [B, HT + 1, num_actions]

        hist_seq = torch.cat([dof_seq, imu_seq, action_seq], dim=-1)

        fut_ref = self._get_obs_fut_ref_v11()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()
        return self.obs_serializer.serialize(
            [
                hist_seq,
                fut_ref,
                fut_ref_valid_mask,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v9_student(self):
        dof_seq = self._get_obs_noisy_dof_with_history_seq()  # [B, HT + 1, num_dofs]
        imu_seq = self._get_obs_noisy_imu_with_history_seq()  # [B, HT + 1, 6]
        action_seq = self._get_obs_action_with_history_seq()  # [B, HT + 1, num_actions]

        hist_seq = torch.cat([dof_seq, imu_seq, action_seq], dim=-1)

        fut_ref = self._get_obs_fut_ref_v12()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()
        return self.obs_serializer.serialize(
            [
                hist_seq,
                fut_ref,
                fut_ref_valid_mask,
            ]
        )

    def _get_obs_root_global_lin_vel(self):
        return self.simulator.robot_root_states[:, 7:10]

    def _get_obs_root_global_ang_vel(self):
        return self.simulator.robot_root_states[:, 10:13]

    def _get_obs_root_relative_reference_root_pos(self):
        ref_root_global_pos = self.ref_root_global_pos_t
        cur_root_global_pos = self.simulator.robot_root_states[:, :3]
        diff_root_global_pos = ref_root_global_pos - cur_root_global_pos
        # transform the difference vector with current root rotation
        cur_root_quat = self.simulator.robot_root_states[:, 3:7]
        diff_root_global_pos = quat_rotate(
            cur_root_quat,
            diff_root_global_pos,
            w_last=True,
        )
        return diff_root_global_pos

    def _get_obs_root_relative_reference_root_rot(self):
        ref_root_rot_quat = self.ref_root_global_rot_quat_t
        cur_root_rot_quat = self.simulator.robot_root_states[:, 3:7]
        diff_root_rot_quat = quat_mul(
            quat_inverse(cur_root_rot_quat, w_last=True),
            ref_root_rot_quat,
            w_last=True,
        )
        diff_root_rot_mat = quaternion_to_matrix(diff_root_rot_quat, w_last=True)
        return diff_root_rot_mat[..., :2].reshape(diff_root_rot_mat.shape[0], -1)

    def _get_obs_beyondmimic_actor(self):
        # reference motion state
        fut_ref_rel_dof_pos_flat = self.ref_dof_pos_fut.reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        fut_ref = torch.cat(
            [fut_ref_rel_dof_pos_flat, fut_ref_rel_dof_vel_flat], dim=-1
        )

        # motion_anchor_pos/rot
        root_relative_reference_root_pos = (
            self._get_obs_root_relative_reference_root_pos()
        )
        root_relative_reference_root_rot = (
            self._get_obs_root_relative_reference_root_rot()
        )

        cur_root_global_lin_vel = self._get_obs_root_global_lin_vel()
        cur_root_global_ang_vel = self._get_obs_root_global_ang_vel()

        cur_dof_pos = self._get_obs_dof_pos()
        cur_dof_vel = self._get_obs_dof_vel()
        prev_actions = self._get_obs_actions()

        return self.obs_serializer.serialize(
            [
                torch.cat(
                    [
                        fut_ref[:, None, :],  # [B, 1, num_dofs * 2]
                        root_relative_reference_root_pos[:, None, :],  # [B, 1, 3]
                        root_relative_reference_root_rot[:, None, :],  # [B, 1, 6]
                        cur_root_global_lin_vel[:, None, :],  # [B, 1, 3]
                        cur_root_global_ang_vel[:, None, :],  # [B, 1, 3]
                        cur_dof_pos[:, None, :],  # [B, 1, num_dofs]
                        cur_dof_vel[:, None, :],  # [B, 1, num_dofs]
                        prev_actions[:, None, :],  # [B, 1, num_actions]
                    ],
                    dim=-1,
                )
            ]
        )

    def _get_obs_beyondmimic_teacher_actor(self):
        # reference motion state
        fut_ref_rel_dof_pos_flat = self.ref_dof_pos_fut.reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        fut_ref = torch.cat(
            [fut_ref_rel_dof_pos_flat, fut_ref_rel_dof_vel_flat], dim=-1
        )

        # motion_anchor_pos/rot
        root_relative_reference_root_pos = (
            self._get_obs_root_relative_reference_root_pos()
        )
        root_relative_reference_root_rot = (
            self._get_obs_root_relative_reference_root_rot()
        )

        cur_root_global_lin_vel = self._get_obs_root_global_lin_vel()
        cur_root_global_ang_vel = self._get_obs_root_global_ang_vel()

        cur_dof_pos = self._get_obs_dof_pos()
        cur_dof_vel = self._get_obs_dof_vel()
        prev_actions = self._get_obs_actions()

        return self.teacher_obs_serializer.serialize(
            [
                torch.cat(
                    [
                        fut_ref[:, None, :],  # [B, 1, num_dofs * 2]
                        root_relative_reference_root_pos[:, None, :],  # [B, 1, 3]
                        root_relative_reference_root_rot[:, None, :],  # [B, 1, 6]
                        cur_root_global_lin_vel[:, None, :],  # [B, 1, 3]
                        cur_root_global_ang_vel[:, None, :],  # [B, 1, 3]
                        cur_dof_pos[:, None, :],  # [B, 1, num_dofs]
                        cur_dof_vel[:, None, :],  # [B, 1, num_dofs]
                        prev_actions[:, None, :],  # [B, 1, num_actions]
                    ],
                    dim=-1,
                )
            ]
        )

    def _get_obs_beyondmimic_critic(self):
        # reference motion state
        fut_ref_rel_dof_pos_flat = self.ref_dof_pos_fut.reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        fut_ref = torch.cat(
            [fut_ref_rel_dof_pos_flat, fut_ref_rel_dof_vel_flat], dim=-1
        )

        # motion_anchor_pos/rot
        root_relative_reference_root_pos = (
            self._get_obs_root_relative_reference_root_pos()
        )
        root_relative_reference_root_rot = (
            self._get_obs_root_relative_reference_root_rot()
        )

        cur_root_global_lin_vel = self._get_obs_root_global_lin_vel()
        cur_root_global_ang_vel = self._get_obs_root_global_ang_vel()

        cur_dof_pos = self._get_obs_dof_pos()
        cur_dof_vel = self._get_obs_dof_vel()
        prev_actions = self._get_obs_actions()

        # root relative body pos/rot
        root_relative_body_pos = self._get_obs_root_rel_bodylink_pos_flat()
        root_relative_body_rot = self._get_obs_root_rel_bodylink_rot_mat_flat()

        return self.critic_obs_serializer.serialize(
            [
                torch.cat(
                    [
                        fut_ref[:, None, :],  # [B, 1, num_dofs * 2]
                        root_relative_reference_root_pos[:, None, :],  # [B, 1, 3]
                        root_relative_reference_root_rot[:, None, :],  # [B, 1, 6]
                        cur_root_global_lin_vel[:, None, :],  # [B, 1, 3]
                        cur_root_global_ang_vel[:, None, :],  # [B, 1, 3]
                        cur_dof_pos[:, None, :],  # [B, 1, num_dofs]
                        cur_dof_vel[:, None, :],  # [B, 1, num_dofs]
                        prev_actions[:, None, :],  # [B, 1, num_actions]
                        root_relative_body_pos[:, None, :],  # [B, 1, num_bodies * 3]
                        root_relative_body_rot[:, None, :],  # [B, 1, num_bodies * 6]
                    ],
                    dim=-1,
                )
            ]
        )

    def _get_obs_global_root_rot(self):
        return self.simulator.robot_root_states[:, 3:7]

    def _get_obs_global_root_pos(self):
        return self.simulator.robot_root_states[:, :3]

    def _get_obs_global_root_lin_vel(self):
        return self.simulator.robot_root_states[:, 7:10]

    def _get_obs_global_root_ang_vel(self):
        return self.simulator.robot_root_states[:, 10:13]

    def _get_obs_global_bodylink_pos(self):
        return self._rigid_body_pos_extend.reshape(self.num_envs, -1)

    def _get_obs_global_bodylink_rot(self):
        return self._rigid_body_rot_extend.reshape(self.num_envs, -1)

    def _get_obs_global_bodylink_vel(self):
        return self._rigid_body_vel_extend.reshape(self.num_envs, -1)

    def _get_obs_global_bodylink_ang_vel(self):
        return self._rigid_body_ang_vel_extend.reshape(self.num_envs, -1)

    def _get_obs_hist_priocep_seq_v9(self):
        HT = self.config.obs.context_length
        NB = self.config.robot.num_extend_bodies + self.config.robot.num_bodies

        cur_global_root_pos = self._get_obs_global_root_pos()
        cur_global_root_rot = self._get_obs_global_root_rot()
        cur_global_root_rot_inv = quat_inverse(
            cur_global_root_rot, w_last=True
        )  # [B, 4]
        cur_global_root_rot_inv_hist_flat = (
            cur_global_root_rot_inv[:, None, :].repeat(1, HT, 1).view(-1, 4)
        )  # [B*hist_len, 4]
        cur_heading_aligned_root_rot_inv = self.cur_heading_inv_quat  # [B, 4]
        cur_heading_aligned_root_rot_inv_hist_flat = (
            cur_heading_aligned_root_rot_inv[:, None, :].repeat(1, HT, 1).view(-1, 4)
        )  # [B*hist_len, 4]

        # get root position
        hist_global_root_pos = self.history_handler.query("global_root_pos")[:, :HT]
        hist_rel_root_pos_seq = (
            hist_global_root_pos - cur_global_root_pos[:, None, :]
        )  # [B, hist_len, 3]

        # get root rotation
        hist_global_root_rot = self.history_handler.query("global_root_rot")[:, :HT]
        relative_root_rotation = quat_mul(
            cur_global_root_rot_inv_hist_flat,
            hist_global_root_rot.reshape(-1, 4),
            w_last=True,
        )  # [B*hist_len, 4]
        relative_root_rot_tannorm = quat_to_tan_norm(
            relative_root_rotation, w_last=True
        )  # [B*hist_len, 6]
        hist_rel_root_rot_tannorm_seq = relative_root_rot_tannorm.reshape(
            self.num_envs, HT, -1
        )  # [B, hist_len, 6]

        # get root roll, pitch and yaw
        hist_root_ryp = self.history_handler.query("base_rpy")[
            :, :HT
        ]  # [B, hist_len, 3]

        # get heading-aligned root linear velocity
        hist_global_root_lin_vel = self.history_handler.query("global_root_lin_vel")[
            :, :HT
        ]
        hist_root_lin_vel_seq = quat_rotate(
            cur_heading_aligned_root_rot_inv_hist_flat,
            hist_global_root_lin_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, HT, -1)  # [B, hist_len, 3]

        # get heading-aligned root angular velocity
        hist_global_root_ang_vel = self.history_handler.query("global_root_ang_vel")[
            :, :HT
        ]
        hist_root_ang_vel_seq = quat_rotate(
            cur_heading_aligned_root_rot_inv_hist_flat,
            hist_global_root_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(self.num_envs, HT, -1)  # [B, hist_len, 3]

        # get DoF position
        hist_dof_pos_seq = self.history_handler.query("dof_pos")[
            :, :HT
        ]  # [B, hist_len, num_dofs]

        # get DoF velocity
        hist_dof_vel_seq = self.history_handler.query("dof_vel")[
            :, :HT
        ]  # [B, hist_len, num_dofs]

        # get actions
        hist_actions_seq = self.history_handler.query("actions")[
            :, :HT
        ]  # [B, hist_len, num_actions]

        full_seq = torch.cat(
            [
                hist_rel_root_pos_seq,  # [B, hist_len, 3]
                hist_rel_root_rot_tannorm_seq,  # [B, hist_len, 6]
                hist_root_ryp,  # [B, hist_len, 3]
                hist_root_lin_vel_seq,  # [B, hist_len, 3]
                hist_root_ang_vel_seq,  # [B, hist_len, 3]
                hist_dof_pos_seq,  # [B, hist_len, num_dofs]
                hist_dof_vel_seq,  # [B, hist_len, num_dofs]
                hist_actions_seq,  # [B, hist_len, num_actions]
            ],
            dim=-1,
        )  # [B, HT, D]
        return full_seq

    def _get_obs_hist_valid_mask(self):
        HT = self.config.obs.context_length
        return self.history_handler.query_valid_mask("dof_pos")[:, :HT]

    def _get_obs_fut_ref_v9(self):
        """
        This observation is used for obtaining the future reference motion state
        for training the teacher policy. Notice that the future bodylink properties
        are expressed in the **current** root reference frame.

        - Root roll and pitch: in future per-frame heading-aligned frame
        - Root linear and angular velocity: in the **heading aligned** reference frame
        - DoF position and velocity: in the absolute frame, default dof pos should be subtracted
        """
        FT = self.ref_body_pos_fut.shape[1]

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, FT, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, FT, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(self.num_envs, -1)  # [B, T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_rel_dof_pos_flat = (
            self.ref_dof_pos_fut - self.default_dof_pos[:, None, :]
        ).reshape(self.num_envs, -1)
        fut_ref_rel_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        # ---

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(self.num_envs, FT, -1),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, 3]
                fut_ref_rel_dof_pos_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_dofs]
                fut_ref_rel_dof_vel_flat.reshape(
                    self.num_envs, FT, -1
                ),  # [B, T, num_dofs]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2]
        return rel_fut_ref_motion_state_seq

    def _get_obs_fut_ref_v10(self):
        num_fut_timesteps = self.ref_body_pos_fut.shape[1]
        num_bodies = self.ref_body_pos_fut.shape[2]

        fut_ref_root_rot_quat = self.ref_base_rot_fut  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(self.num_envs, -1)  # [B, T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_ha_base_lin_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_ha_base_ang_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = (
            self.ref_dof_pos_fut - self.default_dof_pos[:, None, :]
        ).reshape(self.num_envs, -1)
        fut_ref_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = self.ref_body_pos_fut  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = self.ref_body_rot_fut  # [B, T, num_bodies, 4]
        fut_ref_global_bodylink_vel = self.ref_body_vel_fut  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_ang_vel = (
            self.ref_body_ang_vel_fut
        )  # [B, T, num_bodies, 3]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        # get root-relative bodylink velocity
        fut_ref_root_rel_bodylink_vel = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink angular velocity
        fut_ref_root_rel_bodylink_ang_vel = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # ---

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_ha_base_lin_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_ha_base_ang_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
                fut_ref_root_rel_bodylink_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_ang_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6 + 3 + 3)]
        return rel_fut_ref_motion_state_seq

    def _get_obs_fut_ref_v11(self):
        num_fut_timesteps = self.ref_body_pos_fut.shape[1]
        num_bodies = self.ref_body_pos_fut.shape[2]

        fut_ref_root_rot_quat = self.ref_base_rot_fut  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(self.num_envs, -1)  # [B, T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_ha_base_lin_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_ha_base_ang_vel = quat_rotate(
            ref_fut_heading_quat_inv,  # [B*T, 4]
            self.ref_base_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = (
            self.ref_dof_pos_fut - self.default_dof_pos[:, None, :]
        ).reshape(self.num_envs, -1)
        fut_ref_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = self.ref_body_pos_fut  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = self.ref_body_rot_fut  # [B, T, num_bodies, 4]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        # ---

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_ha_base_lin_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_ha_base_ang_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6)]
        return rel_fut_ref_motion_state_seq

    def _get_obs_fut_ref_v12(self):
        num_fut_timesteps = self.ref_body_pos_fut.shape[1]
        num_bodies = self.ref_body_pos_fut.shape[2]

        fut_ref_root_rot_quat = self.ref_base_rot_fut  # [B, T, 4]
        fut_ref_root_rot_quat_inv = quat_inverse(
            fut_ref_root_rot_quat, w_last=True
        )  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            self.ref_base_rot_fut.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            self.num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(self.num_envs, -1)  # [B, T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            fut_ref_root_rot_quat_inv.reshape(-1, 4),  # [B*T, 4]
            self.ref_base_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            fut_ref_root_rot_quat_inv.reshape(-1, 4),  # [B*T, 4]
            self.ref_base_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = (
            self.ref_dof_pos_fut - self.default_dof_pos[:, None, :]
        ).reshape(self.num_envs, -1)
        fut_ref_dof_vel_flat = self.ref_dof_vel_fut.reshape(self.num_envs, -1)
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = self.ref_body_pos_fut  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = self.ref_body_rot_fut  # [B, T, num_bodies, 4]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            self.num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        # ---

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    self.num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6)]
        return rel_fut_ref_motion_state_seq

    def _get_obs_priocep_with_fut_ref_v9_teacher(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priocep_v1()

        # historical priocep
        hist_priocep = self._get_obs_hist_priocep_seq_v9()
        hist_valid_mask = self._get_obs_hist_valid_mask()[:, :, None]
        hist_priocep = hist_priocep * hist_valid_mask.float()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v9()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.obs_serializer.serialize(
            [
                cur_priocep,
                hist_priocep,
                hist_valid_mask,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v10_teacher(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priv_priocep_v1()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v10()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.obs_serializer.serialize(
            [
                cur_priocep,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v11_teacher(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priv_priocep_v2()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v10()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.obs_serializer.serialize(
            [
                cur_priocep,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v12_teacher(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priv_priocep_v2()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v11()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.obs_serializer.serialize(
            [
                cur_priocep,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v12_teacher_distill(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priv_priocep_v2()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v11()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.teacher_obs_serializer.serialize(
            [
                cur_priocep,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    def _get_obs_priocep_with_fut_ref_v13_teacher(self):
        # current timestep priocep
        cur_priocep = self._get_obs_cur_priv_priocep_v3()

        # future reference motion state
        fut_ref = self._get_obs_fut_ref_v11()
        fut_ref_valid_mask = self.ref_fut_valid_mask[:, :, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        # domain parameters
        domain_params = self._get_obs_domain_params()[:, None, :]

        return self.obs_serializer.serialize(
            [
                cur_priocep,
                fut_ref,
                fut_ref_valid_mask,
                domain_params,
            ]
        )

    ########### Rewards ###########
    @torch.compile
    def _reward_teleop_body_position_extend(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(
            -diff_body_pos_dist_upper
            / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos
        )
        r_body_pos_lower = torch.exp(
            -diff_body_pos_dist_lower
            / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos
        )
        r_body_pos = (
            r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight
            + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
        )

        return r_body_pos

    @torch.compile
    def _reward_huber_teleop_body_position_extend(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (
            F.huber_loss(
                upper_body_diff,
                torch.zeros_like(upper_body_diff, device=upper_body_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )

        diff_body_pos_dist_lower = (
            F.huber_loss(
                lower_body_diff,
                torch.zeros_like(lower_body_diff, device=lower_body_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )

        r_body_pos_upper = torch.exp(
            -diff_body_pos_dist_upper
            / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos
        )
        r_body_pos_lower = torch.exp(
            -diff_body_pos_dist_lower
            / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos
        )
        r_body_pos = (
            r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight
            + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
        )

        return r_body_pos

    @torch.compile
    def _reward_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(
            -vr_3point_dist
            / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos
        )
        return r_vr_3point

    @torch.compile
    def _reward_huber_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (
            F.huber_loss(
                vr_3point_diff,
                torch.zeros_like(vr_3point_diff, device=vr_3point_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_vr_3point = torch.exp(
            -vr_3point_dist
            / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos
        )
        return r_vr_3point

    @torch.compile
    def _reward_teleop_body_position_feet(self):
        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(
            -feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos
        )
        return r_feet

    @torch.compile
    def _reward_huber_teleop_body_position_feet(self):
        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        feet_dist = (
            F.huber_loss(
                feet_diff,
                torch.zeros_like(feet_diff, device=feet_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_feet = torch.exp(
            -feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos
        )
        return r_feet

    @torch.compile
    def _reward_teleop_body_rotation_extend(self):
        rotation_diff = self.dif_global_body_rot
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_rot = torch.exp(
            -diff_body_rot_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_rot
        )
        return r_body_rot

    @torch.compile
    def _reward_huber_teleop_body_rotation_extend(self):
        rotation_diff = self.dif_global_body_rot
        diff_body_rot_dist = (
            F.huber_loss(
                rotation_diff,
                torch.zeros_like(rotation_diff, device=rotation_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_body_rot = torch.exp(
            -diff_body_rot_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_rot
        )
        return r_body_rot

    @torch.compile
    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(
            -diff_body_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_vel
        )
        return r_body_vel

    @torch.compile
    def _reward_huber_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel
        diff_body_vel_dist = (
            F.huber_loss(
                velocity_diff,
                torch.zeros_like(velocity_diff, device=velocity_diff.device),
                delta=1.0,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_body_vel = torch.exp(
            -diff_body_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_vel
        )
        return r_body_vel

    @torch.compile
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(
            -diff_body_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel
        )
        return r_body_ang_vel

    @torch.compile
    def _reward_huber_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (
            F.huber_loss(
                ang_velocity_diff,
                torch.zeros_like(ang_velocity_diff, device=ang_velocity_diff.device),
                delta=0.25,
                reduction="none",
            )
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_body_ang_vel = torch.exp(
            -diff_body_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel
        )
        return r_body_ang_vel

    @torch.compile
    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(
            -diff_joint_pos_dist
            / self.config.rewards.reward_tracking_sigma.teleop_joint_pos
        )
        return r_joint_pos

    @torch.compile
    def _reward_huber_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = F.huber_loss(
            joint_pos_diff,
            torch.zeros_like(joint_pos_diff, device=joint_pos_diff.device),
            delta=0.25,
            reduction="none",
        ).mean(dim=-1)
        r_joint_pos = torch.exp(
            -diff_joint_pos_dist
            / self.config.rewards.reward_tracking_sigma.teleop_joint_pos
        )
        return r_joint_pos

    @torch.compile
    def _reward_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(
            -diff_joint_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_joint_vel
        )
        return r_joint_vel

    @torch.compile
    def _reward_huber_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = F.huber_loss(
            joint_vel_diff,
            torch.zeros_like(joint_vel_diff, device=joint_vel_diff.device),
            delta=1.0,
            reduction="none",
        ).mean(dim=-1)
        r_joint_vel = torch.exp(
            -diff_joint_vel_dist
            / self.config.rewards.reward_tracking_sigma.teleop_joint_vel
        )
        return r_joint_vel

    @torch.compile
    def _reward_penalty_upper_body_action_smooth(self):
        # only calculate the action smoothness of the upper body
        upper_body_actions = self.actions[:, self.upper_body_joint_ids]
        upper_body_last_actions = self.last_actions[:, self.upper_body_joint_ids]
        upper_body_last_last_actions = self.last_last_actions[
            :, self.upper_body_joint_ids
        ]
        upper_body_action_smooth = torch.mean(
            torch.square(
                (upper_body_actions - upper_body_last_actions)
                - (upper_body_last_actions - upper_body_last_last_actions)
            ),
            dim=-1,
        )
        return upper_body_action_smooth

    @torch.compile
    def _reward_upper_body_teleop_joint_position(self):
        upper_body_joint_pos_diff = self.dif_joint_angles[:, self.upper_body_joint_ids]
        diff_upper_body_joint_pos_dist = (upper_body_joint_pos_diff**2).mean(dim=-1)
        r_upper_body_joint_pos = torch.exp(
            -diff_upper_body_joint_pos_dist
            / self.config.rewards.reward_tracking_sigma.teleop_joint_pos
        )
        return r_upper_body_joint_pos

    @torch.compile
    def _reward_rel_tracking_root_lin_vel(self):
        diff_root_lin_vel_dist = (self.dif_local_root_lin_vel_t**2).mean(dim=-1)
        r_root_lin_vel = torch.exp(
            -diff_root_lin_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_root_lin_vel", 0.03
            )
        )
        return r_root_lin_vel

    @torch.compile
    def _reward_rel_tracking_root_global_lin_vel(self):
        diff_root_lin_vel_dist = (self.dif_local_root_lin_vel_t**2).mean(dim=-1)
        r_root_lin_vel = torch.exp(
            -diff_root_lin_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_root_lin_vel", 0.03
            )
        )
        return r_root_lin_vel

    @torch.compile
    def _reward_rel_tracking_root_ang_vel(self):
        diff_root_ang_vel_dist = (self.dif_local_root_ang_vel_t**2).mean(dim=-1)
        r_root_ang_vel = torch.exp(
            -diff_root_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_root_ang_vel", 0.1
            )
        )
        return r_root_ang_vel

    @torch.compile
    def _reward_tracking_root_rpy(self):
        diff_root_rpy_dist = (self.dif_base_rpy_t.square()).mean(dim=-1)
        r_root_rpy = torch.exp(
            -diff_root_rpy_dist
            / self.config.rewards.reward_tracking_sigma.get("tracking_root_rpy", 0.1)
        )
        return r_root_rpy

    @torch.compile
    def _reward_tracking_root_rp(self):
        diff_root_rp_dist = (self.dif_base_rpy_t[..., :2].square()).mean(dim=-1)
        r_root_rp = torch.exp(
            -diff_root_rp_dist
            / self.config.rewards.reward_tracking_sigma.get("tracking_root_rp", 0.1)
        )
        return r_root_rp

    @torch.compile
    def _reward_rel_tracking_body_pos(self):
        diff_body_pos_dist = (self.dif_local_body_pos_t**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(
            -diff_body_pos_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_body_pos", 0.01
            )
        )
        return r_body_pos

    @torch.compile
    def _reward_rel_tracking_keybody_pos(self):
        diff_body_pos_dist = (
            (self.dif_local_body_pos_t[:, self.key_body_indices, :] ** 2)
            .mean(dim=-1)
            .mean(dim=-1)
        )
        r_body_pos = torch.exp(
            -diff_body_pos_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_keybody_pos", 0.01
            )
        )
        return r_body_pos

    @torch.compile
    def _reward_rel_tracking_body_rot(self):
        diff_body_rot_dist = (
            (self.dif_local_body_rot_tannorm**2).mean(dim=-1).mean(dim=-1)
        )
        r_body_rot = torch.exp(
            -diff_body_rot_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_body_rot", 0.03
            )
        )
        return r_body_rot

    @torch.compile
    def _reward_rel_tracking_body_vel(self):
        diff_body_vel_dist = (self.dif_local_body_vel_t**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(
            -diff_body_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_body_vel", 0.01
            )
        )
        return r_body_vel

    @torch.compile
    def _reward_rel_tracking_body_ang_vel(self):
        diff_body_ang_vel_dist = (
            (self.dif_local_body_ang_vel_t**2).mean(dim=-1).mean(dim=-1)
        )
        r_body_ang_vel = torch.exp(
            -diff_body_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "rel_tracking_body_ang_vel", 0.01
            )
        )
        return r_body_ang_vel

    @torch.compile
    def _reward_l1_rel_tracking_keybody_pos(self):
        # Ensure key_body_indices are tensor for advanced indexing
        key_body_indices_tensor = torch.tensor(
            self.key_body_indices, device=self.device, dtype=torch.long
        )

        # Filter dif_local_body_pos_t for key bodies
        key_body_pos_diff = self.dif_local_body_pos_t[
            :, key_body_indices_tensor, :
        ]  # [B, N_key, 3]

        upper_body_ids_tensor = torch.tensor(
            self.upper_body_id, device=self.device, dtype=torch.long
        )
        lower_body_ids_tensor = torch.tensor(
            self.lower_body_id, device=self.device, dtype=torch.long
        )

        is_upper_key_body = torch.isin(key_body_indices_tensor, upper_body_ids_tensor)
        is_lower_key_body = torch.isin(key_body_indices_tensor, lower_body_ids_tensor)

        # Upper body keypoint position error
        r_keybody_pos_upper = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_upper_key_body):
            upper_key_body_pos_diff = key_body_pos_diff[:, is_upper_key_body, :]
            if upper_key_body_pos_diff.numel() > 0:
                error_upper = (upper_key_body_pos_diff.abs()).mean(dim=-1).mean(dim=-1)
                sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                    "l1_rel_tracking_keybody_pos_upper", 1.0
                )
                r_keybody_pos_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body keypoint position error
        r_keybody_pos_lower = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_lower_key_body):
            lower_key_body_pos_diff = key_body_pos_diff[:, is_lower_key_body, :]
            if lower_key_body_pos_diff.numel() > 0:
                error_lower = (lower_key_body_pos_diff.abs()).mean(dim=-1).mean(dim=-1)
                sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                    "l1_rel_tracking_keybody_pos_lower", 1.0
                )
                r_keybody_pos_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l1_rel_tracking_keybody_pos_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l1_rel_tracking_keybody_pos_lower_weight", 0.5
        )

        return r_keybody_pos_upper * upper_weight + r_keybody_pos_lower * lower_weight

    @torch.compile
    def _reward_l2_rel_tracking_keybody_pos(self):
        # Ensure key_body_indices are tensor for advanced indexing
        key_body_indices_tensor = torch.tensor(
            self.key_body_indices, device=self.device, dtype=torch.long
        )

        # Filter dif_local_body_pos_t for key bodies
        key_body_pos_diff = self.dif_local_body_pos_t[
            :, key_body_indices_tensor, :
        ]  # [B, N_key, 3]

        upper_body_ids_tensor = torch.tensor(
            self.upper_body_id,
            device=self.device,
            dtype=torch.long,
        )
        lower_body_ids_tensor = torch.tensor(
            self.lower_body_id,
            device=self.device,
            dtype=torch.long,
        )

        is_upper_key_body = torch.isin(key_body_indices_tensor, upper_body_ids_tensor)
        is_lower_key_body = torch.isin(key_body_indices_tensor, lower_body_ids_tensor)

        # Upper body keypoint position error
        r_keybody_pos_upper = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_upper_key_body):
            upper_key_body_pos_diff = key_body_pos_diff[:, is_upper_key_body, :]
            if upper_key_body_pos_diff.numel() > 0:
                error_upper = (
                    (upper_key_body_pos_diff.square()).mean(dim=-1).mean(dim=-1)
                )
                sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                    "l2_rel_tracking_keybody_pos_upper", 1.0
                )
                r_keybody_pos_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body keypoint position error
        r_keybody_pos_lower = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_lower_key_body):
            lower_key_body_pos_diff = key_body_pos_diff[:, is_lower_key_body, :]
            if lower_key_body_pos_diff.numel() > 0:
                error_lower = (
                    (lower_key_body_pos_diff.square()).mean(dim=-1).mean(dim=-1)
                )
                sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                    "l2_rel_tracking_keybody_pos_lower", 1.0
                )
                r_keybody_pos_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l2_rel_tracking_keybody_pos_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l2_rel_tracking_keybody_pos_lower_weight", 0.5
        )

        return r_keybody_pos_upper * upper_weight + r_keybody_pos_lower * lower_weight

    @torch.compile
    def _reward_l2_root_rel_tracking_keybody_pos(self):
        # Ensure key_body_indices are tensor for advanced indexing
        key_body_indices_tensor = torch.tensor(
            self.key_body_indices, device=self.device, dtype=torch.long
        )

        # Filter dif_local_body_pos_t for key bodies
        key_body_pos_diff = self.dif_root_rel_body_pos_t[
            :, key_body_indices_tensor, :
        ]  # [B, N_key, 3]

        upper_body_ids_tensor = torch.tensor(
            self.upper_body_id,
            device=self.device,
            dtype=torch.long,
        )
        lower_body_ids_tensor = torch.tensor(
            self.lower_body_id,
            device=self.device,
            dtype=torch.long,
        )

        is_upper_key_body = torch.isin(key_body_indices_tensor, upper_body_ids_tensor)
        is_lower_key_body = torch.isin(key_body_indices_tensor, lower_body_ids_tensor)

        # Upper body keypoint position error
        r_keybody_pos_upper = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_upper_key_body):
            upper_key_body_pos_diff = key_body_pos_diff[:, is_upper_key_body, :]
            if upper_key_body_pos_diff.numel() > 0:
                error_upper = (
                    (upper_key_body_pos_diff.square()).mean(dim=-1).mean(dim=-1)
                )
                sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                    "l2_root_rel_tracking_keybody_pos_upper", 0.01
                )
                r_keybody_pos_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body keypoint position error
        r_keybody_pos_lower = torch.zeros(self.num_envs, device=self.device)
        if torch.any(is_lower_key_body):
            lower_key_body_pos_diff = key_body_pos_diff[:, is_lower_key_body, :]
            if lower_key_body_pos_diff.numel() > 0:
                error_lower = (
                    (lower_key_body_pos_diff.square()).mean(dim=-1).mean(dim=-1)
                )
                sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                    "l2_root_rel_tracking_keybody_pos_lower", 0.01
                )
                r_keybody_pos_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l2_root_rel_tracking_keybody_pos_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l2_root_rel_tracking_keybody_pos_lower_weight", 0.5
        )

        return r_keybody_pos_upper * upper_weight + r_keybody_pos_lower * lower_weight

    @torch.compile
    def _reward_l2_rel_tracking_headhand_pos(self):
        head_hand_pos_diff = self.dif_local_body_pos_t[
            :, self.head_hand_body_indices, :
        ]
        error = (head_hand_pos_diff.square()).mean(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_headhand_pos", 0.01
        )
        r_headhand_pos = torch.exp(-error / sigma)
        return r_headhand_pos

    @torch.compile
    def _reward_l2_rel_tracking_foothand_pos(self):
        foot_hand_pos_diff = self.dif_local_body_pos_t[
            :, self.foot_hand_body_indices, :
        ]
        error = (foot_hand_pos_diff.square()).mean(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_foothand_pos", 0.01
        )
        r_foothand_pos = torch.exp(-error / sigma)
        return r_foothand_pos

    @torch.compile
    def _reward_l1_rel_tracking_headhand_pos(self):
        head_hand_pos_diff = self.dif_local_body_pos_t[
            :, self.head_hand_body_indices, :
        ]
        error = (head_hand_pos_diff.abs()).mean(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_rel_tracking_headhand_pos", 0.1
        )
        r_headhand_pos = torch.exp(-error / sigma)
        return r_headhand_pos

    @torch.compile
    def _reward_l1_root_rel_tracking_headhand_pos(self):
        head_hand_pos_diff = self.dif_root_rel_body_pos_t[:, self.motion_tracking_id, :]
        error = (head_hand_pos_diff.abs()).mean(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_root_rel_tracking_headhand_pos", 0.1
        )
        r_headhand_pos = torch.exp(-error / sigma)
        return r_headhand_pos

    @torch.compile
    def _reward_l2_rel_tracking_wholebody_rot(self):
        upper_sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_wholebody_rot_upper", 0.1
        )
        lower_sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_wholebody_rot_lower", 0.1
        )
        upper_weight = self.config.rewards.get(
            "l2_rel_tracking_wholebody_rot_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l2_rel_tracking_wholebody_rot_lower_weight", 0.5
        )

        error = (self.dif_local_body_rot_tannorm.square()).mean(dim=-1)
        upper_error = error[:, self.upper_body_joint_ids].mean(dim=-1)
        lower_error = error[:, self.lower_body_joint_ids].mean(dim=-1)
        upper_r = torch.exp(-upper_error / upper_sigma)
        lower_r = torch.exp(-lower_error / lower_sigma)
        return upper_r * upper_weight + lower_r * lower_weight

    @torch.compile
    def _reward_l1_rel_tracking_root_lin_vel(self):
        error = (self.dif_local_root_lin_vel_t.abs()).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_rel_tracking_root_lin_vel", 0.25
        )
        r_root_lin_vel = torch.exp(-error / sigma)
        return r_root_lin_vel

    @torch.compile
    def _reward_l2_rel_tracking_root_lin_vel(self):
        error = (self.dif_local_root_lin_vel_t.square()).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_root_lin_vel", 0.25
        )
        r_root_lin_vel = torch.exp(-error / sigma)
        return r_root_lin_vel

    @torch.compile
    def _reward_l1_rel_tracking_root_ang_vel(self):
        error = (self.dif_local_root_ang_vel_t.abs()).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_rel_tracking_root_ang_vel", 0.25
        )
        r_root_ang_vel = torch.exp(-error / sigma)
        return r_root_ang_vel

    @torch.compile
    def _reward_l2_rel_tracking_root_ang_vel(self):
        error = (self.dif_local_root_ang_vel_t.square()).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_rel_tracking_root_ang_vel", 0.25
        )
        r_root_ang_vel = torch.exp(-error / sigma)
        return r_root_ang_vel

    @torch.compile
    def _reward_l1_tracking_joint_position(self):
        # Upper body joint position error
        r_joint_pos_upper = torch.zeros(self.num_envs, device=self.device)
        if len(self.upper_body_joint_ids) > 0:
            upper_joint_pos_diff = self.dif_joint_angles[:, self.upper_body_joint_ids]
            error_upper = (upper_joint_pos_diff.abs()).mean(dim=-1)
            sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                "l1_tracking_joint_pos_upper", 1.43
            )
            r_joint_pos_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body joint position error
        r_joint_pos_lower = torch.zeros(self.num_envs, device=self.device)
        if len(self.lower_body_joint_ids) > 0:
            lower_joint_pos_diff = self.dif_joint_angles[:, self.lower_body_joint_ids]
            error_lower = (lower_joint_pos_diff.abs()).mean(dim=-1)
            sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                "l1_tracking_joint_pos_lower", 1.43
            )
            r_joint_pos_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l1_tracking_joint_position_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l1_tracking_joint_position_lower_weight", 0.5
        )

        # Handle cases where one of the body parts might not have joints
        if len(self.upper_body_joint_ids) == 0:
            return r_joint_pos_lower  # Only lower body reward
        if len(self.lower_body_joint_ids) == 0:
            return r_joint_pos_upper  # Only upper body reward

        return r_joint_pos_upper * upper_weight + r_joint_pos_lower * lower_weight

    @torch.compile
    def _reward_l2_tracking_joint_position(self):
        # Upper body joint position error
        r_joint_pos_upper = torch.zeros(self.num_envs, device=self.device)
        if len(self.upper_body_joint_ids) > 0:
            upper_joint_pos_diff = self.dif_joint_angles[:, self.upper_body_joint_ids]
            error_upper = (upper_joint_pos_diff.square()).mean(dim=-1)
            sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_joint_pos_upper", 1.0
            )
            r_joint_pos_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body joint position error
        r_joint_pos_lower = torch.zeros(self.num_envs, device=self.device)
        if len(self.lower_body_joint_ids) > 0:
            lower_joint_pos_diff = self.dif_joint_angles[:, self.lower_body_joint_ids]
            error_lower = (lower_joint_pos_diff.square()).mean(dim=-1)
            sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_joint_pos_lower", 1.0
            )
            r_joint_pos_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l2_tracking_joint_position_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l2_tracking_joint_position_lower_weight", 0.5
        )

        # Handle cases where one of the body parts might not have joints
        if len(self.upper_body_joint_ids) == 0:
            return r_joint_pos_lower  # Only lower body reward
        if len(self.lower_body_joint_ids) == 0:
            return r_joint_pos_upper  # Only upper body reward

        return r_joint_pos_upper * upper_weight + r_joint_pos_lower * lower_weight

    @torch.compile
    def _reward_l1_tracking_joint_velocity(self):
        # Upper body joint velocity error
        r_joint_vel_upper = torch.zeros(self.num_envs, device=self.device)
        if len(self.upper_body_joint_ids) > 0:
            upper_joint_vel_diff = self.dif_joint_velocities[
                :, self.upper_body_joint_ids
            ]
            error_upper = (upper_joint_vel_diff.abs()).mean(dim=-1)
            sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                "l1_tracking_joint_vel_upper", 1.0
            )  # Default sigma, adjust as needed
            r_joint_vel_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body joint velocity error
        r_joint_vel_lower = torch.zeros(self.num_envs, device=self.device)
        if len(self.lower_body_joint_ids) > 0:
            lower_joint_vel_diff = self.dif_joint_velocities[
                :, self.lower_body_joint_ids
            ]
            error_lower = (lower_joint_vel_diff.abs()).mean(dim=-1)
            sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                "l1_tracking_joint_vel_lower", 1.0
            )
            r_joint_vel_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l1_tracking_joint_vel_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l1_tracking_joint_vel_lower_weight", 0.5
        )

        if len(self.upper_body_joint_ids) == 0:
            return r_joint_vel_lower
        if len(self.lower_body_joint_ids) == 0:
            return r_joint_vel_upper

        return r_joint_vel_upper * upper_weight + r_joint_vel_lower * lower_weight

    @torch.compile
    def _reward_l2_tracking_joint_velocity(self):
        # Upper body joint velocity error
        r_joint_vel_upper = torch.zeros(self.num_envs, device=self.device)
        if len(self.upper_body_joint_ids) > 0:
            upper_joint_vel_diff = self.dif_joint_velocities[
                :, self.upper_body_joint_ids
            ]
            error_upper = (upper_joint_vel_diff.square()).mean(dim=-1)
            sigma_upper = self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_joint_vel_upper", 1.0
            )  # Default sigma, adjust as needed
            r_joint_vel_upper = torch.exp(-error_upper / sigma_upper)

        # Lower body joint velocity error
        r_joint_vel_lower = torch.zeros(self.num_envs, device=self.device)
        if len(self.lower_body_joint_ids) > 0:
            lower_joint_vel_diff = self.dif_joint_velocities[
                :, self.lower_body_joint_ids
            ]
            error_lower = (lower_joint_vel_diff.square()).mean(dim=-1)
            sigma_lower = self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_joint_vel_lower", 1.0
            )
            r_joint_vel_lower = torch.exp(-error_lower / sigma_lower)

        upper_weight = self.config.rewards.get(
            "l2_tracking_joint_vel_upper_weight", 0.5
        )
        lower_weight = self.config.rewards.get(
            "l2_tracking_joint_vel_lower_weight", 0.5
        )

        if len(self.upper_body_joint_ids) == 0:
            return r_joint_vel_lower
        if len(self.lower_body_joint_ids) == 0:
            return r_joint_vel_upper

        return r_joint_vel_upper * upper_weight + r_joint_vel_lower * lower_weight

    @torch.compile
    def _reward_l1_tracking_root_rpy(self):
        # Stack the differences for roll, pitch, yaw
        rpy_diffs = torch.stack(
            [self.dif_base_roll_t, self.dif_base_pitch_t, self.dif_base_yaw_t],
            dim=-1,
        )  # Shape: [num_envs, 3]

        # Calculate L1 error (mean of absolute differences)
        error = (rpy_diffs.abs()).mean(dim=-1)  # Shape: [num_envs]

        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_tracking_root_rpy", 1.0
        )

        r_root_rpy = torch.exp(-error / sigma)
        return r_root_rpy

    @torch.compile
    def _reward_l2_tracking_root_rpy(self):
        # Stack the differences for roll, pitch, yaw
        rpy_diffs = torch.stack(
            [self.dif_base_roll_t, self.dif_base_pitch_t, self.dif_base_yaw_t],
            dim=-1,
        )  # Shape: [num_envs, 3]

        # Calculate L1 error (mean of absolute differences)
        error = (rpy_diffs.square()).mean(dim=-1)  # Shape: [num_envs]

        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_tracking_root_rpy", 1.0
        )

        r_root_rpy = torch.exp(-error / sigma)
        return r_root_rpy

    @torch.compile
    def _reward_l1_tracking_root_pitch(self):
        error = self.dif_base_pitch_t.abs()
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_tracking_root_pitch", 0.05
        )
        r_root_pitch = torch.exp(-error / sigma)
        return r_root_pitch

    @torch.compile
    def _reward_l1_tracking_waist_roll_pitch_dof(self):
        error = (
            self.dif_joint_angles[:, self.waist_roll_pitch_dof_indices]
            .abs()
            .mean(dim=-1)
        )
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l1_tracking_waist_roll_pitch_dof", 0.05
        )
        r_waist_roll_pitch = torch.exp(-error / sigma)
        return r_waist_roll_pitch

    @torch.compile
    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(
                    self.simulator.contact_forces[:, self.feet_indices, :],
                    dim=-1,
                )
                - self.config.rewards.feet_max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    @torch.compile
    def _reward_l2_feet_pos(self):
        rel_feet_pos_error = (
            self.dif_local_body_pos_t[:, self.feet_indices, :].square()
        ).mean(dim=(-1, -2))
        return torch.exp(
            -rel_feet_pos_error
            / self.config.rewards.reward_tracking_sigma.get("l2_feet_pos", 0.01)
        )

    @torch.compile
    def _reward_l2_feet_vel(self):
        rel_feet_vel_error = (
            self.dif_local_body_vel_t[:, self.feet_indices, :].square()
        ).mean(dim=(-1, -2))
        return torch.exp(
            -rel_feet_vel_error
            / self.config.rewards.reward_tracking_sigma.get("l2_feet_vel", 0.1)
        )

    @torch.compile
    def _reward_l2_hand_pos(self):
        rel_hand_pos_error = (
            self.dif_local_body_pos_t[:, self.hand_indices, :].square()
        ).mean(dim=(-1, -2))
        return torch.exp(
            -rel_hand_pos_error
            / self.config.rewards.reward_tracking_sigma.get("l2_hand_pos", 0.01)
        )

    @torch.compile
    def _reward_l2_root_rel_feet_pos(self):
        rel_feet_pos_error = (
            self.dif_root_rel_body_pos_t[:, self.feet_indices, :].square()
        ).mean(dim=(-1, -2))
        return torch.exp(
            -rel_feet_pos_error
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_feet_pos", 0.01
            )
        )

    @torch.compile
    def _reward_l2_root_rel_feet_vel(self):
        rel_feet_vel_error = (
            self.dif_root_rel_body_vel_t[:, self.feet_indices, :].square()
        ).mean(dim=(-1, -2))
        return torch.exp(
            -rel_feet_vel_error
            / self.config.rewards.reward_tracking_sigma.get("l2_root_rel_feet_vel", 0.1)
        )

    @torch.compile
    def _reward_penalty_l1_waist_roll_pitch_dof(self):
        error = (
            self.dif_joint_angles[:, self.waist_roll_pitch_dof_indices]
            .abs()
            .mean(dim=-1)
        )
        return error

    @torch.compile
    def _reward_l2_root_rel_tracking_endpoints(self):
        """
        Tracks the relative position of the end effectors (head, hands and feet) to the root.
        """

        motion_tracking_tensor = torch.tensor(
            self.motion_tracking_id,
            device=self.device,
            dtype=torch.long,
        )
        endpoint_indicies = torch.cat([motion_tracking_tensor, self.feet_indices])

        error = (self.dif_root_rel_body_pos_t[:, endpoint_indicies, :].square()).mean(
            dim=(-1, -2)
        )
        return torch.exp(
            -error
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_endpoints", 0.01
            )
        )

    @torch.compile
    def _reward_l2_root_rel_tracking_body_pos(self):
        diff_body_pos_dist = (self.dif_root_rel_body_pos_t**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(
            -diff_body_pos_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_body_pos", 0.01
            )
        )
        return r_body_pos

    @torch.compile
    def _reward_l2_root_rel_tracking_body_rot(self):
        diff_body_rot_dist = (
            (self.dif_root_rel_body_rot_tannorm**2).mean(dim=-1).mean(dim=-1)
        )
        r_body_rot = torch.exp(
            -diff_body_rot_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_body_rot", 0.02
            )
        )
        return r_body_rot

    @torch.compile
    def _reward_l2_root_rel_tracking_body_vel(self):
        diff_body_vel_dist = (self.dif_root_rel_body_vel_t**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(
            -diff_body_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_body_vel", 0.05
            )
        )
        return r_body_vel

    @torch.compile
    def _reward_l2_tracking_keybody_vel_v2(self):
        # calculates the global velocity difference between bodylinks and root
        # and then project to the root frame
        robot_keybody_vel_global_diff = (
            self._rigid_body_vel_extend[:, self.key_body_indices]
            - self.simulator.robot_root_states[:, 7:10][:, None, :]
        ).reshape(-1, 3)
        base_quat_body_flat = (
            self.base_quat[:, None, :]
            .repeat(1, len(self.key_body_indices), 1)
            .reshape(-1, 4)
        )
        robot_keybody_vel_root_rel = quat_rotate_inverse(
            base_quat_body_flat,
            robot_keybody_vel_global_diff,
            w_last=True,
        ).reshape(self.num_envs, -1, 3)

        # calculate the refrence global velocity difference between bodylinks
        # and root and then project to the reference root frame
        ref_keybody_vel_global_diff = (
            self.ref_body_vel_t[:, self.key_body_indices]
            - self.ref_root_global_pos_t[:, None, :]
        ).reshape(-1, 3)
        ref_quat_body_flat = (
            self.ref_root_global_rot_quat_t[:, None, :]
            .repeat(1, len(self.key_body_indices), 1)
            .reshape(-1, 4)
        )
        ref_keybody_vel_root_rel = quat_rotate_inverse(
            ref_quat_body_flat,
            ref_keybody_vel_global_diff,
            w_last=True,
        ).reshape(self.num_envs, -1, 3)

        error = robot_keybody_vel_root_rel - ref_keybody_vel_root_rel
        r_body_vel = torch.exp(
            -error.square().sum(-1).mean(-1)
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_keybody_vel_v2", 0.1
            )
        )
        return r_body_vel

    @torch.compile
    def _reward_l2_tracking_keybody_angvel_v2(self):
        # calculates the global velocity difference between bodylinks and root
        # and then project to the root frame
        robot_keybody_angvel_global_diff = (
            self._rigid_body_ang_vel_extend[:, self.key_body_indices]
            - self.simulator.robot_root_states[:, 10:13][:, None, :]
        ).reshape(-1, 3)
        base_quat_body_flat = (
            self.base_quat[:, None, :]
            .repeat(1, len(self.key_body_indices), 1)
            .reshape(-1, 4)
        )
        robot_keybody_angvel_root_rel = quat_rotate_inverse(
            base_quat_body_flat,
            robot_keybody_angvel_global_diff,
            w_last=True,
        ).reshape(self.num_envs, -1, 3)

        # calculate the refrence global velocity difference between bodylinks
        # and root and then project to the reference root frame
        ref_keybody_angvel_global_diff = (
            self.ref_body_ang_vel_t[:, self.key_body_indices]
            - self.ref_root_global_pos_t[:, None, :]
        ).reshape(-1, 3)
        ref_quat_body_flat = (
            self.ref_root_global_rot_quat_t[:, None, :]
            .repeat(1, len(self.key_body_indices), 1)
            .reshape(-1, 4)
        )
        ref_keybody_angvel_root_rel = quat_rotate_inverse(
            ref_quat_body_flat,
            ref_keybody_angvel_global_diff,
            w_last=True,
        ).reshape(self.num_envs, -1, 3)

        error = robot_keybody_angvel_root_rel - ref_keybody_angvel_root_rel
        r_body_angvel = torch.exp(
            -error.square().sum(-1).mean(-1)
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_keybody_angvel_v2", 1.0
            )
        )
        return r_body_angvel

    @torch.compile
    def _reward_l2_root_rel_tracking_body_ang_vel(self):
        diff_body_ang_vel_dist = (
            (self.dif_root_rel_body_ang_vel_t**2).mean(dim=-1).mean(dim=-1)
        )
        r_body_ang_vel = torch.exp(
            -diff_body_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_body_ang_vel", 1.0
            )
        )
        return r_body_ang_vel

    @torch.compile
    def _reward_l2_root_rel_tracking_keybody_vel(self):
        diff_body_vel_dist = (
            (self.dif_root_rel_body_vel_t[:, self.key_body_indices])
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
        )
        r_body_vel = torch.exp(
            -diff_body_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_keybody_vel", 0.05
            )
        )
        return r_body_vel

    @torch.compile
    def _reward_l2_root_rel_tracking_keybody_ang_vel(self):
        diff_body_ang_vel_dist = (
            (self.dif_root_rel_body_ang_vel_t[:, self.key_body_indices])
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
        )
        r_body_ang_vel = torch.exp(
            -diff_body_ang_vel_dist
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_root_rel_tracking_keybody_ang_vel", 1.0
            )
        )
        return r_body_ang_vel

    @torch.compile
    def _reward_l2_tracking_torso_rot(self):
        error = (
            self.dif_root_rel_body_rot_tannorm[:, self.torso_index]
            .square()
            .mean(dim=-1)
        )
        return torch.exp(
            -error
            / self.config.rewards.reward_tracking_sigma.get(
                "l2_tracking_torso_rot", 0.1
            )
        )

    @torch.compile
    def _reward_l1_root_rel_tracking_torso_rot(self):
        error = (
            self.dif_root_rel_body_rot_tannorm[:, self.torso_index].abs().mean(dim=-1)
        )
        return torch.exp(
            -error
            / self.config.rewards.reward_tracking_sigma.get(
                "l1_root_rel_tracking_torso_rot", 0.2
            )
        )

    @torch.compile
    def _reward_l2_root_global_pos(self):
        error = torch.sum(torch.square(self.dif_global_root_pos_t), dim=(-1))
        return torch.exp(
            -error
            / self.config.rewards.reward_tracking_sigma.get("l2_root_global_pos", 0.01)
        )

    @torch.compile
    def _reward_l2_root_global_rot(self):
        error = quat_error_magnitude(
            self.base_quat,
            self.ref_root_global_rot_quat_t,
            w_last=True,
        )
        return torch.exp(
            -error
            / self.config.rewards.reward_tracking_sigma.get("l2_root_global_rot", 0.16)
        )

    @torch.compile
    def _reward_l2_root_rel_tracking_keybody_pos_v2(self):
        key_body_pos_diff = self.dif_root_rel_body_pos_t[
            :, self.key_body_indices, :
        ]  # [B, N_key, 3]
        error = (key_body_pos_diff.square()).sum(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_root_rel_tracking_keybody_pos_v2", 0.09
        )
        r_keybody_pos = torch.exp(-error / sigma)
        return r_keybody_pos

    @torch.compile
    def _reward_l2_root_rel_tracking_keybody_rot_v2(self):
        keybody_quat_diff = quat_error_magnitude(
            self._robot_root_rel_body_rot_quat_t[:, self.key_body_indices].reshape(
                -1, 4
            ),
            self.ref_root_rel_body_rot_quat_t[:, self.key_body_indices].reshape(-1, 4),
            w_last=True,
        ).reshape(self.num_envs, -1)  # [B, N_kb]
        error = (keybody_quat_diff.square()).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_root_rel_tracking_keybody_rot_v2", 0.16
        )
        r_keybody_pos = torch.exp(-error / sigma)
        return r_keybody_pos

    @torch.compile
    def _reward_l2_global_tracking_keybody_linvel(self):
        keybody_global_linvel_diff = self.dif_global_body_vel[:, self.key_body_indices]
        error = (keybody_global_linvel_diff.square()).sum(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_global_tracking_keybody_linvel", 1.0
        )
        r_keybody_pos = torch.exp(-error / sigma)
        return r_keybody_pos

    @torch.compile
    def _reward_l2_global_tracking_keybody_angvel(self):
        keybody_global_angvel_diff = self.dif_global_body_ang_vel[
            :, self.key_body_indices
        ]
        error = (keybody_global_angvel_diff.square()).sum(dim=-1).mean(dim=-1)
        sigma = self.config.rewards.reward_tracking_sigma.get(
            "l2_global_tracking_keybody_angvel", 9.8596
        )
        r_keybody_pos = torch.exp(-error / sigma)
        return r_keybody_pos
    
    @torch.compile
    def _reward_penalty_undesired_contacts(self, threshold=1.0):
        net_contact_forces = self.simulator.contact_forces
        is_contact =  torch.norm(net_contact_forces[:, self.penalised_contact_indices, :3], dim=-1) > threshold
        return torch.sum(is_contact, dim=1)


@torch.compile
def quat_to_tan_norm(q: torch.Tensor, w_last: bool) -> torch.Tensor:
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan, w_last)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm, w_last)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Procrustes-aligned MPJPE between predicted and ground truth.

    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Corrected rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    predicted_aligned = a * np.matmul(predicted, r) + t

    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)


def compute_error_vel(
    joints_pred: np.ndarray,
    joints_gt: np.ndarray,
    vis: Optional[np.ndarray] = None,
    pred_vel: Optional[np.ndarray] = None,
    gt_vel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute velocity error between predicted and ground truth poses.

    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    if pred_vel is not None and gt_vel is not None:
        vel_pred = pred_vel
        vel_gt = gt_vel
    else:
        # Compute velocities from positions
        vel_gt = joints_gt[1:] - joints_gt[:-1]
        vel_pred = joints_pred[1:] - joints_pred[:-1]

    # Compute L2 norm of velocity differences
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    # Apply visibility mask if provided
    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)


def compute_error_accel(
    joints_pred: np.ndarray,
    joints_gt: np.ndarray,
    vis: Optional[np.ndarray] = None,
    pred_accel: Optional[np.ndarray] = None,
    gt_accel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute acceleration error between predicted and ground truth poses.

    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    if pred_accel is not None and gt_accel is not None:
        accel_pred = pred_accel
        accel_gt = gt_accel
    else:
        # Compute accelerations from positions using central difference
        # (N-2)xJx3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    # Compute L2 norm of acceleration differences
    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_metrics_lite(
    pred_pos_all: List[np.ndarray],
    gt_pos_all: List[np.ndarray],
    pred_rot_all: Optional[List[np.ndarray]] = None,
    gt_rot_all: Optional[List[np.ndarray]] = None,
    root_idx: int = 0,
    use_tqdm: bool = True,
    concatenate: bool = True,
    pred_vel: Optional[List[np.ndarray]] = None,
    gt_vel: Optional[List[np.ndarray]] = None,
    pred_accel: Optional[List[np.ndarray]] = None,
    gt_accel: Optional[List[np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Compute tracking metrics between predicted and ground truth poses.

    Args:
        pred_pos_all: List of predicted positions, each with shape (T, J, 3)
        gt_pos_all: List of ground truth positions, each with shape (T, J, 3)
        pred_rot_all: Optional list of predicted rotations, shape (T, J, 4)
        gt_rot_all: Optional list of ground truth rotations, shape (T, J, 4)
        root_idx: Index of the root joint
        use_tqdm: Whether to use tqdm for progress bar
        concatenate: Whether to concatenate metrics across sequences
    Returns:
        Dictionary of metrics
    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    from tqdm import tqdm

    # Initialize metrics dictionary
    metrics = defaultdict(list)
    if use_tqdm:
        pbar = tqdm(range(len(pred_pos_all)))
    else:
        pbar = range(len(pred_pos_all))

    for idx in pbar:
        jpos_pred = pred_pos_all[idx].copy()
        jpos_gt = gt_pos_all[idx].copy()
        rot_pred = pred_rot_all[idx].copy() if pred_rot_all is not None else None
        rot_gt = gt_rot_all[idx].copy() if gt_rot_all is not None else None

        # Global joint position error
        mpjpe_g = np.linalg.norm(jpos_gt - jpos_pred, axis=2) * 1000

        # Velocity and acceleration errors
        vel_dist = (
            compute_error_vel(
                jpos_pred,
                jpos_gt,
                pred_vel=None if pred_vel is None else pred_vel[idx],
                gt_vel=None if gt_vel is None else gt_vel[idx],
            )
            * 1000
        )

        accel_dist = (
            compute_error_accel(
                jpos_pred,
                jpos_gt,
                pred_accel=None if pred_accel is None else pred_accel[idx],
                gt_accel=None if gt_accel is None else gt_accel[idx],
            )
            * 1000
        )

        # Local joint position error (zero out root)
        jpos_pred = jpos_pred - jpos_pred[:, [root_idx]]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[:, [root_idx]]

        # Procrustes-aligned MPJPE
        pa_mpjpe = p_mpjpe(jpos_pred, jpos_gt) * 1000
        mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2) * 1000

        # Rotation error if rotations are provided
        if rot_pred is not None and rot_gt is not None:
            rot_error = np.linalg.norm(
                (
                    sRot.from_quat(rot_gt.reshape(-1, 4))
                    * sRot.from_quat(rot_pred.reshape(-1, 4)).inv()
                ).as_rotvec(),
                axis=-1,
            )

        # Store metrics
        metrics["mpjpe_g"].append(mpjpe_g)
        if rot_pred is not None and rot_gt is not None:
            metrics["rot_error"].append(rot_error)
        metrics["mpjpe_l"].append(mpjpe)
        metrics["mpjpe_pa"].append(pa_mpjpe)
        metrics["accel_dist"].append(accel_dist)
        metrics["vel_dist"].append(vel_dist)

    # Concatenate if requested
    if concatenate:
        metrics = {k: np.concatenate(v) for k, v in metrics.items()}
    return metrics
