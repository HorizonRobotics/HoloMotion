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


import json
import os
import pickle
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from holomotion.src.modules.agent_modules import PPOActor, PPOCritic
from holomotion.src.modules.network_modules import RunningMeanStdNormalizer


class PPO:
    def __init__(
        self,
        env,
        config,
        log_dir=None,
        device="cpu",
    ):
        self.config = config
        self.use_accelerate = config.use_accelerate
        if self.use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                self.is_main_process = torch.distributed.get_rank() == 0
                self.process_rank = torch.distributed.get_rank()
            else:
                self.is_main_process = self.accelerator.is_main_process
                self.process_rank = self.accelerator.process_index
        else:
            self.device = device
            self.is_main_process = True
            self.process_rank = 0

        self.env = env
        self.log_dir = log_dir

        # Only initialize TensorBoard on the main process
        if self.is_main_process:
            # Initialize TensorBoard SummaryWriter
            if self.log_dir:
                self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard logging enabled at: {self.log_dir}")
            else:
                self.tensorboard_writer = None
                logger.warning(
                    "No log directory provided, TensorBoard logging disabled"
                )
        else:
            self.tensorboard_writer = None

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        self.normalize_rewards = self.config.get("normalize_rewards", False)
        self.reward_norm_epsilon = self.config.get("reward_norm_epsilon", 1e-8)
        if self.normalize_rewards:
            self.reward_normalizer = RunningMeanStdNormalizer(
                feature_dim=1, epsilon=self.reward_norm_epsilon
            ).to(self.device)
            logger.info(
                f"Reward normalization enabled with "
                f"epsilon: {self.reward_norm_epsilon}"
            )
            self.clip_normalized_rewards = self.config.get(
                "clip_normalized_rewards", False
            )
            if self.clip_normalized_rewards:
                self.normalized_reward_clip_value = self.config.get(
                    "normalized_reward_clip_value", 10.0
                )
                logger.info(
                    f"Normalized reward clipping enabled with value: "
                    f"{self.normalized_reward_clip_value}"
                )

        if getattr(self.env, "obs_serializer", None) is not None:
            self.obs_serializer = self.env.obs_serializer
        else:
            self.obs_serializer = None

        self.dagger_only = self.config.get("dagger_only", False)

        self.actor_type = self.config.module_dict.get("actor", {}).get(
            "type", "MLP"
        )
        if not self.dagger_only:
            self.critic_type = self.config.module_dict.get("critic", {}).get(
                "type", "MLP"
            )
            self.disc_type = self.config.module_dict.get("disc", {}).get(
                "type", "MLP"
            )
        else:
            self.critic_type = None
            self.disc_type = None

        logger.info(f"Actor type: {self.actor_type}")
        logger.info(f"Critic type: {self.critic_type}")
        logger.info(f"Disc type: {self.disc_type}")

        self.save_interval = self.config.save_interval
        self.eval_interval = self.config.get("eval_interval", None)
        self.log_interval = self.config.log_interval
        # Training related Config
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.load_critic_when_dagger = self.config.load_critic_when_dagger
        self.num_learning_iterations = self.config.num_learning_iterations

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.use_smooth_grad_penalty = self.config.get(
            "use_smooth_penalty", False
        )

        self.entropy_curriculum = self.env.config.get(
            "entropy_curriculum", {}
        ).get("enable_entropy_curriculum", False)

        self.use_amp = self.env.config.get("amp", {}).get("enabled", False)
        if self.use_amp:
            self.task_rew_coef = self.config.get("task_rew_coef", 1.0)
            self.amp_rew_coef = self.config.get("amp_rew_coef", 1.0)
            self.disc_loss_coef = self.config.get("disc_loss_coef", 1.0)
            self.disc_grad_penalty_coef = self.config.get(
                "disc_grad_penalty_coef", 0.1
            )
            self.disc_loss_type = self.config.get("disc_loss_type", "lsgan")
            self.amp_rew_scale = self.config.get("amp_rew_scale", 1.0)
            self.adaptive_disc_rew = self.config.get(
                "adaptive_disc_rew", False
            )
            self.smoothed_task_rew_scale = 0.1
            self.smoothed_disc_rew_scale = 0.1
            self.smooth_gamma = 1.0 - 1e-2

        self.use_dagger = (
            True
            if self.config.get("teacher_actor_ckpt_path", None) is not None
            else False
        )
        if self.use_dagger:
            self.teacher_actor_ckpt_path = self.config.teacher_actor_ckpt_path
            logger.info(
                f"Using Dagger with teacher actor checkpoint from "
                f"{self.config.teacher_actor_ckpt_path}"
            )

            self.dagger_anneal = self.config.get("dagger_anneal", True)
            self.dagger_anneal_degree = self.config.get(
                "dagger_anneal_degree", 1.0e-5
            )
            self.dagger_coef = self.config.get("dagger_init_coef", 1.0)

            self.rl_anneal = self.config.get("rl_anneal", False)
            self.rl_anneal_degree = self.config.get("rl_anneal_degree", 1.0e-5)
            self.rl_coef = self.config.get("rl_init_coef", 1.0)

        self.predict_local_body_pos = self.config.get(
            "predict_local_body_pos",
            False,
        )
        self.predict_local_body_vel = self.config.get(
            "predict_local_body_vel",
            False,
        )
        self.predict_root_lin_vel = self.config.get(
            "predict_root_lin_vel",
            False,
        )

        self.pred_local_body_pos_alpha = self.config.get(
            "pred_local_body_pos_alpha",
            1.0,
        )
        self.pred_local_body_vel_alpha = self.config.get(
            "pred_local_body_vel_alpha",
            0.1,
        )
        self.pred_root_lin_vel_alpha = self.config.get(
            "pred_root_lin_vel_alpha",
            0.1,
        )

    def setup(self):
        self._setup_models_and_optimizer()
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        if self.actor_type == "MLP":
            self.actor = PPOActor(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config_dict=self.config.module_dict.actor,
                num_actions=self.num_act,
                init_noise_std=self.config.init_noise_std,
            ).to(self.device)
        elif self.actor_type == "MoEMLP":
            self.actor = PPOActor(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config_dict=self.config.module_dict.actor,
                num_actions=self.num_act,
                init_noise_std=self.config.init_noise_std,
            ).to(self.device)
        else:
            raise NotImplementedError

        if not self.dagger_only:
            if self.critic_type == "MLP":
                self.critic = PPOCritic(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config_dict=self.config.module_dict.critic,
                ).to(self.device)
            elif self.critic_type == "MoEMLP":
                self.critic = PPOCritic(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config_dict=self.config.module_dict.critic,
                ).to(self.device)
            else:
                raise NotImplementedError

            if self.use_amp:
                if self.disc_type == "MLP":
                    self.disc = PPOCritic(
                        obs_dim_dict=self.algo_obs_dim_dict,
                        module_config_dict=self.config.module_dict.disc,
                    ).to(self.device)
                else:
                    raise NotImplementedError

        logger.info("Actor:\n" + str(self.actor))
        if not self.dagger_only:
            logger.info("Critic:\n" + str(self.critic))
            if self.use_amp:
                logger.info("Disc:\n" + str(self.disc))

        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=self.actor_learning_rate
        )

        if not self.dagger_only:
            self.critic_optimizer = optim.AdamW(
                self.critic.parameters(), lr=self.critic_learning_rate
            )
            if self.use_amp:
                self.disc_optimizer = optim.AdamW(
                    self.disc.parameters(),
                    lr=self.critic_learning_rate,
                    betas=(0.0, 0.99),
                )

        if self.use_accelerate and hasattr(self, "accelerator"):
            # accelerator.prepare should handle compiled models correctly
            self.actor, self.actor_optimizer = self.accelerator.prepare(
                self.actor, self.actor_optimizer
            )
            if not self.dagger_only:
                self.critic, self.critic_optimizer = self.accelerator.prepare(
                    self.critic, self.critic_optimizer
                )
                if self.use_amp:
                    self.disc, self.disc_optimizer = self.accelerator.prepare(
                        self.disc, self.disc_optimizer
                    )

        if self.use_dagger:
            # Setup teacher actor with hydra config
            teacher_actor_type = self.config.module_dict.teacher_actor.get(
                "type", "MLP"
            )
            if teacher_actor_type == "MLP":
                self.teacher_actor = PPOActor(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config_dict=self.config.module_dict.teacher_actor,
                    num_actions=self.num_act,
                    init_noise_std=self.config.init_noise_std,
                ).to(self.device)
            elif teacher_actor_type == "MoEMLP":
                self.teacher_actor = PPOActor(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config_dict=self.config.module_dict.teacher_actor,
                    num_actions=self.num_act,
                    init_noise_std=self.config.init_noise_std,
                ).to(self.device)
            else:
                raise NotImplementedError

            logger.info("Teacher actor:\n" + str(self.teacher_actor))

            # Load teacher actor checkpoint
            teacher_ckpt = torch.load(
                self.teacher_actor_ckpt_path, map_location=self.device
            )
            self.teacher_actor.load_state_dict(
                teacher_ckpt["actor_model_state_dict"], strict=True
            )
            # Freeze teacher actor
            self.teacher_actor.to(self.device)
            self.teacher_actor.eval()
            for param in self.teacher_actor.parameters():
                param.requires_grad = False
            if self.use_accelerate and hasattr(self, "accelerator"):
                self.teacher_actor = self.accelerator.prepare(
                    self.teacher_actor
                )
            logger.info("Teacher actor loaded from checkpoint successfully !")

            if self.config.get("load_critic_when_dagger", False):
                cleaned_critic_state_dict = self._clean_state_dict(
                    teacher_ckpt["critic_model_state_dict"]
                )
                if self.use_accelerate and hasattr(self, "accelerator"):
                    self.accelerator.unwrap_model(self.critic).load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
                else:
                    self.critic.load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
                logger.info(
                    "Strict loading of actor and critic states successful."
                )

                if self.use_amp:
                    if "disc_model_state_dict" in teacher_ckpt:
                        cleaned_disc_state_dict = self._clean_state_dict(
                            teacher_ckpt["disc_model_state_dict"]
                        )
                        if self.use_accelerate and hasattr(
                            self, "accelerator"
                        ):
                            self.accelerator.unwrap_model(
                                self.disc
                            ).load_state_dict(
                                cleaned_disc_state_dict, strict=True
                            )
                        else:
                            self.disc.load_state_dict(
                                cleaned_disc_state_dict, strict=True
                            )
                        logger.info(
                            "Strict loading of discriminator state successful."
                        )
                    else:
                        logger.warning(
                            "use_amp is True, but 'disc_model_state_dict' "
                            "not found in checkpoint. Skipping discriminator "
                            "model loading."
                        )

        # Log brief summaries of the models
        self._log_model_summary(self.actor, "Actor")
        if not self.dagger_only:
            self._log_model_summary(self.critic, "Critic")
            if self.use_amp:
                self._log_model_summary(self.disc, "Discriminator")
        if self.use_dagger and hasattr(self, "teacher_actor"):
            self._log_model_summary(self.teacher_actor, "Teacher Actor")

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.env.num_envs, self.num_steps_per_env, device=self.device
        )
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(
                obs_key, shape=(obs_dim,), dtype=torch.float
            )

        ## Register others
        self.storage.register_key(
            "actions", shape=(self.num_act,), dtype=torch.float
        )
        self.storage.register_key("rewards", shape=(1,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)
        self.storage.register_key("values", shape=(1,), dtype=torch.float)
        self.storage.register_key("returns", shape=(1,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(1,), dtype=torch.float)
        self.storage.register_key(
            "actions_log_prob", shape=(1,), dtype=torch.float
        )
        self.storage.register_key(
            "action_mean", shape=(self.num_act,), dtype=torch.float
        )
        self.storage.register_key(
            "action_sigma", shape=(self.num_act,), dtype=torch.float
        )

        if self.use_amp:
            self.storage.register_key(
                "disc_demo_obs",
                shape=(self.env.config.amp.amp_obs_size,),
                dtype=torch.float,
            )
            self.storage.register_key(
                "amp_valid_sample_mask",
                shape=(1,),
                dtype=torch.bool,
            )
            # Add storage for task and discriminator rewards
            self.storage.register_key(
                "task_rewards", shape=(1,), dtype=torch.float
            )
            self.storage.register_key(
                "disc_rewards", shape=(1,), dtype=torch.float
            )

        if self.use_dagger:
            self.storage.register_key(
                "teacher_actions", shape=(self.num_act,), dtype=torch.float
            )

        self.num_all_bodies = self.config.get(
            "num_rigid_bodies", 0
        ) + self.config.get("num_extended_bodies", 0)

        if self.predict_local_body_pos:
            self.storage.register_key(
                "local_body_pos_extend_flat",
                shape=(self.num_all_bodies * 3,),
                dtype=torch.float,
            )
        if self.predict_local_body_vel:
            self.storage.register_key(
                "local_body_vel_extend_flat",
                shape=(self.num_all_bodies * 3,),
                dtype=torch.float,
            )
        if self.predict_root_lin_vel:
            self.storage.register_key(
                "root_lin_vel",
                shape=(3,),
                dtype=torch.float,
            )

    def _eval_mode(self):
        # Handle both DDP-wrapped and normal models
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.eval()
        if not self.dagger_only:
            critic = (
                self.critic.module
                if hasattr(self.critic, "module")
                else self.critic
            )
            critic.eval()

    def _train_mode(self):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.train()
        if not self.dagger_only:
            critic = (
                self.critic.module
                if hasattr(self.critic, "module")
                else self.critic
            )
            critic.train()

    @staticmethod
    def _clean_state_dict(state_dict):
        """Remove the '_orig_mod.' prefix from keys if it exists."""
        cleaned_dict = {}
        prefix = "_orig_mod."
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                cleaned_dict[k[prefix_len:]] = v
            else:
                cleaned_dict[k] = v
        return cleaned_dict

    def load(self, ckpt_path):
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            # Clean the state dicts to remove potential compilation prefixes
            cleaned_actor_state_dict = self._clean_state_dict(
                loaded_dict["actor_model_state_dict"]
            )
            if not self.dagger_only:
                cleaned_critic_state_dict = self._clean_state_dict(
                    loaded_dict["critic_model_state_dict"]
                )

            # Load with strict=True using the cleaned state dicts
            if self.use_accelerate and hasattr(self, "accelerator"):
                self.accelerator.unwrap_model(self.actor).load_state_dict(
                    cleaned_actor_state_dict, strict=True
                )
                if not self.dagger_only:
                    self.accelerator.unwrap_model(self.critic).load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
            else:
                self.actor.load_state_dict(
                    cleaned_actor_state_dict, strict=True
                )
                if not self.dagger_only and self.load_critic_when_dagger:
                    self.critic.load_state_dict(
                        cleaned_critic_state_dict, strict=True
                    )
                    logger.info(
                        "Strict loading of actor and critic states successful."
                    )

            if self.use_amp and not self.dagger_only:
                if self.config.get("load_disc", False):
                    if "disc_model_state_dict" in loaded_dict:
                        cleaned_disc_state_dict = self._clean_state_dict(
                            loaded_dict["disc_model_state_dict"]
                        )
                        if self.use_accelerate and hasattr(
                            self, "accelerator"
                        ):
                            self.accelerator.unwrap_model(
                                self.disc
                            ).load_state_dict(
                                cleaned_disc_state_dict, strict=True
                            )
                        else:
                            self.disc.load_state_dict(
                                cleaned_disc_state_dict, strict=True
                            )
                        logger.info(
                            "Strict loading of discriminator state successful."
                        )
                    else:
                        logger.warning(
                            "use_amp is True, but 'disc_model_state_dict' not "
                            "found in checkpoint. Skipping disc model loading."
                        )

            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(
                    loaded_dict["actor_optimizer_state_dict"]
                )
                if not self.dagger_only:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )
                if self.use_amp and not self.dagger_only:
                    if self.config.get("load_disc", False):
                        if "disc_optimizer_state_dict" in loaded_dict:
                            self.disc_optimizer.load_state_dict(
                                loaded_dict["disc_optimizer_state_dict"]
                            )
                            logger.info(
                                "Disc optimizer loaded from checkpoint"
                            )

                self.actor_learning_rate = loaded_dict[
                    "actor_optimizer_state_dict"
                ]["param_groups"][0]["lr"]

                if not self.dagger_only:
                    self.critic_learning_rate = loaded_dict[
                        "critic_optimizer_state_dict"
                    ]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                if not self.dagger_only:
                    logger.info(
                        f"Critic Learning rate: {self.critic_learning_rate}"
                    )
            self.current_learning_iteration = loaded_dict["iter"]
            if "env_curriculum" in loaded_dict:
                curriculum_dict = loaded_dict["env_curriculum"]
                if (
                    self.env.use_reward_penalty_curriculum
                    and "reward_penalty_scale" in curriculum_dict
                ):
                    self.env.reward_penalty_scale = curriculum_dict[
                        "reward_penalty_scale"
                    ]
                if (
                    self.env.use_reward_limits_dof_pos_curriculum
                    and "soft_dof_pos_curriculum_value" in curriculum_dict
                ):
                    self.env.soft_dof_pos_curriculum_value = curriculum_dict[
                        "soft_dof_pos_curriculum_value"
                    ]
                if (
                    self.env.use_reward_limits_dof_vel_curriculum
                    and "soft_dof_vel_curriculum_value" in curriculum_dict
                ):
                    self.env.soft_dof_vel_curriculum_value = curriculum_dict[
                        "soft_dof_vel_curriculum_value"
                    ]
                if (
                    self.env.use_reward_limits_torque_curriculum
                    and "soft_torque_curriculum_value" in curriculum_dict
                ):
                    self.env.soft_torque_curriculum_value = curriculum_dict[
                        "soft_torque_curriculum_value"
                    ]
                if (
                    self.env.add_noise_currculum
                    and "current_noise_curriculum_value" in curriculum_dict
                ):
                    self.env.current_noise_curriculum_value = curriculum_dict[
                        "current_noise_curriculum_value"
                    ]
                if (
                    self.env.config.termination.terminate_when_motion_far
                    and self.env.config.termination_curriculum.terminate_when_motion_far_curriculum  # noqa: E501
                    and "average_episode_length" in curriculum_dict
                    and "terminate_when_motion_far_threshold"
                    in curriculum_dict
                ):
                    self.env.average_episode_length = curriculum_dict[
                        "average_episode_length"
                    ]
                    self.env.terminate_when_motion_far_threshold = (
                        curriculum_dict["terminate_when_motion_far_threshold"]
                    )
            return loaded_dict["infos"]

        if self.normalize_rewards and hasattr(self, "reward_normalizer"):
            if "reward_normalizer_state" in loaded_dict:
                self.reward_normalizer.set_state(
                    loaded_dict["reward_normalizer_state"],
                    new_buffer_device=self.device,
                )
                logger.info("Reward normalizer state loaded from checkpoint.")
            else:
                logger.warning(
                    "normalize_rewards is True, but 'reward_normalizer_state' "
                    "not found in checkpoint. Initializing new reward"
                    "normalizer."
                )

    def save(self, path, infos=None):
        if not self.is_main_process:
            return

        logger.info(f"Saving checkpoint to {path}")

        env_curriculum = {}
        if self.env.use_reward_penalty_curriculum:
            env_curriculum["reward_penalty_scale"] = (
                self.env.reward_penalty_scale
            )
        if self.env.use_reward_limits_dof_pos_curriculum:
            env_curriculum["soft_dof_pos_curriculum_value"] = (
                self.env.soft_dof_pos_curriculum_value
            )
        if self.env.use_reward_limits_dof_vel_curriculum:
            env_curriculum["soft_dof_vel_curriculum_value"] = (
                self.env.soft_dof_vel_curriculum_value
            )
        if self.env.use_reward_limits_torque_curriculum:
            env_curriculum["soft_torque_curriculum_value"] = (
                self.env.soft_torque_curriculum_value
            )
        if self.env.add_noise_currculum:
            env_curriculum["current_noise_curriculum_value"] = (
                self.env.current_noise_curriculum_value
            )
        if (
            self.env.config.termination.terminate_when_motion_far
            and self.env.config.termination_curriculum.terminate_when_motion_far_curriculum  # noqa: E501
        ):
            env_curriculum["average_episode_length"] = (
                self.env.average_episode_length
            )
            env_curriculum["terminate_when_motion_far_threshold"] = (
                self.env.terminate_when_motion_far_threshold
            )

        # Get unwrapped model state dict if using accelerate
        actor_state = (
            self.accelerator.unwrap_model(self.actor).state_dict()
            if (self.use_accelerate and hasattr(self, "accelerator"))
            else self.actor.state_dict()
        )
        if not self.dagger_only:
            critic_state = (
                self.accelerator.unwrap_model(self.critic).state_dict()
                if (self.use_accelerate and hasattr(self, "accelerator"))
                else self.critic.state_dict()
            )

        save_dict = {
            "actor_model_state_dict": actor_state,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "env_curriculum": env_curriculum,
        }

        if self.normalize_rewards and hasattr(self, "reward_normalizer"):
            save_dict["reward_normalizer_state"] = (
                self.reward_normalizer.get_state()
            )

        if not self.dagger_only:
            save_dict.update(
                {
                    "critic_model_state_dict": critic_state,
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),  # noqa: E501
                }
            )

        if self.use_amp and not self.dagger_only:
            disc_state = (
                self.accelerator.unwrap_model(self.disc).state_dict()
                if (self.use_accelerate and hasattr(self, "accelerator"))
                else self.disc.state_dict()
            )
            save_dict["disc_model_state_dict"] = disc_state
            save_dict["disc_optimizer_state_dict"] = (
                self.disc_optimizer.state_dict()
            )

        torch.save(save_dict, path)

    def learn(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Initialize distributed barrier for synchronized training
        if self.use_accelerate and hasattr(self, "accelerator"):
            self.accelerator.wait_for_everyone()

        if self.is_main_process:
            logger.info(
                f"Starting training for {num_learning_iterations} iterations "
                f"from iteration {self.current_learning_iteration}"
            )

        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)

            end_of_rollout_time = time.time()
            self.collection_time = end_of_rollout_time - self.start_time

            loss_dict = self._training_step()

            end_of_training_time = time.time()
            self.learn_time = end_of_training_time - end_of_rollout_time

            self.current_learning_iteration = it

            # Logging
            log_dict = {
                "it": it,
                "loss_dict": loss_dict,
                "collection_time": self.collection_time,
                "learn_time": self.learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "entropy_coef": self.entropy_coef,
                "num_learning_iterations": num_learning_iterations,
                "total_learning_iterations": tot_iter,
            }

            # Only log on main process when using distributed training
            if self.is_main_process:
                if it % self.log_interval == 0:
                    self._post_epoch_logging(log_dict)
                    self.env._log_motion_tracking_info()
                if it % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )
            self.ep_infos.clear()

            if self.eval_interval is not None:
                if it > 0 and it % self.eval_interval == 0:
                    if self.use_accelerate:
                        self.accelerator.wait_for_everyone()
                    self.evaluate_policy()

            # Synchronize processes after each iteration
            if self.use_accelerate and hasattr(self, "accelerator"):
                self.accelerator.wait_for_everyone()

        # Only save on main process when using distributed training
        if self.is_main_process:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            logger.info(f"Training completed. Model saved to {self.log_dir}")
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        # with torch.no_grad():
        with torch.inference_mode():
            actions = self._actor_act_step(obs_dict)
        policy_state_dict["actions"] = actions

        if hasattr(self.actor, "module"):
            action_mean = self.actor.module.action_mean.detach()
            action_sigma = self.actor.module.action_std.detach()
            actions_log_prob = (
                self.actor.module.get_actions_log_prob(actions)
                .detach()
                .unsqueeze(1)
            )
        else:
            action_mean = self.actor.action_mean.detach()
            action_sigma = self.actor.action_std.detach()
            actions_log_prob = (
                self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
            )

        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(
                    obs_dict, policy_state_dict
                )
                if not self.dagger_only:
                    values = self._critic_eval_step(obs_dict).detach()
                    policy_state_dict["values"] = values
                for obs_key in obs_dict.keys():
                    if obs_key != "disc_obs":
                        self.storage.update_key(obs_key, obs_dict[obs_key])
                if self.use_amp and not self.dagger_only:
                    valid_env_ids = (
                        self.env.episode_length_buf
                        > self.env.config.amp_context_length
                    )
                    self.storage.update_key(
                        "disc_obs",
                        obs_dict["disc_obs"],
                    )
                    self.storage.update_key(
                        "amp_valid_sample_mask",
                        valid_env_ids[:, None],
                    )
                    disc_demo_obs = self.env._get_obs_amp_demo_seq_v2().to(
                        self.device
                    )
                    self.storage.update_key("disc_demo_obs", disc_demo_obs)

                    disc_r = torch.zeros(self.env.num_envs, device=self.device)
                    # calculate disc reward for amp
                    if valid_env_ids.any():
                        with torch.inference_mode():
                            if self.use_accelerate and hasattr(
                                self.disc, "module"
                            ):
                                disc_logits = self.disc.module.evaluate(
                                    obs_dict["disc_obs"]
                                ).detach()
                            else:
                                disc_logits = self.disc.evaluate(
                                    obs_dict["disc_obs"]
                                ).detach()
                            # Eq. 4: Style Reward Calculation
                            # r = max(0, 1 - 0.25 * (D(s, s') - 1)^2)
                            style_reward_term = (
                                1.0
                                - 0.25 * (disc_logits.squeeze(1) - 1.0) ** 2
                            )
                            disc_r = torch.maximum(
                                torch.zeros_like(style_reward_term),
                                style_reward_term,
                            )
                            disc_r[~valid_env_ids] = 0.0
                    disc_r = disc_r * self.amp_rew_scale

                if self.use_dagger:
                    with torch.inference_mode():
                        policy_state_dict["teacher_actions"] = (
                            self.teacher_actor.act_inference(
                                obs_dict["teacher_obs"]
                            ).detach()
                        )

                if self.predict_local_body_pos:
                    local_body_pos_extend_flat = (
                        self.env._get_obs_local_body_pos_extend_flat()
                    )
                    self.storage.update_key(
                        "local_body_pos_extend_flat",
                        local_body_pos_extend_flat,
                    )
                if self.predict_local_body_vel:
                    local_body_vel_extend_flat = (
                        self.env._get_obs_local_body_vel_extend_flat()
                    )
                    self.storage.update_key(
                        "local_body_vel_extend_flat",
                        local_body_vel_extend_flat,
                    )
                if self.predict_root_lin_vel:
                    root_lin_vel = self.env._get_obs_base_lin_vel()
                    self.storage.update_key("root_lin_vel", root_lin_vel)

                # Add policy output and states into storage
                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                obs_dict, task_rewards, dones, infos = self.env.step(
                    actor_state
                )
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                task_rewards, dones = (
                    task_rewards.to(self.device),
                    dones.to(self.device),
                )

                # Only accumulate logging info on the main process
                if self.is_main_process:
                    self.episode_env_tensors.add(infos["to_log"])

                disc_rewards = torch.zeros_like(task_rewards)
                if self.use_amp:
                    disc_rewards = disc_r * self.amp_rew_coef
                    if self.adaptive_disc_rew:
                        self.smoothed_task_rew_scale = (
                            self.smooth_gamma * self.smoothed_task_rew_scale
                            + (1.0 - self.smooth_gamma)
                            * task_rewards.abs().mean().item()
                        )
                        self.smoothed_disc_rew_scale = (
                            self.smooth_gamma * self.smoothed_disc_rew_scale
                            + (1.0 - self.smooth_gamma)
                            * disc_rewards.abs().mean().item()
                        )
                        task_to_disc_rew_ratio = (
                            self.smoothed_task_rew_scale
                            / self.smoothed_disc_rew_scale
                        )
                        disc_rewards = (
                            disc_rewards * task_to_disc_rew_ratio * 0.25
                        )
                    rewards = (
                        self.task_rew_coef * task_rewards
                        + self.amp_rew_coef * disc_rewards
                    )
                else:
                    rewards = task_rewards

                rewards_for_storage_and_gae = (
                    rewards.clone()
                )  # This will be potentially normalized

                if self.normalize_rewards:
                    self.reward_normalizer.update(
                        rewards_for_storage_and_gae[:, None]
                    )  # Local update

                    if (
                        self.use_accelerate
                        and self.accelerator.num_processes > 1
                        and torch.distributed.is_initialized()
                    ):
                        for buff_name in [
                            "running_mean",
                            "running_var",
                            "running_count",
                        ]:
                            buff = getattr(self.reward_normalizer, buff_name)
                            torch.distributed.all_reduce(
                                buff, op=torch.distributed.ReduceOp.AVG
                            )

                    rewards_for_storage_and_gae = (
                        self.reward_normalizer.normalize(
                            rewards_for_storage_and_gae[:, None]
                        ).squeeze(1)
                    )

                    if self.clip_normalized_rewards:
                        rewards_for_storage_and_gae = torch.clamp(
                            rewards_for_storage_and_gae,
                            -self.normalized_reward_clip_value,
                            self.normalized_reward_clip_value,
                        )

                rewards_stored = rewards_for_storage_and_gae.unsqueeze(
                    1
                )  # Shape [num_envs, 1]

                if not self.dagger_only:
                    if "time_outs" in infos:
                        rewards_stored += (
                            self.gamma
                            * policy_state_dict[
                                "values"
                            ]  # Values are based on (normalized) returns
                            * infos["time_outs"].unsqueeze(1).to(self.device)
                        )
                assert len(rewards_stored.shape) == 2
                self.storage.update_key("rewards", rewards_stored)
                if self.use_amp and not self.dagger_only:
                    self.storage.update_key(
                        "task_rewards", task_rewards.unsqueeze(1)
                    )
                    self.storage.update_key(
                        "disc_rewards", disc_rewards.unsqueeze(1)
                    )
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                # Only do bookkeeping for logging on the main process
                if self.is_main_process:
                    # Book keeping
                    if "episode" in infos:
                        self.ep_infos.append(infos["episode"])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            if not self.dagger_only:
                returns, advantages = self._compute_returns(
                    last_obs_dict=obs_dict,
                    policy_state_dict=dict(
                        values=self.storage.query_key("values"),
                        dones=self.storage.query_key("dones"),
                        rewards=self.storage.query_key("rewards"),
                    ),
                )
                self.storage.batch_update_data("returns", returns)
                self.storage.batch_update_data("advantages", advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        if self.use_accelerate and hasattr(self.actor, "module"):
            self.actor.module.reset(dones)
        else:
            self.actor.reset(dones)

        if not self.dagger_only:
            if self.use_accelerate and hasattr(self.critic, "module"):
                self.critic.module.reset(dones)
            else:
                self.critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        if self.use_accelerate and hasattr(self.critic, "module"):
            last_values = self.critic.module.evaluate(
                last_obs_dict["critic_obs"]
            ).detach()
        else:
            last_values = self.critic.evaluate(
                last_obs_dict["critic_obs"]
            ).detach()
        advantage = 0

        values = policy_state_dict["values"]
        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)

        returns = torch.zeros_like(values)

        num_steps = returns.shape[0]

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - values[step]
            )
            advantage = (
                delta
                + next_is_not_terminal * self.gamma * self.lam * advantage
            )
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )
        return returns, advantages

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for policy_state_dict in generator:
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict["Value"] = 0
        loss_dict["Surrogate"] = 0
        loss_dict["Entropy"] = 0
        loss_dict["Smooth_Grad_Penalty"] = 0
        loss_dict["Disc_Loss"] = 0
        loss_dict["Disc_Grad_Penalty"] = 0
        loss_dict["Disc_Agent_Logits_Mean"] = 0
        loss_dict["Disc_Demo_Logits_Mean"] = 0
        loss_dict["Actor_Load_Balancing_Loss"] = 0
        loss_dict["Critic_Load_Balancing_Loss"] = 0
        loss_dict["Bound_Loss"] = 0
        loss_dict["Dagger_loss"] = 0
        loss_dict["Local_Body_Pos_Reg_Loss"] = 0
        loss_dict["Local_Body_Vel_Reg_Loss"] = 0
        loss_dict["Root_Lin_Vel_Reg_Loss"] = 0
        loss_dict["KL_Mean"] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _actor_act_step(self, obs_dict):
        if self.use_accelerate and hasattr(self.actor, "module"):
            return self.actor.module.act(obs_dict["actor_obs"])
        else:
            return self.actor.act(obs_dict["actor_obs"])

    def _critic_eval_step(self, obs_dict):
        if self.use_accelerate and hasattr(self.critic, "module"):
            return self.critic.module.evaluate(obs_dict["critic_obs"])
        else:
            return self.critic.evaluate(obs_dict["critic_obs"])

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict["actions"]
        old_actions_log_prob_batch = policy_state_dict["actions_log_prob"]
        old_mu_batch = policy_state_dict["action_mean"]
        old_sigma_batch = policy_state_dict["action_sigma"]
        if not self.dagger_only:
            target_values_batch = policy_state_dict["values"]
            advantages_batch = policy_state_dict["advantages"]
            returns_batch = policy_state_dict["returns"]

        self.actor_optimizer.zero_grad()

        if not self.dagger_only:
            self.critic_optimizer.zero_grad()
            if self.use_amp:
                self.disc_optimizer.zero_grad()

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = (
            self.actor.get_actions_log_prob(actions_batch)
            if not hasattr(self.actor, "module")
            else self.actor.module.get_actions_log_prob(actions_batch)
        )
        if not self.dagger_only:
            value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = (
            self.actor.action_mean
            if not hasattr(self.actor, "module")
            else self.actor.module.action_mean
        )
        sigma_batch = (
            self.actor.action_std
            if not hasattr(self.actor, "module")
            else self.actor.module.action_std
        )
        entropy_batch = (
            self.actor.entropy
            if not hasattr(self.actor, "module")
            else self.actor.module.entropy
        )

        if self.use_smooth_grad_penalty:
            grad_log_prob = torch.autograd.grad(
                outputs=actions_log_prob_batch.sum(),
                inputs=self.full_actor_obs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_penalty_loss = torch.sum(
                torch.square(grad_log_prob), dim=-1
            ).mean()

        # KL
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
                loss_dict["KL_Mean"] += kl_mean.item()
                lr_scaler = 1.2
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(
                        1e-6, self.actor_learning_rate / lr_scaler
                    )
                    self.critic_learning_rate = max(
                        1e-6, self.critic_learning_rate / lr_scaler
                    )
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(
                        1e-2, self.actor_learning_rate * lr_scaler
                    )
                    self.critic_learning_rate = min(
                        1e-2, self.critic_learning_rate * lr_scaler
                    )

                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = self.critic_learning_rate

        if not self.dagger_only:
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if (
                self.critic_type == "MoEMLP"
                or self.critic_type == "MoEMLPTFEnc"
            ):
                if self.use_accelerate and hasattr(self.actor, "module"):
                    critic_load_balancing_loss = (
                        self.critic.module.critic_module.compute_load_balancing_loss()
                        * self.config.get("load_balancing_loss_alpha", 1e-2)
                    )
                else:
                    critic_load_balancing_loss = (
                        self.critic.critic_module.compute_load_balancing_loss()
                        * self.config.get("load_balancing_loss_alpha", 1e-2)
                    )
                value_loss = value_loss + critic_load_balancing_loss

                loss_dict["Critic_Load_Balancing_Loss"] += (
                    critic_load_balancing_loss.item()
                )

            if self.entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                if self.entropy_curriculum:
                    self.entropy_coef = self.env.entropy_coef
                actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
            else:
                actor_loss = surrogate_loss

        if self.use_dagger:
            teacher_actions_batch = policy_state_dict["teacher_actions"]
            dagger_loss = F.mse_loss(mu_batch, teacher_actions_batch)
            if not self.dagger_only:
                if self.dagger_anneal:
                    self.dagger_coef = self.dagger_coef * (
                        1.0 - self.dagger_anneal_degree
                    )
                if self.rl_anneal:
                    self.rl_coef = self.rl_coef * (1.0 - self.rl_anneal_degree)
                actor_loss = (
                    self.rl_coef * actor_loss + self.dagger_coef * dagger_loss
                )
            else:
                actor_loss = dagger_loss
            loss_dict["Dagger_loss"] += dagger_loss.item()

        if self.use_smooth_grad_penalty:
            smooth_grad_penalty_coef = self.config.get(
                "smooth_grad_penalty_coef", 0.1
            )
            actor_loss = (
                actor_loss + gradient_penalty_loss * smooth_grad_penalty_coef
            )

        # Load balancing loss (only for MoE-based actors)
        if self.actor_type == "MoEMLP" or self.actor_type == "MoEMLPTFEnc":
            if self.use_accelerate and hasattr(self.actor, "module"):
                load_balancing_loss = (
                    self.actor.module.actor_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            else:
                load_balancing_loss = (
                    self.actor.actor_module.compute_load_balancing_loss()
                    * self.config.get("load_balancing_loss_alpha", 1e-2)
                )
            actor_loss = actor_loss + load_balancing_loss
            loss_dict["Actor_Load_Balancing_Loss"] += (
                load_balancing_loss.item()
            )

        # Bound loss (for all actor types)
        if self.use_accelerate and hasattr(self.actor, "module"):
            bound_loss = (
                self.actor.module.actor_module.compute_bound_loss()
                * self.config.get("bound_loss_alpha", 1.0)
            )
        else:
            bound_loss = (
                self.actor.actor_module.compute_bound_loss()
                * self.config.get("bound_loss_alpha", 1.0)
            )
        actor_loss = actor_loss + bound_loss
        loss_dict["Bound_Loss"] += bound_loss.item()

        if self.predict_local_body_pos:
            gt_local_body_pos_extend_flat = policy_state_dict[
                "local_body_pos_extend_flat"
            ]
            if self.use_accelerate and hasattr(self.actor, "module"):
                local_body_pos_reg_loss = (
                    self.actor.module.actor_module.compute_local_body_pos_reg_loss(
                        gt_local_body_pos_extend_flat
                    )
                    * self.pred_local_body_pos_alpha
                )
            else:
                local_body_pos_reg_loss = (
                    self.actor.actor_module.compute_local_body_pos_reg_loss(
                        gt_local_body_pos_extend_flat
                    )
                    * self.pred_local_body_pos_alpha
                )
            actor_loss = actor_loss + local_body_pos_reg_loss
            loss_dict["Local_Body_Pos_Reg_Loss"] += (
                local_body_pos_reg_loss.item()
            )

        if self.predict_local_body_vel:
            gt_local_body_vel_extend_flat = policy_state_dict[
                "local_body_vel_extend_flat"
            ]
            if self.use_accelerate and hasattr(self.actor, "module"):
                local_body_vel_reg_loss = (
                    self.actor.module.actor_module.compute_local_body_vel_reg_loss(
                        gt_local_body_vel_extend_flat
                    )
                    * self.pred_local_body_vel_alpha
                )
            else:
                local_body_vel_reg_loss = (
                    self.actor.actor_module.compute_local_body_vel_reg_loss(
                        gt_local_body_vel_extend_flat
                    )
                    * self.pred_local_body_vel_alpha
                )
            actor_loss = actor_loss + local_body_vel_reg_loss
            loss_dict["Local_Body_Vel_Reg_Loss"] += (
                local_body_vel_reg_loss.item()
            )
        if self.predict_root_lin_vel:
            gt_root_lin_vel = policy_state_dict["root_lin_vel"]
            if self.use_accelerate and hasattr(self.actor, "module"):
                root_lin_vel_reg_loss = (
                    self.actor.module.actor_module.compute_root_lin_vel_reg_loss(
                        gt_root_lin_vel
                    )
                    * self.pred_root_lin_vel_alpha
                )
            else:
                root_lin_vel_reg_loss = (
                    self.actor.actor_module.compute_root_lin_vel_reg_loss(
                        gt_root_lin_vel
                    )
                    * self.pred_root_lin_vel_alpha
                )
            actor_loss = actor_loss + root_lin_vel_reg_loss
            loss_dict["Root_Lin_Vel_Reg_Loss"] += root_lin_vel_reg_loss.item()

        if self.use_amp and not self.dagger_only:
            valid_sample_mask = policy_state_dict[
                "amp_valid_sample_mask"
            ].squeeze()
            if valid_sample_mask.any():
                disc_agent_obs = policy_state_dict["disc_obs"][
                    valid_sample_mask
                ].to(self.device)
                disc_demo_obs = policy_state_dict["disc_demo_obs"][
                    valid_sample_mask
                ].to(self.device)
                disc_demo_obs.requires_grad_(True)
                if self.use_accelerate and hasattr(self.disc, "module"):
                    disc_agent_logits = self.disc.module.evaluate(
                        disc_agent_obs
                    )
                    disc_demo_logits = self.disc.module.evaluate(disc_demo_obs)
                else:
                    disc_agent_logits = self.disc.evaluate(disc_agent_obs)
                    disc_demo_logits = self.disc.evaluate(disc_demo_obs)

                if self.disc_loss_type == "lsgan":
                    disc_loss_agent = (
                        (disc_agent_logits - (-1.0)) ** 2
                    ).mean()
                    disc_loss_demo = ((disc_demo_logits - 1.0) ** 2).mean()
                    disc_loss = (
                        self.disc_loss_coef
                        * 0.5
                        * (disc_loss_agent + disc_loss_demo)
                    )
                elif self.disc_loss_type == "bce":
                    disc_loss_agent = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            disc_agent_logits,
                            torch.zeros_like(
                                disc_agent_logits, device=self.device
                            ),
                        )
                    )
                    disc_loss_demo = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            disc_demo_logits,
                            torch.ones_like(
                                disc_demo_logits, device=self.device
                            ),
                        )
                    )
                    disc_loss = (
                        self.disc_loss_coef
                        * 0.5
                        * (disc_loss_agent + disc_loss_demo)
                    )
                    disc_logit_loss = torch.sum(
                        torch.square(
                            self.disc.module.logits_weights
                            if hasattr(self.disc, "module")
                            else self.disc.logits_weights
                        )
                    )
                    disc_loss = disc_loss + 1e-2 * disc_logit_loss

                disc_demo_grad = torch.autograd.grad(
                    disc_demo_logits,
                    disc_demo_obs,
                    grad_outputs=torch.ones_like(
                        disc_demo_logits, device=self.device
                    ),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                disc_demo_grad_norm = torch.norm(
                    disc_demo_grad, p=2, dim=-1
                ).mean()
                disc_grad_penalty_loss = (
                    self.disc_grad_penalty_coef * disc_demo_grad_norm
                )

                disc_total_loss = disc_loss + disc_grad_penalty_loss
                disc_agent_logits_mean = disc_agent_logits.mean()
                disc_demo_logits_mean = disc_demo_logits.mean()

            else:
                disc_loss = torch.tensor(0.0, device=self.device)
                disc_grad_penalty_loss = torch.tensor(0.0, device=self.device)
                disc_total_loss = torch.tensor(0.0, device=self.device)
                disc_agent_logits_mean = torch.tensor(0.0, device=self.device)
                disc_demo_logits_mean = torch.tensor(0.0, device=self.device)

        if not self.dagger_only:
            critic_loss = self.value_loss_coef * value_loss
        else:
            value_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)

        actor_critic_loss = actor_loss + critic_loss

        if self.use_accelerate and hasattr(self, "accelerator"):
            self.accelerator.backward(actor_critic_loss)
            if self.use_amp and not self.dagger_only:
                if valid_sample_mask.any():
                    self.accelerator.backward(disc_total_loss)
        else:
            actor_critic_loss.backward()
            if self.use_amp and not self.dagger_only:
                if valid_sample_mask.any():
                    disc_total_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        if not self.dagger_only:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            if self.use_amp:
                nn.utils.clip_grad_norm_(
                    self.disc.parameters(), self.max_grad_norm
                )

        self.actor_optimizer.step()
        if not self.dagger_only:
            self.critic_optimizer.step()
            if self.use_amp:
                self.disc_optimizer.step()

        if not self.dagger_only:
            loss_dict["Value"] += value_loss.item()
            loss_dict["Surrogate"] += surrogate_loss.item()
        if self.entropy_coef > 0.0:
            loss_dict["Entropy"] += entropy_loss.item()
        if self.use_smooth_grad_penalty:
            loss_dict["Smooth_Grad_Penalty"] += gradient_penalty_loss.item()
        if self.use_amp:
            loss_dict["Disc_Loss"] += disc_loss.item()
            loss_dict["Disc_Grad_Penalty"] += disc_grad_penalty_loss.item()
            loss_dict["Disc_Agent_Logits_Mean"] += (
                disc_agent_logits_mean.item()
            )
            loss_dict["Disc_Demo_Logits_Mean"] += disc_demo_logits_mean.item()

        return loss_dict

    @property
    def inference_model(self):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        if not self.dagger_only:
            critic = (
                self.critic.module
                if hasattr(self.critic, "module")
                else self.critic
            )
        else:
            critic = None
        return {"actor": actor, "critic": critic}

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        # Skip logging if not the main process
        if not self.is_main_process:
            return

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict["collection_time"] + log_dict["learn_time"]
        iteration_time = log_dict["collection_time"] + log_dict["learn_time"]

        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                if (
                    self.is_main_process
                    and self.tensorboard_writer is not None
                ):
                    self.tensorboard_writer.add_scalar(
                        f"Episode/{key}", value.item(), log_dict["it"]
                    )

        train_log_dict = {}
        actor_model = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        mean_std = actor_model.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (log_dict["collection_time"] + log_dict["learn_time"])
        )
        train_log_dict["fps"] = fps
        train_log_dict["mean_std"] = mean_std.item()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        # Prepare training log data for tabulate
        training_data = {
            "Learning Iteration": f"{log_dict['it']}/{log_dict['total_learning_iterations']}",  # noqa: E501
            "FPS": f"{train_log_dict['fps']:.0f} steps/s",
            "Collection Time": f"{log_dict['collection_time']:.3f}s",
            "Learning Time": f"{log_dict['learn_time']:.3f}s",
            "Total Time": f"{self.tot_time:.2f}s",
            "Iteration Time": f"{iteration_time:.2f}s",
            "Total Timesteps": f"{self.tot_timesteps}",
            "ETA": f"{(self.tot_time / (log_dict['it'] + 1) * (log_dict['total_learning_iterations'] - log_dict['it'])) / 3600:.2f}H",  # noqa: E501
            "Mean Action Noise Std": f"{train_log_dict['mean_std']:.2f}",
            "Entropy Coef": f"{self.entropy_coef:.4e}",
        }

        # Add reward and episode length if available
        if len(log_dict["rewbuffer"]) > 0:
            training_data["Mean Reward"] = (
                f"{statistics.mean(log_dict['rewbuffer']):.2f}"
            )
            training_data["Mean Episode Length"] = (
                f"{statistics.mean(log_dict['lenbuffer']):.2f}"
            )

        # Add environment log data
        for k, v in env_log_dict.items():
            key_name = k.replace("Env/", "")  # Clean up key names
            training_data[key_name] = f"{v:.4f}"

        training_data.update(
            {
                k: f"{v:.4f}" if isinstance(v, torch.Tensor) else f"{v:.4f}"
                for k, v in log_dict["loss_dict"].items()
            }
        )

        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                training_data[f"Mean Episode {key}"] = f"{value:.4f}"
        table_data = [[key, value] for key, value in training_data.items()]
        log_lines = [
            "\n" + "=" * 80,
            f"TRAINING LOG - Iteration {log_dict['it']}/{log_dict['total_learning_iterations']}",  # noqa: E501
            "=" * 80,
            tabulate(
                table_data,
                headers=["Metric", "Value"],
                tablefmt="simple_outline",
            ),
            "=" * 80,
            f"Logging Directory: {os.path.abspath(self.log_dir)}",
            "=" * 80 + "\n",
        ]
        training_log = "\n".join(log_lines)
        logger.info(training_log)

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        # Skip logging if not the main process or TensorBoard writer is None
        if not self.is_main_process or self.tensorboard_writer is None:
            return

        # Logging Loss Dict
        for loss_key, loss_value in log_dict["loss_dict"].items():
            # Skip logging accuracy metrics if they were removed
            if "Acc" not in loss_key:
                self.tensorboard_writer.add_scalar(
                    f"Loss/{loss_key}", loss_value, log_dict["it"]
                )

        self.tensorboard_writer.add_scalar(
            "Loss/actor_learning_rate",
            self.actor_learning_rate,
            log_dict["it"],
        )
        self.tensorboard_writer.add_scalar(
            "Loss/critic_learning_rate",
            self.critic_learning_rate,
            log_dict["it"],
        )
        self.tensorboard_writer.add_scalar(
            "Policy/mean_noise_std", train_log_dict["mean_std"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/total_fps", train_log_dict["fps"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/collection_time", log_dict["collection_time"], log_dict["it"]
        )
        self.tensorboard_writer.add_scalar(
            "Perf/learning_time", log_dict["learn_time"], log_dict["it"]
        )

        if len(log_dict["rewbuffer"]) > 0:
            self.tensorboard_writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(log_dict["rewbuffer"]),
                log_dict["it"],
            )
            self.tensorboard_writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(log_dict["lenbuffer"]),
                log_dict["it"],
            )

        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.tensorboard_writer.add_scalar(k, v, log_dict["it"])

        self.tensorboard_writer.add_scalar(
            "Train/entropy_coef", self.entropy_coef, log_dict["it"]
        )

        # Log mean task and disc rewards from rollout storage if available
        if self.use_amp:
            mean_task_reward = (
                self.storage.query_key("task_rewards").mean().item()
            )
            mean_disc_reward = (
                self.storage.query_key("disc_rewards").mean().item()
            )
            self.tensorboard_writer.add_scalar(
                "Train/mean_task_reward", mean_task_reward, log_dict["it"]
            )
            self.tensorboard_writer.add_scalar(
                "Train/mean_disc_reward", mean_disc_reward, log_dict["it"]
            )

        if self.use_dagger:
            self.tensorboard_writer.add_scalar(
                "Train/dagger_coef", self.dagger_coef, log_dict["it"]
            )

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {
                "obs": obs_dict,
                "rewards": rewards,
                "dones": dones,
                "extras": extras,
            }
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        self.env.resample_motion()
        if reset_env:
            _ = self.env.reset_all()

    def _post_evaluate_policy(self):
        self.env._init_buffers()
        self.env.resample_motion()
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        # Reset PPO bookkeeping for accurate logging post-evaluation
        self.cur_reward_sum.zero_()
        self.cur_episode_length.zero_()
        self._train_mode()  # Switch model back to training mode
        self.env.is_evaluating = False  # Reset evaluation flag

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs = actor_state["obs"]["actor_obs"]
        actions = self.eval_policy(actor_obs)
        actor_state.update({"actions": actions})
        return actor_state

    def _post_eval_env_step(self, actor_state):
        actor_state["step_log_dict"] = self.env.log_dict
        return actor_state

    def _get_inference_policy(self, device=None):
        actor = (
            self.actor.module if hasattr(self.actor, "module") else self.actor
        )
        actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            actor.to(device)
        return actor.act_inference

    def _log_model_summary(self, model, name):
        if not model:
            logger.info(f"{name}: None")
            return

        # Get total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Format parameter counts in K, M, B for readability
        def format_params(count):
            if count >= 1e9:
                return f"{count / 1e9:.2f}B"
            elif count >= 1e6:
                return f"{count / 1e6:.2f}M"
            elif count >= 1e3:
                return f"{count / 1e3:.1f}K"
            else:
                return f"{count}"

        total_params_str = format_params(total_params)
        trainable_params_str = format_params(trainable_params)

        # Get model structure overview
        if hasattr(model, "__class__"):
            model_type = model.__class__.__name__
        else:
            model_type = str(type(model))

        # Get layer info if available
        layer_info = ""
        if hasattr(model, "children"):
            top_level_modules = list(model.children())
            if len(top_level_modules) <= 5:  # Only show if not too many
                layer_info = f", Modules: {len(top_level_modules)}"

        logger.info(
            f"{name} Summary: {model_type}, Total params: {total_params_str}, "
            f"Trainable: {trainable_params_str}{layer_info}"
        )

    @torch.inference_mode()
    def evaluate_policy(self, keyboard_commander=None):
        """Evaluate the trained policy and save motion tracking metrics.

        This function runs the evaluation loop for the trained policy, collects
        motion tracking metrics for each environment
        and global evaluation results to disk. The evaluation includes running
        the policy on all evaluation motion clips, aggregating metrics, and
        saving input/output samples for further analysis.

        Reference: This evaluation logic is self-developed for the HoloMotion
        project, but is inspired by best practices from open-source RL and
        motion imitation frameworks such as IsaacGym and ProtoMotions.

        Args:
            keyboard_commander (optional): Not used in this implementation, but
                can be used to provide manual control during evaluation.

        Returns:
            dict or None: If called on the main process, returns a
                dictionary of global evaluation metrics (e.g., mean
                MPJPE, joint errors, etc.). On non-main processes,
                returns None.
        """
        self._pre_evaluate_policy()
        self.env.is_evaluating = True
        actor_state = self._create_actor_state()

        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(
            self.env.num_envs, self.num_act, device=self.device
        )
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        if self.is_main_process:
            self.dump_inoutput = {
                "input": [],
                "output": [],
            }

        actor_state = self._pre_eval_env_step(actor_state)

        # Create metrics directory on all processes with iteration number
        try:
            metrics_dump_dir = os.path.join(
                self.log_dir,
                f"eval_metrics_iter-{self.current_learning_iteration}",
            )
        except Exception:
            metrics_dump_dir = os.path.join(
                self.env.config.eval_log_dir,
                f"eval_metrics_iter-{self.current_learning_iteration}",
            )
        os.makedirs(metrics_dump_dir, exist_ok=True)

        # Each process maintains its own metrics list
        tracking_metrics_list = []

        # Only show progress bar on main process
        last_eval_batch = False
        total_eval_clips = len(self.env._motion_lib.eval_allocation_schedule)

        if self.is_main_process:
            pbar = tqdm(
                total=total_eval_clips, desc="holomotion Evaluation Progress"
            )

        while not last_eval_batch:
            cached_max_frame_len = self.env._motion_lib.cache.max_frame_length
            last_eval_batch = self.env.resample_motion_eval()

            # Track motion metrics for each environment
            env_tracking_metrics = [{} for _ in range(self.env.num_envs)]

            # Only show inner progress bar on main process
            inner_range = range(cached_max_frame_len)
            if self.is_main_process:
                inner_range = tqdm(
                    inner_range, desc="Evaluating holomotion Batch "
                )

            for step in inner_range:
                self.env.is_evaluating = True
                self.env.commands = torch.zeros(
                    self.env.num_envs, 4, device=self.device
                )
                self.env.commands[:, 0] = 1.0
                actor_state["step"] = step
                actor_state = self._pre_eval_env_step(actor_state)
                actor_state = self.env_step(actor_state)
                actor_state = self._post_eval_env_step(actor_state)

                # Collect motion tracking metrics at specific intervals
                if step % 50 == 0 or step == cached_max_frame_len - 1:
                    # Calculate motion tracking metrics
                    self.env._log_motion_tracking_info()
                    self.env._log_motion_tracking_holomotion_metrics()

                    # Extract metrics for each environment
                    for env_idx in range(self.env.num_envs):
                        for k, v in self.env.log_dict_nonreduced.items():
                            if k not in env_tracking_metrics[env_idx]:
                                env_tracking_metrics[env_idx][k] = []
                            env_tracking_metrics[env_idx][k].append(
                                v[env_idx].item()
                            )

                        if hasattr(self.env, "log_dict_nonreduced_holomotion"):
                            for (
                                k,
                                v,
                            ) in (
                                self.env.log_dict_nonreduced_holomotion.items()
                            ):
                                if k not in env_tracking_metrics[env_idx]:
                                    env_tracking_metrics[env_idx][k] = []
                                env_tracking_metrics[env_idx][k].append(
                                    v[env_idx].item()
                                )

                        for k, v in self.env.log_dict.items():
                            if k not in env_tracking_metrics[env_idx]:
                                env_tracking_metrics[env_idx][k] = []
                            env_tracking_metrics[env_idx][k].append(v.item())

                        if hasattr(self.env, "log_dict_holomotion"):
                            for k, v in self.env.log_dict_holomotion.items():
                                if k not in env_tracking_metrics[env_idx]:
                                    env_tracking_metrics[env_idx][k] = []
                                env_tracking_metrics[env_idx][k].append(
                                    v.item()
                                )

                # Save input/output sample at step 500
                if (
                    self.is_main_process
                    and step == 500
                    and len(self.dump_inoutput["input"]) == 0
                ):
                    self.dump_inoutput = {
                        "input": actor_state["obs"],
                        "output": {"actions": actor_state["actions"]},
                        "step": 500,
                    }

            # Each process collects metrics for its environments
            for i, clip_info in enumerate(
                self.env._motion_lib.cache.cached_clip_info
            ):
                if i < self.env.num_envs:  # Safety check
                    # Add motion tracking metrics
                    tracking_dict = {"clip_info": clip_info}
                    for k, v in env_tracking_metrics[i].items():
                        if len(v) > 0:
                            tracking_dict[f"{k}_mean"] = sum(v) / len(v)
                            tracking_dict[f"{k}_min"] = min(v)
                            tracking_dict[f"{k}_max"] = max(v)
                    tracking_metrics_list.append(tracking_dict)

            # Each process writes its own metrics files
            process_rank = (
                self.process_rank if hasattr(self, "process_rank") else 0
            )
            tracking_metrics_filename = (
                f"eval_tracking_metrics_rank_{process_rank}.json"
            )

            with open(
                os.path.join(metrics_dump_dir, tracking_metrics_filename), "w+"
            ) as f:
                json.dump(tracking_metrics_list, f, indent=2)

            # Only the main process aggregates metrics for reporting
            if self.is_main_process:
                # Calculate average holomotion metrics across all environments
                holomotion_metrics_mean = {}
                if hasattr(self.env, "log_dict_holomotion"):
                    for k in self.env.log_dict_holomotion.keys():
                        values = [
                            sum(env_metrics.get(k, [0]))
                            / max(len(env_metrics.get(k, [0])), 1)
                            for env_metrics in env_tracking_metrics
                        ]
                        holomotion_metrics_mean[k] = sum(values) / max(
                            len(values), 1
                        )

                if holomotion_metrics_mean:
                    logger.info(
                        "holomotion MPJPE_G: %.4f, "
                        "Upper body error: %.4f, "
                        "Lower body error: %.4f",
                        holomotion_metrics_mean.get("mpjpe_g", 0),
                        holomotion_metrics_mean.get(
                            "upper_body_joints_dist", 0
                        ),
                        holomotion_metrics_mean.get(
                            "lower_body_joints_dist", 0
                        ),
                    )

                # Save global metrics to a separate file
                global_metrics = {
                    "iteration": self.current_learning_iteration,
                }

                # Add holomotion metrics to global metrics
                if holomotion_metrics_mean:
                    global_metrics.update(holomotion_metrics_mean)

                with open(
                    os.path.join(metrics_dump_dir, "global_metrics.json"), "w+"
                ) as f:
                    json.dump(global_metrics, f, indent=2)

                # Save input/output sample if available
                if self.dump_inoutput["input"]:
                    with open(
                        os.path.join(
                            metrics_dump_dir, "eval_inoutput_step-500.pkl"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(self.dump_inoutput, f)

            # Update progress bar on main process
            if self.is_main_process:
                pbar.update(self.env.num_envs)

        # Sync all processes after writing files
        if self.use_accelerate and hasattr(self, "accelerator"):
            self.accelerator.wait_for_everyone()

        # Close progress bar on main process only
        if self.is_main_process:
            pbar.close()
            logger.info(
                f"holomotion evaluation metrics saved to {metrics_dump_dir}"
            )

        self._post_evaluate_policy()

        # Return holomotion metrics if main process
        if self.is_main_process:
            return global_metrics
        else:
            return None


class RolloutStorage(nn.Module):
    def __init__(self, num_envs, num_transitions_per_env, device="cpu"):
        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        # self.saved_hidden_states_a = None
        # self.saved_hidden_states_c = None

        self.step = 0
        self.stored_keys = list()

    def register_key(self, key: str, shape=(), dtype=torch.float):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not hasattr(self, key), key
        assert isinstance(shape, (list, tuple)), (
            "shape must be a list or tuple"
        )
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape,
            dtype=dtype,
            device=self.device,
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)

    def increment_step(self):
        self.step += 1

    def update_key(self, key: str, data: torch.Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, (
            "Rollout buffer overflow"
        )
        getattr(self, key)[self.step].copy_(data)

    def batch_update_data(self, key: str, data: torch.Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        getattr(self, key)[:] = data
        # self.store_dict[key] += self.total_sum()

    def _save_hidden_states(self, hidden_states):
        assert NotImplementedError
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0],
                    *hid_a[i].shape,
                    device=self.device,
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0],
                    *hid_c[i].shape,
                    device=self.device,
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def get_statistics(self):
        raise NotImplementedError

    def query_key(self, key: str):
        assert hasattr(self, key), key
        return getattr(self, key)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        _buffer_dict = {
            key: getattr(self, key)[:].flatten(0, 1)
            for key in self.stored_keys
        }

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                _batch_buffer_dict = {
                    key: _buffer_dict[key][batch_idx]
                    for key in self.stored_keys
                }
                yield _batch_buffer_dict


class TensorAverageMeter:
    def __init__(self):
        self.tensors = []

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        else:
            return cat.mean()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        mean_dict = {k: v.mean() for k, v in self.data.items()}
        return mean_dict

    def clear(self):
        self.data = {}

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean
