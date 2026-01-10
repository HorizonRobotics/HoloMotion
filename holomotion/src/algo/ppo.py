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
import statistics
import time
from collections import deque
import sys
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from omegaconf import OmegaConf


from holomotion.src.modules.agent_modules import PPOActor, PPOCritic
from hydra.utils import get_class


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except
                batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns
                input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "_last_sync_mean", torch.zeros(shape).unsqueeze(0)
        )
        self.register_buffer("_last_sync_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer(
            "_last_sync_count", torch.tensor(0, dtype=torch.long)
        )

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    def normalize_only(self, x):
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var + delta_mean * (mean_x - self._mean)
        )
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def sync_stats_across_processes(self, accelerator):
        """Synchronize normalization statistics across distributed processes."""
        if accelerator.num_processes <= 1:
            return

        # Weighted synchronization with correction to avoid double counting
        device = self._mean.device
        count_local = self.count.to(device=device, dtype=torch.float32)
        mean_local = self._mean.to(device=device, dtype=torch.float32)
        var_local = self._var.to(device=device, dtype=torch.float32)

        # Local weighted sums
        sum_count = accelerator.reduce(count_local, reduction="sum")
        sum_mean_count = accelerator.reduce(
            mean_local * count_local, reduction="sum"
        )
        sum_ex2_count = accelerator.reduce(
            (var_local + mean_local * mean_local) * count_local,
            reduction="sum",
        )

        # Correct for replication of previously-synced global stats across ranks
        last_c = self._last_sync_count.to(device=device, dtype=torch.float32)
        if last_c.item() > 0:
            w_minus_1 = float(accelerator.num_processes - 1)
            last_mean = self._last_sync_mean.to(
                device=device, dtype=torch.float32
            )
            last_var = self._last_sync_var.to(
                device=device, dtype=torch.float32
            )
            sum_count = sum_count - w_minus_1 * last_c
            sum_mean_count = sum_mean_count - w_minus_1 * (last_mean * last_c)
            sum_ex2_count = sum_ex2_count - w_minus_1 * (
                (last_var + last_mean * last_mean) * last_c
            )

        if sum_count.item() <= 0:
            return

        global_mean = sum_mean_count / sum_count
        global_ex2 = sum_ex2_count / sum_count
        global_var = torch.clamp(
            global_ex2 - global_mean * global_mean, min=0.0
        )
        global_std = torch.sqrt(global_var)

        # Copy back (keep original buffer shapes)
        self._mean.copy_(global_mean.to(self._mean.dtype))
        self._var.copy_(global_var.to(self._var.dtype))
        self._std.copy_(global_std.to(self._std.dtype))
        # Set global sample count and remember snapshot for next correction
        self.count.copy_(sum_count.to(self.count.dtype))
        self._last_sync_mean.copy_(global_mean.to(self._last_sync_mean.dtype))
        self._last_sync_var.copy_(global_var.to(self._last_sync_var.dtype))
        self._last_sync_count.copy_(self.count)


class RolloutStorage(nn.Module):
    """Simplified rollout storage that matches rsl_rl behavior exactly."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.teacher_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.velocity_commands = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        device="cpu",
        command_name: str = None,
        storage_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.command_name = command_name

        # Use bfloat16 for large tensors (obs, actions, mu, sigma) to save memory.
        # Keep float32 for values used in GAE/advantage computation for precision.
        low_prec = (
            storage_dtype if storage_dtype is not None else torch.float32
        )
        high_prec = torch.float32

        # Core storage (can use lower precision)
        self.observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actor_obs_shape,
            device=self.device,
            dtype=low_prec,
        )
        self.privileged_observations = (
            torch.zeros(
                num_transitions_per_env,
                num_envs,
                *critic_obs_shape,
                device=self.device,
                dtype=low_prec,
            )
            if critic_obs_shape
            else None
        )
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
            dtype=low_prec,
        )
        self.teacher_actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
            dtype=low_prec,
        )
        self.mu = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
            dtype=low_prec,
        )
        self.sigma = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
            dtype=low_prec,
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # PPO specific (keep high precision for numerical stability)
        self.rewards = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
            dtype=high_prec,
        )
        self.values = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
            dtype=high_prec,
        )
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
            dtype=high_prec,
        )
        self.returns = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
            dtype=high_prec,
        )
        self.advantages = torch.zeros(
            num_transitions_per_env,
            num_envs,
            1,
            device=self.device,
            dtype=high_prec,
        )

        # Store velocity commands for advantage normalization
        self.velocity_commands = (
            torch.zeros(
                num_transitions_per_env,
                num_envs,
                4,
                device=self.device,
                dtype=high_prec,
            )
            if command_name == "base_velocity"
            else None
        )

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.privileged_observations
            )
        self.actions[self.step].copy_(transition.actions)
        if transition.teacher_actions is not None:
            self.teacher_actions[self.step].copy_(transition.teacher_actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if (
            self.velocity_commands is not None
            and transition.velocity_commands is not None
        ):
            self.velocity_commands[self.step].copy_(
                transition.velocity_commands
            )

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(
        self, last_values, gamma, lam, normalize_advantage: bool = False
    ):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute raw advantages
        self.advantages = self.returns - self.values
        # Optional local normalization (RSL-style)
        if normalize_advantage:
            flat = self.advantages.view(-1)
            mean = flat.mean()
            std = flat.std().clamp_min(1.0e-8)
            self.advantages = (self.advantages - mean) / std

    def mini_batch_generator(
        self, num_mini_batches, num_epochs=8, accelerator=None
    ):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        privileged_observations = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else observations
        )
        actions = self.actions.flatten(0, 1)
        teacher_actions = self.teacher_actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    teacher_actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )


class PPO:
    """PPO implementation that exactly matches rsl_rl behavior."""

    def __init__(
        self,
        env_config,
        config,
        log_dir=None,
        headless=True,
        is_offline_eval=False,
    ):
        """Initialize PPO algorithm.

        Args:
            env_config: Environment configuration (OmegaConf object with _target_ and config).
            config: PPO algorithm configuration.
            log_dir: Directory for logging and checkpoints.
            headless: Whether to run in headless mode.
            is_offline_eval: Whether this is for offline evaluation (uses "offline_eval.log" instead of "run.log").
        """
        self.config = config
        self.env_config = env_config
        self.log_dir = log_dir
        self.headless = headless
        self.is_offline_eval = is_offline_eval

        self._setup_accelerator()
        self._preview_weighted_bin_config()
        self._setup_environment()
        self._setup_configs()
        self._setup_seeding()
        self._setup_data_buffers()
        self._setup_models_and_optimizer()
        self._setup_simulator()

        # Video recording toggle for offline evaluation (env.render() rgb_array)
        self.record_video: bool = bool(self.config.get("record_video", False))

    def _preview_weighted_bin_config(self) -> None:
        """Preview weighted-bin sampling using manifest-level stats before cache init."""
        sampling_strategy_cfg = self.config.get("sampling_strategy", None)
        if sampling_strategy_cfg is None:
            curriculum_cfg = self.config.get("curriculum", {})
            if bool(curriculum_cfg.get("enabled", False)):
                return
            sampling_strategy = "uniform"
        else:
            sampling_strategy = str(sampling_strategy_cfg).lower()
        if sampling_strategy != "weighted_bin":
            return

        weighted_bin_cfg = dict(self.config.get("weighted_bin", {}))

        # Resolve motion config and manifest path(s) from env_config without constructing env.
        motion_cfg = self.env_config.config.robot.motion
        backend = motion_cfg.get("backend", "hdf5_simple")
        if backend != "hdf5_simple":
            return
        train_hdf5_roots = motion_cfg.get("train_hdf5_roots", None)
        manifest_paths = []
        if train_hdf5_roots:
            for root in train_hdf5_roots:
                manifest_paths.append(os.path.join(str(root), "manifest.json"))
        else:
            hdf5_root = motion_cfg.get("hdf5_root", None)
            if not hdf5_root:
                return
            manifest_paths.append(
                os.path.join(str(hdf5_root), "manifest.json")
            )

        cache_cfg = motion_cfg.get("cache", {})
        batch_size = int(cache_cfg.get("max_num_clips", 1))

        from holomotion.src.training.h5_dataloader import (
            preview_weighted_bin_from_manifest,
        )

        preview_weighted_bin_from_manifest(
            manifest_path=manifest_paths
            if len(manifest_paths) > 1
            else manifest_paths[0],
            batch_size=batch_size,
            cfg=weighted_bin_cfg,
        )

    def _setup_accelerator(self):
        if not self.is_offline_eval:
            os.makedirs(self.log_dir, exist_ok=True)

        accelerator_kwargs = {}
        mixed_precision = self.config.get("mixed_precision", None)
        if mixed_precision in ("fp16", "bf16"):
            accelerator_kwargs["mixed_precision"] = mixed_precision
        dynamo_backend = self.config.get("dynamo_backend", None)
        if dynamo_backend in ("inductor", "aot_eager", "cudagraphs"):
            accelerator_kwargs["dynamo_backend"] = dynamo_backend

        accelerator_kwargs["log_with"] = "tensorboard"
        project_config = ProjectConfiguration(
            project_dir=self.log_dir,
            logging_dir=self.log_dir,
        )
        accelerator_kwargs["project_config"] = project_config

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process

        self.accelerator.init_trackers(
            project_name="holomotion",
            config={
                "precision": mixed_precision if mixed_precision else "fp32",
                "dynamo_backend": dynamo_backend if dynamo_backend else "none",
            },
        )

        logger.remove()
        if self.is_main_process:
            logger.add(
                sys.stdout,
                level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
                colorize=True,
            )
            log_file_name = (
                "offline_eval.log" if self.is_offline_eval else "run.log"
            )
            logger.add(
                os.path.join(self.log_dir, log_file_name),
                level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
                colorize=False,
            )

            used_precision = mixed_precision if mixed_precision else "fp32"
            logger.info(
                f"Accelerator initialized with precision: {used_precision}"
            )
            if dynamo_backend:
                logger.info(f"Accelerator dynamo_backend: {dynamo_backend}")
            logger.info(f"TensorBoard logging enabled at: {self.log_dir}")

        self.process_rank = self.accelerator.process_index
        self.gpu_world_size = self.accelerator.num_processes
        self.gpu_global_rank = self.accelerator.process_index
        self.is_distributed = self.gpu_world_size > 1

    def _setup_environment(self):
        """Setup IsaacLab AppLauncher and environment instance."""
        # Device string from accelerator (handles distributed training)
        device_str = str(self.device)

        # Delayed import to ensure Accelerate is fully initialized before IsaacLab
        from isaaclab.app import AppLauncher

        # Stagger IsaacSim AppLauncher initialization across distributed ranks
        # Use local rank per node to stagger independently on each node
        if self.is_distributed:
            self.accelerator.wait_for_everyone()
            base_delay_s = float(
                os.environ.get("HOLOMOTION_ISAAC_STAGGER_SEC", "5.0")
            )
            # Get local rank within the node (per-node staggering)
            local_rank = getattr(self.accelerator, "local_process_index", None)
            if local_rank is None:
                # Fallback to environment variable if Accelerate doesn't provide it
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            delay_s = base_delay_s * float(local_rank)
            if delay_s > 0.0:
                logger.info(
                    f"[Global Rank {self.gpu_global_rank}, Local Rank {local_rank}] "
                    f"Sleeping {delay_s:.1f}s before IsaacSim AppLauncher init"
                )
            time.sleep(delay_s)

        # Create AppLauncher with accelerator device
        # Enable cameras only when needed:
        # - headless & recording: True (offscreen rendering)
        # - headless & not recording: False (maximize performance)
        # - with GUI: True
        _record_video = bool(self.config.get("record_video", False))
        enable_cameras = _record_video or (not self.headless)

        # Explicitly disable Omniverse multi-GPU rendering to avoid per-process
        # MGPU context creation across all visible GPUs.
        kit_args_str = (
            "--/renderer/multiGpu/enabled=false "
            "--/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1"
        )

        app_launcher_flags = {
            "headless": self.headless,
            "enable_cameras": enable_cameras,
            # Hint to keep viewport active in headless mode when recording
            "video": _record_video,
            "device": device_str,
            "kit_args": kit_args_str,
        }
        self._sim_app_launcher = AppLauncher(**app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app

        # Create environment instance
        env_class = get_class(self.env_config._target_)

        # If the actor/critic modules define obs_schema, propagate them into the
        # environment obs config so that serializers can be built there using
        # IsaacLab atomic observation functions as the shape oracle.
        actor_cfg = getattr(self.config.module_dict, "actor", None)
        critic_cfg = getattr(self.config.module_dict, "critic", None)
        if hasattr(self.env_config, "config"):
            env_cfg = self.env_config.config
            if hasattr(env_cfg, "obs"):
                env_obs_cfg = env_cfg.obs
                merged_schema = {}
                if actor_cfg is not None:
                    actor_schema = actor_cfg.get("obs_schema", None)
                    if actor_schema is not None:
                        # Expect actor_schema to be keyed by group (e.g. "policy").
                        for group_name, group_cfg in actor_schema.items():
                            merged_schema[group_name] = group_cfg
                if critic_cfg is not None:
                    critic_schema = critic_cfg.get("obs_schema", None)
                    if critic_schema is not None:
                        for group_name, group_cfg in critic_schema.items():
                            merged_schema[group_name] = group_cfg
                if merged_schema:
                    env_obs_struct = OmegaConf.is_struct(env_obs_cfg)
                    OmegaConf.set_struct(env_obs_cfg, False)
                    env_obs_cfg.obs_schema = merged_schema
                    OmegaConf.set_struct(env_obs_cfg, env_obs_struct)

        # Set render_mode="rgb_array" only when recording is requested to avoid overhead otherwise
        render_mode = (
            "rgb_array"
            if bool(self.config.get("record_video", False))
            else None
        )
        self.env = env_class(
            config=self.env_config.config,
            device=device_str,
            headless=self.headless,
            log_dir=self.log_dir,
            accelerator=self.accelerator,
            render_mode=render_mode,
        )

    def _setup_configs(self):
        self.num_envs: int = self.env.config.num_envs
        algo_obs_dim_dict = self.config.get("algo_obs_dim_dict", None)
        if (
            algo_obs_dim_dict is not None
            and "policy" in algo_obs_dim_dict
            and "critic" in algo_obs_dim_dict
        ):
            self.num_obs = int(algo_obs_dim_dict["policy"])
            self.num_privileged_obs = int(algo_obs_dim_dict["critic"])
        else:
            obs_serializer = getattr(self.env, "obs_serializer", None)
            critic_obs_serializer = getattr(
                self.env, "critic_obs_serializer", None
            )
            if obs_serializer is None:
                raise RuntimeError(
                    "algo_obs_dim_dict not set and env.obs_serializer "
                    "is missing; cannot infer observation dimensions."
                )
            self.num_obs = int(obs_serializer.obs_flat_dim)
            if critic_obs_serializer is not None:
                self.num_privileged_obs = int(
                    critic_obs_serializer.obs_flat_dim
                )
            else:
                self.num_privileged_obs = 0
        self.num_actions = self.env.config.robot.actions_dim

        self.command_name = list(self.env.config.commands.keys())[0]

        self.save_interval = self.config.save_interval
        self.log_interval = self.config.log_interval
        self.num_steps_per_env = self.config.num_steps_per_env
        self.num_learning_iterations = self.config.num_learning_iterations

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.get(
            "actor_learning_rate", self.config.get("learning_rate", 3e-4)
        )
        self.critic_learning_rate = self.config.get(
            "critic_learning_rate", self.config.get("learning_rate", 3e-4)
        )

        self.optimizer_type = self.config.optimizer_type
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        base_num_mini_batches = int(self.config.num_mini_batches)
        if self.is_distributed:
            scaled_num_mini_batches = base_num_mini_batches * int(
                self.gpu_world_size
            )
        else:
            scaled_num_mini_batches = base_num_mini_batches
        self.num_mini_batches = int(max(1, scaled_num_mini_batches))
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = bool(
            self.config.get("normalize_advantage_per_mini_batch", False)
        )
        self.global_advantage_norm: bool = bool(
            self.config.get("global_advantage_norm", True)
        )
        # Curriculum / sampling strategy config
        self.curriculum_cfg = dict(self.config.get("curriculum", {}))
        sampling_strategy_cfg = self.config.get("sampling_strategy", None)
        if sampling_strategy_cfg is None:
            if bool(self.curriculum_cfg.get("enabled", False)):
                sampling_strategy = "curriculum"
            else:
                sampling_strategy = "uniform"
        else:
            sampling_strategy = str(sampling_strategy_cfg).lower()
        valid_strategies = {"uniform", "weighted_bin", "curriculum"}
        if sampling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid sampling_strategy '{sampling_strategy}'. "
                f"Expected one of {sorted(valid_strategies)}."
            )
        self.sampling_strategy: str = sampling_strategy
        self.curriculum_enabled: bool = self.sampling_strategy == "curriculum"
        self.weighted_bin_cfg = dict(self.config.get("weighted_bin", {}))
        self.dump_sampled_keys: bool = bool(
            self.weighted_bin_cfg.get("dump_sampled_keys", False)
        )
        self.dump_sampled_keys_interval: int = int(
            self.weighted_bin_cfg.get("dump_sampled_keys_interval", 0)
        )

        # Observation normalization
        obs_norm_cfg = self.config.get("obs_norm", {})
        self.obs_norm_enabled = obs_norm_cfg.get("enabled", False)
        self.obs_norm_epsilon = float(obs_norm_cfg.get("epsilon", 1.0e-8))
        self.obs_norm_clip_range = float(obs_norm_cfg.get("clip_range", 10.0))
        self.obs_norm_enable_clipping = bool(
            obs_norm_cfg.get("enable_clipping", False)
        )
        # Periodic cross-rank sync during rollout to avoid per-rank drift
        self.obs_norm_sync_interval_steps = int(
            obs_norm_cfg.get("sync_interval_steps", 0)
        )

        # Distillation / DAgger configuration (teacher-student)
        self.teacher_actor_ckpt_path = self.config.get(
            "teacher_actor_ckpt_path", None
        )
        self.dagger_only: bool = bool(self.config.get("dagger_only", False))
        self.dagger_anneal: bool = bool(self.config.get("dagger_anneal", True))
        self.dagger_anneal_degree: float = float(
            self.config.get("dagger_anneal_degree", 1.0e-5)
        )
        self.dagger_coef: float = float(
            self.config.get("dagger_init_coef", 1.0)
        )
        self.rl_warmup: bool = bool(self.config.get("rl_warmup", False))
        self.rl_warmup_degree: float = float(
            self.config.get("rl_warmup_degree", 1.0e-5)
        )
        self.rl_coef: float = float(self.config.get("rl_init_coef", 1.0))
        # Teacher rollout annealing
        self.use_teacher_rollout_annealing: bool = bool(
            self.config.get("use_teacher_rollout_annealing", False)
        )
        self.teacher_rollout_prob: float = float(
            self.config.get("teacher_rollout_init_prob", 0.0)
        )
        self.teacher_rollout_anneal_degree: float = float(
            self.config.get("teacher_rollout_anneal_degree", 1.0e-5)
        )
        self.teacher_rollout_min_prob: float = float(
            self.config.get("teacher_rollout_min_prob", 0.0)
        )
        # Teacher observation normalization (frozen)
        self.teacher_normalize_observations: bool = False
        self.teacher_obs_normalizer = None

        self.enable_online_eval = self.config.get("enable_online_eval", False)

    def _setup_seeding(self) -> None:
        self.base_seed = int(self.config.get("seed", int(time.time())))
        self.seed = int(self.base_seed + int(self.process_rank))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.env.seed(self.seed)

    def _setup_data_buffers(self):
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self.online_eval_metrics_history = []

        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        self.cur_reward_sum = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self.cur_episode_length = torch.zeros(
            self.env.num_envs,
            dtype=torch.float,
            device=self.device,
        )

        # Storage dtype for memory efficiency (bfloat16 saves ~50% on large tensors)
        storage_dtype_str = self.config.get("storage_dtype", "float32")
        storage_dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        storage_dtype = storage_dtype_map.get(storage_dtype_str, torch.float32)

        self.storage = RolloutStorage(
            self.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [self.num_privileged_obs],
            [self.num_actions],
            device=self.device,
            command_name=self.command_name,
            storage_dtype=storage_dtype,
        )
        self.transition = RolloutStorage.Transition()

    def _setup_models_and_optimizer(self):
        self.obs_serializer = self.env.obs_serializer
        self.critic_obs_serializer = self.env.critic_obs_serializer
        if self.teacher_actor_ckpt_path is not None:
            self.teacher_obs_serializer = self.env.teacher_obs_serializer
            self.use_dagger = True
        else:
            self.use_dagger = False
        self.actor = PPOActor(
            obs_dim_dict=self.obs_serializer,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
        ).to(self.device)

        self.critic = PPOCritic(
            obs_dim_dict=self.critic_obs_serializer,
            module_config_dict=self.config.module_dict.critic,
        ).to(self.device)

        if self.is_main_process:
            logger.info("Actor:\n" + str(self.actor))
            logger.info("Critic:\n" + str(self.critic))
            # Log actor and critic parameter counts (in millions)
            actor_params = sum(p.numel() for p in self.actor.parameters())
            critic_params = sum(p.numel() for p in self.critic.parameters())
            params_table = [
                ["Actor", f"{actor_params / 1.0e6:.3f}"],
                ["Critic", f"{critic_params / 1.0e6:.3f}"],
                ["Total", f"{(actor_params + critic_params) / 1.0e6:.3f}"],
            ]
            logger.info(
                "Model Summary:\n"
                + tabulate(
                    params_table,
                    headers=["Model", "Params (M)"],
                    tablefmt="simple_outline",
                )
            )

        optimizer_class = getattr(optim, self.optimizer_type)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
        )

        # Prepare models and optimizers with accelerator
        # Note: If dynamo_backend is configured, Accelerate will automatically
        # compile models during prepare() using the specified backend.
        dynamo_backend = self.config.get("dynamo_backend", None)
        if dynamo_backend and self.is_main_process:
            logger.info(
                f"Models will be compiled with dynamo_backend='{dynamo_backend}' "
                "during accelerator.prepare()"
            )
        (
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.accelerator.prepare(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        )

        # Setup teacher actor for distillation (frozen)
        self.teacher_actor = None
        if self.use_dagger and self.teacher_actor_ckpt_path is not None:
            self.teacher_actor = PPOActor(
                obs_dim_dict=self.teacher_obs_serializer,
                module_config_dict=self.config.module_dict.actor,
                num_actions=self.num_actions,
                init_noise_std=self.config.init_noise_std,
            ).to(self.device)

            # Freeze teacher
            self.teacher_actor.eval()
            for p in self.teacher_actor.parameters():
                p.requires_grad = False

            # Load teacher checkpoint (support both separate and rsl formats)
            if self.is_main_process:
                logger.info(
                    f"Loading teacher actor checkpoint from {self.teacher_actor_ckpt_path}"
                )
            teacher_ckpt = torch.load(
                self.teacher_actor_ckpt_path, map_location=self.device
            )

            if "actor_model_state_dict" in teacher_ckpt:
                teacher_actor_state = self._clean_state_dict(
                    teacher_ckpt["actor_model_state_dict"]
                )
            elif "model_state_dict" in teacher_ckpt:
                cleaned = self._clean_state_dict(
                    teacher_ckpt["model_state_dict"]
                )
                teacher_actor_state = {
                    k[6:]: v
                    for k, v in cleaned.items()
                    if k.startswith("actor.")
                }
            else:
                teacher_actor_state = None

            if teacher_actor_state is not None:
                self.teacher_actor.load_state_dict(
                    teacher_actor_state, strict=True
                )
                if self.is_main_process:
                    logger.info("Teacher actor weights loaded successfully.")
            else:
                if self.is_main_process:
                    logger.warning(
                        "Teacher checkpoint missing actor weights. Distillation will run uninitialized."
                    )

        if not self.obs_norm_enabled:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(
                self.device
            )
            return

        self.obs_normalizer = EmpiricalNormalization(
            shape=[self.num_obs], eps=self.obs_norm_epsilon
        ).to(self.device)

        self.privileged_obs_normalizer = EmpiricalNormalization(
            shape=[self.num_privileged_obs],
            eps=self.obs_norm_epsilon,
        ).to(self.device)

    def _normalize_advantages_global_by_move_mask(self) -> None:
        """Global advantage normalization split by move vs static for base_velocity.

        Assumes:
        - self.storage.advantages: [T, N, 1]
        - self.storage.velocity_commands: [T, N, 4] with first channel as move_mask
          (1.0 for move, 0.0 for static) and remaining channels [vx, vy, vyaw].
        """
        if self.storage.velocity_commands is None:
            return

        advantages_flat = self.storage.advantages.view(-1).float()
        if advantages_flat.numel() == 0:
            return

        vel_flat = self.storage.velocity_commands.view(
            -1, self.storage.velocity_commands.shape[-1]
        )
        move_channel = vel_flat[:, 0]
        move_mask = move_channel > 0.5
        static_mask = ~move_mask

        count_all = torch.tensor(
            [advantages_flat.numel()],
            device=self.device,
            dtype=torch.float32,
        )
        sum_all_local = advantages_flat.sum()
        sqsum_all_local = (advantages_flat * advantages_flat).sum()
        if self.is_distributed:
            count_all_g = self.accelerator.reduce(count_all, reduction="sum")
            sum_all_g = self.accelerator.reduce(sum_all_local, reduction="sum")
            sqsum_all_g = self.accelerator.reduce(
                sqsum_all_local, reduction="sum"
            )
        else:
            count_all_g = count_all
            sum_all_g = sum_all_local
            sqsum_all_g = sqsum_all_local
        mean_all = sum_all_g / count_all_g
        var_all = (sqsum_all_g / count_all_g) - mean_all * mean_all
        std_all = torch.sqrt(var_all.clamp_min(1.0e-8))

        def _group_stats(mask: torch.Tensor):
            if not mask.any():
                return mean_all, std_all
            mask_f = mask.float()
            count_local = mask_f.sum()
            sum_local = (advantages_flat * mask_f).sum()
            sqsum_local = (advantages_flat * advantages_flat * mask_f).sum()
            if self.is_distributed:
                count_g = self.accelerator.reduce(count_local, reduction="sum")
                sum_g = self.accelerator.reduce(sum_local, reduction="sum")
                sqsum_g = self.accelerator.reduce(sqsum_local, reduction="sum")
            else:
                count_g = count_local
                sum_g = sum_local
                sqsum_g = sqsum_local
            if count_g.item() <= 0:
                return mean_all, std_all
            mean = sum_g / count_g
            var = (sqsum_g / count_g) - mean * mean
            std = torch.sqrt(var.clamp_min(1.0e-8))
            return mean, std

        move_mean, move_std = _group_stats(move_mask)
        static_mean, static_std = _group_stats(static_mask)

        advantages_norm = advantages_flat.clone()
        if move_mask.any():
            advantages_norm[move_mask] = (
                advantages_flat[move_mask] - move_mean
            ) / move_std
        if static_mask.any():
            advantages_norm[static_mask] = (
                advantages_flat[static_mask] - static_mean
            ) / static_std

        self.storage.advantages = advantages_norm.view_as(
            self.storage.advantages
        )

    def _setup_simulator(self):
        _ = self.env.reset_all()

    def act(self, obs, critic_obs):
        """Act function using separate actor and critic."""
        with self.accelerator.autocast():
            actions, actions_log_prob, mu, sigma, _ = self.actor(
                obs, actions=None, mode="sampling"
            )
            values = self.critic(critic_obs)
        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = actions_log_prob.detach()
        self.transition.action_mean = mu.detach()
        self.transition.action_sigma = sigma.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, time_outs, infos):
        """Process environment step that matches rsl_rl exactly."""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * time_outs.unsqueeze(1), 1
        )

        # Extract velocity commands for advantage normalization
        if self.command_name == "base_velocity":
            velocity_cmd = self.env._env.command_manager.get_command(
                "base_velocity"
            )
            # Compute move_mask: norm of velocity > 0.1
            move_mask = velocity_cmd.norm(dim=-1) > 0.1
            # Store [move_mask, velocity_cmd] with shape [num_envs, 4]
            self.transition.velocity_commands = torch.cat(
                [move_mask[..., None], velocity_cmd], dim=-1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        """Compute returns and optionally normalize advantages (RSL-style)."""
        with self.accelerator.autocast():
            last_values = self.critic(last_critic_obs)
        last_values = last_values.detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=(
                False
                if self.global_advantage_norm
                else (not self.normalize_advantage_per_mini_batch)
            ),
        )

    def update(self):
        """Update function that matches rsl_rl exactly."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_dagger_loss = 0

        adaptive_kl_enabled = (
            self.desired_kl is not None and self.schedule == "adaptive"
        )

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            self.accelerator,
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            teacher_actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    flat = advantages_batch.view(-1).float()
                    if self.global_advantage_norm and self.is_distributed:
                        count = torch.tensor(
                            [flat.numel()],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        sum_g = self.accelerator.reduce(
                            flat.sum(), reduction="sum"
                        )
                        sqsum_g = self.accelerator.reduce(
                            (flat * flat).sum(), reduction="sum"
                        )
                        count_g = self.accelerator.reduce(
                            count, reduction="sum"
                        )
                        mean = sum_g / count_g
                        var = (sqsum_g / count_g) - mean * mean
                        std = torch.sqrt(var.clamp_min(1.0e-8))
                    else:
                        mean = flat.mean()
                        std = flat.std().clamp_min(1.0e-8)
                    advantages_batch = (advantages_batch - mean) / std

            with self.accelerator.autocast():
                (
                    _,
                    actions_log_prob_batch,
                    mu_batch,
                    sigma_batch,
                    entropy_batch,
                ) = self.actor(obs_batch, actions=actions_batch, mode="logp")
                value_pred = self.critic(critic_obs_batch)
            actions_log_prob_batch = actions_log_prob_batch.float()

            value_batch = value_pred
            returns_batch_norm = returns_batch
            target_values_batch_norm = target_values_batch

            if adaptive_kl_enabled:
                with torch.no_grad():
                    kl_vec = torch.sum(
                        torch.log(
                            (sigma_batch + 1.0e-8) / (old_sigma_batch + 1.0e-8)
                        )
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch) + 1.0e-8)
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean_local = kl_vec.mean()
                    if self.is_distributed:
                        kl_mean_global = self.accelerator.reduce(
                            kl_mean_local, reduction="mean"
                        )
                    else:
                        kl_mean_global = kl_mean_local

                    km = float(kl_mean_global.item())
                    min_lr = 1e-6
                    max_lr = 1
                    lr_scaler = 1.2
                    if km > self.desired_kl * 2.0:
                        self.actor_learning_rate = max(
                            min_lr, self.actor_learning_rate / lr_scaler
                        )
                        self.critic_learning_rate = max(
                            min_lr, self.critic_learning_rate / lr_scaler
                        )
                    elif km > 0.0 and km < self.desired_kl / 2.0:
                        self.actor_learning_rate = min(
                            max_lr, self.actor_learning_rate * lr_scaler
                        )
                        self.critic_learning_rate = min(
                            max_lr, self.critic_learning_rate * lr_scaler
                        )

                # Update learning rates for all param groups
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = self.critic_learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch).float()
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch_norm + (
                    value_batch - target_values_batch_norm
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch_norm).pow(2)
                value_losses_clipped = (
                    value_clipped - returns_batch_norm
                ).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch_norm - value_batch).pow(2).mean()

            # Separate actor and critic losses with auxiliary losses
            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss

            # Entropy loss
            if self.entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                actor_loss = actor_loss - self.entropy_coef * entropy_loss

            # Distillation (DAgger) loss: MSE between student mean and teacher actions
            dagger_loss = None
            if self.use_dagger and teacher_actions_batch is not None:
                dagger_loss = F.mse_loss(mu_batch, teacher_actions_batch)
                # Anneal and warmup coefficients
                if not self.dagger_only:
                    if self.dagger_anneal:
                        self.dagger_coef = self.dagger_coef * (
                            1.0 - self.dagger_anneal_degree
                        )
                    if self.rl_warmup:
                        self.rl_coef = min(
                            1.0, self.rl_coef * (1.0 + self.rl_warmup_degree)
                        )
                    actor_loss = (
                        self.rl_coef * actor_loss
                        + self.dagger_coef * dagger_loss
                    )
                else:
                    actor_loss = self.dagger_coef * dagger_loss

            # Zero gradients for both optimizers
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Separate backward passes for actor and critic
            self.accelerator.backward(actor_loss)
            if not self.dagger_only:
                self.accelerator.backward(critic_loss)

            # Gradient clipping for actor
            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )

            # Gradient clipping for critic
            if self.max_grad_norm is not None and not self.dagger_only:
                self.accelerator.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

            # Separate optimizer steps
            self.actor_optimizer.step()
            if not self.dagger_only:
                self.critic_optimizer.step()

            mean_value_loss += 0.0 if self.dagger_only else value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if self.use_dagger and dagger_loss is not None:
                mean_dagger_loss += dagger_loss.item()

            # Track dagger loss to return/log
            if self.use_dagger and dagger_loss is not None:
                if "Dagger_loss" not in locals():
                    pass

        num_updates = self.num_learning_epochs * self.num_mini_batches
        denom = max(1, num_updates)
        mean_value_loss /= denom
        mean_surrogate_loss /= denom
        mean_entropy /= denom
        if self.use_dagger:
            mean_dagger_loss /= denom

        self.storage.clear()

        loss_out = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.use_dagger:
            loss_out["Dagger_loss"] = mean_dagger_loss

        # Teacher rollout annealing schedule update
        if self.use_dagger and self.use_teacher_rollout_annealing:
            self.teacher_rollout_prob = max(
                self.teacher_rollout_min_prob,
                self.teacher_rollout_prob
                * (1.0 - self.teacher_rollout_anneal_degree),
            )

        # Reduce losses across processes for consistent logging on rank 0
        if self.is_distributed:
            reduced_out = {}
            for k, v in loss_out.items():
                if v is None:
                    reduced_out[k] = None
                    continue
                t = torch.tensor(v, device=self.device, dtype=torch.float32)
                reduced_t = self.accelerator.reduce(t, reduction="mean")
                reduced_out[k] = float(reduced_t.item())
            loss_out = reduced_out

        return loss_out

    def learn(self):
        """Main learning loop that matches rsl_rl exactly."""
        obs_dict = self.env.reset_all()[0]
        obs_raw = obs_dict["policy"].to(self.device)
        privileged_obs_raw = obs_dict["critic"].to(self.device)

        # Configure motion-cache sampling strategy if available
        motion_cmd = None
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")

        # Normalize initial observations using current stats without updating
        if self.obs_norm_enabled:
            obs = self.obs_normalizer.normalize_only(obs_raw)
            privileged_obs = self.privileged_obs_normalizer.normalize_only(
                privileged_obs_raw
            )
            if self.obs_norm_enable_clipping:
                obs = torch.clamp(
                    obs, -self.obs_norm_clip_range, self.obs_norm_clip_range
                )
                privileged_obs = torch.clamp(
                    privileged_obs,
                    -self.obs_norm_clip_range,
                    self.obs_norm_clip_range,
                )
        else:
            obs = obs_raw
            privileged_obs = privileged_obs_raw

        self.actor.train()
        self.critic.train()

        num_learning_iterations = self.num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Synchronize all processes before starting
        self.accelerator.wait_for_everyone()

        if self.is_main_process:
            logger.info(
                f"Starting training for {num_learning_iterations} iterations from iteration {self.current_learning_iteration}"
            )

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Rollout
            with torch.no_grad():
                for step_idx in range(self.num_steps_per_env):
                    # Sample actions using normalized observations
                    actions = self.act(obs, privileged_obs)

                    # If distillation is enabled, compute teacher actions (using separate teacher obs)
                    if self.use_dagger and self.teacher_actor is not None:
                        teacher_obs = obs_dict.get("teacher", None)
                        if teacher_obs is None:
                            logger.error(
                                "Teacher observations not found in obs_dict under key 'teacher'"
                            )
                        else:
                            teacher_obs = teacher_obs.to(self.device)

                        # Apply frozen teacher normalizer if available
                        if (
                            teacher_obs is not None
                            and self.teacher_normalize_observations
                            and self.teacher_obs_normalizer is not None
                        ):
                            with torch.no_grad():
                                teacher_obs = (
                                    self.teacher_obs_normalizer.normalize(
                                        teacher_obs
                                    )
                                )

                        if teacher_obs is not None:
                            teacher_actions, _, _, _, _ = self.teacher_actor(
                                teacher_obs, actions=None, mode="inference"
                            )
                            teacher_actions = teacher_actions.detach()
                            self.transition.teacher_actions = teacher_actions

                    # Step the environment
                    obs_dict, rewards, dones, time_outs, infos = self.env.step(
                        actions
                    )

                    # Raw observations from environment
                    obs_raw = obs_dict["policy"].to(self.device)
                    privileged_obs_raw = obs_dict["critic"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Update normalization statistics with raw observations
                    if self.obs_norm_enabled:
                        self.obs_normalizer.update(obs_raw)
                        self.privileged_obs_normalizer.update(
                            privileged_obs_raw
                        )

                        obs = self.obs_normalizer.normalize_only(obs_raw)
                        privileged_obs = (
                            self.privileged_obs_normalizer.normalize_only(
                                privileged_obs_raw
                            )
                        )
                        if self.obs_norm_enable_clipping:
                            obs = torch.clamp(
                                obs,
                                -self.obs_norm_clip_range,
                                self.obs_norm_clip_range,
                            )
                            privileged_obs = torch.clamp(
                                privileged_obs,
                                -self.obs_norm_clip_range,
                                self.obs_norm_clip_range,
                            )
                    else:
                        obs = obs_raw
                        privileged_obs = privileged_obs_raw

                    # Periodically synchronize observation normalizers mid-rollout
                    if (
                        self.obs_norm_enabled
                        and self.obs_norm_sync_interval_steps > 0
                        and (
                            (step_idx + 1) % self.obs_norm_sync_interval_steps
                            == 0
                        )
                    ):
                        self.synchronize_normalizers()

                    # Process the step
                    self.process_env_step(rewards, dones, time_outs, infos)

                    self.ep_infos.append(infos["log"])

                    # Update reward tracking
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1

                    # Handle episode completion
                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[done_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[done_ids][:, 0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    self.cur_reward_sum[done_ids] = 0
                    self.cur_episode_length[done_ids] = 0

                # Synchronize normalizers across ranks before computing returns
                if self.obs_norm_enabled:
                    self.synchronize_normalizers()
                # Compute returns using the last normalized critic observations
                self.compute_returns(privileged_obs)
                if (
                    self.global_advantage_norm
                    and not self.normalize_advantage_per_mini_batch
                ):
                    if (
                        self.command_name == "base_velocity"
                        and self.storage.velocity_commands is not None
                    ):
                        self._normalize_advantages_global_by_move_mask()
                    else:
                        adv = self.storage.advantages.view(-1).float()
                        count = torch.tensor(
                            [adv.numel()],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        sum_local = adv.sum()
                        sqsum_local = (adv * adv).sum()
                        if self.is_distributed:
                            count_g = self.accelerator.reduce(
                                count, reduction="sum"
                            )
                            sum_g = self.accelerator.reduce(
                                sum_local, reduction="sum"
                            )
                            sqsum_g = self.accelerator.reduce(
                                sqsum_local, reduction="sum"
                            )
                        else:
                            count_g = count
                            sum_g = sum_local
                            sqsum_g = sqsum_local
                        mean = sum_g / count_g
                        var = (sqsum_g / count_g) - mean * mean
                        std = torch.sqrt(var.clamp_min(1.0e-8))
                        self.storage.advantages = (
                            self.storage.advantages - mean
                        ) / std

            stop = time.time()
            collection_time = stop - start
            start = stop

            # Update policy
            loss_dict = self.update()
            # Log Dagger loss if used
            if self.use_dagger:
                if "Dagger_loss" not in loss_dict:
                    loss_dict["Dagger_loss"] = None

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging and saving on main process
            if self.is_main_process and it % self.log_interval == 0:
                self._log(locals())

            if it % self.save_interval == 0:
                if self.is_main_process:
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )

                # Trigger online evaluation after checkpoint save (all ranks)
                if self.enable_online_eval:
                    if self.is_main_process:
                        logger.info(
                            "Starting distributed online evaluation after checkpoint save..."
                        )
                    eval_metrics = self.online_evaluate_policy()

                    # Only main process records TB and history
                    if self.is_main_process and eval_metrics:
                        self.online_eval_metrics_history.append(
                            {"iteration": it, "metrics": eval_metrics}
                        )
                        if self.log_dir:
                            # Log online evaluation metrics using Accelerate
                            online_eval_metrics = {
                                f"OnlineEval/{key}": value
                                for key, value in eval_metrics.items()
                            }
                            self.accelerator.log(online_eval_metrics, step=it)

            # Barrier-applied motion cache swap (avoids mid-rollout teleports)
            if self.command_name == "ref_motion":
                motion_cmd = self.env._env.command_manager.get_term(
                    "ref_motion"
                )
                motion_cmd.apply_cache_swap_if_pending_barrier()

            self.ep_infos.clear()

            # Synchronize processes after each iteration
            self.accelerator.wait_for_everyone()

        # Only save on main process when using distributed training
        if self.is_main_process:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )

        # Gracefully release motion cache resources (if any).
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")
            if motion_cmd is not None:
                motion_cmd.close()

        # End training and finalize trackers
        if self.log_dir:
            self.accelerator.end_training()
            if self.is_main_process:
                logger.info(
                    f"Training completed. Model saved to {self.log_dir}"
                )

    def record_video(self, num_steps: int, out_path: str, fps: int) -> dict:
        """Record viewport rendering to an MP4 using rgb_array frames.
        Args:
            num_steps: Number of simulation steps to record.
            out_path: Absolute path to the output MP4 file.
            fps: Target frames per second for the video.
        Returns:
            dict: Metadata with output path and recorded frames.
        """
        import imageio
        import numpy as np

        # Switch models and normalizers to eval for deterministic rollouts
        self.actor.eval()
        self.critic.eval()
        if self.obs_norm_enabled:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

        # Reset environment
        obs_dict = self.env.reset_all()[0]
        obs_raw = obs_dict["policy"].to(self.device)
        if self.obs_norm_enabled:
            obs = self.obs_normalizer.normalize_only(obs_raw)
            if self.obs_norm_enable_clipping:
                obs = torch.clamp(
                    obs, -self.obs_norm_clip_range, self.obs_norm_clip_range
                )
        else:
            obs = obs_raw

        # Prepare video writer; macro_block_size=None allows arbitrary resolution
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = imageio.get_writer(
            out_path, fps=int(fps), macro_block_size=None
        )

        # Prime a first frame
        first_frame = self.env._env.render()
        if isinstance(first_frame, np.ndarray):
            writer.append_data(first_frame)

        with torch.no_grad():
            for _ in range(int(max(0, num_steps - 1))):
                actions, _, _, _, _ = self.actor(
                    obs, actions=None, mode="inference"
                )
                obs_dict, _, _, _, _ = self.env.step(actions)
                obs_raw = obs_dict["policy"].to(self.device)
                if self.obs_norm_enabled:
                    obs = self.obs_normalizer.normalize_only(obs_raw)
                    if self.obs_norm_enable_clipping:
                        obs = torch.clamp(
                            obs,
                            -self.obs_norm_clip_range,
                            self.obs_norm_clip_range,
                        )
                else:
                    obs = obs_raw

                frame = self.env._env.render()
                if isinstance(frame, np.ndarray):
                    writer.append_data(frame)

        writer.close()
        if self.is_main_process:
            logger.info(f"Saved video to: {out_path}")
        # Restore train mode for continued use
        self.actor.train()
        self.critic.train()
        if self.obs_norm_enabled:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()
        return {"video_path": out_path, "frames": int(num_steps)}

    def online_evaluate_policy(self):
        """Run online evaluation using validation motion library.

        Returns:
            dict: Aggregated evaluation metrics
        """
        command_name = list(self.env.config.commands.keys())[0]
        if command_name != "ref_motion":
            logger.warning(
                "Online evaluation only supported for ref_motion command"
            )
            return {}

        motion_cmd = self.env._env.command_manager.get_term("ref_motion")

        cache = motion_cmd._motion_cache
        logger.info("Starting online evaluation (simple cache)...")

        # Switch to validation mode
        cache.set_mode("val")
        motion_cmd._is_evaluating = True

        # Set models to eval mode
        self.actor.eval()
        self.critic.eval()
        if self.obs_norm_enabled:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

        # Ensure command metrics are normal tensors (not inference tensors)
        try:
            if hasattr(motion_cmd, "metrics") and isinstance(
                motion_cmd.metrics, dict
            ):
                for k, v in list(motion_cmd.metrics.items()):
                    if (
                        isinstance(v, torch.Tensor)
                        and getattr(v, "_is_zerocopy", False) is False
                    ):
                        motion_cmd.metrics[k] = v.detach().clone()
        except Exception:
            pass

        num_envs = self.env.num_envs
        all_env_metrics = []
        episode_lengths_all: list = []

        with torch.no_grad():
            # Exhaustively iterate through all validation batches sequentially
            total_batches = cache.num_batches
            for batch_idx in range(total_batches):
                # Uniformly sample clips for envs from current cache batch; start at frame 0
                clip_idx, frame_idx = cache.sample_env_assignments(
                    num_envs,
                    motion_cmd.cfg.n_fut_frames,
                    self.device,
                    deterministic_start=True,
                )
                motion_cmd._clip_indices[:] = clip_idx
                motion_cmd._frame_indices[:] = frame_idx

                # Reset and update reference state from cache for selected clips
                obs_dict = self.env.reset_all()[0]
                motion_cmd._update_ref_motion_state_from_cache()
                all_ids = torch.arange(
                    num_envs, dtype=torch.long, device=self.device
                )
                motion_cmd._align_root_to_ref(all_ids)
                motion_cmd._align_dof_to_ref(all_ids)

                # Recompute observations post-alignment
                obs_dict = self.env._env.observation_manager.compute(
                    update_history=True
                )
                obs = obs_dict["policy"].to(self.device)
                privileged_obs = obs_dict["critic"].to(self.device)
                obs = self.obs_normalizer(obs)
                privileged_obs = self.privileged_obs_normalizer(privileged_obs)

                # Track first-done per env and episode lengths
                env_has_done = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                episode_lengths = torch.zeros(
                    num_envs, dtype=torch.long, device=self.device
                )
                collected_metrics: dict = {}

                max_steps = cache.max_frame_length
                for step in range(max_steps):
                    # Increment episode lengths for active envs
                    episode_lengths += (~env_has_done).long()

                    # Inference and env step
                    actions, _, _, _, _ = self.actor(
                        obs, actions=None, mode="inference"
                    )
                    obs_dict, rewards, dones, time_outs, infos = self.env.step(
                        actions
                    )
                    obs = obs_dict["policy"].to(self.device)
                    privileged_obs = obs_dict["critic"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    obs = self.obs_normalizer(obs)
                    privileged_obs = self.privileged_obs_normalizer(
                        privileged_obs
                    )

                    newly_done = dones.bool() & ~env_has_done
                    if torch.any(newly_done):
                        done_envs = torch.nonzero(newly_done).squeeze(-1)
                        # Episode length for these envs
                        episode_lengths_all.append(
                            episode_lengths[done_envs].detach().float().cpu()
                        )
                        # Metrics from infos['log'] with ref_motion prefix
                        if isinstance(infos, dict) and infos.get("log"):
                            for key, value in infos["log"].items():
                                if not isinstance(key, str):
                                    continue
                                if not key.startswith("Metrics/ref_motion/"):
                                    continue
                                v = value
                                if not isinstance(v, torch.Tensor):
                                    v = torch.tensor(v, device=self.device)
                                if v.dim() == 0:
                                    v = v.unsqueeze(0).expand(num_envs)
                                done_values = v[done_envs]
                                if done_values.numel() == 0:
                                    continue
                                collected_metrics.setdefault(key, []).append(
                                    done_values.detach().float().cpu()
                                )
                        env_has_done |= newly_done

                    if torch.all(env_has_done):
                        break

                # Store metrics for this batch
                if collected_metrics:
                    all_env_metrics.append(collected_metrics)

                # Advance to next cache batch except after the last iteration
                if batch_idx < total_batches - 1:
                    cache.advance()

        # Aggregate metrics
        if not all_env_metrics and not episode_lengths_all:
            logger.warning("No evaluation metrics collected")
            cache.set_mode("train")
            motion_cmd._is_evaluating = False

            # Restore models and normalizers to train mode after evaluation
            self.actor.train()
            self.critic.train()
            if self.obs_norm_enabled:
                self.obs_normalizer.train()
                self.privileged_obs_normalizer.train()
            return {}

        aggregated = {}
        for batch_metrics in all_env_metrics:
            for key, values in batch_metrics.items():
                aggregated.setdefault(key, []).extend(values)

        final_metrics = {}

        for key, values in aggregated.items():
            # Concatenate lists of tensors then mean
            if len(values) == 0:
                continue
            stacked = torch.cat([torch.as_tensor(v) for v in values])
            final_metrics[f"{key}_mean"] = float(stacked.mean().item())

        # Mean episode length across all finished envs
        if episode_lengths_all:
            elens = torch.cat(episode_lengths_all)
            final_metrics["Episode/length_mean"] = float(elens.mean().item())

        # Reduce across processes if distributed
        if final_metrics:
            reduced = {}
            for k, v in final_metrics.items():
                t = torch.tensor(v, device=self.device, dtype=torch.float32)
                rv = self.accelerator.reduce(t, reduction="mean")
                reduced[k] = float(rv.item())
            final_metrics = reduced

        self._back_to_train(motion_cmd)
        return final_metrics

    def _back_to_train(self, motion_cmd):
        """Restore training mode after evaluation and reset episode tracking.

        Args:
            motion_cmd: Reference motion command term
        """
        cache = motion_cmd._motion_cache
        cache.set_mode("train")
        motion_cmd._is_evaluating = False

        # Restore models and normalizers to train mode
        self.actor.train()
        self.critic.train()
        if self.obs_norm_enabled:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

        # Sample new clips and reset environment for training
        num_envs = self.env.num_envs
        clip_idx, frame_idx = cache.sample_env_assignments(
            num_envs,
            motion_cmd.cfg.n_fut_frames,
            self.device,
            deterministic_start=False,
        )
        motion_cmd._clip_indices[:] = clip_idx
        motion_cmd._frame_indices[:] = frame_idx
        motion_cmd._update_ref_motion_state_from_cache()
        all_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        motion_cmd._align_root_to_ref(all_ids)
        motion_cmd._align_dof_to_ref(all_ids)

        # Reset episode tracking to avoid contamination from eval episodes
        self.cur_reward_sum.zero_()
        self.cur_episode_length.zero_()

        logger.info("Restored training mode after evaluation")

    def _log_eval_results(self, metrics: dict, num_clips: int):
        """Format and log evaluation results with tabulate, and save to JSON.

        Args:
            metrics: Dictionary of evaluation metrics
            num_clips: Number of clips evaluated
        """
        import json
        from datetime import datetime

        if not metrics:
            logger.warning("No evaluation metrics to log")
            return

        # Build table data
        table_data = []

        # Summary section
        table_data.append(["=== SUMMARY ===", ""])
        table_data.append(["Total Clips Evaluated", f"{num_clips}"])
        table_data.append(
            ["Current Iteration", f"{self.current_learning_iteration}"]
        )
        table_data.append(["", ""])

        # Metrics section - group by metric type
        table_data.append(["=== EVALUATION METRICS ===", ""])

        # Separate mean and std metrics
        mean_metrics = {
            k: v for k, v in metrics.items() if k.endswith("_mean")
        }

        # Display metrics with their std
        for key in sorted(mean_metrics.keys()):
            metric_name = key.replace("_mean", "")
            mean_val = mean_metrics[key]
            display_name = metric_name.replace("_", " ").title()
            table_data.append([display_name, f"{mean_val:.4f}"])

        # Print formatted table
        log_lines = [
            "\n" + "=" * 80,
            f"ONLINE EVALUATION RESULTS - Iteration {self.current_learning_iteration}",
            "=" * 80,
            tabulate(
                table_data,
                headers=["Metric", "Value"],
                tablefmt="simple_outline",
            ),
            "=" * 80 + "\n",
        ]
        eval_log = "\n".join(log_lines)
        logger.info(eval_log)

        # Save to JSON file
        if self.log_dir:
            eval_results = {
                "iteration": self.current_learning_iteration,
                "timestamp": datetime.now().isoformat(),
                "num_clips_evaluated": num_clips,
                "metrics": metrics,
            }

            # Save to checkpoint-specific file
            json_filename = (
                f"eval_metrics_iter_{self.current_learning_iteration}.json"
            )
            json_path = os.path.join(self.log_dir, json_filename)

            with open(json_path, "w") as f:
                json.dump(eval_results, f, indent=2)

            logger.info(f"Evaluation metrics saved to: {json_path}")

            # Also save to a "latest" file for easy access
            latest_json_path = os.path.join(
                self.log_dir, "eval_metrics_latest.json"
            )
            with open(latest_json_path, "w") as f:
                json.dump(eval_results, f, indent=2)

    def record_metric(self, env_tracking_metrics, motion_cmd, record_path):
        holomotion_metrics_mean = {}
        if hasattr(motion_cmd, "log_dict_holomotion"):
            for k in motion_cmd.log_dict_holomotion.keys():
                values = [
                    sum(env_metrics.get(k, [0]))
                    / max(len(env_metrics.get(k, [0])), 1)
                    for env_metrics in env_tracking_metrics
                ]
                holomotion_metrics_mean[k] = sum(values) / max(len(values), 1)

        if holomotion_metrics_mean:
            holomotion_metrics = {
                "MPJPE_G": f"{holomotion_metrics_mean.get('mpjpe_g', 0):.4f}",  # noqa: E501
                "MPJPE_L": f"{holomotion_metrics_mean.get('mpjpe_l', 0):.4f}",  # noqa: E501
                "MPJPE_PA": f"{holomotion_metrics_mean.get('mpjpe_pa', 0):.4f}",  # noqa: E501
                "ACCELERATION_DIST": f"{holomotion_metrics_mean.get('accel_dist', 0):.4f}",  # noqa: E501
                "VELOCITY_DIST": f"{holomotion_metrics_mean.get('vel_dist', 0):.4f}",  # noqa: E501
                "UPPER_BODY_JOINTS_DIST": f"{holomotion_metrics_mean.get('upper_body_joints_dist', 0):.4f}",  # noqa: E501
                "LOWER_BODY_JOINTS_DIST": f"{holomotion_metrics_mean.get('lower_body_joints_dist', 0):.4f}",  # noqa: E501
                "ROOT_Roll_ERROR": f"{holomotion_metrics_mean.get('root_r_error', 0):.4f}",  # noqa: E501
                "ROOT_Pitch_ERROR": f"{holomotion_metrics_mean.get('root_p_error', 0):.4f}",  # noqa: E501
                "ROOT_Yaw_ERROR": f"{holomotion_metrics_mean.get('root_y_error', 0):.4f}",  # noqa: E501
                "ROOT_VEL_ERROR": f"{holomotion_metrics_mean.get('root_vel_error', 0):.4f}",  # noqa: E501
                "ROOT_HEIGHT_ERROR": f"{holomotion_metrics_mean.get('root_height_error', 0):.4f}",  # noqa: E501
            }
            logger.info(
                "\n"
                + tabulate(
                    [[k, v] for k, v in holomotion_metrics.items()],
                    headers=["Metric", "Value"],
                    tablefmt="simple_outline",
                )
                + "\n"
            )

        # Save global metrics to a separate file
        global_metrics = {
            "iteration": self.current_learning_iteration,
        }

        # Add holomotion metrics to global metrics
        if holomotion_metrics_mean:
            global_metrics.update(holomotion_metrics_mean)

        with open(record_path, "w+") as f:
            json.dump(global_metrics, f, indent=2)

    def offline_evaluate_policy(self, dump_npzs: bool = False):
        """Dump NPZs (no metrics) from validation cache using ref_motion command.

        - Iterates validation batches; env i -> clip i (deterministic) starting at frame 0.
        - Collect robot and reference sequences each step and save one NPZ per clip.
        - NPZ conforms to holomotion_retargeted format keys.
        - Optionally records viewport MP4(s) aligned with target_fps and rollout length.
        """
        import numpy as np
        import imageio

        ckpt_path = self.config.checkpoint
        # log_dir is already set to checkpoint directory in eval script
        model_name = os.path.basename(ckpt_path).replace(".pt", "")

        # Eval modes (freeze normalizers if enabled)
        self.actor.eval()
        self.critic.eval()
        if self.obs_norm_enabled:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

        # Require ref_motion command and simple cache backend
        command_name = list(self.env.config.commands.keys())[0]
        if command_name != "ref_motion":
            logger.warning(
                "Offline evaluation only supported for ref_motion command"
            )
            return {}
        motion_cmd = self.env._env.command_manager.get_term("ref_motion")
        cache = getattr(motion_cmd, "_motion_cache", None)
        if cache is None:
            logger.error(
                "Offline evaluation requires hdf5_simple cache backend (no LMDB support)"
            )
            return {}

        # Evaluation flag and cache batch-size adjustment (ensure batch_size == num_envs)
        motion_cmd._is_evaluating = True
        num_envs = self.env.num_envs
        try:
            if getattr(cache, "_batch_size", None) != num_envs:
                from holomotion.src.training.h5_dataloader import (
                    MotionClipBatchCache,
                )

                cache = MotionClipBatchCache(
                    train_dataset=cache._datasets["train"],
                    val_dataset=cache._datasets["val"],
                    batch_size=num_envs,
                    stage_device=getattr(cache, "_stage_device", None),
                    num_workers=getattr(cache, "_num_workers", 0),
                    prefetch_factor=getattr(cache, "_prefetch_factor", None),
                    pin_memory=getattr(cache, "_pin_memory", True),
                    persistent_workers=getattr(
                        cache, "_persistent_workers", False
                    ),
                    sampler_rank=getattr(cache, "_sampler_rank", 0),
                    sampler_world_size=getattr(
                        cache, "_sampler_world_size", 1
                    ),
                    allowed_prefixes=getattr(cache, "_allowed_prefixes", None),
                    swap_interval_steps=getattr(
                        cache, "swap_interval_steps", None
                    ),
                    force_timeout_on_swap=getattr(
                        cache, "force_timeout_on_swap", True
                    ),
                )
                motion_cmd._motion_cache = cache
        except Exception as e:
            logger.warning(
                f"Offline eval: failed to rebuild cache to batch_size={num_envs}: {e}"
            )

        # Derive HDF5 dataset base name (from validation dataset root) for output naming
        dataset_suffix = None
        val_dataset = cache._datasets["val"]
        dataset_root = str(val_dataset.hdf5_root).rstrip(os.sep)
        if dataset_root:
            dataset_suffix = os.path.basename(dataset_root)

        # Output directory (respect existing log_dir derived from checkpoint)
        suffix = f"isaaclab_eval_output_{model_name}"
        if dataset_suffix is not None:
            suffix = f"{suffix}_{dataset_suffix}"
        output_dir = os.path.join(self.log_dir, suffix)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving evaluation outputs to: {output_dir}")

        # Switch to validation cache and iterate all batches
        if hasattr(cache, "set_mode"):
            cache.set_mode("val")
        # Determine policy/video FPS from command config (align wallclock time)
        motion_fps = int(getattr(motion_cmd.cfg, "target_fps", 50))
        # Prepare video root under checkpoint directory
        videos_dir = os.path.join(self.log_dir, "videos")
        if self.record_video and self.is_main_process:
            os.makedirs(videos_dir, exist_ok=True)
        total_batches = int(getattr(cache, "num_batches", 1))
        with torch.no_grad():
            for batch_idx in tqdm(
                range(total_batches), desc="Evaluating batches"
            ):
                if batch_idx > 0:
                    cache.advance()
                # Reset envs first, then apply deterministic mapping on the active cache batch
                _ = self.env.reset_all()
                if hasattr(motion_cmd, "setup_offline_eval_deterministic"):
                    motion_cmd.setup_offline_eval_deterministic(
                        apply_pending_swap=True
                    )

                # Read current batch metadata AFTER reset + setup
                current = getattr(cache, "current_batch", None)
                if current is None or not hasattr(current, "motion_keys"):
                    logger.warning(
                        "Current cache batch missing motion_keys; skipping batch"
                    )
                    continue
                motion_keys = list(current.motion_keys)
                raw_motion_keys = list(
                    getattr(current, "raw_motion_keys", current.motion_keys)
                )

                # Determine active env count for this batch
                clip_count = int(cache.clip_count)
                active_count = min(num_envs, clip_count)

                # Recompute observations after deterministic setup
                obs_dict = self.env._env.observation_manager.compute(
                    update_history=True
                )
                obs = obs_dict["policy"].to(self.device)
                obs = (
                    self.obs_normalizer(obs) if self.obs_norm_enabled else obs
                )

                # Map env -> motion_key for active envs
                env_motion_keys = {
                    int(i): motion_keys[int(i)] for i in range(active_count)
                }
                env_raw_motion_keys = {
                    int(i): raw_motion_keys[int(i)]
                    for i in range(active_count)
                }

                # Prepare video writer per batch (reflects viewport frames)
                writer = None
                if self.record_video and self.is_main_process:

                    def _sanitize_key(key: str) -> str:
                        return (
                            key.replace("/", "+")
                            .replace(" ", "_")
                            .replace("\\", "+")
                        )

                    if active_count == 1 and 0 in env_motion_keys:
                        vid_name = f"{_sanitize_key(env_motion_keys[0])}.mp4"
                    else:
                        vid_name = f"batch_{batch_idx:04d}.mp4"
                    vid_path = os.path.join(videos_dir, vid_name)
                    writer = imageio.get_writer(
                        vid_path, fps=motion_fps, macro_block_size=None
                    )

                # Prepare per-env collectors
                env_has_done = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                episode_lengths = torch.zeros(
                    num_envs, dtype=torch.long, device=self.device
                )

                active_mask = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                if active_count > 0:
                    active_mask[:active_count] = True

                # Reference collectors (URDF order)
                ref_dof_pos = [[] for _ in range(active_count)]
                ref_dof_vel = [[] for _ in range(active_count)]
                ref_body_pos = [[] for _ in range(active_count)]
                ref_body_rot_xyzw = [[] for _ in range(active_count)]
                ref_body_vel = [[] for _ in range(active_count)]
                ref_body_ang_vel = [[] for _ in range(active_count)]

                # Robot collectors (URDF order)
                robot_dof_pos = [[] for _ in range(active_count)]
                robot_dof_vel = [[] for _ in range(active_count)]
                robot_body_pos = [[] for _ in range(active_count)]
                robot_body_rot_xyzw = [[] for _ in range(active_count)]
                robot_body_vel = [[] for _ in range(active_count)]
                robot_body_ang_vel = [[] for _ in range(active_count)]

                # Per-env bookkeeping
                clip_lengths_np = (
                    current.lengths.detach().cpu().numpy()
                    if hasattr(current, "lengths")
                    else np.array(
                        [getattr(cache, "max_frame_length", 1000)]
                        * active_count
                    )
                )
                # Persist an explicit mapping file for verification
                try:
                    mapping_records = []
                    for i in range(active_count):
                        mapping_records.append(
                            {
                                "env_id": int(i),
                                "motion_key": env_motion_keys[int(i)],
                                "raw_motion_key": env_raw_motion_keys[int(i)],
                                "clip_length": int(clip_lengths_np[int(i)]),
                            }
                        )
                    mapping_path = os.path.join(
                        output_dir, f"batch_{batch_idx:04d}_mapping.json"
                    )
                    with open(mapping_path, "w") as f:
                        json.dump(mapping_records, f, indent=2)
                except Exception:
                    pass

                env_frame_counts = [0 for _ in range(active_count)]
                encountered_done = [False for _ in range(active_count)]
                valid_masks = [[] for _ in range(active_count)]

                def _sanitize_key(key: str) -> str:
                    return (
                        key.replace("/", "+")
                        .replace(" ", "_")
                        .replace("\\", "+")
                    )

                def _save_env_npz(idx: int):
                    if idx >= active_count:
                        return
                    # Total collected frames
                    total_len = int(min(env_frame_counts[idx], max_steps))
                    if total_len <= 0:
                        return

                    # Compute contiguous valid prefix length and slice_len
                    vm = valid_masks[idx][:total_len]
                    valid_prefix_len = 0
                    for b in vm:
                        if b:
                            valid_prefix_len += 1
                        else:
                            break
                    clip_len = int(clip_lengths_np[idx])
                    slice_len = int(min(valid_prefix_len, clip_len, total_len))
                    if slice_len <= 0:
                        return

                    # Reference arrays (sliced)
                    ref_dof_pos_arr = np.stack(
                        ref_dof_pos[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    ref_dof_vel_arr = np.stack(
                        ref_dof_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    ref_body_pos_arr = np.stack(
                        ref_body_pos[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    ref_body_rot_xyzw_arr = np.stack(
                        ref_body_rot_xyzw[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    ref_body_vel_arr = np.stack(
                        ref_body_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    ref_body_ang_vel_arr = np.stack(
                        ref_body_ang_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)

                    # Robot arrays (sliced)
                    robot_dof_pos_arr = np.stack(
                        robot_dof_pos[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_dof_vel_arr = np.stack(
                        robot_dof_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_body_pos_arr = np.stack(
                        robot_body_pos[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_body_rot_xyzw_arr = np.stack(
                        robot_body_rot_xyzw[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_body_vel_arr = np.stack(
                        robot_body_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_body_ang_vel_arr = np.stack(
                        robot_body_ang_vel[idx][:slice_len], axis=0
                    ).astype(np.float32)

                    # Metadata
                    motion_fps = int(getattr(motion_cmd.cfg, "target_fps", 50))
                    num_dofs = int(ref_dof_pos_arr.shape[1])
                    num_bodies = int(ref_body_pos_arr.shape[1])
                    wallclock_len = (
                        float(slice_len - 1) / float(motion_fps)
                        if motion_fps > 0 and slice_len > 0
                        else 0.0
                    )
                    meta = {
                        "motion_key": env_motion_keys[idx],
                        "raw_motion_key": env_raw_motion_keys[idx],
                        "motion_fps": float(motion_fps),
                        "num_frames": int(slice_len),
                        "wallclock_len": float(wallclock_len),
                        "num_dofs": int(num_dofs),
                        "num_bodies": int(num_bodies),
                        "clip_length": int(clip_lengths_np[idx]),
                        "valid_prefix_len": int(valid_prefix_len),
                    }

                    # Output filename: flattened motion_key
                    out_name = f"{_sanitize_key(env_motion_keys[idx])}.npz"
                    out_path = os.path.join(output_dir, out_name)

                    np.savez_compressed(
                        out_path,
                        metadata=json.dumps(meta),
                        robot_dof_pos=robot_dof_pos_arr,
                        robot_dof_vel=robot_dof_vel_arr,
                        robot_global_translation=robot_body_pos_arr,
                        robot_global_rotation_quat=robot_body_rot_xyzw_arr,
                        robot_global_velocity=robot_body_vel_arr,
                        robot_global_angular_velocity=robot_body_ang_vel_arr,
                        ref_dof_pos=ref_dof_pos_arr,
                        ref_dof_vel=ref_dof_vel_arr,
                        ref_global_translation=ref_body_pos_arr,
                        ref_global_rotation_quat=ref_body_rot_xyzw_arr,
                        ref_global_velocity=ref_body_vel_arr,
                        ref_global_angular_velocity=ref_body_ang_vel_arr,
                    )

                max_steps = int(
                    getattr(cache, "max_frame_length", 1000)
                )  # decide the max_length to evaluate
                for rollout_step in tqdm(
                    range(max_steps), desc="Rollout steps"
                ):
                    # PRE-STEP: collect states for all active envs
                    active = [i for i in range(active_count)]
                    if len(active) > 0:
                        # Reference step tensors (URDF order)
                        ref_dp = (
                            motion_cmd.get_ref_motion_dof_pos_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ref_dv = (
                            motion_cmd.get_ref_motion_dof_vel_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ref_bp = (
                            motion_cmd.get_ref_motion_bodylink_global_pos_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ref_br = (
                            motion_cmd.get_ref_motion_bodylink_global_rot_xyzw_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ref_bv = (
                            motion_cmd.get_ref_motion_bodylink_global_lin_vel_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ref_bav = (
                            motion_cmd.get_ref_motion_bodylink_global_ang_vel_cur_urdf_order()
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        # Robot step tensors (URDF order)
                        rob_dp = (
                            motion_cmd.robot_dof_pos_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        rob_dv = (
                            motion_cmd.robot_dof_vel_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        rob_bp = (
                            motion_cmd.robot_bodylink_global_pos_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        rob_br = (
                            motion_cmd.robot_bodylink_global_rot_xyzw_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        rob_bv = (
                            motion_cmd.robot_bodylink_global_lin_vel_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        rob_bav = (
                            motion_cmd.robot_bodylink_global_ang_vel_cur_urdf_order.detach()
                            .cpu()
                            .numpy()
                        )
                        for idx in active:
                            ref_dof_pos[idx].append(ref_dp[idx])
                            ref_dof_vel[idx].append(ref_dv[idx])
                            ref_body_pos[idx].append(ref_bp[idx])
                            ref_body_rot_xyzw[idx].append(ref_br[idx])
                            ref_body_vel[idx].append(ref_bv[idx])
                            ref_body_ang_vel[idx].append(ref_bav[idx])

                            robot_dof_pos[idx].append(rob_dp[idx])
                            robot_dof_vel[idx].append(rob_dv[idx])
                            robot_body_pos[idx].append(rob_bp[idx])
                            robot_body_rot_xyzw[idx].append(rob_br[idx])
                            robot_body_vel[idx].append(rob_bv[idx])
                            robot_body_ang_vel[idx].append(rob_bav[idx])

                            # Record valid mask for current frame (before step)
                            clip_limit = int(clip_lengths_np[idx])
                            valid_now = (
                                (idx < active_count)
                                and (not encountered_done[idx])
                                and (env_frame_counts[idx] < clip_limit)
                            )
                            valid_masks[idx].append(bool(valid_now))

                            # Increment local frame counter
                            env_frame_counts[idx] += 1

                    # No mid-rollout finalize; we defer to end using valid masks

                    # Inference and step (advance sim)
                    actions, _, _, _, _ = self.actor(
                        obs, actions=None, mode="inference"
                    )
                    obs_dict, _, dones, _, infos = self.env.step(actions)
                    # Append one rendered frame per sim step (post-step) to align wallclock time
                    if writer is not None:
                        frame = self.env._env.render()
                        if isinstance(frame, np.ndarray):
                            writer.append_data(frame)
                    obs = obs_dict["policy"].to(self.device)
                    obs = (
                        self.obs_normalizer(obs)
                        if self.obs_norm_enabled
                        else obs
                    )

                    # Handle RL dones (first-done policy): mark done for future frames
                    step_dones = (
                        dones.bool().reshape(-1).detach().cpu().numpy()
                    )
                    for idx in range(min(active_count, len(step_dones))):
                        if step_dones[idx] and not encountered_done[idx]:
                            encountered_done[idx] = True

                    if rollout_step == max_steps - 1:
                        # End of rollout: save once per env with full rollout arrays + valid_mask
                        if dump_npzs:
                            for idx in range(active_count):
                                _save_env_npz(idx)
                        # Close batch video writer
                        if writer is not None:
                            writer.close()
                        break

                # No manual cache.advance(); handled by command setup on next reset
        logger.info(
            f"Offline evaluation complete: saved clips to {output_dir}"
        )
        return {"output_dir": output_dir}

    def offline_evaluate_velocity_tracking(self):
        """Roll out indefinitely for visualizing the velocity tracking policy.

        This method runs a continuous rollout without time limits, suitable for
        visualization and interactive evaluation. The policy will track velocity
        commands generated by the environment's command manager.

        Returns:
            dict: Empty dict (method runs indefinitely until interrupted)
        """
        command_name = list(self.env.config.commands.keys())[0]
        if command_name != "base_velocity":
            logger.warning(
                "Velocity tracking evaluation only supported for base_velocity command"
            )
            return {}

        if self.is_main_process:
            logger.info(
                "Starting indefinite velocity tracking rollout for visualization..."
            )
            logger.info("Press Ctrl+C to stop")

        # Set models to eval mode
        self.actor.eval()
        self.critic.eval()
        if self.obs_norm_enabled:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

        # Reset environment
        obs_dict = self.env.reset_all()[0]
        obs_raw = obs_dict["policy"].to(self.device)
        privileged_obs_raw = obs_dict["critic"].to(self.device)

        # Normalize initial observations
        if self.obs_norm_enabled:
            obs = self.obs_normalizer.normalize_only(obs_raw)
            privileged_obs = self.privileged_obs_normalizer.normalize_only(
                privileged_obs_raw
            )
            if self.obs_norm_enable_clipping:
                obs = torch.clamp(
                    obs, -self.obs_norm_clip_range, self.obs_norm_clip_range
                )
                privileged_obs = torch.clamp(
                    privileged_obs,
                    -self.obs_norm_clip_range,
                    self.obs_norm_clip_range,
                )
        else:
            obs = obs_raw
            privileged_obs = privileged_obs_raw

        step_count = 0
        with torch.no_grad():
            while True:
                # Inference: compute actions
                actions, _, _, _, _ = self.actor(
                    obs, actions=None, mode="inference"
                )

                # Step environment
                obs_dict, rewards, dones, time_outs, infos = self.env.step(
                    actions
                )
                obs_raw = obs_dict["policy"].to(self.device)
                privileged_obs_raw = obs_dict["critic"].to(self.device)

                # Normalize observations
                if self.obs_norm_enabled:
                    obs = self.obs_normalizer.normalize_only(obs_raw)
                    privileged_obs = (
                        self.privileged_obs_normalizer.normalize_only(
                            privileged_obs_raw
                        )
                    )
                    if self.obs_norm_enable_clipping:
                        obs = torch.clamp(
                            obs,
                            -self.obs_norm_clip_range,
                            self.obs_norm_clip_range,
                        )
                        privileged_obs = torch.clamp(
                            privileged_obs,
                            -self.obs_norm_clip_range,
                            self.obs_norm_clip_range,
                        )
                else:
                    obs = obs_raw
                    privileged_obs = privileged_obs_raw

                step_count += 1

                # Handle resets (when dones occur, environment will auto-reset)
                if torch.any(dones):
                    if self.is_main_process:
                        logger.debug(
                            f"Episode resets detected at step {step_count}"
                        )

        return {}

    def _log(self, locs: dict):
        """Enhanced logging function with beautiful tabulate formatting."""
        if not self.log_dir:
            return

        it = locs["it"]
        loss_dict = locs["loss_dict"]
        collection_time = locs["collection_time"]
        learn_time = locs["learn_time"]

        # Prepare metrics dictionary for Accelerate's logging
        metrics = {}

        # Episode info logging to TensorBoard
        ep_info_data = {}
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )

                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    metric_key = key if "/" in key else f"Episode/{key}"
                    metrics[metric_key] = value.item()
                    ep_info_data[metric_key] = value.item()

        # Estimate policy noise std without accessing DDP-wrapped internals
        base_actor = self.accelerator.unwrap_model(self.actor)
        if hasattr(base_actor, "std"):
            mean_std = base_actor.std.mean()
        elif hasattr(base_actor, "log_std"):
            mean_std = torch.exp(base_actor.log_std).mean()
        else:
            mean_std = torch.tensor(0.0, device=self.device)

        fps = int(
            self.num_steps_per_env
            * self.num_envs
            * self.gpu_world_size
            / (collection_time + learn_time)
        )

        # Add loss metrics
        for key, value in loss_dict.items():
            metrics[f"Loss/{key}"] = value

        metrics["Loss/actor_learning_rate"] = self.actor_learning_rate
        metrics["Loss/critic_learning_rate"] = self.critic_learning_rate
        metrics["Policy/mean_noise_std"] = mean_std.item()
        metrics["Perf/total_fps"] = fps
        metrics["Perf/collection_time"] = collection_time
        metrics["Perf/learning_time"] = learn_time

        synced_mean_reward = locs.get("synced_mean_reward", None)
        synced_mean_episode_length = locs.get(
            "synced_mean_episode_length", None
        )
        if (
            synced_mean_reward is not None
            and synced_mean_episode_length is not None
        ):
            metrics["Train/mean_reward"] = synced_mean_reward
            metrics["Train/mean_episode_length"] = synced_mean_episode_length
        elif len(self.rewbuffer) > 0:
            metrics["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            metrics["Train/mean_episode_length"] = statistics.mean(
                self.lenbuffer
            )

        # Log all metrics using Accelerate's native logging
        # Cache diagnostics for ref_motion command
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")
            metrics["Cache/swap_index"] = float(
                motion_cmd._motion_cache.swap_index
            )
        self.accelerator.log(metrics, step=it)

        # Beautiful console logging with tabulate
        self._post_epoch_logging(
            {
                "it": it,
                "total_learning_iterations": self.num_learning_iterations,
                "loss_dict": loss_dict,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "synced_mean_reward": synced_mean_reward,
                "synced_mean_episode_length": synced_mean_episode_length,
                "mean_std": mean_std.item(),
                "fps": fps,
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
            }
        )

    def _post_epoch_logging(self, log_dict):
        """Beautiful console logging with tabulate formatting."""
        # Episode info processing
        ep_metrics = {}
        if log_dict["ep_infos"]:
            for key in log_dict["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )

                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    if "/" in key:
                        ep_metrics[key] = f"{value:.4f}"
                    else:
                        ep_metrics[f"Mean Episode {key}"] = f"{value:.4f}"

        # Build training data dictionary
        training_data = {
            "Learning Iteration": f"{log_dict['it']}/{log_dict['total_learning_iterations']}",
            "FPS": f"{log_dict['fps']:.0f} steps/s",
            "Collection Time": f"{log_dict['collection_time']:.3f}s",
            "Learning Time": f"{log_dict['learn_time']:.3f}s",
            "Mean Action Noise Std": f"{log_dict['mean_std']:.2f}",
            "Actor Learning Rate": f"{log_dict['actor_learning_rate']:.4e}",
            "Critic Learning Rate": f"{log_dict['critic_learning_rate']:.4e}",
        }

        # Add reward and episode info (synced across ranks if available)
        smr = log_dict.get("synced_mean_reward", None)
        smel = log_dict.get("synced_mean_episode_length", None)
        if smr is not None and smel is not None:
            training_data["Mean Episode Reward"] = f"{smr:.2f}"
            training_data["Mean Episode Length"] = f"{smel:.2f}"
        elif len(log_dict["rewbuffer"]) > 0:
            training_data["Mean Episode Reward"] = (
                f"{statistics.mean(log_dict['rewbuffer']):.2f}"
            )
            training_data["Mean Episode Length"] = (
                f"{statistics.mean(log_dict['lenbuffer']):.2f}"
            )

        # Add loss data
        training_data.update(
            {
                k: f"{v:.4f}" if isinstance(v, (int, float)) else f"{v:.4f}"
                for k, v in log_dict["loss_dict"].items()
                if v is not None
            }
        )

        # Add episode metrics
        training_data.update(ep_metrics)

        # Organize and display
        table_data = self._organize_training_data(training_data)
        log_lines = [
            "\n" + "=" * 80,
            f"TRAINING LOG - Iteration {log_dict['it']}/{log_dict['total_learning_iterations']}",
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

    def _organize_training_data(self, training_data):
        """Organize training data into logical groups for better console display."""
        # Define priority order for key display
        priority_keys = [
            # Core training info (highest priority)
            "Learning Iteration",
            "FPS",
            "Collection Time",
            "Learning Time",
            "",  # separator
            # Episode statistics
            "Mean Episode Reward",
            "Mean Episode Length",
            "",  # separator
            # Model metrics
            "Mean Action Noise Std",
            "Actor Learning Rate",
            "Critic Learning Rate",
            "",  # separator
        ]

        # Create organized list
        organized_data = []
        used_keys = set()

        # Helper function to add section header
        def add_section_header(title):
            organized_data.append([f"=== {title.upper()} ===", "======"])

        # Add priority keys first
        current_section = None
        for key in priority_keys:
            if key == "":  # section break
                current_section = None
            elif key in training_data:
                # Add section header for performance metrics
                if current_section != "training" and key in [
                    "Learning Iteration",
                    "FPS",
                    "Collection Time",
                    "Learning Time",
                ]:
                    add_section_header("Performance")
                    current_section = "training"
                # Add section header for episode stats
                elif current_section != "episode" and key in [
                    "Mean Episode Reward",
                    "Mean Episode Length",
                ]:
                    add_section_header("Episode Statistics")
                    current_section = "episode"
                # Add section header for model metrics
                elif current_section != "model" and key in [
                    "Mean Action Noise Std",
                    "Actor Learning Rate",
                    "Critic Learning Rate",
                ]:
                    add_section_header("Model")
                    current_section = "model"

                organized_data.append([key, training_data[key]])
                used_keys.add(key)

        loss_keys = sorted(
            [
                k
                for k in training_data.keys()
                if k in ["value_function", "surrogate", "entropy"]
                and k not in used_keys
            ]
        )
        if loss_keys:
            add_section_header("Loss")
            for key in loss_keys:
                display_key = f"Loss/{key}"
                organized_data.append([display_key, training_data[key]])
                used_keys.add(key)

        remaining_keys = sorted(
            [k for k in training_data.keys() if k not in used_keys]
        )
        if remaining_keys:
            add_section_header("Other Metrics")
            for key in remaining_keys:
                organized_data.append([key, training_data[key]])

        return organized_data

    @staticmethod
    def _clean_state_dict(state_dict):
        """Remove '_orig_mod.' prefix from torch.compile wrapped models.

        Args:
            state_dict: State dict that may contain '_orig_mod.' prefixed keys

        Returns:
            Cleaned state dict with prefixes removed
        """
        cleaned_dict = {}
        prefix = "_orig_mod."
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            new_k = k[prefix_len:] if k.startswith(prefix) else k
            cleaned_dict[new_k] = v
        return cleaned_dict

    def _load_model_state(self, model, state_dict, *, strict: bool = True):
        """Load a state dict into a (possibly compiled) model safely.

        - Always unwrap Accelerate wrappers first.
        - If the model is a compiled OptimizedModule (has ``_orig_mod``),
          load into the original module and strip any ``_orig_mod.`` prefixes
          from the incoming state dict for robustness.
        """
        target = self.accelerator.unwrap_model(model)
        cleaned = self._clean_state_dict(state_dict)
        if hasattr(target, "_orig_mod"):
            target._orig_mod.load_state_dict(cleaned, strict=strict)
        else:
            target.load_state_dict(cleaned, strict=strict)

    def load(self, ckpt_path):
        """Load checkpoint using Accelerate's built-in methods when available.

        Supports both Accelerate format (actor/ and critic/ directories) and
        legacy formats for backward compatibility.
        """
        if ckpt_path is not None:
            if self.is_main_process:
                logger.info(f"Loading checkpoint from {ckpt_path}")

            base_path = ckpt_path.replace(".pt", "")
            actor_model_path = os.path.join(base_path, "actor")
            critic_model_path = os.path.join(base_path, "critic")

            # Check if this is an Accelerate-saved checkpoint (new format)
            is_accelerate_format = os.path.exists(
                actor_model_path
            ) and os.path.exists(critic_model_path)

            if is_accelerate_format:
                # Accelerate format checkpoint detected
                # accelerator.save_model() saves state dicts, so we load them manually
                # Find the actual model file (Accelerate may save as
                # pytorch_model.bin or model.safetensors)
                actor_files = [
                    f
                    for f in os.listdir(actor_model_path)
                    if f.endswith((".bin", ".safetensors"))
                ]
                critic_files = [
                    f
                    for f in os.listdir(critic_model_path)
                    if f.endswith((".bin", ".safetensors"))
                ]

                if not actor_files or not critic_files:
                    raise FileNotFoundError(
                        f"Model files not found in Accelerate checkpoint format. "
                        f"Actor dir: {actor_model_path}, "
                        f"Critic dir: {critic_model_path}"
                    )

                # Prefer pytorch_model.bin, fallback to first file
                actor_file = next(
                    (f for f in actor_files if f == "pytorch_model.bin"),
                    actor_files[0],
                )
                critic_file = next(
                    (f for f in critic_files if f == "pytorch_model.bin"),
                    critic_files[0],
                )

                # Load model state dicts from Accelerate format
                actor_file_path = os.path.join(actor_model_path, actor_file)
                critic_file_path = os.path.join(critic_model_path, critic_file)

                if actor_file.endswith(".safetensors") or critic_file.endswith(
                    ".safetensors"
                ):
                    try:
                        from safetensors import safe_open

                        actor_state = {}
                        critic_state = {}

                        if actor_file.endswith(".safetensors"):
                            with safe_open(
                                actor_file_path,
                                framework="pt",
                                device=str(self.device),
                            ) as f:
                                for key in f.keys():
                                    actor_state[key] = f.get_tensor(key)
                        else:
                            actor_state = torch.load(
                                actor_file_path, map_location=self.device
                            )

                        if critic_file.endswith(".safetensors"):
                            with safe_open(
                                critic_file_path,
                                framework="pt",
                                device=str(self.device),
                            ) as f:
                                for key in f.keys():
                                    critic_state[key] = f.get_tensor(key)
                        else:
                            critic_state = torch.load(
                                critic_file_path, map_location=self.device
                            )
                    except ImportError:
                        raise ImportError(
                            "safetensors library required to load "
                            ".safetensors files. Install with: pip install safetensors"
                        )
                else:
                    actor_state = torch.load(
                        actor_file_path, map_location=self.device
                    )
                    critic_state = torch.load(
                        critic_file_path, map_location=self.device
                    )

                # Load state dicts with proper unwrapping/compile handling
                self._load_model_state(self.actor, actor_state, strict=True)
                self._load_model_state(self.critic, critic_state, strict=True)

                # Load custom state (optimizers, normalizers, etc.)
                loaded_dict = torch.load(ckpt_path, map_location=self.device)

                if "actor_optimizer_state_dict" in loaded_dict:
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["actor_optimizer_state_dict"]
                    )
                if "critic_optimizer_state_dict" in loaded_dict:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )

                if self.obs_norm_enabled:
                    if "obs_norm_state_dict" in loaded_dict and hasattr(
                        self, "obs_normalizer"
                    ):
                        self.obs_normalizer.load_state_dict(
                            loaded_dict["obs_norm_state_dict"],
                            strict=False,
                        )
                    if (
                        "privileged_obs_norm_state_dict" in loaded_dict
                        and hasattr(self, "privileged_obs_normalizer")
                    ):
                        self.privileged_obs_normalizer.load_state_dict(
                            loaded_dict["privileged_obs_norm_state_dict"],
                            strict=False,
                        )

                self.current_learning_iteration = loaded_dict.get("iter", 0)
                return loaded_dict.get("infos", None)

            # Fallback to old format (backward compatibility)
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            # Handle both old and new checkpoint formats
            if "actor_model_state_dict" in loaded_dict:
                # Separate actor/critic format (preferred)
                actor_state = self._clean_state_dict(
                    loaded_dict["actor_model_state_dict"]
                )
                critic_state = self._clean_state_dict(
                    loaded_dict["critic_model_state_dict"]
                )

                # Load state dicts with unwrapping/compile handling
                self._load_model_state(self.actor, actor_state, strict=True)
                self._load_model_state(self.critic, critic_state, strict=True)

                # Load optimizers (new format with separate optimizers)
                if "actor_optimizer_state_dict" in loaded_dict:
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["actor_optimizer_state_dict"]
                    )
                if "critic_optimizer_state_dict" in loaded_dict:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )
                # Backward compatibility: single optimizer
                if (
                    "optimizer_state_dict" in loaded_dict
                    and "actor_optimizer_state_dict" not in loaded_dict
                ):
                    logger.warning(
                        "Loading from old checkpoint format with combined optimizer. "
                        "Only actor optimizer state will be loaded."
                    )
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["optimizer_state_dict"]
                    )

                self.current_learning_iteration = loaded_dict.get("iter", 0)
            elif "model_state_dict" in loaded_dict:
                # rsl_rl format (single policy)
                cleaned_state_dict = self._clean_state_dict(
                    loaded_dict["model_state_dict"]
                )

                # Split into actor and critic parts
                actor_state = {}
                critic_state = {}
                for key, value in cleaned_state_dict.items():
                    if key.startswith("actor."):
                        actor_state[key[6:]] = value
                    elif key.startswith("critic."):
                        critic_state[key[7:]] = value

                # Load state dicts with unwrapping/compile handling
                if actor_state:
                    self._load_model_state(
                        self.actor, actor_state, strict=False
                    )
                if critic_state:
                    self._load_model_state(
                        self.critic, critic_state, strict=False
                    )

                # Load optimizer - try separate optimizers first, then combined
                if "actor_optimizer_state_dict" in loaded_dict:
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["actor_optimizer_state_dict"]
                    )
                if "critic_optimizer_state_dict" in loaded_dict:
                    self.critic_optimizer.load_state_dict(
                        loaded_dict["critic_optimizer_state_dict"]
                    )
                if (
                    "optimizer_state_dict" in loaded_dict
                    and "actor_optimizer_state_dict" not in loaded_dict
                ):
                    logger.warning(
                        "Loading from old checkpoint format with combined optimizer. "
                        "Only actor optimizer state will be loaded."
                    )
                    self.actor_optimizer.load_state_dict(
                        loaded_dict["optimizer_state_dict"]
                    )

                self.current_learning_iteration = loaded_dict.get("iter", 0)

            # Load normalizers if present
            if self.obs_norm_enabled:
                if "obs_norm_state_dict" in loaded_dict and hasattr(
                    self, "obs_normalizer"
                ):
                    self.obs_normalizer.load_state_dict(
                        loaded_dict["obs_norm_state_dict"],
                        strict=False,
                    )
                if "privileged_obs_norm_state_dict" in loaded_dict and hasattr(
                    self, "privileged_obs_normalizer"
                ):
                    self.privileged_obs_normalizer.load_state_dict(
                        loaded_dict["privileged_obs_norm_state_dict"],
                        strict=False,
                    )

            return loaded_dict.get("infos", None)

    def save(self, path, infos=None):
        """Save checkpoint using Accelerate's built-in methods when available."""
        if not self.is_main_process:
            return

        logger.info(f"Saving checkpoint to {path}")

        # Always use Accelerate's save_model() which handles unwrapping and compilation
        # Save models separately with Accelerate (handles compilation automatically)
        base_path = path.replace(".pt", "")
        os.makedirs(
            os.path.dirname(base_path) if os.path.dirname(base_path) else ".",
            exist_ok=True,
        )

        self.accelerator.save_model(
            self.actor, os.path.join(base_path, "actor")
        )
        self.accelerator.save_model(
            self.critic, os.path.join(base_path, "critic")
        )

        # Save optimizers and custom state separately
        custom_state = {
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        if self.obs_norm_enabled and hasattr(self, "obs_normalizer"):
            custom_state["obs_norm_state_dict"] = (
                self.obs_normalizer.state_dict()
            )
        if self.obs_norm_enabled and hasattr(
            self, "privileged_obs_normalizer"
        ):
            custom_state["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )

        torch.save(custom_state, path)

    @property
    def inference_model(self):
        """Return the separate actor and critic for inference."""
        return {
            "actor": self.actor,
            "critic": self.critic,
        }

    def synchronize_normalizers(self):
        """Synchronize observation normalizers across all processes."""
        any_synced = False
        if self.obs_norm_enabled:
            self.obs_normalizer.sync_stats_across_processes(self.accelerator)
            self.privileged_obs_normalizer.sync_stats_across_processes(
                self.accelerator
            )
            any_synced = True
        if any_synced:
            # Ensure all ranks have synced before proceeding
            self.accelerator.wait_for_everyone()
