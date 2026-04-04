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

import os
import random
import statistics
import sys
import time
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (
    ProjectConfiguration,
    TorchDynamoPlugin,
    load_checkpoint_in_model,
    load_state_dict,
)
from hydra.utils import get_class
from loguru import logger
from tensordict import TensorDict

from holomotion.src.algo.algo_utils import AlgoLogger


class BaseOnpolicyRL:
    """Base class for on-policy RL algorithms in HoloMotion."""

    def __init__(
        self,
        env_config,
        config,
        log_dir=None,
        headless: bool = True,
        is_offline_eval: bool = False,
    ) -> None:
        self.config = config
        self.env_config = env_config
        self.log_dir = log_dir
        self.headless = headless
        self.is_offline_eval = is_offline_eval

        self._setup_accelerator()
        self.algo_logger = AlgoLogger(
            self.accelerator,
            self.log_dir,
            is_main_process=self.is_main_process,
        )
        self._setup_environment()
        self._setup_configs()
        self._setup_seeding()
        self._setup_data_buffers()
        self._setup_algo_components()
        self._setup_models_and_optimizer()

    def _setup_accelerator(self) -> None:
        if not self.is_offline_eval:
            os.makedirs(self.log_dir, exist_ok=True)

        accelerator_kwargs = {}
        mixed_precision = self.config.get("mixed_precision", None)
        if mixed_precision in ("fp16", "bf16"):
            accelerator_kwargs["mixed_precision"] = mixed_precision
        dynamo_backend = self.config.get("dynamo_backend", None)
        if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1":
            dynamo_backend = None
        if dynamo_backend in ("inductor", "aot_eager", "cudagraphs"):
            dynamo_dynamic = bool(self.config.get("dynamo_dynamic", True))
            dynamo_fullgraph = bool(self.config.get("dynamo_fullgraph", False))
            dynamo_mode = self.config.get("dynamo_mode", "default")
            accelerator_kwargs["dynamo_plugin"] = TorchDynamoPlugin(
                backend=str(dynamo_backend),
                mode=str(dynamo_mode),
                fullgraph=bool(dynamo_fullgraph),
                dynamic=bool(dynamo_dynamic),
            )

        accelerator_kwargs["log_with"] = "tensorboard"
        project_config = ProjectConfiguration(
            project_dir=self.log_dir,
            logging_dir=self.log_dir,
        )
        accelerator_kwargs["project_config"] = project_config
        self.accelerator = Accelerator(**accelerator_kwargs)
        self.local_rank = getattr(
            self.accelerator, "local_process_index", None
        )
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.device = self.accelerator.device
        if torch.cuda.is_available() and self.device.type == "cuda":
            dev_index = self.device.index
            if dev_index is None:
                dev_index = int(self.local_rank)
                self.device = torch.device("cuda", dev_index)
            else:
                dev_index = int(dev_index)
            torch.cuda.set_device(dev_index)
        self.is_main_process = self.accelerator.is_main_process

        self.accelerator.init_trackers(
            project_name="holomotion",
            config={
                "precision": mixed_precision if mixed_precision else "fp32",
                "dynamo_backend": dynamo_backend if dynamo_backend else "none",
                "dynamo_dynamic": bool(self.config.get("dynamo_dynamic", True))
                if dynamo_backend
                else False,
            },
        )
        self._release_cuda_cache()

        logger.remove()
        log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        if self.log_dir:
            rank_log_file_name = (
                "offline_eval_rank" if self.is_offline_eval else "run_rank"
            )
            logger.add(
                os.path.join(
                    self.log_dir,
                    f"{rank_log_file_name}_{int(self.accelerator.process_index):04d}.log",
                ),
                level=log_level,
                colorize=False,
            )
        if self.is_main_process:
            logger.add(
                sys.stdout,
                level=log_level,
                colorize=True,
            )
            log_file_name = (
                "offline_eval.log" if self.is_offline_eval else "run.log"
            )
            logger.add(
                os.path.join(self.log_dir, log_file_name),
                level=log_level,
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
        env_rank = os.environ.get("RANK", "unset")
        env_local_rank = os.environ.get("LOCAL_RANK", "unset")
        env_world_size = os.environ.get("WORLD_SIZE", "unset")
        env_local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "unset")
        env_node_rank = os.environ.get(
            "NODE_RANK", os.environ.get("MACHINE_RANK", "unset")
        )
        env_master_addr = os.environ.get("MASTER_ADDR", "unset")
        env_master_port = os.environ.get("MASTER_PORT", "unset")
        env_cuda_visible_devices = os.environ.get(
            "CUDA_VISIBLE_DEVICES", "unset"
        )
        cuda_device_count = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        logger.info(
            "[Accelerate setup] "
            f"distributed_type={self.accelerator.distributed_type}, "
            f"num_processes={int(self.accelerator.num_processes)}, "
            f"process_index={int(self.accelerator.process_index)}, "
            f"local_process_index={int(self.local_rank)}, "
            f"is_main_process={bool(self.accelerator.is_main_process)}"
        )
        logger.info(
            "[Accelerate env] "
            f"RANK={env_rank}, LOCAL_RANK={env_local_rank}, "
            f"WORLD_SIZE={env_world_size}, "
            f"LOCAL_WORLD_SIZE={env_local_world_size}, "
            f"NODE_RANK={env_node_rank}, MASTER_ADDR={env_master_addr}, "
            f"MASTER_PORT={env_master_port}"
        )
        logger.info(
            "[Accelerate cuda] "
            f"CUDA_VISIBLE_DEVICES={env_cuda_visible_devices}, "
            f"torch_cuda_device_count={cuda_device_count}, "
            f"selected_device={self.device}"
        )

    def _setup_environment(self) -> None:
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
            local_rank = int(self.local_rank)
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
            "video": _record_video,
            "device": device_str,
            "kit_args": kit_args_str,
        }

        self._sim_app_launcher = AppLauncher(**app_launcher_flags)
        self._sim_app = self._sim_app_launcher.app

        logger.info(
            f"AppLauncher initialized with flags: {app_launcher_flags}"
        )

        env_class = get_class(self.env_config._target_)

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

        _ = self.env.reset_all()

        logger.info(f"Environment initialized with render_mode: {render_mode}")

    def _setup_configs(self) -> None:
        self.num_envs: int = self.env.config.num_envs
        self.num_privileged_obs = 0
        self.num_actions = self.env.config.robot.actions_dim

        self.command_name = list(self.env.config.commands.keys())[0]
        self.command_term = self.env._env.command_manager.get_term(
            self.command_name
        )
        if self.command_name == "ref_motion":
            self.command_term.set_runtime_distributed_context(
                process_id=int(self.accelerator.process_index),
                num_processes=int(self.accelerator.num_processes),
            )
            self.command_term.setup_dumping_dir(self.log_dir)

        self.save_interval = self.config.save_interval
        self.log_interval = self.config.log_interval
        self.num_steps_per_env = self.config.num_steps_per_env
        self.num_learning_iterations = self.config.num_learning_iterations
        self.total_learning_iterations = int(self.num_learning_iterations)

    def _setup_seeding(self) -> None:
        if self.command_name == "ref_motion":
            self.seed = int(self.command_term.cfg.seed)
            self.base_seed = int(self.seed - int(self.process_rank))
        else:
            self.base_seed = int(self.config.get("seed", int(time.time())))
            self.seed = int(self.base_seed + int(self.process_rank))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        self.env.seed(self.seed)
        if self.command_name == "ref_motion":
            self.command_term.set_motion_cache_seed(
                self.seed, reinitialize=False
            )

    def _setup_data_buffers(self) -> None:
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

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

        self.storage = None
        self.transition_td = None
        self._last_rollout_dones = None
        self._last_rollout_actions = None

    def _setup_algo_components(self) -> None:
        """Hook for algorithm-specific components (AMP, DAgger, PULSE)."""
        return

    def _setup_models_and_optimizer(self) -> None:
        raise NotImplementedError(
            "Subclasses must implement _setup_models_and_optimizer."
        )

    def _build_storage(self, obs_td: TensorDict) -> Any:
        """Hook for custom RolloutStorage. Override for specialized storage; default no-op."""
        return None

    def _post_env_step_hook(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        infos: Dict[str, Any],
    ) -> None:
        """Hook after each env step for auxiliary data collection."""
        if self.command_name != "ref_motion":
            return
        motion_term = self.env._env.command_manager.get_term("ref_motion")
        if motion_term is None:
            return
        motion_term.update_curriculum_reward_accumulators(rewards)

    def _post_update_hook(self, loss_dict: Dict[str, Any]) -> None:
        """Hook after each PPO update for auxiliary losses or logging."""
        return

    def _extra_checkpoint_state(self) -> Dict[str, Any]:
        """Additional state to save in checkpoints."""
        return {}

    def _load_extra_checkpoint_state(
        self, loaded_dict: Dict[str, Any]
    ) -> None:
        """Load additional checkpoint state if present."""
        return

    def _build_transition(
        self,
        obs_td: TensorDict,
        actor_out: TensorDict,
        critic_out: TensorDict,
    ):
        raise NotImplementedError(
            "Subclasses must implement _build_transition."
        )

    def _post_iteration_hook(self, it: int) -> None:
        return

    def _post_training_hook(self) -> None:
        return

    def _release_cuda_cache(self) -> None:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_additional_log_metrics(self) -> Dict[str, Any]:
        return {}

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def _ensure_storage(self, obs_td: TensorDict) -> None:
        if self.storage is not None:
            return
        self.storage = self._build_storage(obs_td)
        if self.storage is None:
            raise RuntimeError(
                "Storage is not initialized. Override _build_storage() or initialize self.storage in subclass."
            )

    def _reset_rollout_forward_state(self) -> None:
        """Hook for algorithm-specific rollout state reset."""
        return

    def _rollout_forward(
        self,
        obs_td: TensorDict,
        *,
        actor_mode: str = "sampling",
        collect_transition: bool = True,
        track_episode_stats: bool = True,
    ) -> TensorDict:
        update_obs_norm = not self.is_offline_eval
        with self.accelerator.autocast():
            actor_out: TensorDict = self.actor(
                obs_td,
                actions=None,
                mode=actor_mode,
                update_obs_norm=update_obs_norm,
            )
            critic_out: TensorDict | None = None
            if collect_transition:
                critic_out = self.critic(
                    obs_td, update_obs_norm=update_obs_norm
                )

        if collect_transition:
            self.transition_td = self._build_transition(
                obs_td,
                actor_out,
                critic_out,
            )

        actions = actor_out.get("actions")
        self._last_rollout_actions = actions
        obs_dict, rewards, dones, time_outs, infos = self.env.step(actions)

        next_obs_td = self._wrap_obs_dict(obs_dict)
        dones = dones.to(self.device)
        self._last_rollout_dones = dones

        if collect_transition:
            rewards = rewards.to(self.device)
            time_outs = time_outs.to(self.device)
            self.process_env_step(rewards, dones, time_outs, infos)

        if track_episode_stats:
            rewards_for_stats = rewards.to(self.device)
            self._track_episode_stats(rewards_for_stats, dones, infos)
        return next_obs_td

    def _track_episode_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: Dict[str, Any],
    ) -> None:
        log_info = infos.get("log")
        if self.is_main_process and isinstance(log_info, dict):
            cpu_log_info: Dict[str, torch.Tensor] = {}
            for key, value in log_info.items():
                cpu_value = self._log_value_to_cpu_tensor(value)
                if cpu_value is not None and cpu_value.numel() > 0:
                    cpu_log_info[key] = cpu_value
            if len(cpu_log_info) > 0:
                self.ep_infos.append(cpu_log_info)
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        done_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(
            self.cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist()
        )
        self.lenbuffer.extend(
            self.cur_episode_length[done_ids][:, 0].cpu().numpy().tolist()
        )
        self.cur_reward_sum[done_ids] = 0
        self.cur_episode_length[done_ids] = 0

    def _compute_returns(self, obs_td: TensorDict) -> None:
        update_obs_norm = not self.is_offline_eval
        with self.accelerator.autocast():
            last_values = (
                self.critic(obs_td, update_obs_norm=update_obs_norm)
                .get("values")
                .detach()
            )
            self.storage.compute_returns(
                last_values,
                self.gamma,
                self.lam,
                normalize_advantage=False,
            )

        if getattr(self, "global_advantage_norm", False):
            accelerator = self.accelerator if self.is_distributed else None
            self.storage.normalize_advantages_global_by_command(
                command_name=self.command_name,
                accelerator=accelerator,
                eps=1.0e-8,
            )

    def rollout_policy(self, obs_td: TensorDict) -> TensorDict:
        """Collect one rollout with current policy and compute returns."""
        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            self._reset_rollout_forward_state()
            for _ in range(self.num_steps_per_env):
                obs_td = self._rollout_forward(obs_td)
            self._compute_returns(obs_td)
        if actor_was_training:
            self.actor.train()
        if critic_was_training:
            self.critic.train()
        return obs_td

    def learn(self):
        """Main learning loop with runner logic shared across on-policy algorithms."""
        obs_dict = self.env.reset_all()[0]
        obs_td = self._wrap_obs_dict(obs_dict)
        self._ensure_storage(obs_td)
        self.train_mode()

        start_it = self.current_learning_iteration
        total_it = start_it + int(self.num_learning_iterations)
        self.total_learning_iterations = total_it

        self.accelerator.wait_for_everyone()
        if self.is_main_process:
            logger.info(
                f"Starting training for {self.num_learning_iterations} iterations "
                f"from iteration {self.current_learning_iteration}"
            )

        for it in range(start_it, total_it):
            self.current_learning_iteration = it
            start = time.time()
            obs_td = self.rollout_policy(obs_td)

            stop = time.time()
            collection_time = stop - start
            start = stop

            loss_dict = self.update()

            stop = time.time()
            learn_time = stop - start

            if self.is_main_process and it % self.log_interval == 0:
                self._log_iteration(
                    it=it,
                    loss_dict=loss_dict,
                    collection_time=collection_time,
                    learn_time=learn_time,
                )

            if self.is_main_process and it % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
                self._release_cuda_cache()

            self._post_iteration_hook(it)
            self.ep_infos.clear()
            self.accelerator.wait_for_everyone()

        final_checkpoint_path = os.path.join(
            self.log_dir, f"model_{self.current_learning_iteration}.pt"
        )
        if self.is_main_process:
            self.save(final_checkpoint_path)
            self._release_cuda_cache()

        self._post_training_hook()

        if self.log_dir:
            self.accelerator.wait_for_everyone()
            self.accelerator.end_training()
            if self.is_main_process:
                logger.info(
                    f"Training completed. Model saved to {self.log_dir}"
                )

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        infos: Dict[str, Any],
    ) -> None:
        """Process env step results and append to storage.

        Args:
            rewards: [N, 1] rewards (env step output).
            dones: [N, 1] done flags (env step output).
            time_outs: [N] time out flags (env step output).
            infos: Environment info dictionary.
        """
        raw_rewards = rewards.clone().view(-1, 1)
        rewards = raw_rewards.clone()
        dones = dones.view(-1, 1)

        # Bootstrapping on time outs
        rewards += self.gamma * (
            self.transition_td.values * time_outs[:, None]
        )
        self.transition_td.rewards = rewards
        self.transition_td.dones = dones.to(dtype=torch.bool)

        self.storage.add(self.transition_td)
        self._post_env_step_hook(raw_rewards, dones, time_outs, infos)

        self.transition_td = None

    def _wrap_obs_dict(self, obs_dict: dict) -> TensorDict:
        """Wrap env obs dict into a native nested TensorDict on device."""
        return TensorDict.from_dict(
            obs_dict,
            batch_size=[self.env.num_envs],
            device=self.device,
        )

    @staticmethod
    def _clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
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

    def _resolve_model_file_path(self, ckpt_path: str, model_name: str) -> str:
        """Resolve per-model Accelerate checkpoint directory from *.pt path."""
        base_path = ckpt_path.replace(".pt", "")
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Missing accelerate checkpoint directory for {model_name}: "
                f"{model_path}"
            )
        return model_path

    def _load_accelerate_model(
        self, model, model_path: str, *, strict: bool = True
    ) -> None:
        """Load model params from Accelerate checkpoint directory/file."""
        checkpoint_path = model_path
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            pytorch_bin_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.isfile(safetensors_path):
                checkpoint_path = safetensors_path
            elif os.path.isfile(pytorch_bin_path):
                checkpoint_path = pytorch_bin_path
            else:
                target = self.accelerator.unwrap_model(model)
                load_checkpoint_in_model(target, model_path, strict=strict)
                return
        state_dict = load_state_dict(checkpoint_path)
        self._load_model_state(model, state_dict, strict=strict)

    def _aggregate_episode_log_metrics(
        self,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if len(self.ep_infos) == 0:
            return metrics

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        for ep_info in self.ep_infos:
            for key, value in ep_info.items():
                cpu_value = self._log_value_to_cpu_tensor(value)
                if cpu_value is None or cpu_value.numel() == 0:
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(
                    cpu_value.sum().item()
                )
                metric_counts[key] = metric_counts.get(key, 0) + int(
                    cpu_value.numel()
                )

        for key, total in metric_sums.items():
            count = metric_counts.get(key, 0)
            if count <= 0:
                continue
            mean_value = total / float(count)
            metric_key = key if "/" in key else f"Episode/{key}"
            metrics[metric_key] = mean_value

        return metrics

    @staticmethod
    def _log_value_to_cpu_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            return tensor.to(device="cpu", dtype=torch.float32).reshape(-1)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if isinstance(value, (int, float)):
            return torch.tensor([float(value)], dtype=torch.float32)
        return None

    def _log_iteration(
        self,
        *,
        it: int,
        loss_dict: Dict[str, Any],
        collection_time: float,
        learn_time: float,
        synced_mean_reward: float | None = None,
        synced_mean_episode_length: float | None = None,
    ) -> None:
        if not self.log_dir:
            return

        world_size = max(1, int(self.gpu_world_size))
        fps = int(
            self.num_steps_per_env
            * self.num_envs
            * world_size
            / max(collection_time + learn_time, 1.0e-8)
        )
        total_learning_iterations = int(
            getattr(
                self,
                "total_learning_iterations",
                self.current_learning_iteration
                + int(self.num_learning_iterations),
            )
        )

        iteration_metrics: Dict[str, Any] = {
            "0-Train/iteration": int(it),
            "0-Train/iterations_total": total_learning_iterations,
        }

        for key, value in loss_dict.items():
            if value is None:
                continue
            scalar = float(value)
            iteration_metrics[f"Loss/{key}"] = scalar

        iteration_metrics.update(
            {
                "1-Perf/total_fps": float(fps),
                "1-Perf/collection_time": float(collection_time),
                "1-Perf/learning_time": float(learn_time),
            }
        )

        if (
            synced_mean_reward is not None
            and synced_mean_episode_length is not None
        ):
            iteration_metrics["0-Train/mean_reward"] = float(
                synced_mean_reward
            )
            iteration_metrics["0-Train/mean_episode_length"] = float(
                synced_mean_episode_length
            )
        elif len(self.rewbuffer) > 0:
            mean_reward = float(statistics.mean(self.rewbuffer))
            mean_episode_length = float(statistics.mean(self.lenbuffer))
            iteration_metrics["0-Train/mean_reward"] = mean_reward
            iteration_metrics["0-Train/mean_episode_length"] = (
                mean_episode_length
            )

        iteration_metrics.update(self._aggregate_episode_log_metrics())
        iteration_metrics.update(self._get_additional_log_metrics())

        self.algo_logger.log_iteration(
            step=it,
            total_learning_iterations=total_learning_iterations,
            metrics=iteration_metrics,
        )

    def load(self, ckpt_path):
        raise NotImplementedError("Subclasses must implement load().")

    def save(self, path, infos=None):
        raise NotImplementedError("Subclasses must implement save().")
