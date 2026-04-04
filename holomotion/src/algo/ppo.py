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

import inspect
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import imageio
from omegaconf import OmegaConf

from holomotion.src.algo.algo_base import BaseOnpolicyRL
from holomotion.src.algo.algo_utils import (
    PpoTransition,
    PpoVelocityTransition,
    RolloutStorage,
)
from holomotion.src.utils.onnx_export import (
    export_policy_to_onnx as export_policy_to_onnx_common,
)
from tensordict import TensorDict


def _checkpoint_state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _checkpoint_state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_checkpoint_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_state_to_cpu(v) for v in value)
    return value


class PPO(BaseOnpolicyRL):
    def _setup_configs(self):
        super()._setup_configs()
        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.get(
            "actor_learning_rate", self.config.get("learning_rate", 3e-4)
        )
        self.critic_learning_rate = self.config.get(
            "critic_learning_rate", self.config.get("learning_rate", 3e-4)
        )
        self.base_actor_learning_rate = float(self.actor_learning_rate)
        self.base_critic_learning_rate = float(self.critic_learning_rate)
        self.actor_beta1 = self.config.get("actor_beta1", 0.9)
        self.actor_beta2 = self.config.get("actor_beta2", 0.999)
        self.critic_beta1 = self.config.get("critic_beta1", 0.9)
        self.critic_beta2 = self.config.get("critic_beta2", 0.999)
        self.optimizer_type = self.config.optimizer_type
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = int(self.config.num_learning_epochs)
        self.configured_num_mini_batches = int(self.config.num_mini_batches)
        if self.configured_num_mini_batches < 1:
            raise ValueError("num_mini_batches must be >= 1.")
        distributed_update_cfg = self.config.get("distributed_update", {})
        self.distributed_update_mode = str(
            distributed_update_cfg.get("mode", "legacy")
        ).lower()
        if self.distributed_update_mode not in {"legacy", "scalable"}:
            raise ValueError(
                "distributed_update.mode must be one of "
                "{'legacy', 'scalable'}."
            )
        self.requested_num_mini_batches = self._resolve_num_mini_batches(
            self.configured_num_mini_batches
        )
        self.num_mini_batches = self.requested_num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.initial_entropy_coef = float(self.config.entropy_coef)
        self.anneal_entropy = bool(self.config.get("anneal_entropy", False))
        self.zero_entropy_point = float(
            self.config.get("zero_entropy_point", 1.0)
        )
        self._validate_entropy_schedule_config(
            initial_entropy_coef=self.initial_entropy_coef,
            anneal_entropy=self.anneal_entropy,
            zero_entropy_point=self.zero_entropy_point,
        )
        self.entropy_coef = self.initial_entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        adaptive_lr_cfg = self.config.get("adaptive_lr", {})
        self.adaptive_lr_adapt_critic = bool(
            adaptive_lr_cfg.get("adapt_critic", False)
        )
        self.adaptive_lr_factor = float(adaptive_lr_cfg.get("lr_scaler", 1.2))
        self.adaptive_lr_kl_high_factor = float(
            adaptive_lr_cfg.get("kl_high_factor", 2.0)
        )
        self.adaptive_lr_kl_low_factor = float(
            adaptive_lr_cfg.get("kl_low_factor", 0.5)
        )
        self.adaptive_lr_min = float(
            adaptive_lr_cfg.get("min_learning_rate", 1.0e-6)
        )
        self.adaptive_lr_max = float(
            adaptive_lr_cfg.get("max_learning_rate", 1.0)
        )
        if self.adaptive_lr_factor <= 1.0:
            raise ValueError("adaptive_lr.lr_scaler must be > 1.")
        if self.adaptive_lr_kl_high_factor <= 0.0:
            raise ValueError("adaptive_lr.kl_high_factor must be > 0.")
        if self.adaptive_lr_kl_low_factor <= 0.0:
            raise ValueError("adaptive_lr.kl_low_factor must be > 0.")
        if self.adaptive_lr_min <= 0.0:
            raise ValueError("adaptive_lr.min_learning_rate must be > 0.")
        if self.adaptive_lr_max < self.adaptive_lr_min:
            raise ValueError(
                "adaptive_lr.max_learning_rate must be >= "
                "adaptive_lr.min_learning_rate."
            )
        kl_early_stop_cfg = distributed_update_cfg.get("kl_early_stop", {})
        self.kl_early_stop_enabled = bool(
            kl_early_stop_cfg.get("enabled", False)
        )
        kl_signal_mode = str(
            kl_early_stop_cfg.get("signal", "window_mean")
        ).lower()
        if kl_signal_mode != "window_mean":
            raise ValueError(
                "Only distributed_update.kl_early_stop.signal='window_mean' "
                "is supported."
            )
        self.kl_early_stop_window_size = int(
            kl_early_stop_cfg.get("window_size", 3)
        )
        self.kl_early_stop_factor = float(kl_early_stop_cfg.get("factor", 2.0))
        self.kl_early_stop_min_updates = int(
            kl_early_stop_cfg.get("min_updates", 1)
        )
        if self.kl_early_stop_window_size < 1:
            raise ValueError(
                "distributed_update.kl_early_stop.window_size must be >= 1."
            )
        if self.kl_early_stop_factor <= 0.0:
            raise ValueError(
                "distributed_update.kl_early_stop.factor must be > 0."
            )
        if self.kl_early_stop_min_updates < 1:
            raise ValueError(
                "distributed_update.kl_early_stop.min_updates must be >= 1."
            )
        if self.kl_early_stop_enabled and self.desired_kl is None:
            raise ValueError(
                "distributed_update.kl_early_stop requires desired_kl to be set."
            )
        self.global_advantage_norm = bool(
            self.config.get("global_advantage_norm", True)
        )
        self.normalize_advantage_per_mini_batch = bool(
            self.config.get("normalize_advantage_per_mini_batch", False)
        )
        self.distributed_lr_scale_factor = self._compute_lr_scale_factor(
            distributed_update_cfg.get("lr_scale", {})
        )
        self.actor_learning_rate = (
            self.base_actor_learning_rate * self.distributed_lr_scale_factor
        )
        self.critic_learning_rate = (
            self.base_critic_learning_rate * self.distributed_lr_scale_factor
        )
        self._last_update_metrics = {
            "0-Train/configured_num_mini_batches": float(
                self.configured_num_mini_batches
            ),
            "0-Train/requested_num_mini_batches": float(
                self.requested_num_mini_batches
            ),
            "0-Train/effective_num_mini_batches": float(
                self.requested_num_mini_batches
            ),
            "0-Train/mini_batch_size_per_rank": 0.0,
            "0-Train/num_updates_executed": 0.0,
            "0-Train/lr_scale_factor": float(self.distributed_lr_scale_factor),
            "0-Train/scalable_distributed_update": float(
                self.distributed_update_mode == "scalable"
            ),
            "0-Train/kl_windowed": 0.0,
            "0-Train/kl_stop_triggered": 0.0,
            "0-Train/kl_stop_analytic": 0.0,
        }
        self._offline_evaluating: bool = False

        motion_cfg = self.env_config.config.robot.motion
        sampling_strategy_cfg = motion_cfg.get("sampling_strategy", None)
        if sampling_strategy_cfg is None:
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
        self.weighted_bin_cfg = dict(motion_cfg.get("weighted_bin", {}))

        sym_cfg = self.config.get("symmetry_loss", {})
        self.symmetry_loss_enabled = bool(sym_cfg.get("enabled", False))
        self.symmetry_loss_coef = float(sym_cfg.get("coef", 0.0))
        self._sym_dof_perm: torch.Tensor | None = None
        self._sym_dof_sign: torch.Tensor | None = None
        self._obs_mirror_map: dict[str, callable] = {}
        if self._symmetry_loss_active():
            self._setup_symmetry()

    def _resolve_num_mini_batches(self, base_num_mini_batches: int) -> int:
        if self.distributed_update_mode == "legacy" and self.is_distributed:
            return max(1, base_num_mini_batches * int(self.gpu_world_size))
        return max(1, base_num_mini_batches)

    def _compute_lr_scale_factor(self, lr_scale_cfg) -> float:
        scale_mode = str(lr_scale_cfg.get("mode", "none")).lower()
        if scale_mode not in {
            "none",
            "sqrt_world_size",
            "linear_world_size",
        }:
            raise ValueError(
                "distributed_update.lr_scale.mode must be one of "
                "{'none', 'sqrt_world_size', 'linear_world_size'}."
            )
        reference_world_size = float(
            lr_scale_cfg.get("reference_world_size", 1)
        )
        if reference_world_size <= 0.0:
            raise ValueError(
                "distributed_update.lr_scale.reference_world_size must be > 0."
            )
        runtime_world_size = float(
            self.gpu_world_size if self.is_distributed else 1
        )
        world_ratio = runtime_world_size / reference_world_size
        if scale_mode == "none":
            scale = 1.0
        elif scale_mode == "sqrt_world_size":
            scale = math.sqrt(world_ratio)
        else:
            scale = world_ratio
        max_scale = lr_scale_cfg.get("max_scale", None)
        if max_scale is not None:
            max_scale = float(max_scale)
            if max_scale <= 0.0:
                raise ValueError(
                    "distributed_update.lr_scale.max_scale must be > 0 when set."
                )
            scale = min(scale, max_scale)
        return float(scale)

    def _symmetry_loss_active(self) -> bool:
        return bool(
            getattr(self, "command_name", None) == "base_velocity"
            and getattr(self, "symmetry_loss_enabled", False)
            and float(getattr(self, "symmetry_loss_coef", 0.0)) > 0.0
        )

    @staticmethod
    def _omega_or_obj_to_dict(value):
        if value is None:
            return {}
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            return vars(value)
        return {}

    def _setup_symmetry(self) -> None:
        robot_asset = self.env._env.scene["robot"]
        joint_names = list(getattr(robot_asset, "joint_names", []))
        if len(joint_names) != int(self.num_actions):
            raise ValueError(
                "symmetry_loss requires simulator joint_names to match "
                f"num_actions, got {len(joint_names)} vs {self.num_actions}."
            )

        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        perm: list[int] = []
        for name in joint_names:
            if name.startswith("left_"):
                mirror_name = "right_" + name[len("left_") :]
            elif name.startswith("right_"):
                mirror_name = "left_" + name[len("right_") :]
            else:
                mirror_name = name
            perm.append(int(name_to_idx.get(mirror_name, name_to_idx[name])))

        sym_cfg = self._omega_or_obj_to_dict(
            self.config.get("symmetry_loss", {})
        )
        sign_by_name = sym_cfg.get("dof_sign_by_name", None)
        if not sign_by_name:
            robot_cfg = self._omega_or_obj_to_dict(
                getattr(
                    getattr(self.env_config, "config", None), "robot", None
                )
            )
            sign_by_name = robot_cfg.get("dof_sign_by_name", None)
        sign_by_name = self._omega_or_obj_to_dict(sign_by_name)
        if len(sign_by_name) == 0:
            raise ValueError(
                "symmetry_loss requires dof_sign_by_name in algo or robot config."
            )

        sign = [float(sign_by_name.get(name, 1.0)) for name in joint_names]
        self._sym_dof_perm = torch.tensor(
            perm, device=self.device, dtype=torch.long
        )
        self._sym_dof_sign = torch.tensor(
            sign, device=self.device, dtype=torch.float32
        )
        self._build_obs_mirror_map()

    def _extract_obs_mirror_metadata(self) -> dict[str, dict]:
        obs_cfg = getattr(
            getattr(self.env_config, "config", None), "obs", None
        )
        obs_root = self._omega_or_obj_to_dict(obs_cfg)
        obs_groups = obs_root.get("obs_groups", {})
        metadata: dict[str, dict] = {}
        for group_name, group_cfg in obs_groups.items():
            if not isinstance(group_cfg, dict):
                continue
            for term_entry in group_cfg.get("atomic_obs_list", []):
                if not isinstance(term_entry, dict):
                    continue
                for term_name, term_cfg in term_entry.items():
                    term_cfg = self._omega_or_obj_to_dict(term_cfg)
                    mirror_func = term_cfg.get("mirror_func", None)
                    if not mirror_func:
                        continue
                    metadata[f"{group_name}/{term_name}"] = {
                        "mirror_func": str(mirror_func),
                        "mirror_config": self._omega_or_obj_to_dict(
                            term_cfg.get("mirror_config", {})
                        ),
                    }
        return metadata

    def _get_actor_schema_terms(self) -> set[str]:
        module_dict = self._omega_or_obj_to_dict(
            self.config.get("module_dict", {})
        )
        actor_cfg = self._omega_or_obj_to_dict(module_dict.get("actor", {}))
        actor_schema = self._omega_or_obj_to_dict(
            actor_cfg.get("obs_schema", {})
        )
        actor_terms: set[str] = set()
        for seq_cfg in actor_schema.values():
            if not isinstance(seq_cfg, dict):
                continue
            for term in seq_cfg.get("terms", []):
                actor_terms.add(str(term))
        return actor_terms

    def _build_obs_mirror_map(self) -> None:
        from holomotion.src.env.isaaclab_components.isaaclab_observation import (
            MirrorFunctions,
        )

        self._obs_mirror_map = {}
        if self._sym_dof_perm is None or self._sym_dof_sign is None:
            return

        term_meta = self._extract_obs_mirror_metadata()
        actor_terms = self._get_actor_schema_terms()
        for term in actor_terms:
            meta = term_meta.get(term)
            if meta is None:
                continue
            mirror_func = str(meta.get("mirror_func", ""))
            if mirror_func == "mirror_dof":
                perm = self._sym_dof_perm
                sign = self._sym_dof_sign

                def _fn(x, perm=perm, sign=sign):
                    perm_local = perm.to(device=x.device, dtype=torch.long)
                    sign_local = sign.to(device=x.device, dtype=x.dtype)
                    return MirrorFunctions.mirror_dof(
                        x, perm=perm_local, sign=sign_local
                    )

            elif mirror_func == "mirror_vec3":

                def _fn(x):
                    return MirrorFunctions.mirror_vec3(x)

            elif mirror_func == "mirror_axial_vec3":

                def _fn(x):
                    return MirrorFunctions.mirror_axial_vec3(x)

            elif mirror_func == "mirror_velocity_command":

                def _fn(x):
                    return MirrorFunctions.mirror_velocity_command(x)

            else:
                continue

            self._obs_mirror_map[term] = _fn

    @staticmethod
    def _td_key_to_path(key) -> str:
        if isinstance(key, tuple):
            return "/".join(str(part) for part in key)
        return str(key)

    def _mirror_actor_obs(self, obs_td: TensorDict) -> TensorDict:
        if (
            not self._symmetry_loss_active()
            or not isinstance(obs_td, TensorDict)
            or len(getattr(self, "_obs_mirror_map", {})) == 0
        ):
            return obs_td

        mirrored = TensorDict(
            {},
            batch_size=list(obs_td.batch_size),
            device=obs_td.device,
        )
        for key in obs_td.keys(include_nested=True, leaves_only=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            value = obs_td.get(key_tuple)
            mirror_fn = self._obs_mirror_map.get(
                self._td_key_to_path(key_tuple)
            )
            mirrored.set(
                key_tuple,
                mirror_fn(value) if mirror_fn is not None else value,
            )
        return mirrored

    def _mirror_env_action(self, actions: torch.Tensor) -> torch.Tensor:
        from holomotion.src.env.isaaclab_components.isaaclab_observation import (
            MirrorFunctions,
        )

        if not self._symmetry_loss_active():
            return actions
        if self._sym_dof_perm is None or self._sym_dof_sign is None:
            raise RuntimeError(
                "Symmetry DOF permutation/signs are not initialized."
            )
        return MirrorFunctions.mirror_action(
            actions,
            perm=self._sym_dof_perm.to(
                device=actions.device, dtype=torch.long
            ),
            sign=self._sym_dof_sign.to(
                device=actions.device, dtype=actions.dtype
            ),
        )

    def _compute_analytic_kl(
        self,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        new_mu: torch.Tensor,
        new_sigma: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> float:
        with torch.no_grad():
            kl_vec = torch.sum(
                torch.log((new_sigma + 1.0e-8) / (old_sigma + 1.0e-8))
                + (torch.square(old_sigma) + torch.square(old_mu - new_mu))
                / (2.0 * torch.square(new_sigma) + 1.0e-8)
                - 0.5,
                dim=-1,
            )
            if weight is None:
                kl_sum = kl_vec.sum()
                kl_count = torch.tensor(
                    float(kl_vec.numel()),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                kl_weight = weight.to(dtype=torch.float32)
                kl_sum = (kl_vec * kl_weight).sum()
                kl_count = kl_weight.sum()
            if self.is_distributed:
                kl_sum = self.accelerator.reduce(kl_sum, reduction="sum")
                kl_count = self.accelerator.reduce(kl_count, reduction="sum")
            kl_mean = kl_sum / kl_count.clamp_min(1.0)
        return float(kl_mean.item())

    def _compute_clip_fraction(
        self,
        ratio: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> float:
        with torch.no_grad():
            clipped = (
                (ratio < (1.0 - self.clip_param))
                | (ratio > (1.0 + self.clip_param))
            ).to(torch.float32)
            if weight is None:
                clip_sum = clipped.sum()
                clip_count = torch.tensor(
                    float(clipped.numel()),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                clip_weight = weight.to(dtype=torch.float32)
                clip_sum = (clipped * clip_weight).sum()
                clip_count = clip_weight.sum()
            if self.is_distributed:
                clip_sum = self.accelerator.reduce(clip_sum, reduction="sum")
                clip_count = self.accelerator.reduce(
                    clip_count, reduction="sum"
                )
            clip_fraction = clip_sum / clip_count.clamp_min(1.0)
        return float(clip_fraction.item())

    def _compute_explained_variance(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> float:
        with torch.no_grad():
            target_f = target.float()
            prediction_f = prediction.float()
            residual = target_f - prediction_f
            if weight is None:
                weight_f = torch.ones_like(target_f, dtype=torch.float32)
            else:
                weight_f = weight.to(dtype=torch.float32)

            count = weight_f.sum()
            target_sum = (target_f * weight_f).sum()
            target_sq_sum = (target_f.square() * weight_f).sum()
            residual_sum = (residual * weight_f).sum()
            residual_sq_sum = (residual.square() * weight_f).sum()

            if self.is_distributed:
                count = self.accelerator.reduce(count, reduction="sum")
                target_sum = self.accelerator.reduce(
                    target_sum, reduction="sum"
                )
                target_sq_sum = self.accelerator.reduce(
                    target_sq_sum, reduction="sum"
                )
                residual_sum = self.accelerator.reduce(
                    residual_sum, reduction="sum"
                )
                residual_sq_sum = self.accelerator.reduce(
                    residual_sq_sum, reduction="sum"
                )

            denom = count.clamp_min(1.0)
            target_mean = target_sum / denom
            residual_mean = residual_sum / denom
            target_var = target_sq_sum / denom - target_mean.square()
            residual_var = residual_sq_sum / denom - residual_mean.square()
            if float(target_var.item()) <= 1.0e-8:
                return 0.0
            explained_variance = 1.0 - residual_var / target_var
        return float(explained_variance.item())

    def _set_optimizer_learning_rates(self) -> None:
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    @staticmethod
    def _validate_entropy_schedule_config(
        *,
        initial_entropy_coef: float,
        anneal_entropy: bool,
        zero_entropy_point: float,
    ) -> None:
        if float(initial_entropy_coef) < 0.0:
            raise ValueError("entropy_coef must be >= 0.")
        if anneal_entropy and not (0.0 < float(zero_entropy_point) <= 1.0):
            raise ValueError(
                "zero_entropy_point must be in (0.0, 1.0] when "
                "anneal_entropy is enabled."
            )

    def _get_effective_entropy_coef(self) -> float:
        if self.initial_entropy_coef <= 0.0 or not self.anneal_entropy:
            return float(self.initial_entropy_coef)
        total_learning_iterations = int(
            getattr(
                self,
                "total_learning_iterations",
                self.current_learning_iteration
                + int(self.num_learning_iterations),
            )
        )
        total_learning_iterations = max(1, total_learning_iterations)
        zero_entropy_iteration = float(self.zero_entropy_point) * float(
            total_learning_iterations
        )
        anneal_scale = max(
            0.0,
            1.0
            - float(self.current_learning_iteration) / zero_entropy_iteration,
        )
        return float(self.initial_entropy_coef * anneal_scale)

    def _apply_adaptive_lr(self, kl_signal: float | None) -> None:
        if (
            self.desired_kl is None
            or self.schedule != "adaptive"
            or kl_signal is None
        ):
            return
        if kl_signal > self.desired_kl * self.adaptive_lr_kl_high_factor:
            self.actor_learning_rate = max(
                self.adaptive_lr_min,
                self.actor_learning_rate / self.adaptive_lr_factor,
            )
            if self.adaptive_lr_adapt_critic:
                self.critic_learning_rate = max(
                    self.adaptive_lr_min,
                    self.critic_learning_rate / self.adaptive_lr_factor,
                )
        elif (
            kl_signal > 0.0
            and kl_signal < self.desired_kl * self.adaptive_lr_kl_low_factor
        ):
            self.actor_learning_rate = min(
                self.adaptive_lr_max,
                self.actor_learning_rate * self.adaptive_lr_factor,
            )
            if self.adaptive_lr_adapt_critic:
                self.critic_learning_rate = min(
                    self.adaptive_lr_max,
                    self.critic_learning_rate * self.adaptive_lr_factor,
                )
        self._set_optimizer_learning_rates()

    def _compute_windowed_kl_signal(
        self, recent_analytic_kls: list[float]
    ) -> float | None:
        if len(recent_analytic_kls) < self.kl_early_stop_window_size:
            return None
        window = recent_analytic_kls[-self.kl_early_stop_window_size :]
        return float(sum(window) / len(window))

    def _should_early_stop_for_kl(
        self,
        kl_signal: float | None,
        num_kl_measurements: int,
    ) -> bool:
        if not self.kl_early_stop_enabled or self.desired_kl is None:
            return False
        if kl_signal is None:
            return False
        required_measurements = max(
            self.kl_early_stop_min_updates, self.kl_early_stop_window_size
        )
        if num_kl_measurements < required_measurements:
            return False
        return kl_signal > self.desired_kl * self.kl_early_stop_factor

    def _setup_data_buffers(self):
        super()._setup_data_buffers()
        self.use_velocity_transition: bool = (
            self.command_name == "base_velocity"
        )
        self.transition_cls = (
            PpoVelocityTransition
            if self.use_velocity_transition
            else PpoTransition
        )
        self.transition_td: PpoTransition | PpoVelocityTransition | None = None

    def _build_optimizer_kwargs(self, optimizer_class: type) -> dict:
        if self.optimizer_type != "AdamW":
            return {}
        signature = inspect.signature(optimizer_class.__init__)
        parameters = signature.parameters
        use_fused = bool(
            self.config.get(
                "adamw_use_fused", bool(self.device.type == "cuda")
            )
        )
        use_foreach = bool(self.config.get("adamw_use_foreach", True))
        if (
            use_fused
            and ("fused" in parameters)
            and (self.device.type == "cuda")
        ):
            return {"fused": True}
        if use_foreach and ("foreach" in parameters):
            return {"foreach": True}
        return {}

    def _setup_models_and_optimizer(self):
        from holomotion.src.modules.agent_modules import PPOActor, PPOCritic

        # Build sample TensorDict for schema-based assembly
        sample_obs_dict = self.env.reset_all()[0]
        sample_td = self._wrap_obs_dict(sample_obs_dict)
        actor_cfg = OmegaConf.to_container(
            self.config.module_dict.actor, resolve=True
        )
        critic_cfg = OmegaConf.to_container(
            self.config.module_dict.critic, resolve=True
        )

        self.actor_type = actor_cfg.get("type", "MLP")
        self.critic_type = critic_cfg.get("type", "MLP")

        actor_schema = actor_cfg.get("obs_schema", None)
        critic_schema = critic_cfg.get("obs_schema", None)

        self.actor = PPOActor(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)

        self.critic = PPOCritic(
            obs_schema=critic_schema,
            module_config_dict=critic_cfg,
            obs_example=sample_td,
        ).to(self.device)

        if self.is_main_process:
            actor = self.accelerator.unwrap_model(self.actor)
            critic = self.accelerator.unwrap_model(self.critic)

            logger.info("Actor (TensorDict module):\n{!r}", actor)
            logger.info(
                "Actor keys: in_keys={} out_keys={}",
                list(actor.in_keys),
                list(actor.out_keys),
            )
            logger.info("Actor core nn module:\n{!r}", actor.actor_module)

            logger.info("Critic (TensorDict module):\n{!r}", critic)
            logger.info(
                "Critic keys: in_keys={} out_keys={}",
                list(critic.in_keys),
                list(critic.out_keys),
            )
            logger.info("Critic core nn module:\n{!r}", critic.critic_module)

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
        optimizer_kwargs = self._build_optimizer_kwargs(optimizer_class)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
            betas=(self.actor_beta1, self.actor_beta2),
            **optimizer_kwargs,
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(self.critic_beta1, self.critic_beta2),
            **optimizer_kwargs,
        )

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

    def _build_storage(self, obs_td: TensorDict):
        return RolloutStorage(
            self.num_envs,
            self.num_steps_per_env,
            obs_template=obs_td,
            actions_shape=[self.num_actions],
            device=self.device,
            command_name=self.command_name,
            transition_cls=self.transition_cls,
        )

    def _build_transition(
        self,
        obs_td: TensorDict,
        actor_out: TensorDict,
        critic_out: TensorDict,
    ):
        actions = actor_out.get("actions")
        actions_log_prob = actor_out.get("actions_log_prob")
        mu = actor_out.get("mu")
        sigma = actor_out.get("sigma")
        values = critic_out.get("values")

        zero_scalar = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
            dtype=torch.float32,
        )
        zero_scalar_bool = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
            dtype=torch.bool,
        )

        transition_kwargs = {
            "obs": obs_td,
            "actions": actions.detach(),
            "teacher_actions": torch.zeros_like(actions),
            "mu": mu.detach(),
            "sigma": sigma.detach(),
            "actions_log_prob": actions_log_prob[..., None].detach(),
            "values": values.detach(),
            "rewards": zero_scalar.clone(),
            "dones": zero_scalar_bool,
            "returns": zero_scalar.clone(),
            "advantages": zero_scalar.clone(),
            "batch_size": [self.num_envs],
            "device": self.device,
        }

        if self.use_velocity_transition:
            import isaaclab.envs.mdp as isaaclab_mdp

            velocity_cmd = isaaclab_mdp.generated_commands(
                self.env._env, command_name="base_velocity"
            )
            if velocity_cmd.shape[-1] > 3:
                velocity_cmd = velocity_cmd[..., :3]
            move_mask = (velocity_cmd.norm(dim=-1) > 0.1).to(
                dtype=velocity_cmd.dtype
            )
            transition_kwargs["velocity_commands"] = torch.cat(
                [move_mask[..., None], velocity_cmd],
                dim=-1,
            ).detach()

        return self.transition_cls(**transition_kwargs)

    def _post_iteration_hook(self, it: int) -> None:
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")
            motion_cmd.apply_cache_swap_if_pending_barrier(
                accelerator=self.accelerator
            )

    def _post_training_hook(self) -> None:
        if self.command_name == "ref_motion":
            motion_cmd = self.env._env.command_manager.get_term("ref_motion")
            if motion_cmd is not None:
                motion_cmd.close()

    def _get_mean_policy_std(self) -> torch.Tensor:
        base_actor = self.accelerator.unwrap_model(self.actor)
        if hasattr(base_actor, "std"):
            return base_actor.std.mean()
        if hasattr(base_actor, "log_std"):
            return torch.exp(base_actor.log_std).mean()
        return torch.tensor(0.0, device=self.device)

    def _maybe_override_loaded_actor_sigma(self) -> None:
        if not bool(self.config.get("override_sigma", False)):
            return

        sigma_override = self.config.get("sigma_override", None)
        if sigma_override is None:
            raise ValueError(
                "config.override_sigma is enabled but config.sigma_override is not set."
            )

        actor_unwrapped = self.accelerator.unwrap_model(self.actor)
        orig_mod = getattr(actor_unwrapped, "_orig_mod", None)
        if orig_mod is not None:
            actor_unwrapped = orig_mod

        override_sigma = getattr(actor_unwrapped, "override_sigma", None)
        if override_sigma is None:
            raise AttributeError(
                f"{type(actor_unwrapped).__name__} does not implement override_sigma()."
            )

        override_sigma(sigma_override)
        if self.is_main_process:
            logger.info(
                "Reapplied sigma override after checkpoint load: {}",
                sigma_override,
            )

    def _get_additional_log_metrics(self) -> dict[str, float]:
        """Build auxiliary training/cache metrics."""
        iteration_metrics = {}

        if "actor_learning_rate" in self.__dict__:
            iteration_metrics["0-Train/actor_learning_rate"] = float(
                self.actor_learning_rate
            )

        if "critic_learning_rate" in self.__dict__:
            iteration_metrics["0-Train/critic_learning_rate"] = float(
                self.critic_learning_rate
            )

        if "initial_entropy_coef" in self.__dict__:
            iteration_metrics["0-Train/entropy_coef_effective"] = float(
                self._get_effective_entropy_coef()
            )

        if "_last_update_metrics" in self.__dict__:
            iteration_metrics.update(self._last_update_metrics)

        mean_std = self._get_mean_policy_std()
        iteration_metrics["0-Train/mean_noise_std"] = float(mean_std.item())

        if self.command_name != "ref_motion":
            return iteration_metrics

        motion_cmd = self.env._env.command_manager.get_term("ref_motion")
        cache = motion_cmd._motion_cache
        iteration_metrics["1-Perf/Cache/swap_index"] = float(cache.swap_index)
        pool_stats = cache.cache_curriculum_pool_statistics()
        if pool_stats is not None:
            core_cache_metric_keys = {
                "prioritized_pool_size": "1-Perf/Cache/prioritized_pool_size",
                "prioritized_pool_mean_score": "1-Perf/Cache/prioritized_pool_mean_score",
                "uniform_pool_mean_score": "1-Perf/Cache/uniform_pool_mean_score",
                "entered_prioritized_pool_count": "1-Perf/Cache/entered_prioritized_pool_count",
                "exited_prioritized_pool_count": "1-Perf/Cache/exited_prioritized_pool_count",
            }
            for src_key, dst_key in core_cache_metric_keys.items():
                if src_key in pool_stats:
                    iteration_metrics[dst_key] = float(pool_stats[src_key])
        return iteration_metrics

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_analytic = 0.0
        mean_symmetry_loss = 0.0
        critic_explained_variance = self._compute_explained_variance(
            target=self.storage.data["returns"],
            prediction=self.storage.data["values"],
        )

        batch_size = int(
            self.storage.num_envs * self.storage.num_transitions_per_env
        )
        (
            effective_num_mini_batches,
            mini_batch_size,
        ) = RolloutStorage.resolve_mini_batch_partition(
            batch_size, self.num_mini_batches
        )
        self._last_update_metrics = {
            "0-Train/configured_num_mini_batches": float(
                self.configured_num_mini_batches
            ),
            "0-Train/requested_num_mini_batches": float(
                self.requested_num_mini_batches
            ),
            "0-Train/effective_num_mini_batches": float(
                effective_num_mini_batches
            ),
            "0-Train/mini_batch_size_per_rank": float(mini_batch_size),
            "0-Train/num_updates_executed": 0.0,
            "0-Train/lr_scale_factor": float(self.distributed_lr_scale_factor),
            "0-Train/scalable_distributed_update": float(
                self.distributed_update_mode == "scalable"
            ),
            "0-Train/kl_windowed": 0.0,
            "0-Train/kl_stop_triggered": 0.0,
            "0-Train/kl_stop_analytic": 0.0,
            "0-Train/kl_analytic_batch_last": 0.0,
            "0-Train/kl_analytic_batch_max": 0.0,
            "0-Train/clip_fraction_batch_mean": 0.0,
            "0-Train/clip_fraction_batch_last": 0.0,
        }
        entropy_coef = self._get_effective_entropy_coef()

        generator = self.storage.iter_minibatches(
            effective_num_mini_batches,
            self.num_learning_epochs,
        )
        measure_analytic_kl = self.desired_kl is not None
        num_updates = 0
        num_kl_measurements = 0
        kl_stop_triggered = False
        kl_stop_analytic = 0.0
        kl_windowed = None
        recent_analytic_kls: list[float] = []
        kl_analytic_batch_last = 0.0
        kl_analytic_batch_max = 0.0
        clip_fraction_batch_mean = 0.0
        clip_fraction_batch_last = 0.0

        for batch in generator:
            obs_batch = batch.obs
            actions_batch = batch.actions
            target_values_batch = batch.values
            advantages_batch = batch.advantages
            returns_batch = batch.returns
            old_actions_log_prob_batch = batch.actions_log_prob
            old_mu_batch = batch.mu
            old_sigma_batch = batch.sigma
            with self.accelerator.autocast():
                actor_out = self.actor(
                    obs_batch,
                    actions=actions_batch,
                    mode="logp",
                    update_obs_norm=False,
                )
                critic_out = self.critic(obs_batch, update_obs_norm=False)
                actions_log_prob_batch = actor_out.get("actions_log_prob")
                mu_batch = actor_out.get("mu")
                sigma_batch = actor_out.get("sigma")
                entropy_batch = actor_out.get("entropy")
                value_pred = critic_out.get("values")
                symmetry_loss = None
                if self._symmetry_loss_active():
                    mirrored_obs_batch = self._mirror_actor_obs(obs_batch)
                    mirrored_actor_out = self.actor(
                        mirrored_obs_batch,
                        actions=None,
                        mode="inference",
                        update_obs_norm=False,
                    )
                    mirrored_actions = mirrored_actor_out.get("actions")
                    mirrored_actions_back = self._mirror_env_action(
                        mirrored_actions
                    )
                    symmetry_loss = F.mse_loss(
                        mu_batch.float(),
                        mirrored_actions_back.float(),
                    )

            value_batch = value_pred
            returns_batch_norm = returns_batch
            target_values_batch_norm = target_values_batch

            analytic_kl = None
            if measure_analytic_kl:
                analytic_kl = self._compute_analytic_kl(
                    old_mu=old_mu_batch.float(),
                    old_sigma=old_sigma_batch.float(),
                    new_mu=mu_batch.float(),
                    new_sigma=sigma_batch.float(),
                )
                mean_kl_analytic += analytic_kl
                num_kl_measurements += 1
                kl_analytic_batch_last = analytic_kl
                kl_analytic_batch_max = max(kl_analytic_batch_max, analytic_kl)
                recent_analytic_kls.append(analytic_kl)
                if len(recent_analytic_kls) > self.kl_early_stop_window_size:
                    recent_analytic_kls.pop(0)
                kl_windowed = self._compute_windowed_kl_signal(
                    recent_analytic_kls
                )
                if self._should_early_stop_for_kl(
                    kl_windowed, num_kl_measurements
                ):
                    kl_stop_triggered = True
                    kl_stop_analytic = analytic_kl
                    break

            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch).float()
            )
            clip_fraction = self._compute_clip_fraction(ratio)
            clip_fraction_batch_mean += clip_fraction
            clip_fraction_batch_last = clip_fraction
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

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

            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss

            if entropy_coef > 0.0:
                entropy_loss = entropy_batch.mean()
                actor_loss = actor_loss - entropy_coef * entropy_loss
            if symmetry_loss is not None:
                actor_loss = (
                    actor_loss + self.symmetry_loss_coef * symmetry_loss
                )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.accelerator.backward(actor_loss)
            self.accelerator.backward(critic_loss)

            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )
                self.accelerator.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            num_updates += 1
            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy_batch.mean().item())
            if symmetry_loss is not None:
                mean_symmetry_loss += float(symmetry_loss.item())

        denom = max(1, num_updates)
        mean_value_loss /= denom
        mean_surrogate_loss /= denom
        mean_entropy /= denom
        mean_kl_analytic /= max(1, num_kl_measurements)
        mean_symmetry_loss /= denom
        clip_fraction_batch_mean /= denom
        if self.schedule == "adaptive":
            self._apply_adaptive_lr(kl_windowed)
        self._last_update_metrics["0-Train/num_updates_executed"] = float(
            num_updates
        )
        self._last_update_metrics["0-Train/kl_windowed"] = float(
            kl_windowed or 0.0
        )
        self._last_update_metrics["0-Train/kl_stop_triggered"] = float(
            kl_stop_triggered
        )
        self._last_update_metrics["0-Train/kl_stop_analytic"] = float(
            kl_stop_analytic
        )
        self._last_update_metrics["0-Train/kl_analytic_batch_last"] = float(
            kl_analytic_batch_last
        )
        self._last_update_metrics["0-Train/kl_analytic_batch_max"] = float(
            kl_analytic_batch_max
        )
        self._last_update_metrics["0-Train/clip_fraction_batch_mean"] = float(
            clip_fraction_batch_mean
        )
        self._last_update_metrics["0-Train/clip_fraction_batch_last"] = float(
            clip_fraction_batch_last
        )

        self.storage.clear()

        loss_out = {
            "value_function": mean_value_loss,
            "critic_explained_variance": critic_explained_variance,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_analytic": mean_kl_analytic,
        }
        if self._symmetry_loss_active():
            loss_out["symmetry_loss"] = mean_symmetry_loss
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

        self._post_update_hook(loss_out)
        return loss_out

    def load(self, ckpt_path):
        if ckpt_path is None:
            return None
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {ckpt_path}")

        actor_model_path = self._resolve_model_file_path(ckpt_path, "actor")
        critic_model_path = self._resolve_model_file_path(ckpt_path, "critic")
        self._load_accelerate_model(self.actor, actor_model_path, strict=True)
        self._load_accelerate_model(
            self.critic, critic_model_path, strict=True
        )

        loaded_dict = torch.load(ckpt_path, map_location=self.device)
        if not getattr(self, "is_offline_eval", False):
            self._restore_optimizer_state(
                self.actor_optimizer,
                loaded_dict["actor_optimizer_state_dict"],
                optimizer_name="actor",
            )
            self._restore_optimizer_state(
                self.critic_optimizer,
                loaded_dict["critic_optimizer_state_dict"],
                optimizer_name="critic",
            )
        elif self.is_main_process:
            logger.info(
                "Skipping optimizer state restore during offline evaluation."
            )
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        self._maybe_override_loaded_actor_sigma()
        self._load_extra_checkpoint_state(loaded_dict)
        return loaded_dict.get("infos", None)

    def _restore_optimizer_state(
        self,
        optimizer,
        loaded_state_dict,
        *,
        optimizer_name: str,
    ) -> bool:
        compatible, reason = self._optimizer_state_is_compatible(
            optimizer, loaded_state_dict
        )
        if not compatible:
            if self.is_main_process:
                logger.warning(
                    "Skipping {} optimizer state restore from checkpoint: {}",
                    optimizer_name,
                    reason,
                )
            return False

        try:
            optimizer.load_state_dict(loaded_state_dict)
        except ValueError as exc:
            if self.is_main_process:
                logger.warning(
                    "Skipping {} optimizer state restore from checkpoint: {}",
                    optimizer_name,
                    exc,
                )
            return False
        return True

    def _optimizer_state_is_compatible(
        self, optimizer, loaded_state_dict
    ) -> tuple[bool, str | None]:
        current_state_dict = optimizer.state_dict()
        current_groups = current_state_dict.get("param_groups")
        loaded_groups = loaded_state_dict.get("param_groups")
        if not isinstance(current_groups, list) or not isinstance(
            loaded_groups, list
        ):
            return True, None
        if len(current_groups) != len(loaded_groups):
            return (
                False,
                "param group count mismatch "
                f"(current={len(current_groups)}, loaded={len(loaded_groups)})",
            )

        for group_idx, (current_group, loaded_group) in enumerate(
            zip(current_groups, loaded_groups)
        ):
            current_param_count = len(current_group.get("params", []))
            loaded_param_count = len(loaded_group.get("params", []))
            if current_param_count != loaded_param_count:
                return (
                    False,
                    "param group size mismatch for group "
                    f"{group_idx} (current={current_param_count}, "
                    f"loaded={loaded_param_count})",
                )
        return True, None

    def save(self, path, infos=None):
        if not self.is_main_process:
            return

        logger.info(f"Saving checkpoint to {path}")
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

        custom_state = {
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        custom_state.update(self._extra_checkpoint_state())
        torch.save(_checkpoint_state_to_cpu(custom_state), path)

        if bool(self.config.get("export_policy", False)):
            export_policy_to_onnx_common(
                self,
                path,
                onnx_name_suffix=self.config.get("onnx_name_suffix", None),
                use_kv_cache=bool(self.config.get("use_kv_cache", True)),
            )

    def offline_evaluate_policy(self, dump_npzs: bool = False):
        """Dump NPZs (no metrics) from validation cache using ref_motion command.

        - Iterates validation batches; env i -> clip i (deterministic) starting at frame 0.
        - Collect robot and reference sequences each step and save one NPZ per clip.
        - NPZ conforms to holomotion_retargeted format keys.
        - Optionally records viewport MP4(s) aligned with target_fps and rollout length.
        """

        ckpt_path = self.config.checkpoint
        n_fut_frames = self.env.config.commands.ref_motion.params.get(
            "n_fut_frames", 8
        )
        # log_dir is already set to checkpoint directory in eval script
        model_name = os.path.basename(ckpt_path).replace(".pt", "")

        # Eval modes (freeze normalizers if enabled)
        self.actor.eval()
        self.critic.eval()

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

        self._offline_evaluating = True

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
                    seed=getattr(cache, "_seed", None),
                    loader_timeout=getattr(cache, "_loader_timeout", 0.0),
                )
                motion_cmd._motion_cache = cache
        except Exception as e:
            logger.warning(
                f"Offline eval: failed to rebuild cache to batch_size={num_envs}: {e}"
            )

        # Derive HDF5 dataset base name (from validation dataset root) for output naming
        dataset_suffix = None
        val_dataset = cache._datasets["val"]
        dataset_root = None
        if hasattr(val_dataset, "hdf5_root"):
            dataset_root = str(val_dataset.hdf5_root).rstrip(os.sep)
        elif hasattr(val_dataset, "ts_roots"):
            ts_roots = getattr(val_dataset, "ts_roots")
            if ts_roots:
                dataset_root = str(ts_roots[0]).rstrip(os.sep)
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
                        apply_pending_swap=False
                    )
                self._reset_rollout_forward_state()

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

                if active_count > 0:
                    active_ids = torch.arange(
                        active_count,
                        dtype=torch.long,
                        device=self.device,
                    )
                    motion_cmd.force_realign_offline_eval_no_perturb(
                        active_ids
                    )

                # Recompute observations after deterministic setup
                obs_mgr = self.env._env.observation_manager
                if active_count > 0:
                    obs_mgr.reset(active_ids)
                    obs_dict = obs_mgr.compute(update_history=True)
                else:
                    obs_dict = obs_mgr.compute(update_history=True)
                obs = self._wrap_obs_dict(obs_dict)

                # Map env -> motion_key for active envs
                env_motion_keys = {
                    int(i): motion_keys[int(i)] for i in range(active_count)
                }
                env_raw_motion_keys = {
                    int(i): raw_motion_keys[int(i)]
                    for i in range(active_count)
                }

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
                robot_dof_acc = [[] for _ in range(active_count)]
                robot_dof_torque = [[] for _ in range(active_count)]
                robot_action_rate = [[] for _ in range(active_count)]
                prev_robot_dof_vel = [None for _ in range(active_count)]
                prev_robot_actions = [None for _ in range(active_count)]
                step_dt = float(self.env._env.step_dt)

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

                def _get_out_path(idx: int) -> str:
                    out_name = f"{_sanitize_key(env_motion_keys[idx])}.npz"
                    return os.path.join(output_dir, out_name)

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
                    robot_dof_acc_arr = np.stack(
                        robot_dof_acc[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_dof_torque_arr = np.stack(
                        robot_dof_torque[idx][:slice_len], axis=0
                    ).astype(np.float32)
                    robot_action_rate_arr = np.asarray(
                        robot_action_rate[idx][:slice_len], dtype=np.float32
                    )
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
                    out_path = _get_out_path(idx)

                    np.savez_compressed(
                        out_path,
                        metadata=json.dumps(meta),
                        robot_dof_pos=robot_dof_pos_arr,
                        robot_dof_vel=robot_dof_vel_arr,
                        robot_dof_acc=robot_dof_acc_arr,
                        robot_dof_torque=robot_dof_torque_arr,
                        robot_action_rate=robot_action_rate_arr,
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
                            if prev_robot_dof_vel[idx] is None:
                                dof_acc_cur = np.zeros_like(
                                    rob_dv[idx], dtype=np.float32
                                )
                            else:
                                dof_acc_cur = (
                                    rob_dv[idx] - prev_robot_dof_vel[idx]
                                ) / step_dt
                            prev_robot_dof_vel[idx] = rob_dv[idx].copy()

                            ref_dof_pos[idx].append(ref_dp[idx])
                            ref_dof_vel[idx].append(ref_dv[idx])
                            ref_body_pos[idx].append(ref_bp[idx])
                            ref_body_rot_xyzw[idx].append(ref_br[idx])
                            ref_body_vel[idx].append(ref_bv[idx])
                            ref_body_ang_vel[idx].append(ref_bav[idx])

                            robot_dof_pos[idx].append(rob_dp[idx])
                            robot_dof_vel[idx].append(rob_dv[idx])
                            robot_dof_acc[idx].append(
                                dof_acc_cur.astype(np.float32)
                            )
                            robot_body_pos[idx].append(rob_bp[idx])
                            robot_body_rot_xyzw[idx].append(rob_br[idx])
                            robot_body_vel[idx].append(rob_bv[idx])
                            robot_body_ang_vel[idx].append(rob_bav[idx])

                            # Record valid mask for current frame (before step)
                            clip_limit = int(clip_lengths_np[idx])
                            valid_now = (
                                (idx < active_count)
                                and (not encountered_done[idx])
                                and (
                                    env_frame_counts[idx]
                                    < clip_limit - n_fut_frames
                                )
                            )
                            valid_masks[idx].append(bool(valid_now))

                            # Increment local frame counter
                            env_frame_counts[idx] += 1

                    # No mid-rollout finalize; we defer to end using valid masks
                    # Inference and step (advance sim)
                    obs = self._rollout_forward(
                        obs,
                        actor_mode="inference",
                        collect_transition=False,
                        track_episode_stats=False,
                    )
                    dones = self._last_rollout_dones
                    if dones is None:
                        raise RuntimeError(
                            "Rollout forward did not return dones during offline evaluation."
                        )
                    actions_step = self._last_rollout_actions
                    if actions_step is None:
                        raise RuntimeError(
                            "Rollout forward did not return actions during offline evaluation."
                        )

                    actions_np = actions_step.detach().cpu().numpy()
                    torque_urdf = (
                        motion_cmd.robot.data.applied_torque[
                            ..., motion_cmd.sim2urdf_dof_idx
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    for idx in range(active_count):
                        if prev_robot_actions[idx] is None:
                            action_rate_cur = 0.0
                        else:
                            action_rate_cur = float(
                                np.linalg.norm(
                                    actions_np[idx] - prev_robot_actions[idx]
                                )
                                / step_dt
                            )
                        prev_robot_actions[idx] = actions_np[idx].copy()
                        robot_action_rate[idx].append(
                            np.float32(action_rate_cur)
                        )
                        robot_dof_torque[idx].append(
                            torque_urdf[idx].astype(np.float32)
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
                        if dump_npzs and active_count > 0:
                            out_path_to_last_idx = {}
                            for idx in range(active_count):
                                out_path_to_last_idx[_get_out_path(idx)] = idx
                            save_indices = list(out_path_to_last_idx.values())
                            max_npz_save_workers = max(
                                1, min(16, len(save_indices))
                            )
                            with ThreadPoolExecutor(
                                max_workers=max_npz_save_workers
                            ) as executor:
                                futures = [
                                    executor.submit(_save_env_npz, idx)
                                    for idx in save_indices
                                ]
                                for future in tqdm(
                                    as_completed(futures),
                                    total=len(futures),
                                    desc="Saving NPZs",
                                ):
                                    future.result()
                        break

        logger.info(
            f"Offline evaluation complete: saved clips to {output_dir}"
        )
        return {"output_dir": output_dir}
