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
from collections.abc import Mapping
from typing import Any, Generator

import torch
import torch.nn as nn
from loguru import logger
from tabulate import tabulate
from tensordict import TensorDict, tensorclass


class AlgoLogger:
    def __init__(
        self,
        accelerator,
        log_dir: str | None,
        *,
        is_main_process: bool,
    ) -> None:
        self.accelerator = accelerator
        self.log_dir = log_dir
        self.is_main_process = bool(is_main_process)

    @staticmethod
    def _is_scalar_metric(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, torch.Tensor):
            return value.numel() == 1
        return False

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    @staticmethod
    def _format_console_value(value: Any) -> str:
        if isinstance(value, (int, float)):
            value_f = float(value)
            abs_value = abs(value_f)
            if abs_value > 0.0 and (abs_value < 1.0e-4 or abs_value >= 1.0e4):
                return f"{value_f:.4e}"
            return f"{value_f:.4f}"
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value_f = float(value.item())
            abs_value = abs(value_f)
            if abs_value > 0.0 and (abs_value < 1.0e-4 or abs_value >= 1.0e4):
                return f"{value_f:.4e}"
            return f"{value_f:.4f}"
        return str(value)

    def _build_console_log(
        self,
        *,
        step: int,
        total_learning_iterations: int | None,
        console_metrics: Mapping[str, Any],
    ) -> str:
        if total_learning_iterations is None:
            title = f"TRAINING LOG - Iteration {step}"
        else:
            title = (
                f"TRAINING LOG - Iteration {step}/{total_learning_iterations}"
            )
        table_data = [
            [key, str(console_metrics[key])]
            for key in sorted(console_metrics.keys())
        ]
        log_lines = [
            "\n" + "=" * 80,
            title,
            "=" * 80,
            tabulate(
                table_data,
                headers=["Metric", "Value"],
                tablefmt="simple_outline",
                colalign=("left", "left"),
                disable_numparse=True,
            ),
            "=" * 80,
            f"Logging Directory: {os.path.abspath(self.log_dir)}",
            "=" * 80 + "\n",
        ]
        return "\n".join(log_lines)

    def log_iteration(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any],
        total_learning_iterations: int | None = None,
    ) -> None:
        if not self.log_dir or not self.is_main_process:
            return

        tensorboard_metrics: dict[str, float] = {}
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if value is None or not self._is_scalar_metric(value):
                continue
            tensorboard_metrics[key] = self._to_scalar(value)

        if len(tensorboard_metrics) > 0:
            self.accelerator.log(tensorboard_metrics, step=int(step))

        console_metrics = {
            key: self._format_console_value(value)
            for key, value in metrics.items()
            if value is not None
        }
        console_log = self._build_console_log(
            step=step,
            total_learning_iterations=total_learning_iterations,
            console_metrics=console_metrics,
        )
        logger.info(console_log)


@tensorclass(shadow=True)
class PpoTransition:
    """PPO rollout transition tensorclass.

    Batch axes:
    - N: num_envs (per-step)
    - B: minibatch_size (for minibatches)

    Shapes (batch dims = [N] or [B]):
    - obs: TensorDict with leaf tensors [*, ...]
    - actions, teacher_actions, mu, sigma: [*, A]
    - actions_log_prob, values, rewards, returns, advantages, dones: [*, 1]

    All float tensors are float32. `dones` is bool.
    """

    FIELD_SPECS = {
        "obs": {"kind": "obs"},
        "actions": {"shape": ("A",), "dtype": torch.float32},
        "teacher_actions": {"shape": ("A",), "dtype": torch.float32},
        "mu": {"shape": ("A",), "dtype": torch.float32},
        "sigma": {"shape": ("A",), "dtype": torch.float32},
        "actions_log_prob": {"shape": (1,), "dtype": torch.float32},
        "values": {"shape": (1,), "dtype": torch.float32},
        "rewards": {"shape": (1,), "dtype": torch.float32},
        "dones": {"shape": (1,), "dtype": torch.bool},
        "returns": {"shape": (1,), "dtype": torch.float32},
        "advantages": {"shape": (1,), "dtype": torch.float32},
    }

    obs: TensorDict
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    actions_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@tensorclass(shadow=True)
class PpoVelocityTransition:
    """PPO rollout transition tensorclass.

    Batch axes:
    - N: num_envs (per-step)
    - B: minibatch_size (for minibatches)

    Shapes (batch dims = [N] or [B]):
    - obs: TensorDict with leaf tensors [*, ...]
    - actions, teacher_actions, mu, sigma: [*, A]
    - actions_log_prob, values, rewards, returns, advantages, dones: [*, 1]
    - velocity_commands: [*, 4]

    All float tensors are float32. `dones` is bool.
    """

    FIELD_SPECS = {
        "obs": {"kind": "obs"},
        "actions": {"shape": ("A",), "dtype": torch.float32},
        "teacher_actions": {"shape": ("A",), "dtype": torch.float32},
        "mu": {"shape": ("A",), "dtype": torch.float32},
        "sigma": {"shape": ("A",), "dtype": torch.float32},
        "actions_log_prob": {"shape": (1,), "dtype": torch.float32},
        "values": {"shape": (1,), "dtype": torch.float32},
        "rewards": {"shape": (1,), "dtype": torch.float32},
        "dones": {"shape": (1,), "dtype": torch.bool},
        "returns": {"shape": (1,), "dtype": torch.float32},
        "advantages": {"shape": (1,), "dtype": torch.float32},
        "velocity_commands": {"shape": (4,), "dtype": torch.float32},
    }

    obs: TensorDict
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    actions_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    velocity_commands: torch.Tensor


@tensorclass(shadow=True)
class PpoAuxTransition:
    """PPO transition with auxiliary state-prediction supervision targets."""

    SHAPE_TOKENS = {"C": 0, "K": 0}

    FIELD_SPECS = {
        "obs": {"kind": "obs"},
        "actions": {"shape": ("A",), "dtype": torch.float32},
        "teacher_actions": {"shape": ("A",), "dtype": torch.float32},
        "mu": {"shape": ("A",), "dtype": torch.float32},
        "sigma": {"shape": ("A",), "dtype": torch.float32},
        "actions_log_prob": {"shape": (1,), "dtype": torch.float32},
        "values": {"shape": (1,), "dtype": torch.float32},
        "rewards": {"shape": (1,), "dtype": torch.float32},
        "dones": {"shape": (1,), "dtype": torch.bool},
        "returns": {"shape": (1,), "dtype": torch.float32},
        "advantages": {"shape": (1,), "dtype": torch.float32},
        "gt_base_lin_vel_b": {"shape": (3,), "dtype": torch.float32},
        "gt_root_height_rel_terrain": {"shape": (1,), "dtype": torch.float32},
        "gt_keybody_contacts": {"shape": ("C",), "dtype": torch.float32},
        "gt_ref_keybody_rel_pos": {
            "shape": ("K", 3),
            "dtype": torch.float32,
        },
        "gt_robot_keybody_rel_pos": {
            "shape": ("K", 3),
            "dtype": torch.float32,
        },
    }

    obs: TensorDict
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    actions_log_prob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    gt_base_lin_vel_b: torch.Tensor
    gt_root_height_rel_terrain: torch.Tensor
    gt_keybody_contacts: torch.Tensor
    gt_ref_keybody_rel_pos: torch.Tensor
    gt_robot_keybody_rel_pos: torch.Tensor


class RolloutStorage(nn.Module):
    """Rollout storage as a single TensorDict buffer with batch size [T, N]."""

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_template: TensorDict,
        actions_shape,
        device="cpu",
        command_name: str | None = None,
        transition_cls: type[PpoTransition] = PpoTransition,
    ):
        super().__init__()
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.command_name = command_name
        self._float_dtype = torch.float32
        self._dones_dtype = torch.bool
        self._transition_cls = transition_cls

        obs_template = obs_template.to(self.device)
        self.data = TensorDict(
            {},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self._allocate_from_transition(
            obs_template=obs_template,
            actions_shape=actions_shape,
        )

        self.step = 0

    def _resolve_shape(self, spec_shape, actions_shape) -> tuple:
        if spec_shape is None:
            return tuple()
        resolved = []
        shape_tokens = getattr(self._transition_cls, "SHAPE_TOKENS", {})
        for dim in spec_shape:
            if dim == "A":
                resolved.extend(tuple(actions_shape))
            elif isinstance(dim, str) and dim in shape_tokens:
                resolved.append(int(shape_tokens[dim]))
            else:
                resolved.append(int(dim))
        return tuple(resolved)

    def _allocate_from_transition(
        self,
        *,
        obs_template: TensorDict,
        actions_shape,
    ) -> None:
        specs = getattr(self._transition_cls, "FIELD_SPECS", None)
        if not isinstance(specs, dict):
            raise ValueError(
                "Transition class must define FIELD_SPECS for allocation."
            )
        for name, spec in specs.items():
            if spec.get("kind") == "obs":
                leaf_keys = obs_template.keys(
                    include_nested=True, leaves_only=True
                )
                for key in leaf_keys:
                    value = obs_template.get(key)
                    if not torch.is_tensor(value):
                        continue
                    dtype = (
                        self._float_dtype
                        if torch.is_floating_point(value)
                        else value.dtype
                    )
                    key_tuple = key if isinstance(key, tuple) else (key,)
                    self.data.set(
                        ("obs",) + key_tuple,
                        torch.empty(
                            (self.num_transitions_per_env, self.num_envs)
                            + tuple(value.shape[1:]),
                            device=self.device,
                            dtype=dtype,
                        ),
                    )
                continue
            shape_spec = spec.get("shape")
            dtype = spec.get("dtype", self._float_dtype)
            resolved = self._resolve_shape(shape_spec, actions_shape)
            self.data.set(
                name,
                torch.empty(
                    (self.num_transitions_per_env, self.num_envs) + resolved,
                    device=self.device,
                    dtype=dtype,
                ),
            )

    def _to_storage_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError("Expected a tensor for RolloutStorage update.")
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if (
            torch.is_floating_point(tensor)
            and tensor.dtype != self._float_dtype
        ):
            tensor = tensor.to(dtype=self._float_dtype)
        return tensor

    def add(self, transition: PpoTransition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")
        if not isinstance(transition, self._transition_cls):
            raise TypeError(
                "Transition must match the RolloutStorage transition class."
            )
        if transition.batch_size is None or len(transition.batch_size) < 1:
            raise ValueError("Transition must have batch size [N].")
        if int(transition.batch_size[0]) != int(self.num_envs):
            raise ValueError(
                f"Transition batch size {transition.batch_size} "
                f"does not match num_envs={self.num_envs}."
            )

        td = transition.to_tensordict()
        td = td.apply(self._to_storage_tensor, inplace=False)
        if "dones" in td.keys():
            dones = td.get("dones")
            if torch.is_tensor(dones) and dones.dtype != self._dones_dtype:
                td.set("dones", dones.to(dtype=self._dones_dtype))
        self.data[self.step].update_(td)

        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self,
        last_values,
        gamma,
        lam,
        normalize_advantage: bool = False,
    ) -> None:
        advantage = 0
        rewards = self.data["rewards"]
        values = self.data["values"]
        dones = self.data["dones"]
        returns = self.data["returns"]
        advantages = self.data["advantages"]
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * gamma * next_values
                - values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]

        advantages.copy_(returns - values)
        if normalize_advantage:
            flat = advantages.view(-1)
            mean = flat.mean()
            std = flat.std().clamp_min(1.0e-8)
            advantages.copy_((advantages - mean) / std)

    @torch.no_grad()
    def normalize_advantages_global(
        self,
        *,
        accelerator=None,
        eps: float = 1.0e-8,
    ) -> None:
        """Global advantage normalization over the full rollout buffer.

        This normalizes `self.data["advantages"]` in-place using mean/std over
        all `[T * N]` samples. If `accelerator` is provided, the moments are
        aggregated across processes via `accelerator.reduce(..., reduction="sum")`.
        """
        advantages = self.data["advantages"]
        advantages_flat = advantages.view(-1).float()

        count = torch.tensor(
            [advantages_flat.numel()], device=self.device, dtype=torch.float32
        )
        sum_local = advantages_flat.sum()
        sqsum_local = (advantages_flat * advantages_flat).sum()
        if accelerator is not None and int(accelerator.num_processes) > 1:
            count_g = accelerator.reduce(count, reduction="sum")
            sum_g = accelerator.reduce(sum_local, reduction="sum")
            sqsum_g = accelerator.reduce(sqsum_local, reduction="sum")
        else:
            count_g = count
            sum_g = sum_local
            sqsum_g = sqsum_local

        mean = sum_g / count_g
        var = (sqsum_g / count_g) - mean * mean
        std = torch.sqrt(var.clamp_min(eps))
        advantages.copy_((advantages - mean) / std)

    @torch.no_grad()
    def normalize_advantages_global_by_move_mask(
        self,
        *,
        accelerator=None,
        eps: float = 1.0e-8,
        move_threshold: float = 0.5,
    ) -> None:
        """Global advantage normalization split by move vs static (velocity commands).

        Assumes:
        - `advantages`: [T, N, 1]
        - `velocity_commands`: [T, N, 4], where channel 0 is move_mask in {0,1}.
        """
        velocity_commands = self.data.get("velocity_commands", None)
        if velocity_commands is None:
            raise ValueError(
                "velocity_commands is required for global advantage normalization by move mask."
            )

        advantages = self.data["advantages"]
        advantages_flat = advantages.view(-1).float()

        vel_flat = velocity_commands.view(-1, int(velocity_commands.shape[-1]))
        move_mask = vel_flat[:, 0] > float(move_threshold)
        static_mask = ~move_mask

        count_all = torch.tensor(
            [advantages_flat.numel()], device=self.device, dtype=torch.float32
        )
        sum_all_local = advantages_flat.sum()
        sqsum_all_local = (advantages_flat * advantages_flat).sum()
        if accelerator is not None and int(accelerator.num_processes) > 1:
            count_all_g = accelerator.reduce(count_all, reduction="sum")
            sum_all_g = accelerator.reduce(sum_all_local, reduction="sum")
            sqsum_all_g = accelerator.reduce(sqsum_all_local, reduction="sum")
        else:
            count_all_g = count_all
            sum_all_g = sum_all_local
            sqsum_all_g = sqsum_all_local

        mean_all = sum_all_g / count_all_g
        var_all = (sqsum_all_g / count_all_g) - mean_all * mean_all
        std_all = torch.sqrt(var_all.clamp_min(eps))

        def _group_stats(
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not bool(mask.any().item()):
                return mean_all, std_all
            mask_f = mask.to(dtype=torch.float32)
            count_local = mask_f.sum()
            sum_local = (advantages_flat * mask_f).sum()
            sqsum_local = (advantages_flat * advantages_flat * mask_f).sum()
            if accelerator is not None and int(accelerator.num_processes) > 1:
                count_g = accelerator.reduce(count_local, reduction="sum")
                sum_g = accelerator.reduce(sum_local, reduction="sum")
                sqsum_g = accelerator.reduce(sqsum_local, reduction="sum")
            else:
                count_g = count_local
                sum_g = sum_local
                sqsum_g = sqsum_local
            if float(count_g.item()) <= 0.0:
                return mean_all, std_all
            mean = sum_g / count_g
            var = (sqsum_g / count_g) - mean * mean
            std = torch.sqrt(var.clamp_min(eps))
            return mean, std

        move_mean, move_std = _group_stats(move_mask)
        static_mean, static_std = _group_stats(static_mask)

        advantages_norm = advantages_flat.clone()
        if bool(move_mask.any().item()):
            advantages_norm[move_mask] = (
                advantages_flat[move_mask] - move_mean
            ) / move_std
        if bool(static_mask.any().item()):
            advantages_norm[static_mask] = (
                advantages_flat[static_mask] - static_mean
            ) / static_std

        self.data["advantages"].copy_(advantages_norm.view_as(advantages))

    @torch.no_grad()
    def normalize_advantages_global_by_command(
        self,
        *,
        command_name: str | None,
        accelerator=None,
        eps: float = 1.0e-8,
    ) -> None:
        """Dispatch global advantage normalization based on command type/name."""
        if (
            command_name == "base_velocity"
            and self.data.get("velocity_commands", None) is not None
        ):
            self.normalize_advantages_global_by_move_mask(
                accelerator=accelerator, eps=eps
            )
            return
        self.normalize_advantages_global(accelerator=accelerator, eps=eps)

    def iter_minibatches(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator[PpoTransition, None, None]:
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                f"RolloutStorage buffer not full: step={self.step}, "
                f"expected={self.num_transitions_per_env}. "
                "This would mix stale entries from a previous rollout."
            )
        batch_size = self.num_envs * self.num_transitions_per_env
        (
            effective_num_mini_batches,
            mini_batch_size,
        ) = self.resolve_mini_batch_partition(batch_size, num_mini_batches)

        indices = torch.randperm(
            batch_size,
            requires_grad=False,
            device=self.device,
        )[: effective_num_mini_batches * mini_batch_size]

        flat = self.data.flatten(0, 1)  # [T * N, ...]

        for _ in range(num_epochs):
            for i in range(effective_num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                batch = flat[batch_idx]
                yield self._transition_cls.from_tensordict(batch)

    @staticmethod
    def resolve_mini_batch_partition(
        batch_size: int,
        num_mini_batches: int,
    ) -> tuple[int, int]:
        if batch_size <= 0:
            raise RuntimeError(
                "RolloutStorage minibatch partition requires batch_size > 0."
            )
        effective_num_mini_batches = max(
            1, min(int(num_mini_batches), int(batch_size))
        )
        mini_batch_size = max(1, batch_size // effective_num_mini_batches)
        return effective_num_mini_batches, mini_batch_size
