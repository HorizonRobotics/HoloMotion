# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Paired feedforward PPO rollouts and AMP transitions."""

import torch

from holoboxing.training.tasks.combat.config.sandbag import ROLES


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.clear()

        def clear(self):
            for role in ROLES:
                for name in (
                    "observations", "privileged_observations", "actions", "rewards",
                    "dones", "values", "actions_log_prob", "action_mean",
                    "action_sigma", "amp_obs",
                ):
                    setattr(self, f"{name}_{role}", None)

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0
        for role in ROLES:
            for name, shape in (
                ("observations", obs_shape),
                ("privileged_observations", privileged_obs_shape),
                ("actions", actions_shape),
                ("mu", actions_shape),
                ("sigma", actions_shape),
                ("rewards", [1]),
                ("dones", [1]),
                ("values", [1]),
                ("actions_log_prob", [1]),
                ("returns", [1]),
                ("advantages", [1]),
            ):
                value = torch.zeros(
                    num_transitions_per_env, num_envs, *shape, device=device,
                    dtype=torch.uint8 if name == "dones" else torch.float32,
                )
                setattr(self, f"{name}_{role}", value)
            setattr(self, f"amp_obs_{role}", None)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow; update before collecting again")
        for role in ROLES:
            for name in (
                "observations", "privileged_observations", "actions", "rewards",
                "dones", "values", "actions_log_prob", "mu", "sigma",
            ):
                source = {"mu": "action_mean", "sigma": "action_sigma"}.get(name, name)
                value = getattr(transition, f"{source}_{role}")
                if name in ("rewards", "dones", "actions_log_prob"):
                    value = value.view(-1, 1)
                getattr(self, f"{name}_{role}")[self.step].copy_(value)
            amp = getattr(transition, f"amp_obs_{role}")
            buffer = getattr(self, f"amp_obs_{role}")
            if buffer is None:
                buffer = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *amp.shape[1:],
                    device=self.device,
                )
                setattr(self, f"amp_obs_{role}", buffer)
            buffer[self.step].copy_(amp)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, role, last_values, gamma, lam):
        values = getattr(self, f"values_{role}")
        rewards = getattr(self, f"rewards_{role}")
        dones = getattr(self, f"dones_{role}")
        returns = getattr(self, f"returns_{role}")
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.num_transitions_per_env - 1
                else values[step + 1]
            )
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * gamma * next_values
                - values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        setattr(self, f"advantages_{role}", advantages)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # Keep the baseline's single permutation, shared across roles and epochs.
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device,
        )
        columns = (
            "observations", "privileged_observations", "actions", "values",
            "advantages", "returns", "actions_log_prob", "mu", "sigma",
        )
        batches = [
            tuple(getattr(self, f"{name}_{role}").flatten(0, 1) for name in columns)
            for role in ROLES
        ]
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                selected = indices[start:start + mini_batch_size]
                yield tuple(
                    tuple(value[selected] for value in batch) for batch in batches
                )
