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
            self.valid = None
            for role in ROLES:
                for name in (
                    "observations",
                    "privileged_observations",
                    "actions",
                    "rewards",
                    "dones",
                    "values",
                    "actions_log_prob",
                    "action_mean",
                    "action_sigma",
                    "amp_obs",
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
        *,
        active_roles=ROLES,
    ):
        self.device = device
        self.active_roles = tuple(active_roles)
        if self.active_roles not in (ROLES, ("ego",)):
            raise ValueError("Expected paired learning or one ego learner")
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0
        self.has_invalid = False
        self.valid = torch.ones(
            num_transitions_per_env, num_envs, dtype=torch.bool, device=device
        )
        for role in self.active_roles:
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
                    num_transitions_per_env,
                    num_envs,
                    *shape,
                    device=device,
                    dtype=torch.uint8 if name == "dones" else torch.float32,
                )
                setattr(self, f"{name}_{role}", value)
            setattr(self, f"amp_obs_{role}", None)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow; update before collecting again"
            )
        if transition.valid is None:
            self.valid[self.step].fill_(True)
        else:
            self.valid[self.step].copy_(transition.valid)
            self.has_invalid |= not bool(transition.valid.all())
        for role in self.active_roles:
            for name in (
                "observations",
                "privileged_observations",
                "actions",
                "rewards",
                "dones",
                "values",
                "actions_log_prob",
                "mu",
                "sigma",
            ):
                source = {"mu": "action_mean", "sigma": "action_sigma"}.get(
                    name, name
                )
                value = getattr(transition, f"{source}_{role}")
                if name in ("rewards", "dones", "actions_log_prob"):
                    value = value.view(-1, 1)
                getattr(self, f"{name}_{role}")[self.step].copy_(value)
            amp = getattr(transition, f"amp_obs_{role}")
            buffer = getattr(self, f"amp_obs_{role}")
            if buffer is None:
                buffer = torch.zeros(
                    self.num_transitions_per_env,
                    self.num_envs,
                    *amp.shape[1:],
                    device=self.device,
                )
                setattr(self, f"amp_obs_{role}", buffer)
            buffer[self.step].copy_(amp)
        self.step += 1

    def clear(self):
        self.step = 0
        self.has_invalid = False

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
            # Drop this sample and stop its GAE from reaching the valid prefix.
            advantage = torch.where(self.valid[step, :, None], advantage, 0.0)
            returns[step] = advantage + values[step]
        advantages = returns - values
        selected = advantages[self.valid]
        if len(selected) < 2:
            raise FloatingPointError("Not enough valid transitions for PPO")
        advantages = (advantages - selected.mean()) / (selected.std() + 1e-8)
        advantages[~self.valid] = 0
        setattr(self, f"advantages_{role}", advantages)

    def valid_flattened(self, name):
        values = getattr(self, name).flatten(0, 1)
        return values[self.valid.flatten()] if self.has_invalid else values

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        valid_indices = (
            self.valid.flatten().nonzero().flatten()
            if self.has_invalid
            else None
        )
        batch_size = (
            len(valid_indices)
            if self.has_invalid
            else self.num_envs * self.num_transitions_per_env
        )
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size < 1:
            raise FloatingPointError(
                "Not enough valid transitions for PPO batches"
            )
        # Keep the baseline's single permutation, shared across roles and epochs.
        size = num_mini_batches * mini_batch_size
        indices = torch.randperm(
            batch_size if self.has_invalid else size,
            requires_grad=False,
            device=self.device,
        )
        if self.has_invalid:
            indices = valid_indices[indices]
        selections = indices.tensor_split(num_mini_batches)
        columns = (
            "observations",
            "privileged_observations",
            "actions",
            "values",
            "advantages",
            "returns",
            "actions_log_prob",
            "mu",
            "sigma",
        )
        batches = [
            tuple(
                getattr(self, f"{name}_{role}").flatten(0, 1)
                for name in columns
            )
            for role in self.active_roles
        ]
        for _ in range(num_epochs):
            for selected in selections:
                yield tuple(
                    tuple(value[selected] for value in batch)
                    for batch in batches
                )
