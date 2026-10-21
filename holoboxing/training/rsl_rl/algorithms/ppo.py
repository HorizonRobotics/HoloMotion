# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""RoboStriker paired PPO/GAE/AMP recipe for joint C policies."""

import math
from contextlib import nullcontext

import torch
from torch import nn

from holoboxing.training.rsl_rl.modules.exploration import (
    GenericExplorationHistory,
)
from holoboxing.training.rsl_rl.storage.rollout_storage import (
    RolloutStorage,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    AGENTS,
    CRITIC_DIM,
    ROLES,
)


def require_finite(value, context):
    checks = {}

    def describe(item, path="value"):
        if torch.is_tensor(item):
            invalid = ~torch.isfinite(item)
            if bool(invalid.any()):
                indices = invalid.nonzero()[:8].tolist()
                return [
                    f"{path}: shape={tuple(item.shape)}, "
                    f"invalid={int(invalid.sum())}, first_indices={indices}"
                ]
        elif isinstance(item, dict):
            return [
                detail
                for key, child in item.items()
                for detail in describe(child, f"{path}[{key!r}]")
            ]
        elif isinstance(item, (list, tuple)):
            return [
                detail
                for index, child in enumerate(item)
                for detail in describe(child, f"{path}[{index}]")
            ]
        elif isinstance(item, float) and not math.isfinite(item):
            return [f"{path}: {item!r}"]
        return []

    def collect(item):
        if torch.is_tensor(item):
            checks.setdefault(item.device, []).append(
                torch.isfinite(item).all()
            )
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise FloatingPointError(
                f"Nonfinite {context}: {'; '.join(describe(value)[:8])}"
            )

    collect(value)
    # Keep every check at the same update boundary, with one host wait per device.
    for flags in checks.values():
        valid = flags[0] if len(flags) == 1 else torch.stack(flags).all()
        if not bool(valid):
            raise FloatingPointError(
                f"Nonfinite {context}: {'; '.join(describe(value)[:8])}"
            )


def gaussian_kl(old_mean, old_std, mean, std):
    # No +1e-5 inside log: unchanged policies must have zero KL.
    return (
        torch.log(std / old_std)
        + (old_std.square() + (old_mean - mean).square())
        / (2.0 * std.square())
        - 0.5
    ).sum(-1)


def factors(law):
    return law.kl_factors() if hasattr(law, "kl_factors") else law


class PPO:
    def __init__(self, policy, amp, config, device, *, active_roles=ROLES):
        self.policy, self.amp, self.config = policy, amp, config
        self.active_roles = tuple(active_roles)
        if self.active_roles not in (ROLES, ("ego",)):
            raise ValueError("Expected paired learning or one ego learner")
        self.device = device
        self.optimizers = {
            role: torch.optim.Adam(
                policy.role_parameters(role), lr=config.learning_rate
            )
            for role in ROLES
        }
        self.learning_rates = dict.fromkeys(ROLES, config.learning_rate)
        self.histories = {
            role: GenericExplorationHistory(
                config.num_envs, device, config.actor_dim, policy.action_dim
            )
            for role in ROLES
        }
        shape = [policy.actor_storage_dim]
        self.storage = RolloutStorage(
            config.num_envs,
            config.num_steps_per_env,
            shape,
            [CRITIC_DIM],
            [policy.action_dim],
            device=device,
            active_roles=self.active_roles,
        )
        self.transition = RolloutStorage.Transition()
        self.last_parity = {}
        self.optimizer_steps = dict.fromkeys(ROLES, 0)

    def begin_rollout(self):
        if self.storage.step:
            raise RuntimeError("Previous rollout has not been updated")
        for history in self.histories.values():
            history.begin_rollout()

    def reset_exploration(self, done=None):
        for history in self.histories.values():
            history.reset(done)

    @torch.no_grad()
    def act(self, observations, *, opponent_actions=None):
        actions = {} if opponent_actions is None else dict(opponent_actions)
        for role, agent in zip(ROLES, AGENTS, strict=True):
            if role not in self.active_roles:
                if agent not in actions:
                    raise ValueError(
                        "Inactive role requires an external action"
                    )
                continue
            if agent in actions:
                raise ValueError("Cannot override an active learner action")
            actor, critic = observations[agent]
            packed = (
                self.histories[role].pack(actor)
                if self.config.exploration_rho
                else actor
            )
            law = self.policy.distribution(role, packed)
            action = law.sample()
            logp = law.log_prob(action).sum(-1)
            distribution_factors = factors(law)
            base = self.policy.base_distributions[role]
            self.histories[role].record(actor, action, base.mean, base.stddev)
            values = {
                "actions": action,
                "values": self.policy.value(role, critic),
                "actions_log_prob": logp,
                "action_mean": distribution_factors.mean,
                "action_sigma": distribution_factors.stddev,
                "observations": packed,
                "privileged_observations": critic,
            }
            require_finite(values, f"{role} rollout")
            for name, value in values.items():
                setattr(self.transition, f"{name}_{role}", value.detach())
            actions[agent] = (
                self.policy.execute(role, action)
                if hasattr(self.policy, "execute")
                else action
            )
        return actions

    @torch.no_grad()
    def process_env_step(self, rewards, done, infos):
        require_finite({"rewards": rewards, "infos": infos}, "transition")
        if done.dtype != torch.bool or done.shape != (self.config.num_envs,):
            raise ValueError("Expected paired boolean done mask")
        invalid = infos.get("invalid_envs", torch.zeros_like(done))
        if invalid.dtype != torch.bool or invalid.shape != done.shape:
            raise ValueError("Expected paired boolean invalid mask")
        if bool((invalid & ~done).any()):
            raise ValueError("Invalid worlds must be reset before storage")
        self.transition.valid = ~invalid
        for role, agent in zip(ROLES, AGENTS, strict=True):
            if role not in self.active_roles:
                continue
            setattr(self.transition, f"rewards_{role}", rewards[agent])
            setattr(self.transition, f"dones_{role}", done)
            setattr(
                self.transition, f"amp_obs_{role}", infos[f"amp_obs_{role}"]
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.reset_exploration(done)

    @torch.no_grad()
    def compute_returns(self, observations):
        if self.storage.step != self.config.num_steps_per_env:
            raise ValueError("Cannot update a partial rollout")
        for role, agent in zip(ROLES, AGENTS, strict=True):
            if role not in self.active_roles:
                continue
            value = self.policy.value(role, observations[agent][1])
            self.storage.compute_returns(
                role, value, self.config.gamma, self.config.lam
            )

    @torch.no_grad()
    def check_collection_probabilities(self):
        """Fail before either optimizer runs; never rewrite saved probabilities."""
        self.last_parity = {}
        valid_indices = (
            self.storage.valid.flatten().nonzero().flatten()
            if self.storage.has_invalid
            else None
        )
        for role in self.active_roles:
            obs = getattr(self.storage, f"observations_{role}").flatten(0, 1)
            actions = getattr(self.storage, f"actions_{role}").flatten(0, 1)
            old_logp = getattr(
                self.storage, f"actions_log_prob_{role}"
            ).flatten()
            old_mean = getattr(self.storage, f"mu_{role}").flatten(0, 1)
            old_std = getattr(self.storage, f"sigma_{role}").flatten(0, 1)
            count = len(obs) if valid_indices is None else len(valid_indices)
            if not count:
                raise FloatingPointError("No valid transitions for PPO")
            maximum, kl_sum = 0.0, 0.0
            for start in range(0, count, self.config.actor_forward_rows):
                sl = slice(start, start + self.config.actor_forward_rows)
                if valid_indices is not None:
                    sl = valid_indices[sl]
                law = self.policy.distribution(role, obs[sl])
                logp = law.log_prob(actions[sl]).sum(-1)
                factor = factors(law)
                delta = logp - old_logp[sl]
                kl = gaussian_kl(
                    old_mean[sl], old_std[sl], factor.mean, factor.stddev
                )
                require_finite((delta, kl), "pre-update likelihood")
                maximum = max(maximum, delta.abs().max().item())
                kl_sum += kl.double().sum().item()
            self.last_parity[role] = {
                "max_abs_log_ratio": maximum,
                "mean_kl": kl_sum / count,
                "rows": count,
            }
        if any(
            record["max_abs_log_ratio"] > 0.1 or abs(record["mean_kl"]) > 0.001
            for record in self.last_parity.values()
        ):
            raise FloatingPointError(
                f"Pre-update probability mismatch: {self.last_parity}"
            )

    def _role_loss(self, role, batch):
        (
            obs,
            critic,
            action,
            old_value,
            advantage,
            returns,
            old_logp,
            old_mu,
            old_std,
        ) = batch
        require_finite(batch, f"{role} PPO batch")
        law = self.policy.distribution(role, obs)
        logp = law.log_prob(action).sum(-1)
        factor = factors(law)
        value = self.policy.value(role, critic)
        entropy = law.entropy().sum(-1).mean()
        cfg = self.config
        with torch.no_grad():
            kl = gaussian_kl(
                old_mu, old_std, factor.mean, factor.stddev
            ).mean()
            require_finite(kl, f"{role} adaptive KL")
            lr = self.learning_rates[role]
            if kl > cfg.desired_kl * 2:
                lr = max(cfg.min_learning_rate, lr / 1.5)
            elif 0 < kl < cfg.desired_kl / 2:
                lr = min(cfg.max_learning_rate, lr * 1.5)
            self.learning_rates[role] = lr
            for group in self.optimizers[role].param_groups:
                group["lr"] = lr
        log_ratio = logp - old_logp.squeeze(-1)
        ratio = torch.exp(log_ratio)
        require_finite(ratio, f"{role} probability ratio")
        zero_ratios = int((ratio == 0).sum().item())
        if zero_ratios:
            raise FloatingPointError(
                f"{role} probability ratio underflow: {zero_ratios}/{ratio.numel()}, "
                f"KL={kl.item()}, min_log_ratio={log_ratio.min().item()}"
            )
        surrogate = -advantage.squeeze(-1) * ratio
        clipped = -advantage.squeeze(-1) * ratio.clamp(
            1 - cfg.clip_param, 1 + cfg.clip_param
        )
        surrogate_loss = torch.maximum(surrogate, clipped).mean()
        clipped_value = old_value + (value - old_value).clamp(
            -cfg.clip_param, cfg.clip_param
        )
        value_loss = torch.maximum(
            (value - returns).square(), (clipped_value - returns).square()
        ).mean()
        loss = (
            surrogate_loss
            + cfg.value_loss_coef * value_loss
            - cfg.entropy_coef * entropy
        )
        require_finite((ratio, loss), f"{role} PPO loss")
        return loss, {
            "surrogate": surrogate_loss.item(),
            "value": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "max_minibatch_kl": kl.item(),
            "clip_fraction": ((ratio - 1).abs() > cfg.clip_param)
            .float()
            .mean()
            .item(),
            "zero_ratio_count": zero_ratios,
            "max_abs_log_ratio": log_ratio.detach().abs().max().item(),
        }

    def update(self):
        self.check_collection_probabilities()
        metrics = {}
        cfg = self.config
        for batch in self.storage.mini_batch_generator(
            cfg.num_mini_batches, cfg.num_learning_epochs
        ):
            # Filtered minibatches no longer fit the fixed CUDA graph shape.
            with (
                nullcontext()
                if self.storage.has_invalid
                else self.policy.actor_graph_batch()
            ):
                losses = []
                for role, role_batch in zip(
                    self.active_roles, batch, strict=True
                ):
                    loss, record = self._role_loss(role, role_batch)
                    losses.append(loss)
                    self.optimizers[role].zero_grad(set_to_none=True)
                    for name, value in record.items():
                        key = f"{role}/{name}"
                        if name.startswith("max_"):
                            metrics[key] = max(metrics.get(key, value), value)
                        else:
                            metrics[key] = metrics.get(key, 0.0) + value
                sum(losses).backward()
            # Validate BOTH gradients before either optimizer changes parameters.
            for role in self.active_roles:
                parameters = self.policy.role_parameters(role)
                require_finite(
                    [p.grad for p in parameters], f"{role} gradients"
                )
                nn.utils.clip_grad_norm_(
                    parameters, cfg.max_grad_norm, error_if_nonfinite=True
                )
            for role in self.active_roles:
                self.optimizers[role].step()
                self.optimizer_steps[role] += 1
                require_finite(
                    (
                        self.policy.role_parameters(role),
                        self.optimizers[role].state,
                    ),
                    f"{role} optimizer state",
                )
        count = cfg.num_mini_batches * cfg.num_learning_epochs
        metrics = {
            name: value
            if name.split("/")[-1].startswith("max_")
            or name.endswith("zero_ratio_count")
            else value / count
            for name, value in metrics.items()
        }
        metrics.update(
            self.amp.update(
                self.storage.valid_flattened("amp_obs_ego"),
                self.storage.valid_flattened("amp_obs_opponent")
                if "opponent" in self.active_roles
                else None,
                count,
            )
        )
        self.storage.clear()
        return metrics
