from __future__ import annotations
import math

import torch
from torch.distributions import Normal


def validate_rho(value):
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError("Correlation must be finite in [0, 1)")
    if value and not 0 < torch.tensor(value, dtype=torch.float32).item() < 1:
        raise ValueError("Correlation must remain inside (0, 1) in float32")
    return value


def unpack_context(packed, actor_dim, action_dim):
    if (
        packed.ndim != 2
        or packed.shape[1] != 2 * actor_dim + action_dim + 1
        or not torch.isfinite(packed).all()
    ):
        raise ValueError("Invalid complete conditional exploration context")
    current, previous, value, mode = torch.split(
        packed, [actor_dim, actor_dim, action_dim, 1], dim=-1
    )
    if not ((mode == 0) | (mode == 1) | (mode == 2)).all():
        raise ValueError("Unknown exploration mode")
    return current, previous, value, mode


def conditional_distribution(base, previous_mean, value, mode, rho):
    offset = torch.where(
        mode == 2, rho * (value - previous_mean), rho * base.stddev * value
    )
    return Normal(
        torch.where(mode == 0, base.mean, base.mean + offset),
        torch.where(
            mode == 0, base.stddev, base.stddev * math.sqrt(1 - rho * rho)
        ),
    )


class HorizonNormal:
    """Original five-offset AR law for either supported action encoding."""

    def __init__(self, base, rho):
        self.rho = validate_rho(rho)
        if not rho or base.mean.shape[-1] != 225:
            raise ValueError(
                "Horizon distribution requires 225D and positive rho"
            )
        self.base, self.factor = base, math.sqrt(1 - rho * rho)
        self.scored_factors = None

    @property
    def mean(self):
        return self.base.mean

    @property
    def stddev(self):
        return self.base.stddev

    def _offsets(self, value):
        return value.reshape(*value.shape[:-1], 5, value.shape[-1] // 5)

    @torch.no_grad()
    def sample(self):
        noise = self._offsets(
            Normal(
                torch.zeros_like(self.mean), torch.ones_like(self.stddev)
            ).sample()
        )
        values = [noise[..., 0, :]]
        for offset in range(1, 5):
            values.append(
                self.rho * values[-1] + self.factor * noise[..., offset, :]
            )
        return self.mean + self.stddev * torch.stack(values, -2).reshape_as(
            self.mean
        )

    def factors(self, action):
        if action.shape != self.mean.shape or not torch.isfinite(action).all():
            raise ValueError("Invalid realized action")
        mean, std, value = map(self._offsets, (self.mean, self.stddev, action))
        conditional_mean = torch.cat(
            (
                mean[..., :1, :],
                mean[..., 1:, :]
                + self.rho
                * std[..., 1:, :]
                / std[..., :-1, :]
                * (value[..., :-1, :] - mean[..., :-1, :]),
            ),
            -2,
        )
        conditional_std = torch.cat(
            (std[..., :1, :], std[..., 1:, :] * self.factor), -2
        )
        return Normal(
            conditional_mean.reshape_as(self.mean),
            conditional_std.reshape_as(self.stddev),
        )

    def log_prob(self, action):
        self.scored_factors = self.factors(action)
        return self.scored_factors.log_prob(action)

    def entropy(self):
        entropy = self._offsets(self.base.entropy())
        return torch.cat(
            (entropy[..., :1, :], entropy[..., 1:, :] + math.log(self.factor)),
            -2,
        ).reshape_as(self.mean)

    def kl_factors(self):
        if self.scored_factors is None:
            raise RuntimeError(
                "Score the realized action before reading conditional factors"
            )
        return self.scored_factors


class GenericExplorationHistory:
    """Collection snapshots; mode 1 saves a residual, mode 2 a realized action."""

    def __init__(self, num_envs, device, actor_dim, action_dim):
        if action_dim not in (32, 225) or actor_dim < 1 or num_envs < 1:
            raise ValueError("Invalid history dimensions")
        self.previous_obs = torch.zeros(num_envs, actor_dim, device=device)
        self.previous_action = torch.zeros(num_envs, action_dim, device=device)
        self.last_noise = torch.zeros_like(self.previous_action)
        self.mode = torch.zeros(num_envs, 1, device=device)

    @torch.no_grad()
    def reset(self, dones=None):
        if dones is not None and (
            dones.shape != (len(self.mode),) or dones.dtype != torch.bool
        ):
            raise ValueError("Reset requires one boolean per environment")
        selected = slice(None) if dones is None else dones
        for value in (
            self.previous_obs,
            self.previous_action,
            self.last_noise,
            self.mode,
        ):
            value[selected] = 0

    @torch.no_grad()
    def begin_rollout(self):
        self.mode[self.mode != 0] = 1

    def pack(self, observations):
        if observations.shape != self.previous_obs.shape:
            raise ValueError("History batch differs from current observations")
        value = torch.where(
            self.mode == 1, self.last_noise, self.previous_action
        )
        return torch.cat(
            (observations, self.previous_obs, value, self.mode), -1
        )

    @torch.no_grad()
    def record(self, observations, action, base_mean, base_std):
        if (
            observations.shape != self.previous_obs.shape
            or action.shape != self.previous_action.shape
            or base_mean.shape != action.shape
            or base_std.shape != action.shape
            or not all(
                torch.isfinite(v).all()
                for v in (observations, action, base_mean, base_std)
            )
            or not (base_std > 0).all()
        ):
            raise ValueError("Invalid actual collection state")
        self.previous_obs.copy_(observations)
        self.previous_action.copy_(action)
        self.last_noise.copy_((action - base_mean) / base_std)
        self.mode.fill_(2)

    def state_dict(self):
        return {
            key: value.detach().clone() for key, value in vars(self).items()
        }
