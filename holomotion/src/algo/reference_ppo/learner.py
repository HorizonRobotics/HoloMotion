"""Simulator PPO control for the shared small motion-tracking benchmark.

Actor, critic, observations and scoring are shared with the model-based arm.
This module changes the policy update, rather than the physical task.
"""

import torch
from torch import nn
from torch.distributions import Normal

from holomotion.src.algo.supermimic.networks import Critic, Policy
from holomotion.src.env.reference_tracking.mdp import (
    CRITIC_EXTRA,
    INV_DIM,
    normalize_xi,
)


def generalized_advantage(
    rewards, values, next_values, terminated, ended, gamma=0.99, lam=0.95
):
    """Bootstrap time limits from their physical next state, and cut GAE there."""
    if not (
        rewards.shape
        == values.shape
        == next_values.shape
        == terminated.shape
        == ended.shape
    ):
        raise ValueError(
            "PPO rollout arrays must have identical time/environment dimensions"
        )
    result = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in range(len(rewards) - 1, -1, -1):
        delta = (
            rewards[step]
            + gamma * next_values[step] * (~terminated[step])
            - values[step]
        )
        carry = delta + gamma * lam * (~ended[step]) * carry
        result[step] = carry
    return result, result + values


def clipped_losses(
    log_probability,
    old_log_probability,
    advantage,
    value,
    old_value,
    target,
    entropy,
    *,
    clip=0.2,
    entropy_coefficient=0.005,
):
    ratio = (log_probability - old_log_probability).exp()
    policy = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()
    clipped_value = old_value + (value - old_value).clamp(-clip, clip)
    value_loss = torch.maximum(
        (value - target).square(), (clipped_value - target).square()
    ).mean()
    return (
        policy + value_loss - entropy_coefficient * entropy.mean(),
        policy,
        value_loss,
    )


class SimulatorPPOLearner(nn.Module):
    def __init__(
        self,
        mdp,
        motion,
        *,
        device,
        hist=10,
        goal_frames=10,
        initial=None,
        learning_rate=0.001,
        epochs=5,
        minibatches=4,
        microbatch=2048,
    ):
        super().__init__()
        self.mdp, self.motion = mdp, motion
        self.pi = Policy(mdp.obs_dim, 73 * goal_frames, hist=hist).to(device)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + 73 * goal_frames).to(
            device
        )
        self.log_std = nn.Parameter(torch.zeros(29, device=device))
        if initial is not None:
            self.pi.load_state_dict(initial["pi"], strict=True)
            self.critic.load_state_dict(initial["critic"], strict=True)
        self.goal_frames, self.hist = goal_frames, hist
        self.epochs, self.minibatches, self.microbatch = (
            epochs,
            minibatches,
            microbatch,
        )
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def critic_input(self, state, frame, context):
        return torch.cat(
            (
                self.mdp.critic_features(state, frame, normalize_xi(context)),
                self.motion.goal(frame, self.goal_frames),
            ),
            -1,
        )

    def distribution(self, history, goal):
        return Normal(self.pi(history, goal), self.log_std.exp())

    def update(self, rollout):
        advantage, target = generalized_advantage(
            rollout["reward"],
            rollout["value"],
            rollout["next_value"],
            rollout["terminated"],
            rollout["ended"],
        )
        advantage = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp_min(1e-8)
        arrays = {name: value.flatten(0, 1) for name, value in rollout.items()}
        arrays.update(advantage=advantage.flatten(), target=target.flatten())
        count = len(arrays["target"])
        if count % self.minibatches:
            raise ValueError("PPO minibatches must exactly cover the rollout")
        metrics = []
        for _ in range(self.epochs):
            for ids in torch.randperm(count, device=target.device).chunk(
                self.minibatches
            ):
                self.optimizer.zero_grad(set_to_none=True)
                kl_sum, loss_sum = 0.0, 0.0
                for part in ids.split(self.microbatch):
                    distribution = self.distribution(
                        arrays["history"][part], arrays["goal"][part]
                    )
                    log_probability = distribution.log_prob(
                        arrays["sampled_action"][part]
                    ).sum(-1)
                    value = self.critic(arrays["critic_input"][part])
                    loss, _, _ = clipped_losses(
                        log_probability,
                        arrays["log_probability"][part],
                        arrays["advantage"][part],
                        value,
                        arrays["value"][part],
                        arrays["target"][part],
                        distribution.entropy().sum(-1),
                    )
                    weight = len(part) / len(ids)
                    (loss * weight).backward()
                    with torch.no_grad():
                        old_std = arrays["std"][part]
                        old_mean = arrays["mean"][part]
                        kl = (
                            (
                                torch.log(distribution.scale / old_std)
                                + (
                                    old_std.square()
                                    + (old_mean - distribution.loc).square()
                                )
                                / (2 * distribution.scale.square())
                                - 0.5
                            )
                            .sum(-1)
                            .mean()
                        )
                        kl_sum += float(kl) * weight
                        loss_sum += float(loss) * weight
                if kl_sum > 0.02:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0 < kl_sum < 0.005:
                    self.learning_rate = min(0.01, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
                gradient = nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise FloatingPointError(
                        "PPO produced a nonfinite gradient"
                    )
                self.optimizer.step()
                if not torch.stack(
                    [
                        torch.isfinite(parameter).all()
                        for parameter in self.parameters()
                    ]
                ).all():
                    raise FloatingPointError(
                        "PPO produced a nonfinite parameter"
                    )
                metrics.append(
                    dict(
                        loss=loss_sum, kl=kl_sum, gradient_norm=float(gradient)
                    )
                )
        return {
            key: sum(value[key] for value in metrics) / len(metrics)
            for key in metrics[0]
        } | {
            "learning_rate": self.learning_rate,
            "optimizer_steps": len(metrics),
            "minimum_std": float(self.log_std.detach().exp().min()),
            "maximum_std": float(self.log_std.detach().exp().max()),
        }
