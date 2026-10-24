# Adapted from zita-ch/supermimic_vibed, a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Logical-batch imagination with bounded actor activation memory."""

import torch
from torch import nn

from holomotion.src.env.reference_tracking.mdp import (
    normalize_xi,
    unflatten_state,
)


def set_requires_grad(module, flag):
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def policy_update_microbatched(self):
    batch = self.args.pi_batch
    width = getattr(self.args, "pi_microbatch", batch)
    start, history = self.sample_valid_starts(batch)
    # Draw one logical population in the original time-major RNG order.
    # Execution chunks then reuse it without changing sampled noise.
    with torch.no_grad():
        prepared_history = self.mdp.add_noise(
            self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        )
        prepared_noise = [
            self.mdp.add_noise(
                torch.zeros(batch, self.mdp.obs_dim, device=self.dev)
            )
            for _ in range(self.HZ)
        ]
    self.opt_pi.zero_grad(set_to_none=True)
    set_requires_grad(self.wm, False)
    set_requires_grad(self.critic, False)
    inputs, reward_batches, done_batches = [], [], []
    total_loss, total_reward, total_survival = 0.0, 0.0, 0.0
    try:
        for first in range(0, batch, width):
            last = min(first + width, batch)
            size = last - first
            s = {key: value[first:last] for key, value in start.items()}
            state = unflatten_state(s["s"])
            initial_state = state
            context, frame, previous = s["xi"], s["f"], s["a_prev"]
            normalized = normalize_xi(context)
            goal = initial_goal = self.motion.goal(frame, self.K)
            observations = prepared_history[first:last]
            states, actions, previous_actions, frames, goals = (
                [],
                [],
                [],
                [],
                [],
            )
            for t in range(self.HZ):
                previous_actions.append(previous)
                state, previous, observations, frame, goal = self.imag_step(
                    state,
                    observations,
                    previous,
                    frame,
                    context,
                    normalized,
                    goal,
                    prepared_noise[t][first:last],
                )
                states.append(state)
                actions.append(previous)
                frames.append(frame)
                goals.append(goal)

            def flat(values):
                return torch.stack(values).flatten(0, 1)

            all_states = tuple(
                flat([value[i] for value in states]) for i in range(6)
            )
            reward, done = self.reward_term(
                all_states,
                flat(frames),
                flat(actions),
                flat(previous_actions),
            )
            reward, done = (
                reward.reshape(self.HZ, size),
                done.reshape(self.HZ, size),
            )
            alive = torch.cumprod(1 - done, 0)
            survival = torch.cat(
                (torch.ones(1, size, device=self.dev), alive[:-1]), 0
            )
            discount = (
                self.args.gamma
                ** torch.arange(self.HZ, device=self.dev, dtype=reward.dtype)[
                    :, None
                ]
            )
            objective = (discount * survival * reward).sum(0)
            if not self.args.no_critic:
                objective = objective + self.args.gamma**self.HZ * alive[
                    -1
                ] * self.critic(
                    self._critic_input(state, frame, normalized, goal)
                )
            loss = -objective.mean() * (size / batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite imagined objective")
            loss.backward()
            total_loss += float(loss.detach())
            total_reward += float((survival * reward).detach().sum()) / batch
            total_survival += float(alive[-1].detach().sum()) / batch
            if not self.args.no_critic:
                with torch.no_grad():
                    critic_states = tuple(
                        torch.cat(
                            (
                                initial_state[i][None],
                                torch.stack(
                                    [value[i].detach() for value in states]
                                ),
                            ),
                            0,
                        ).flatten(0, 1)
                        for i in range(6)
                    )
                    critic_frames = torch.cat(
                        (s["f"][None], torch.stack(frames)), 0
                    ).flatten()
                    critic_goal = torch.cat(
                        (initial_goal[None], torch.stack(goals)), 0
                    ).flatten(0, 1)
                    critic_context = (
                        normalized[None]
                        .expand(self.HZ + 1, -1, -1)
                        .flatten(0, 1)
                    )
                    inputs.append(
                        self._critic_input(
                            critic_states,
                            critic_frames,
                            critic_context,
                            critic_goal,
                        ).reshape(self.HZ + 1, size, -1)
                    )
                    reward_batches.append(reward.detach())
                    done_batches.append(done.detach())
        gradient = nn.utils.clip_grad_norm_(
            self.pi.parameters(), 1.0, error_if_nonfinite=True
        )
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        self.opt_pi.step()
    finally:
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
    value_loss = torch.zeros((), device=self.dev)
    if not self.args.no_critic:
        x = torch.cat(inputs, 1)
        rewards, dones = (
            torch.cat(reward_batches, 1),
            torch.cat(done_batches, 1),
        )
        self.critic.norm.update(x)
        with torch.no_grad():
            values = self.critic(x)
            carry, targets = values[-1], []
            for t in range(self.HZ - 1, -1, -1):
                carry = rewards[t] + self.args.gamma * (1 - dones[t]) * (
                    (1 - self.args.lam) * values[t + 1] + self.args.lam * carry
                )
                targets.append(carry)
            target = torch.stack(targets[::-1]).flatten()
        x = x[:-1].flatten(0, 1)
        if len(target) % self.args.critic_minibatches:
            raise ValueError("Critic minibatches must cover the logical batch")
        for _ in range(self.args.critic_epochs):
            for ids in torch.randperm(len(target), device=self.dev).chunk(
                self.args.critic_minibatches
            ):
                value_loss = (
                    (self.critic(x[ids]) - target[ids]).square().mean()
                )
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 1.0, error_if_nonfinite=True
                )
                self.opt_v.step()

    def scalar(value):
        return torch.tensor(value, device=self.dev)

    return (
        scalar(total_loss),
        value_loss.detach(),
        scalar(total_reward / self.HZ / 0.1),
        gradient.detach(),
        scalar(total_survival),
    )
