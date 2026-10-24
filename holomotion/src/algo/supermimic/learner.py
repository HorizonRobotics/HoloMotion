# Adapted from zita-ch/supermimic_vibed, a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Pinned reference world-model, actor, and critic updates.

The update equations are kept together as one reference recipe.
The framework runner owns physics, HDF5 data, clocks and validation.
"""

import time

import torch
from torch import nn

from holomotion.src.env.reference_tracking.mdp import (
    CRITIC_EXTRA,
    INV_DIM,
    normalize_xi,
    unflatten_state,
)

from .imagination import set_requires_grad
from .networks import Critic, Policy, WorldModel


class ReferenceLearner:
    def __init__(self, args, motion, mdp, buffer):
        self.args, self.motion, self.mdp, self.buffer = (
            args,
            motion,
            mdp,
            buffer,
        )
        self.dev = args.device
        self.K, self.Hh, self.HZ = args.goal_frames, args.hist, args.horizon
        goal_dim = 73 * self.K
        self.wm = WorldModel().to(self.dev)
        self.pi = Policy(mdp.obs_dim, goal_dim, hist=self.Hh).to(self.dev)
        self.critic = Critic(INV_DIM + CRITIC_EXTRA + goal_dim).to(self.dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=args.lr)
        self.opt_pi = torch.optim.Adam(
            self.pi.parameters(), lr=args.lr, betas=(0.9, 0.95)
        )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.ERR_FLOOR = torch.tensor(
            [0.003] * 3
            + [0.005] * 3
            + [0.02] * 3
            + [0.05] * 3
            + [0.003] * 29
            + [0.05] * 29,
            device=self.dev,
        )
        self.PROFILE = False
        for target, name in (
            ("world_loss", "_world_loss"),
            ("imag_step", "_imag_step"),
            ("reward_term", "_reward_term"),
            ("critic_input_c", "_critic_input"),
        ):
            fn = getattr(self, name)
            setattr(self, target, torch.compile(fn) if args.compile else fn)

    def set_learning_rate(self, rate):
        for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple((torch.stack([P[i] for P in preds], 1) for i in range(6)))

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.ERR_FLOOR)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2
        e = e.mean(-1)
        return ((e * mask).sum() / s1.shape[0], e[:, -1])

    def _world_loss(self, s, a, xi, s1, mask):
        return self._wm_err(self._wm_rollout(s, a, xi), s1, mask)

    def _imag_step(
        self, S, o_hist, a_prev, f, xi, xi_n, g, observation_noise=None
    ):
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = self.mdp.obs(S1, a, xi, f1)
        o1 = (
            self.mdp.add_noise(o1)
            if observation_noise is None
            else o1 + observation_noise
        )
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return (S1, a, o_hist, f1, self.motion.goal(f1, self.K))

    def _reward_term(self, S, f, a, a_prev):
        return (
            self.mdp.reward(S, f, a, a_prev),
            self.mdp.soft_termination(S, f),
        )

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    def wm_update(self):
        w = self.buffer.sample_windows(self.args.wm_batch, self.args.wm_window)
        loss, e_last = self.world_loss(
            w["s"], w["a"], w["xi"], w["s1"], w["mask"]
        )
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        self.last_world_gradient_norm = nn.utils.clip_grad_norm_(
            self.wm.parameters(), 1.0
        ).detach()
        if not torch.isfinite(self.last_world_gradient_norm):
            raise FloatingPointError("Nonfinite world-model gradient")
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return (
            loss.detach() / self.args.wm_window,
            (e_last.detach() * m).sum() / m.sum().clamp(min=1),
        )

    def sample_valid_starts(self, B: int):
        (starts, hists, n) = ([], [], 0)
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 64:
                raise ValueError(
                    "Replay does not provide enough starts with a complete imagination horizon"
                )
            (s, h) = self.buffer.sample_starts(2 * B, self.Hh)
            f = s["f"]
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            (
                starts.append({k: v[idx] for (k, v) in s.items()}),
                hists.append({k: v[idx] for (k, v) in h.items()}),
            )
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
        return (cat(starts), cat(hists))

    def policy_update(self):
        if (
            getattr(self.args, "pi_microbatch", self.args.pi_batch)
            < self.args.pi_batch
        ):
            from .imagination import policy_update_microbatched

            return policy_update_microbatched(self)
        if self.PROFILE:
            torch.cuda.synchronize()
            tq0 = time.time()
        (s, h) = self.sample_valid_starts(self.args.pi_batch)
        S = unflatten_state(s["s"])
        (xi, f, a_prev) = (s["xi"], s["f"], s["a_prev"])
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = self.mdp.obs(
                unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]
            )
        o_hist = self.mdp.add_noise(o_hist)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        B = self.args.pi_batch
        g0 = self.motion.goal(f, self.K)
        g = g0
        S0 = S
        (Ss, As, Aps, Fs, Gs) = ([], [], [], [], [])
        for t in range(self.HZ):
            Aps.append(a_prev)
            (S, a_prev, o_hist, f, g) = self.imag_step(
                S, o_hist, a_prev, f, xi, xi_n, g
            )
            (Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g))
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        S_all = tuple((flat([St[i] for St in Ss]) for i in range(6)))
        f_all = flat(Fs)
        (r, d) = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        (r, d) = (r.view(self.HZ, B), d.view(self.HZ, B))
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
        disc = self.args.gamma ** torch.arange(
            self.HZ, device=self.dev, dtype=torch.float32
        ).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        (rews, dones) = (r.detach(), d.detach())
        if not self.args.no_critic:
            J = J + self.args.gamma**self.HZ * c * self.critic(
                self._critic_input(S, f, xi_n, g)
            )
        loss = -J.mean()
        if self.PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if self.PROFILE:
            torch.cuda.synchronize()
            tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if not torch.isfinite(gn):
            raise FloatingPointError(
                "Nonfinite reference-policy gradient; candidate rejected"
            )
        self.opt_pi.step()
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        v_loss = torch.zeros(())
        if not self.args.no_critic:
            with torch.no_grad():
                S_c = tuple(
                    (
                        torch.cat(
                            (
                                S0[i].unsqueeze(0),
                                torch.stack([St[i].detach() for St in Ss]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for i in range(6)
                    )
                )
                f_c = torch.cat(
                    (s["f"].unsqueeze(0), torch.stack(Fs)), 0
                ).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(
                    0, 1
                )
                xi_c = (
                    xi_n.unsqueeze(0).expand(self.HZ + 1, -1, -1).flatten(0, 1)
                )
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(
                    self.HZ + 1, B, -1
                )
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(self.HZ - 1, -1, -1):
                    R = rews[t] + self.args.gamma * (1 - dones[t]) * (
                        (1 - self.args.lam) * V[t + 1] + self.args.lam * R
                    )
                    targets.append(R)
                targets = torch.stack(targets[::-1])
            (Xf, Yf) = (X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1))
            n = Xf.shape[0]
            mb = n // self.args.critic_minibatches
            for _ in range(self.args.critic_epochs):
                perm = torch.randperm(n, device=self.dev)
                for i in range(self.args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        if self.PROFILE:
            torch.cuda.synchronize()
            print(
                f"[prof] imagine fwd {1000.0 * (tq1 - tq0):.0f} ms | bwd {1000.0 * (tq2 - tq1):.0f} ms | critic {1000.0 * (time.time() - tq2):.0f} ms",
                flush=True,
            )
        return (
            loss.detach(),
            v_loss.detach(),
            (r_sum / self.HZ / 0.1).mean(),
            gn.detach(),
            c.mean().detach(),
        )
