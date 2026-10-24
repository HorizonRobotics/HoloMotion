# Adapted from zita-ch/supermimic_vibed, commit a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from holomotion.src.env.reference_tracking.mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str):
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        self.break_after = torch.zeros(T, N, dtype=torch.bool, device=device)

    def add(self, tr: dict):
        self.break_after[self.ptr].zero_()
        for k in FIELDS:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int, *, generator=None):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(
            0, self.size - k + 1, (B,), device=self.device, generator=generator
        )
        env = torch.randint(
            0, self.N, (B,), device=self.device, generator=generator
        )
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        fields = ("s", "s1", "a", "xi", "done")
        out = {key: self.data[key][rows, e] for key in fields}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = (out["done"] | self.break_after[rows, e]).float()
        mask = torch.cumprod(
            torch.cat(
                (torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1
            ),
            -1,
        )
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(
            hist - 1, -1, -1, device=self.device
        )  # hist-1 ... 0
        back = torch.minimum(
            back.unsqueeze(0), ep_step.unsqueeze(-1)
        )  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {
            k: self.data[k][row, env]
            for k in ("s", "a_prev", "xi", "f", "ep_step")
        }
        hist_d = {
            k: self.data[k][hrows, henv]
            for k in ("s", "a_prev", "xi", "f", "ep_step")
        }
        return start, hist_d

    def state_dict(self):
        """Copy only populated storage rows to CPU, without a full GPU clone."""
        return dict(
            capacity=self.T,
            environments=self.N,
            ptr=self.ptr,
            size=self.size,
            data={
                key: value[: self.size].to("cpu", copy=True)
                for key, value in self.data.items()
            },
            break_after=self.break_after[: self.size].to("cpu", copy=True),
        )

    def load_state_dict(self, value):
        if value["capacity"] != self.T or value["environments"] != self.N:
            raise ValueError("Replay capacity or environment count changed")
        size, ptr = value["size"], value["ptr"]
        if (
            type(size) is not int
            or type(ptr) is not int
            or not 0 <= size <= self.T
            or not 0 <= ptr < self.T
            or (size < self.T and ptr != size)
        ):
            raise ValueError("Invalid replay storage cursor")
        if value["data"].keys() != self.data.keys():
            raise ValueError("Replay field schema changed")
        pairs = [
            (value["data"][key], target) for key, target in self.data.items()
        ]
        pairs.append((value["break_after"], self.break_after))
        for source, target in pairs:
            if (
                source.shape != target[:size].shape
                or source.dtype != target.dtype
            ):
                raise ValueError("Replay tensor shape or dtype changed")
        for source, target in pairs:
            target[:size].copy_(source)
        self.ptr, self.size = ptr, size

    def mark_episode_boundary(self):
        """Cut model windows at a physical restart without relabelling data."""
        if self.size:
            self.break_after[(self.ptr - 1) % self.T].fill_(True)
