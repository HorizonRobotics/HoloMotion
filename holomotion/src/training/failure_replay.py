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

"""Pure decisions for same-motion failure replay.

The rolling source bound gates additional replay assignments only. Ordinary
assignments always proceed and can themselves exceed the configured fraction.
"""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class FailureReplayConfig:
    enabled: bool = False
    trigger_termination_terms: tuple[str, ...] = ()
    replay_probability: float = 0.5
    max_retries_per_motion: int = 2
    replay_start_mode: str = "next_frame"
    backtrack_seconds: float = 0.75
    preserve_mode_mask: bool = True
    count_against_source_budget: bool = True
    source_budget_upper_multiplier: float = 2.0
    source_budget_window_assignments: int = 65536

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any] | None
    ) -> FailureReplayConfig | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("failure_replay must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown failure_replay settings: {sorted(unknown)}"
            )
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("failure_replay.enabled must be boolean")
        if not enabled:
            return None

        values = dict(raw)
        terms = values.get("trigger_termination_terms", ())
        if (
            not isinstance(terms, Sequence)
            or isinstance(terms, (str, bytes))
            or not terms
            or any(
                not isinstance(term, str) or not term.strip() for term in terms
            )
        ):
            raise ValueError(
                "failure_replay.trigger_termination_terms must be "
                "a nonempty sequence"
            )
        values["trigger_termination_terms"] = tuple(terms)
        config = cls(**values)
        if config.replay_start_mode not in ("next_frame", "backtrack"):
            raise ValueError("Unknown failure replay start mode")
        for name in ("preserve_mode_mask", "count_against_source_budget"):
            if type(getattr(config, name)) is not bool:
                raise ValueError(f"failure_replay.{name} must be boolean")
        for name in (
            "replay_probability",
            "backtrack_seconds",
            "source_budget_upper_multiplier",
        ):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"failure_replay.{name} must be finite")
        if not 0.0 <= config.replay_probability <= 1.0:
            raise ValueError("replay_probability must be in [0, 1]")
        if config.backtrack_seconds < 0.0:
            raise ValueError("backtrack_seconds must be nonnegative")
        if config.source_budget_upper_multiplier < 1.0:
            raise ValueError("source_budget_upper_multiplier must be >= 1")
        if (
            type(config.max_retries_per_motion) is not int
            or config.max_retries_per_motion < 0
        ):
            raise ValueError(
                "max_retries_per_motion must be a nonnegative integer"
            )
        if (
            type(config.source_budget_window_assignments) is not int
            or config.source_budget_window_assignments <= 0
        ):
            raise ValueError(
                "source_budget_window_assignments must be positive"
            )
        return config


def select_failure_replays(
    failure_mask: torch.Tensor,
    retry_counts: torch.Tensor,
    probability_draws: torch.Tensor,
    *,
    config: FailureReplayConfig | None,
    evaluating: bool = False,
) -> torch.Tensor:
    """Select replay candidates without changing retries or exposure."""
    if failure_mask.dtype != torch.bool:
        raise ValueError("failure_mask must be boolean")
    if config is None or not config.enabled or evaluating:
        return torch.zeros_like(failure_mask)
    if (
        failure_mask.shape != retry_counts.shape
        or failure_mask.shape != probability_draws.shape
    ):
        raise ValueError("Replay decision inputs must align")
    if retry_counts.dtype not in _INTEGER_DTYPES:
        raise ValueError("retry_counts must contain integers")
    if bool((retry_counts < 0).any()):
        raise ValueError("retry_counts must be nonnegative")
    if not torch.is_floating_point(probability_draws):
        raise ValueError("probability_draws must be floating point")
    if (
        not bool(torch.isfinite(probability_draws).all())
        or bool((probability_draws < 0).any())
        or bool((probability_draws >= 1).any())
    ):
        raise ValueError("probability_draws must be in [0, 1)")
    return (
        failure_mask
        & (retry_counts < config.max_retries_per_motion)
        & (probability_draws < config.replay_probability)
    )


def retry_start_frames(
    failure_frames: torch.Tensor,
    lengths: torch.Tensor,
    *,
    n_future_frames: int,
    backtrack_frames: int,
    replay_start_mode: str = "next_frame",
) -> torch.Tensor:
    """Choose a valid same-motion retry start with future frames available."""
    if failure_frames.shape != lengths.shape:
        raise ValueError("failure frames and lengths must align")
    if (
        failure_frames.dtype not in _INTEGER_DTYPES
        or lengths.dtype not in _INTEGER_DTYPES
    ):
        raise ValueError("Frame indices and lengths must be integers")
    if bool((failure_frames < 0).any()) or bool((lengths <= 0).any()):
        raise ValueError("Frame indices and lengths must be valid")
    if (
        type(n_future_frames) is not int
        or n_future_frames < 0
        or type(backtrack_frames) is not int
        or backtrack_frames < 0
    ):
        raise ValueError("Frame offsets must be nonnegative integers")
    maximum = torch.clamp(lengths - 1 - n_future_frames, min=0)
    if replay_start_mode == "next_frame":
        return torch.remainder(failure_frames + 1, maximum + 1)
    if replay_start_mode == "backtrack":
        return torch.clamp(failure_frames - backtrack_frames, min=0).minimum(
            maximum
        )
    raise ValueError("Unknown failure replay start mode")


@dataclass(frozen=True)
class ReplayBudgetPlan:
    accepted_mask: torch.Tensor
    final_source_indices: torch.Tensor


class RollingSourceExposure:
    """Track recent assignments and preview ordered replay-gated writes."""

    def __init__(self, num_sources: int, capacity: int, device: Any) -> None:
        if (
            type(num_sources) is not int
            or num_sources <= 0
            or type(capacity) is not int
            or capacity <= 0
        ):
            raise ValueError(
                "Rolling source exposure needs positive dimensions"
            )
        self.num_sources = num_sources
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = torch.full(
            (capacity,), -1, dtype=torch.long, device=self.device
        )
        self.counts = torch.zeros(
            num_sources, dtype=torch.long, device=self.device
        )
        self.position = 0
        self.size = 0

    def record(self, source_indices: torch.Tensor) -> None:
        """Record the assignments actually made, in their write order."""
        if source_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError("Source indices must be integers")
        values = source_indices.to(self.device, dtype=torch.long).reshape(-1)
        if values.numel() == 0:
            return
        if bool(((values < 0) | (values >= self.num_sources)).any()):
            raise ValueError("Source exposure references an invalid source")
        if values.numel() >= self.capacity:
            values = values[-self.capacity :]
            self.buffer.copy_(values)
            self.counts.copy_(
                torch.bincount(values, minlength=self.num_sources)
            )
            self.position = 0
            self.size = self.capacity
            return
        positions = (
            self.position
            + torch.arange(
                values.numel(), device=self.device, dtype=torch.long
            )
        ) % self.capacity
        overwritten = self.buffer[positions]
        valid = overwritten >= 0
        if bool(valid.any()):
            self.counts.scatter_add_(
                0,
                overwritten[valid],
                -torch.ones_like(overwritten[valid]),
            )
        self.buffer[positions] = values
        self.counts.scatter_add_(0, values, torch.ones_like(values))
        self.position = int((self.position + values.numel()) % self.capacity)
        self.size = min(self.capacity, self.size + int(values.numel()))

    def plan_replays(
        self,
        ordinary_source_indices: torch.Tensor,
        candidate_replay_sources: torch.Tensor,
        upper_probabilities: torch.Tensor,
    ) -> ReplayBudgetPlan:
        """Preview every assignment slot in the order it will be recorded.

        A candidate source of -1 means ordinary-only. For each slot, remove
        the overwritten source before testing the candidate replay fraction.
        Accepted replays and ordinary fallbacks then affect later candidates.
        This preview does not mutate exposure; record the returned final
        sources after the assignments really occur. Only extra replays are
        gated: ordinary assignments may exceed the stated fraction.
        """
        if (
            ordinary_source_indices.ndim != 1
            or candidate_replay_sources.ndim != 1
            or ordinary_source_indices.shape != candidate_replay_sources.shape
        ):
            raise ValueError("Assignment source vectors must align in 1D")
        if (
            ordinary_source_indices.dtype not in _INTEGER_DTYPES
            or candidate_replay_sources.dtype not in _INTEGER_DTYPES
        ):
            raise ValueError("Assignment source indices must be integers")
        ordinary = ordinary_source_indices.to(self.device, dtype=torch.long)
        candidates = candidate_replay_sources.to(self.device, dtype=torch.long)
        upper = upper_probabilities.to(
            self.device, dtype=torch.float64
        ).reshape(-1)
        if upper.shape != (self.num_sources,):
            raise ValueError("Replay source upper probabilities misalign")
        if not bool(torch.isfinite(upper).all()) or bool((upper <= 0).any()):
            raise ValueError(
                "Replay source upper probabilities must be positive"
            )
        if bool(
            ((ordinary < 0) | (ordinary >= self.num_sources)).any()
        ) or bool(
            ((candidates < -1) | (candidates >= self.num_sources)).any()
        ):
            raise ValueError("Assignment references an invalid source")

        ring = self.buffer.detach().cpu().tolist()
        counts = self.counts.detach().cpu().tolist()
        ordinary_values = ordinary.detach().cpu().tolist()
        candidate_values = candidates.detach().cpu().tolist()
        upper_values = upper.detach().cpu().tolist()
        position = self.position
        size = self.size
        accepted: list[bool] = []
        final_sources: list[int] = []
        for ordinary_source, replay_source in zip(
            ordinary_values, candidate_values, strict=True
        ):
            old_source = ring[position]
            if old_source >= 0:
                counts[old_source] -= 1
            next_size = min(self.capacity, size + 1)
            allow_replay = (
                replay_source >= 0
                and counts[replay_source] + 1
                <= upper_values[replay_source] * next_size + 1e-12
            )
            selected_source = (
                replay_source if allow_replay else ordinary_source
            )
            ring[position] = selected_source
            counts[selected_source] += 1
            position = (position + 1) % self.capacity
            size = next_size
            accepted.append(allow_replay)
            final_sources.append(selected_source)
        return ReplayBudgetPlan(
            accepted_mask=torch.tensor(
                accepted, dtype=torch.bool, device=self.device
            ),
            final_source_indices=torch.tensor(
                final_sources, dtype=torch.long, device=self.device
            ),
        )

    def metrics(self) -> dict[str, float]:
        if self.size == 0:
            return {
                "source_budget_window_size": 0.0,
                "source_exposure_min": 0.0,
                "source_exposure_max": 0.0,
            }
        fractions = self.counts.to(torch.float64) / float(self.size)
        return {
            "source_budget_window_size": float(self.size),
            "source_exposure_min": float(fractions.min()),
            "source_exposure_max": float(fractions.max()),
        }
