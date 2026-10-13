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

"""Fixed source-budget sampling for motion windows.

Each source receives its configured share of draws regardless of how many
windows it contains. Draws within a source are uniform and with replacement.
An optional controller can instead adapt bounded source shares and prioritize
harder windows within each source.
"""

from __future__ import annotations
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import Sampler


class SourceWeightController(Protocol):
    """The sampler's interface to a fixed or adaptive source controller."""

    source_names: tuple[str, ...]
    source_probabilities: torch.Tensor
    version: int

    def choose(
        self, indices: torch.Tensor, count: int, generator: torch.Generator
    ) -> list[int]: ...

    def record_selected_sources(
        self, source_indices: torch.Tensor
    ) -> None: ...


@dataclass(frozen=True)
class SourceAdaptiveConfig:
    """Opt-in source and within-source adaptation settings."""

    enabled: bool = True
    group_by: str = "source_id"
    lower_multiplier: float = 0.5
    upper_multiplier: float = 2.0
    uniform_fraction_within_source: float = 0.2
    difficulty_ema_alpha: float = 0.2
    difficulty_scale_m: float = 0.25
    min_difficulty_multiplier: float = 0.5
    max_difficulty_multiplier: float = 2.0
    min_window_weight: float = 0.05
    update_interval_swaps: int = 10
    warmup_swaps: int = 10
    min_observations: int = 32

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any] | None
    ) -> SourceAdaptiveConfig | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("source_adaptive must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown source_adaptive settings: {sorted(unknown)}"
            )
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("source_adaptive.enabled must be boolean")
        if not enabled:
            return None
        config = cls(**dict(raw))
        if config.group_by != "source_id":
            raise ValueError("source_adaptive.group_by must be 'source_id'")
        float_fields = (
            "lower_multiplier",
            "upper_multiplier",
            "uniform_fraction_within_source",
            "difficulty_ema_alpha",
            "difficulty_scale_m",
            "min_difficulty_multiplier",
            "max_difficulty_multiplier",
            "min_window_weight",
        )
        for name in float_fields:
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"source_adaptive.{name} must be finite")
        if not 0.0 <= config.lower_multiplier <= 1.0:
            raise ValueError("lower_multiplier must be in [0, 1]")
        if config.upper_multiplier < 1.0:
            raise ValueError("upper_multiplier must be at least one")
        if not 0.0 < config.uniform_fraction_within_source <= 1.0:
            raise ValueError(
                "uniform_fraction_within_source must be in (0, 1]"
            )
        if not 0.0 < config.difficulty_ema_alpha <= 1.0:
            raise ValueError("difficulty_ema_alpha must be in (0, 1]")
        if config.difficulty_scale_m <= 0.0:
            raise ValueError("difficulty_scale_m must be positive")
        if not (
            0.0
            < config.min_difficulty_multiplier
            <= config.max_difficulty_multiplier
        ):
            raise ValueError(
                "difficulty multipliers must be positive and ordered"
            )
        if not 0.0 < config.min_window_weight <= 1.0:
            raise ValueError("min_window_weight must be in (0, 1]")
        for name in (
            "update_interval_swaps",
            "warmup_swaps",
            "min_observations",
        ):
            value = getattr(config, name)
            minimum = 0 if name == "warmup_swaps" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid source_adaptive.{name}")
        return config


def project_bounded_probabilities(
    raw: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Normalize positive scores while respecting each source's bounds."""
    raw = raw.detach().cpu().to(torch.float64)
    lower = lower.detach().cpu().to(torch.float64)
    upper = upper.detach().cpu().to(torch.float64)
    if raw.ndim != 1 or lower.shape != raw.shape or upper.shape != raw.shape:
        raise ValueError("raw/lower/upper must be aligned 1D tensors")
    if not bool(
        torch.isfinite(raw).all()
        and torch.isfinite(lower).all()
        and torch.isfinite(upper).all()
    ):
        raise ValueError("Source probability inputs must be finite")
    if bool((raw <= 0).any() or (lower < 0).any() or (upper < lower).any()):
        raise ValueError("Invalid source probability bounds")
    if float(lower.sum()) > 1.0 or float(upper.sum()) < 1.0:
        raise ValueError("Source probability bounds cannot sum to one")

    lo, hi = 0.0, 1.0
    for _ in range(128):
        if float(torch.clamp(hi * raw, lower, upper).sum()) >= 1.0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("Source probability projection did not reach one")
    for _ in range(96):
        mid = (lo + hi) * 0.5
        if float(torch.clamp(mid * raw, lower, upper).sum()) < 1.0:
            lo = mid
        else:
            hi = mid
    result = torch.clamp((lo + hi) * 0.5 * raw, lower, upper)
    residual = 1.0 - float(result.sum())
    if abs(residual) > 1.0e-12:
        room = upper - result if residual > 0 else result - lower
        available = room > 1.0e-15
        if not bool(available.any()):
            raise RuntimeError("Source probability projection has no capacity")
        result[available] += residual * room[available] / room[available].sum()
    return result


class FixedSourceWeights:
    """Fixed source probabilities and uniform window draws."""

    def __init__(
        self,
        *,
        size: int,
        source_names: Sequence[str],
        source_window_indices: Sequence[Sequence[int] | torch.Tensor],
        base_probabilities: Sequence[float] | torch.Tensor,
    ) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("Fixed source sampling needs a nonempty dataset")
        if not source_names or len(source_names) != len(source_window_indices):
            raise ValueError(
                "Source names and groups must be nonempty and aligned"
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError("Source names must be unique")

        self.size = size
        self.source_names = tuple(str(name) for name in source_names)
        self.source_window_indices = tuple(
            torch.as_tensor(indices, dtype=torch.long)
            for indices in source_window_indices
        )
        covered = torch.zeros(size, dtype=torch.bool)
        for indices in self.source_window_indices:
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError(
                    "Every configured source needs a nonempty 1D group"
                )
            if bool(((indices < 0) | (indices >= size)).any()):
                raise ValueError("Source group references an invalid window")
            if torch.unique(indices).numel() != indices.numel():
                raise ValueError("Source group contains a duplicate window")
            if bool(covered[indices].any()):
                raise ValueError("Source groups must be disjoint")
            covered[indices] = True
        if not bool(covered.all()):
            raise ValueError("Source groups must cover every training window")

        if (
            isinstance(base_probabilities, torch.Tensor)
            and base_probabilities.dtype == torch.bool
        ) or any(isinstance(value, bool) for value in base_probabilities):
            raise ValueError("Base source probabilities must be numeric")
        base = torch.as_tensor(base_probabilities, dtype=torch.float64).clone()
        if base.shape != (len(self.source_names),):
            raise ValueError("Base source probabilities have the wrong shape")
        if not bool(torch.isfinite(base).all()) or bool((base <= 0).any()):
            raise ValueError(
                "Base source probabilities must be finite and positive"
            )
        base /= base.sum()
        self.base_probabilities = base
        self.source_probabilities = base.clone()
        self.source_selection_counts = torch.zeros(
            len(self.source_names), dtype=torch.float64
        )
        self.version = 0

    def choose(
        self, indices: torch.Tensor, count: int, generator: torch.Generator
    ) -> list[int]:
        if indices.numel() == 0 and count:
            raise ValueError("Cannot sample an empty source")
        offsets = torch.randint(
            indices.numel(), (int(count),), generator=generator
        )
        return indices[offsets].tolist()

    def record_selected_sources(self, source_indices: torch.Tensor) -> None:
        values = source_indices.detach().cpu().to(torch.long)
        self.source_selection_counts += torch.bincount(
            values, minlength=len(self.source_names)
        ).to(torch.float64)

    def metrics(self) -> dict[str, float]:
        probabilities = self.source_probabilities
        entropy = -float(
            torch.sum(
                probabilities
                * torch.log(torch.clamp(probabilities, min=1e-15))
            )
        )
        result = {
            "weight_version": 0.0,
            "adaptation_enabled": 0.0,
            "source_probability_min": float(probabilities.min()),
            "source_probability_max": float(probabilities.max()),
            "source_probability_entropy": entropy,
        }
        selected_total = float(self.source_selection_counts.sum())
        for index, source_name in enumerate(self.source_names):
            metric_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
            prefix = f"source/{index:02d}_{metric_name}/"
            probability = float(probabilities[index])
            result[prefix + "baseline_probability"] = probability
            result[prefix + "target_probability"] = probability
            result[prefix + "rank_local_dispatched_count"] = float(
                self.source_selection_counts[index]
            )
            result[prefix + "rank_local_dispatched_fraction"] = (
                float(self.source_selection_counts[index]) / selected_total
                if selected_total > 0.0
                else 0.0
            )
        return result


class SourceAdaptiveWeights(FixedSourceWeights):
    """Adapt source budgets and window difficulty from aggregated errors."""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        size: int,
        source_names: Sequence[str],
        source_window_indices: Sequence[Sequence[int] | torch.Tensor],
        base_probabilities: Sequence[float] | torch.Tensor,
        config: SourceAdaptiveConfig,
        identity: str,
    ) -> None:
        if not isinstance(config, SourceAdaptiveConfig):
            raise ValueError("Source-adaptive configuration is required")
        if SourceAdaptiveConfig.parse(asdict(config)) is None:
            raise ValueError("Source-adaptive configuration is disabled")
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError(
                "Source-adaptive sampling needs a SHA256 identity"
            )
        try:
            bytes.fromhex(identity)
        except ValueError as exc:
            raise ValueError(
                "Source-adaptive identity must be hexadecimal"
            ) from exc
        super().__init__(
            size=size,
            source_names=source_names,
            source_window_indices=source_window_indices,
            base_probabilities=base_probabilities,
        )
        self.config = config
        self.identity = identity
        self.window_to_source = torch.full((size,), -1, dtype=torch.long)
        for source_index, indices in enumerate(self.source_window_indices):
            self.window_to_source[indices] = source_index
        self.lower_probabilities = (
            self.base_probabilities * config.lower_multiplier
        )
        self.upper_probabilities = torch.clamp(
            self.base_probabilities * config.upper_multiplier, max=1.0
        )
        if (
            float(self.lower_probabilities.sum()) > 1.0
            or float(self.upper_probabilities.sum()) < 1.0
        ):
            raise ValueError("Configured source bounds cannot sum to one")
        self.window_weights = torch.ones(size, dtype=torch.float32)
        self.window_ema_error = torch.full(
            (size,), config.difficulty_scale_m, dtype=torch.float64
        )
        self.window_seen = torch.zeros(size, dtype=torch.bool)
        self.source_ema_error = torch.full(
            (len(self.source_names),),
            config.difficulty_scale_m,
            dtype=torch.float64,
        )
        self.source_seen = torch.zeros(
            len(self.source_names), dtype=torch.bool
        )
        self.pending: dict[int, tuple[float, float]] = {}
        self.source_pending_sums = torch.zeros_like(self.source_ema_error)
        self.source_pending_counts = torch.zeros_like(self.source_ema_error)
        self.round = 0
        self.observed_samples = 0.0
        self.last_observed_samples = 0.0
        self.consecutive_empty_statistics_rounds = 0
        self.last_updated_windows = 0
        self.last_updated_sources = 0

    def choose(
        self, indices: torch.Tensor, count: int, generator: torch.Generator
    ) -> list[int]:
        if indices.numel() == 0 and count:
            raise ValueError("Cannot sample an empty source")
        output: list[int] = []
        for _ in range(int(count)):
            uniform = (
                float(torch.rand((), generator=generator))
                < self.config.uniform_fraction_within_source
            )
            while True:
                offset = int(
                    torch.randint(indices.numel(), (), generator=generator)
                )
                index = int(indices[offset])
                if uniform or float(
                    torch.rand((), generator=generator)
                ) < float(self.window_weights[index]):
                    output.append(index)
                    break
        return output

    def observe(
        self,
        indices: torch.Tensor,
        error_sums: torch.Tensor,
        counts: torch.Tensor,
    ) -> bool:
        if (
            indices.ndim != 1
            or error_sums.shape != indices.shape
            or counts.shape != indices.shape
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Adaptive statistics must be aligned 1D arrays")
        idx = indices.detach().cpu().to(torch.long)
        sums = error_sums.detach().cpu().to(torch.float64)
        cnt = counts.detach().cpu().to(torch.float64)
        if bool(((idx < 0) | (idx >= self.size)).any()):
            raise ValueError("Adaptive statistics reference an invalid window")
        if not bool(torch.isfinite(sums).all() and torch.isfinite(cnt).all()):
            raise ValueError("Non-finite adaptive statistics")
        if bool(
            (cnt < 0).any()
            or (sums < 0).any()
            or ((cnt == 0) & (sums != 0)).any()
        ):
            raise ValueError("Adaptive error sums/counts are inconsistent")

        observed = float(cnt.sum())
        self.last_observed_samples = observed
        if observed:
            self.consecutive_empty_statistics_rounds = 0
        else:
            self.consecutive_empty_statistics_rounds += 1
        for index, total, number in zip(
            idx.tolist(), sums.tolist(), cnt.tolist(), strict=True
        ):
            if number:
                old_sum, old_count = self.pending.get(index, (0.0, 0.0))
                self.pending[index] = old_sum + total, old_count + number
        source_indices = self.window_to_source[idx]
        self.source_pending_sums.scatter_add_(0, source_indices, sums)
        self.source_pending_counts.scatter_add_(0, source_indices, cnt)
        self.observed_samples += observed
        self.round += 1
        self.last_updated_windows = 0
        self.last_updated_sources = 0
        if (
            self.round < self.config.warmup_swaps
            or (self.round - self.config.warmup_swaps)
            % self.config.update_interval_swaps
        ):
            return False

        alpha = self.config.difficulty_ema_alpha
        scale = self.config.difficulty_scale_m
        for index in sorted(self.pending):
            total, count = self.pending[index]
            if count < self.config.min_observations:
                continue
            observation = min(total / count, scale)
            value = (
                (1.0 - alpha) * float(self.window_ema_error[index])
                + alpha * observation
                if self.window_seen[index]
                else observation
            )
            self.window_ema_error[index] = value
            self.window_weights[index] = (
                self.config.min_window_weight
                + (1.0 - self.config.min_window_weight) * value / scale
            )
            self.window_seen[index] = True
            del self.pending[index]
            self.last_updated_windows += 1

        enough = self.source_pending_counts >= self.config.min_observations
        for source_index in torch.nonzero(enough).view(-1).tolist():
            observation = min(
                float(
                    self.source_pending_sums[source_index]
                    / self.source_pending_counts[source_index]
                ),
                scale,
            )
            value = (
                (1.0 - alpha) * float(self.source_ema_error[source_index])
                + alpha * observation
                if self.source_seen[source_index]
                else observation
            )
            self.source_ema_error[source_index] = value
            self.source_seen[source_index] = True
            self.source_pending_sums[source_index] = 0.0
            self.source_pending_counts[source_index] = 0.0
            self.last_updated_sources += 1
        if self.last_updated_sources:
            normalized = torch.clamp(self.source_ema_error / scale, 0.0, 1.0)
            multiplier = (
                self.config.min_difficulty_multiplier
                + (
                    self.config.max_difficulty_multiplier
                    - self.config.min_difficulty_multiplier
                )
                * normalized
            )
            self.source_probabilities = project_bounded_probabilities(
                self.base_probabilities * multiplier,
                self.lower_probabilities,
                self.upper_probabilities,
            )
        if self.last_updated_windows or self.last_updated_sources:
            self.version += 1
        return bool(self.last_updated_windows or self.last_updated_sources)

    def metrics(self) -> dict[str, float]:
        result = super().metrics()
        result.update(
            {
                "weight_version": float(self.version),
                "adaptation_enabled": 1.0,
                "statistics_round": float(self.round),
                "observed_samples": self.observed_samples,
                "last_observed_samples": self.last_observed_samples,
                "consecutive_empty_statistics_rounds": float(
                    self.consecutive_empty_statistics_rounds
                ),
                "last_updated_windows": float(self.last_updated_windows),
                "last_updated_sources": float(self.last_updated_sources),
                "pending_windows": float(len(self.pending)),
                "uniform_fraction": self.config.uniform_fraction_within_source,
            }
        )
        for index, source_name in enumerate(self.source_names):
            metric_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
            prefix = f"source/{index:02d}_{metric_name}/"
            result[prefix + "baseline_probability"] = float(
                self.base_probabilities[index]
            )
            result[prefix + "lower_probability"] = float(
                self.lower_probabilities[index]
            )
            result[prefix + "upper_probability"] = float(
                self.upper_probabilities[index]
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_VERSION,
            "kind": "source_adaptive",
            "identity": self.identity,
            "size": self.size,
            "source_names": list(self.source_names),
            "config": asdict(self.config),
            "window_weights": self.window_weights.clone(),
            "window_ema_error": self.window_ema_error.clone(),
            "window_seen": self.window_seen.clone(),
            "source_ema_error": self.source_ema_error.clone(),
            "source_seen": self.source_seen.clone(),
            "source_probabilities": self.source_probabilities.clone(),
            "pending": {
                key: list(value) for key, value in self.pending.items()
            },
            "source_pending_sums": self.source_pending_sums.clone(),
            "source_pending_counts": self.source_pending_counts.clone(),
            "source_selection_counts": self.source_selection_counts.clone(),
            "round": self.round,
            "version": self.version,
            "observed_samples": self.observed_samples,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema_version") != self.STATE_VERSION
            or state.get("kind") != "source_adaptive"
            or state.get("identity") != self.identity
            or state.get("size") != self.size
            or state.get("source_names") != list(self.source_names)
            or state.get("config") != asdict(self.config)
        ):
            raise ValueError("Source-adaptive checkpoint identity mismatch")
        targets = {
            "window_weights": self.window_weights,
            "window_ema_error": self.window_ema_error,
            "window_seen": self.window_seen,
            "source_ema_error": self.source_ema_error,
            "source_seen": self.source_seen,
            "source_probabilities": self.source_probabilities,
            "source_pending_sums": self.source_pending_sums,
            "source_pending_counts": self.source_pending_counts,
            "source_selection_counts": self.source_selection_counts,
        }
        loaded: dict[str, torch.Tensor] = {}
        for name, target in targets.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Invalid source-adaptive tensor {name}")
            value = value.detach().cpu()
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Invalid source-adaptive tensor {name}")
            if value.dtype != torch.bool and not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"Non-finite source-adaptive tensor {name}")
            loaded[name] = value
        scale = self.config.difficulty_scale_m
        weights = loaded["window_weights"]
        if bool(
            (weights <= 0).any()
            or (weights < self.config.min_window_weight * (1.0 - 1e-6)).any()
            or (weights > 1.0).any()
        ):
            raise ValueError("Invalid source-adaptive window weights")
        for name in ("window_ema_error", "source_ema_error"):
            values = loaded[name]
            if bool((values < 0).any() or (values > scale + 1e-7).any()):
                raise ValueError(f"Invalid source-adaptive error {name}")
        probabilities = loaded["source_probabilities"]
        if (
            bool(
                (probabilities <= 0).any()
                or (probabilities < self.lower_probabilities - 1e-7).any()
                or (probabilities > self.upper_probabilities + 1e-7).any()
            )
            or abs(float(probabilities.sum()) - 1.0) > 1e-7
        ):
            raise ValueError("Invalid source-adaptive probabilities")
        sums = loaded["source_pending_sums"]
        counts = loaded["source_pending_counts"]
        if bool(
            (sums < 0).any()
            or (counts < 0).any()
            or ((counts == 0) & (sums != 0)).any()
            or (loaded["source_selection_counts"] < 0).any()
        ):
            raise ValueError("Invalid source-adaptive counters")
        pending: dict[int, tuple[float, float]] = {}
        pending_state = state.get("pending")
        if not isinstance(pending_state, Mapping):
            raise ValueError("Invalid source-adaptive pending windows")
        for key, values in pending_state.items():
            if (
                type(key) is not int
                or not 0 <= key < self.size
                or not isinstance(values, (list, tuple))
                or len(values) != 2
            ):
                raise ValueError("Invalid source-adaptive pending window")
            try:
                total, count = map(float, values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid source-adaptive pending statistics"
                ) from exc
            if not math.isfinite(total + count) or total < 0 or count <= 0:
                raise ValueError("Invalid source-adaptive pending statistics")
            pending[key] = (total, count)
        round_, version = state.get("round"), state.get("version")
        if (
            type(round_) is not int
            or type(version) is not int
            or not 0 <= version <= round_
        ):
            raise ValueError("Invalid source-adaptive counters")
        try:
            if isinstance(state.get("observed_samples"), bool):
                raise ValueError
            observed = float(state.get("observed_samples"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid source-adaptive observation count"
            ) from exc
        if not math.isfinite(observed) or observed < 0:
            raise ValueError("Invalid source-adaptive observation count")
        for name, target in targets.items():
            target.copy_(loaded[name])
        self.pending = pending
        self.round = round_
        self.version = version
        self.observed_samples = observed
        self.last_observed_samples = 0.0
        self.consecutive_empty_statistics_rounds = 0
        self.last_updated_windows = 0
        self.last_updated_sources = 0


class SourceAdaptiveInfiniteSampler(Sampler[int]):
    """Draw source-budgeted batches using a pluggable weight controller."""

    def __init__(
        self,
        *,
        source_window_indices: Sequence[Sequence[int] | torch.Tensor],
        controller: SourceWeightController,
        batch_size: int,
        seed: int,
    ) -> None:
        self._groups = tuple(
            torch.as_tensor(indices, dtype=torch.long)
            for indices in source_window_indices
        )
        if len(self._groups) != len(controller.source_names):
            raise ValueError("Source sampler groups do not match controller")
        self._controller = controller
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._epoch = 0
        if self._batch_size <= 0 or any(
            group.numel() == 0 for group in self._groups
        ):
            raise ValueError(
                "Source sampler needs a positive batch and nonempty groups"
            )

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(self._seed + self._epoch)
            source_draws = torch.multinomial(
                self._controller.source_probabilities,
                self._batch_size,
                replacement=True,
                generator=generator,
            )
            counts = torch.bincount(source_draws, minlength=len(self._groups))
            batch: list[int] = []
            batch_sources: list[int] = []
            for source_index, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                selected = self._controller.choose(
                    self._groups[source_index], count, generator
                )
                batch.extend(selected)
                batch_sources.extend([source_index] * count)
            permutation = torch.randperm(len(batch), generator=generator)
            ordered_sources = torch.tensor(batch_sources, dtype=torch.long)[
                permutation
            ]
            self._controller.record_selected_sources(ordered_sources)
            for offset in permutation.tolist():
                yield int(batch[offset])
            self._epoch += 1

    def __len__(self) -> int:
        return 2**31 - 1
