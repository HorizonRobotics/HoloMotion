"""Simplified HDF5 motion cache backed by a PyTorch ``DataLoader``.

This module provides two core utilities:

* ``Hdf5MotionDataset`` – loads contiguous motion windows directly from HDF5
  shards using metadata stored in ``manifest.json``.
* ``MotionClipBatchCache`` – maintains a double-buffered cache of motion clips
  with deterministic swapping semantics suitable for high-throughput
  reinforcement learning.

Compared to the legacy slot-based prefetcher, this implementation keeps the
pipeline intentionally simple:

* A dataset-worker keeps shard handles open locally; no Ray dependency.
* Each cached batch has a fixed shape
  ``[max_num_clips, max_frame_length, feature_dims]``.
* Swapping a batch is handled via an O(1) pointer flip once the next batch is
  staged on the desired device (CPU or GPU).

The cache exposes helper methods that mirror the data access patterns required
by ``RefMotionCommand``:

* ``sample_env_assignments`` for initial clip/frame sampling.
* ``gather_state`` to fetch ``1 + n_future`` frames per environment.

All tensors returned by this module are ``torch.float32`` unless stated
otherwise; tensor shapes are noted explicitly in type annotations.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger
from tabulate import tabulate

import torch.distributed as dist  # type: ignore

Tensor = torch.Tensor


def _configure_weighted_bins(
    keys: List[str],
    cfg: Mapping[str, Any],
    batch_size_for_log: int,
) -> Tuple[List[List[int]], List[float], List[Dict[str, Any]]]:
    """Common helper to parse config, assign bins, and compute batch fractions."""
    if batch_size_for_log <= 0:
        batch_size_for_log = 1

    cfg_local: Dict[str, Any] = dict(cfg or {})

    patterns_cfg = cfg_local.get("bin_regex_patterns")
    if patterns_cfg is None:
        patterns_cfg = cfg_local.get("bin_regrex_patterns")
    if not patterns_cfg:
        raise ValueError(
            "weighted_bin configuration requires 'bin_regex_patterns' "
            "(list of {regex, ratio}) to be configured"
        )

    compiled_patterns: List[Dict[str, Any]] = []
    ratios: List[float] = []
    for idx, entry in enumerate(patterns_cfg):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns must be a mapping, "
                f"got {type(entry)}"
            )
        regex_str = entry.get("regex", entry.get("regrex", None))
        if not isinstance(regex_str, str) or not regex_str:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing a non-empty "
                f"'regex' field"
            )
        ratio_val = entry.get("ratio", None)
        if ratio_val is None:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns is missing 'ratio'"
            )
        ratio_f = float(ratio_val)
        if ratio_f < 0.0 or ratio_f > 1.0:
            raise ValueError(
                f"Entry {idx} in bin_regex_patterns has invalid ratio "
                f"{ratio_f:.6f}; expected in [0.0, 1.0]"
            )
        compiled_patterns.append(
            {
                "name": str(entry.get("name", f"bin_{idx}")),
                "regex": regex_str,
                "compiled": re.compile(regex_str),
            }
        )
        ratios.append(ratio_f)

    sum_explicit = float(sum(ratios))
    if sum_explicit > 1.0 + 1.0e-6:
        raise ValueError(
            f"Sum of weighted-bin ratios is {sum_explicit:.6f} (> 1.0). "
            "Please reduce the ratios so that their sum is <= 1.0."
        )
    others_ratio = max(0.0, 1.0 - sum_explicit)

    if len(keys) == 0:
        raise ValueError("weighted_bin configuration received an empty key set")

    num_items_total = float(len(keys))
    num_explicit = len(compiled_patterns)
    bin_indices: List[List[int]] = [[] for _ in range(num_explicit + 1)]

    for idx, motion_key in enumerate(keys):
        assigned = False
        for b_idx, pat in enumerate(compiled_patterns):
            if pat["compiled"].search(motion_key):
                bin_indices[b_idx].append(idx)
                assigned = True
                break
        if not assigned:
            bin_indices[-1].append(idx)

    # Combine explicit ratios with implicit "others" ratio
    all_ratios: List[float] = list(ratios)
    all_ratios.append(others_ratio)

    # If all motion keys are covered by explicit regex bins, but the specified
    # ratios sum to less than 1.0, linearly reweight explicit ratios so that
    # they sum to 1.0 and disable the implicit "others" bin.
    others_count = len(bin_indices[-1])
    if others_count == 0 and others_ratio > 0.0 and sum_explicit > 0.0:
        scale = 1.0 / sum_explicit
        ratios = [r * scale for r in ratios]
        others_ratio = 0.0
        all_ratios = list(ratios)
        all_ratios.append(others_ratio)
        logger.info(
            "Weighted-bin: all regex bins cover the dataset; "
            "linearly reweighted explicit ratios to sum to 1.0 and disabled "
            "the implicit 'others' bin."
        )

    # Validate non-empty bins for any positive ratio (including others)
    for b_idx, r in enumerate(all_ratios):
        if r > 0.0 and len(bin_indices[b_idx]) == 0:
            if b_idx < num_explicit:
                name = compiled_patterns[b_idx]["name"]
                regex_s = compiled_patterns[b_idx]["regex"]
                raise ValueError(
                    f"Weighted-bin '{name}' (regex='{regex_s}') has ratio "
                    f"{r:.6f} but matched no motion keys"
                )
            raise ValueError(
                f"Weighted-bin 'others' has ratio {r:.6f} but matched no "
                "motion keys"
            )

    # Prepare logging summary using the configured cache batch size
    raw_counts_log = [ratio * batch_size_for_log for ratio in all_ratios]
    base_counts_log = [int(c) for c in raw_counts_log]
    residuals_log = [c - int(c) for c in raw_counts_log]
    remaining = batch_size_for_log - int(sum(base_counts_log))
    if remaining != 0:
        order = sorted(
            range(len(residuals_log)),
            key=lambda i: residuals_log[i],
            reverse=True,
        )
        idx_pos = 0
        while remaining > 0:
            j = order[idx_pos % len(order)]
            base_counts_log[j] += 1
            remaining -= 1
            idx_pos += 1
    batch_fractions_log = [
        float(c) / float(batch_size_for_log) for c in base_counts_log
    ]

    # Build specs using the final, actually used batch fractions
    specs: List[Dict[str, Any]] = []
    total_items = float(max(1, num_items_total))
    for b_idx in range(num_explicit):
        name = compiled_patterns[b_idx]["name"]
        regex_s = compiled_patterns[b_idx]["regex"]
        n = len(bin_indices[b_idx])
        ds_frac = float(n) / total_items
        bf = batch_fractions_log[b_idx]
        specs.append(
            {
                "name": name,
                "regex": regex_s,
                "ratio": bf,
                "count": n,
                "dataset_fraction": ds_frac,
                "batch_fraction": bf,
            }
        )
    # Others bin
    others_name = "others"
    others_regex = "<unmatched>"
    n_o = len(bin_indices[-1])
    ds_frac_o = float(n_o) / total_items
    bf_o = batch_fractions_log[-1]
    specs.append(
        {
            "name": others_name,
            "regex": others_regex,
            "ratio": bf_o,
            "count": n_o,
            "dataset_fraction": ds_frac_o,
            "batch_fraction": bf_o,
        }
    )

    return bin_indices, all_ratios, specs


def preview_weighted_bin_from_manifest(
    manifest_path: str | Sequence[str],
    batch_size: int,
    cfg: Mapping[str, Any],
) -> None:
    """Lightweight preview of weighted-bin sampling using manifest.json only.

    This helper is intended to be called at configuration time before any
    MotionClipBatchCache/DataLoader is constructed, so that invalid regex or
    ratio settings can fail fast without incurring the cost of cache setup.
    """
    if batch_size <= 0:
        batch_size = 1

    if isinstance(manifest_path, (str, os.PathLike)):
        manifest_paths: List[str] = [str(manifest_path)]
    else:
        manifest_paths = [str(p) for p in manifest_path]
    if len(manifest_paths) == 0:
        raise ValueError("preview_weighted_bin_from_manifest requires at least one manifest path")

    key_source: Dict[str, str] = {}
    for mp in manifest_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"HDF5 manifest not found at {mp}. "
                "Please set robot.motion.hdf5_root/train_hdf5_roots "
                "to the correct path."
            )
        with open(mp, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        clips = manifest.get("clips", {})
        if not clips:
            raise ValueError(
                f"Manifest at {mp} contains no clips; cannot preview "
                "weighted-bin sampling."
            )
        for key in clips.keys():
            if key in key_source:
                raise ValueError(
                    f"Duplicate motion clip key '{key}' found in multiple "
                    "manifests; clip keys must be globally unique."
                )
            key_source[key] = mp

    keys = list(key_source.keys())
    _, _, specs = _configure_weighted_bins(
        keys=keys,
        cfg=cfg,
        batch_size_for_log=batch_size,
    )

    table_rows = []
    for item in specs:
        table_rows.append(
            [
                item["name"],
                item["regex"],
                f"{item['ratio']:.4f}",
                int(item["count"]),
                f"{item['dataset_fraction']:.4f}",
                f"{item['batch_fraction']:.4f}",
            ]
        )
    headers = [
        "bin",
        "regex",
        "final_ratio",
        "num_clips",
        "clip_fraction",
        "batch_fraction",
    ]
    logger.info(
        "Weighted-bin config preview (manifest-level):\n"
        + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
    )

class AbstractClipScorer:
    """Interface for clip score-based curriculum strategies."""

    def update(self, stats: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def probabilities(
        self, keys: List[str], step: int
    ) -> Optional[torch.Tensor]:
        """Optional: return normalized sampling probabilities for provided keys.
        If None is returned, the cache will compute probabilities by itself."""
        return None

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        """Return non-negative, unnormalized scores for the provided motion keys.
        Shape: [len(keys)]
        """
        raise NotImplementedError

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        """Notify scorer that the given keys were sampled at 'step'.
        probs: Optional vector of per-key sampling probabilities aligned with keys.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return


class AdvantageRecencyScorer(AbstractClipScorer):
    """EMA difficulty + recency bonus scorer (label-free).

    Score for key m:
      s_m = EMA_median_abs_advantage
      recency_bonus = 1 + kappa * min(1, steps_since_last/τ)
      progress_bonus = 1 + progress_beta * ema(relative_improvement)
      stagnation_decay = exp(-stagnation_beta * max(0, steps_since_improve)/stagnation_tau)
      S_m = (min(s_m, adv_cap) ** gamma) * recency_bonus * progress_bonus * stagnation_decay + epsilon
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        gamma: float = 1.5,
        kappa: float = 0.3,
        tau: int = 1000,
        epsilon: float = 1.0e-3,
        adv_cap: float = 0.0,
        progress_alpha: float = 0.1,
        progress_beta: float = 0.5,
        improve_threshold: float = 0.02,
        stagnation_tau: int = 2000,
        stagnation_beta: float = 0.5,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.tau = int(max(1, tau))
        self.epsilon = float(max(1e-12, epsilon))
        self.adv_cap = float(max(0.0, adv_cap))
        self.progress_alpha = float(progress_alpha)
        self.progress_beta = float(progress_beta)
        self.improve_threshold = float(max(0.0, improve_threshold))
        self.stagnation_tau = int(max(1, stagnation_tau))
        self.stagnation_beta = float(stagnation_beta)
        self._ema: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._progress: Dict[str, float] = {}
        self._last_improve_step: Dict[str, int] = {}

    def update(self, stats: Dict[str, float], step: int) -> None:
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = self._ema.get(k, 1.0)
            ema = (1.0 - self.alpha) * prev + self.alpha * v_f
            self._ema[k] = ema
            # Track relative improvement
            rel_improve = 0.0
            if prev > 1.0e-8:
                rel_improve = max(0.0, (prev - ema) / prev)
            prog_prev = self._progress.get(k, 0.0)
            prog_new = (
                1.0 - self.progress_alpha
            ) * prog_prev + self.progress_alpha * rel_improve
            self._progress[k] = prog_new
            if rel_improve >= self.improve_threshold:
                self._last_improve_step[k] = int(step)
            # Do not touch last_step here; only set when actually sampled

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = []
        for k in keys:
            s = float(self._ema.get(k, 1.0))
            last = self._last_step.get(k, None)
            if last is None:
                since = self.tau
            else:
                since = max(0, step - last)
            recency = 1.0 + self.kappa * min(1.0, since / float(self.tau))
            # Saturate extreme advantage to avoid bad/outlier data dominating
            s_core = max(0.0, s)
            if self.adv_cap > 0.0:
                s_core = min(s_core, self.adv_cap)
            # Progress and stagnation
            prog = float(self._progress.get(k, 0.0))
            progress_bonus = 1.0 + self.progress_beta * prog
            last_impr = self._last_improve_step.get(k, None)
            if last_impr is None:
                since_impr = self.stagnation_tau
            else:
                since_impr = max(0, step - last_impr)
            stagnation_decay = float(
                torch.exp(
                    torch.tensor(
                        -self.stagnation_beta
                        * since_impr
                        / float(self.stagnation_tau),
                        dtype=torch.float32,
                    )
                ).item()
            )
            score = (
                s_core**self.gamma
            ) * recency * progress_bonus * stagnation_decay + self.epsilon
            out.append(score)
        return torch.tensor(out, dtype=torch.float32)

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        for k in keys:
            self._last_step[k] = int(step)

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "tau": self.tau,
            "epsilon": self.epsilon,
            "adv_cap": self.adv_cap,
            "progress_alpha": self.progress_alpha,
            "progress_beta": self.progress_beta,
            "improve_threshold": self.improve_threshold,
            "stagnation_tau": self.stagnation_tau,
            "stagnation_beta": self.stagnation_beta,
            "ema": self._ema,
            "last_step": self._last_step,
            "progress": self._progress,
            "last_improve_step": self._last_improve_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        self.gamma = float(state.get("gamma", self.gamma))
        self.kappa = float(state.get("kappa", self.kappa))
        self.tau = int(state.get("tau", self.tau))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.adv_cap = float(state.get("adv_cap", self.adv_cap))
        self.progress_alpha = float(
            state.get("progress_alpha", self.progress_alpha)
        )
        self.progress_beta = float(
            state.get("progress_beta", self.progress_beta)
        )
        self.improve_threshold = float(
            state.get("improve_threshold", self.improve_threshold)
        )
        self.stagnation_tau = int(
            state.get("stagnation_tau", self.stagnation_tau)
        )
        self.stagnation_beta = float(
            state.get("stagnation_beta", self.stagnation_beta)
        )
        self._ema = dict(state.get("ema", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._progress = dict(state.get("progress", {}))
        self._last_improve_step = dict(state.get("last_improve_step", {}))


class Exp3ProgressScorer(AbstractClipScorer):
    """EXP3 over clips using learning progress as reward.

    - Keeps EMA difficulty S_t(m) from median(|adv|) (fed via update()).
    - Progress reward r_t(m) = clamp((S_{t-1}-S_t)/max(S_{t-1}, eps), 0, 1).
    - EXP3 weights w(m) updated with importance-corrected reward r̂ = r / p(m).
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.05,
        eta: float = 0.2,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self._ema: Dict[str, float] = {}
        self._log_weights: Dict[
            str, float
        ] = {}  # default lazily to 0.0 → weight=1.0
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        # record candidate set size for this step only if not set yet
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = []
        for k in keys:
            lw.append(float(self._log_weights.get(k, 0.0)))
        lw_t = torch.tensor(lw, dtype=torch.float32)
        # stable softmax over log-weights
        lw_t = lw_t - lw_t.max()
        p_core = torch.softmax(lw_t, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        p = p / p.sum()
        return p

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            # Cannot importance-correct; still record step.
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # Update EMA difficulty; compute progress and apply EXP3 updates using last-sampled p.
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = float(self._ema.get(k, 1.0))
            ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * v_f
            self._ema[k] = ema

            # progress in [0,1]
            denom = max(prev, self.eps)
            progress = max(0.0, min(1.0, (prev - ema) / denom))

            if k in self._last_p_sampled:
                p = float(self._last_p_sampled.pop(k))
                # scale update by candidate set size at sampling step (if available)
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (progress / max(p, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                # clamp to keep numbers well-behaved; probabilities computed via softmax are shift-invariant
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new
            # else: not sampled this round; only EMA is updated

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        # For compatibility: return EMA difficulty (non-negative).
        out = [float(max(0.0, self._ema.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "ema": self._ema,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self._ema = dict(state.get("ema", {}))
        # Backward compatibility: support both 'log_weights' and legacy 'weights'
        if "log_weights" in state:
            self._log_weights = dict(state.get("log_weights", {}))
        else:
            # convert legacy positive weights to log-space
            w = dict(state.get("weights", {}))
            self._log_weights = {
                k: float(
                    torch.log(torch.tensor(max(1.0e-12, float(v)))).item()
                )
                for k, v in w.items()
            }
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )


class Exp3CombinedProgressScorer(AbstractClipScorer):
    """EXP3 with combined actor+critic progress.

    - S_A(m): EMA of median(|adv|) per clip
    - S_D(m): EMA of RMS-TD per clip
    - p_A = rel_drop(S_A), p_D = rel_drop(S_D)
    - reward r = sqrt(p_A * p_D) if include_critic_progress else p_A
    - EXP3 update on log-weights with IPS and |K|-scaled step size
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        eta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
        include_critic_progress: bool = True,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.include_critic_progress = bool(include_critic_progress)
        self._ema_adv: Dict[str, float] = {}
        self._ema_td: Dict[str, float] = {}
        self._log_weights: Dict[str, float] = {}
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}
        # last-step diagnostics
        self._last_prog_a: Dict[str, float] = {}
        self._last_prog_d: Dict[str, float] = {}
        self._last_reward: Dict[str, float] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = torch.tensor(
            [float(self._log_weights.get(k, 0.0)) for k in keys],
            dtype=torch.float32,
        )
        lw = lw - lw.max()
        p_core = torch.softmax(lw, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        return p / p.sum()

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # Backward-compatible: use actor progress only
        self.update_combined(stats, {}, step)

    def update_combined(
        self,
        adv_stats: Dict[str, float],
        td_stats: Dict[str, float],
        step: int,
    ) -> None:
        keys = set(adv_stats.keys()) | set(td_stats.keys())
        for k in keys:
            # Update EMAs
            if k in adv_stats:
                v_a = float(max(0.0, adv_stats[k]))
                prev_a = float(self._ema_adv.get(k, 1.0))
                ema_a = (1.0 - self.ema_alpha) * prev_a + self.ema_alpha * v_a
                self._ema_adv[k] = ema_a
                denom_a = max(prev_a, self.eps)
                p_a = max(0.0, min(1.0, (prev_a - ema_a) / denom_a))
                self._last_prog_a[k] = p_a
            else:
                p_a = float(self._last_prog_a.get(k, 0.0))

            if k in td_stats:
                v_d = float(max(0.0, td_stats[k]))
                prev_d = float(self._ema_td.get(k, 1.0))
                ema_d = (1.0 - self.ema_alpha) * prev_d + self.ema_alpha * v_d
                self._ema_td[k] = ema_d
                denom_d = max(prev_d, self.eps)
                p_d = max(0.0, min(1.0, (prev_d - ema_d) / denom_d))
                self._last_prog_d[k] = p_d
            else:
                p_d = float(self._last_prog_d.get(k, 0.0))

            # Combined reward
            if self.include_critic_progress:
                r = float(torch.sqrt(torch.tensor(p_a * p_d)).item())
            else:
                r = p_a
            self._last_reward[k] = r

            # EXP3 log-weight update if sampled
            if k in self._last_p_sampled:
                p = float(self._last_p_sampled.pop(k))
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (r / max(p, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = [float(max(0.0, self._ema_adv.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "include_critic_progress": self.include_critic_progress,
            "ema_adv": self._ema_adv,
            "ema_td": self._ema_td,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
            "last_prog_a": self._last_prog_a,
            "last_prog_d": self._last_prog_d,
            "last_reward": self._last_reward,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self.include_critic_progress = bool(
            state.get("include_critic_progress", self.include_critic_progress)
        )
        self._ema_adv = dict(state.get("ema_adv", {}))
        self._ema_td = dict(state.get("ema_td", {}))
        self._log_weights = dict(state.get("log_weights", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )
        self._last_prog_a = dict(state.get("last_prog_a", {}))
        self._last_prog_d = dict(state.get("last_prog_d", {}))
        self._last_reward = dict(state.get("last_reward", {}))


class Exp3PoseProgressScorer(AbstractClipScorer):
    """EXP3 with pose-error (MPJPE/MPKPE) progress as reward.

    - S_P(m): EMA of combined pose error per clip (provided by caller)
    - p_P = rel_drop(S_P) clipped to [0, progress_clip]
    - reward r = p_P
    - EXP3 update on log-weights with IPS and |K|-scaled step size
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        eta: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1.0e-6,
        progress_clip: float = 0.25,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.progress_clip = float(max(0.0, progress_clip))
        self._ema_pose: Dict[str, float] = {}
        self._log_weights: Dict[str, float] = {}
        self._last_p_sampled: Dict[str, float] = {}
        self._last_step: Dict[str, int] = {}
        self._population_size_by_step: Dict[int, int] = {}
        self._last_prog_p: Dict[str, float] = {}
        self._last_reward: Dict[str, float] = {}

    def probabilities(self, keys: List[str], step: int) -> torch.Tensor:
        if len(keys) == 0:
            return torch.zeros(0, dtype=torch.float32)
        if step not in self._population_size_by_step:
            self._population_size_by_step[step] = len(keys)
        lw = torch.tensor(
            [float(self._log_weights.get(k, 0.0)) for k in keys],
            dtype=torch.float32,
        )
        lw = lw - lw.max()
        p_core = torch.softmax(lw, dim=0)
        if self.gamma > 0.0:
            uni = torch.full_like(p_core, 1.0 / len(keys))
            p = (1.0 - self.gamma) * p_core + self.gamma * uni
        else:
            p = p_core
        p = torch.clamp(p, min=1.0e-12)
        return p / p.sum()

    def on_sampled(
        self, keys: List[str], step: int, probs: Optional[torch.Tensor] = None
    ) -> None:
        if probs is None:
            for k in keys:
                self._last_step[k] = int(step)
            return
        probs = probs.detach().cpu().float()
        for i, k in enumerate(keys):
            self._last_p_sampled[k] = float(
                max(1.0e-12, float(probs[i].item()))
            )
            self._last_step[k] = int(step)

    def update(self, stats: Dict[str, float], step: int) -> None:
        # stats: per-key combined pose error (non-negative)
        for k, v in stats.items():
            v_f = float(max(0.0, v))
            prev = float(self._ema_pose.get(k, 1.0))
            ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * v_f
            self._ema_pose[k] = ema
            denom = max(prev, self.eps)
            p = (prev - ema) / denom
            # clip symmetric to be robust to noise (allow minor negative)
            p = float(max(-self.progress_clip, min(self.progress_clip, p)))
            # reward is non-negative progress only
            r = float(max(0.0, p))
            self._last_prog_p[k] = (
                r  # store non-negative progress actually rewarded
            )
            self._last_reward[k] = r
            if k in self._last_p_sampled:
                prob = float(self._last_p_sampled.pop(k))
                k_step = int(self._last_step.get(k, step))
                pop_size = int(self._population_size_by_step.get(k_step, 1))
                scale = self.eta / max(1, pop_size)
                delta = scale * (r / max(prob, 1.0e-12))
                lw_old = float(self._log_weights.get(k, 0.0))
                lw_new = lw_old + float(delta)
                if lw_new > 50.0:
                    lw_new = 50.0
                elif lw_new < -50.0:
                    lw_new = -50.0
                self._log_weights[k] = lw_new

    def scores(self, keys: List[str], step: int) -> torch.Tensor:
        out = [float(max(0.0, self._ema_pose.get(k, 1.0))) for k in keys]
        return torch.tensor(out, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "ema_alpha": self.ema_alpha,
            "eta": self.eta,
            "gamma": self.gamma,
            "eps": self.eps,
            "progress_clip": self.progress_clip,
            "ema_pose": self._ema_pose,
            "log_weights": self._log_weights,
            "last_step": self._last_step,
            "population_size_by_step": self._population_size_by_step,
            "last_prog_p": self._last_prog_p,
            "last_reward": self._last_reward,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.eta = float(state.get("eta", self.eta))
        self.gamma = float(state.get("gamma", self.gamma))
        self.eps = float(state.get("eps", self.eps))
        self.progress_clip = float(
            state.get("progress_clip", self.progress_clip)
        )
        self._ema_pose = dict(state.get("ema_pose", {}))
        self._log_weights = dict(state.get("log_weights", {}))
        self._last_step = dict(state.get("last_step", {}))
        self._population_size_by_step = dict(
            state.get("population_size_by_step", {})
        )
        self._last_prog_p = dict(state.get("last_prog_p", {}))
        self._last_reward = dict(state.get("last_reward", {}))


MANDATORY_DATASETS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "rg_pos": "global_translation",
    "rb_rot": "global_rotation_quat",
    "body_vel": "global_velocity",
    "body_ang_vel": "global_angular_velocity",
}


def _normalize_window_world_frame(arrays: Dict[str, Tensor]) -> None:
    """Normalize a motion window into a canonical z-up world frame in-place.

    Behavior:
    - Uses the canonical root (body 0) at frame 0 to:
      - Subtract its XY position from all body positions (Z is unchanged).
      - Remove its yaw around +Z from all body orientations.
    - Applies the same SE(3) transform to:
      - Positions: rg_pos[...]
      - Rotations: rb_rot[...]
      - Linear velocities: body_vel[...]
      - Angular velocities: body_ang_vel[...]
    - If prefixed variants (robot_, ref_, ft_ref_) exist, applies the same
      transform to them as well.
    """
    if "rg_pos" not in arrays or "rb_rot" not in arrays:
        return
    if "body_vel" not in arrays or "body_ang_vel" not in arrays:
        return

    rg_pos = arrays["rg_pos"]
    rb_rot = arrays["rb_rot"]
    body_vel = arrays["body_vel"]
    body_ang_vel = arrays["body_ang_vel"]

    if rg_pos.ndim != 3 or rb_rot.ndim != 3:
        return
    if body_vel.ndim != 3 or body_ang_vel.ndim != 3:
        return
    if rg_pos.shape[0] == 0 or rg_pos.shape[1] == 0:
        return

    # Root pose at frame 0, body 0 (XYZW quaternion, z-up).
    p_root0 = rg_pos[0, 0]  # [3]
    q_root0 = rb_rot[0, 0]  # [4]

    # Center XY so that root XY at frame 0 becomes (0, 0).
    offset_xy = p_root0.clone()
    offset_xy[2] = 0.0
    rg_pos[..., 0] -= offset_xy[0]
    rg_pos[..., 1] -= offset_xy[1]

    # Extract yaw from q_root0 (XYZW) using z-up convention.
    x = q_root0[0]
    y = q_root0[1]
    z = q_root0[2]
    w = q_root0[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = w * w + x * x - y * y - z * z
    yaw0 = torch.atan2(siny_cosp, cosy_cosp)

    # Quaternion for rotation around +Z by -yaw0 (remove initial heading).
    half = -0.5 * yaw0
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    q_heading_inv = torch.stack(
        [
            torch.zeros_like(sin_half),
            torch.zeros_like(sin_half),
            sin_half,
            cos_half,
        ],
        dim=-1,
    )  # [4], XYZW

    T, B, _ = rg_pos.shape
    q_flat = q_heading_inv.view(1, 1, 4).expand(T, B, 4).reshape(-1, 4)

    def _quat_rotate_xyzw(q: Tensor, v: Tensor) -> Tensor:
        # q: [..., 4] (XYZW), v: [..., 3]
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        qx = q[..., 0]
        qy = q[..., 1]
        qz = q[..., 2]
        qw = q[..., 3]
        # v' = v + 2 * q_vec x (q_vec x v + w * v)
        uvx = qy * vz - qz * vy
        uvy = qz * vx - qx * vz
        uvz = qx * vy - qy * vx
        uuvx = qy * uvz - qz * uvy
        uuvy = qz * uvx - qx * uvz
        uuvz = qx * uvy - qy * uvx
        scale = 2.0 * qw
        uvx = uvx * scale
        uvy = uvy * scale
        uvz = uvz * scale
        uuvx = uuvx * 2.0
        uuvy = uuvy * 2.0
        uuvz = uuvz * 2.0
        rx = vx + uvx + uuvx
        ry = vy + uvy + uuvy
        rz = vz + uvz + uuvz
        return torch.stack([rx, ry, rz], dim=-1)

    def _quat_mul_xyzw(q1: Tensor, q2: Tensor) -> Tensor:
        # q1, q2: [..., 4] (XYZW)
        x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
        x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w], dim=-1)

    def _apply_to_set(prefix: str) -> None:
        pos_key = f"{prefix}rg_pos" if prefix else "rg_pos"
        rot_key = f"{prefix}rb_rot" if prefix else "rb_rot"
        vel_key = f"{prefix}body_vel" if prefix else "body_vel"
        ang_key = f"{prefix}body_ang_vel" if prefix else "body_ang_vel"
        if (
            pos_key not in arrays
            or rot_key not in arrays
            or vel_key not in arrays
            or ang_key not in arrays
        ):
            return
        pos = arrays[pos_key]
        rot = arrays[rot_key]
        vel = arrays[vel_key]
        ang = arrays[ang_key]
        if pos.shape != rg_pos.shape or rot.shape != rb_rot.shape:
            return
        # Center XY using canonical offset.
        pos[..., 0] -= offset_xy[0]
        pos[..., 1] -= offset_xy[1]
        # Rotate vectors.
        pos_flat = pos.reshape(-1, 3)
        vel_flat = vel.reshape(-1, 3)
        ang_flat = ang.reshape(-1, 3)
        pos[:] = _quat_rotate_xyzw(q_flat, pos_flat).reshape_as(pos)
        vel[:] = _quat_rotate_xyzw(q_flat, vel_flat).reshape_as(vel)
        ang[:] = _quat_rotate_xyzw(q_flat, ang_flat).reshape_as(ang)
        # Rotate orientations.
        rot_flat = rot.reshape(-1, 4)
        rot[:] = _quat_mul_xyzw(q_flat, rot_flat).reshape_as(rot)

    _apply_to_set("")
    for pfx in ["robot_", "ref_", "ft_ref_"]:
        _apply_to_set(pfx)


@dataclass
class MotionWindow:
    """Metadata describing a contiguous motion window within an HDF5 shard."""

    motion_key: str  # unique per window
    shard_index: int
    start: int
    length: int
    raw_motion_key: str  # original clip key
    window_index: int


@dataclass
class MotionClipSample:
    """In-memory representation of a motion window.

    Attributes:
        motion_key: Unique window identifier (includes slice info).
        raw_motion_key: Original clip identifier from manifest.
        tensors: Mapping from tensor name to data tensor of shape
            ``[window_length, ...]`` (float32 unless specified otherwise).
        length: Number of valid frames contained in the sample (``<=``
            ``max_frame_length``).
    """

    motion_key: str
    raw_motion_key: str
    tensors: Dict[str, Tensor]
    length: int


@dataclass
class ClipBatch:
    """Batch of motion clips ready for consumption by the environment.

    Attributes:
        tensors: Mapping from tensor name to tensor with shape
            ``[batch_size, max_frame_length, ...]`` placed on the staging
            device.
        lengths: Valid frame counts per clip ``[batch_size]``.
        motion_keys: List of motion keys corresponding to each clip.
        max_frame_length: Fixed length configured for the cache.
    """

    tensors: Dict[str, Tensor]
    lengths: Tensor
    motion_keys: List[str]
    raw_motion_keys: List[str]
    max_frame_length: int

    @staticmethod
    def collate_fn(samples: List[MotionClipSample]) -> "ClipBatch":
        if len(samples) == 0:
            raise ValueError(
                "ClipBatch collate_fn received an empty sample list"
            )

        max_frame_length = max(
            sample.tensors["dof_pos"].shape[0] for sample in samples
        )
        max_frame_length = int(max_frame_length)

        batched_tensors: Dict[str, Tensor] = {}
        lengths = torch.zeros(len(samples), dtype=torch.long)
        motion_keys = []
        raw_motion_keys = []

        for batch_idx, sample in enumerate(samples):
            lengths[batch_idx] = sample.length
            motion_keys.append(sample.motion_key)
            raw_motion_keys.append(sample.raw_motion_key)

            for name, tensor in sample.tensors.items():
                if name not in batched_tensors:
                    pad_shape = (
                        len(samples),
                        max_frame_length,
                    ) + tensor.shape[1:]
                    batched_tensors[name] = torch.zeros(
                        pad_shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                target = batched_tensors[name]
                valid_frames = sample.length
                target[batch_idx, :valid_frames] = tensor

                if valid_frames < max_frame_length and valid_frames > 0:
                    target[batch_idx, valid_frames:] = tensor[valid_frames - 1]

        return ClipBatch(
            tensors=batched_tensors,
            lengths=lengths,
            motion_keys=motion_keys,
            raw_motion_keys=raw_motion_keys,
            max_frame_length=max_frame_length,
        )


def _cache_collate_fn(
    samples: List[MotionClipSample],
    mode: str,
    batch_size: int,
) -> ClipBatch:
    """Collate function for motion cache DataLoader (supports validation padding)."""
    if (
        mode == "val"
        and batch_size > len(samples)
        and len(samples) > 0
    ):
        extra = batch_size - len(samples)
        gen = torch.Generator()
        idx = torch.randint(0, len(samples), size=(extra,), generator=gen)
        padded = list(samples)
        for i in idx.tolist():
            padded.append(samples[i])
        return ClipBatch.collate_fn(padded)
    return ClipBatch.collate_fn(samples)


class InfiniteDistributedSampler(DistributedSampler):
    """Distributed sampler that yields an infinite stream by cycling epochs."""

    def __iter__(self):
        # Infinite stream by cycling epochs
        while True:
            self.set_epoch(getattr(self, "_epoch", 0))
            for idx in super().__iter__():
                yield idx
            self._epoch = getattr(self, "_epoch", 0) + 1


class InfiniteRandomSampler(Sampler[int]):
    """Random sampler that yields infinite reshuffled passes over the dataset."""

    def __init__(self, data_source: Dataset, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        # Yield infinite permutations of indices
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.data_source), generator=g)
            for idx in perm.tolist():
                yield int(idx)
            self.epoch += 1

    def __len__(self) -> int:
        # Large sentinel to satisfy components that query length
        return 2**31 - 1


class WeightedBinInfiniteSampler(Sampler[int]):
    """Infinite sampler that respects regex-based weighted bins over indices."""

    def __init__(
        self,
        dataset_len: int,
        bin_indices: List[List[int]],
        ratios: List[float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._ds_len = int(max(0, dataset_len))
        self._bins = [torch.tensor(b, dtype=torch.long) for b in bin_indices]
        self._ratios = list(ratios)
        self._batch_size = int(max(1, batch_size))
        self._seed = int(seed)
        self._epoch = 0

        raw_counts = [r * float(self._batch_size) for r in self._ratios]
        base_counts = [int(c) for c in raw_counts]
        residuals = [c - int(c) for c in raw_counts]
        remaining = self._batch_size - int(sum(base_counts))
        if remaining != 0:
            order = sorted(
                range(len(residuals)),
                key=lambda i: residuals[i],
                reverse=True,
            )
            idx_pos = 0
            while remaining > 0:
                j = order[idx_pos % len(order)]
                base_counts[j] += 1
                remaining -= 1
                idx_pos += 1
        self._counts = [max(0, int(c)) for c in base_counts]

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            batch: List[int] = []
            for bin_idx, count in zip(self._bins, self._counts):
                if count <= 0 or bin_idx.numel() == 0:
                    continue
                choice = torch.randint(
                    0,
                    int(bin_idx.numel()),
                    size=(count,),
                    generator=g,
                )
                selected = bin_idx[choice].tolist()
                batch.extend(int(x) for x in selected)

            if not batch:
                # Fallback: uniform over dataset indices
                if self._ds_len == 0:
                    raise ValueError(
                        "WeightedBinInfiniteSampler cannot sample from an empty dataset"
                    )
                all_idx = torch.randint(
                    0,
                    self._ds_len,
                    size=(self._batch_size,),
                    generator=g,
                )
                batch = [int(x) for x in all_idx.tolist()]

            if len(batch) > self._batch_size:
                batch = batch[: self._batch_size]
            elif len(batch) < self._batch_size:
                pad = self._batch_size - len(batch)
                if pad > 0:
                    batch.extend(batch[:pad])

            perm = torch.randperm(len(batch), generator=g)
            for idx in perm.tolist():
                yield int(batch[idx])
            self._epoch += 1

    def __len__(self) -> int:
        return 2**31 - 1


class Hdf5MotionDataset(Dataset[MotionClipSample]):
    """Dataset that materializes fixed-length motion windows from HDF5 shards."""

    def __init__(
        self,
        manifest_path: str | Sequence[str],
        max_frame_length: int,
        min_window_length: int = 1,
        handpicked_motion_names: Optional[List[str]] = None,
        excluded_motion_names: Optional[List[str]] = None,
        world_frame_normalization: bool = True,
        allowed_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")

        self.max_frame_length = int(max_frame_length)
        self.min_window_length = int(min_window_length)
        self.handpicked_motion_names = (
            set(handpicked_motion_names)
            if handpicked_motion_names is not None
            else None
        )
        self.excluded_motion_names = (
            set(excluded_motion_names)
            if excluded_motion_names is not None
            else None
        )
        self._world_frame_normalization_enabled = bool(world_frame_normalization)
        self._allowed_prefixes: Optional[Tuple[str, ...]] = (
            tuple(allowed_prefixes) if allowed_prefixes is not None else None
        )

        # Normalize manifest path(s) to a list for aggregation.
        if isinstance(manifest_path, (str, os.PathLike)):
            manifest_paths: List[str] = [str(manifest_path)]
        else:
            manifest_paths = [str(p) for p in manifest_path]
        if len(manifest_paths) == 0:
            raise ValueError("At least one manifest_path must be provided")

        # Aggregate shards and clips across one or many manifests into a single
        # logical dataset. Clip keys must be globally unique.
        self.hdf5_root = os.path.dirname(manifest_paths[0])
        self._manifest_paths: List[str] = manifest_paths
        self._shard_paths: List[str] = []
        self.shards: List[Dict[str, Any]] = []
        self.clips: Dict[str, Dict[str, Any]] = {}

        for mp in manifest_paths:
            if not os.path.exists(mp):
                raise FileNotFoundError(
                    f"HDF5 manifest not found at {mp}. "
                    "Please set robot.motion.hdf5_root/train_hdf5_roots "
                    "to the correct path."
                )
            with open(mp, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            root = os.path.dirname(mp)
            shards_local = list(manifest.get("hdf5_shards", []))
            clips_local = manifest.get("clips", {})

            shard_offset = len(self.shards)
            for shard_meta in shards_local:
                self.shards.append(shard_meta)
                rel = shard_meta.get("file", None)
                if not isinstance(rel, str) or not rel:
                    raise ValueError(
                        f"Shard entry in manifest {mp} is missing a valid 'file' field"
                    )
                self._shard_paths.append(os.path.join(root, rel))

            for key, meta in clips_local.items():
                if key in self.clips:
                    raise ValueError(
                        f"Duplicate motion clip key '{key}' found in multiple "
                        "manifests; clip keys must be globally unique."
                    )
                meta_global = dict(meta)
                meta_global["shard"] = int(meta_global.get("shard", 0)) + shard_offset
                self.clips[key] = meta_global

        if len(self.shards) == 0:
            raise ValueError(
                f"No HDF5 shards listed in manifests: {', '.join(manifest_paths)}"
            )

        self.windows: List[MotionWindow] = self._enumerate_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No motion windows satisfy the requested frame length constraints"
            )

        # LRU cache of open HDF5 shard handles; size is bounded to avoid
        # unbounded host-memory usage from per-file raw chunk caches.
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        if max_open_env is None:
            self._max_open_files = 64
        else:
            self._max_open_files = max(1, int(max_open_env))

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
            if (
                self.handpicked_motion_names is not None
                and motion_key not in self.handpicked_motion_names
            ):
                continue
            if (
                self.excluded_motion_names is not None
                and motion_key in self.excluded_motion_names
            ):
                continue

            shard_index = int(meta.get("shard", 0))
            start = int(meta.get("start", 0))
            length = int(meta.get("length", 0))

            if length <= 0:
                continue

            remaining = length
            offset = 0
            window_index = 0
            while remaining > 0:
                window_length = min(self.max_frame_length, remaining)
                if window_length >= self.min_window_length:
                    win_start = start + offset
                    unique_key = (
                        f"{motion_key}__start_{win_start}_len_{window_length}"
                    )
                    windows.append(
                        MotionWindow(
                            motion_key=unique_key,
                            shard_index=shard_index,
                            start=win_start,
                            length=window_length,
                            raw_motion_key=motion_key,
                            window_index=window_index,
                        )
                    )
                offset += window_length
                remaining = max(0, length - offset)
                window_index += 1

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length

        arrays: Dict[str, Tensor] = {}

        # Helper: choose first-present dataset name from candidates
        def _first_present(candidates: List[str]) -> Optional[str]:
            for name in candidates:
                if name in shard_handle:
                    return name
            return None

        # Prefer canonical datasets; fall back to prefixed aliases to be robust to legacy/new shards
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            candidates: List[str] = [dataset_name]
            # Backward-compat aliases and prefixed variants
            if logical_name == "dof_vel":
                # legacy alias
                candidates.append("dof_vels")
            # prefixed variants (future-proof)
            candidates.extend(
                [
                    f"robot_{dataset_name}",
                    f"ref_{dataset_name}",
                    f"ft_ref_{dataset_name}",
                ]
            )
            chosen = _first_present(candidates)
            if chosen is None:
                raise KeyError(
                    f"Dataset '{dataset_name}' (or aliases {candidates[1:]}) missing in shard index {window.shard_index}"
                )
            np_array = shard_handle[chosen][start:end]
            arrays[logical_name] = torch.from_numpy(np_array).to(torch.float32)

        # Optionally read additional prefixed datasets when present to make multiple sources available at runtime
        _prefixes = (
            list(self._allowed_prefixes)
            if getattr(self, "_allowed_prefixes", None) is not None
            else ["robot_", "ref_", "ft_ref_"]
        )
        for pfx in _prefixes:
            for logical_name, dataset_name in MANDATORY_DATASETS.items():
                dname = f"{pfx}{dataset_name}"
                if dname in shard_handle:
                    np_array = shard_handle[dname][start:end]
                    arrays[f"{pfx}{logical_name}"] = torch.from_numpy(
                        np_array
                    ).to(torch.float32)

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 0:
                frame_flag[0] = 0
                frame_flag[-1] = 2
        arrays["frame_flag"] = frame_flag

        if self._world_frame_normalization_enabled:
            _normalize_window_world_frame(arrays)

        # Derived root_* for canonical source (after normalization)
        arrays["root_pos"] = arrays["rg_pos"][:, 0, :]
        arrays["root_rot"] = arrays["rb_rot"][:, 0, :]
        arrays["root_vel"] = arrays["body_vel"][:, 0, :]
        arrays["root_ang_vel"] = arrays["body_ang_vel"][:, 0, :]

        # Derived root_* for optional prefixed sources if present (after normalization)
        for pfx in _prefixes:
            rg_key = f"{pfx}rg_pos"
            rb_key = f"{pfx}rb_rot"
            bv_key = f"{pfx}body_vel"
            bav_key = f"{pfx}body_ang_vel"
            if (
                rg_key in arrays
                and rb_key in arrays
                and bv_key in arrays
                and bav_key in arrays
            ):
                arrays[f"{pfx}root_pos"] = arrays[rg_key][:, 0, :]
                arrays[f"{pfx}root_rot"] = arrays[rb_key][:, 0, :]
                arrays[f"{pfx}root_vel"] = arrays[bv_key][:, 0, :]
                arrays[f"{pfx}root_ang_vel"] = arrays[bav_key][:, 0, :]

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                # Mark as most recently used.
                self._file_handles[shard_index] = handle
                return handle

        if shard_index < 0 or shard_index >= len(self._shard_paths):
            raise IndexError(
                f"Shard index {shard_index} out of range for "
                f"{len(self._shard_paths)} available shards"
            )
        shard_path = self._shard_paths[shard_index]
        # Open with SWMR and a configurable raw chunk cache to speed up repeated reads.
        # The default cache size (in bytes) can be overridden via the
        # HOLOMOTION_HDF5_RDCC_NBYTES environment variable.
        rdcc_nbytes_env = os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES")
        if rdcc_nbytes_env is None:
            rdcc_nbytes = 256 * 1024 * 1024  # 256MB default
        else:
            rdcc_nbytes = int(rdcc_nbytes_env)
        handle = h5py.File(
            shard_path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=rdcc_nbytes,
            rdcc_w0=0.75,
        )
        # Enforce LRU limit on the number of simultaneously open shard files.
        if (
            self._max_open_files is not None
            and len(self._file_handles) >= self._max_open_files
        ):
            old_index, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        """Close all open HDF5 shard handles for this dataset."""
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()


class MotionClipBatchCache:
    """Double-buffered motion cache for RL training and evaluation."""

    def __init__(
        self,
        train_dataset: Hdf5MotionDataset,
        *,
        val_dataset: Optional[Hdf5MotionDataset] = None,
        batch_size: int,
        stage_device: Optional[torch.device] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler_rank: int = 0,
        sampler_world_size: int = 1,
        allowed_prefixes: Optional[Sequence[str]] = None,
        swap_interval_steps: Optional[int] = None,
        force_timeout_on_swap: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._datasets = {
            "train": train_dataset,
            "val": val_dataset if val_dataset is not None else train_dataset,
        }
        self._mode = "train"
        self._seed = int(time.time_ns() & 0x7FFFFFFF)
        self._stage_device = stage_device
        self._sampler_rank = int(sampler_rank)
        self._sampler_world_size = int(max(1, sampler_world_size))
        self._batch_size = int(batch_size)
        self._allowed_prefixes: Optional[Tuple[str, ...]] = (
            tuple(allowed_prefixes) if allowed_prefixes is not None else None
        )

        self.swap_interval_steps = (
            swap_interval_steps
            if swap_interval_steps is not None
            else train_dataset.max_frame_length
        )
        self.force_timeout_on_swap = force_timeout_on_swap

        self._num_workers = int(max(0, num_workers))
        self._prefetch_factor = (
            prefetch_factor if prefetch_factor is not None else None
        )
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers and num_workers > 0)

        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[DistributedSampler] = None
        self._iterator: Optional[Iterator[ClipBatch]] = None

        self._current_batch: Optional[ClipBatch] = None
        self._next_batch: Optional[ClipBatch] = None
        self._swap_index = 0

        self._effective_batch_size: Optional[int] = None
        self._num_batches: Optional[int] = None

        # Curriculum (clip-scorer) state
        self._curriculum_enabled: bool = False
        self._scorer: Optional[AbstractClipScorer] = None
        self._cur_uniform_ratio: float = 0.1
        self._cur_temperature: float = 0.7
        self._step_counter: int = 0
        self._last_unique_count: int = 0

        # Weighted-bin sampling state
        self._weighted_bin_enabled: bool = False
        self._weighted_bin_bins: Optional[List[List[int]]] = None
        self._weighted_bin_ratios: Optional[List[float]] = None
        self._weighted_bin_specs: Optional[List[Dict[str, Any]]] = None

        # Async GPU staging helpers
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        self._build_dataloader()
        if self._stage_device is not None and (
            getattr(self._stage_device, "type", None) == "cuda"
            or (
                isinstance(self._stage_device, str)
                and self._stage_device.startswith("cuda")
            )
        ):
            import torch.cuda

            try:
                # Normalize to device index and set context explicitly
                if isinstance(self._stage_device, torch.device):
                    dev_index = (
                        0
                        if self._stage_device.index is None
                        else int(self._stage_device.index)
                    )
                elif isinstance(
                    self._stage_device, str
                ) and self._stage_device.startswith("cuda"):
                    parts = self._stage_device.split(":")
                    dev_index = (
                        int(parts[1])
                        if len(parts) > 1
                        else torch.cuda.current_device()
                    )
                else:
                    dev_index = torch.cuda.current_device()
                torch.cuda.set_device(dev_index)
                self._copy_stream = torch.cuda.Stream()
                # logger.info(
                #     f"Perf/Cache: created CUDA copy stream on cuda:{dev_index}"
                # )
            except Exception as e:
                logger.warning(
                    f"Perf/Cache: failed to create CUDA copy stream ({self._stage_device}): {e}"
                )
        self._prime_buffers()

    @property
    def current_batch(self) -> ClipBatch:
        assert self._current_batch is not None
        return self._current_batch

    @property
    def max_frame_length(self) -> int:
        return self.current_batch.max_frame_length

    @property
    def clip_count(self) -> int:
        return self.current_batch.lengths.shape[0]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def swap_index(self) -> int:
        return self._swap_index

    @property
    def num_batches(self) -> int:
        if self._num_batches is None:
            raise RuntimeError("DataLoader is not initialised")
        return int(self._num_batches)

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        if mode not in self._datasets:
            raise ValueError(f"Unknown cache mode: {mode}")
        self._mode = mode
        self._build_dataloader()
        self._prime_buffers()

    def advance(self) -> None:
        if self._next_batch is None:
            self._next_batch = self._fetch_next_batch()
        # Ensure asynchronous staging finished before swapping in next batch
        if (
            self._next_ready_event is not None
            and self._stage_device is not None
            and getattr(self._stage_device, "type", None) == "cuda"
        ):
            import torch.cuda

            torch.cuda.current_stream(self._stage_device).wait_event(
                self._next_ready_event
            )
        self._current_batch = self._next_batch
        self._next_batch = self._fetch_next_batch()
        self._swap_index += 1

    # -------------------------
    # Curriculum configuration
    # -------------------------
    def enable_curriculum(
        self,
        cfg: Optional[Dict[str, float]] = None,
        scorer: Optional[AbstractClipScorer] = None,
    ) -> None:
        """Enable clip-score-based curriculum with a pluggable scorer."""
        self._curriculum_enabled = True
        if scorer is None:
            scorer_name = str((cfg or {}).get("scorer", "adv_recency")).lower()
            alpha = float((cfg or {}).get("alpha", 0.05))
            gamma = float((cfg or {}).get("gamma", 1.5))
            kappa = float((cfg or {}).get("kappa", 0.3))
            tau = int((cfg or {}).get("tau", 1000))
            epsilon = float((cfg or {}).get("epsilon", 1.0e-3))
            adv_cap = float((cfg or {}).get("adv_cap", 0.0))
            progress_alpha = float((cfg or {}).get("progress_alpha", 0.1))
            progress_beta = float((cfg or {}).get("progress_beta", 0.5))
            improve_threshold = float(
                (cfg or {}).get("improve_threshold", 0.02)
            )
            stagnation_tau = int((cfg or {}).get("stagnation_tau", 2000))
            stagnation_beta = float((cfg or {}).get("stagnation_beta", 0.5))
            if scorer_name == "exp3_progress":
                ema_alpha = float((cfg or {}).get("ema_alpha", alpha))
                exp3_eta = float((cfg or {}).get("exp3_eta", 0.2))
                exp3_gamma = float((cfg or {}).get("exp3_gamma", 0.1))
                include_critic_progress = bool(
                    (cfg or {}).get("include_critic_progress", False)
                )
                if include_critic_progress:
                    self._scorer = Exp3CombinedProgressScorer(
                        ema_alpha=ema_alpha,
                        eta=exp3_eta,
                        gamma=exp3_gamma,
                        include_critic_progress=True,
                    )
                else:
                    self._scorer = Exp3ProgressScorer(
                        ema_alpha=ema_alpha, eta=exp3_eta, gamma=exp3_gamma
                    )
            elif scorer_name == "exp3_pose_progress":
                ema_alpha = float((cfg or {}).get("ema_alpha", 0.2))
                exp3_eta = float((cfg or {}).get("exp3_eta", 0.5))
                exp3_gamma = float((cfg or {}).get("exp3_gamma", 0.1))
                progress_clip = float((cfg or {}).get("progress_clip", 0.25))
                self._scorer = Exp3PoseProgressScorer(
                    ema_alpha=ema_alpha,
                    eta=exp3_eta,
                    gamma=exp3_gamma,
                    progress_clip=progress_clip,
                )
            else:
                self._scorer = AdvantageRecencyScorer(
                    alpha=alpha,
                    gamma=gamma,
                    kappa=kappa,
                    tau=tau,
                    epsilon=epsilon,
                    adv_cap=adv_cap,
                    progress_alpha=progress_alpha,
                    progress_beta=progress_beta,
                    improve_threshold=improve_threshold,
                    stagnation_tau=stagnation_tau,
                    stagnation_beta=stagnation_beta,
                )
        else:
            self._scorer = scorer
        self._cur_uniform_ratio = float((cfg or {}).get("uniform_ratio", 0.1))
        self._cur_temperature = float((cfg or {}).get("temperature", 0.7))

    def disable_curriculum(self) -> None:
        self._curriculum_enabled = False
        self._scorer = None

    # -------------------------
    # Weighted-bin configuration
    # -------------------------
    def enable_weighted_bin_sampling(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enable regex-based weighted-bin sampling over manifest motion keys.

        The configuration must provide a list under ``bin_regex_patterns`` (or the
        legacy name ``bin_regrex_patterns``), where each element is a mapping with:

        - ``regex`` (or ``regrex``): Python regular expression applied to the
          manifest clip key (e.g., ``AMASS_.*``, ``VR_pico_.*``).
        - ``ratio``: Target sampling ratio in [0, 1].

        The sum of explicit bin ratios must be <= 1.0. Any remaining mass is
        assigned to an implicit ``others`` bin that collects all clips not
        matched by any regex.
        """
        cfg_local: Dict[str, Any] = dict(cfg or {})

        dataset = self._datasets.get("train")
        if dataset is None:
            raise ValueError(
                "Weighted-bin sampling requires a training dataset"
            )

        # Collect manifest-level motion keys for all windows in order
        window_keys: List[str] = []
        for window in dataset.windows:
            motion_key = getattr(window, "raw_motion_key", None)
            if motion_key is None:
                full_key = getattr(window, "motion_key", "")
                if "__start_" in full_key:
                    motion_key = full_key.split("__start_", 1)[0]
                else:
                    motion_key = full_key
            window_keys.append(motion_key)

        bin_indices, all_ratios, specs = _configure_weighted_bins(
            keys=window_keys,
            cfg=cfg_local,
            batch_size_for_log=int(self._batch_size),
        )

        # Log summary in terms of windows
        table_rows = []
        for item in specs:
            table_rows.append(
                [
                    item["name"],
                    item["regex"],
                    f"{item['ratio']:.4f}",
                    int(item["count"]),
                    f"{item['dataset_fraction']:.4f}",
                    f"{item['batch_fraction']:.4f}",
                ]
            )
        headers = [
            "bin",
            "regex",
            "final_ratio",
            "num_windows",
            "dataset_fraction",
            "batch_fraction",
        ]
        logger.info(
            "Motion cache weighted-bin sampling configured:\n"
            + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
        )

        # Activate weighted-bin sampling and rebuild dataloader/cache
        self._weighted_bin_enabled = True
        self._weighted_bin_bins = bin_indices
        self._weighted_bin_ratios = all_ratios
        self._weighted_bin_specs = specs
        # Curriculum and weighted-bin sampling are mutually exclusive at cache level
        self.disable_curriculum()
        self._build_dataloader()
        self._prime_buffers()

    def update_advantage_scores(self, stats: Dict[str, float]) -> None:
        """Update scorer EMA from per-clip median(|adv|) values."""
        if not (self._curriculum_enabled and self._scorer is not None):
            return
        self._scorer.update(stats, self._step_counter)

    def update_curriculum(
        self,
        adv_stats: Optional[Dict[str, float]] = None,
        td_stats: Optional[Dict[str, float]] = None,
        pose_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update actor and critic progress stats together when supported."""
        if not (self._curriculum_enabled and self._scorer is not None):
            return
        # Pose-only scorer
        if isinstance(self._scorer, Exp3PoseProgressScorer):
            if pose_stats:
                self._scorer.update(pose_stats, self._step_counter)
            return
        # Combined or actor-only scorers
        if hasattr(self._scorer, "update_combined"):
            getattr(self._scorer, "update_combined")(
                adv_stats or {}, td_stats or {}, self._step_counter
            )
        else:
            self._scorer.update(adv_stats or {}, self._step_counter)

    def get_curriculum_metrics(self) -> Dict[str, float]:
        """Return lightweight metrics for logging."""
        if not (self._curriculum_enabled and self._scorer is not None):
            return {}
        state = self._scorer.state_dict()
        # A-scores (actor)
        if "ema_adv" in state:
            ema_a_vals = list(state.get("ema_adv", {}).values())
        else:
            ema_a_vals = list(state.get("ema", {}).values())
        if len(ema_a_vals) == 0:
            scoresA_mean = 0.0
            scoresA_std = 0.0
        else:
            ta = torch.tensor(ema_a_vals, dtype=torch.float32)
            scoresA_mean = float(ta.mean().item())
            scoresA_std = float(ta.std().item())
        # D-scores (critic)
        ema_d_vals = list(state.get("ema_td", {}).values())
        if len(ema_d_vals) == 0:
            scoresD_mean = 0.0
            scoresD_std = 0.0
        else:
            td = torch.tensor(ema_d_vals, dtype=torch.float32)
            scoresD_mean = float(td.mean().item())
            scoresD_std = float(td.std().item())
        # P-scores (pose)
        ema_p_vals = list(state.get("ema_pose", {}).values())
        if len(ema_p_vals) == 0:
            scoresP_mean = 0.0
            scoresP_std = 0.0
        else:
            tp = torch.tensor(ema_p_vals, dtype=torch.float32)
            scoresP_mean = float(tp.mean().item())
            scoresP_std = float(tp.std().item())
        # Optional probability metrics (current batch)
        prob_entropy = 0.0
        prob_max = 0.0
        top1p_mass = 0.0
        topk10_mass = 0.0
        coverage_1e3 = 0.0
        try:
            keys = self.current_batch.motion_keys
            p = self._scorer.probabilities(keys, self._step_counter)
            if p is None:
                raw = self._scorer.scores(keys, self._step_counter)
                p = torch.clamp(raw, min=1.0e-12)
                p = p / p.sum()
            prob_max = float(p.max().item())
            prob_entropy = float(-(p * torch.log(p)).sum().item())
            # distribution shape diagnostics
            n = int(p.numel())
            if n > 0:
                k1 = max(1, n // 100)
                sorted_p, _ = torch.sort(p, descending=True)
                top1p_mass = float(sorted_p[:k1].sum().item())
                k10 = min(10, n)
                topk10_mass = float(sorted_p[:k10].sum().item())
                coverage_1e3 = float((p >= 1.0e-3).float().mean().item())
        except Exception:
            pass
        return {
            "scores_mean": scoresA_mean,
            "scores_std": scoresA_std,
            "scoresA_mean": scoresA_mean,
            "scoresA_std": scoresA_std,
            "scoresD_mean": scoresD_mean,
            "scoresD_std": scoresD_std,
            "unique_in_last_sample": float(self._last_unique_count),
            "prob_max": prob_max,
            "prob_entropy": prob_entropy,
            "prob_top1pct_mass": top1p_mass,
            "prob_top10_mass": topk10_mass,
            "prob_coverage_ge_1e3": coverage_1e3,
            # progress/reward aggregates (if available)
            "progA_mean": float(
                torch.tensor(
                    list(state.get("last_prog_a", {}).values()) or [0.0]
                )
                .float()
                .mean()
                .item()
            ),
            "progD_mean": float(
                torch.tensor(
                    list(state.get("last_prog_d", {}).values()) or [0.0]
                )
                .float()
                .mean()
                .item()
            ),
            "progP_mean": float(
                torch.tensor(
                    list(state.get("last_prog_p", {}).values()) or [0.0]
                )
                .float()
                .mean()
                .item()
            ),
            "reward_mean": float(
                torch.tensor(
                    list(state.get("last_reward", {}).values()) or [0.0]
                )
                .float()
                .mean()
                .item()
            ),
            "scoresP_mean": scoresP_mean,
            "scoresP_std": scoresP_std,
        }

    def sync_curriculum_from_rank0(self) -> None:
        """Broadcast scorer state from rank 0 to all ranks (if distributed)."""
        if not (self._curriculum_enabled and self._scorer is not None):
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        rank = dist.get_rank()
        obj_list = [self._scorer.state_dict() if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        if rank != 0 and obj_list[0] is not None:
            self._scorer.load_state_dict(obj_list[0])  # type: ignore

    def _all_motion_keys(self) -> List[str]:
        """Return motion keys for all windows from the training dataset only."""
        keys_set = set()
        ds = self._datasets.get("train")
        if ds is not None:
            for w in getattr(ds, "windows", []):
                keys_set.add(getattr(w, "motion_key", None))
        # Remove Nones and return a stable list
        keys = [k for k in keys_set if k is not None]
        keys.sort()
        return keys

    def dump_curriculum_json(self, output_dir: str, iteration: int) -> str:
        """Dump scorer state (global scores for all keys) and global probabilities to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "iteration": int(iteration),
            "step": int(self._step_counter),
            "config": {"scorer": type(self._scorer).__name__},
            "scores": {},  # final scores per key
            "last_step": {},
            "probabilities": {},  # global probabilities over all keys
        }
        if self._curriculum_enabled and self._scorer is not None:
            # Collect all motion keys across datasets
            all_keys = self._all_motion_keys()
            # Compute final scores (S_m) for all keys
            scores_t = self._scorer.scores(all_keys, self._step_counter)
            # Compute global probabilities via scorer if available
            probs_all = None
            try:
                probs_all = self._scorer.probabilities(
                    all_keys, self._step_counter
                )
            except Exception:
                probs_all = None
            if probs_all is None:
                # Minimal fallback: normalize non-negative scores
                s = torch.clamp(scores_t, min=1.0e-12)
                probs_all = s / s.sum()
            probs_all = torch.clamp(probs_all, min=1.0e-12)
            probs_all = probs_all / probs_all.sum()
            # Convert to plain Python mappings
            score_list = scores_t.detach().cpu().tolist()
            prob_list = probs_all.detach().cpu().tolist()
            data["scores"] = {
                all_keys[i]: float(score_list[i]) for i in range(len(all_keys))
            }
            # last step info (if any)
            state = self._scorer.state_dict()
            last_step = state.get("last_step", {}) or {}
            data["last_step"] = {k: int(v) for k, v in last_step.items()}
            data["probabilities"] = {
                all_keys[i]: float(prob_list[i]) for i in range(len(all_keys))
            }
            # include log-weights when available (EXP3, numerically stable)
            log_weights = state.get("log_weights", None)
            if isinstance(log_weights, dict):
                data["log_weights"] = {
                    k: float(v) for k, v in log_weights.items()
                }
            # aggregates (lightweight)
            agg = {}
            # actor score aggregates
            if "ema_adv" in state:
                ta = torch.tensor(
                    list(state.get("ema_adv", {}).values()) or [0.0]
                ).float()
            else:
                ta = torch.tensor(
                    list(state.get("ema", {}).values()) or [0.0]
                ).float()
            agg["scoresA_mean"] = float(ta.mean().item())
            agg["scoresA_std"] = float(ta.std().item())
            # critic score aggregates
            td = torch.tensor(
                list(state.get("ema_td", {}).values()) or [0.0]
            ).float()
            agg["scoresD_mean"] = float(td.mean().item())
            agg["scoresD_std"] = float(td.std().item())
            # pose score aggregates
            tp = torch.tensor(
                list(state.get("ema_pose", {}).values()) or [0.0]
            ).float()
            agg["scoresP_mean"] = float(tp.mean().item())
            agg["scoresP_std"] = float(tp.std().item())
            # reward/progress aggregates
            r = torch.tensor(
                list(state.get("last_reward", {}).values()) or [0.0]
            ).float()
            agg["reward_mean"] = float(r.mean().item())
            agg["reward_std"] = float(r.std().item())
            pa = torch.tensor(
                list(state.get("last_prog_a", {}).values()) or [0.0]
            ).float()
            pd = torch.tensor(
                list(state.get("last_prog_d", {}).values()) or [0.0]
            ).float()
            agg["progA_mean"] = float(pa.mean().item())
            agg["progD_mean"] = float(pd.mean().item())
            pp = torch.tensor(
                list(state.get("last_prog_p", {}).values()) or [0.0]
            ).float()
            agg["progP_mean"] = float(pp.mean().item())
            data["aggregates"] = agg
        out_path = os.path.join(
            output_dir, f"curriculum_iter_{int(iteration):06d}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path

    def sample_env_assignments(
        self,
        num_envs: int,
        n_future_frames: int,
        device: torch.device,
        *,
        deterministic_start: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        batch = self.current_batch
        lengths = batch.lengths.to(device)

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        # Advance internal "step" on each assignment call
        self._step_counter += 1

        total = int(lengths.shape[0])
        if (
            not (self._curriculum_enabled and self._scorer is not None)
            or total == 0
        ):
            # Fallback to uniform
            clip_indices = torch.randint(
                low=0, high=total, size=(num_envs,), device=device
            )
        else:
            # Build probabilities for current batch motion keys
            motion_keys = batch.motion_keys
            probs = None
            try:
                probs = self._scorer.probabilities(
                    motion_keys, self._step_counter
                )
            except Exception:
                probs = None
            if probs is None:
                raw_scores = self._scorer.scores(
                    motion_keys, self._step_counter
                ).to(device)
                # Minimal fallback: normalize non-negative scores
                probs = torch.clamp(raw_scores, min=1.0e-12)
                probs = probs / probs.sum()
            probs = torch.clamp(probs, min=1.0e-12)
            probs = probs / probs.sum()

            # Unique-first sampling
            unique_n = int(min(num_envs, total))
            unique_idx = torch.multinomial(
                probs.detach().cpu(),
                num_samples=unique_n,
                replacement=False,
            ).to(device)
            self._last_unique_count = int(unique_idx.numel())

            if unique_n < num_envs:
                extra_n = int(num_envs - unique_n)
                # Continue with replacement for the remainder
                extra_idx = torch.multinomial(
                    probs.detach().cpu(),
                    num_samples=extra_n,
                    replacement=True,
                ).to(device)
                clip_indices = torch.cat([unique_idx, extra_idx], dim=0)
            else:
                clip_indices = unique_idx

            # Notify scorer about sampled keys
            sampled_keys = [motion_keys[int(i)] for i in clip_indices.tolist()]
            sampled_probs = probs.index_select(
                0, clip_indices.to(probs.device)
            )
            self._scorer.on_sampled(
                sampled_keys, self._step_counter, probs=sampled_probs
            )

        max_start = torch.clamp(
            lengths[clip_indices] - 1 - n_future_frames, min=0
        )
        if deterministic_start:
            frame_starts = torch.zeros_like(max_start)
        else:
            rand = torch.rand_like(max_start, dtype=torch.float32)
            frame_starts = torch.floor(rand * (max_start + 1).float()).to(
                torch.long
            )

        return clip_indices, frame_starts

    def gather_state(
        self,
        clip_indices: Tensor,
        frame_indices: Tensor,
        n_future_frames: int,
    ) -> Dict[str, Tensor]:
        batch = self.current_batch
        staged_device = batch.lengths.device
        selected_clips = clip_indices.to(staged_device, dtype=torch.long)
        frame_indices = frame_indices.to(staged_device, dtype=torch.long)

        temporal_span = 1 + int(n_future_frames)
        time_offsets = torch.arange(
            temporal_span, device=staged_device, dtype=torch.long
        )
        gather_timesteps = frame_indices[:, None] + time_offsets[None, :]

        lengths = batch.lengths
        max_valid = torch.clamp(
            lengths.index_select(0, selected_clips) - 1, min=0
        )
        gather_timesteps = torch.minimum(gather_timesteps, max_valid[:, None])

        state: Dict[str, Tensor] = {}
        for name, tensor in batch.tensors.items():
            source = tensor.index_select(0, selected_clips)
            # Build index tensor to gather along the temporal dimension (dim=1)
            # Start from [B, T] and only add singleton dims if needed to match source.ndim.
            indices = gather_timesteps
            while indices.ndim < source.ndim:
                indices = indices[..., None]
            if source.ndim > 2:
                expanded = indices.expand(-1, -1, *source.shape[2:])
            else:
                expanded = indices  # shape [B, T] matches 2D source [B, L]
            gathered = torch.take_along_dim(source, expanded, dim=1)
            state[name] = gathered

        return state

    def lengths_for_indices(self, clip_indices: Tensor) -> Tensor:
        lengths = self.current_batch.lengths.to(clip_indices.device)
        return lengths.index_select(0, clip_indices.long())

    def motion_keys_for_indices(self, clip_indices: Tensor) -> List[str]:
        result = []
        base_keys = self.current_batch.motion_keys
        for idx in clip_indices.tolist():
            result.append(base_keys[int(idx)])
        return result

    def export_motion_clip(self, motion_key: str) -> Dict[str, np.ndarray]:
        dataset = self._datasets[self._mode]
        if motion_key not in dataset.clips:
            raise KeyError(f"Motion key '{motion_key}' not found in manifest")

        meta = dataset.clips[motion_key]
        shard_index = int(meta.get("shard", 0))
        shard_handle = dataset._get_shard_handle(shard_index)
        start = int(meta.get("start", 0))
        length = int(meta.get("length", 0))
        end = start + length

        output: Dict[str, np.ndarray] = {}
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            if dataset_name in shard_handle:
                output[logical_name] = shard_handle[dataset_name][start:end]
            elif logical_name == "dof_vel" and "dof_vels" in shard_handle:
                # Backward-compat: allow 'dof_vels' when 'dof_vel' is missing
                output[logical_name] = shard_handle["dof_vels"][start:end]

        if "frame_flag" in shard_handle:
            output["frame_flag"] = shard_handle["frame_flag"][start:end]

        return output

    def _prime_buffers(self) -> None:
        self._current_batch = self._fetch_next_batch()
        # Ensure first staged batch is ready before consumption
        if (
            self._current_ready_event is not None
            and self._stage_device is not None
            and getattr(self._stage_device, "type", None) == "cuda"
        ):
            import torch.cuda

            t0 = time.time()
            torch.cuda.current_stream(self._stage_device).wait_event(
                self._current_ready_event
            )
            t1 = time.time()
            logger.info(
                f"Perf/Cache/cuda_wait_event_ms={((t1 - t0) * 1e3):.2f} (first)"
            )
        self._next_batch = self._fetch_next_batch()

    def _fetch_next_batch(self) -> ClipBatch:
        if self._iterator is None:
            self._iterator = self._build_iterator()

        try:
            t0 = time.time()
            batch = next(self._iterator)
            t1 = time.time()
            # logger.info(
            #     f"Perf/Cache/dataloader_next_ms={((t1 - t0) * 1e3):.2f}"
            # )
        except StopIteration:
            # For training (infinite sampler), this path shouldn't trigger often; safeguard anyway
            self._iterator = self._build_iterator(reset_epoch=True)
            t0 = time.time()
            batch = next(self._iterator)
            t1 = time.time()
            # logger.info(
            #     f"Perf/Cache/dataloader_next_ms={((t1 - t0) * 1e3):.2f} (reset)"
            # )

        batch = self._filter_clip_batch_prefixes(batch)
        staged = self._stage_batch(batch, record_event=True)
        # Move pending event into current/next slot
        if self._current_batch is None:
            self._current_ready_event = self._pending_ready_event
        else:
            self._next_ready_event = self._pending_ready_event
        self._pending_ready_event = None
        return staged

    def _filter_clip_batch_prefixes(self, batch: ClipBatch) -> ClipBatch:
        """Drop unused motion sources at cache level to reduce device memory.

        Behavior:
        - Always keep canonical bases required by RefMotionCommand:
          dof_pos, dof_vel, rg_pos, rb_rot, body_vel, body_ang_vel,
          root_pos, root_rot, root_vel, root_ang_vel.
        - Always keep frame_flag (time metadata).
        - If _allowed_prefixes is not None, additionally keep any tensor whose
          name starts with one of the allowed prefixes (e.g. "ref_", "ft_ref_").
        - Drop all other tensors.
        """
        if self._allowed_prefixes is None:
            return batch

        allowed_bases = {
            "dof_pos",
            "dof_vel",
            "rg_pos",
            "rb_rot",
            "body_vel",
            "body_ang_vel",
            "root_pos",
            "root_rot",
            "root_vel",
            "root_ang_vel",
        }

        tensors = batch.tensors
        filtered: Dict[str, Tensor] = {}
        for name, tensor in tensors.items():
            if name == "frame_flag" or name in allowed_bases:
                filtered[name] = tensor
                continue
            keep = False
            for pfx in self._allowed_prefixes:
                if name.startswith(pfx):
                    keep = True
                    break
            if keep:
                filtered[name] = tensor

        if len(filtered) == len(tensors):
            return batch

        return ClipBatch(
            tensors=filtered,
            lengths=batch.lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=batch.raw_motion_keys,
            max_frame_length=batch.max_frame_length,
        )

    def _stage_batch(
        self, batch: ClipBatch, record_event: bool = False
    ) -> ClipBatch:
        if self._stage_device is None:
            return batch

        # If CUDA, copy on a dedicated stream and record readiness
        if self._copy_stream is None and (
            self._stage_device is not None
            and (
                getattr(self._stage_device, "type", None) == "cuda"
                or (
                    isinstance(self._stage_device, str)
                    and self._stage_device.startswith("cuda")
                )
            )
        ):
            # Fallback: lazily create copy stream if it wasn't created at init
            import torch.cuda

            try:
                if isinstance(self._stage_device, torch.device):
                    dev_index = (
                        0
                        if self._stage_device.index is None
                        else int(self._stage_device.index)
                    )
                elif isinstance(
                    self._stage_device, str
                ) and self._stage_device.startswith("cuda"):
                    parts = self._stage_device.split(":")
                    dev_index = (
                        int(parts[1])
                        if len(parts) > 1
                        else torch.cuda.current_device()
                    )
                else:
                    dev_index = torch.cuda.current_device()
                torch.cuda.set_device(dev_index)
                self._copy_stream = torch.cuda.Stream()
                logger.info(
                    f"Perf/Cache: created CUDA copy stream lazily on cuda:{dev_index}"
                )
            except Exception as e:
                logger.warning(
                    f"Perf/Cache: failed to lazily create CUDA copy stream: {e}"
                )

        if self._copy_stream is not None:
            import torch.cuda

            # estimate payload size for logging
            try:
                total_bytes = 0
                for tensor in batch.tensors.values():
                    total_bytes += int(tensor.element_size() * tensor.numel())
                total_bytes += int(
                    batch.lengths.element_size() * batch.lengths.numel()
                )
            except Exception:
                total_bytes = -1
            t0 = time.time()
            with torch.cuda.stream(self._copy_stream):
                tensors = {
                    name: tensor.to(self._stage_device, non_blocking=True)
                    for name, tensor in batch.tensors.items()
                }
                lengths = batch.lengths.to(
                    self._stage_device, non_blocking=True
                )
            if record_event:
                ev = torch.cuda.Event()
                ev.record(self._copy_stream)
                self._pending_ready_event = ev
            t1 = time.time()
            # logger.info(
            #     f"Perf/Cache/stage_schedule_ms={((t1 - t0) * 1e3):.2f} bytes={total_bytes} (copy-stream)"
            # )
        else:
            t0 = time.time()
            tensors = {
                name: tensor.to(self._stage_device, non_blocking=True)
                for name, tensor in batch.tensors.items()
            }
            lengths = batch.lengths.to(self._stage_device, non_blocking=True)
            t1 = time.time()
            logger.info(
                f"Perf/Cache/stage_schedule_ms={((t1 - t0) * 1e3):.2f} (same-stream)"
            )
        return ClipBatch(
            tensors=tensors,
            lengths=lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=getattr(
                batch, "raw_motion_keys", batch.motion_keys
            ),
            max_frame_length=batch.max_frame_length,
        )

    def _build_iterator(
        self, *, reset_epoch: bool = False
    ) -> Iterator[ClipBatch]:
        if self._dataloader is None:
            raise RuntimeError("DataLoader is not initialised")

        if self._sampler is not None and reset_epoch:
            self._sampler.set_epoch(self._swap_index + 1)

        return iter(self._dataloader)

    def _build_dataloader(self) -> None:
        dataset = self._datasets[self._mode]

        # Clamp batch size to dataset length to avoid empty iterator when drop_last is disabled
        effective_batch_size = self._batch_size
        ds_len = len(dataset)
        if isinstance(ds_len, int) and ds_len > 0:
            effective_batch_size = max(1, min(self._batch_size, ds_len))

        # Sampler selection: validation uses standard distributed/sequential samplers;
        # training can optionally use weighted-bin sampling.
        if self._mode == "val":
            if self._sampler_world_size > 1:
                self._sampler = DistributedSampler(
                    dataset,
                    num_replicas=self._sampler_world_size,
                    rank=self._sampler_rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                self._sampler = None
        else:
            if (
                self._weighted_bin_enabled
                and self._weighted_bin_bins is not None
                and self._weighted_bin_ratios is not None
            ):
                seed = self._seed + self._sampler_rank * 100003
                self._sampler = WeightedBinInfiniteSampler(
                    dataset_len=ds_len,
                    bin_indices=self._weighted_bin_bins,
                    ratios=self._weighted_bin_ratios,
                    batch_size=effective_batch_size,
                    seed=seed,
                )
            else:
                if self._sampler_world_size > 1:
                    # Infinite sampler for training: no epoch boundaries
                    self._sampler = InfiniteDistributedSampler(
                        dataset,
                        num_replicas=self._sampler_world_size,
                        rank=self._sampler_rank,
                        shuffle=True,
                        drop_last=False,
                    )
                else:
                    # Infinite sampler for single-process training
                    self._sampler = InfiniteRandomSampler(dataset)

        # Only pass prefetch_factor when using workers
        pf = (
            self._prefetch_factor
            if (self._num_workers and self._num_workers > 0)
            else None
        )
        pw = (
            self._persistent_workers
            if (self._num_workers and self._num_workers > 0)
            else False
        )

        # Collate wrapper: in validation, pad the batch up to cache size by
        # uniformly repeating samples when dataset is smaller than batch size.
        collate = partial(
            _cache_collate_fn,
            mode=self._mode,
            batch_size=self._batch_size,
        )

        mp_ctx = None
        if self._num_workers and self._num_workers > 0:
            mp_ctx = mp.get_context("spawn")

        self._dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            sampler=self._sampler,
            shuffle=(self._sampler is None and self._mode != "val"),
            num_workers=self._num_workers,
            prefetch_factor=pf,
            pin_memory=self._pin_memory,
            persistent_workers=pw,
            collate_fn=collate,
            drop_last=False,
            multiprocessing_context=mp_ctx,
        )
        self._iterator = None
        self._current_batch = None
        self._next_batch = None
        self._swap_index = 0

        # Compute number of batches only for validation; training is infinite
        local_len = ds_len
        if self._mode == "val":
            if self._sampler is not None:
                local_len = (
                    ds_len + self._sampler_world_size - 1
                ) // self._sampler_world_size
            self._effective_batch_size = int(effective_batch_size)
            self._num_batches = (
                local_len + self._effective_batch_size - 1
            ) // self._effective_batch_size
        else:
            self._effective_batch_size = int(effective_batch_size)
            self._num_batches = 2**31  # effectively infinite for logging

    def close(self) -> None:
        """Release DataLoader workers and close underlying HDF5 datasets."""
        self._iterator = None
        self._current_batch = None
        self._next_batch = None
        self._dataloader = None
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        for ds in self._datasets.values():
            if ds is not None:
                ds.close()

    def __del__(self) -> None:
        self.close()
