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
* ``gather_tensor`` to fetch exactly one tensor field for ``1 + n_future``
  frames per environment.

All tensors returned by this module are ``torch.float32`` unless stated
otherwise; tensor shapes are noted explicitly in type annotations.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from holomotion.src.utils import torch_utils
from holomotion.src.training.reference_fk import TrainingReferenceFK

Tensor = torch.Tensor


def _cpu_only_dataloader_worker_init_fn(worker_id: int) -> None:
    """Keep cache workers lightweight without mutating CUDA visibility."""
    del worker_id
    torch.set_num_threads(1)


def _allocate_batch_counts(
    raw_counts: List[float], target_total: int
) -> List[int]:
    """Allocate integer counts that sum exactly to target_total."""
    total = int(max(0, target_total))
    if len(raw_counts) == 0:
        return []
    base_counts = [max(0, int(c)) for c in raw_counts]
    residuals = [float(c) - float(int(c)) for c in raw_counts]
    remaining = total - int(sum(base_counts))
    if remaining > 0:
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
    elif remaining < 0:
        order = sorted(range(len(residuals)), key=lambda i: residuals[i])
        idx_pos = 0
        while remaining < 0:
            j = order[idx_pos % len(order)]
            if base_counts[j] > 0:
                base_counts[j] -= 1
                remaining += 1
            idx_pos += 1
    if sum(base_counts) != total:
        raise RuntimeError(
            "Internal error: integer batch-count allocation did not preserve total."
        )
    return [max(0, int(c)) for c in base_counts]


def _configure_weighted_bins(
    keys: List[str],
    cfg: Mapping[str, Any],
    batch_size_for_log: int,
) -> Tuple[List[List[int]], List[float], List[Dict[str, Any]]]:
    """Common helper to parse config, assign bins, and compute batch fractions."""
    if batch_size_for_log <= 0:
        batch_size_for_log = 1

    cfg_local: Dict[str, Any] = dict(cfg or {})

    dataset_ratios = cfg_local.get("dataset_ratios")
    normalize_dataset_ratios = dataset_ratios is not None
    if dataset_ratios is not None:
        if not isinstance(dataset_ratios, Mapping) or not dataset_ratios:
            raise ValueError(
                "weighted_bin dataset_ratios must be a non-empty mapping"
            )
        patterns_cfg = [
            {
                "name": str(dataset_name),
                "regex": rf"^{re.escape(_normalize_dataset_id(dataset_name))}::",
                "ratio": ratio,
            }
            for dataset_name, ratio in dataset_ratios.items()
        ]
    else:
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
    if normalize_dataset_ratios:
        if sum_explicit <= 0.0:
            raise ValueError(
                "weighted_bin root ratios must contain at least one positive value"
            )
        ratios = [ratio / sum_explicit for ratio in ratios]
        sum_explicit = 1.0
    elif sum_explicit > 1.0 + 1.0e-6:
        raise ValueError(
            f"Sum of weighted-bin ratios is {sum_explicit:.6f} (> 1.0). "
            "Please reduce the ratios so that their sum is <= 1.0."
        )
    if sum_explicit > 1.0:
        sum_explicit = 1.0
    others_ratio = max(0.0, 1.0 - sum_explicit)

    if len(keys) == 0:
        raise ValueError(
            "weighted_bin configuration received an empty key set"
        )

    num_items_total = float(len(keys))
    num_explicit = len(compiled_patterns)
    bin_count = num_explicit if normalize_dataset_ratios else num_explicit + 1
    bin_indices: List[List[int]] = [[] for _ in range(bin_count)]

    for idx, motion_key in enumerate(keys):
        assigned = False
        for b_idx, pat in enumerate(compiled_patterns):
            if pat["compiled"].search(motion_key):
                bin_indices[b_idx].append(idx)
                assigned = True
                break
        if not assigned and not normalize_dataset_ratios:
            bin_indices[-1].append(idx)

    # Combine explicit ratios with implicit "others" ratio
    all_ratios: List[float] = list(ratios)
    if not normalize_dataset_ratios:
        all_ratios.append(others_ratio)

    # If all motion keys are covered by explicit regex bins, but the specified
    # ratios sum to less than 1.0, linearly reweight explicit ratios so that
    # they sum to 1.0 and disable the implicit "others" bin.
    others_count = 0 if normalize_dataset_ratios else len(bin_indices[-1])
    if (
        not normalize_dataset_ratios
        and others_count == 0
        and others_ratio > 0.0
        and sum_explicit > 0.0
    ):
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
                f"Weighted-bin 'others' has ratio {r:.6f} but matched no motion keys"
            )

    # Prepare logging summary using the configured cache batch size
    raw_counts_log = [ratio * batch_size_for_log for ratio in all_ratios]
    base_counts_log = _allocate_batch_counts(
        raw_counts=raw_counts_log,
        target_total=batch_size_for_log,
    )
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
    if not normalize_dataset_ratios:
        n_o = len(bin_indices[-1])
        ds_frac_o = float(n_o) / total_items
        bf_o = batch_fractions_log[-1]
        specs.append(
            {
                "name": "others",
                "regex": "<unmatched>",
                "ratio": bf_o,
                "count": n_o,
                "dataset_fraction": ds_frac_o,
                "batch_fraction": bf_o,
            }
        )

    return bin_indices, all_ratios, specs


def _normalize_dataset_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _manifest_dataset_id(
    manifest: Mapping[str, Any], manifest_path: str
) -> str:
    explicit = manifest.get("dataset_id")
    if explicit:
        return _normalize_dataset_id(explicit)

    path_candidates = [
        manifest.get("source_holosmpl_h5_root"),
        os.path.dirname(manifest_path),
    ]
    for path_value in path_candidates:
        if not path_value:
            continue
        for part in reversed(Path(str(path_value)).parts):
            lowered = part.lower()
            if lowered.startswith(("train_", "eval_")):
                return _normalize_dataset_id(part)

    for field in ("dataset_name", "dataset", "source_name"):
        value = manifest.get(field)
        if value:
            return _normalize_dataset_id(value)
    return "unknown"


def _weighted_bin_motion_key(
    motion_key: str, metadata: Mapping[str, Any] | None = None
) -> str:
    dataset_id = None if metadata is None else metadata.get("_dataset_id")
    if not dataset_id:
        return str(motion_key)
    return f"{dataset_id}::{motion_key}"


def _collect_manifest_keys(
    manifest_path: str | Sequence[str],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    if isinstance(manifest_path, (str, os.PathLike)):
        manifest_paths: List[str] = [str(manifest_path)]
    else:
        manifest_paths = [str(p) for p in manifest_path]
    if len(manifest_paths) == 0:
        raise ValueError("Expected at least one manifest path")

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
                f"Manifest at {mp} contains no clips; cannot preview sampling."
            )
        dataset_id = _manifest_dataset_id(manifest, mp)
        for key in clips.keys():
            weighted_key = _weighted_bin_motion_key(
                key, {"_dataset_id": dataset_id}
            )
            if weighted_key in key_source:
                raise ValueError(
                    f"Duplicate weighted-bin key '{weighted_key}' found in multiple "
                    "manifests; clip keys must be globally unique."
                )
            key_source[weighted_key] = mp

    return list(key_source.keys()), key_source, manifest_paths


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

    keys, _, _ = _collect_manifest_keys(manifest_path=manifest_path)
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


def preview_uniform_from_manifest(
    manifest_path: str | Sequence[str],
    batch_size: int,
    *,
    max_frame_length: int,
    min_window_length: int,
    handpicked_motion_names: Optional[Sequence[str]] = None,
    excluded_motion_names: Optional[Sequence[str]] = None,
) -> None:
    """Manifest-level preview table for uniform/curriculum sampling."""
    if batch_size <= 0:
        batch_size = 1
    if max_frame_length <= 0:
        raise ValueError("max_frame_length must be positive")
    if min_window_length <= 0:
        raise ValueError("min_window_length must be positive")

    _, _, manifest_paths = _collect_manifest_keys(manifest_path=manifest_path)
    handpicked_set = (
        set(handpicked_motion_names)
        if handpicked_motion_names is not None
        else None
    )
    excluded_set = (
        set(excluded_motion_names)
        if excluded_motion_names is not None
        else None
    )

    def _normalize_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        key = value if isinstance(value, str) else str(value)
        if not key:
            return None
        return key

    def _build_aliases(motion_key: str, meta: Mapping[str, Any]) -> List[str]:
        aliases: List[str] = []

        def _add(value: Any) -> None:
            key = _normalize_key(value)
            if key is None or key in aliases:
                return
            aliases.append(key)

        _add(motion_key)
        if isinstance(meta, Mapping):
            _add(meta.get("motion_key"))
            metadata = meta.get("metadata")
            if isinstance(metadata, Mapping):
                _add(metadata.get("motion_key"))
                _add(metadata.get("raw_motion_key"))
        return aliases

    def _count_windows(clip_length: int) -> Tuple[int, int]:
        remaining = clip_length
        offset = 0
        num_windows = 0
        num_frames = 0
        while remaining > 0:
            window_length = min(max_frame_length, remaining)
            if window_length >= min_window_length:
                num_windows += 1
                num_frames += int(window_length)
            offset += int(window_length)
            remaining = max(0, clip_length - offset)
        return num_windows, num_frames

    stats_by_manifest: Dict[str, Dict[str, float]] = {}
    for mp in manifest_paths:
        with open(mp, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        clips = manifest.get("clips", {})
        if not clips:
            raise ValueError(
                f"Manifest at {mp} contains no clips; cannot preview sampling."
            )
        num_windows = 0
        num_frames = 0
        duration_s = 0.0
        for key, meta in clips.items():
            if isinstance(meta, Mapping):
                aliases = _build_aliases(key, meta)
            else:
                aliases = [key]
            if handpicked_set is not None and not any(
                alias in handpicked_set for alias in aliases
            ):
                continue
            if excluded_set is not None and any(
                alias in excluded_set for alias in aliases
            ):
                continue
            length = (
                int(meta.get("length", 0)) if isinstance(meta, Mapping) else 0
            )
            if length <= 0:
                continue
            metadata = (
                meta.get("metadata") if isinstance(meta, Mapping) else None
            )
            motion_fps_val = None
            if isinstance(metadata, Mapping):
                motion_fps_val = metadata.get("motion_fps")
            if motion_fps_val is None and isinstance(meta, Mapping):
                motion_fps_val = meta.get("motion_fps")
            if motion_fps_val is None:
                raise ValueError(
                    f"motion_fps missing for clip {key} in manifest {mp}"
                )
            motion_fps = float(motion_fps_val)
            if motion_fps <= 0.0:
                raise ValueError(
                    f"Invalid motion_fps {motion_fps} for clip {key} in {mp}"
                )
            clip_windows, clip_frames = _count_windows(length)
            num_windows += int(clip_windows)
            num_frames += int(clip_frames)
            duration_s += float(clip_frames) / float(motion_fps)
        stats_by_manifest[mp] = {
            "num_windows": float(num_windows),
            "num_frames": float(num_frames),
            "duration_s": float(duration_s),
        }

    total_windows = int(
        sum(stats["num_windows"] for stats in stats_by_manifest.values())
    )
    if total_windows == 0:
        raise ValueError(
            "No motion windows satisfy the requested frame length constraints"
        )

    table_rows = []
    denom = float(max(1, total_windows))
    for mp in manifest_paths:
        stats = stats_by_manifest.get(mp, {})
        count = int(stats.get("num_windows", 0))
        frames = int(stats.get("num_frames", 0))
        duration_h = float(stats.get("duration_s", 0.0)) / 3600.0
        frac = float(count) / denom
        table_rows.append(
            [
                os.path.dirname(mp),
                count,
                f"{frac:.4f}",
                frames,
                f"{duration_h:.2f}",
                f"{frac:.4f}",
            ]
        )
    headers = [
        "dataset_root",
        "num_windows",
        "window_fraction",
        "num_frames",
        "duration_h",
        "batch_fraction",
    ]
    logger.info(
        "Uniform sampling preview (manifest-level):\n"
        + tabulate(table_rows, headers=headers, tablefmt="simple_outline")
    )


def preview_sampling_from_cfg(motion_cfg: Mapping[str, Any]) -> None:
    """Preview manifest-level sampling table for uniform/weighted-bin."""
    sampling_strategy_cfg = motion_cfg.get("sampling_strategy", None)
    if sampling_strategy_cfg is None:
        sampling_strategy = "uniform"
    else:
        sampling_strategy = str(sampling_strategy_cfg).lower()
    if sampling_strategy not in ("uniform", "weighted_bin", "curriculum"):
        return

    backend = str(motion_cfg.get("backend", "hdf5")).lower()
    if backend not in ("hdf5", "hdf5_simple", "hdf5_v2"):
        return

    train_roots = _normalize_root_list(
        motion_cfg.get("train_hdf5_roots", None)
    )
    if len(train_roots) == 0:
        hdf5_root = motion_cfg.get("hdf5_root", None)
        if not hdf5_root:
            return
        train_roots = [str(hdf5_root)]
    manifest_paths = [
        os.path.join(str(root), "manifest.json") for root in train_roots
    ]
    cache_cfg = motion_cfg.get("cache", {})
    batch_size = int(cache_cfg.get("max_num_clips", 1))

    if sampling_strategy == "weighted_bin":
        weighted_bin_cfg = weighted_bin_cfg_from_motion_cfg(motion_cfg)
        preview_weighted_bin_from_manifest(
            manifest_path=manifest_paths
            if len(manifest_paths) > 1
            else manifest_paths[0],
            batch_size=batch_size,
            cfg=weighted_bin_cfg,
        )
        return

    max_frame_length = int(motion_cfg.get("max_frame_length", 1))
    min_window_length = int(motion_cfg.get("min_frame_length", 1))
    handpicked_motion_names = motion_cfg.get("handpicked_motion_names", None)
    excluded_motion_names = motion_cfg.get("excluded_motion_names", None)
    preview_uniform_from_manifest(
        manifest_path=manifest_paths
        if len(manifest_paths) > 1
        else manifest_paths[0],
        batch_size=batch_size,
        max_frame_length=max_frame_length,
        min_window_length=min_window_length,
        handpicked_motion_names=handpicked_motion_names,
        excluded_motion_names=excluded_motion_names,
    )


MANDATORY_DATASETS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "rg_pos": "global_translation",
    "rb_rot": "global_rotation_quat",
    "body_vel": "global_velocity",
    "body_ang_vel": "global_angular_velocity",
}


class _WorldFrameNormalizeTransform:
    """Normalize motion tensors into a canonical z-up world frame in-place."""

    @staticmethod
    def _apply_prefix(
        arrays: Dict[str, Tensor],
        prefix: str,
        *,
        offset_xy: Tensor,
        q_flat_wxyz: Tensor,
        ref_rg_pos_shape: torch.Size,
        ref_rb_rot_shape: torch.Size,
    ) -> None:
        pos_key = f"{prefix}rg_pos"
        rot_key = f"{prefix}rb_rot"
        vel_key = f"{prefix}body_vel"
        ang_key = f"{prefix}body_ang_vel"
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
        if pos.shape != ref_rg_pos_shape or rot.shape != ref_rb_rot_shape:
            return

        # Center XY using canonical offset.
        pos[..., 0] -= offset_xy[0]
        pos[..., 1] -= offset_xy[1]

        # Rotate vectors using shared quaternion utilities (WXYZ convention).
        pos_flat = pos.reshape(-1, 3)
        vel_flat = vel.reshape(-1, 3)
        ang_flat = ang.reshape(-1, 3)
        pos[:] = torch_utils.quat_apply(q_flat_wxyz, pos_flat).reshape_as(pos)
        vel[:] = torch_utils.quat_apply(q_flat_wxyz, vel_flat).reshape_as(vel)
        ang[:] = torch_utils.quat_apply(q_flat_wxyz, ang_flat).reshape_as(ang)

        # Rotate orientations: q' = q_heading_inv * q.
        rot_flat_xyzw = rot.reshape(-1, 4)
        rot_flat_wxyz = torch_utils.xyzw_to_wxyz(rot_flat_xyzw)
        rot_out_wxyz = torch_utils.quat_mul(q_flat_wxyz, rot_flat_wxyz)
        rot[:] = torch_utils.wxyz_to_xyzw(rot_out_wxyz).reshape_as(rot)

    def __call__(self, arrays: Dict[str, Tensor]) -> None:
        if "ref_rg_pos" not in arrays or "ref_rb_rot" not in arrays:
            raise ValueError("ref_rg_pos and ref_rb_rot are required")
        if "ref_body_vel" not in arrays or "ref_body_ang_vel" not in arrays:
            raise ValueError("ref_body_vel and ref_body_ang_vel are required")

        rg_pos = arrays["ref_rg_pos"]
        rb_rot = arrays["ref_rb_rot"]

        # Root pose at frame 0, body 0 (XYZW quaternion, z-up).
        p_root0 = rg_pos[0, 0]  # [3]
        q_root0 = rb_rot[0, 0]  # [4]

        # Compute XY offset from root at frame 0 (will be applied in _apply_to_set).
        offset_xy = p_root0.clone()
        offset_xy[2] = 0.0

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

        t, b, _ = rg_pos.shape
        q_flat = q_heading_inv.view(1, 1, 4).expand(t, b, 4).reshape(-1, 4)
        q_flat_wxyz = torch_utils.xyzw_to_wxyz(q_flat)

        self._apply_prefix(
            arrays,
            "ref_",
            offset_xy=offset_xy,
            q_flat_wxyz=q_flat_wxyz,
            ref_rg_pos_shape=rg_pos.shape,
            ref_rb_rot_shape=rb_rot.shape,
        )


class _CpuFKTransform:
    """Compute FK on CPU and write ref_* tensors in-place."""

    def __init__(self, robot_file_path: str) -> None:
        self._fk = TrainingReferenceFK(
            robot_file_path=str(robot_file_path), device=torch.device("cpu")
        )
        self._fk = self._fk.to(torch.device("cpu"))

    def __call__(
        self,
        arrays: Dict[str, Tensor],
        fps: float,
        prefix: str = "ref_",
    ) -> None:
        root_pos_key = f"{prefix}root_pos"
        root_rot_key = f"{prefix}root_rot"
        dof_pos_key = f"{prefix}dof_pos"
        if (
            root_pos_key not in arrays
            or root_rot_key not in arrays
            or dof_pos_key not in arrays
        ):
            raise KeyError(f"Missing {prefix}root_* or {prefix}dof_pos for FK")
        with torch.no_grad():
            fk_out = self._fk(
                root_pos=arrays[root_pos_key][None, ...],
                root_quat=arrays[root_rot_key][None, ...],
                dof_pos=arrays[dof_pos_key][None, ...],
                fps=float(fps),
                quat_format="xyzw",
            )
        arrays[f"{prefix}rg_pos"] = fk_out["global_translation"][0]
        arrays[f"{prefix}rb_rot"] = fk_out["global_rotation_quat"][0]
        arrays[f"{prefix}body_vel"] = fk_out["global_velocity"][0]
        arrays[f"{prefix}body_ang_vel"] = fk_out["global_angular_velocity"][0]
        arrays[f"{prefix}dof_vel"] = fk_out["dof_vel"][0]


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
    window_index: int
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
    window_indices: Tensor
    max_frame_length: int

    @staticmethod
    def collate_fn(samples: List[MotionClipSample]) -> "ClipBatch":
        if len(samples) == 0:
            raise ValueError(
                "ClipBatch collate_fn received an empty sample list"
            )

        max_frame_length = max(
            sample.tensors["ref_dof_pos"].shape[0] for sample in samples
        )
        max_frame_length = int(max_frame_length)

        batched_tensors: Dict[str, Tensor] = {}
        lengths = torch.zeros(len(samples), dtype=torch.long)
        motion_keys = []
        raw_motion_keys = []
        window_indices = torch.zeros(len(samples), dtype=torch.long)

        for batch_idx, sample in enumerate(samples):
            lengths[batch_idx] = sample.length
            motion_keys.append(sample.motion_key)
            raw_motion_keys.append(sample.raw_motion_key)
            window_indices[batch_idx] = int(sample.window_index)

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
            window_indices=window_indices,
            max_frame_length=max_frame_length,
        )


class Hdf5RootDofDataset(Dataset[MotionClipSample]):
    """HDF5 dataset reading ref_root_* + ref_dof_pos only."""

    def __init__(
        self,
        manifest_path: str | Sequence[str],
        max_frame_length: int,
        min_window_length: int = 1,
        handpicked_motion_names: Optional[List[str]] = None,
        excluded_motion_names: Optional[List[str]] = None,
        fk_robot_file_path: Optional[str] = None,
        allowed_prefixes: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if max_frame_length <= 0:
            raise ValueError("max_frame_length must be positive")
        if min_window_length <= 0:
            raise ValueError("min_window_length must be positive")

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
        self._fk_robot_file_path = (
            str(fk_robot_file_path) if fk_robot_file_path is not None else ""
        )
        if not self._fk_robot_file_path:
            raise ValueError("fk_robot_file_path is required for hdf5_v2 FK")
        self._fk_transform = _CpuFKTransform(self._fk_robot_file_path)
        self._world_frame_transform = _WorldFrameNormalizeTransform()
        if allowed_prefixes is None:
            self._allowed_prefixes = ("ref_",)
        else:
            self._allowed_prefixes = tuple(str(v) for v in allowed_prefixes)
        if "ref_" not in self._allowed_prefixes:
            raise ValueError(
                "Hdf5RootDofDataset requires 'ref_' in allowed_prefixes"
            )
        self._progress_counter: Optional[mp.Value] = None

        if isinstance(manifest_path, (str, os.PathLike)):
            manifest_paths: List[str] = [str(manifest_path)]
        else:
            manifest_paths = [str(p) for p in manifest_path]
        if len(manifest_paths) == 0:
            raise ValueError("At least one manifest_path must be provided")

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
            dataset_id = _manifest_dataset_id(manifest, mp)

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
                meta_global["_dataset_id"] = dataset_id
                meta_global["shard"] = (
                    int(meta_global.get("shard", 0)) + shard_offset
                )
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

        # Setting up hdf5 file handles management for bounded host-memory usage
        self._file_handles: "OrderedDict[int, h5py.File]" = OrderedDict()
        max_open_env = os.getenv("HOLOMOTION_HDF5_MAX_OPEN_SHARDS")
        if max_open_env is None:
            self._h5_max_open_files = 16
        else:
            self._h5_max_open_files = max(1, int(max_open_env))
        self._h5_access_counter = 0
        self._h5_cleanup_interval = int(
            1.0e6
        )  # clean h5 handles every 1 million samples

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self._progress_counter = counter

    @staticmethod
    def _normalize_motion_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            key = value
        else:
            key = str(value)
        if not key:
            return None
        return key

    def _build_motion_key_aliases(
        self, motion_key: str, meta: Mapping[str, Any]
    ) -> Tuple[str, ...]:
        aliases: List[str] = []

        def _add(value: Any) -> None:
            key = self._normalize_motion_key(value)
            if key is None:
                return
            if key in aliases:
                return
            aliases.append(key)

        _add(motion_key)
        if isinstance(meta, Mapping):
            _add(meta.get("motion_key"))
            metadata = meta.get("metadata")
            if isinstance(metadata, Mapping):
                _add(metadata.get("motion_key"))
                _add(metadata.get("raw_motion_key"))
        return tuple(aliases)

    def _enumerate_windows(self) -> List[MotionWindow]:
        windows: List[MotionWindow] = []
        for motion_key, meta in self.clips.items():
            aliases = self._build_motion_key_aliases(motion_key, meta)
            if self.handpicked_motion_names is not None and not any(
                alias in self.handpicked_motion_names for alias in aliases
            ):
                continue
            if self.excluded_motion_names is not None and any(
                alias in self.excluded_motion_names for alias in aliases
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
                    window_index += 1
                offset += window_length
                remaining = max(0, length - offset)
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _cast_motion_np(np_array: np.ndarray, name: str) -> Tensor:
        if np_array.dtype == np.float32:
            pass
        elif np_array.dtype.kind == "O":
            raise ValueError(f"{name} has object dtype")
        elif np.issubdtype(np_array.dtype, np.integer):
            logger.warning(
                "Casting {} from {} to float32.", name, np_array.dtype
            )
            np_array = np_array.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"{name} has dtype {np_array.dtype}, expected float32 or integer."
            )
        return torch.from_numpy(np_array).to(torch.float32)

    @staticmethod
    def _make_scalar_metadata_tensor(value: float, length: int) -> Tensor:
        return torch.full((int(length), 1), float(value), dtype=torch.float32)

    @staticmethod
    def _derive_root_state_tensors(
        arrays: Dict[str, Tensor],
        prefix: str = "ref_",
    ) -> None:
        rg_pos_key = f"{prefix}rg_pos"
        rb_rot_key = f"{prefix}rb_rot"
        body_vel_key = f"{prefix}body_vel"
        body_ang_vel_key = f"{prefix}body_ang_vel"
        if (
            rg_pos_key not in arrays
            or rb_rot_key not in arrays
            or body_vel_key not in arrays
            or body_ang_vel_key not in arrays
        ):
            return
        # Keep root-level tensors consistent with the FK-derived body tensors.
        arrays[f"{prefix}root_pos"] = arrays[rg_pos_key][:, 0, :]
        arrays[f"{prefix}root_rot"] = arrays[rb_rot_key][:, 0, :]
        arrays[f"{prefix}root_vel"] = arrays[body_vel_key][:, 0, :]
        arrays[f"{prefix}root_ang_vel"] = arrays[body_ang_vel_key][:, 0, :]

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length
        arrays: Dict[str, Tensor] = {}

        for dataset_name in ("ref_root_pos", "ref_root_rot", "ref_dof_pos"):
            if dataset_name not in shard_handle:
                raise KeyError(
                    f"Missing mandatory dataset '{dataset_name}' in shard index "
                    f"{window.shard_index}"
                )
            np_array = np.asarray(shard_handle[dataset_name][start:end, ...])
            arrays[dataset_name] = self._cast_motion_np(np_array, dataset_name)

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            if frame_flag_np.dtype.kind == "O":
                raise ValueError("frame_flag has object dtype")
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 1:
                frame_flag[0] = 0
                frame_flag[-1] = 2
            elif window.length == 1:
                frame_flag[0] = 2
        arrays["frame_flag"] = frame_flag

        clip_meta = self.clips.get(window.raw_motion_key, {})
        metadata = clip_meta.get("metadata", {})
        motion_fps_val = metadata.get(
            "motion_fps", clip_meta.get("motion_fps")
        )
        if motion_fps_val is None:
            raise ValueError(
                f"motion_fps missing for clip {window.raw_motion_key}"
            )
        motion_fps = float(motion_fps_val)
        if motion_fps <= 0.0:
            raise ValueError(
                f"Invalid motion_fps {motion_fps} for clip {window.raw_motion_key}"
            )
        arrays["motion_fps"] = self._make_scalar_metadata_tensor(
            motion_fps, window.length
        )
        self._fk_transform(arrays, motion_fps)
        self._world_frame_transform(arrays)

        self._derive_root_state_tensors(arrays, prefix="ref_")

        if self._progress_counter is not None:
            with self._progress_counter.get_lock():
                self._progress_counter.value += 1

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            window_index=int(index),
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        # periodically clean up the file handles
        self._h5_access_counter += 1
        if self._h5_access_counter >= self._h5_cleanup_interval:
            self.close()
            self._h5_access_counter = 0

        if shard_index in self._file_handles:
            handle = self._file_handles.pop(shard_index)
            if handle.id:
                self._file_handles[shard_index] = handle
                return handle

        if shard_index < 0 or shard_index >= len(self._shard_paths):
            raise IndexError(
                f"Shard index {shard_index} out of range for "
                f"{len(self._shard_paths)} available shards"
            )
        shard_path = self._shard_paths[shard_index]
        rdcc_nbytes_env = os.getenv("HOLOMOTION_HDF5_RDCC_NBYTES")
        if rdcc_nbytes_env is None:
            rdcc_nbytes = 4 * 1024 * 1024
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
        if (
            self._h5_max_open_files is not None
            and len(self._file_handles) >= self._h5_max_open_files
        ):
            old_index, old_handle = self._file_handles.popitem(last=False)
            old_handle.close()
        self._file_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        logger.info("Clearing HDF5 file handles ...")
        for handle in self._file_handles.values():
            if handle.id:
                handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:
        self.close()


def _normalize_root_list(value: Any) -> List[str]:
    roots, _ = normalize_hdf5_root_entries(value)
    return roots


def normalize_hdf5_root_entries(
    value: Any,
) -> Tuple[List[str], Dict[str, float]]:
    """Parse string roots or ``{root, ratio}`` entries."""

    if value is None:
        return [], {}
    if isinstance(value, (str, os.PathLike)):
        return [str(value)], {}

    roots: List[str] = []
    dataset_ratios: Dict[str, float] = {}
    for index, entry in enumerate(value):
        if isinstance(entry, Mapping):
            root = entry.get("root", entry.get("path"))
            if not root:
                raise ValueError(
                    f"train_hdf5_roots entry {index} requires 'root'"
                )
            root_str = str(root)
            ratio = entry.get("ratio")
            if ratio is not None:
                dataset_id = _manifest_dataset_id(
                    {}, os.path.join(root_str, "manifest.json")
                )
                if dataset_id in dataset_ratios:
                    raise ValueError(
                        f"Duplicate ratio-bearing HDF5 dataset '{dataset_id}'"
                    )
                dataset_ratios[dataset_id] = float(ratio)
            roots.append(root_str)
            continue
        roots.append(str(entry))
    return roots, dataset_ratios


def weighted_bin_cfg_from_motion_cfg(
    motion_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = dict(motion_cfg.get("weighted_bin", {}) or {})
    _, root_ratios = normalize_hdf5_root_entries(
        motion_cfg.get("train_hdf5_roots")
    )
    if not root_ratios:
        return cfg
    cfg.pop("bin_regex_patterns", None)
    cfg.pop("bin_regrex_patterns", None)
    cfg.pop("dataset_ratios", None)
    cfg["dataset_ratios"] = root_ratios
    return cfg


def build_motion_datasets_from_cfg(
    motion_cfg: Mapping[str, Any],
    *,
    max_frame_length: int,
    min_window_length: int,
    world_frame_normalization: bool = True,
    handpicked_motion_names: Optional[List[str]] = None,
    excluded_motion_names: Optional[List[str]] = None,
    allowed_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[
    Dataset[MotionClipSample],
    Optional[Dataset[MotionClipSample]],
    Dict[str, Any],
]:
    preview_sampling_from_cfg(motion_cfg=motion_cfg)
    backend = str(motion_cfg.get("backend", "hdf5")).lower()
    if backend in ("hdf5", "hdf5_simple"):
        train_roots = _normalize_root_list(
            motion_cfg.get("train_hdf5_roots", None)
        )
        if len(train_roots) == 0:
            hdf5_root = motion_cfg.get("hdf5_root", None)
            if not hdf5_root:
                raise ValueError(
                    "HDF5 backend requires train_hdf5_roots or hdf5_root"
                )
            train_roots = [str(hdf5_root)]
        manifest_paths = [
            os.path.join(str(root), "manifest.json") for root in train_roots
        ]
        train_dataset = Hdf5MotionDataset(
            manifest_path=manifest_paths
            if len(manifest_paths) > 1
            else manifest_paths[0],
            max_frame_length=max_frame_length,
            min_window_length=min_window_length,
            handpicked_motion_names=handpicked_motion_names,
            excluded_motion_names=excluded_motion_names,
            world_frame_normalization=world_frame_normalization,
            allowed_prefixes=allowed_prefixes,
        )

        val_roots = _normalize_root_list(
            motion_cfg.get("val_hdf5_roots", motion_cfg.get("val_hdf5_root"))
        )
        val_dataset = None
        if len(val_roots) > 0:
            val_manifest_paths = [
                os.path.join(str(root), "manifest.json") for root in val_roots
            ]
            val_dataset = Hdf5MotionDataset(
                manifest_path=val_manifest_paths
                if len(val_manifest_paths) > 1
                else val_manifest_paths[0],
                max_frame_length=max_frame_length,
                min_window_length=min_window_length,
                handpicked_motion_names=handpicked_motion_names,
                excluded_motion_names=excluded_motion_names,
                world_frame_normalization=world_frame_normalization,
                allowed_prefixes=allowed_prefixes,
            )
        return train_dataset, val_dataset, {}

    if backend == "hdf5_v2":
        fk_robot_file_path = motion_cfg.get("fk_robot_file_path")
        cache_cfg = motion_cfg.get("cache", {})
        allowed_prefixes = cache_cfg.get(
            "allowed_prefixes",
            ["ref_"],
        )
        train_roots = _normalize_root_list(
            motion_cfg.get("train_hdf5_roots", None)
        )
        if len(train_roots) == 0:
            hdf5_root = motion_cfg.get("hdf5_root", None)
            if not hdf5_root:
                raise ValueError(
                    "HDF5 v2 backend requires train_hdf5_roots or hdf5_root"
                )
            train_roots = [str(hdf5_root)]
        train_manifest_paths = [
            os.path.join(str(root), "manifest.json") for root in train_roots
        ]
        train_dataset = Hdf5RootDofDataset(
            manifest_path=train_manifest_paths
            if len(train_manifest_paths) > 1
            else train_manifest_paths[0],
            max_frame_length=max_frame_length,
            min_window_length=min_window_length,
            handpicked_motion_names=handpicked_motion_names,
            excluded_motion_names=excluded_motion_names,
            fk_robot_file_path=fk_robot_file_path,
            allowed_prefixes=allowed_prefixes,
        )

        val_roots = _normalize_root_list(
            motion_cfg.get("val_hdf5_roots", motion_cfg.get("val_hdf5_root"))
        )
        val_dataset = None
        if len(val_roots) > 0:
            val_manifest_paths = [
                os.path.join(str(root), "manifest.json") for root in val_roots
            ]
            val_dataset = Hdf5RootDofDataset(
                manifest_path=val_manifest_paths
                if len(val_manifest_paths) > 1
                else val_manifest_paths[0],
                max_frame_length=max_frame_length,
                min_window_length=min_window_length,
                handpicked_motion_names=handpicked_motion_names,
                excluded_motion_names=excluded_motion_names,
                fk_robot_file_path=fk_robot_file_path,
                allowed_prefixes=allowed_prefixes,
            )
        cache_kwargs = {
            "stage_on_swap_only": bool(
                motion_cfg.get("stage_on_swap_only", True)
            )
        }
        return train_dataset, val_dataset, cache_kwargs

    raise ValueError(f"Unsupported motion backend: {backend}")


def _cache_collate_fn(
    samples: List[MotionClipSample],
    mode: str,
    batch_size: int,
) -> ClipBatch:
    """Collate function for motion cache DataLoader (supports validation padding)."""
    if mode == "val" and batch_size > len(samples) and len(samples) > 0:
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
        self._counts = _allocate_batch_counts(
            raw_counts=raw_counts,
            target_total=self._batch_size,
        )

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


class PrioritizedInfiniteSampler(Sampler[int]):
    """Infinite sampler with persistent prioritized and fresh uniform pools."""

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        seed: int,
        *,
        p_a_ratio: float = 0.2,
        ema_alpha_signal: float = 0.2,
        ema_alpha_rel_improve: float = 0.2,
        relative_eps: float = 1.0e-6,
    ) -> None:
        self._ds_len = int(max(0, dataset_len))
        self._batch_size = int(max(1, batch_size))
        self._seed = int(seed)
        self._epoch = 0

        self._p_a_ratio = float(min(1.0, max(0.0, p_a_ratio)))
        self._ema_alpha_signal = float(min(1.0, max(0.0, ema_alpha_signal)))
        self._ema_alpha_rel_improve = float(
            min(1.0, max(0.0, ema_alpha_rel_improve))
        )
        self._relative_eps = float(max(1.0e-12, relative_eps))

        if self._ds_len <= 0:
            self._ema_completion_rate = torch.zeros(0, dtype=torch.float32)
            self._ema_completion_rate_sq = torch.zeros(0, dtype=torch.float32)
            self._ema_completion_rel_improve = torch.zeros(
                0, dtype=torch.float32
            )
            self._selection_counts = torch.zeros(0, dtype=torch.long)
            self._seen_mask = torch.zeros(0, dtype=torch.bool)
            self._prioritized_pool_indices = torch.zeros(0, dtype=torch.long)
            self._prioritized_pool_mask = torch.zeros(0, dtype=torch.bool)
        else:
            self._ema_completion_rate = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._ema_completion_rate_sq = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._ema_completion_rel_improve = torch.zeros(
                self._ds_len, dtype=torch.float32
            )
            self._selection_counts = torch.zeros(
                self._ds_len, dtype=torch.long
            )
            self._seen_mask = torch.zeros(self._ds_len, dtype=torch.bool)
            self._prioritized_pool_indices = torch.zeros(0, dtype=torch.long)
            self._prioritized_pool_mask = torch.zeros(
                self._ds_len, dtype=torch.bool
            )
        self._state_version = 0
        self._last_updated_swap = -1
        self._last_prioritized_pool_mean_score = 0.0
        self._last_uniform_pool_mean_score = 0.0
        self._last_entered_prioritized_pool_count = 0
        self._last_exited_prioritized_pool_count = 0
        self._uniform_cycle_start = 0
        self._uniform_cycle_step = 1
        self._uniform_cycle_offset = self._ds_len
        self._uniform_cycle_epoch = 0

    @property
    def state_version(self) -> int:
        return int(self._state_version)

    def get_pool_statistics(self) -> Optional[Dict[str, float]]:
        if self._ds_len <= 0:
            return None
        return self._pool_metric_stats()

    @staticmethod
    def _aggregate_by_index(
        window_indices: Tensor,
        values: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if window_indices.numel() == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float32),
            )
        unique_indices, inverse = torch.unique(
            window_indices.to(dtype=torch.long),
            sorted=False,
            return_inverse=True,
        )
        out_weighted_sum = torch.zeros(
            unique_indices.numel(), dtype=torch.float32
        )
        out_count = torch.zeros(unique_indices.numel(), dtype=torch.float32)
        out_weighted_sum.scatter_add_(0, inverse, values * counts)
        out_count.scatter_add_(0, inverse, counts)
        return unique_indices, out_weighted_sum, out_count

    def _pool_batch_sizes(self) -> Tuple[int, int]:
        if self._ds_len <= 0:
            return 0, 0
        uniform_count = int(round(self._p_a_ratio * float(self._batch_size)))
        uniform_count = max(0, min(self._batch_size, uniform_count))
        prioritized_count = max(0, self._batch_size - uniform_count)
        return uniform_count, prioritized_count

    def _priority_scores_for_indices(self, indices: Tensor) -> Tensor:
        if indices.numel() == 0 or self._ds_len <= 0:
            return torch.zeros(0, dtype=torch.float32)
        idx = indices.to(dtype=torch.long)
        progress = torch.clamp(
            self._ema_completion_rel_improve.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        remaining_difficulty = torch.clamp(
            1.0 - self._ema_completion_rate.index_select(0, idx),
            min=0.0,
            max=1.0,
        )
        seen = self._seen_mask.index_select(0, idx).to(dtype=torch.float32)
        return progress * remaining_difficulty * seen

    def _pool_metric_stats(self) -> Dict[str, float]:
        prioritized_pool_size = int(self._prioritized_pool_indices.numel())
        return {
            "prioritized_pool_size": float(prioritized_pool_size),
            "prioritized_pool_mean_score": float(
                self._last_prioritized_pool_mean_score
            ),
            "uniform_pool_mean_score": float(
                self._last_uniform_pool_mean_score
            ),
            "entered_prioritized_pool_count": float(
                self._last_entered_prioritized_pool_count
            ),
            "exited_prioritized_pool_count": float(
                self._last_exited_prioritized_pool_count
            ),
        }

    def get_window_state_for_indices(
        self, window_indices: Tensor
    ) -> Dict[str, Tensor]:
        if self._ds_len <= 0:
            empty_bool = torch.zeros(0, dtype=torch.bool)
            empty_float = torch.zeros(0, dtype=torch.float32)
            return {
                "ema_completion_rate": empty_float,
                "completion_rate_rel_improve": empty_float,
                "selection_count": torch.zeros(0, dtype=torch.long),
                "seen": empty_bool,
                "in_prioritized_pool": empty_bool,
            }
        idx = window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        if idx.numel() == 0:
            empty_bool = torch.zeros(0, dtype=torch.bool)
            empty_float = torch.zeros(0, dtype=torch.float32)
            return {
                "ema_completion_rate": empty_float,
                "completion_rate_rel_improve": empty_float,
                "selection_count": torch.zeros(0, dtype=torch.long),
                "seen": empty_bool,
                "in_prioritized_pool": empty_bool,
            }
        return {
            "ema_completion_rate": self._ema_completion_rate.index_select(
                0, idx
            ).to(dtype=torch.float32),
            "completion_rate_rel_improve": (
                self._ema_completion_rel_improve.index_select(0, idx).to(
                    dtype=torch.float32
                )
            ),
            "selection_count": self._selection_counts.index_select(0, idx),
            "seen": self._seen_mask.index_select(0, idx),
            "in_prioritized_pool": self._prioritized_pool_mask.index_select(
                0, idx
            ),
        }

    def _rebuild_prioritized_pool(self, candidate_indices: Tensor) -> None:
        if self._ds_len <= 0:
            return
        _, prioritized_count = self._pool_batch_sizes()
        previous_indices = self._prioritized_pool_indices
        selected = torch.zeros(0, dtype=torch.long)
        if prioritized_count > 0:
            candidates = torch.cat(
                [
                    previous_indices.to(dtype=torch.long),
                    candidate_indices.to(dtype=torch.long).reshape(-1),
                ]
            )
            candidates = torch.unique(candidates, sorted=False)
            scores = self._priority_scores_for_indices(candidates)
            positive = scores > 0.0
            if bool(positive.any().item()):
                candidates = candidates[positive]
                scores = scores[positive]
                order = torch.argsort(scores, descending=True)
                selected = candidates.index_select(
                    0, order[: min(prioritized_count, candidates.numel())]
                )
                scores = scores.index_select(
                    0, order[: min(prioritized_count, scores.numel())]
                )
                self._last_prioritized_pool_mean_score = float(
                    scores.mean().item()
                )
            else:
                self._last_prioritized_pool_mean_score = 0.0
            if candidates.numel() > selected.numel():
                selected_mask = torch.zeros(
                    candidates.numel(), dtype=torch.bool
                )
                if selected.numel() > 0:
                    matches = candidates[:, None] == selected[None, :]
                    selected_mask = matches.any(dim=1)
                nonselected_scores = self._priority_scores_for_indices(
                    candidates[~selected_mask]
                )
                self._last_uniform_pool_mean_score = (
                    float(nonselected_scores.mean().item())
                    if nonselected_scores.numel() > 0
                    else 0.0
                )
            else:
                self._last_uniform_pool_mean_score = 0.0
        else:
            self._last_prioritized_pool_mean_score = 0.0
            self._last_uniform_pool_mean_score = 0.0
        if previous_indices.numel() > 0:
            self._prioritized_pool_mask[previous_indices] = False
        if selected.numel() > 0:
            self._prioritized_pool_mask[selected] = True
        previous_set = set(previous_indices.tolist())
        selected_set = set(selected.tolist())
        self._last_entered_prioritized_pool_count = len(
            selected_set - previous_set
        )
        self._last_exited_prioritized_pool_count = len(
            previous_set - selected_set
        )
        self._prioritized_pool_indices = selected

    def maybe_update_from_observations(
        self,
        *,
        window_indices: Tensor,
        mpkpe_signal_means: Tensor,
        completion_rate_means: Tensor,
        counts: Tensor,
        swap_index: int,
    ) -> bool:
        if self._ds_len <= 0:
            return False
        swap_idx = int(swap_index)
        if swap_idx <= 0:
            return False
        if self._last_updated_swap == swap_idx:
            return False

        indices = (
            window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        )
        # Keep validating the MPKPE tensor shape so the command-side
        # curriculum aggregation stays aligned with completion-rate updates.
        mpkpe_signal_numel = int(mpkpe_signal_means.numel())
        completion_rate = (
            completion_rate_means.detach()
            .to(dtype=torch.float32)
            .reshape(-1)
            .cpu()
        )
        cnt = counts.detach().to(dtype=torch.float32).reshape(-1).cpu()
        if not (
            indices.numel() == mpkpe_signal_numel
            and mpkpe_signal_numel == completion_rate.numel()
            and completion_rate.numel() == cnt.numel()
        ):
            raise ValueError(
                "Prioritized sampler update tensors must have matching shape."
            )

        valid_dataset_idx = (indices >= 0) & (indices < self._ds_len)
        valid = (
            valid_dataset_idx & torch.isfinite(completion_rate) & (cnt > 0.0)
        )
        current_batch_indices = torch.unique(
            indices[valid_dataset_idx], sorted=False
        )
        if not bool(valid.any().item()):
            self._last_entered_prioritized_pool_count = 0
            self._last_exited_prioritized_pool_count = 0
            self._last_updated_swap = swap_idx
            return False

        idx_valid = indices[valid]
        completion_rate_valid = completion_rate[valid]
        cnt_valid = cnt[valid]

        touched_idx, completion_rate_sum, completion_rate_count_sum = (
            self._aggregate_by_index(
                idx_valid,
                completion_rate_valid,
                cnt_valid,
            )
        )
        if touched_idx.numel() == 0:
            self._last_entered_prioritized_pool_count = 0
            self._last_exited_prioritized_pool_count = 0
            self._last_updated_swap = swap_idx
            return False

        completion_rate_obs = (
            completion_rate_sum / completion_rate_count_sum.clamp_min(1.0e-12)
        )
        completion_rate_obs = torch.clamp(
            completion_rate_obs, min=0.0, max=1.0
        )

        prev_seen = self._seen_mask[touched_idx]
        prev_completion_rate = self._ema_completion_rate[touched_idx]
        prev_completion_rate_sq = self._ema_completion_rate_sq[touched_idx]
        prev_completion_rate_var = torch.clamp(
            prev_completion_rate_sq
            - prev_completion_rate * prev_completion_rate,
            min=1.0e-6,
        )
        prev_completion_rate_std = torch.sqrt(prev_completion_rate_var)
        next_completion_rate = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_signal) * prev_completion_rate
            + self._ema_alpha_signal * completion_rate_obs,
            completion_rate_obs,
        )
        next_completion_rate_sq = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_signal) * prev_completion_rate_sq
            + self._ema_alpha_signal
            * (completion_rate_obs * completion_rate_obs),
            completion_rate_obs * completion_rate_obs,
        )

        completion_rel_improve_obs = torch.zeros_like(next_completion_rate)
        completion_rel_improve_obs[prev_seen] = torch.tanh(
            (completion_rate_obs[prev_seen] - prev_completion_rate[prev_seen])
            / (prev_completion_rate_std[prev_seen] + self._relative_eps)
        )
        prev_completion_rel = self._ema_completion_rel_improve[touched_idx]
        next_completion_rel = torch.where(
            prev_seen,
            (1.0 - self._ema_alpha_rel_improve) * prev_completion_rel
            + self._ema_alpha_rel_improve * completion_rel_improve_obs,
            completion_rel_improve_obs,
        )

        self._ema_completion_rate[touched_idx] = next_completion_rate
        self._ema_completion_rate_sq[touched_idx] = next_completion_rate_sq
        self._ema_completion_rel_improve[touched_idx] = next_completion_rel
        self._seen_mask[touched_idx] = True

        self._rebuild_prioritized_pool(touched_idx)
        self._state_version += 1
        self._last_updated_swap = swap_idx
        return True

    def _reset_uniform_cycle(self) -> None:
        if self._ds_len <= 0:
            self._uniform_cycle_start = 0
            self._uniform_cycle_step = 1
            self._uniform_cycle_offset = 0
            return
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._uniform_cycle_epoch * 1000003)
        self._uniform_cycle_epoch += 1
        self._uniform_cycle_start = int(
            torch.randint(
                low=0,
                high=self._ds_len,
                size=(1,),
                generator=generator,
            ).item()
        )
        if self._ds_len <= 1:
            self._uniform_cycle_step = 1
        else:
            step = int(
                torch.randint(
                    low=1,
                    high=self._ds_len,
                    size=(1,),
                    generator=generator,
                ).item()
            )
            while math.gcd(step, self._ds_len) != 1:
                step += 1
                if step >= self._ds_len:
                    step = 1
            self._uniform_cycle_step = step
        self._uniform_cycle_offset = 0

    def _next_uniform_index(self) -> int:
        if self._uniform_cycle_offset >= self._ds_len:
            self._reset_uniform_cycle()
        next_index = (
            self._uniform_cycle_start
            + self._uniform_cycle_offset * self._uniform_cycle_step
        ) % self._ds_len
        self._uniform_cycle_offset += 1
        return int(next_index)

    def _sample_uniform_indices(
        self,
        generator: torch.Generator,
        count: int,
        *,
        exclude: Optional[Tensor] = None,
    ) -> Tensor:
        del generator
        if count <= 0 or self._ds_len <= 0:
            return torch.zeros(0, dtype=torch.long)
        blocked = set()
        if exclude is not None and exclude.numel() > 0:
            blocked.update(
                exclude.detach().to(dtype=torch.long).reshape(-1).tolist()
            )
        take = min(int(count), max(0, self._ds_len - len(blocked)))
        if take <= 0:
            return torch.zeros(0, dtype=torch.long)
        selected: List[int] = []
        stagnant_steps = 0
        while len(selected) < take and stagnant_steps < self._ds_len:
            next_index = self._next_uniform_index()
            if next_index in blocked:
                stagnant_steps += 1
                continue
            selected.append(next_index)
            blocked.add(next_index)
            stagnant_steps = 0
        return torch.tensor(selected, dtype=torch.long)

    def _sample_prioritized_indices(
        self, generator: torch.Generator, count: int
    ) -> Tensor:
        if count <= 0 or self._prioritized_pool_indices.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        perm = torch.randperm(
            self._prioritized_pool_indices.numel(), generator=generator
        )
        take = min(count, int(self._prioritized_pool_indices.numel()))
        return self._prioritized_pool_indices.index_select(0, perm[:take])

    def _sample_batch_indices(self, generator: torch.Generator) -> Tensor:
        uniform_count, prioritized_count = self._pool_batch_sizes()
        prioritized_indices = self._sample_prioritized_indices(
            generator, prioritized_count
        )
        uniform_indices = self._sample_uniform_indices(
            generator,
            uniform_count,
            exclude=prioritized_indices,
        )
        sampled_indices = torch.cat(
            [uniform_indices, prioritized_indices], dim=0
        )
        if sampled_indices.numel() < self._batch_size:
            extra_indices = self._sample_uniform_indices(
                generator,
                self._batch_size - int(sampled_indices.numel()),
                exclude=sampled_indices,
            )
            sampled_indices = torch.cat(
                [sampled_indices, extra_indices], dim=0
            )
        if sampled_indices.numel() != self._batch_size:
            raise ValueError(
                "Prioritized sampler failed to assemble a full cache batch."
            )

        if sampled_indices.numel() > 0:
            self._selection_counts[sampled_indices] += 1
        return sampled_indices

    def get_scores_for_indices(self, window_indices: Tensor) -> Tensor:
        if self._ds_len <= 0:
            return torch.zeros_like(window_indices, dtype=torch.float32)
        idx = window_indices.detach().to(dtype=torch.long).reshape(-1).cpu()
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.float32)
        scores = self._priority_scores_for_indices(idx)
        return scores.to(dtype=torch.float32)

    def __iter__(self):
        while True:
            if self._ds_len <= 0:
                raise ValueError(
                    "PrioritizedInfiniteSampler cannot sample from "
                    "an empty dataset."
                )
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampled_indices = self._sample_batch_indices(generator=g)
            perm = torch.randperm(sampled_indices.numel(), generator=g)
            yielded_indices = sampled_indices.index_select(0, perm)
            for idx in yielded_indices.tolist():
                yield int(idx)
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
        self._world_frame_transform = (
            _WorldFrameNormalizeTransform()
            if bool(world_frame_normalization)
            else None
        )
        self._allowed_prefixes: Tuple[str, ...] = ("ref_",)
        self._progress_counter: Optional[mp.Value] = None

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
            dataset_id = _manifest_dataset_id(manifest, mp)

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
                meta_global["_dataset_id"] = dataset_id
                meta_global["shard"] = (
                    int(meta_global.get("shard", 0)) + shard_offset
                )
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

    def set_progress_counter(self, counter: Optional[mp.Value]) -> None:
        self._progress_counter = counter

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
                    window_index += 1
                offset += window_length
                remaining = max(0, length - offset)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MotionClipSample:
        window = self.windows[index]
        shard_handle = self._get_shard_handle(window.shard_index)
        start, end = window.start, window.start + window.length

        arrays: Dict[str, Tensor] = {}

        # Mandatory reference source: ref_*
        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            dname = f"ref_{dataset_name}"
            if dname not in shard_handle:
                raise KeyError(
                    f"Missing mandatory dataset '{dname}' in shard index {window.shard_index}"
                )
            np_array = shard_handle[dname][start:end]
            arrays[f"ref_{logical_name}"] = torch.from_numpy(np_array).to(
                torch.float32
            )

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 1:
                frame_flag[0] = 0
                frame_flag[-1] = 2
            elif window.length == 1:
                # Single-frame window: mark as both start and end (use 2 for end)
                frame_flag[0] = 2
        arrays["frame_flag"] = frame_flag

        if self._world_frame_transform is not None:
            self._world_frame_transform(arrays)

        # Derived root_* for ref_* (after normalization)
        arrays["ref_root_pos"] = arrays["ref_rg_pos"][:, 0, :]
        arrays["ref_root_rot"] = arrays["ref_rb_rot"][:, 0, :]
        arrays["ref_root_vel"] = arrays["ref_body_vel"][:, 0, :]
        arrays["ref_root_ang_vel"] = arrays["ref_body_ang_vel"][:, 0, :]

        if self._progress_counter is not None:
            with self._progress_counter.get_lock():
                self._progress_counter.value += 1

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            window_index=int(index),
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

    @staticmethod
    def _infer_cuda_device_index() -> int:
        device_count = int(torch.cuda.device_count())
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            if 0 <= local_rank < device_count:
                return local_rank
        return int(torch.cuda.current_device())

    @classmethod
    def _normalize_stage_device(
        cls, stage_device: Optional[object]
    ) -> Optional[torch.device]:
        if stage_device is None:
            return None

        if isinstance(stage_device, torch.device):
            if stage_device.type == "cpu":
                return None
            if stage_device.type != "cuda":
                raise ValueError(
                    f"Unsupported stage_device type: {stage_device.type}"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "stage_device requested CUDA but CUDA is not available"
                )
            if stage_device.index is not None:
                return stage_device
            return torch.device("cuda", cls._infer_cuda_device_index())

        if isinstance(stage_device, str):
            stage_device_str = stage_device.strip().lower()
            if stage_device_str in ("none", "cpu"):
                return None
            if stage_device_str == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "stage_device requested CUDA but CUDA is not available"
                    )
                return torch.device("cuda", cls._infer_cuda_device_index())
            if stage_device_str.startswith("cuda:"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "stage_device requested CUDA but CUDA is not available"
                    )
                return torch.device(stage_device_str)
            raise ValueError(
                f"Unsupported stage_device string: {stage_device}"
            )

        raise TypeError(
            f"Unsupported stage_device value type: {type(stage_device)}"
        )

    def __init__(
        self,
        train_dataset: Dataset[MotionClipSample],
        *,
        val_dataset: Optional[Dataset[MotionClipSample]] = None,
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
        stage_on_swap_only: bool = False,
        batch_progress_bar: bool = False,
        seed: Optional[int] = None,
        loader_timeout: float = 0.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if float(loader_timeout) < 0.0:
            raise ValueError("loader_timeout must be >= 0")

        self._datasets = {
            "train": train_dataset,
            "val": val_dataset if val_dataset is not None else train_dataset,
        }
        self._mode = "train"
        self._seed = (
            int(seed) if seed is not None else int(time.time_ns() & 0x7FFFFFFF)
        )
        self._stage_device = self._normalize_stage_device(stage_device)
        self._sampler_rank = int(sampler_rank)
        self._sampler_world_size = int(max(1, sampler_world_size))
        self._batch_size = int(batch_size)
        self._allowed_prefixes: Optional[Tuple[str, ...]] = (
            tuple(allowed_prefixes) if allowed_prefixes is not None else None
        )

        # If enabled, keep the prefetched batch on CPU (FK on CPU) and stage to GPU
        # only during cache swapping (advance).
        self._stage_on_swap_only = bool(stage_on_swap_only)
        self._batch_progress_bar = bool(batch_progress_bar)
        self._loader_timeout = float(loader_timeout)
        self.force_timeout_on_swap = bool(force_timeout_on_swap)
        self._batch_progress_counter: Optional[mp.Value] = None
        if self._should_use_batch_progress():
            ctx = mp.get_context("spawn")
            self._batch_progress_counter = ctx.Value("i", 0)

        self.swap_interval_steps = (
            swap_interval_steps
            if swap_interval_steps is not None
            else train_dataset.max_frame_length
        )

        self._num_workers = int(max(0, num_workers))
        self._prefetch_factor = (
            prefetch_factor if prefetch_factor is not None else None
        )
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers and num_workers > 0)

        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[Sampler[int]] = None
        self._iterator: Optional[Iterator[ClipBatch]] = None

        self._current_batch: Optional[ClipBatch] = None
        self._next_batch: Optional[ClipBatch] = None
        self._swap_index = 0

        self._effective_batch_size: Optional[int] = None
        self._num_batches: Optional[int] = None

        # Weighted-bin sampling state
        self._weighted_bin_enabled: bool = False
        self._weighted_bin_bins: Optional[List[List[int]]] = None
        self._weighted_bin_ratios: Optional[List[float]] = None
        self._weighted_bin_specs: Optional[List[Dict[str, Any]]] = None
        self._cache_curriculum_enabled: bool = False
        self._cache_curriculum_cfg: Dict[str, Any] = {}
        self._cache_curriculum_sampler: Optional[
            PrioritizedInfiniteSampler
        ] = None
        self._cache_curriculum_dump_enabled: bool = False
        self._cache_curriculum_dump_every_swaps: int = 10
        self._cache_curriculum_dump_chunk_size: int = 4096
        self._cache_curriculum_dump_dir: Path = Path(
            "cache_curriculum_window_scores"
        )
        self._cache_curriculum_last_dump_swap: int = -1

        # Async GPU staging helpers
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        self._build_dataloader()
        if (
            self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            self._copy_stream = torch.cuda.Stream(device=self._stage_device)
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

    def set_seed(self, seed: int, *, reinitialize: bool = True) -> None:
        self._seed = int(seed)
        if reinitialize:
            self._build_dataloader()
            self._prime_buffers()

    def advance(self) -> None:
        if self._stage_on_swap_only:
            if self._next_batch is None:
                self._next_batch = self._fetch_next_batch()
            old_current = self._current_batch
            self._current_batch = None
            del old_current
            # Stage the prefetched CPU batch to GPU only at swap time.
            try:
                staged = self._stage_batch_blocking(self._next_batch)
            except torch.cuda.OutOfMemoryError:
                if (
                    self._stage_device is None
                    or self._stage_device.type != "cuda"
                ):
                    raise
                torch.cuda.empty_cache()
                staged = self._stage_batch_blocking(self._next_batch)
            self._current_batch = staged
            self._next_batch = self._fetch_next_batch()
            self._swap_index += 1
            return

        if self._next_batch is None:
            self._next_batch = self._fetch_next_batch()
        # Ensure asynchronous staging finished before swapping in next batch
        if (
            self._next_ready_event is not None
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            torch.cuda.current_stream(self._stage_device).wait_event(
                self._next_ready_event
            )
        self._current_batch = self._next_batch
        self._next_batch = self._fetch_next_batch()
        self._swap_index += 1

    # -------------------------
    # Weighted-bin configuration
    # -------------------------
    def enable_weighted_bin_sampling(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enable regex-based weighted-bin sampling over manifest motion keys.

        Prefer ``dataset_ratios`` to classify clips by their manifest's
        ``train_*``/``eval_*`` directory. ``bin_regex_patterns`` (or the legacy
        name ``bin_regrex_patterns``) remains available for clip-level matching.
        Each generated or explicit pattern provides:

        - ``regex`` (or ``regrex``): Python regular expression applied to the
          manifest clip key (e.g., ``AMASS_.*``, ``VR_pico_.*``).
        - ``ratio``: Target sampling ratio in [0, 1].

        The sum of explicit bin ratios must be <= 1.0. Any remaining mass is
        assigned to an implicit ``others`` bin that collects all clips not
        matched by any regex.
        """
        cfg_local: Dict[str, Any] = dict(cfg or {})
        if self._cache_curriculum_enabled:
            raise ValueError(
                "weighted-bin and cache curriculum sampling cannot be enabled together."
            )

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
            clip_metadata = dataset.clips.get(motion_key, {})
            window_keys.append(
                _weighted_bin_motion_key(motion_key, clip_metadata)
            )

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
        self._build_dataloader()
        self._prime_buffers()

    def enable_cache_curriculum_sampling(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._weighted_bin_enabled:
            raise ValueError(
                "cache curriculum and weighted-bin sampling cannot be enabled together."
            )
        self._cache_curriculum_enabled = True
        self._cache_curriculum_cfg = dict(cfg or {})
        self._cache_curriculum_dump_enabled = bool(
            self._cache_curriculum_cfg.get(
                "dump_whole_window_scores_json", True
            )
        )
        self._cache_curriculum_dump_every_swaps = max(
            1,
            int(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_every_swaps", 10
                )
            ),
        )
        self._cache_curriculum_dump_chunk_size = max(
            1,
            int(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_chunk_size", 4096
                )
            ),
        )
        self._cache_curriculum_dump_dir = Path(
            str(
                self._cache_curriculum_cfg.get(
                    "dump_whole_window_scores_dir",
                    "cache_curriculum_window_scores",
                )
            )
        )
        self._cache_curriculum_last_dump_swap = -1
        self._prepare_cache_curriculum_dump_dir(
            self._cache_curriculum_dump_dir,
            reason="enabled",
        )
        self._build_dataloader()
        self._prime_buffers()

    def _prepare_cache_curriculum_dump_dir(
        self, dump_dir: Path, *, reason: str
    ) -> None:
        self._cache_curriculum_dump_dir = Path(str(dump_dir))
        if not self._cache_curriculum_dump_enabled:
            return
        self._cache_curriculum_dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Cache curriculum whole-window score dump "
            f"{reason}: dir={self._cache_curriculum_dump_dir}, "
            f"every_swaps={self._cache_curriculum_dump_every_swaps}, "
            f"rank={self._sampler_rank}"
        )

    def set_cache_curriculum_dump_dir(self, dump_dir: str) -> None:
        self._prepare_cache_curriculum_dump_dir(
            Path(str(dump_dir)),
            reason="directory set",
        )

    def update_cache_curriculum(
        self,
        *,
        window_indices: Tensor,
        mpkpe_signal_means: Tensor,
        completion_rate_means: Tensor,
        counts: Tensor,
        swap_index: int,
    ) -> bool:
        if self._cache_curriculum_sampler is None:
            return False
        updated = (
            self._cache_curriculum_sampler.maybe_update_from_observations(
                window_indices=window_indices,
                mpkpe_signal_means=mpkpe_signal_means,
                completion_rate_means=completion_rate_means,
                counts=counts,
                swap_index=swap_index,
            )
        )
        if updated:
            self._refresh_prefetched_batch()
        self._maybe_dump_cache_curriculum_scores_json(swap_index=swap_index)
        return updated

    def _refresh_prefetched_batch(self) -> None:
        if self._next_batch is None:
            return
        self._next_batch = self._fetch_next_batch()

    def _maybe_dump_cache_curriculum_scores_json(
        self, *, swap_index: int
    ) -> None:
        if not self._cache_curriculum_dump_enabled:
            return
        if self._cache_curriculum_sampler is None:
            return

        swap_idx = int(swap_index)
        if swap_idx <= 0:
            return
        if swap_idx % self._cache_curriculum_dump_every_swaps != 0:
            return
        if self._cache_curriculum_last_dump_swap == swap_idx:
            return

        dataset = self._datasets["train"]
        ds_len = int(len(dataset))
        if ds_len <= 0:
            return

        self._cache_curriculum_dump_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._cache_curriculum_dump_dir / (
            "whole_window_scores_"
            f"rank_{self._sampler_rank:04d}_swap_{swap_idx:06d}.json"
        )
        sampler_version = int(self._cache_curriculum_sampler.state_version)
        windows = dataset.windows
        score_values: List[float] = []
        completion_values: List[float] = []
        rel_improve_values: List[float] = []
        selection_count_values: List[int] = []
        seen_values: List[bool] = []
        in_pool_values: List[bool] = []
        chunk_size = max(1, int(self._cache_curriculum_dump_chunk_size))
        for chunk_start in range(0, ds_len, chunk_size):
            chunk_end = min(ds_len, chunk_start + chunk_size)
            chunk_indices = torch.arange(
                chunk_start, chunk_end, dtype=torch.long
            )
            chunk_scores = (
                self._cache_curriculum_sampler.get_scores_for_indices(
                    chunk_indices
                )
            )
            chunk_state = (
                self._cache_curriculum_sampler.get_window_state_for_indices(
                    chunk_indices
                )
            )
            if chunk_scores.numel() != chunk_indices.numel():
                raise ValueError(
                    "Whole-window score dump shape mismatch for "
                    "cache curriculum sampler."
                )
            score_values.extend(chunk_scores.tolist())
            completion_values.extend(
                chunk_state["ema_completion_rate"].tolist()
            )
            rel_improve_values.extend(
                chunk_state["completion_rate_rel_improve"].tolist()
            )
            selection_count_values.extend(
                chunk_state["selection_count"].tolist()
            )
            seen_values.extend(chunk_state["seen"].tolist())
            in_pool_values.extend(chunk_state["in_prioritized_pool"].tolist())
        rows: List[Dict[str, Any]] = []
        for window_index in range(ds_len):
            window = windows[window_index]
            rows.append(
                {
                    "swap_index": int(swap_idx),
                    "rank": int(self._sampler_rank),
                    "sampler_state_version": sampler_version,
                    "window_index": int(window_index),
                    "raw_motion_key": str(window.raw_motion_key),
                    "motion_key": str(window.motion_key),
                    "start": int(window.start),
                    "length": int(window.length),
                    "score": float(score_values[window_index]),
                    "selection_count": int(
                        selection_count_values[window_index]
                    ),
                    "ema_completion_rate": float(
                        completion_values[window_index]
                    ),
                    "completion_rate_rel_improve": float(
                        rel_improve_values[window_index]
                    ),
                    "seen": bool(seen_values[window_index]),
                    "in_prioritized_pool": bool(in_pool_values[window_index]),
                }
            )
        payload: Dict[str, Any] = {
            "swap_index": int(swap_idx),
            "rank": int(self._sampler_rank),
            "sampler_state_version": sampler_version,
            "num_windows": int(ds_len),
            "pool_metrics": self._cache_curriculum_sampler.get_pool_statistics()
            or {},
            "rows": rows,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        self._cache_curriculum_last_dump_swap = swap_idx

    def cache_curriculum_scores_for_window_indices(
        self, window_indices: Tensor
    ) -> Optional[Tuple[Tensor, Dict[str, Tensor], int]]:
        if self._cache_curriculum_sampler is None:
            return None
        scores = self._cache_curriculum_sampler.get_scores_for_indices(
            window_indices
        )
        state = self._cache_curriculum_sampler.get_window_state_for_indices(
            window_indices
        )
        version = self._cache_curriculum_sampler.state_version
        return scores, state, version

    def cache_curriculum_pool_statistics(
        self,
    ) -> Optional[Dict[str, float]]:
        if self._cache_curriculum_sampler is None:
            return None
        return self._cache_curriculum_sampler.get_pool_statistics()

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

        total = int(lengths.shape[0])
        if total == 0:
            raise ValueError(
                "Cannot sample from an empty batch. Ensure the cache contains "
                "at least one motion clip before calling sample_env_assignments."
            )
        clip_indices = torch.randint(
            low=0, high=total, size=(num_envs,), device=device
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

    def _prepare_gather_indices(
        self,
        *,
        clip_indices: Tensor,
        frame_indices: Tensor,
        n_future_frames: int,
    ) -> Tuple[Tensor, Tensor]:
        batch = self.current_batch
        staged_device = batch.lengths.device
        selected_clips = clip_indices.to(
            staged_device, dtype=torch.long
        ).clone()
        frame_indices = frame_indices.to(
            staged_device, dtype=torch.long
        ).clone()

        temporal_span = 1 + int(n_future_frames)
        time_offsets = torch.arange(
            temporal_span, device=staged_device, dtype=torch.long
        )
        gather_timesteps = frame_indices[:, None] + time_offsets[None, :]

        lengths = batch.lengths
        max_valid = torch.clamp(
            lengths.index_select(0, selected_clips) - 1, min=0
        )
        gather_timesteps = torch.minimum(
            gather_timesteps, max_valid[:, None]
        ).clone()

        return selected_clips, gather_timesteps

    def gather_tensor(
        self,
        tensor_name: str,
        *,
        clip_indices: Tensor,
        frame_indices: Tensor,
        n_future_frames: int,
    ) -> Tensor:
        batch = self.current_batch
        if tensor_name not in batch.tensors:
            raise KeyError(
                f"Tensor '{tensor_name}' is not present in current_batch"
            )
        selected_clips, gather_timesteps = self._prepare_gather_indices(
            clip_indices=clip_indices,
            frame_indices=frame_indices,
            n_future_frames=n_future_frames,
        )
        tensor = batch.tensors[tensor_name]
        return tensor[selected_clips[:, None], gather_timesteps, ...]

    def lengths_for_indices(self, clip_indices: Tensor) -> Tensor:
        lengths = self.current_batch.lengths.to(clip_indices.device)
        return lengths.index_select(0, clip_indices.long())

    def motion_keys_for_indices(self, clip_indices: Tensor) -> List[str]:
        result = []
        base_keys = self.current_batch.motion_keys
        for idx in clip_indices.tolist():
            result.append(base_keys[int(idx)])
        return result

    def window_indices_for_indices(self, clip_indices: Tensor) -> Tensor:
        base_indices = self.current_batch.window_indices.to(
            clip_indices.device
        )
        return base_indices.index_select(0, clip_indices.long())

    def _prime_buffers(self) -> None:
        if self._stage_on_swap_only:
            # Prefetch on CPU; stage to GPU only for current batch.
            cpu_current = self._fetch_next_batch()
            self._current_batch = self._stage_batch_blocking(cpu_current)
            self._next_batch = self._fetch_next_batch()
            self._pending_ready_event = None
            self._current_ready_event = None
            self._next_ready_event = None
            return

        self._current_batch = self._fetch_next_batch()
        # Ensure first staged batch is ready before consumption
        if (
            self._current_ready_event is not None
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
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
        batch = self._load_next_batch()
        if self._stage_on_swap_only:
            # Prefetch raw batch on CPU.
            return batch

        staged = self._stage_batch(batch, record_event=True)
        # Move pending event into current/next slot
        if self._current_batch is None:
            self._current_ready_event = self._pending_ready_event
        else:
            self._next_ready_event = self._pending_ready_event
        self._pending_ready_event = None
        return staged

    def _load_next_batch(self) -> ClipBatch:
        if self._should_use_batch_progress():
            return self._load_next_batch_with_progress()
        return self._load_next_batch_raw()

    def _load_next_batch_raw(self) -> ClipBatch:
        if self._iterator is None:
            self._iterator = self._build_iterator()

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = self._build_iterator(reset_epoch=True)
            batch = next(self._iterator)
        return batch

    def _load_next_batch_with_progress(self) -> ClipBatch:
        if self._iterator is None:
            self._iterator = self._build_iterator()

        expected = int(self._effective_batch_size or self._batch_size)
        counter = self._batch_progress_counter
        if counter is None:
            return self._load_next_batch_raw()

        with counter.get_lock():
            counter.value = 0

        pbar = tqdm(
            total=expected,
            desc="Collecting motion batch",
            leave=False,
            dynamic_ncols=True,
        )
        last = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._load_next_batch_raw)
            while not future.done():
                with counter.get_lock():
                    value = counter.value
                if value > last:
                    step = min(value, expected) - last
                    if step > 0:
                        pbar.update(step)
                        last += step
                time.sleep(0.05)
            batch = future.result(timeout=self._result_timeout())

        with counter.get_lock():
            value = counter.value
        if value > last:
            step = min(value, expected) - last
            if step > 0:
                pbar.update(step)
        pbar.close()
        return batch

    def _stage_batch_blocking(self, batch: ClipBatch) -> ClipBatch:
        """Stage a CPU batch to the configured device on the current stream.

        This path is used when `stage_on_swap_only=True` so that only the current
        cache batch resides on GPU.
        """
        if self._stage_device is None:
            return batch
        non_blocking = bool(
            self._pin_memory and self._stage_device.type == "cuda"
        )
        tensors = {
            name: tensor.to(self._stage_device, non_blocking=non_blocking)
            for name, tensor in batch.tensors.items()
        }
        lengths = batch.lengths.to(
            self._stage_device, non_blocking=non_blocking
        )
        window_indices = batch.window_indices.to(
            self._stage_device, non_blocking=non_blocking
        )
        staged = ClipBatch(
            tensors=tensors,
            lengths=lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=getattr(
                batch, "raw_motion_keys", batch.motion_keys
            ),
            window_indices=window_indices,
            max_frame_length=batch.max_frame_length,
        )
        return staged

    def _stage_batch(
        self,
        batch: ClipBatch,
        record_event: bool = False,
    ) -> ClipBatch:
        if self._stage_device is None:
            return batch

        # If CUDA, copy on a dedicated stream and record readiness
        if self._copy_stream is None and (
            self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            self._copy_stream = torch.cuda.Stream(device=self._stage_device)
            logger.info(
                f"Perf/Cache: created CUDA copy stream lazily on {self._stage_device}"
            )

        if self._copy_stream is not None:
            # estimate payload size for logging
            try:
                total_bytes = 0
                for tensor in batch.tensors.values():
                    total_bytes += int(tensor.element_size() * tensor.numel())
                total_bytes += int(
                    batch.lengths.element_size() * batch.lengths.numel()
                )
                total_bytes += int(
                    batch.window_indices.element_size()
                    * batch.window_indices.numel()
                )
            except Exception:
                total_bytes = -1
            with torch.cuda.stream(self._copy_stream):
                tensors = {
                    name: tensor.to(self._stage_device, non_blocking=True)
                    for name, tensor in batch.tensors.items()
                }
                lengths = batch.lengths.to(
                    self._stage_device, non_blocking=True
                )
                window_indices = batch.window_indices.to(
                    self._stage_device, non_blocking=True
                )
            if record_event:
                ev = torch.cuda.Event()
                ev.record(self._copy_stream)
                self._pending_ready_event = ev

        else:
            tensors = {
                name: tensor.to(self._stage_device, non_blocking=True)
                for name, tensor in batch.tensors.items()
            }
            lengths = batch.lengths.to(self._stage_device, non_blocking=True)
            window_indices = batch.window_indices.to(
                self._stage_device, non_blocking=True
            )

        return ClipBatch(
            tensors=tensors,
            lengths=lengths,
            motion_keys=batch.motion_keys,
            raw_motion_keys=getattr(
                batch, "raw_motion_keys", batch.motion_keys
            ),
            window_indices=window_indices,
            max_frame_length=batch.max_frame_length,
        )

    def _build_iterator(
        self, *, reset_epoch: bool = False
    ) -> Iterator[ClipBatch]:
        if self._dataloader is None:
            raise RuntimeError("DataLoader is not initialised")

        if isinstance(self._sampler, DistributedSampler) and reset_epoch:
            self._sampler.set_epoch(self._swap_index + 1)

        return iter(self._dataloader)

    def _build_dataloader(self) -> None:
        dataset = self._datasets[self._mode]
        dataset.set_progress_counter(self._batch_progress_counter)

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
            self._cache_curriculum_sampler = None
        else:
            if self._cache_curriculum_enabled:
                seed = self._seed + self._sampler_rank * 100003
                cfg = dict(self._cache_curriculum_cfg)
                self._cache_curriculum_sampler = PrioritizedInfiniteSampler(
                    dataset_len=ds_len,
                    batch_size=effective_batch_size,
                    seed=seed,
                    p_a_ratio=float(cfg.get("p_a_ratio", 0.2)),
                    ema_alpha_signal=float(cfg.get("ema_alpha_signal", 0.2)),
                    ema_alpha_rel_improve=float(
                        cfg.get("ema_alpha_rel_improve", 0.2)
                    ),
                    relative_eps=float(cfg.get("relative_eps", 1.0e-6)),
                )
                self._cache_curriculum_last_dump_swap = -1
                self._sampler = self._cache_curriculum_sampler
            elif (
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
                self._cache_curriculum_sampler = None
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
                self._cache_curriculum_sampler = None

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

        worker_init_fn = None
        if (
            self._num_workers > 0
            and self._stage_device is not None
            and self._stage_device.type == "cuda"
        ):
            worker_init_fn = _cpu_only_dataloader_worker_init_fn

        self._dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            sampler=self._sampler,
            shuffle=(self._sampler is None and self._mode != "val"),
            num_workers=self._num_workers,
            prefetch_factor=pf,
            pin_memory=self._pin_memory,
            timeout=self._loader_timeout_seconds(),
            persistent_workers=pw,
            collate_fn=collate,
            drop_last=False,
            multiprocessing_context=mp_ctx,
            worker_init_fn=worker_init_fn,
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
        datasets = self.__dict__.get("_datasets")
        if datasets is None:
            return
        self._iterator = None
        self._current_batch = None
        self._next_batch = None
        self._dataloader = None
        self._copy_stream = None
        self._pending_ready_event = None
        self._current_ready_event = None
        self._next_ready_event = None

        for ds in datasets.values():
            if ds is not None:
                ds.close()

    def __del__(self) -> None:
        self.close()

    def _loader_timeout_seconds(self) -> float:
        if not self.force_timeout_on_swap:
            return 0.0
        return self._loader_timeout

    def _result_timeout(self) -> Optional[float]:
        timeout_s = self._loader_timeout_seconds()
        if timeout_s <= 0.0:
            return None
        return timeout_s + 1.0

    def _should_use_batch_progress(self) -> bool:
        if not self._batch_progress_bar:
            return False
        if self._sampler_world_size > 1:
            return False
        if self._loader_timeout_seconds() > 0.0:
            return False
        return True
