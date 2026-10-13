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

"""PPO-barrier synchronization for opt-in source-adaptive window errors.

Cache-row statistics are gathered once in a rank-major, fixed-size int64
packet. The rank-local cache rows may point to different dataset windows;
the controller merges their dataset indices deterministically after gather.
"""

from __future__ import annotations
import math

import torch

_HEADER = (
    "schema",
    "enabled",
    "swap_pending",
    "training",
    "clip_count",
    "capacity",
    "round",
    "version",
    "swap_index",
    "invalid_count",
    "malformed",
)
_SCHEMA = 1
_IDENTITY_BYTES = 32
_DATA_OFFSET = len(_HEADER) + _IDENTITY_BYTES


class WindowErrorAccumulator:
    """Accumulate device-local MPKPE by current cache row."""

    def __init__(
        self, window_indices: torch.Tensor, device: torch.device | str
    ):
        if window_indices.ndim != 1:
            raise ValueError("Cache dataset-window indices must be 1D")
        self.window_indices = window_indices.to(
            device=device, dtype=torch.long
        ).clone()
        self.sums = torch.zeros(
            window_indices.numel(), device=device, dtype=torch.float64
        )
        self.counts = torch.zeros_like(self.sums)
        self.invalid = torch.zeros(1, device=device, dtype=torch.float64)

    @torch.no_grad()
    def record(self, clip_indices: torch.Tensor, errors: torch.Tensor) -> None:
        if (
            clip_indices.ndim != 1
            or errors.shape != clip_indices.shape
            or clip_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("MPKPE must align with integer cache-row indices")
        rows = clip_indices.detach().to(
            device=self.sums.device, dtype=torch.long
        )
        values = errors.detach().to(
            device=self.sums.device, dtype=torch.float64
        )
        valid_rows = (rows >= 0) & (rows < self.sums.numel())
        valid_values = torch.isfinite(values) & (values >= 0)
        valid = valid_rows & valid_values
        self.invalid += (~valid).sum()
        if bool(valid.any()):
            self.sums.scatter_add_(0, rows[valid], values[valid])
            self.counts.scatter_add_(
                0, rows[valid], torch.ones_like(values[valid])
            )


def record_command_mpkpe(command, values: torch.Tensor) -> None:
    """Record training tracking error before command index advancement.

    Terminal environments already reset by Isaac Lab are excluded. This is
    a tracking-error proxy, not a complete-episode failure measurement.
    """
    cache = getattr(command, "_motion_cache", None)
    if (
        cache is None
        or not getattr(cache, "adaptive_sampling_enabled", False)
        or cache.mode != "train"
        or getattr(command, "_is_evaluating", False)
    ):
        return
    if values.ndim != 1 or values.numel() != command.num_envs:
        raise ValueError("MPKPE must contain one value per environment")
    env_ids = command._filter_env_ids_for_motion_task(
        torch.arange(command.num_envs, device=command.device)
    )
    lengths = getattr(command._env, "episode_length_buf", None)
    if lengths is not None:
        env_ids = env_ids[lengths[env_ids] > 0]
    batch = cache.current_batch
    marker = (id(batch), cache.swap_index)
    if getattr(command, "_adaptive_metrics_batch", None) != marker:
        command._adaptive_window_metrics = WindowErrorAccumulator(
            batch.window_indices, command.device
        )
        command._adaptive_metrics_batch = marker
    command._adaptive_window_metrics.record(
        command._clip_indices[env_ids], values[env_ids]
    )


def pack_sync_payload(
    *,
    identity: str | None,
    enabled: bool,
    swap_pending: bool,
    training: bool,
    clip_count: int,
    capacity: int,
    round: int,
    version: int,
    swap_index: int,
    window_indices: torch.Tensor,
    sums: torch.Tensor,
    counts: torch.Tensor,
    invalid_count: float = 0.0,
    device: torch.device | str,
) -> torch.Tensor:
    """Create a fixed-size, lossless int64 packet for one collective.

    Float64 sums and counts are bit-cast rather than numerically converted.
    Malformed local state is encoded so every rank can fail after the gather.
    """
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(
            "Adaptive synchronization needs positive cache capacity"
        )
    packet = torch.zeros(
        _DATA_OFFSET + 3 * capacity, dtype=torch.int64, device=device
    )
    packet[0] = _SCHEMA
    packet[1] = int(enabled)
    packet[2] = int(swap_pending)
    packet[3] = int(training)
    packet[4] = int(clip_count)
    packet[5] = capacity
    packet[6] = int(round)
    packet[7] = int(version)
    packet[8] = int(swap_index)

    malformed = False
    if not math.isfinite(float(invalid_count)) or invalid_count < 0:
        malformed = True
    else:
        packet[9] = int(invalid_count)
    try:
        digest = bytes.fromhex(identity) if identity is not None else b""
    except (TypeError, ValueError):
        digest = b""
    if len(digest) != _IDENTITY_BYTES:
        malformed = True
    else:
        packet[len(_HEADER) : _DATA_OFFSET] = torch.tensor(
            list(digest), dtype=torch.int64, device=device
        )

    if (
        window_indices.ndim != 1
        or sums.ndim != 1
        or counts.ndim != 1
        or window_indices.numel() != clip_count
        or sums.numel() != clip_count
        or counts.numel() != clip_count
        or clip_count < 0
        or clip_count > capacity
    ):
        malformed = True
    count = min(capacity, window_indices.numel(), sums.numel(), counts.numel())
    if count:
        packet[_DATA_OFFSET : _DATA_OFFSET + count] = (
            window_indices.reshape(-1)[:count]
            .detach()
            .to(device=device, dtype=torch.int64)
        )
        packet[_DATA_OFFSET + capacity : _DATA_OFFSET + capacity + count] = (
            sums.reshape(-1)[:count]
            .detach()
            .to(device=device, dtype=torch.float64)
            .contiguous()
            .view(torch.int64)
        )
        packet[
            _DATA_OFFSET + 2 * capacity : _DATA_OFFSET + 2 * capacity + count
        ] = (
            counts.reshape(-1)[:count]
            .detach()
            .to(device=device, dtype=torch.float64)
            .contiguous()
            .view(torch.int64)
        )
    packet[10] = int(malformed)
    return packet


def reduce_sync_payload(
    gathered: torch.Tensor, *, world: int, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Validate rank identity/state and return rank-major observations."""
    stride = _DATA_OFFSET + 3 * capacity
    if gathered.dtype != torch.int64 or gathered.numel() != world * stride:
        raise RuntimeError("Adaptive gather payload shape is invalid")
    packets = gathered.reshape(world, stride).detach().cpu()
    for index, name in enumerate(_HEADER[:9]):
        if not bool((packets[:, index] == packets[0, index]).all()):
            raise RuntimeError(f"Adaptive sampler ranks disagree on {name}")
    if int(packets[0, 0]) != _SCHEMA:
        raise RuntimeError("Unknown adaptive synchronization packet")
    if not bool(
        (
            packets[:, len(_HEADER) : _DATA_OFFSET]
            == packets[0, len(_HEADER) : _DATA_OFFSET]
        ).all()
    ):
        raise RuntimeError(
            "Adaptive sampler dataset/config identity differs across ranks"
        )
    if int(packets[0, 5]) != capacity:
        raise RuntimeError("Adaptive cache capacities do not match")
    if int(packets[0, 1]) != 1:
        raise RuntimeError(
            "Source-adaptive sampling is not enabled on all ranks"
        )
    if bool((packets[:, 10] != 0).any()):
        raise RuntimeError("Malformed source-adaptive cache-row statistics")
    if bool((packets[:, 9] != 0).any()):
        raise FloatingPointError(
            "Invalid MPKPE or cache-row index; refusing adaptive update"
        )
    if not int(packets[0, 2]) or not int(packets[0, 3]):
        return None
    clip_count = int(packets[0, 4])
    if not 0 <= clip_count <= capacity:
        raise RuntimeError("Adaptive cache row count exceeds fixed capacity")
    indices = (
        packets[:, _DATA_OFFSET : _DATA_OFFSET + clip_count]
        .reshape(-1)
        .clone()
    )
    sums = (
        packets[
            :, _DATA_OFFSET + capacity : _DATA_OFFSET + capacity + clip_count
        ]
        .contiguous()
        .view(torch.float64)
        .reshape(-1)
        .clone()
    )
    counts = (
        packets[
            :,
            _DATA_OFFSET + 2 * capacity : _DATA_OFFSET
            + 2 * capacity
            + clip_count,
        ]
        .contiguous()
        .view(torch.float64)
        .reshape(-1)
        .clone()
    )
    if not bool(torch.isfinite(sums).all() and torch.isfinite(counts).all()):
        raise FloatingPointError("Non-finite gathered MPKPE statistics")
    if bool(
        (sums < 0).any()
        or (counts < 0).any()
        or ((counts == 0) & (sums != 0)).any()
    ):
        raise RuntimeError("Inconsistent gathered MPKPE sums/counts")
    return indices, sums, counts


def synchronize_command_sampler(command, accelerator=None) -> None:
    """Synchronize one PPO iteration, before a possible cache swap.

    Every source-adaptive rank calls the same one gather, even with no swap or
    zero observations. Its fixed capacity lets mismatched actual clip counts
    fail on every rank after the collective rather than hang in it.
    """
    cache = command._motion_cache
    controller = getattr(cache, "adaptive_controller", None)
    enabled = bool(getattr(cache, "adaptive_sampling_enabled", False))
    batch = cache.current_batch
    marker = (id(batch), cache.swap_index)
    stats = getattr(command, "_adaptive_window_metrics", None)
    if (
        stats is None
        or getattr(command, "_adaptive_metrics_batch", None) != marker
    ):
        stats = WindowErrorAccumulator(batch.window_indices, command.device)
    packet = pack_sync_payload(
        identity=getattr(controller, "identity", None),
        enabled=enabled,
        swap_pending=bool(command._swap_pending),
        training=cache.mode == "train" and not command._is_evaluating,
        clip_count=int(cache.clip_count),
        capacity=int(cache._batch_size),
        round=getattr(controller, "round", -1),
        version=getattr(controller, "version", -1),
        swap_index=int(cache.swap_index),
        window_indices=stats.window_indices,
        sums=stats.sums,
        counts=stats.counts,
        invalid_count=float(stats.invalid.item()),
        device=command.device,
    )
    world = int(accelerator.num_processes) if accelerator is not None else 1
    gathered = accelerator.gather(packet) if world > 1 else packet
    observations = reduce_sync_payload(
        gathered, world=world, capacity=int(cache._batch_size)
    )
    if observations is None:
        return
    if controller is None:
        raise RuntimeError("Source-adaptive controller is missing")
    controller.observe(*observations)
    command._adaptive_window_metrics = None
    command._adaptive_metrics_batch = None
