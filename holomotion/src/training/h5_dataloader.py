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
from dataclasses import dataclass
import time
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from loguru import logger

Tensor = torch.Tensor


MANDATORY_DATASETS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vels",
    "rg_pos": "global_translation",
    "rb_rot": "global_rotation_quat",
    "body_vel": "global_velocity",
    "body_ang_vel": "global_angular_velocity",
}


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


class Hdf5MotionDataset(Dataset[MotionClipSample]):
    """Dataset that materializes fixed-length motion windows from HDF5 shards."""

    def __init__(
        self,
        manifest_path: str,
        max_frame_length: int,
        min_window_length: int = 1,
        handpicked_motion_names: Optional[List[str]] = None,
        excluded_motion_names: Optional[List[str]] = None,
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

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        root = os.path.dirname(manifest_path)
        self.hdf5_root = root
        self.shards = list(manifest.get("hdf5_shards", []))
        self.clips = manifest.get("clips", {})

        if len(self.shards) == 0:
            raise ValueError(
                f"No HDF5 shards listed in manifest: {manifest_path}"
            )

        self.windows: List[MotionWindow] = self._enumerate_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No motion windows satisfy the requested frame length constraints"
            )

        self._file_handles: Dict[int, h5py.File] = {}

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
                    unique_key = f"{motion_key}__start_{win_start}_len_{window_length}"
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

        for logical_name, dataset_name in MANDATORY_DATASETS.items():
            if dataset_name not in shard_handle:
                raise KeyError(
                    f"Dataset '{dataset_name}' missing in shard index {window.shard_index}"
                )
            np_array = shard_handle[dataset_name][start:end]
            arrays[logical_name] = torch.from_numpy(np_array).to(torch.float32)

        if "frame_flag" in shard_handle:
            frame_flag_np = shard_handle["frame_flag"][start:end]
            frame_flag = torch.from_numpy(frame_flag_np).to(torch.long)
        else:
            frame_flag = torch.ones(window.length, dtype=torch.long)
            if window.length > 0:
                frame_flag[0] = 0
                frame_flag[-1] = 2
        arrays["frame_flag"] = frame_flag

        arrays["root_pos"] = arrays["rg_pos"][:, 0, :]
        arrays["root_rot"] = arrays["rb_rot"][:, 0, :]
        arrays["root_vel"] = arrays["body_vel"][:, 0, :]
        arrays["root_ang_vel"] = arrays["body_ang_vel"][:, 0, :]

        return MotionClipSample(
            motion_key=window.motion_key,
            raw_motion_key=window.raw_motion_key,
            tensors=arrays,
            length=window.length,
        )

    def _get_shard_handle(self, shard_index: int) -> h5py.File:
        if (
            shard_index in self._file_handles
            and self._file_handles[shard_index].id
        ):
            return self._file_handles[shard_index]

        shard_rel = self.shards[shard_index]["file"]
        shard_path = os.path.join(self.hdf5_root, shard_rel)
        # Open with SWMR and a larger raw chunk cache to speed up repeated reads
        handle = h5py.File(
            shard_path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=256 * 1024 * 1024,
            rdcc_w0=0.75,
        )
        self._file_handles[shard_index] = handle
        return handle


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

        clip_indices = torch.randint(
            low=0,
            high=lengths.shape[0],
            size=(num_envs,),
            device=device,
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

        staged = self._stage_batch(batch, record_event=True)
        # Move pending event into current/next slot
        if self._current_batch is None:
            self._current_ready_event = self._pending_ready_event
        else:
            self._next_ready_event = self._pending_ready_event
        self._pending_ready_event = None
        return staged

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

        class InfiniteDistributedSampler(DistributedSampler):
            def __iter__(self):
                # Infinite stream by cycling epochs
                while True:
                    self.set_epoch(getattr(self, "_epoch", 0))
                    for idx in super().__iter__():
                        yield idx
                    self._epoch = getattr(self, "_epoch", 0) + 1

        class InfiniteRandomSampler(Sampler[int]):
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

        if self._sampler_world_size > 1:
            if self._mode == "val":
                self._sampler = DistributedSampler(
                    dataset,
                    num_replicas=self._sampler_world_size,
                    rank=self._sampler_rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                # Infinite sampler for training: no epoch boundaries
                self._sampler = InfiniteDistributedSampler(
                    dataset,
                    num_replicas=self._sampler_world_size,
                    rank=self._sampler_rank,
                    shuffle=True,
                    drop_last=False,
                )
        else:
            if self._mode == "val":
                self._sampler = None
            else:
                # Infinite sampler for single-process training
                self._sampler = InfiniteRandomSampler(dataset)

        # Clamp batch size to dataset length to avoid empty iterator when drop_last is disabled
        effective_batch_size = self._batch_size
        ds_len = len(dataset)
        if isinstance(ds_len, int) and ds_len > 0:
            effective_batch_size = max(1, min(self._batch_size, ds_len))

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
        def _collate(samples):
            if (
                self._mode == "val"
                and self._batch_size > len(samples)
                and len(samples) > 0
            ):
                extra = self._batch_size - len(samples)
                gen = torch.Generator()
                idx = torch.randint(
                    0, len(samples), size=(extra,), generator=gen
                )
                padded = list(samples)
                for i in idx.tolist():
                    padded.append(samples[i])
                return ClipBatch.collate_fn(padded)
            return ClipBatch.collate_fn(samples)

        self._dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            sampler=self._sampler,
            shuffle=(self._sampler is None and self._mode != "val"),
            num_workers=self._num_workers,
            prefetch_factor=pf,
            pin_memory=self._pin_memory,
            persistent_workers=pw,
            collate_fn=_collate,
            drop_last=False,
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
