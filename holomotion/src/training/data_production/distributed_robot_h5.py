"""Distributed HoloSMPL-H5 to HoloRetarget robot-H5 production.

This module is intended to run under ``accelerate launch`` on Orchard.  Work is
split at clip granularity because HoloRetarget uses sequence state: previous
joint positions, quaternion continuity, and sliding ground calibration.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class H5RootInfo:
    root_index: int
    input_root: Path
    output_root: Path
    relative_key: str
    display_name: str
    source_manifest: dict[str, Any]


@dataclass(frozen=True)
class ClipTask:
    root_index: int
    global_index: int
    input_root: Path
    output_root: Path
    shard_rel: str
    clip_index: int
    start: int
    length: int
    motion_key: str
    metadata: dict[str, Any]


class ProgressReporter:
    """Best-effort TCP progress reporter with rank0 in-memory aggregation."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        rank_totals: dict[int, int],
        host: str,
        port: int,
        log_interval_sec: float,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.rank_totals = dict(rank_totals)
        self.host = str(host)
        self.port = int(port)
        self.log_interval_sec = max(1.0, float(log_interval_sec))
        self._server_socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._payloads: dict[int, dict[str, Any]] = {}
        self._last_print_time = 0.0
        if self.rank == 0 and self.world_size > 1:
            self._start_server()

    def close(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)

    def publish(self, payload: dict[str, Any], *, force: bool = False) -> None:
        payload = dict(payload)
        payload["sent_monotonic"] = time.monotonic()
        if self.rank == 0:
            self._accept_payload(payload, force=force)
            return
        if self.world_size <= 1:
            return
        self._send_payload(payload)

    def _start_server(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.listen(max(16, self.world_size))
        sock.settimeout(0.5)
        self._server_socket = sock
        self._server_thread = threading.Thread(
            target=self._server_loop,
            name="holoretarget-progress-server",
            daemon=True,
        )
        self._server_thread.start()

    def _server_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                conn, _addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    data = b""
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    if not data:
                        continue
                    payload = json.loads(data.decode("utf-8"))
                    self._accept_payload(payload)
                except Exception:
                    continue

    def _send_payload(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for _attempt in range(2):
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=1.0
                ) as sock:
                    sock.sendall(data)
                return
            except OSError:
                time.sleep(0.1)

    def _accept_payload(
        self, payload: dict[str, Any], *, force: bool = False
    ) -> None:
        try:
            rank = int(payload["rank"])
        except Exception:
            return
        with self._lock:
            self._payloads[rank] = dict(payload)
            now = time.monotonic()
            should_print = (
                force
                or self._last_print_time <= 0.0
                or now - self._last_print_time >= self.log_interval_sec
            )
            if not should_print:
                return
            self._last_print_time = now
            text = _format_progress_from_payloads(
                self._payloads,
                world_size=self.world_size,
                rank_totals=self.rank_totals,
            )
        print(f"[holoretarget-h5] progress | {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    source_config = _read_optional_config(
        args.source_config or args.source_config_json
    )
    input_root_value = (
        args.input_root
        or source_config.get("holosmpl_run_root")
        or source_config.get("holosmpl_full_runs_root")
    )
    if not input_root_value:
        parser.error(
            "provide --input-root or set holosmpl_run_root in source config"
        )
    output_root_value = (
        args.output_root
        or source_config.get("holoretarget_output_root")
    )
    if not output_root_value:
        parser.error(
            "provide --output-root or set holoretarget_output_root in "
            "source config"
        )
    input_root = Path(input_root_value)
    output_root = Path(output_root_value)

    rank, world_size, local_rank = _distributed_env()
    _configure_cuda_device(local_rank=local_rank, rank=rank)
    skip_names = [name.lower() for name in args.skip_name]

    roots = discover_holosmpl_h5_roots(
        input_root=input_root,
        output_root=output_root,
        source_config=source_config,
        skip_names=skip_names,
    )
    root_infos, tasks = build_clip_tasks(roots)
    rank_assignment = assign_tasks_by_frames(tasks, world_size)
    rank_frame_totals = {
        int(load["rank"]): int(load["frames"])
        for load in _rank_loads(tasks, rank_assignment)
    }

    if args.dry_run:
        _print_dry_run(
            root_infos,
            tasks,
            rank=rank,
            world_size=world_size,
            rank_assignment=rank_assignment,
        )
        return 0

    if rank == 0:
        _print_run_plan(
            root_infos,
            tasks,
            world_size=world_size,
            rank_assignment=rank_assignment,
        )

    run_id = _sanitize_run_id(args.run_id)
    coord_root = output_root / "_distributed" / run_id
    if rank == 0:
        _prepare_output_root(
            output_root=output_root,
            coord_root=coord_root,
            root_infos=root_infos,
            tasks=tasks,
            rank_assignment=rank_assignment,
            overwrite=args.overwrite,
        )
    _wait_for_init(coord_root=coord_root, timeout_sec=args.init_timeout_sec)

    assigned = [
        task for task in tasks if rank_assignment[_task_id(task)] == rank
    ]
    progress_reporter = _create_progress_reporter(
        rank=rank,
        world_size=world_size,
        rank_totals=rank_frame_totals,
        log_interval_sec=args.progress_log_interval_sec,
    )
    try:
        result = process_assigned_tasks(
            assigned,
            root_infos=root_infos,
            rank=rank,
            world_size=world_size,
            coord_root=coord_root,
            compression=args.compression,
            chunks_t=args.chunks_t,
            shard_target_frames=args.shard_target_frames,
            progress_interval=args.progress_interval,
            progress_log_interval_sec=args.progress_log_interval_sec,
            progress_reporter=progress_reporter,
        )
        _write_json(coord_root / f"rank_{rank:05d}.done.json", result)

        if rank == 0:
            _wait_for_rank_results(
                coord_root=coord_root,
                world_size=world_size,
                timeout_sec=args.finalize_timeout_sec,
                progress_log_interval_sec=args.progress_log_interval_sec,
            )
            finalize_manifests(
                output_root=output_root,
                coord_root=coord_root,
                root_infos=root_infos,
                tasks=tasks,
                compression=args.compression,
                chunks_t=args.chunks_t,
            )
    finally:
        if progress_reporter is not None:
            progress_reporter.close()
    return 0


def discover_holosmpl_h5_roots(
    *,
    input_root: Path,
    output_root: Path,
    source_config: dict[str, Any],
    skip_names: list[str],
) -> list[H5RootInfo]:
    """Find HoloSMPL formal-H5 roots and map them to robot-H5 outputs."""

    candidates: list[Path] = []
    labels_by_path: dict[Path, str] = {}
    entries = (
        source_config.get("sources") or source_config.get("datasets") or []
    )
    for entry in entries:
        if entry.get("enabled", True) is False:
            continue
        label = _source_label(entry)
        explicit_roots = entry.get("holosmpl_h5_roots", [])
        if entry.get("holosmpl_h5_root"):
            explicit_roots = [*explicit_roots, entry["holosmpl_h5_root"]]
        for root_value in explicit_roots:
            root_path = Path(str(root_value))
            if not root_path.is_absolute():
                root_path = input_root / root_path
            candidates.append(root_path)
            labels_by_path[root_path.resolve()] = label
        for pattern in entry.get("holosmpl_h5_globs", []):
            for root_path in input_root.glob(str(pattern)):
                candidates.append(root_path)
                labels_by_path[root_path.resolve()] = label

    if not candidates:
        candidates = [
            path.parent for path in input_root.rglob("manifest.json")
        ]

    unique_roots = _filter_shadowed_job_shards(
        sorted({path.resolve() for path in candidates})
    )
    roots: list[H5RootInfo] = []
    for candidate in unique_roots:
        if any(name in str(candidate).lower() for name in skip_names):
            continue
        manifest = _read_json(candidate / "manifest.json")
        if not _is_holosmpl_h5_manifest(manifest):
            continue
        relative = _safe_relative(candidate, input_root)
        roots.append(
            H5RootInfo(
                root_index=len(roots),
                input_root=candidate,
                output_root=_output_root_for(
                    input_root=input_root,
                    h5_root=candidate,
                    output_root=output_root,
                ),
                relative_key=relative.as_posix(),
                display_name=labels_by_path.get(
                    candidate, _manifest_label(manifest, candidate)
                ),
                source_manifest=manifest,
            )
        )
    if not roots:
        raise FileNotFoundError(
            f"no HoloSMPL H5 roots found under {input_root}"
        )
    return roots


def build_clip_tasks(
    roots: list[H5RootInfo],
) -> tuple[list[H5RootInfo], list[ClipTask]]:
    """Read clip indexes without loading frame-major pose arrays."""

    import h5py

    all_tasks: list[ClipTask] = []
    for root in roots:
        local_index = 0
        for shard in root.source_manifest.get("shards", []):
            rel = shard.get("path") or shard.get("file")
            if not rel:
                raise ValueError(
                    f"HoloSMPL shard entry missing path/file: {shard}"
                )
            shard_path = root.input_root / str(rel)
            with h5py.File(shard_path, "r") as handle:
                clips_group = handle["clips"]
                starts = np.asarray(clips_group["start"][:], dtype=np.int64)
                lengths = np.asarray(clips_group["length"][:], dtype=np.int64)
                motion_keys = [
                    _decode_string(x) for x in clips_group["motion_key_id"][:]
                ]
                metadata_json = [
                    _decode_string(x) for x in clips_group["metadata_json"][:]
                ]
                for clip_index, (start, length) in enumerate(
                    zip(starts, lengths, strict=True)
                ):
                    task = ClipTask(
                        root_index=root.root_index,
                        global_index=local_index,
                        input_root=root.input_root,
                        output_root=root.output_root,
                        shard_rel=str(rel),
                        clip_index=int(clip_index),
                        start=int(start),
                        length=int(length),
                        motion_key=str(motion_keys[clip_index]),
                        metadata=json.loads(metadata_json[clip_index]),
                    )
                    all_tasks.append(task)
                    local_index += 1
    return roots, all_tasks


def assign_tasks_by_frames(
    tasks: list[ClipTask],
    world_size: int,
) -> dict[tuple[int, int], int]:
    """Assign whole clips to ranks with deterministic frame balancing."""

    world_size = max(1, int(world_size))
    rank_loads = [0 for _ in range(world_size)]
    assignment: dict[tuple[int, int], int] = {}
    sorted_tasks = sorted(
        tasks,
        key=lambda task: (-task.length, task.root_index, task.global_index),
    )
    for task in sorted_tasks:
        rank = min(range(world_size), key=lambda idx: (rank_loads[idx], idx))
        assignment[_task_id(task)] = rank
        rank_loads[rank] += int(task.length)
    return assignment


def process_assigned_tasks(
    tasks: list[ClipTask],
    *,
    root_infos: list[H5RootInfo],
    rank: int,
    world_size: int,
    coord_root: Path,
    compression: str,
    chunks_t: int,
    shard_target_frames: int,
    progress_interval: int,
    progress_log_interval_sec: float,
    progress_reporter: ProgressReporter | None,
) -> dict[str, Any]:
    """Retarget this rank's clips and write rank-local H5 shards."""

    from holoretarget.config import DEFAULT_CONFIG
    from holoretarget.online import HoloRetargeter
    from holosmpl.converters.smpl_to_body_poses import SmplToBodyPosesConverter
    from holomotion.src.training.data_production.robot_h5 import (
        ROBOT_H5_ARRAY_NAMES,
        RobotH5ShardWriter,
        _finalize_robot_shard,
        _robot_metadata,
        retarget_holosmpl_clip_to_robot_arrays,
    )

    progress_log_interval_sec = max(1.0, float(progress_log_interval_sec))

    by_root: dict[int, list[ClipTask]] = {}
    for task in tasks:
        by_root.setdefault(task.root_index, []).append(task)

    start_time = time.monotonic()
    last_progress_time = start_time
    last_rate_time = start_time
    last_rate_frames = 0
    result_roots: dict[str, Any] = {}
    processed_clips = 0
    processed_frames = 0
    rank_total_clips = len(tasks)
    rank_total_frames = int(sum(task.length for task in tasks))
    for root_position, root in enumerate(root_infos, start=1):
        root_tasks = sorted(
            by_root.get(root.root_index, []), key=lambda x: x.global_index
        )
        if not root_tasks:
            result_roots[root.relative_key] = _empty_rank_root_result(
                root, rank
            )
            continue

        retargeter = HoloRetargeter(DEFAULT_CONFIG)
        body_pose_converter = SmplToBodyPosesConverter(retargeter)
        writer: RobotH5ShardWriter | None = None
        shard_index = 0
        hdf5_shards: list[dict[str, Any]] = []
        clips_manifest: dict[str, dict[str, Any]] = {}
        frames_this_root = 0
        root_total_frames = int(sum(task.length for task in root_tasks))
        for local_count, task in enumerate(root_tasks, start=1):
            clip = load_clip_task(task)
            if writer is None or (
                writer.clip_count > 0
                and writer.frame_count + clip.frame_count > shard_target_frames
            ):
                if writer is not None:
                    hdf5_shards.append(
                        _finalize_robot_shard(writer, root.output_root)
                    )
                    shard_index += 1
                writer = RobotH5ShardWriter(
                    root.output_root
                    / "shards"
                    / f"holoretarget_rank{rank:05d}_{shard_index:06d}.h5",
                    chunks_t=chunks_t,
                    compression=compression,
                )

            arrays = retarget_holosmpl_clip_to_robot_arrays(
                retargeter=retargeter,
                body_pose_converter=body_pose_converter,
                clip=clip,
            )
            metadata = _robot_metadata(clip)
            metadata["distributed_global_clip_index"] = int(task.global_index)
            start, length = writer.append_motion(
                motion_id=task.global_index,
                arrays=arrays,
                metadata_json=json.dumps(
                    metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            clip_key = _unique_clip_key(
                clips_manifest,
                clip.motion_key,
                task.global_index,
            )
            clips_manifest[clip_key] = {
                "motion_key": clip.motion_key,
                "shard": str(Path(writer.path).relative_to(root.output_root)),
                "clip_idx": writer.clip_count - 1,
                "start": int(start),
                "length": int(length),
                "available_arrays": list(ROBOT_H5_ARRAY_NAMES),
                "metadata": metadata,
                "distributed_global_clip_index": int(task.global_index),
            }
            frames_this_root += length
            processed_clips += 1
            processed_frames += length
            now = time.monotonic()
            should_log_by_clip = (
                progress_interval > 0
                and local_count % max(1, progress_interval) == 0
            )
            should_log_by_time = (
                now - last_progress_time >= progress_log_interval_sec
            )
            should_log_final = local_count == len(root_tasks)
            if should_log_by_clip or should_log_by_time or should_log_final:
                last_progress_time = now
                elapsed = max(now - start_time, 1.0e-6)
                rank_fps = processed_frames / elapsed
                recent_elapsed = max(now - last_rate_time, 1.0e-6)
                recent_frames = max(0, processed_frames - last_rate_frames)
                recent_fps = recent_frames / recent_elapsed
                last_rate_time = now
                last_rate_frames = processed_frames
                rank_eta_sec = _eta_seconds(
                    done=processed_frames,
                    total=rank_total_frames,
                    rate=rank_fps,
                )
                progress_payload = _write_rank_progress(
                    coord_root=coord_root,
                    rank=rank,
                    processed_clips=processed_clips,
                    total_clips=rank_total_clips,
                    processed_frames=processed_frames,
                    total_frames=rank_total_frames,
                    fps=rank_fps,
                    recent_fps=recent_fps,
                    eta_sec=rank_eta_sec,
                    dataset_position=root_position,
                    dataset_count=len(root_infos),
                    dataset_name=root.display_name,
                    dataset_key=root.relative_key,
                    dataset_processed_frames=frames_this_root,
                    dataset_total_frames=root_total_frames,
                )
                if progress_reporter is not None:
                    progress_reporter.publish(progress_payload)
                elif rank == 0:
                    print(
                        "[holoretarget-h5] progress | "
                        + _format_job_progress(
                            coord_root=coord_root,
                            world_size=world_size,
                        ),
                        flush=True,
                    )

        if writer is not None:
            hdf5_shards.append(_finalize_robot_shard(writer, root.output_root))
        if progress_reporter is None and rank == 0:
            print(
                f"[holoretarget-h5] progress | {_format_job_progress(coord_root=coord_root, world_size=world_size)}",
                flush=True,
            )
        result_roots[root.relative_key] = {
            "rank": rank,
            "input_root": str(root.input_root),
            "output_root": str(root.output_root),
            "hdf5_shards": hdf5_shards,
            "clips": clips_manifest,
            "processed_clips": len(root_tasks),
            "processed_frames": int(frames_this_root),
            "dof_names": list(retargeter.dof_names),
        }

    elapsed = time.monotonic() - start_time
    return {
        "rank": rank,
        "processed_clips": processed_clips,
        "processed_frames": processed_frames,
        "elapsed_sec": elapsed,
        "frames_per_sec": processed_frames / max(elapsed, 1.0e-6),
        "roots": result_roots,
    }


def load_clip_task(task: ClipTask):
    """Load one HoloSMPL clip from a frame-major H5 shard."""

    import h5py

    from holomotion.src.training.data_production.robot_h5 import (
        HoloSmplClip,
        _normalize_shape_beta,
        _read_clip_shape_beta,
    )

    shard_path = task.input_root / task.shard_rel
    end = task.start + task.length
    with h5py.File(shard_path, "r") as handle:
        pose = np.asarray(
            handle["human_pose_aa"][task.start : end], dtype=np.float32
        )
        trans = np.asarray(
            handle["human_root_trans"][task.start : end], dtype=np.float32
        )
        beta_raw = _read_clip_shape_beta(
            handle,
            clip_index=task.clip_index,
            start=task.start,
            end=end,
            clip_count=int(handle["clips"]["start"].shape[0]),
            total_frames=int(handle["human_pose_aa"].shape[0]),
        )
    return HoloSmplClip(
        shard_path=shard_path,
        clip_index=task.clip_index,
        motion_key=task.motion_key,
        metadata=dict(task.metadata),
        pose_aa=pose,
        root_trans=trans,
        shape_beta=_normalize_shape_beta(beta_raw, frame_count=task.length),
    )


def finalize_manifests(
    *,
    output_root: Path,
    coord_root: Path,
    root_infos: list[H5RootInfo],
    tasks: list[ClipTask],
    compression: str,
    chunks_t: int,
) -> None:
    """Merge rank-local result manifests into training-compatible manifests."""

    from holoretarget.config import DEFAULT_CONFIG
    from holomotion.src.training.data_production.robot_h5 import (
        ROBOT_H5_ARRAY_NAMES,
        _config_to_manifest,
        _normalize_compression_name,
    )

    rank_results = [
        _read_json(path)
        for path in sorted(coord_root.glob("rank_*.done.json"))
    ]
    tasks_by_root: dict[int, list[ClipTask]] = {}
    for task in tasks:
        tasks_by_root.setdefault(task.root_index, []).append(task)

    run_roots: list[dict[str, Any]] = []
    for root in root_infos:
        root_key = root.relative_key
        hdf5_shards: list[dict[str, Any]] = []
        clips_manifest: dict[str, dict[str, Any]] = {}
        processed_clips = 0
        processed_frames = 0
        dof_names: list[str] = []
        for rank_result in rank_results:
            root_result = rank_result.get("roots", {}).get(root_key)
            if not root_result:
                continue
            hdf5_shards.extend(root_result.get("hdf5_shards", []))
            clips_manifest.update(root_result.get("clips", {}))
            processed_clips += int(root_result.get("processed_clips", 0))
            processed_frames += int(root_result.get("processed_frames", 0))
            if not dof_names:
                dof_names = list(root_result.get("dof_names", []))

        root_tasks = sorted(
            tasks_by_root.get(root.root_index, []),
            key=lambda x: x.global_index,
        )
        motion_keys = [task.motion_key for task in root_tasks]
        hdf5_shards = sorted(hdf5_shards, key=lambda x: x["file"])
        shard_indices = {
            str(shard["file"]): index
            for index, shard in enumerate(hdf5_shards)
        }
        for clip_key, clip_meta in clips_manifest.items():
            shard_file = str(clip_meta.get("shard", ""))
            if shard_file not in shard_indices:
                raise ValueError(
                    f"clip {clip_key} references unknown shard: {shard_file}"
                )
            clip_meta["shard"] = shard_indices[shard_file]
        output_manifest = {
            "version": 2,
            "root": str(root.output_root),
            "dataset_name": str(root.display_name),
            "hdf5_shards": hdf5_shards,
            "clips": clips_manifest,
            "motion_keys": motion_keys,
            "dof_names": dof_names,
            "array_names": list(ROBOT_H5_ARRAY_NAMES),
            "chunks_t": int(chunks_t),
            "compression": _normalize_compression_name(compression),
            "source_schema": str(root.source_manifest.get("schema_version")),
            "source_holosmpl_h5_root": str(root.input_root),
            "retarget_backend": "HoloRetarget",
            "retarget_config": _config_to_manifest(DEFAULT_CONFIG),
            "distributed": {
                "clip_partition": "greedy_balance_by_clip_frame_count",
                "processed_clips": processed_clips,
                "processed_frames": processed_frames,
            },
        }
        _write_json(root.output_root / "manifest.json", output_manifest)
        _write_json(root.output_root / "nan_npz_paths.json", [])
        run_roots.append(
            {
                "input_root": str(root.input_root),
                "output_root": str(root.output_root),
                "relative_key": root.relative_key,
                "clip_count": processed_clips,
                "frame_count": processed_frames,
                "shard_count": len(hdf5_shards),
            }
        )

    _write_json(
        output_root / "run_manifest.json",
        {
            "version": 2,
            "pipeline": "holosmpl_h5_to_holoretarget_robot_h5",
            "output_root": str(output_root),
            "roots": run_roots,
        },
    )


def _prepare_output_root(
    *,
    output_root: Path,
    coord_root: Path,
    root_infos: list[H5RootInfo],
    tasks: list[ClipTask],
    rank_assignment: dict[tuple[int, int], int],
    overwrite: bool,
) -> None:
    if coord_root.exists():
        shutil.rmtree(coord_root)
    coord_root.mkdir(parents=True, exist_ok=True)

    try:
        existing_roots = [
            root.output_root
            for root in root_infos
            if root.output_root.exists()
        ]
        if existing_roots and not overwrite:
            preview = ", ".join(str(path) for path in existing_roots[:5])
            suffix = "" if len(existing_roots) <= 5 else " ..."
            raise FileExistsError(
                "output dataset roots already exist: "
                f"{preview}{suffix}; pass --overwrite or set OVERWRITE=1"
            )

        if overwrite:
            for root_path in existing_roots:
                shutil.rmtree(root_path)

        output_root.mkdir(parents=True, exist_ok=True)
        for root in root_infos:
            (root.output_root / "shards").mkdir(parents=True, exist_ok=True)
        _write_json(
            coord_root / "task_manifest.json",
            {
                "input_roots": [
                    {
                        "root_index": root.root_index,
                        "input_root": str(root.input_root),
                        "output_root": str(root.output_root),
                        "relative_key": root.relative_key,
                        "clip_count": sum(
                            1
                            for task in tasks
                            if task.root_index == root.root_index
                        ),
                    }
                    for root in root_infos
                ],
                "total_clips": len(tasks),
                "total_frames": int(sum(task.length for task in tasks)),
                "rank_loads": _rank_loads(tasks, rank_assignment),
            },
        )
        (coord_root / "init.done").write_text("ok\n", encoding="utf-8")
    except Exception as exc:
        _write_json(
            coord_root / "init.error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def _wait_for_rank_results(
    *,
    coord_root: Path,
    world_size: int,
    timeout_sec: float,
    progress_log_interval_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    last_log_time = 0.0
    while True:
        done = sorted(coord_root.glob("rank_*.done.json"))
        if len(done) >= world_size:
            return
        now = time.monotonic()
        if now - last_log_time >= max(1.0, float(progress_log_interval_sec)):
            last_log_time = now
            print(
                "[holoretarget-h5] waiting_for_ranks | "
                f"done_ranks={len(done)}/{world_size} | "
                f"{_format_job_progress(coord_root=coord_root, world_size=world_size)}",
                flush=True,
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"waited for {len(done)}/{world_size} rank results "
                f"under {coord_root}"
            )
        time.sleep(5.0)


def _wait_for_file(path: Path, *, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(1.0)


def _wait_for_init(*, coord_root: Path, timeout_sec: float) -> None:
    done_path = coord_root / "init.done"
    error_path = coord_root / "init.error.json"
    deadline = time.monotonic() + timeout_sec
    while True:
        if done_path.exists():
            return
        if error_path.exists():
            payload = _read_json(error_path)
            raise RuntimeError(
                "rank0 output initialization failed: "
                f"{payload.get('type', 'Error')}: {payload.get('message')}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {done_path}")
        time.sleep(1.0)


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, max(1, world_size), max(0, local_rank)


def _create_progress_reporter(
    *,
    rank: int,
    world_size: int,
    rank_totals: dict[int, int],
    log_interval_sec: float,
) -> ProgressReporter | None:
    if world_size <= 1:
        return None
    host = (
        os.environ.get("HOLORETARGET_PROGRESS_HOST")
        or os.environ.get("MASTER_ADDR")
        or os.environ.get("MAIN_PROCESS_IP")
        or os.environ.get("HOST_NODE_ADDR")
        or "127.0.0.1"
    )
    try:
        port = int(
            os.environ.get(
                "HOLORETARGET_PROGRESS_PORT",
                str(int(os.environ.get("MASTER_PORT", "29500")) + 137),
            )
        )
        return ProgressReporter(
            rank=rank,
            world_size=world_size,
            rank_totals=rank_totals,
            host=host,
            port=port,
            log_interval_sec=log_interval_sec,
        )
    except Exception as exc:
        if rank == 0:
            print(
                "[holoretarget-h5] progress TCP reporter disabled: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return None


def _configure_cuda_device(*, local_rank: int, rank: int) -> None:
    cache_root = Path(
        os.environ.get(
            "HOLORETARGET_WARP_CACHE_ROOT", "/tmp/holoretarget_warp_cache"
        )
    )
    warp_cache_path = cache_root / f"rank_{int(rank):05d}"
    warp_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(warp_cache_path)

    restricted_visible_devices = "CUDA_VISIBLE_DEVICES" not in os.environ
    if restricted_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    try:
        import torch

        if torch.cuda.is_available():
            device_count = max(1, torch.cuda.device_count())
            device_index = (
                0 if restricted_visible_devices else local_rank % device_count
            )
            torch.cuda.set_device(device_index)
    except Exception as exc:  # pragma: no cover - best-effort device binding.
        print(
            f"[holoretarget-h5] CUDA device binding skipped: {exc}", flush=True
        )
    if rank == 0:
        print(
            f"[holoretarget-h5] WARP_CACHE_PATH={cache_root}/rank_XXXXX",
            flush=True,
        )


def _sanitize_run_id(run_id: str | None) -> str:
    value = (
        run_id
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("AIDI_JOB_NAME")
        or f"run_{time.strftime('%Y%m%d%H%M%S')}"
    )
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return clean[:120] or f"run_{int(time.time())}"


def _is_holosmpl_h5_manifest(manifest: dict[str, Any]) -> bool:
    arrays = set(manifest.get("available_arrays", []))
    clip_arrays = set(manifest.get("clip_level_arrays", []))
    has_shape_beta = (
        "human_shape_beta" in arrays or "clips/human_shape_beta" in clip_arrays
    )
    return (
        isinstance(manifest.get("shards"), list)
        and "human_pose_aa" in arrays
        and "human_root_trans" in arrays
        and has_shape_beta
    )


def _output_root_for(
    *, input_root: Path, h5_root: Path, output_root: Path
) -> Path:
    relative = _safe_relative(h5_root, input_root)
    if relative.name == "formal_h5":
        relative = relative.parent / "holoretarget_h5"
    else:
        relative = relative / "holoretarget_h5"
    return output_root / relative


def _filter_shadowed_job_shards(roots: list[Path]) -> list[Path]:
    final_bases = {
        path.parent.parent
        for path in roots
        if (
            path.name == "formal_h5"
            and path.parent.name == "final"
            and path.parent.parent.name == "job_shards"
        )
    }
    if not final_bases:
        return roots
    filtered: list[Path] = []
    for path in roots:
        is_intermediate_job_shard = (
            path.name == "formal_h5"
            and path.parent.name.startswith("shard_")
            and path.parent.parent.name == "job_shards"
            and path.parent.parent in final_bases
        )
        if not is_intermediate_job_shard:
            filtered.append(path)
    return filtered


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _empty_rank_root_result(root: H5RootInfo, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "input_root": str(root.input_root),
        "output_root": str(root.output_root),
        "hdf5_shards": [],
        "clips": {},
        "processed_clips": 0,
        "processed_frames": 0,
    }


def _unique_clip_key(
    existing: dict[str, Any],
    motion_key: str,
    global_index: int,
) -> str:
    if motion_key not in existing:
        return motion_key
    return f"{motion_key}__global_{global_index:08d}"


def _task_id(task: ClipTask) -> tuple[int, int]:
    return (int(task.root_index), int(task.global_index))


def _rank_loads(
    tasks: list[ClipTask],
    rank_assignment: dict[tuple[int, int], int],
) -> list[dict[str, int]]:
    loads: dict[int, dict[str, int]] = {}
    for task in tasks:
        rank = int(rank_assignment[_task_id(task)])
        load = loads.setdefault(rank, {"rank": rank, "clips": 0, "frames": 0})
        load["clips"] += 1
        load["frames"] += int(task.length)
    return [loads[rank] for rank in sorted(loads)]


def _write_rank_progress(
    *,
    coord_root: Path,
    rank: int,
    processed_clips: int,
    total_clips: int,
    processed_frames: int,
    total_frames: int,
    fps: float,
    recent_fps: float,
    eta_sec: float | None,
    dataset_position: int,
    dataset_count: int,
    dataset_name: str,
    dataset_key: str,
    dataset_processed_frames: int,
    dataset_total_frames: int,
) -> dict[str, Any]:
    progress_dir = coord_root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": int(rank),
        "processed_clips": int(processed_clips),
        "total_clips": int(total_clips),
        "processed_frames": int(processed_frames),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "recent_fps": float(recent_fps),
        "eta_sec": None if eta_sec is None else float(eta_sec),
        "done_at": _format_done_at(eta_sec),
        "dataset_position": int(dataset_position),
        "dataset_count": int(dataset_count),
        "dataset_name": str(dataset_name),
        "dataset_key": str(dataset_key),
        "dataset_processed_frames": int(dataset_processed_frames),
        "dataset_total_frames": int(dataset_total_frames),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    path = progress_dir / f"rank_{rank:05d}.json"
    tmp_path = progress_dir / f".rank_{rank:05d}.tmp"
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return payload


def _format_progress_from_payloads(
    payloads: dict[int, dict[str, Any]],
    *,
    world_size: int,
    rank_totals: dict[int, int],
) -> str:
    total_frames = int(sum(rank_totals.values()))
    processed_frames = 0
    active_fps = 0.0
    active_ranks = 0
    seen_ranks = 0
    dataset_counts: dict[tuple[int, int, str], int] = {}

    for rank in range(world_size):
        payload = payloads.get(rank)
        if payload is None:
            continue
        seen_ranks += 1
        total = int(payload.get("total_frames", rank_totals.get(rank, 0)))
        processed = min(int(payload.get("processed_frames", 0)), total)
        processed_frames += processed
        dataset_key = (
            int(payload.get("dataset_position", 0)),
            int(payload.get("dataset_count", 0)),
            str(payload.get("dataset_name", "unknown")),
        )
        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + 1
        if total > 0 and processed >= total:
            continue
        fps = payload.get("recent_fps", payload.get("fps"))
        if fps is not None and float(fps) > 0.0 and processed > 0:
            active_ranks += 1
            active_fps += float(fps)

    if dataset_counts:
        (dataset_position, dataset_count, dataset_name), _ = max(
            dataset_counts.items(), key=lambda item: (item[1], item[0][0])
        )
        dataset_text = f"{dataset_position}/{dataset_count} {dataset_name}"
    else:
        dataset_text = "unknown"

    if total_frames > 0 and processed_frames >= total_frames:
        eta_text = "done"
        done_at_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elif seen_ranks < world_size or active_fps <= 0.0:
        eta_text = f"warming_up({seen_ranks}/{world_size} ranks)"
        done_at_text = "unknown"
    else:
        eta_sec = _eta_seconds(
            done=processed_frames,
            total=total_frames,
            rate=active_fps,
        )
        eta_text = _format_duration(eta_sec)
        done_at_text = _format_done_at(eta_sec)

    frame_percent = (
        100.0 * processed_frames / total_frames if total_frames > 0 else 0.0
    )
    return (
        f"{frame_percent:5.1f}% | "
        f"dataset {dataset_text} | "
        f"frames {_format_count(processed_frames)}/{_format_count(total_frames)} | "
        f"fps {active_fps:.1f} | eta {eta_text} | done_at {done_at_text}"
    )


def _format_job_progress(*, coord_root: Path, world_size: int) -> str:
    summary = _job_progress_summary(
        coord_root=coord_root, world_size=world_size
    )
    total_frames = summary["total_frames"]
    processed_frames = summary["processed_frames_est"]
    active_ranks = summary["active_ranks"]
    fps = summary["fps_est"]
    dataset_text = summary["dataset_text"]
    if total_frames > 0 and processed_frames >= total_frames:
        eta_text = "done"
        done_at_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elif summary["eta_sec"] is None:
        eta_text = f"warming_up({active_ranks}/{world_size} active ranks)"
        done_at_text = "unknown"
    else:
        eta_text = _format_duration(summary["eta_sec"])
        done_at_text = _format_done_at(summary["eta_sec"])
    frame_percent = (
        100.0 * processed_frames / total_frames if total_frames > 0 else 0.0
    )
    return (
        f"{frame_percent:5.1f}% | "
        f"dataset {dataset_text} | "
        f"frames {_format_count(processed_frames)}/{_format_count(total_frames)} | "
        f"fps {fps:.1f} | eta {eta_text} | done_at {done_at_text}"
    )


def _job_progress_summary(
    *, coord_root: Path, world_size: int
) -> dict[str, Any]:
    rank_totals = _rank_frame_totals(
        coord_root=coord_root, world_size=world_size
    )
    total_frames = int(sum(rank_totals.values()))
    visible_processed_frames = 0
    active_processed_frames = 0
    active_fps = 0.0
    seen_ranks = 0
    active_ranks = 0
    done_ranks = 0
    done_frames = 0
    dataset_counts: dict[tuple[int, int, str], int] = {}

    for rank in range(world_size):
        payload = _read_rank_done_or_progress(coord_root=coord_root, rank=rank)
        if payload is None:
            continue
        seen_ranks += 1
        dataset_key = (
            int(payload.get("dataset_position", 0)),
            int(payload.get("dataset_count", 0)),
            str(payload.get("dataset_name", "unknown")),
        )
        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + 1
        processed = int(payload.get("processed_frames", 0))
        total = int(payload.get("total_frames", rank_totals.get(rank, 0)))
        clipped_processed = min(processed, total)
        visible_processed_frames += clipped_processed
        if processed >= total and total > 0:
            done_ranks += 1
            done_frames += total
            continue
        fps = payload.get("recent_fps", payload.get("fps"))
        if fps is not None and float(fps) > 0.0 and processed > 0:
            active_ranks += 1
            active_processed_frames += clipped_processed
            active_fps += float(fps)

    if seen_ranks < world_size or active_fps <= 0.0:
        eta_sec: float | None = None
        fps_est = 0.0
        processed_frames_est = visible_processed_frames
    else:
        remaining_ranks = max(0, world_size - done_ranks)
        fps_per_active_rank = active_fps / max(1, active_ranks)
        fps_est = fps_per_active_rank * remaining_ranks
        active_processed_per_rank = active_processed_frames / max(
            1, active_ranks
        )
        processed_frames_est = int(
            min(
                total_frames,
                done_frames + active_processed_per_rank * remaining_ranks,
            )
        )
        eta_sec = _eta_seconds(
            done=processed_frames_est,
            total=total_frames,
            rate=fps_est,
        )
    if dataset_counts:
        (dataset_position, dataset_count, dataset_name), _ = max(
            dataset_counts.items(),
            key=lambda item: (item[1], item[0][0]),
        )
        dataset_text = f"{dataset_position}/{dataset_count} {dataset_name}"
    else:
        dataset_text = "unknown"
    return {
        "processed_frames": int(visible_processed_frames),
        "processed_frames_est": int(processed_frames_est),
        "total_frames": int(total_frames),
        "seen_ranks": int(seen_ranks),
        "active_ranks": int(active_ranks),
        "eta_sec": eta_sec,
        "fps": float(active_fps),
        "fps_est": float(fps_est),
        "dataset_text": dataset_text,
    }


def _rank_frame_totals(*, coord_root: Path, world_size: int) -> dict[int, int]:
    manifest_path = coord_root / "task_manifest.json"
    if not manifest_path.exists():
        return {rank: 0 for rank in range(world_size)}
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return {rank: 0 for rank in range(world_size)}
    totals = {rank: 0 for rank in range(world_size)}
    for load in manifest.get("rank_loads", []):
        totals[int(load["rank"])] = int(load.get("frames", 0))
    return totals


def _read_rank_done_or_progress(
    *,
    coord_root: Path,
    rank: int,
) -> dict[str, Any] | None:
    done_path = coord_root / f"rank_{rank:05d}.done.json"
    if done_path.exists():
        try:
            payload = _read_json(done_path)
            payload["total_frames"] = int(payload.get("processed_frames", 0))
            payload["eta_sec"] = 0.0
            return payload
        except Exception:
            return None

    progress_path = coord_root / "progress" / f"rank_{rank:05d}.json"
    if not progress_path.exists():
        return None
    try:
        return _read_json(progress_path)
    except Exception:
        return None


def _source_label(entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or entry.get("source") or "unknown")
    split = entry.get("split")
    if split:
        return f"{split}/{name}"
    return name


def _manifest_label(manifest: dict[str, Any], root: Path) -> str:
    dataset = manifest.get("dataset") or manifest.get("source_name")
    if dataset:
        return str(dataset)
    return root.parent.name if root.name == "formal_h5" else root.name


def _print_run_plan(
    roots: list[H5RootInfo],
    tasks: list[ClipTask],
    *,
    world_size: int,
    rank_assignment: dict[tuple[int, int], int],
) -> None:
    total_frames = int(sum(task.length for task in tasks))
    total_clips = len(tasks)
    print(
        "[holoretarget-h5] plan | "
        f"datasets={len(roots)} | clips={total_clips} | "
        f"frames={total_frames} | ranks={world_size}",
        flush=True,
    )
    for position, root in enumerate(roots, start=1):
        root_tasks = [
            task for task in tasks if task.root_index == root.root_index
        ]
        print(
            "[holoretarget-h5] dataset "
            f"{position}/{len(roots)} {root.display_name} | "
            f"key={root.relative_key} | clips={len(root_tasks)} | "
            f"frames={int(sum(task.length for task in root_tasks))}",
            flush=True,
        )
    print(
        "[holoretarget-h5] rank_loads "
        + json.dumps(_rank_loads(tasks, rank_assignment), ensure_ascii=False),
        flush=True,
    )


def _eta_seconds(*, done: int, total: int, rate: float) -> float | None:
    remaining = max(0, int(total) - int(done))
    if remaining == 0:
        return 0.0
    if rate <= 1.0e-9:
        return None
    return remaining / rate


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds_i = max(0, int(round(seconds)))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_count(value: int | float) -> str:
    value_f = float(value)
    abs_value = abs(value_f)
    if abs_value >= 1.0e9:
        return f"{value_f / 1.0e9:.1f}B"
    if abs_value >= 1.0e6:
        return f"{value_f / 1.0e6:.1f}M"
    if abs_value >= 1.0e3:
        return f"{value_f / 1.0e3:.1f}K"
    return str(int(value_f))


def _format_done_at(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(time.time() + max(0.0, float(seconds))),
    )


def _print_dry_run(
    roots: list[H5RootInfo],
    tasks: list[ClipTask],
    *,
    rank: int,
    world_size: int,
    rank_assignment: dict[tuple[int, int], int],
) -> None:
    payload = {
        "rank": rank,
        "world_size": world_size,
        "root_count": len(roots),
        "clip_count": len(tasks),
        "frame_count": int(sum(task.length for task in tasks)),
        "clip_partition": "greedy_balance_by_clip_frame_count",
        "rank_loads": _rank_loads(tasks, rank_assignment),
        "roots": [
            {
                "input_root": str(root.input_root),
                "output_root": str(root.output_root),
                "relative_key": root.relative_key,
                "display_name": root.display_name,
                "clip_count": sum(
                    1 for task in tasks if task.root_index == root.root_index
                ),
                "frame_count": int(
                    sum(
                        task.length
                        for task in tasks
                        if task.root_index == root.root_index
                    )
                ),
            }
            for root in roots
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def _read_optional_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config = _read_yaml_config(path, stack=[])
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "YAML interpolation requires OmegaConf."
            ) from exc
        resolved = OmegaConf.to_container(
            OmegaConf.create(config), resolve=True
        )
        if not isinstance(resolved, dict):
            raise ValueError(f"YAML config must resolve to a mapping: {path}")
        return resolved
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"unsupported config file extension: {path}")


def _read_yaml_config(path: Path, *, stack: list[Path]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML source configs require PyYAML. Install pyyaml or use JSON."
        ) from exc

    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in [*stack, path])
        raise ValueError(f"cyclic YAML defaults: {chain}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")

    defaults = data.get("defaults") or []
    body = {key: value for key, value in data.items() if key != "defaults"}
    if not defaults:
        return body

    merged: dict[str, Any] = {}
    inserted_self = False
    for item in defaults:
        if item == "_self_":
            merged = _merge_dicts(merged, body)
            inserted_self = True
            continue
        default_path = _resolve_yaml_default(path, item)
        merged = _merge_dicts(
            merged,
            _read_yaml_config(default_path, stack=[*stack, path]),
        )

    if not inserted_self:
        merged = _merge_dicts(merged, body)
    return merged


def _resolve_yaml_default(current_path: Path, item: Any) -> Path:
    if isinstance(item, str):
        default_name = item
    elif isinstance(item, dict) and len(item) == 1:
        group, name = next(iter(item.items()))
        default_name = f"{group}/{name}"
    else:
        raise ValueError(f"unsupported YAML defaults entry: {item!r}")

    if default_name.endswith((".yaml", ".yml")):
        suffix_path = Path(default_name)
    else:
        suffix_path = Path(default_name + ".yaml")

    if default_name.startswith("/"):
        config_root = _find_config_root(current_path)
        return config_root / str(suffix_path).lstrip("/")
    return current_path.parent / suffix_path


def _find_config_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if parent.name == "config":
            return parent
    return path.parent


def _merge_dicts(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(old_value, value)
        else:
            merged[key] = value
    return merged


def _read_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return _read_json(Path(path))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed HoloSMPL H5 -> HoloRetarget robot H5 production"
        ),
    )
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--source-config-json", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--compression", default="lzf")
    parser.add_argument("--chunks-t", type=int, default=1024)
    parser.add_argument("--shard-target-frames", type=int, default=250_000)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument(
        "--progress-log-interval-sec", type=float, default=60.0
    )
    parser.add_argument("--skip-name", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--init-timeout-sec", type=float, default=600.0)
    parser.add_argument("--finalize-timeout-sec", type=float, default=86400.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
