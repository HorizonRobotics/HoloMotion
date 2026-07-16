"""Distributed raw source -> HoloSMPL formal H5 production.

Each rank writes independent per-source shards under ``_rank_outputs``.  Rank 0
then merges the H5 manifests and shard files into ``<source>/final/formal_h5``.
This keeps clip ownership simple: one clip is produced by exactly one rank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from holosmpl.core.config import load_config
from holosmpl.sources import get_source_config
from holosmpl.workflows.combine_formal_h5 import combine_formal_h5_roots
from holosmpl.workflows.combine_npz_roots import (
    combine_canonical_roots,
    combine_formal_npz_roots,
)
from holosmpl.workflows.convert_dataset import convert_dataset_to_canonical
from holosmpl.workflows.convert_formal import convert_canonical_to_formal_npz
from holosmpl.workflows.direct_formal_h5 import stream_dataset_to_formal_h5
from holosmpl.workflows.pack_formal_h5 import pack_formal_npz_to_h5


@dataclass(frozen=True)
class SourceJob:
    index: int
    split: str
    name: str
    source_key: str
    input_root: Path
    output_key: str
    source_glob: str
    multi_clip_source: bool

    @property
    def display(self) -> str:
        return f"{self.split}/{self.name}"


class GlobalProgressReporter:
    """Periodically aggregate rank-local progress files on rank 0."""

    def __init__(
        self,
        *,
        coord_root: Path,
        world_size: int,
        total_units: int,
        interval_sec: float,
    ) -> None:
        self.coord_root = coord_root
        self.world_size = int(world_size)
        self.total_units = int(total_units)
        self.interval_sec = max(1.0, float(interval_sec))
        self.start_time = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="holosmpl-progress-reporter",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.print_progress()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.print_progress()

    def print_progress(self) -> None:
        payloads = _read_rank_progress(self.coord_root, self.world_size)
        completed_units = sum(int(item.get("completed_units", 0)) for item in payloads)
        processed_frames = sum(int(item.get("processed_frames", 0)) for item in payloads)
        elapsed = max(time.monotonic() - self.start_time, 1.0e-6)
        unit_rate = completed_units / elapsed
        frame_rate = processed_frames / elapsed
        eta_sec = (
            (self.total_units - completed_units) / unit_rate
            if unit_rate > 0.0 and completed_units < self.total_units
            else 0.0
        )
        if unit_rate <= 0.0:
            eta_text = f"warming_up({len(payloads)}/{self.world_size} ranks)"
            done_at_text = "unknown"
        else:
            eta_text = _format_duration(eta_sec)
            done_at_text = _format_done_at(eta_sec)
        latest = max(
            payloads,
            key=lambda item: float(item.get("updated_at_epoch", 0.0)),
            default={},
        )
        dataset = str(latest.get("dataset", "warming_up"))
        percent = 100.0 * completed_units / self.total_units if self.total_units else 100.0
        print(
            "[holosmpl-h5] progress | "
            f"{percent:5.1f}% | dataset {dataset} | "
            f"work {_format_count(completed_units)}/{_format_count(self.total_units)} | "
            f"frames {_format_count(processed_frames)} | fps {frame_rate:.1f} | "
            f"eta {eta_text} | done_at {done_at_text}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _read_optional_config(args.source_config)
    raw_data_root_value = args.raw_data_root or config.get("raw_data_root")
    if not raw_data_root_value:
        parser.error(
            "provide --raw-data-root or set raw_data_root in source config"
        )
    output_root_value = args.output_root or config.get("holosmpl_output_root")
    if not output_root_value:
        parser.error(
            "provide --output-root or set holosmpl_output_root in "
            "source config"
        )
    raw_data_root = Path(raw_data_root_value)
    output_root = Path(output_root_value)

    rank, world_size, _local_rank = _distributed_env()
    jobs = _discover_jobs(
        config=config,
        raw_data_root=raw_data_root,
        only_names=args.only_name,
        skip_names=args.skip_name,
    )
    if not jobs:
        raise RuntimeError("no HoloSMPL source jobs selected")

    run_id = _sanitize_run_id(args.run_id or output_root.name)
    coord_root = output_root / "_distributed" / run_id
    if args.dry_run:
        if rank == 0:
            _print_plan(jobs, world_size=world_size)
        return 0

    if rank == 0:
        _prepare_output(output_root=output_root, coord_root=coord_root, overwrite=args.overwrite)
        _print_plan(jobs, world_size=world_size)
    _wait_for_init(coord_root, timeout_sec=args.init_timeout_sec)

    if args.pipeline_mode == "direct_h5":
        rank_jobs = _build_global_rank_jobs(jobs, rank=rank, world_size=world_size)
        total_work_units = _global_work_unit_count(jobs, world_size=world_size)
    else:
        rank_jobs = [(job, None) for job in jobs]
        total_work_units = sum(_job_work_units(job, paths) for job, paths in rank_jobs) * world_size
    rank_total_units = sum(_job_work_units(job, paths) for job, paths in rank_jobs)
    _write_rank_progress(
        coord_root=coord_root,
        rank=rank,
        completed_units=0,
        total_units=rank_total_units,
        processed_frames=0,
        dataset="warming_up",
    )
    progress_reporter = (
        GlobalProgressReporter(
            coord_root=coord_root,
            world_size=world_size,
            total_units=total_work_units,
            interval_sec=args.progress_log_interval_sec,
        )
        if rank == 0
        else None
    )
    rank_results: list[dict[str, Any]] = []
    start_time = time.monotonic()
    completed_units = 0
    processed_frames = 0
    for job, selected_source_paths in rank_jobs:
        job_units = _job_work_units(job, selected_source_paths)

        def progress_callback(
            payload: dict[str, Any],
            *,
            base_units: int = completed_units,
            base_frames: int = processed_frames,
            current_job: SourceJob = job,
        ) -> None:
            current_units = min(job_units, int(payload.get("completed_units", 0)))
            _write_rank_progress(
                coord_root=coord_root,
                rank=rank,
                completed_units=base_units + current_units,
                total_units=rank_total_units,
                processed_frames=base_frames + int(payload.get("processed_frames", 0)),
                dataset=current_job.display,
            )

        result = _process_source_job(
            job,
            output_root=output_root,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers_per_rank,
            progress_interval=args.progress_interval,
            compression=args.compression,
            shard_target_gb=args.shard_target_gb,
            chunks_t=args.chunks_t,
            pipeline_mode=args.pipeline_mode,
            shard_strategy=args.shard_strategy,
            validate_rank_output=args.validate_rank_output,
            selected_source_paths=selected_source_paths,
            progress_callback=progress_callback,
            progress_log_interval_sec=args.progress_log_interval_sec,
        )
        completed_units += job_units
        processed_frames += int(result.get("formal_h5_summary", {}).get("frame_count", 0))
        _write_rank_progress(
            coord_root=coord_root,
            rank=rank,
            completed_units=completed_units,
            total_units=rank_total_units,
            processed_frames=processed_frames,
            dataset=job.display,
        )
        rank_results.append(result)
        _log_rank_progress(
            rank=rank,
            job=job,
            result=result,
            done=len(rank_results),
            total=len(rank_jobs),
            start_time=start_time,
        )

    rank_done = coord_root / "ranks" / f"rank_{rank:05d}.json"
    _write_json(
        rank_done,
        {
            "rank": rank,
            "world_size": world_size,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
            "results": rank_results,
        },
    )
    if rank != 0:
        return 0

    _wait_for_rank_done(coord_root=coord_root, world_size=world_size, timeout_sec=args.rank_timeout_sec)
    if progress_reporter is not None:
        progress_reporter.close()
    combine_summary = _combine_all_sources(
        jobs,
        output_root=output_root,
        world_size=world_size,
        combine_mode=args.combine_mode,
        combine_npz=args.combine_npz,
        validate_final=args.validate_final,
        progress_interval=args.progress_interval,
    )
    _write_json(
        output_root / "manifest.json",
        {
            "schema_version": "distributed_holosmpl_run_v1",
            "run_id": run_id,
            "raw_data_root": str(raw_data_root),
            "output_root": str(output_root),
            "world_size": world_size,
            "sources": combine_summary,
            "elapsed_sec": round(time.monotonic() - start_time, 3),
        },
    )
    (output_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(f"[holosmpl-h5] done | output_root={output_root}", flush=True)
    return 0


def _process_source_job(
    job: SourceJob,
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    num_workers: int,
    progress_interval: int,
    compression: str | None,
    shard_target_gb: float,
    chunks_t: int,
    pipeline_mode: str,
    shard_strategy: str,
    validate_rank_output: bool,
    selected_source_paths: list[Path] | None,
    progress_callback: Callable[[dict[str, Any]], None],
    progress_log_interval_sec: float,
) -> dict[str, Any]:
    if selected_source_paths is None:
        file_count, selected_count = _source_file_counts(
            job,
            rank=rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )
    else:
        file_count = len(sorted(job.input_root.rglob(job.source_glob)))
        selected_count = len(selected_source_paths)
    if selected_count <= 0:
        return {
            "source": job.display,
            "output_key": job.output_key,
            "status": "skipped_empty_rank",
            "source_file_count": file_count,
            "selected_source_file_count": selected_count,
        }

    rank_root = _rank_source_root(output_root, rank=rank, job=job)
    if rank_root.exists():
        shutil.rmtree(rank_root)
    rank_root.mkdir(parents=True, exist_ok=True)
    report_root = rank_root / "reports"
    run_config_root = rank_root / "run_configs"
    report_root.mkdir(parents=True, exist_ok=True)
    run_config_root.mkdir(parents=True, exist_ok=True)

    if job.multi_clip_source:
        source_shard_index = 0
        source_shard_count = 1
        multi_clip_shard_index = rank
        multi_clip_shard_count = world_size
    else:
        source_shard_index = rank
        source_shard_count = world_size
        multi_clip_shard_index = rank
        multi_clip_shard_count = world_size

    canonical_root = rank_root / "canonical"
    formal_npz_root = rank_root / "formal_npz"
    formal_h5_root = rank_root / "formal_h5"
    if pipeline_mode == "direct_h5":
        formal_h5 = stream_dataset_to_formal_h5(
            dataset=job.source_key,
            input_root=job.input_root,
            output_root=formal_h5_root,
            target_fps=50.0,
            report_json=report_root / "formal_h5_validation.json",
            report_md=report_root / "formal_h5_validation.md",
            run_config_json=run_config_root / "direct_formal_h5_run_config.json",
            compression=compression,
            overwrite=True,
            shard_target_gb=shard_target_gb,
            chunks_t=chunks_t,
            progress_interval=progress_interval,
            shard_index=source_shard_index,
            shard_count=source_shard_count,
            multi_clip_shard_index=multi_clip_shard_index,
            multi_clip_shard_count=multi_clip_shard_count,
            shard_strategy=shard_strategy,
            validate_output=validate_rank_output,
            source_paths=selected_source_paths,
            progress_callback=progress_callback,
            progress_log_interval_sec=progress_log_interval_sec,
        )
        result = {
            "source": job.display,
            "output_key": job.output_key,
            "status": "ok",
            "pipeline_mode": pipeline_mode,
            "source_file_count": file_count,
            "selected_source_file_count": selected_count,
            "rank_root": str(rank_root),
            "formal_h5_root": str(formal_h5_root),
            "formal_h5_summary": formal_h5["summary"],
        }
        _write_json(rank_root / "_SUCCESS.json", result)
        return result
    if pipeline_mode != "staged":
        raise ValueError(f"unsupported pipeline_mode: {pipeline_mode}")

    canonical = convert_dataset_to_canonical(
        dataset=job.source_key,
        input_root=job.input_root,
        output_root=canonical_root,
        target_fps=50.0,
        report_json=report_root / "canonical_validation.json",
        report_md=report_root / "canonical_validation.md",
        run_config_json=run_config_root / "canonical_run_config.json",
        overwrite=True,
        num_workers=num_workers,
        progress_interval=progress_interval,
        shard_index=source_shard_index,
        shard_count=source_shard_count,
        multi_clip_shard_index=multi_clip_shard_index,
        multi_clip_shard_count=multi_clip_shard_count,
    )
    sample_count = int(canonical["summary"].get("converted_count", 0))
    if sample_count <= 0:
        _write_json(
            rank_root / "_EMPTY.json",
            {
                "source": job.display,
                "rank": rank,
                "world_size": world_size,
                "reason": "no clips selected after source conversion",
            },
        )
        return {
            "source": job.display,
            "output_key": job.output_key,
            "status": "empty_after_canonical",
            "source_file_count": file_count,
            "selected_source_file_count": selected_count,
            "sample_count": sample_count,
        }

    formal_npz = convert_canonical_to_formal_npz(
        canonical_root=canonical_root,
        output_root=formal_npz_root,
        report_json=report_root / "formal_npz_validation.json",
        report_md=report_root / "formal_npz_validation.md",
        run_config_json=run_config_root / "formal_npz_run_config.json",
        overwrite=True,
        num_workers=num_workers,
        progress_interval=progress_interval,
    )
    formal_h5 = pack_formal_npz_to_h5(
        formal_npz_root=formal_npz_root,
        output_root=formal_h5_root,
        report_json=report_root / "formal_h5_validation.json",
        report_md=report_root / "formal_h5_validation.md",
        run_config_json=run_config_root / "formal_h5_run_config.json",
        compression=compression,
        overwrite=True,
        shard_target_gb=shard_target_gb,
        chunks_t=chunks_t,
        progress_interval=progress_interval,
    )
    result = {
        "source": job.display,
        "output_key": job.output_key,
        "status": "ok",
        "pipeline_mode": pipeline_mode,
        "source_file_count": file_count,
        "selected_source_file_count": selected_count,
        "rank_root": str(rank_root),
        "canonical_root": str(canonical_root),
        "formal_npz_root": str(formal_npz_root),
        "formal_h5_root": str(formal_h5_root),
        "canonical_summary": canonical["summary"],
        "formal_npz_summary": formal_npz["summary"],
        "formal_h5_summary": formal_h5["summary"],
    }
    _write_json(rank_root / "_SUCCESS.json", result)
    return result


def _combine_all_sources(
    jobs: list[SourceJob],
    *,
    output_root: Path,
    world_size: int,
    combine_mode: str,
    combine_npz: bool,
    validate_final: bool,
    progress_interval: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        rank_roots = [
            _rank_source_root(output_root, rank=rank, job=job)
            for rank in range(world_size)
        ]
        h5_inputs = [
            root / "formal_h5"
            for root in rank_roots
            if (root / "formal_h5" / "manifest.json").is_file()
        ]
        final_root = output_root / job.output_key / "final"
        if not h5_inputs:
            summaries.append(
                {
                    "source": job.display,
                    "output_key": job.output_key,
                    "status": "skipped_no_rank_outputs",
                }
            )
            continue
        final_h5 = combine_formal_h5_roots(
            inputs=h5_inputs,
            output_root=final_root / "formal_h5",
            mode=combine_mode,
            overwrite=True,
            progress_interval=progress_interval,
            validate=validate_final,
        )
        item: dict[str, Any] = {
            "source": job.display,
            "output_key": job.output_key,
            "status": "ok",
            "rank_output_count": len(h5_inputs),
            "formal_h5_root": str(final_root / "formal_h5"),
            "clip_count": int(final_h5["manifest"].get("clip_count", 0)),
            "frame_count": int(final_h5["manifest"].get("frame_count", 0)),
        }
        if combine_npz:
            canonical_inputs = [
                root / "canonical"
                for root in rank_roots
                if (root / "canonical" / "manifest.json").is_file()
            ]
            formal_npz_inputs = [
                root / "formal_npz"
                for root in rank_roots
                if (root / "formal_npz" / "manifest.json").is_file()
            ]
            if canonical_inputs:
                combine_canonical_roots(
                    inputs=canonical_inputs,
                    output_root=final_root / "canonical",
                    mode=combine_mode,
                    overwrite=True,
                    progress_interval=progress_interval,
                )
                item["canonical_root"] = str(final_root / "canonical")
            if formal_npz_inputs:
                combine_formal_npz_roots(
                    inputs=formal_npz_inputs,
                    output_root=final_root / "formal_npz",
                    mode=combine_mode,
                    overwrite=True,
                    progress_interval=progress_interval,
                )
                item["formal_npz_root"] = str(final_root / "formal_npz")
        summaries.append(item)
        print(
            "[holosmpl-h5] combined "
            f"{job.display} | clips={item['clip_count']} "
            f"frames={item['frame_count']} | {item['formal_h5_root']}",
            flush=True,
        )
    return summaries


def _discover_jobs(
    *,
    config: dict[str, Any],
    raw_data_root: Path,
    only_names: list[str],
    skip_names: list[str],
) -> list[SourceJob]:
    only = {name.lower() for name in only_names}
    skip = {name.lower() for name in skip_names}
    sources = config.get("sources") or []
    if not isinstance(sources, list):
        raise TypeError("source config field 'sources' must be a list")
    jobs: list[SourceJob] = []
    for index, item in enumerate(sources):
        if not isinstance(item, dict):
            raise TypeError(f"sources[{index}] must be a mapping")
        if item.get("enabled", True) is False:
            continue
        name = str(item.get("name") or f"source_{index:03d}")
        split = str(item.get("split") or "unspecified")
        labels = {name.lower(), f"{split}/{name}".lower()}
        if only and not labels.intersection(only):
            continue
        if labels.intersection(skip):
            continue
        source_key = str(item.get("holosmpl_source_key") or item.get("source_key") or "")
        if not source_key:
            raise ValueError(f"{name}: missing holosmpl_source_key")
        source_config = get_source_config(source_key)
        if source_config is None:
            raise ValueError(f"{name}: unknown HoloSMPL source key {source_key}")
        raw_path = item.get("raw_path") or item.get("input_root")
        if raw_path:
            input_root = Path(str(raw_path))
        else:
            raw_relative_path = item.get("raw_relative_path")
            if raw_relative_path is None:
                raise ValueError(f"{name}: missing raw_relative_path or raw_path")
            input_root = raw_data_root / str(raw_relative_path)
        output_key = str(item.get("output_key") or _slug(f"{split}_{name}"))
        jobs.append(
            SourceJob(
                index=index,
                split=split,
                name=name,
                source_key=source_key,
                input_root=input_root,
                output_key=output_key,
                source_glob=str(source_config["source_glob"]),
                multi_clip_source=bool(source_config.get("multi_clip_source")),
            )
        )
    return jobs


def _source_file_counts(
    job: SourceJob,
    *,
    rank: int,
    world_size: int,
    shard_strategy: str,
) -> tuple[int, int]:
    paths = sorted(job.input_root.rglob(job.source_glob)) if job.input_root.is_dir() else []
    if job.multi_clip_source:
        return len(paths), len(paths)
    return len(paths), len(_select_source_paths_for_rank(paths, rank=rank, world_size=world_size, strategy=shard_strategy))


def _build_global_rank_jobs(
    jobs: list[SourceJob],
    *,
    rank: int,
    world_size: int,
) -> list[tuple[SourceJob, list[Path] | None]]:
    """Assign independent source files globally instead of dataset by dataset.

    File-size greedy assignment is deterministic on every rank, so it needs no
    extra coordinator or cross-rank work queue. Multi-clip sources remain a
    shared job because their converter already shards sequences by rank.
    """
    bins: list[list[tuple[int, int, SourceJob, Path]]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    shared_jobs: list[SourceJob] = []
    indexed_tasks: list[tuple[int, int, SourceJob, Path]] = []
    order = 0
    for job in jobs:
        paths = sorted(job.input_root.rglob(job.source_glob)) if job.input_root.is_dir() else []
        if job.multi_clip_source:
            if paths:
                shared_jobs.append(job)
            continue
        for path in paths:
            try:
                size = int(path.stat().st_size)
            except OSError:
                size = 0
            indexed_tasks.append((size, order, job, path))
            order += 1
    for size, original_order, job, path in sorted(
        indexed_tasks,
        key=lambda item: (-item[0], item[1]),
    ):
        target = min(range(world_size), key=lambda idx: (loads[idx], idx))
        bins[target].append((size, original_order, job, path))
        loads[target] += size

    grouped: dict[int, list[Path]] = {}
    job_by_index = {job.index: job for job in jobs}
    for _size, _order, job, path in bins[rank]:
        grouped.setdefault(job.index, []).append(path)
    rank_jobs = [
        (job_by_index[job_index], sorted(paths))
        for job_index, paths in sorted(grouped.items())
    ]
    rank_jobs.extend((job, None) for job in shared_jobs)
    return rank_jobs


def _job_work_units(job: SourceJob, source_paths: list[Path] | None) -> int:
    if source_paths is not None:
        return len(source_paths)
    if not job.input_root.is_dir():
        return 0
    return len(list(job.input_root.rglob(job.source_glob)))


def _global_work_unit_count(jobs: list[SourceJob], *, world_size: int) -> int:
    total = 0
    for job in jobs:
        count = _job_work_units(job, None)
        total += count * world_size if job.multi_clip_source else count
    return total


def _write_rank_progress(
    *,
    coord_root: Path,
    rank: int,
    completed_units: int,
    total_units: int,
    processed_frames: int,
    dataset: str,
) -> None:
    progress_root = coord_root / "progress"
    progress_root.mkdir(parents=True, exist_ok=True)
    path = progress_root / f"rank_{rank:05d}.json"
    temp = progress_root / f".rank_{rank:05d}.tmp"
    payload = {
        "rank": int(rank),
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "processed_frames": int(processed_frames),
        "dataset": str(dataset),
        "updated_at_epoch": time.time(),
    }
    temp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_rank_progress(coord_root: Path, world_size: int) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    progress_root = coord_root / "progress"
    for rank in range(world_size):
        path = progress_root / f"rank_{rank:05d}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _select_source_paths_for_rank(
    paths: list[Path],
    *,
    rank: int,
    world_size: int,
    strategy: str,
) -> list[Path]:
    strategy = str(strategy).strip().lower()
    if strategy in {"", "round_robin", "index"}:
        return list(paths[rank::world_size])
    if strategy not in {"size_greedy", "filesize", "bytes_greedy"}:
        raise ValueError(f"unsupported shard_strategy: {strategy}")
    bins: list[list[Path]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    indexed = []
    for order, path in enumerate(paths):
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        indexed.append((size, order, path))
    for size, _order, path in sorted(indexed, key=lambda item: (-item[0], item[1])):
        target = min(range(world_size), key=lambda idx: (loads[idx], idx))
        bins[target].append(path)
        loads[target] += size
    return sorted(bins[rank])


def _rank_source_root(output_root: Path, *, rank: int, job: SourceJob) -> Path:
    return output_root / "_rank_outputs" / f"rank_{rank:05d}" / job.output_key


def _distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _prepare_output(*, output_root: Path, coord_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    coord_root.mkdir(parents=True, exist_ok=True)
    (coord_root / "ranks").mkdir(parents=True, exist_ok=True)
    _write_json(coord_root / "init.json", {"status": "ready", "time": time.time()})


def _wait_for_init(coord_root: Path, *, timeout_sec: float) -> None:
    init_path = coord_root / "init.json"
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if init_path.is_file():
            return
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {init_path}")


def _wait_for_rank_done(*, coord_root: Path, world_size: int, timeout_sec: float) -> None:
    rank_root = coord_root / "ranks"
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        done = list(rank_root.glob("rank_*.json"))
        if len(done) >= world_size:
            return
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {world_size} rank done files")


def _read_optional_config(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {}
    return _load_config_with_defaults(Path(path_text))


def _load_config_with_defaults(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise RuntimeError(f"cyclic config defaults: {path}")
    data = load_config(path)
    defaults = data.get("defaults") or []
    body = {key: value for key, value in data.items() if key != "defaults"}
    merged: dict[str, Any] = {}
    inserted_self = False
    for item in defaults:
        if item == "_self_":
            merged = _merge_dict(merged, body)
            inserted_self = True
            continue
        if isinstance(item, str):
            child = Path(item)
        elif isinstance(item, dict) and len(item) == 1:
            group, name = next(iter(item.items()))
            child = Path(str(group)) / str(name)
        else:
            continue
        if child.suffix not in {".yaml", ".yml", ".json"}:
            child = child.with_suffix(".yaml")
        if not child.is_absolute():
            child = path.parent / child
        merged = _merge_dict(merged, _load_config_with_defaults(child, (*stack, path)))
    if not inserted_self:
        merged = _merge_dict(merged, body)
    return merged


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _print_plan(jobs: list[SourceJob], *, world_size: int) -> None:
    print(f"[holosmpl-h5] plan | sources={len(jobs)} world_size={world_size}", flush=True)
    for idx, job in enumerate(jobs, start=1):
        file_count = len(sorted(job.input_root.rglob(job.source_glob))) if job.input_root.is_dir() else 0
        print(
            f"[holosmpl-h5] source {idx}/{len(jobs)} {job.display} "
            f"| key={job.source_key} | files={file_count} "
            f"| multi_clip={job.multi_clip_source} | input={job.input_root}",
            flush=True,
        )


def _log_rank_progress(
    *,
    rank: int,
    job: SourceJob,
    result: dict[str, Any],
    done: int,
    total: int,
    start_time: float,
) -> None:
    elapsed = time.monotonic() - start_time
    print(
        f"[holosmpl-h5][rank {rank}] {done}/{total} {job.display} "
        f"| status={result.get('status')} | elapsed={elapsed:.1f}s",
        flush=True,
    )


def _sanitize_run_id(value: str) -> str:
    return _slug(value)[:120] or "holosmpl_run"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", value).strip("._")


def _format_count(value: int | float) -> str:
    value = float(value)
    if value >= 1.0e9:
        return f"{value / 1.0e9:.1f}B"
    if value >= 1.0e6:
        return f"{value / 1.0e6:.1f}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:.1f}K"
    return str(int(value))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_done_at(seconds: float) -> str:
    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(time.time() + max(0.0, float(seconds))),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m holosmpl.distributed_holosmpl_h5")
    parser.add_argument("--source-config", type=str)
    parser.add_argument("--raw-data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--only-name", action="append", default=[])
    parser.add_argument("--skip-name", action="append", default=[])
    parser.add_argument("--num-workers-per-rank", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--progress-log-interval-sec", type=float, default=60.0)
    parser.add_argument(
        "--compression",
        default="gzip",
        help="H5 compression filter; use lzf explicitly when write throughput matters more.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("direct_h5", "staged"),
        default="direct_h5",
        help="direct_h5 skips canonical/formal NPZ intermediates; staged preserves the old three-step path.",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=("size_greedy", "round_robin"),
        default="size_greedy",
        help="Per-rank source-file assignment for direct_h5.",
    )
    parser.add_argument(
        "--validate-rank-output",
        action="store_true",
        help="Read back each rank's H5 shards after writing. Off by default for production speed.",
    )
    parser.add_argument("--shard-target-gb", type=float, default=2.0)
    parser.add_argument("--chunks-t", type=int, default=1024)
    parser.add_argument("--combine-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--combine-npz", action="store_true")
    parser.add_argument("--validate-final", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-timeout-sec", type=float, default=600.0)
    parser.add_argument("--rank-timeout-sec", type=float, default=86400.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
