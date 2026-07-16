from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from holosmpl.core.timing import StageTimer, format_timing_summary, summarize_timing
from holosmpl.core.validation.formal_h5_check import check_formal_h5_root
from holosmpl.core.writers.formal_h5 import (
    FORMAL_H5_ARRAY_FIELDS,
    FORMAL_H5_CLIP_FIELDS,
    FormalH5ShardWriter,
)
from holosmpl.sources import get_source_config
from holosmpl.workflows.convert_dataset import (
    _clip_id_from_relative_path,
    _exception_to_rejection,
)
from holosmpl.workflows.convert_formal import canonical_clip_to_formal_clip
from holosmpl.workflows.pack_formal_h5 import (
    _finalize_current_shard,
    _should_roll_shard,
)


def stream_dataset_to_formal_h5(
    *,
    dataset: str,
    input_root: str | Path,
    output_root: str | Path,
    target_fps: float = 50.0,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    run_config_json: str | Path | None = None,
    compression: str | None = "gzip",
    overwrite: bool = False,
    shard_target_gb: float = 2.0,
    shard_target_bytes: int | None = None,
    shard_target_clips: int = 0,
    shard_target_frames: int = 0,
    chunks_t: int = 1024,
    progress_interval: int = 100,
    shard_index: int = 0,
    shard_count: int = 1,
    multi_clip_shard_index: int | None = None,
    multi_clip_shard_count: int = 1,
    shard_strategy: str = "size_greedy",
    validate_output: bool = False,
    source_paths: list[Path] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_log_interval_sec: float = 60.0,
) -> dict[str, Any]:
    """Convert raw source files directly to formal H5 shards.

    This keeps the same source converters and formal field construction as the
    staged raw -> canonical NPZ -> formal NPZ -> H5 pipeline, but skips both
    intermediate NPZ write/read passes.
    """

    dataset_key = dataset.lower()
    input_root = Path(input_root)
    output_root = Path(output_root)
    dataset_config = get_source_config(dataset_key)
    if dataset_config is None:
        raise NotImplementedError(f"direct H5 conversion is not implemented for dataset: {dataset}")
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root does not exist or is not a directory: {input_root}")

    source_name = str(dataset_config["source_name"])
    source_glob = str(dataset_config["source_glob"])
    all_source_paths = sorted(input_root.rglob(source_glob))
    if not all_source_paths:
        raise FileNotFoundError(f"no {source_glob} files found under {input_root}")
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    multi_clip_shard_count = max(1, int(multi_clip_shard_count))
    if multi_clip_shard_index is None:
        multi_clip_shard_index = int(shard_index)
    multi_clip_shard_index = int(multi_clip_shard_index)
    if multi_clip_shard_index < 0 or multi_clip_shard_index >= multi_clip_shard_count:
        raise ValueError(
            "multi_clip_shard_index must be in "
            f"[0, {multi_clip_shard_count}), got {multi_clip_shard_index}"
        )
    if source_paths is None:
        source_paths = _select_source_paths(
            all_source_paths,
            shard_index=shard_index,
            shard_count=shard_count,
            strategy=shard_strategy,
            multi_clip_source=bool(dataset_config.get("multi_clip_source")),
        )
    else:
        source_paths = sorted(Path(path) for path in source_paths)
        all_source_set = set(all_source_paths)
        if any(path not in all_source_set for path in source_paths):
            raise ValueError("source_paths contains a path outside input_root")
    if not source_paths:
        raise ValueError(
            f"no source files selected for shard_index={shard_index}, shard_count={shard_count}"
        )

    progress_interval = max(1, int(progress_interval))
    progress_log_interval_sec = max(1.0, float(progress_log_interval_sec))
    chunks_t = max(1, int(chunks_t))
    shard_target_clips = max(0, int(shard_target_clips))
    shard_target_frames = max(0, int(shard_target_frames))
    if shard_target_bytes is None:
        shard_target_bytes = int(float(shard_target_gb) * (1 << 30))
    shard_target_bytes = max(1, int(shard_target_bytes))

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    shards_root = output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    if run_config_json is not None:
        _write_json(
            Path(run_config_json),
            {
                "schema_version": "direct_formal_h5_run_config_v1",
                "dataset": source_name,
                "input_root": str(input_root),
                "output_root": str(output_root),
                "source_glob": source_glob,
                "target_fps": float(target_fps),
                "compression": compression,
                "overwrite": bool(overwrite),
                "total_source_count": len(all_source_paths),
                "selected_source_count": len(source_paths),
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "multi_clip_shard_index": int(multi_clip_shard_index),
                "multi_clip_shard_count": int(multi_clip_shard_count),
                "shard_strategy": shard_strategy,
                "validate_output": bool(validate_output),
                "shard_target_gb": float(shard_target_gb),
                "shard_target_bytes": int(shard_target_bytes),
                "shard_target_clips": int(shard_target_clips),
                "shard_target_frames": int(shard_target_frames),
                "chunks_t": int(chunks_t),
                "progress_interval": int(progress_interval),
            },
        )

    print(
        "[direct_h5] discovered "
        f"{len(all_source_paths)} source files, selected {len(source_paths)} "
        f"for shard {shard_index}/{shard_count}, strategy={shard_strategy}",
        flush=True,
    )

    start_time = time.monotonic()
    run_timer = StageTimer()
    rejected_samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    h5_shards: list[dict[str, Any]] = []
    h5_samples: list[dict[str, Any]] = []
    current_shard_samples: list[dict[str, Any]] = []
    writer: FormalH5ShardWriter | None = None
    current_shard_index = 0
    current_shard_metric_bytes = 0
    total_clips = 0
    total_frames = 0
    first_beta_dim: int | None = None
    last_progress_time = start_time

    def append_formal_clip(
        *,
        index: int,
        source_file_index: int,
        source_path: Path,
        source_relative_path: str,
        canonical_clip: dict[str, Any],
        clip_id: str,
    ) -> None:
        nonlocal current_shard_index
        nonlocal current_shard_metric_bytes
        nonlocal current_shard_samples
        nonlocal first_beta_dim
        nonlocal total_clips
        nonlocal total_frames
        nonlocal writer
        nonlocal last_progress_time

        sample_timer = StageTimer()
        metadata = dict(canonical_clip["metadata"])
        metadata["clip_id"] = clip_id
        metadata["canonical_relative_path"] = f"direct/{clip_id}"
        canonical_clip["metadata"] = metadata
        with sample_timer.measure("formal_convert_ms"):
            formal_clip, sample_metadata = canonical_clip_to_formal_clip(
                root_orient=canonical_clip["root_orient"],
                pose_body=canonical_clip["pose_body"],
                trans=canonical_clip["trans"],
                betas=canonical_clip["betas"],
                canonical_metadata=metadata,
                clip_id=clip_id,
                canonical_path=None,
                canonical_relative_path=f"direct/{clip_id}",
            )
        formal_relative_path = f"stream/{clip_id}"
        formal_clip["metadata"]["formal_relative_path"] = formal_relative_path
        metadata_json = json.dumps(
            formal_clip["metadata"],
            ensure_ascii=False,
            sort_keys=True,
        )
        h5_clip = {
            "clip_id": clip_id,
            "formal_relative_path": formal_relative_path,
            "metadata_json": metadata_json,
            "human_pose_aa": formal_clip["human_pose_aa"],
            "human_shape_beta": formal_clip["human_shape_beta"],
            "human_root_trans": formal_clip["human_root_trans"],
            "human_root_height": formal_clip["human_root_height"],
            "human_gravity_projection": formal_clip["human_gravity_projection"],
        }
        with sample_timer.measure("plan_ms"):
            clip_frames = int(h5_clip["human_pose_aa"].shape[0])
            clip_bytes = _estimate_stream_clip_bytes(h5_clip)
            beta_dim = int(h5_clip["human_shape_beta"].shape[0])
        if first_beta_dim is None:
            first_beta_dim = beta_dim
        elif beta_dim != first_beta_dim:
            timing_ms = sample_timer.finish()
            errors.append(
                {
                    "index": index,
                    "source_path": str(source_path),
                    "error": f"beta dim mismatch: {beta_dim} != {first_beta_dim}",
                    "timing_ms": timing_ms,
                }
            )
            return
        with sample_timer.measure("roll_finalize_ms"):
            if writer is not None and _should_roll_shard(
                writer=writer,
                current_bytes=current_shard_metric_bytes,
                next_clip_frames=clip_frames,
                next_clip_bytes=clip_bytes,
                shard_target_bytes=shard_target_bytes,
                shard_target_clips=shard_target_clips,
                shard_target_frames=shard_target_frames,
            ):
                _finalize_current_shard(
                    writer=writer,
                    output_root=output_root,
                    shard_index=current_shard_index,
                    current_samples=current_shard_samples,
                    h5_shards=h5_shards,
                    h5_samples=h5_samples,
                )
                current_shard_index += 1
                writer = None
                current_shard_metric_bytes = 0
                current_shard_samples = []
        with sample_timer.measure("open_shard_ms"):
            if writer is None:
                writer = FormalH5ShardWriter(
                    shards_root / f"shard_{current_shard_index:06d}.h5",
                    beta_dim=beta_dim,
                    chunks_t=chunks_t,
                    compression=compression,
                )
        with sample_timer.measure("write_ms"):
            start, length = writer.append_clip(h5_clip)
        timing_ms = sample_timer.finish()
        run_timer.update(timing_ms)
        current_shard_metric_bytes += clip_bytes
        current_shard_samples.append(
            {
                "index": index,
                "source_file_index": source_file_index,
                "clip_id": clip_id,
                "source_path": str(source_path),
                "source_relative_path": source_relative_path,
                "formal_relative_path": formal_relative_path,
                "start": int(start),
                "length": int(length),
                "frame_count": int(sample_metadata["frame_count"]),
                "beta_dim": int(sample_metadata["beta_dim"]),
                "formal_pose_72_policy": sample_metadata["formal_pose_72_policy"],
                "timing_ms": timing_ms,
            }
        )
        total_clips += 1
        total_frames += clip_frames
        last_progress_time = _maybe_log_direct_progress(
            done=total_clips,
            total_sources=len(source_paths),
            total_frames=total_frames,
            shard_index=current_shard_index,
            writer=writer,
            start_time=start_time,
            progress_interval=progress_interval,
            progress_log_interval_sec=progress_log_interval_sec,
            last_progress_time=last_progress_time,
            progress_callback=progress_callback,
        )

    for source_file_index, source_path in enumerate(source_paths):
        rel = source_path.relative_to(input_root).as_posix()
        source_timer = StageTimer()
        with source_timer.measure("classify_ms"):
            source_info = dataset_config["classify"](source_path)
        if source_info["status"] != dataset_config["convertible_status"]:
            rejected_samples.append(
                {
                    "index": source_file_index,
                    "dataset": source_name,
                    "source_path": str(source_path),
                    "source_relative_path": rel,
                    "status": "rejected",
                    "reject_status": source_info["status"],
                    "reason": source_info["reason"],
                    "source_fields": source_info.get("source_fields", []),
                    "missing_field_groups": source_info.get("missing_field_groups", []),
                    "timing_ms": source_timer.finish(),
                }
            )
            continue
        try:
            if dataset_config.get("multi_clip_source"):
                iter_convert = dataset_config["iter_convert"]
                iterator = iter(
                    iter_convert(
                        source_path,
                        input_root=input_root,
                        target_fps=float(target_fps),
                        sequence_shard_index=multi_clip_shard_index,
                        sequence_shard_count=multi_clip_shard_count,
                    )
                )
                clip_index = 0
                while True:
                    try:
                        with source_timer.measure("source_convert_ms"):
                            canonical_clip = next(iterator)
                    except StopIteration:
                        break
                    metadata = dict(canonical_clip["metadata"])
                    source_relative_path = str(metadata.get("source_relative_path") or rel)
                    clip_id = _clip_id_from_relative_path(source_relative_path)
                    append_formal_clip(
                        index=clip_index,
                        source_file_index=source_file_index,
                        source_path=source_path,
                        source_relative_path=source_relative_path,
                        canonical_clip=canonical_clip,
                        clip_id=clip_id,
                    )
                    clip_index += 1
            else:
                with source_timer.measure("source_convert_ms"):
                    canonical_clip = dataset_config["convert"](
                        source_path,
                        input_root=input_root,
                        target_fps=float(target_fps),
                    )
                clip_id = _clip_id_from_relative_path(rel)
                append_formal_clip(
                    index=source_file_index,
                    source_file_index=source_file_index,
                    source_path=source_path,
                    source_relative_path=rel,
                    canonical_clip=canonical_clip,
                    clip_id=clip_id,
                )
        except Exception as exc:
            rejection = _exception_to_rejection(dataset_key=dataset_key, exc=exc)
            timing_ms = source_timer.finish()
            if rejection is not None:
                rejected_samples.append(
                    {
                        "index": source_file_index,
                        "dataset": source_name,
                        "source_path": str(source_path),
                        "source_relative_path": rel,
                        "status": "rejected",
                        "reject_status": rejection["status"],
                        "reason": rejection["reason"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "source_fields": source_info.get("source_fields", []),
                        "missing_field_groups": source_info.get("missing_field_groups", []),
                        "timing_ms": timing_ms,
                    }
                )
            else:
                errors.append(
                    {
                        "index": source_file_index,
                        "dataset": source_name,
                        "source_path": str(source_path),
                        "source_relative_path": rel,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "timing_ms": timing_ms,
                    }
                )

    if writer is not None:
        with run_timer.measure("finalize_ms"):
            _finalize_current_shard(
                writer=writer,
                output_root=output_root,
                shard_index=current_shard_index,
                current_samples=current_shard_samples,
                h5_shards=h5_shards,
                h5_samples=h5_samples,
            )

    if progress_callback is not None:
        elapsed = max(time.monotonic() - start_time, 1.0e-6)
        progress_callback(
            {
                "completed_units": int(len(source_paths)),
                "total_units": int(len(source_paths)),
                "processed_frames": int(total_frames),
                "dataset_units": int(len(source_paths)),
                "dataset_total_units": int(len(source_paths)),
                "fps": float(total_frames / elapsed),
                "updated_monotonic": time.monotonic(),
            }
        )

    timing_summary = summarize_timing(h5_samples, frame_key="length")
    h5_manifest = {
        "schema_version": "formal_h5_manifest_v1",
        "pipeline_mode": "direct_h5",
        "formal_npz_root": None,
        "output_root": str(output_root),
        "dataset": source_name,
        "source_glob": source_glob,
        "input_root": str(input_root),
        "total_source_count": len(all_source_paths),
        "source_count": len(source_paths),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "multi_clip_shard_index": int(multi_clip_shard_index),
        "multi_clip_shard_count": int(multi_clip_shard_count),
        "shard_strategy": shard_strategy,
        "clip_count": total_clips,
        "frame_count": total_frames,
        "available_arrays": list(FORMAL_H5_ARRAY_FIELDS),
        "clip_level_arrays": list(FORMAL_H5_CLIP_FIELDS),
        "layout": "frame_major_flat",
        "clip_index_fields": [
            "clips/start",
            "clips/length",
            "clips/motion_key_id",
            "clips/metadata_json",
            "clips/formal_relative_path",
            "clips/human_shape_beta",
        ],
        "compression": compression,
        "chunks_t": int(chunks_t),
        "shard_target_mode": "uncompressed_nbytes",
        "shard_target_bytes": int(shard_target_bytes),
        "shard_target_clips": int(shard_target_clips),
        "shard_target_frames": int(shard_target_frames),
        "shards": h5_shards,
        "samples": h5_samples,
        "rejected_samples": rejected_samples,
        "errors": errors,
        "timing_summary": timing_summary,
    }
    _write_json(output_root / "manifest.json", h5_manifest)
    if errors:
        raise RuntimeError(f"direct H5 conversion failed for {len(errors)} source files")

    if total_clips > 0 and validate_output:
        with run_timer.measure("validate_ms"):
            report = check_formal_h5_root(
                output_root,
                expected_clip_count=total_clips,
                report_json=report_json,
                report_md=report_md,
                progress_interval=progress_interval,
            )
        validation = report["validation"]
        summary = dict(report["summary"])
    else:
        validation = {"status": "skipped_empty" if total_clips <= 0 else "skipped_by_config"}
        summary = {
            "clip_count": total_clips,
            "frame_count": total_frames,
            "timing_summary": timing_summary,
        }
    summary["timing_summary"] = timing_summary
    timing_text = format_timing_summary(
        timing_summary,
        stages=(
            "formal_convert_ms",
            "plan_ms",
            "write_ms",
            "roll_finalize_ms",
            "total_ms",
        ),
    )
    if timing_text:
        print(f"[direct_h5] timing | {timing_text}", flush=True)
    return {
        "manifest": h5_manifest,
        "validation": validation,
        "summary": summary,
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "run_config_json": None if run_config_json is None else str(run_config_json),
    }


def _estimate_stream_clip_bytes(clip: dict[str, Any]) -> int:
    total = 0
    for field in FORMAL_H5_ARRAY_FIELDS:
        total += int(clip[field].nbytes)
    total += int(clip["human_shape_beta"].nbytes)
    return total


def _select_source_paths(
    paths: list[Path],
    *,
    shard_index: int,
    shard_count: int,
    strategy: str,
    multi_clip_source: bool,
) -> list[Path]:
    if multi_clip_source:
        return list(paths)
    strategy = str(strategy).strip().lower()
    if strategy in {"", "round_robin", "index"}:
        return list(paths[shard_index::shard_count])
    if strategy not in {"size_greedy", "filesize", "bytes_greedy"}:
        raise ValueError(f"unsupported direct H5 shard_strategy: {strategy}")
    bins: list[list[Path]] = [[] for _ in range(shard_count)]
    loads = [0 for _ in range(shard_count)]
    indexed = []
    for order, path in enumerate(paths):
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        indexed.append((size, order, path))
    for size, _order, path in sorted(indexed, key=lambda item: (-item[0], item[1])):
        target = min(range(shard_count), key=lambda idx: (loads[idx], idx))
        bins[target].append(path)
        loads[target] += size
    return sorted(bins[shard_index])


def _maybe_log_direct_progress(
    *,
    done: int,
    total_sources: int,
    total_frames: int,
    shard_index: int,
    writer: FormalH5ShardWriter,
    start_time: float,
    progress_interval: int,
    progress_log_interval_sec: float,
    last_progress_time: float,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> float:
    now = time.monotonic()
    if done % progress_interval != 0 and now - last_progress_time < progress_log_interval_sec:
        return last_progress_time
    elapsed = max(now - start_time, 1e-6)
    print(
        f"[direct_h5] clips={done} sources={total_sources} frames={total_frames} "
        f"| current_shard={shard_index} shard_clips={writer.clip_count} "
        f"shard_frames={writer.frame_count} | {done / elapsed:.2f} clips/s "
        f"| elapsed={elapsed:.1f}s",
        flush=True,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "completed_units": int(done),
                "total_units": int(total_sources),
                "processed_frames": int(total_frames),
                "dataset_units": int(done),
                "dataset_total_units": int(total_sources),
                "fps": float(total_frames / elapsed),
                "updated_monotonic": now,
            }
        )
    return now


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
