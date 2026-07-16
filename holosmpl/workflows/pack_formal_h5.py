from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from holosmpl.core.timing import StageTimer, format_timing_summary, summarize_timing
from holosmpl.core.validation.formal_h5_check import check_formal_h5_root
from holosmpl.core.writers.formal_h5 import (
    FORMAL_H5_ARRAY_FIELDS,
    FORMAL_H5_CLIP_FIELDS,
    FormalH5ShardWriter,
)


def pack_formal_npz_to_h5(
    *,
    formal_npz_root: str | Path,
    output_root: str | Path,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    run_config_json: str | Path | None = None,
    compression: str | None = "gzip",
    overwrite: bool = False,
    shard_target_gb: float = 2.0,
    shard_target_bytes: int | None = None,
    shard_target_mode: str = "uncompressed_nbytes",
    shard_target_clips: int = 0,
    shard_target_frames: int = 0,
    chunks_t: int = 1024,
    progress_interval: int = 100,
) -> dict[str, Any]:
    formal_npz_root = Path(formal_npz_root)
    output_root = Path(output_root)
    manifest_path = formal_npz_root / "manifest.json"
    if not formal_npz_root.is_dir():
        raise FileNotFoundError(f"formal_npz_root does not exist: {formal_npz_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"formal NPZ manifest does not exist: {manifest_path}")

    formal_manifest = _read_json(manifest_path)
    samples = formal_manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"formal NPZ manifest has no samples: {manifest_path}")
    progress_interval = max(1, int(progress_interval))
    chunks_t = max(1, int(chunks_t))
    shard_target_clips = max(0, int(shard_target_clips))
    shard_target_frames = max(0, int(shard_target_frames))
    if shard_target_bytes is None:
        shard_target_bytes = int(float(shard_target_gb) * (1 << 30))
    shard_target_bytes = max(1, int(shard_target_bytes))
    shard_target_mode = str(shard_target_mode).lower().strip()
    if shard_target_mode not in {
        "uncompressed_nbytes",
        "nbytes",
        "uncompressed",
        "npz_filesize",
        "npz_size",
        "npz_bytes",
        "h5_filesize",
        "h5_size",
        "output_filesize",
        "disk",
    }:
        raise ValueError(f"unsupported shard_target_mode: {shard_target_mode}")

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
                "schema_version": "formal_h5_run_config_v1",
                "formal_npz_root": str(formal_npz_root),
                "output_root": str(output_root),
                "compression": compression,
                "report_json": None if report_json is None else str(report_json),
                "report_md": None if report_md is None else str(report_md),
                "overwrite": bool(overwrite),
                "shard_target_gb": float(shard_target_gb),
                "shard_target_bytes": int(shard_target_bytes),
                "shard_target_mode": shard_target_mode,
                "shard_target_clips": int(shard_target_clips),
                "shard_target_frames": int(shard_target_frames),
                "chunks_t": int(chunks_t),
                "progress_interval": int(progress_interval),
            },
        )

    errors = []
    writer: FormalH5ShardWriter | None = None
    current_shard_index = 0
    current_shard_metric_bytes = 0
    current_shard_samples: list[dict[str, Any]] = []
    h5_shards: list[dict[str, Any]] = []
    h5_samples: list[dict[str, Any]] = []
    total_frames = 0
    total_clips = 0
    first_beta_dim: int | None = None
    start_time = time.monotonic()
    run_timer = StageTimer()

    print(
        f"[h5] packing {len(samples)} formal clips, compression={compression}, "
        f"target_bytes={shard_target_bytes}, target_clips={shard_target_clips}, "
        f"target_frames={shard_target_frames}, mode={shard_target_mode}",
        flush=True,
    )

    for index, sample in enumerate(samples):
        formal_rel = sample.get("formal_path")
        if not formal_rel:
            errors.append({"index": index, "error": "formal manifest sample missing formal_path"})
            continue
        formal_path = formal_npz_root / str(formal_rel)
        sample_timer = StageTimer()
        try:
            with sample_timer.measure("load_ms"):
                clip = _load_formal_npz_clip(formal_path, formal_npz_root=formal_npz_root)
        except Exception as exc:
            timing_ms = sample_timer.finish()
            errors.append(
                {
                    "index": index,
                    "formal_path": str(formal_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing_ms": timing_ms,
                }
            )
            continue
        with sample_timer.measure("plan_ms"):
            clip_frames = int(clip["human_pose_aa"].shape[0])
            clip_bytes = _estimate_clip_bytes(clip, formal_path, shard_target_mode)
            beta_dim = int(clip["human_shape_beta"].shape[0])
        if first_beta_dim is None:
            first_beta_dim = beta_dim
        elif beta_dim != first_beta_dim:
            timing_ms = sample_timer.finish()
            errors.append(
                {
                    "index": index,
                    "formal_path": str(formal_path),
                    "error": f"beta dim mismatch: {beta_dim} != {first_beta_dim}",
                    "timing_ms": timing_ms,
                }
            )
            continue

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
                shard_path = shards_root / f"shard_{current_shard_index:06d}.h5"
                writer = FormalH5ShardWriter(
                    shard_path,
                    beta_dim=beta_dim,
                    chunks_t=chunks_t,
                    compression=compression,
                )

        with sample_timer.measure("write_ms"):
            start, length = writer.append_clip(clip)
        timing_ms = sample_timer.finish()
        current_shard_samples.append(
            {
                "index": index,
                "clip_id": clip["clip_id"],
                "formal_relative_path": clip["formal_relative_path"],
                "start": int(start),
                "length": int(length),
                "timing_ms": timing_ms,
            }
        )
        run_timer.update(timing_ms)
        total_clips += 1
        total_frames += clip_frames
        if shard_target_mode in {"h5_filesize", "h5_size", "output_filesize", "disk"}:
            with run_timer.measure("flush_ms"):
                writer.flush()
                current_shard_metric_bytes = int(Path(writer.path).stat().st_size)
        else:
            current_shard_metric_bytes += clip_bytes
        _maybe_log_h5_progress(
            done=index + 1,
            total=len(samples),
            total_clips=total_clips,
            total_frames=total_frames,
            shard_index=current_shard_index,
            writer=writer,
            start_time=start_time,
            progress_interval=progress_interval,
        )
    if errors:
        timing_summary = summarize_timing(errors)
        _write_json(
            output_root / "manifest.json",
            {
                "schema_version": "formal_h5_manifest_v1",
                "formal_npz_root": str(formal_npz_root),
                "output_root": str(output_root),
                "dataset": formal_manifest.get("dataset"),
                "clip_count": 0,
                "frame_count": 0,
                "shards": [],
                "errors": errors,
                "timing_summary": timing_summary,
            },
        )
        raise RuntimeError(f"failed to load {len(errors)} formal NPZ clips")

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
    timing_summary = summarize_timing(h5_samples, frame_key="length")
    h5_manifest = {
        "schema_version": "formal_h5_manifest_v1",
        "formal_npz_root": str(formal_npz_root),
        "output_root": str(output_root),
        "dataset": formal_manifest.get("dataset"),
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
        "shard_target_mode": shard_target_mode,
        "shard_target_bytes": int(shard_target_bytes),
        "shard_target_clips": int(shard_target_clips),
        "shard_target_frames": int(shard_target_frames),
        "shards": h5_shards,
        "samples": h5_samples,
        "errors": [],
        "timing_summary": timing_summary,
    }
    _write_json(output_root / "manifest.json", h5_manifest)

    with run_timer.measure("validate_ms"):
        report = check_formal_h5_root(
            output_root,
            expected_clip_count=total_clips,
            report_json=report_json,
            report_md=report_md,
            progress_interval=progress_interval,
        )
    timing_text = format_timing_summary(
        timing_summary,
        stages=("load_ms", "plan_ms", "write_ms", "roll_finalize_ms", "total_ms"),
    )
    if timing_text:
        print(f"[h5] timing | {timing_text}", flush=True)
    summary = dict(report["summary"])
    summary["timing_summary"] = timing_summary
    return {
        "manifest": h5_manifest,
        "validation": report["validation"],
        "summary": summary,
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "run_config_json": None if run_config_json is None else str(run_config_json),
    }


def _load_formal_npz_clip(path: Path, *, formal_npz_root: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        metadata_json = str(data["metadata"].item())
        metadata = json.loads(metadata_json)
        clip = {
            "clip_id": str(metadata.get("clip_id") or path.stem),
            "formal_relative_path": path.relative_to(formal_npz_root).as_posix(),
            "metadata_json": metadata_json,
        }
        for field in FORMAL_H5_ARRAY_FIELDS:
            clip[field] = np.asarray(data[field], dtype=np.float32)
        clip["human_shape_beta"] = np.asarray(data["human_shape_beta"], dtype=np.float32)
    return clip


def _estimate_clip_bytes(
    clip: dict[str, Any],
    formal_path: Path,
    shard_target_mode: str,
) -> int:
    if shard_target_mode in {"npz_filesize", "npz_size", "npz_bytes"}:
        return int(formal_path.stat().st_size)
    total = 0
    for field in FORMAL_H5_ARRAY_FIELDS:
        total += int(clip[field].nbytes)
    total += int(clip["human_shape_beta"].nbytes)
    return total


def _should_roll_shard(
    *,
    writer: FormalH5ShardWriter,
    current_bytes: int,
    next_clip_frames: int,
    next_clip_bytes: int,
    shard_target_bytes: int,
    shard_target_clips: int,
    shard_target_frames: int,
) -> bool:
    if writer.clip_count <= 0:
        return False
    if shard_target_clips > 0 and writer.clip_count >= shard_target_clips:
        return True
    if shard_target_frames > 0 and writer.frame_count + next_clip_frames > shard_target_frames:
        return True
    if current_bytes + next_clip_bytes > shard_target_bytes:
        return True
    return False


def _finalize_current_shard(
    *,
    writer: FormalH5ShardWriter,
    output_root: Path,
    shard_index: int,
    current_samples: list[dict[str, Any]],
    h5_shards: list[dict[str, Any]],
    h5_samples: list[dict[str, Any]],
) -> None:
    shard_info = writer.finalize()
    shard_id = f"shard_{shard_index:06d}"
    shard_relative_path = Path(shard_info["path"]).relative_to(output_root).as_posix()
    clip_start_index = len(h5_samples)
    h5_shards.append(
        {
            "shard_id": shard_id,
            "path": shard_relative_path,
            "clip_count": int(shard_info["clip_count"]),
            "frame_count": int(shard_info["frame_count"]),
            "clip_start_index": clip_start_index,
            "clip_end_index": clip_start_index + int(shard_info["clip_count"]),
        }
    )
    for sample in current_samples:
        out_sample = dict(sample)
        out_sample["shard_id"] = shard_id
        out_sample["shard_path"] = shard_relative_path
        h5_samples.append(out_sample)


def _maybe_log_h5_progress(
    *,
    done: int,
    total: int,
    total_clips: int,
    total_frames: int,
    shard_index: int,
    writer: FormalH5ShardWriter,
    start_time: float,
    progress_interval: int,
) -> None:
    if done != total and done % progress_interval != 0:
        return
    elapsed = max(time.monotonic() - start_time, 1e-6)
    rate = done / elapsed
    print(
        f"[h5] {done}/{total} inputs | clips={total_clips} frames={total_frames} "
        f"| current_shard={shard_index} shard_clips={writer.clip_count} "
        f"shard_frames={writer.frame_count} | {rate:.2f} clips/s | elapsed={elapsed:.1f}s",
        flush=True,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
