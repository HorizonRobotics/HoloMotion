from __future__ import annotations

import json
import re
import shutil
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from holosmpl.sources import get_source_config
from holosmpl.core.timing import StageTimer, format_timing_summary, summarize_timing
from holosmpl.core.validation.canonical_check import check_canonical_root
from holosmpl.core.writers.canonical_npz import write_canonical_npz


def convert_dataset_to_canonical(
    *,
    dataset: str,
    input_root: str | Path,
    output_root: str | Path,
    target_fps: float = 50.0,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    run_config_json: str | Path | None = None,
    overwrite: bool = False,
    num_workers: int = 1,
    progress_interval: int = 100,
    shard_index: int = 0,
    shard_count: int = 1,
    multi_clip_shard_index: int | None = None,
    multi_clip_shard_count: int = 1,
) -> dict[str, object]:
    dataset_key = dataset.lower()
    input_root = Path(input_root)
    output_root = Path(output_root)
    dataset_config = _dataset_config(dataset_key)
    if dataset_config is None:
        raise NotImplementedError(f"canonical conversion is not implemented for dataset: {dataset}")
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
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    multi_clip_shard_count = max(1, int(multi_clip_shard_count))
    if multi_clip_shard_index is None:
        multi_clip_shard_index = int(shard_index)
    multi_clip_shard_index = int(multi_clip_shard_index)
    if multi_clip_shard_index < 0 or multi_clip_shard_index >= multi_clip_shard_count:
        raise ValueError(
            "multi_clip_shard_index must be in "
            f"[0, {multi_clip_shard_count}), got {multi_clip_shard_index}"
        )
    source_paths = all_source_paths[shard_index::shard_count]
    if not source_paths:
        raise ValueError(
            f"no source files selected for shard_index={shard_index}, shard_count={shard_count}"
        )

    num_workers = max(1, int(num_workers))
    progress_interval = max(1, int(progress_interval))

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    if run_config_json is not None:
        _write_json(
            Path(run_config_json),
            {
                "schema_version": "canonical_run_config_v1",
                "dataset": source_name,
                "input_root": str(input_root),
                "output_root": str(output_root),
                "source_glob": source_glob,
                "target_fps": float(target_fps),
                "report_json": None if report_json is None else str(report_json),
                "report_md": None if report_md is None else str(report_md),
                "overwrite": bool(overwrite),
                "num_workers": int(num_workers),
                "progress_interval": int(progress_interval),
                "total_source_count": len(all_source_paths),
                "selected_source_count": len(source_paths),
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "multi_clip_shard_index": int(multi_clip_shard_index),
                "multi_clip_shard_count": int(multi_clip_shard_count),
            },
        )

    manifest_samples: list[dict[str, Any]] = []
    rejected_samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    print(
        "[canonical] discovered "
        f"{len(all_source_paths)} source files, selected {len(source_paths)} "
        f"for shard {shard_index}/{shard_count}, workers={num_workers}",
        flush=True,
    )
    start_time = time.monotonic()

    results: list[dict[str, Any]] = []
    if dataset_config.get("multi_clip_source"):
        for index, source_path in enumerate(source_paths):
            results.append(
                _process_multi_clip_source(
                    (
                        index,
                        str(source_path),
                        str(input_root),
                        str(output_root),
                        str(clips_root),
                        float(target_fps),
                        dataset_key,
                        int(multi_clip_shard_index),
                        int(multi_clip_shard_count),
                    ),
                    progress_interval=progress_interval,
                )
            )
            _maybe_log_progress(
                stage="canonical_source",
                done=len(results),
                total=len(source_paths),
                results=results,
                start_time=start_time,
                progress_interval=1,
            )
    else:
        worker_args = [
            (
                index,
                str(source_path),
                str(input_root),
                str(output_root),
                str(clips_root),
                float(target_fps),
                dataset_key,
            )
            for index, source_path in enumerate(source_paths)
        ]
        if num_workers == 1:
            for args in worker_args:
                results.append(_process_source(args))
                _maybe_log_progress(
                    stage="canonical",
                    done=len(results),
                    total=len(worker_args),
                    results=results,
                    start_time=start_time,
                    progress_interval=progress_interval,
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_source, args) for args in worker_args]
                for future in as_completed(futures):
                    results.append(future.result())
                    _maybe_log_progress(
                        stage="canonical",
                        done=len(results),
                        total=len(worker_args),
                        results=results,
                        start_time=start_time,
                        progress_interval=progress_interval,
                    )

    for result in sorted(results, key=lambda item: int(item["index"])):
        status = result.get("status")
        if status == "ok":
            manifest_samples.append(result["sample"])
        elif status == "many_ok":
            manifest_samples.extend(result["samples"])
        elif status == "rejected":
            rejected_samples.append(result["sample"])
        elif status == "many_rejected":
            rejected_samples.extend(result["samples"])
        elif status == "many_error":
            errors.extend(result["samples"])
        else:
            errors.append(result["sample"])

    timing_summary = summarize_timing(
        [*manifest_samples, *rejected_samples, *errors],
        frame_key="canonical_frame_count",
    )
    manifest = {
        "schema_version": "canonical_manifest_v1",
        "dataset": source_name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_glob": source_glob,
        "target_fps": float(target_fps),
        "total_source_count": len(all_source_paths),
        "source_count": len(source_paths),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "multi_clip_shard_index": int(multi_clip_shard_index),
        "multi_clip_shard_count": int(multi_clip_shard_count),
        "sample_count": len(manifest_samples),
        "rejected_count": len(rejected_samples),
        "error_count": len(errors),
        "samples": manifest_samples,
        "rejected_samples": rejected_samples,
        "errors": errors,
        "timing_summary": timing_summary,
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "rejected_manifest.json",
        {
            "schema_version": "canonical_rejected_manifest_v1",
            "dataset": source_name,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "total_source_count": len(all_source_paths),
            "source_count": len(source_paths),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "rejected_count": len(rejected_samples),
            "rejected_samples": rejected_samples,
        },
    )
    if errors:
        raise RuntimeError(f"conversion failed for {len(errors)} of {len(source_paths)} samples")

    report = check_canonical_root(
        output_root,
        expected_dataset=source_name,
        expected_count=len(manifest_samples),
        target_fps=target_fps,
        report_json=report_json,
        report_md=report_md,
        progress_interval=progress_interval,
    )
    summary = dict(report["summary"])
    summary.update(
        {
            "source_count": len(source_paths),
            "total_source_count": len(all_source_paths),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "converted_count": len(manifest_samples),
            "rejected_count": len(rejected_samples),
            "unexpected_error_count": len(errors),
            "timing_summary": timing_summary,
        }
    )
    timing_text = format_timing_summary(
        timing_summary,
        stages=("classify_ms", "convert_ms", "write_ms", "total_ms"),
    )
    if timing_text:
        print(f"[canonical] timing | {timing_text}", flush=True)
    return {
        "manifest": manifest,
        "validation": report["validation"],
        "summary": summary,
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "run_config_json": None if run_config_json is None else str(run_config_json),
    }


def _process_multi_clip_source(
    args: tuple[int, str, str, str, str, float, str, int, int],
    *,
    progress_interval: int,
) -> dict[str, Any]:
    (
        index,
        source_path_text,
        input_root_text,
        output_root_text,
        clips_root_text,
        target_fps,
        dataset_key,
        multi_clip_shard_index,
        multi_clip_shard_count,
    ) = args
    source_path = Path(source_path_text)
    input_root = Path(input_root_text)
    output_root = Path(output_root_text)
    clips_root = Path(clips_root_text)
    rel = source_path.relative_to(input_root).as_posix()
    dataset_config = _dataset_config(dataset_key)
    if dataset_config is None:
        raise NotImplementedError(f"canonical conversion is not implemented for dataset: {dataset_key}")
    source_name = str(dataset_config["source_name"])
    source_timer = StageTimer()
    with source_timer.measure("classify_ms"):
        source_info = dataset_config["classify"](source_path)
    if source_info["status"] != dataset_config["convertible_status"]:
        timing_ms = source_timer.finish()
        return {
            "index": index,
            "status": "rejected",
            "sample": {
                "index": index,
                "dataset": source_name,
                "source_path": str(source_path),
                "source_relative_path": rel,
                "status": "rejected",
                "reject_status": source_info["status"],
                "reason": source_info["reason"],
                "source_fields": source_info["source_fields"],
                "missing_field_groups": source_info["missing_field_groups"],
                "timing_ms": timing_ms,
            },
        }

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    iter_convert = dataset_config["iter_convert"]
    try:
        start_time = time.monotonic()
        iterator = iter(iter_convert(source_path, input_root=input_root, target_fps=target_fps))
        clip_index = 0
        while True:
            clip_timer = StageTimer()
            try:
                with clip_timer.measure("convert_ms"):
                    clip = next(iterator)
            except StopIteration:
                break
            metadata = dict(clip["metadata"])
            if clip_index % multi_clip_shard_count != multi_clip_shard_index:
                clip_index += 1
                continue
            source_relative_path = str(metadata.get("source_relative_path") or rel)
            clip_id = _clip_id_from_relative_path(source_relative_path)
            output_path = clips_root / f"{clip_id}.npz"
            metadata["clip_id"] = clip_id
            metadata["canonical_relative_path"] = output_path.relative_to(output_root).as_posix()
            clip["metadata"] = metadata
            with clip_timer.measure("write_ms"):
                write_canonical_npz(output_path, clip)
            clip_timing_ms = clip_timer.finish()
            source_timer.update(clip_timing_ms)
            samples.append(
                {
                    "index": clip_index,
                    "source_file_index": index,
                    "clip_id": clip_id,
                    "dataset": source_name,
                    "source_path": str(source_path),
                    "source_relative_path": source_relative_path,
                    "source_file_relative_path": rel,
                    "canonical_path": output_path.relative_to(output_root).as_posix(),
                    "source_fps": float(clip["source_fps"]),
                    "target_fps": float(clip["target_fps"]),
                    "source_frame_count": int(metadata["source_frame_count"]),
                    "canonical_frame_count": int(metadata["canonical_frame_count"]),
                    "pose_body_dim": int(metadata["pose_body_dim"]),
                    "pose_body_layout": str(metadata["pose_body_layout"]),
                    "coordinate_system": str(metadata["canonical_coordinate_system"]),
                    "resample_policy": str(metadata["resample_policy"]),
                    "status": "ok",
                    "timing_ms": clip_timing_ms,
                }
            )
            if (
                progress_interval
                and ((clip_index + 1) % progress_interval == 0)
            ):
                elapsed = max(time.monotonic() - start_time, 1e-6)
                print(
                    f"[canonical:{dataset_key}] source={rel} "
                    f"{clip_index + 1} clips written | "
                    f"{(clip_index + 1) / elapsed:.2f} clips/s | elapsed={elapsed:.1f}s",
                    flush=True,
                )
            clip_index += 1
    except Exception as exc:
        timing_ms = source_timer.finish()
        errors.append(
            {
                "index": index,
                "dataset": source_name,
                "source_path": str(source_path),
                "source_relative_path": rel,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "timing_ms": timing_ms,
            }
        )

    if errors:
        return {"index": index, "status": "many_error", "samples": errors}
    return {"index": index, "status": "many_ok", "samples": samples}


def _process_source(args: tuple[int, str, str, str, str, float, str]) -> dict[str, Any]:
    (
        index,
        source_path_text,
        input_root_text,
        output_root_text,
        clips_root_text,
        target_fps,
        dataset_key,
    ) = args
    source_path = Path(source_path_text)
    input_root = Path(input_root_text)
    output_root = Path(output_root_text)
    clips_root = Path(clips_root_text)
    rel = source_path.relative_to(input_root).as_posix()
    dataset_config = _dataset_config(dataset_key)
    if dataset_config is None:
        raise NotImplementedError(f"canonical conversion is not implemented for dataset: {dataset_key}")
    source_name = str(dataset_config["source_name"])
    timer = StageTimer()
    with timer.measure("classify_ms"):
        source_info = dataset_config["classify"](source_path)
    if source_info["status"] != dataset_config["convertible_status"]:
        timing_ms = timer.finish()
        return {
            "index": index,
            "status": "rejected",
            "sample": {
                "index": index,
                "dataset": source_name,
                "source_path": str(source_path),
                "source_relative_path": rel,
                "status": "rejected",
                "reject_status": source_info["status"],
                "reason": source_info["reason"],
                "source_fields": source_info["source_fields"],
                "missing_field_groups": source_info["missing_field_groups"],
                "timing_ms": timing_ms,
            },
        }
    try:
        convert_kwargs: dict[str, Any] = {
            "input_root": input_root,
            "target_fps": target_fps,
        }
        with timer.measure("convert_ms"):
            clip = dataset_config["convert"](source_path, **convert_kwargs)
        clip_id = _clip_id_from_relative_path(rel)
        output_path = clips_root / f"{clip_id}.npz"
        metadata = dict(clip["metadata"])
        metadata["clip_id"] = clip_id
        metadata["canonical_relative_path"] = output_path.relative_to(output_root).as_posix()
        clip["metadata"] = metadata
        with timer.measure("write_ms"):
            write_canonical_npz(output_path, clip)
        timing_ms = timer.finish()
        return {
            "index": index,
            "status": "ok",
            "sample": {
                "index": index,
                "clip_id": clip_id,
                "dataset": source_name,
                "source_path": str(source_path),
                "source_relative_path": rel,
                "canonical_path": output_path.relative_to(output_root).as_posix(),
                "source_fps": float(clip["source_fps"]),
                "target_fps": float(clip["target_fps"]),
                "source_frame_count": int(metadata["source_frame_count"]),
                "canonical_frame_count": int(metadata["canonical_frame_count"]),
                "pose_body_dim": int(metadata["pose_body_dim"]),
                "pose_body_layout": str(metadata["pose_body_layout"]),
                "coordinate_system": str(metadata["canonical_coordinate_system"]),
                "resample_policy": str(metadata["resample_policy"]),
                "status": "ok",
                "timing_ms": timing_ms,
            },
        }
    except Exception as exc:
        timing_ms = timer.finish()
        rejection = _exception_to_rejection(dataset_key=dataset_key, exc=exc)
        if rejection is not None:
            with timer.measure("classify_error_ms"):
                source_info = dataset_config["classify"](source_path)
            return {
                "index": index,
                "status": "rejected",
                "sample": {
                    "index": index,
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
                },
            }
        return {
            "index": index,
            "status": "error",
            "sample": {
                "index": index,
                "source_path": str(source_path),
                "source_relative_path": rel,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "timing_ms": timing_ms,
            },
        }


def _exception_to_rejection(*, dataset_key: str, exc: Exception) -> dict[str, str] | None:
    message = str(exc)
    if dataset_key == "motionmillion" and "contains NaN or Inf" in message:
        return {
            "status": "rejected_nonfinite_motion_fields",
            "reason": "source_motion_field_contains_nan_or_inf",
        }
    return None


def _dataset_config(dataset_key: str) -> dict[str, Any] | None:
    return get_source_config(dataset_key)


def _maybe_log_progress(
    *,
    stage: str,
    done: int,
    total: int,
    results: list[dict[str, Any]],
    start_time: float,
    progress_interval: int,
) -> None:
    if done != total and done % progress_interval != 0:
        return
    elapsed = max(time.monotonic() - start_time, 1e-6)
    ok = sum(1 for item in results if item.get("status") in {"ok", "many_ok"})
    rejected = sum(1 for item in results if item.get("status") in {"rejected", "many_rejected"})
    errors = sum(1 for item in results if item.get("status") in {"error", "many_error"})
    rate = done / elapsed
    print(
        f"[{stage}] {done}/{total} done | ok={ok} rejected={rejected} "
        f"errors={errors} | {rate:.2f} clips/s | elapsed={elapsed:.1f}s",
        flush=True,
    )


def _clip_id_from_relative_path(relative_path: str) -> str:
    path = Path(relative_path)
    stem = path.with_suffix("").as_posix()
    slug = re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", stem.replace("/", "__"))
    slug = slug.strip("._") or "clip"
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:170]}__{digest}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
