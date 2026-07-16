from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from holosmpl.core.processing.derived_fields import derive_formal_fields
from holosmpl.core.processing.pose72 import canonical_pose_to_human_pose_aa
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
    WORLD_GRAVITY,
)
from holosmpl.core.timing import StageTimer, format_timing_summary, summarize_timing
from holosmpl.core.validation.formal_check import check_formal_root
from holosmpl.core.writers.formal_npz import write_formal_npz


def convert_canonical_to_formal_npz(
    *,
    canonical_root: str | Path,
    output_root: str | Path,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    run_config_json: str | Path | None = None,
    overwrite: bool = False,
    num_workers: int = 1,
    progress_interval: int = 100,
) -> dict[str, object]:
    import numpy as np

    canonical_root = Path(canonical_root)
    output_root = Path(output_root)
    manifest_path = canonical_root / "manifest.json"
    clips_root = canonical_root / "clips"
    if not canonical_root.is_dir():
        raise FileNotFoundError(f"canonical_root does not exist: {canonical_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical manifest does not exist: {manifest_path}")
    if not clips_root.is_dir():
        raise FileNotFoundError(f"canonical clips dir does not exist: {clips_root}")

    canonical_manifest = _read_json(manifest_path)
    canonical_samples = canonical_manifest.get("samples", [])
    if not isinstance(canonical_samples, list) or not canonical_samples:
        raise ValueError(f"canonical manifest has no samples: {manifest_path}")
    num_workers = max(1, int(num_workers))
    progress_interval = max(1, int(progress_interval))

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    formal_clips_root = output_root / "clips"
    formal_clips_root.mkdir(parents=True, exist_ok=True)
    if run_config_json is not None:
        _write_json(
            Path(run_config_json),
            {
                "schema_version": "formal_npz_run_config_v1",
                "canonical_root": str(canonical_root),
                "output_root": str(output_root),
                "report_json": None if report_json is None else str(report_json),
                "report_md": None if report_md is None else str(report_md),
                "overwrite": bool(overwrite),
                "num_workers": int(num_workers),
                "progress_interval": int(progress_interval),
            },
        )

    print(
        f"[formal_npz] converting {len(canonical_samples)} canonical clips, "
        f"workers={num_workers}",
        flush=True,
    )
    start_time = time.monotonic()
    worker_args = [
        (
            index,
            canonical_sample,
            str(canonical_root),
            str(output_root),
            str(formal_clips_root),
        )
        for index, canonical_sample in enumerate(canonical_samples)
    ]
    results: list[dict[str, Any]] = []
    if num_workers == 1:
        for args in worker_args:
            results.append(_process_canonical_formal_clip(args))
            _maybe_log_progress(
                stage="formal_npz",
                done=len(results),
                total=len(worker_args),
                results=results,
                start_time=start_time,
                progress_interval=progress_interval,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_canonical_formal_clip, args) for args in worker_args]
            for future in as_completed(futures):
                results.append(future.result())
                _maybe_log_progress(
                    stage="formal_npz",
                    done=len(results),
                    total=len(worker_args),
                    results=results,
                    start_time=start_time,
                    progress_interval=progress_interval,
                )

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["index"])):
        if result.get("status") == "ok":
            samples.append(result["sample"])
        else:
            errors.append(result["sample"])

    timing_summary = summarize_timing(samples, frame_key="frame_count")
    manifest = {
        "schema_version": "formal_npz_manifest_v1",
        "canonical_root": str(canonical_root),
        "output_root": str(output_root),
        "dataset": canonical_manifest.get("dataset"),
        "source_count": len(canonical_samples),
        "sample_count": len(samples),
        "error_count": len(errors),
        "samples": samples,
        "errors": errors,
        "timing_summary": timing_summary,
    }
    _write_json(output_root / "manifest.json", manifest)
    if errors:
        raise RuntimeError(f"formal conversion failed for {len(errors)} clips")

    report = check_formal_root(
        output_root,
        expected_dataset=str(canonical_manifest.get("dataset")),
        expected_count=len(samples),
        report_json=report_json,
        report_md=report_md,
        progress_interval=progress_interval,
    )
    summary = dict(report["summary"])
    summary["timing_summary"] = timing_summary
    timing_text = format_timing_summary(
        timing_summary,
        stages=("load_ms", "convert_ms", "write_ms", "total_ms"),
    )
    if timing_text:
        print(f"[formal_npz] timing | {timing_text}", flush=True)
    return {
        "manifest": manifest,
        "validation": report["validation"],
        "summary": summary,
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "run_config_json": None if run_config_json is None else str(run_config_json),
    }


def _process_canonical_formal_clip(
    args: tuple[int, dict[str, Any], str, str, str],
) -> dict[str, Any]:
    index, canonical_sample, canonical_root_text, output_root_text, formal_clips_root_text = args
    canonical_root = Path(canonical_root_text)
    output_root = Path(output_root_text)
    formal_clips_root = Path(formal_clips_root_text)
    canonical_rel = canonical_sample.get("canonical_path")
    if not canonical_rel:
        timer = StageTimer()
        timing_ms = timer.finish()
        return {
            "index": index,
            "status": "error",
            "sample": {
                "index": index,
                "status": "error",
                "error": "canonical manifest sample missing canonical_path",
                "timing_ms": timing_ms,
            },
        }
    canonical_path = canonical_root / str(canonical_rel)
    timer = StageTimer()
    try:
        formal_clip, sample_metadata, convert_timing_ms = _convert_one_canonical_clip(
            canonical_path=canonical_path,
            canonical_root=canonical_root,
        )
        timer.update(convert_timing_ms)
        clip_id = str(sample_metadata["clip_id"])
        formal_path = formal_clips_root / f"{clip_id}.npz"
        formal_relative_path = formal_path.relative_to(output_root).as_posix()
        formal_clip["metadata"]["formal_relative_path"] = formal_relative_path
        with timer.measure("write_ms"):
            write_formal_npz(formal_path, formal_clip)
        timing_ms = timer.finish()
        return {
            "index": index,
            "status": "ok",
            "sample": {
                "index": index,
                "clip_id": clip_id,
                "dataset": sample_metadata["dataset"],
                "canonical_path": str(canonical_path),
                "canonical_relative_path": canonical_path.relative_to(canonical_root).as_posix(),
                "formal_path": formal_relative_path,
                "frame_count": int(sample_metadata["frame_count"]),
                "beta_dim": int(sample_metadata["beta_dim"]),
                "formal_pose_72_policy": sample_metadata["formal_pose_72_policy"],
                "status": "ok",
                "timing_ms": timing_ms,
            },
        }
    except Exception as exc:
        timing_ms = timer.finish()
        return {
            "index": index,
            "status": "error",
            "sample": {
                "index": index,
                "canonical_path": str(canonical_path),
                "canonical_relative_path": str(canonical_rel),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "timing_ms": timing_ms,
            },
        }


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
    ok = sum(1 for item in results if item.get("status") == "ok")
    errors = sum(1 for item in results if item.get("status") == "error")
    rate = done / elapsed
    print(
        f"[{stage}] {done}/{total} done | ok={ok} errors={errors} | "
        f"{rate:.2f} clips/s | elapsed={elapsed:.1f}s",
        flush=True,
    )


def _convert_one_canonical_clip(
    *,
    canonical_path: Path,
    canonical_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    import numpy as np

    timer = StageTimer()
    with timer.measure("load_ms"):
        with np.load(canonical_path, allow_pickle=False) as data:
            root_orient = np.asarray(data["root_orient"], dtype=np.float32)
            pose_body = np.asarray(data["pose_body"], dtype=np.float32)
            trans = np.asarray(data["trans"], dtype=np.float32)
            betas = np.asarray(data["betas"], dtype=np.float32)
            canonical_metadata = json.loads(str(data["metadata"].item()))

    clip_id = str(canonical_metadata.get("clip_id") or canonical_path.stem)
    try:
        canonical_relative_path = canonical_path.relative_to(canonical_root).as_posix()
    except ValueError:
        canonical_relative_path = canonical_path.as_posix()
    with timer.measure("convert_ms"):
        formal_clip, sample_metadata = canonical_clip_to_formal_clip(
            root_orient=root_orient,
            pose_body=pose_body,
            trans=trans,
            betas=betas,
            canonical_metadata=canonical_metadata,
            clip_id=clip_id,
            canonical_path=str(canonical_path),
            canonical_relative_path=canonical_relative_path,
        )
    return formal_clip, sample_metadata, timer.finish()


def canonical_clip_to_formal_clip(
    *,
    root_orient: Any,
    pose_body: Any,
    trans: Any,
    betas: Any,
    canonical_metadata: dict[str, Any],
    clip_id: str,
    canonical_path: str | None,
    canonical_relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one in-memory canonical clip to formal arrays.

    This is shared by staged NPZ production and direct H5 production so both
    paths keep identical formal semantics.
    """

    import numpy as np

    root_orient = np.asarray(root_orient, dtype=np.float32)
    pose_body = np.asarray(pose_body, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    body_pose_layout = str(canonical_metadata["pose_body_layout"])
    human_pose_aa, pose72_metadata = canonical_pose_to_human_pose_aa(
        root_orient,
        pose_body,
        body_pose_layout,
    )
    frame_count = int(human_pose_aa.shape[0])
    human_root_trans = trans.astype(np.float32, copy=True)
    human_shape_beta = betas.astype(np.float32, copy=True)
    derived = derive_formal_fields(human_pose_aa, human_root_trans)

    metadata = {
        "schema_version": "formal_human_npz_v1",
        "dataset": canonical_metadata.get("dataset"),
        "clip_id": clip_id,
        "canonical_path": canonical_path,
        "canonical_relative_path": canonical_relative_path,
        "source_path": canonical_metadata.get("source_path"),
        "source_relative_path": canonical_metadata.get("source_relative_path"),
        "source_fps": canonical_metadata.get("source_fps"),
        "target_fps": canonical_metadata.get("target_fps"),
        "frame_count": frame_count,
        "beta_dim": int(betas.shape[0]),
        "canonical_pose_body_dim": canonical_metadata.get("pose_body_dim"),
        "canonical_pose_body_layout": body_pose_layout,
        "canonical_coordinate_system": CANONICAL_COORDINATE_FRAME,
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "slice_policy": "none",
        "formal_shape_beta_policy": "copy_canonical_betas_clip_level",
        "formal_root_trans_policy": "copy_canonical_trans",
        "formal_root_height_policy": "derive_from_human_root_trans_z",
        "formal_gravity_projection_policy": "derive_from_human_pose_root_orient",
        **pose72_metadata,
    }
    for key in (
        "source_pose_field",
        "source_translation_field",
        "source_fps_field",
        "original_fps",
        "original_pose_aa_shape",
        "original_pose_aa_policy",
        "betas_policy",
        "source_native_beta_present",
        "source_coordinate_system",
        "coordinate_transform",
        "coordinate_transform_matrix",
        "coordinate_transform_matrix_source_trans_to_canonical",
        "bones_seed_smpl_transl_axis_policy",
        "bones_seed_smpl_joints_axis_policy",
        "bones_seed_smpl_axis_consistency_check",
        "bones_seed_smpl_root_trans_source",
        "beta_clip_policy",
        "beta_clip_abs",
        "beta_clip_applied",
        "beta_abs_before_clip_max",
        "beta_abs_after_clip_max",
        "root_frame_semantics",
        "root_orient_policy",
        "root_frame_certified",
    ):
        if key in canonical_metadata:
            metadata[key] = canonical_metadata[key]
    formal_clip = {
        "human_pose_aa": human_pose_aa.astype(np.float32),
        "human_shape_beta": human_shape_beta,
        "human_root_trans": human_root_trans,
        "human_root_height": derived["human_root_height"],
        "human_gravity_projection": derived["human_gravity_projection"],
        "metadata": metadata,
    }
    sample_metadata = {
        "clip_id": clip_id,
        "dataset": metadata["dataset"],
        "frame_count": frame_count,
        "beta_dim": int(betas.shape[0]),
        "formal_pose_72_policy": metadata["formal_pose_72_policy"],
    }
    return formal_clip, sample_metadata


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
