from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from holosmpl.core.processing.derived_fields import derive_formal_fields
from holosmpl.core.schema.canonical import CANONICAL_COORDINATE_FRAME
from holosmpl.core.writers.formal_h5 import FORMAL_H5_ARRAY_FIELDS


def check_formal_h5_root(
    formal_h5_root: str | Path,
    *,
    expected_clip_count: int | None = None,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    progress_interval: int | None = None,
) -> dict[str, Any]:
    import h5py
    import numpy as np

    formal_h5_root = Path(formal_h5_root)
    manifest_path = formal_h5_root / "manifest.json"
    shards_root = formal_h5_root / "shards"
    errors: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []

    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if manifest is None:
        errors.append({"scope": "manifest", "error": f"missing {manifest_path}"})

    shard_paths = sorted(shards_root.glob("*.h5"))
    if not shard_paths:
        errors.append({"scope": "shards", "error": f"no h5 shards found under {shards_root}"})

    total_clip_count = 0
    total_frame_count = 0
    beta_dims: set[int] = set()
    pose_dims: set[int] = set()
    coordinate_counts: dict[str, int] = {}
    progress_interval = max(1, int(progress_interval or 0)) if progress_interval else 0
    start_time = time.monotonic()

    for shard_index, shard_path in enumerate(shard_paths, start=1):
        shard_report: dict[str, Any] = {
            "shard_path": str(shard_path.relative_to(formal_h5_root)),
        }
        try:
            with h5py.File(shard_path, "r") as handle:
                missing = [field for field in FORMAL_H5_ARRAY_FIELDS if field not in handle]
                if missing:
                    raise ValueError(f"missing datasets: {missing}")
                if "clips" not in handle:
                    raise ValueError("missing clips group")

                arrays = {field: handle[field] for field in FORMAL_H5_ARRAY_FIELDS}
                frame_count = int(arrays["human_pose_aa"].shape[0])
                if arrays["human_pose_aa"].shape[1] != 72:
                    raise ValueError("human_pose_aa must be [N,72]")
                if arrays["human_root_trans"].shape != (frame_count, 3):
                    raise ValueError("human_root_trans must be [N,3]")
                if arrays["human_root_height"].shape != (frame_count, 1):
                    raise ValueError("human_root_height must be [N,1]")
                if arrays["human_gravity_projection"].shape != (frame_count, 3):
                    raise ValueError("human_gravity_projection must be [N,3]")
                for field, dataset in arrays.items():
                    if str(dataset.dtype) != "float32":
                        raise ValueError(f"{field} dtype must be float32, got {dataset.dtype}")

                clips_group = handle["clips"]
                for name in (
                    "start",
                    "length",
                    "motion_key_id",
                    "metadata_json",
                    "human_shape_beta",
                ):
                    if name not in clips_group:
                        raise ValueError(f"missing clips/{name}")
                starts = clips_group["start"][:]
                lengths = clips_group["length"][:]
                clip_count = int(len(lengths))
                if len(starts) != clip_count:
                    raise ValueError("clips/start and clips/length length mismatch")
                if clip_count == 0:
                    raise ValueError("shard has zero clips")
                expected_starts = np.zeros_like(starts)
                if clip_count > 1:
                    expected_starts[1:] = np.cumsum(lengths[:-1])
                if not np.array_equal(starts, expected_starts):
                    raise ValueError("clip starts are not contiguous")
                if int(lengths.sum()) != frame_count:
                    raise ValueError(
                        f"sum(clips/length) {int(lengths.sum())} != frame_count {frame_count}"
                    )
                shape_beta = clips_group["human_shape_beta"][:]
                if shape_beta.ndim != 2 or shape_beta.shape[0] != clip_count:
                    raise ValueError(
                        "clips/human_shape_beta must be [num_clips,B], "
                        f"got {shape_beta.shape}"
                    )

                pose = arrays["human_pose_aa"][:]
                trans = arrays["human_root_trans"][:]
                height = arrays["human_root_height"][:]
                gravity = arrays["human_gravity_projection"][:]
                for field, array in {
                    "human_pose_aa": pose,
                    "clips/human_shape_beta": shape_beta,
                    "human_root_trans": trans,
                    "human_root_height": height,
                    "human_gravity_projection": gravity,
                }.items():
                    if not np.isfinite(array).all():
                        raise ValueError(f"{field} contains NaN or Inf")

                derived = derive_formal_fields(pose, trans)
                max_height_error = float(np.max(np.abs(height - derived["human_root_height"])))
                max_gravity_error = float(
                    np.max(np.abs(gravity - derived["human_gravity_projection"]))
                )
                if max_height_error > 1e-6:
                    raise ValueError(f"human_root_height mismatch: {max_height_error}")
                if max_gravity_error > 1e-5:
                    raise ValueError(f"human_gravity_projection mismatch: {max_gravity_error}")

                metadata_json = [_decode_string(x) for x in clips_group["metadata_json"][:]]
                motion_key_ids = [_decode_string(x) for x in clips_group["motion_key_id"][:]]
                if len(metadata_json) != clip_count or len(motion_key_ids) != clip_count:
                    raise ValueError("clip string index length mismatch")
                for text in metadata_json:
                    metadata = json.loads(text)
                    coord = str(metadata.get("canonical_coordinate_system"))
                    coordinate_counts[coord] = coordinate_counts.get(coord, 0) + 1
                    if coord != CANONICAL_COORDINATE_FRAME:
                        raise ValueError(f"non-canonical coordinate system: {coord}")

                beta_dims.add(int(shape_beta.shape[1]))
                pose_dims.add(int(pose.shape[1]))
                total_clip_count += clip_count
                total_frame_count += frame_count
                shard_report.update(
                    {
                        "status": "ok",
                        "clip_count": clip_count,
                        "frame_count": frame_count,
                        "beta_dim": int(shape_beta.shape[1]),
                        "max_height_error": max_height_error,
                        "max_gravity_error": max_gravity_error,
                    }
                )
        except Exception as exc:
            shard_report.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            errors.append(shard_report)
        shard_reports.append(shard_report)
        if progress_interval and (
            shard_index % progress_interval == 0 or shard_index == len(shard_paths)
        ):
            elapsed = max(time.monotonic() - start_time, 1e-9)
            print(
                "[formal_h5_validation] "
                f"{shard_index}/{len(shard_paths)} shards checked | "
                f"clips={total_clip_count} frames={total_frame_count} "
                f"errors={len(errors)} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest_clip_count = None if manifest is None else manifest.get("clip_count")
    manifest_frame_count = None if manifest is None else manifest.get("frame_count")
    if manifest is not None:
        if int(manifest_clip_count) != total_clip_count:
            errors.append(
                {
                    "scope": "manifest",
                    "error": f"manifest clip_count {manifest_clip_count} != actual {total_clip_count}",
                }
            )
        if int(manifest_frame_count) != total_frame_count:
            errors.append(
                {
                    "scope": "manifest",
                    "error": f"manifest frame_count {manifest_frame_count} != actual {total_frame_count}",
                }
            )
    if expected_clip_count is not None and total_clip_count != expected_clip_count:
        errors.append(
            {
                "scope": "formal_h5_root",
                "error": f"clip count {total_clip_count} != expected {expected_clip_count}",
            }
        )

    report = {
        "schema_version": "formal_h5_validation_report_v1",
        "formal_h5_root": str(formal_h5_root),
        "validation": {
            "shard_count": len(shard_paths),
            "clip_count": total_clip_count,
            "frame_count": total_frame_count,
            "manifest_clip_count": manifest_clip_count,
            "manifest_frame_count": manifest_frame_count,
            "expected_clip_count": expected_clip_count,
            "error_count": len(errors),
            "all_ok": len(errors) == 0,
        },
        "summary": {
            "beta_dims": sorted(beta_dims),
            "pose_dims": sorted(pose_dims),
            "coordinate_system_counts": dict(sorted(coordinate_counts.items())),
        },
        "shards": shard_reports,
        "errors": errors,
    }
    if report_json is not None:
        _write_json(Path(report_json), report)
    if report_md is not None:
        _write_text(Path(report_md), _render_report_md(report))
    return report


def _render_report_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Formal H5 校验报告",
        "",
        "## 总体结论",
        "",
        f"- shard 数量: {validation['shard_count']}",
        f"- clip 数量: {validation['clip_count']}",
        f"- frame 数量: {validation['frame_count']}",
        f"- manifest clip 数量: {validation['manifest_clip_count']}",
        f"- manifest frame 数量: {validation['manifest_frame_count']}",
        f"- 期望 clip 数量: {validation['expected_clip_count']}",
        f"- error_count: {validation['error_count']}",
        f"- all_ok: {validation['all_ok']}",
        "",
        "## Summary",
        "",
        f"- beta_dims: {report['summary']['beta_dims']}",
        f"- pose_dims: {report['summary']['pose_dims']}",
        f"- coordinate_system_counts: {report['summary']['coordinate_system_counts']}",
        "",
        "## Shards",
        "",
    ]
    for shard in report["shards"]:
        lines.append(f"- {shard}")
    if report["errors"]:
        lines.extend(["", "## 错误", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
    lines.append("")
    return "\n".join(lines)


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
