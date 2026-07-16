from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
)


REQUIRED_CANONICAL_FIELDS = {
    "root_orient",
    "pose_body",
    "trans",
    "betas",
    "gender",
    "source_fps",
    "target_fps",
    "metadata",
}
FORBIDDEN_CANONICAL_FIELDS = {
    "human_pose_aa",
    "human_shape_beta",
    "human_root_trans",
    "human_root_height",
    "human_gravity_projection",
}
ROBOT_LIKE_KEYWORDS = (
    "robot",
    "dof",
    "torque",
    "actuator",
    "motor",
    "qpos",
    "qvel",
    "rigid_body",
    "root_states",
)


def check_canonical_root(
    canonical_root: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_count: int | None = None,
    target_fps: float = 50.0,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    progress_interval: int | None = None,
) -> dict[str, object]:
    import numpy as np

    canonical_root = Path(canonical_root)
    clips_root = canonical_root / "clips"
    manifest_path = canonical_root / "manifest.json"
    clip_paths = sorted(clips_root.glob("*.npz"))
    samples = []
    errors = []
    pose_dim_counts: Counter[int] = Counter()
    fps_counts: Counter[str] = Counter()
    coordinate_counts: Counter[str] = Counter()
    progress_interval = max(1, int(progress_interval or 0)) if progress_interval else 0
    start_time = time.monotonic()

    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if manifest is None:
        errors.append({"scope": "manifest", "error": f"missing {manifest_path}"})

    for index, clip_path in enumerate(clip_paths, start=1):
        sample = {"clip_path": str(clip_path.relative_to(canonical_root))}
        try:
            with np.load(clip_path, allow_pickle=False) as data:
                fields = set(data.files)
                missing = sorted(REQUIRED_CANONICAL_FIELDS - fields)
                forbidden = sorted(fields & FORBIDDEN_CANONICAL_FIELDS)
                robot_like = [
                    field
                    for field in fields
                    if any(keyword in field.lower() for keyword in ROBOT_LIKE_KEYWORDS)
                ]
                if missing:
                    raise ValueError(f"missing fields: {missing}")
                if forbidden:
                    raise ValueError(f"forbidden formal fields in canonical: {forbidden}")
                if robot_like:
                    raise ValueError(f"robot-like fields in canonical: {robot_like}")

                root_orient = data["root_orient"]
                pose_body = data["pose_body"]
                trans = data["trans"]
                betas = data["betas"]
                source_fps = float(data["source_fps"])
                actual_target_fps = float(data["target_fps"])
                metadata = json.loads(str(data["metadata"].item()))

                _check_array("root_orient", root_orient, ndim=2, second_dim=3)
                _check_array("pose_body", pose_body, ndim=2, second_dim_one_of=(63, 69))
                _check_array("trans", trans, ndim=2, second_dim=3)
                _check_array("betas", betas, ndim=1)
                frame_count = int(root_orient.shape[0])
                if pose_body.shape[0] != frame_count or trans.shape[0] != frame_count:
                    raise ValueError("root_orient, pose_body, and trans frame counts differ")
                for name, array in {
                    "root_orient": root_orient,
                    "pose_body": pose_body,
                    "trans": trans,
                    "betas": betas,
                }.items():
                    if array.dtype != np.float32:
                        raise ValueError(f"{name} dtype must be float32, got {array.dtype}")
                    if not np.isfinite(array).all():
                        raise ValueError(f"{name} contains NaN or Inf")
                if abs(actual_target_fps - target_fps) > 1e-5:
                    raise ValueError(f"target_fps must be {target_fps}, got {actual_target_fps}")
                if metadata.get("target_fps") != target_fps:
                    raise ValueError(f"metadata target_fps must be {target_fps}")
                if metadata.get("canonical_coordinate_system") != CANONICAL_COORDINATE_FRAME:
                    raise ValueError("canonical_coordinate_system is not project canonical")
                if metadata.get("up_axis") != CANONICAL_UP_AXIS:
                    raise ValueError("up_axis is not canonical")
                if metadata.get("unit") != CANONICAL_UNIT:
                    raise ValueError("unit is not canonical")
                if metadata.get("slice_policy") != "none":
                    raise ValueError("slice_policy must be none")
                if expected_dataset and metadata.get("dataset") != expected_dataset:
                    raise ValueError(f"dataset must be {expected_dataset}")
                expected_frames = int(round(int(metadata["source_frame_count"]) * target_fps / source_fps))
                expected_frames = max(1, expected_frames)
                if frame_count != expected_frames:
                    raise ValueError(
                        f"frame count mismatch: got {frame_count}, expected {expected_frames}"
                    )
                if int(metadata["canonical_frame_count"]) != frame_count:
                    raise ValueError("metadata canonical_frame_count mismatch")

                pose_dim = int(pose_body.shape[1])
                pose_dim_counts[pose_dim] += 1
                fps_counts[f"{source_fps:g}->{actual_target_fps:g}"] += 1
                coordinate_counts[str(metadata["canonical_coordinate_system"])] += 1
                sample.update(
                    {
                        "status": "ok",
                        "frame_count": frame_count,
                        "source_frame_count": int(metadata["source_frame_count"]),
                        "source_fps": source_fps,
                        "target_fps": actual_target_fps,
                        "pose_body_dim": pose_dim,
                        "source_path": metadata.get("source_path"),
                        "resample_policy": metadata.get("resample_policy"),
                        "coordinate_transform": metadata.get("coordinate_transform"),
                    }
                )
        except Exception as exc:
            sample.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            errors.append(sample)
        samples.append(sample)
        if progress_interval and (index % progress_interval == 0 or index == len(clip_paths)):
            elapsed = max(time.monotonic() - start_time, 1e-9)
            print(
                "[canonical_validation] "
                f"{index}/{len(clip_paths)} checked | errors={len(errors)} | "
                f"{index / elapsed:.2f} clips/s | elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest_count = None
    manifest_source_count = None
    manifest_rejected_count = None
    manifest_error_count = None
    if manifest is not None:
        manifest_count = len(manifest.get("samples", []))
        manifest_source_count = manifest.get("source_count")
        manifest_rejected_count = manifest.get("rejected_count")
        manifest_error_count = manifest.get("error_count")
        if manifest_count != len(clip_paths):
            errors.append(
                {
                    "scope": "manifest",
                    "error": f"manifest sample count {manifest_count} != clip count {len(clip_paths)}",
                }
            )
    if expected_count is not None and len(clip_paths) != expected_count:
        errors.append(
            {
                "scope": "canonical_root",
                "error": f"clip count {len(clip_paths)} != expected_count {expected_count}",
            }
        )

    report = {
        "schema_version": "canonical_validation_report_v1",
        "canonical_root": str(canonical_root),
        "validation": {
            "clip_count": len(clip_paths),
            "manifest_sample_count": manifest_count,
            "manifest_source_count": manifest_source_count,
            "manifest_rejected_count": manifest_rejected_count,
            "manifest_error_count": manifest_error_count,
            "expected_count": expected_count,
            "error_count": len(errors),
            "all_ok": len(errors) == 0,
        },
        "summary": {
            "pose_body_dim_counts": dict(sorted(pose_dim_counts.items())),
            "fps_resample_counts": dict(sorted(fps_counts.items())),
            "coordinate_system_counts": dict(sorted(coordinate_counts.items())),
        },
        "errors": errors,
        "samples": samples,
    }
    if report_json is not None:
        _write_json(Path(report_json), report)
    if report_md is not None:
        _write_text(Path(report_md), _render_report_md(report))
    return report


def _check_array(
    name: str,
    array: Any,
    *,
    ndim: int,
    second_dim: int | None = None,
    second_dim_one_of: tuple[int, ...] | None = None,
) -> None:
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, got {array.ndim}")
    if second_dim is not None and array.shape[1] != second_dim:
        raise ValueError(f"{name} second dim must be {second_dim}, got {array.shape}")
    if second_dim_one_of is not None and array.shape[1] not in second_dim_one_of:
        raise ValueError(f"{name} second dim must be one of {second_dim_one_of}, got {array.shape}")


def _render_report_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Canonical 校验报告",
        "",
        "## 总体结论",
        "",
        f"- clip 数量: {validation['clip_count']}",
        f"- manifest 样本数: {validation['manifest_sample_count']}",
        f"- manifest 源文件数: {validation['manifest_source_count']}",
        f"- manifest 拒绝数: {validation['manifest_rejected_count']}",
        f"- manifest 转换异常数: {validation['manifest_error_count']}",
        f"- 期望数量: {validation['expected_count']}",
        f"- error_count: {validation['error_count']}",
        f"- all_ok: {validation['all_ok']}",
        "",
        "## Pose Body 维度",
        "",
    ]
    for key, value in report["summary"]["pose_body_dim_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## FPS 重采样", ""])
    for key, value in report["summary"]["fps_resample_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 坐标系", ""])
    for key, value in report["summary"]["coordinate_system_counts"].items():
        lines.append(f"- {key}: {value}")
    if report["errors"]:
        lines.extend(["", "## 错误", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
