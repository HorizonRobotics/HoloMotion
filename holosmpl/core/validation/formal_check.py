from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from holosmpl.core.processing.derived_fields import derive_formal_fields
from holosmpl.core.schema.canonical import (
    CANONICAL_COORDINATE_FRAME,
    CANONICAL_UNIT,
    CANONICAL_UP_AXIS,
)


REQUIRED_FORMAL_FIELDS = {
    "human_pose_aa",
    "human_shape_beta",
    "human_root_trans",
    "human_root_height",
    "human_gravity_projection",
    "metadata",
}
FORBIDDEN_FORMAL_FIELDS = {
    "root_orient",
    "pose_body",
    "trans",
    "betas",
    "source_fps",
    "target_fps",
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


def check_formal_root(
    formal_root: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_count: int | None = None,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    progress_interval: int | None = None,
) -> dict[str, object]:
    import numpy as np

    formal_root = Path(formal_root)
    clips_root = formal_root / "clips"
    manifest_path = formal_root / "manifest.json"
    clip_paths = sorted(clips_root.glob("*.npz"))
    samples = []
    errors = []
    beta_dim_counts: Counter[int] = Counter()
    frame_count_min = None
    frame_count_max = None
    coordinate_counts: Counter[str] = Counter()
    pose_policy_counts: Counter[str] = Counter()
    progress_interval = max(1, int(progress_interval or 0)) if progress_interval else 0
    start_time = time.monotonic()

    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    manifest_count = None
    if manifest is None:
        errors.append({"scope": "manifest", "error": f"missing {manifest_path}"})

    for index, clip_path in enumerate(clip_paths, start=1):
        sample = {"clip_path": str(clip_path.relative_to(formal_root))}
        try:
            with np.load(clip_path, allow_pickle=False) as data:
                fields = set(data.files)
                missing = sorted(REQUIRED_FORMAL_FIELDS - fields)
                forbidden = sorted(fields & FORBIDDEN_FORMAL_FIELDS)
                robot_like = [
                    field
                    for field in fields
                    if any(keyword in field.lower() for keyword in ROBOT_LIKE_KEYWORDS)
                ]
                if missing:
                    raise ValueError(f"missing fields: {missing}")
                if forbidden:
                    raise ValueError(f"canonical fields leaked into formal: {forbidden}")
                if robot_like:
                    raise ValueError(f"robot-like fields in formal: {robot_like}")

                human_pose_aa = data["human_pose_aa"]
                human_shape_beta = data["human_shape_beta"]
                human_root_trans = data["human_root_trans"]
                human_root_height = data["human_root_height"]
                human_gravity_projection = data["human_gravity_projection"]
                metadata = json.loads(str(data["metadata"].item()))

                _check_array("human_pose_aa", human_pose_aa, ndim=2, second_dim=72)
                _check_array("human_shape_beta", human_shape_beta, ndim=1)
                _check_array("human_root_trans", human_root_trans, ndim=2, second_dim=3)
                _check_array("human_root_height", human_root_height, ndim=2, second_dim=1)
                _check_array(
                    "human_gravity_projection",
                    human_gravity_projection,
                    ndim=2,
                    second_dim=3,
                )

                frame_count = int(human_pose_aa.shape[0])
                if frame_count <= 0:
                    raise ValueError("formal clip has zero frames")
                for name, array in {
                    "human_root_trans": human_root_trans,
                    "human_root_height": human_root_height,
                    "human_gravity_projection": human_gravity_projection,
                }.items():
                    if int(array.shape[0]) != frame_count:
                        raise ValueError(
                            f"{name} frame count {array.shape[0]} != pose frame count {frame_count}"
                        )
                for name, array in {
                    "human_pose_aa": human_pose_aa,
                    "human_shape_beta": human_shape_beta,
                    "human_root_trans": human_root_trans,
                    "human_root_height": human_root_height,
                    "human_gravity_projection": human_gravity_projection,
                }.items():
                    if array.dtype != np.float32:
                        raise ValueError(f"{name} dtype must be float32, got {array.dtype}")
                    if not np.isfinite(array).all():
                        raise ValueError(f"{name} contains NaN or Inf")

                if human_shape_beta.shape[0] <= 0:
                    raise ValueError("human_shape_beta must have positive beta dimension")

                derived = derive_formal_fields(human_pose_aa, human_root_trans)
                if not np.allclose(
                    human_root_height,
                    derived["human_root_height"],
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError("human_root_height does not match human_root_trans[:, 2:3]")
                if not np.allclose(
                    human_gravity_projection,
                    derived["human_gravity_projection"],
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise ValueError("human_gravity_projection does not match root orientation")

                if metadata.get("schema_version") != "formal_human_npz_v1":
                    raise ValueError("metadata schema_version must be formal_human_npz_v1")
                if expected_dataset and metadata.get("dataset") != expected_dataset:
                    raise ValueError(f"dataset must be {expected_dataset}")
                if metadata.get("canonical_coordinate_system") != CANONICAL_COORDINATE_FRAME:
                    raise ValueError("canonical_coordinate_system is not project canonical")
                if metadata.get("up_axis") != CANONICAL_UP_AXIS:
                    raise ValueError("up_axis is not canonical")
                if metadata.get("unit") != CANONICAL_UNIT:
                    raise ValueError("unit is not canonical")
                if metadata.get("slice_policy") != "none":
                    raise ValueError("slice_policy must be none")
                if int(metadata.get("frame_count", -1)) != frame_count:
                    raise ValueError("metadata frame_count mismatch")
                if metadata.get("formal_pose_72_policy") is None:
                    raise ValueError("metadata missing formal_pose_72_policy")

                beta_dim = int(human_shape_beta.shape[0])
                beta_dim_counts[beta_dim] += 1
                frame_count_min = frame_count if frame_count_min is None else min(frame_count_min, frame_count)
                frame_count_max = frame_count if frame_count_max is None else max(frame_count_max, frame_count)
                coordinate_counts[str(metadata["canonical_coordinate_system"])] += 1
                pose_policy_counts[str(metadata["formal_pose_72_policy"])] += 1
                sample.update(
                    {
                        "status": "ok",
                        "frame_count": frame_count,
                        "beta_dim": beta_dim,
                        "canonical_path": metadata.get("canonical_relative_path"),
                        "formal_pose_72_policy": metadata.get("formal_pose_72_policy"),
                    }
                )
        except Exception as exc:
            sample.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            errors.append(sample)
        samples.append(sample)
        if progress_interval and (index % progress_interval == 0 or index == len(clip_paths)):
            elapsed = max(time.monotonic() - start_time, 1e-9)
            print(
                "[formal_npz_validation] "
                f"{index}/{len(clip_paths)} checked | errors={len(errors)} | "
                f"{index / elapsed:.2f} clips/s | elapsed={elapsed:.1f}s",
                flush=True,
            )

    if manifest is not None:
        manifest_count = len(manifest.get("samples", []))
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
                "scope": "formal_root",
                "error": f"clip count {len(clip_paths)} != expected_count {expected_count}",
            }
        )

    report = {
        "schema_version": "formal_validation_report_v1",
        "formal_root": str(formal_root),
        "validation": {
            "clip_count": len(clip_paths),
            "manifest_sample_count": manifest_count,
            "expected_count": expected_count,
            "error_count": len(errors),
            "all_ok": len(errors) == 0,
        },
        "summary": {
            "beta_dim_counts": dict(sorted(beta_dim_counts.items())),
            "frame_count_min": frame_count_min,
            "frame_count_max": frame_count_max,
            "coordinate_system_counts": dict(sorted(coordinate_counts.items())),
            "formal_pose_72_policy_counts": dict(sorted(pose_policy_counts.items())),
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
) -> None:
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, got {array.ndim}")
    if second_dim is not None and array.shape[1] != second_dim:
        raise ValueError(f"{name} second dim must be {second_dim}, got {array.shape}")


def _render_report_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Formal NPZ 校验报告",
        "",
        "## 总体结论",
        "",
        f"- clip 数量: {validation['clip_count']}",
        f"- manifest 样本数: {validation['manifest_sample_count']}",
        f"- 期望数量: {validation['expected_count']}",
        f"- error_count: {validation['error_count']}",
        f"- all_ok: {validation['all_ok']}",
        "",
        "## Beta 维度",
        "",
    ]
    for key, value in report["summary"]["beta_dim_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Frame Count", ""])
    lines.append(f"- min: {report['summary']['frame_count_min']}")
    lines.append(f"- max: {report['summary']['frame_count_max']}")
    lines.extend(["", "## Pose-72 Policy", ""])
    for key, value in report["summary"]["formal_pose_72_policy_counts"].items():
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
