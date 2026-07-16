from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from holosmpl.visualization.body_model import BodyModelCache
from holosmpl.visualization.formal_loader import load_formal_clip_as_render_clip
from holosmpl.visualization.video_render import render_mesh_video


def render_formal_video_root(
    *,
    formal_root: str | Path,
    output_root: str | Path,
    smpl_models_root: str | Path,
    num_clips: int = 20,
    video_fps: float = 50.0,
    max_seconds: float = 4.0,
    width: int = 960,
    height: int = 720,
    floor_policy: str = "first5_min",
    camera_mode: str = "fixed_3quarter",
    batch_size: int = 32,
    seed: int = 20260703,
    overwrite: bool = False,
) -> dict[str, Any]:
    formal_root = Path(formal_root)
    output_root = Path(output_root)
    smpl_models_root = Path(smpl_models_root)
    clips_root = formal_root / "clips"
    if not formal_root.is_dir():
        raise FileNotFoundError(f"formal_root does not exist: {formal_root}")
    if not clips_root.is_dir():
        raise FileNotFoundError(f"formal clips dir does not exist: {clips_root}")
    clip_paths = sorted(clips_root.glob("*.npz"))
    if not clip_paths:
        raise FileNotFoundError(f"no formal clips found under {clips_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    (output_root / "thumbnails").mkdir(parents=True, exist_ok=True)
    (output_root / "per_clip_config").mkdir(parents=True, exist_ok=True)

    selected_paths = _select_paths(clip_paths, num_clips=num_clips, seed=seed)
    body_cache = BodyModelCache(smpl_models_root)
    samples: list[dict[str, Any]] = []
    rendered_count = 0
    error_count = 0

    for clip_path in selected_paths:
        sample: dict[str, Any] = {"clip_path": str(clip_path), "status": "pending"}
        try:
            clip = load_formal_clip_as_render_clip(clip_path)
            video_path = output_root / "videos" / f"{clip.clip_id}.mp4"
            thumbnail_dir = output_root / "thumbnails"
            config_path = output_root / "per_clip_config" / f"{clip.clip_id}.json"
            result = render_mesh_video(
                clip=clip,
                body_cache=body_cache,
                output_path=video_path,
                thumbnail_dir=thumbnail_dir,
                thumbnail_prefix=f"{clip.clip_id}__",
                video_fps=video_fps,
                max_seconds=max_seconds,
                width=width,
                height=height,
                floor_policy=floor_policy,
                camera_mode=camera_mode,
                batch_size=batch_size,
                label="Formal SMPL mesh",
            )
            render_config = {
                "clip_id": clip.clip_id,
                "formal_relative_path": clip.path.relative_to(formal_root).as_posix(),
                "source_relative_path": clip.metadata.get("source_relative_path"),
                "video_path": result.video_path.relative_to(output_root).as_posix(),
                "thumbnails": {
                    key: value.relative_to(output_root).as_posix()
                    for key, value in result.thumbnails.items()
                },
                "frame_count": clip.frame_count,
                "rendered_frame_count": len(result.frame_indices),
                "rendered_frame_start": result.frame_indices[0],
                "rendered_frame_end": result.frame_indices[-1],
                "video_fps": result.video_fps,
                "width": result.width,
                "height": result.height,
                "floor_policy": result.floor_policy,
                "floor_z": result.floor_z,
                "floor_reference_lowest_before_shift": result.floor_reference_lowest_before_shift,
                "floor_reference_lowest_after_shift": result.floor_reference_lowest_after_shift,
                "rendered_lowest_vertex_before_shift": result.lowest_vertex_before_shift,
                "rendered_lowest_vertex_after_shift": result.lowest_vertex_after_shift,
                "camera_mode": result.camera_mode,
                "layout": "flat",
            }
            _write_json(config_path, render_config)
            sample.update({"status": "ok", **render_config})
            rendered_count += 1
        except Exception as exc:
            sample.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            error_count += 1
        samples.append(sample)

    report = {
        "schema_version": "formal_video_render_report_v1",
        "formal_root": str(formal_root),
        "output_root": str(output_root),
        "smpl_models_root": str(smpl_models_root),
        "seed": int(seed),
        "num_clips_requested": int(num_clips),
        "video_fps": float(video_fps),
        "max_seconds": float(max_seconds),
        "width": int(width),
        "height": int(height),
        "floor_policy": floor_policy,
        "camera_mode": camera_mode,
        "layout": "flat",
        "validation": {
            "available_clip_count": len(clip_paths),
            "selected_clip_count": len(selected_paths),
            "rendered_clip_count": rendered_count,
            "error_count": error_count,
            "all_ok": error_count == 0,
        },
        "samples": samples,
    }
    _write_json(output_root / "video_report.json", report)
    _write_text(output_root / "video_report.md", _render_report_md(report))
    return report


def _select_paths(clip_paths: list[Path], *, num_clips: int, seed: int) -> list[Path]:
    if num_clips <= 0:
        raise ValueError(f"num_clips must be positive, got {num_clips}")
    if len(clip_paths) <= num_clips:
        return clip_paths
    rng = random.Random(seed)
    return sorted(rng.sample(clip_paths, num_clips))


def _render_report_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Formal SMPL Mesh 视频预览报告",
        "",
        "## 总体结论",
        "",
        f"- formal_root: `{report['formal_root']}`",
        f"- output_root: `{report['output_root']}`",
        f"- selected_clip_count: {validation['selected_clip_count']}",
        f"- rendered_clip_count: {validation['rendered_clip_count']}",
        f"- error_count: {validation['error_count']}",
        f"- all_ok: {validation['all_ok']}",
        f"- video_fps: {report['video_fps']}",
        f"- max_seconds: {report['max_seconds']}",
        f"- floor_policy: {report['floor_policy']}",
        f"- camera_mode: {report['camera_mode']}",
        f"- layout: {report['layout']}",
        "",
        "## Clips",
        "",
    ]
    for sample in report["samples"]:
        title = sample.get("clip_id") or Path(str(sample.get("clip_path", "unknown"))).stem
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- status: {sample['status']}")
        if sample["status"] == "ok":
            lines.append(f"- source: `{sample.get('source_relative_path')}`")
            lines.append(f"- video: `{sample.get('video_path')}`")
            lines.append(f"- rendered_frame_count: {sample.get('rendered_frame_count')}")
            lines.append(
                f"- rendered_frame_range: {sample.get('rendered_frame_start')} -> {sample.get('rendered_frame_end')}"
            )
            lines.append(f"- floor_z: {sample.get('floor_z')}")
            lines.append(
                f"- floor_reference_lowest_before/after: {sample.get('floor_reference_lowest_before_shift')} / {sample.get('floor_reference_lowest_after_shift')}"
            )
            lines.append(
                f"- rendered_lowest_vertex_before/after: {sample.get('rendered_lowest_vertex_before_shift')} / {sample.get('rendered_lowest_vertex_after_shift')}"
            )
        else:
            lines.append(f"- error: `{sample.get('error')}`")
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
