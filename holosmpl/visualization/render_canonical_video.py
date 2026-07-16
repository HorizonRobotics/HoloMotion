from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from holosmpl.visualization.body_model import BodyModelCache
from holosmpl.visualization.canonical_loader import load_canonical_clip
from holosmpl.visualization.video_render import render_mesh_video


def render_canonical_video_root(
    *,
    canonical_root: str | Path,
    output_root: str | Path,
    smpl_models_root: str | Path,
    num_clips: int = 8,
    video_fps: float = 50.0,
    max_seconds: float = 10.0,
    width: int = 960,
    height: int = 720,
    floor_policy: str = "first5_min",
    camera_mode: str = "fixed_3quarter",
    batch_size: int = 32,
    seed: int = 20260703,
    layout: str = "nested",
    overwrite: bool = False,
) -> dict[str, Any]:
    canonical_root = Path(canonical_root)
    output_root = Path(output_root)
    smpl_models_root = Path(smpl_models_root)
    if not canonical_root.is_dir():
        raise FileNotFoundError(f"canonical_root does not exist: {canonical_root}")
    clip_paths = sorted((canonical_root / "clips").glob("*.npz"))
    if not clip_paths:
        raise FileNotFoundError(f"no canonical clips found under {canonical_root / 'clips'}")
    if layout not in {"nested", "flat"}:
        raise ValueError(f"layout must be nested or flat, got {layout}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    if layout == "nested":
        (output_root / "previews").mkdir(parents=True, exist_ok=True)
    else:
        (output_root / "videos").mkdir(parents=True, exist_ok=True)
        (output_root / "thumbnails").mkdir(parents=True, exist_ok=True)
        (output_root / "per_clip_config").mkdir(parents=True, exist_ok=True)

    selected_paths = _select_paths(clip_paths, num_clips=num_clips, seed=seed)
    body_cache = BodyModelCache(smpl_models_root)
    samples: list[dict[str, Any]] = []
    rendered_count = 0
    error_count = 0

    for clip_path in selected_paths:
        sample: dict[str, Any] = {
            "clip_path": str(clip_path),
            "status": "pending",
        }
        try:
            clip = load_canonical_clip(clip_path)
            if layout == "nested":
                preview_dir = output_root / "previews" / clip.clip_id
                video_path = preview_dir / "mesh_preview.mp4"
                thumbnail_dir = preview_dir
                config_path = preview_dir / "render_config.json"
                thumbnail_prefix = ""
            else:
                video_path = output_root / "videos" / f"{clip.clip_id}.mp4"
                thumbnail_dir = output_root / "thumbnails"
                config_path = output_root / "per_clip_config" / f"{clip.clip_id}.json"
                thumbnail_prefix = f"{clip.clip_id}__"
            result = render_mesh_video(
                clip=clip,
                body_cache=body_cache,
                output_path=video_path,
                thumbnail_dir=thumbnail_dir,
                thumbnail_prefix=thumbnail_prefix,
                video_fps=video_fps,
                max_seconds=max_seconds,
                width=width,
                height=height,
                floor_policy=floor_policy,
                camera_mode=camera_mode,
                batch_size=batch_size,
            )
            render_config = {
                "clip_id": clip.clip_id,
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
                "layout": layout,
            }
            _write_json(config_path, render_config)
            sample.update({"status": "ok", **render_config})
            rendered_count += 1
        except Exception as exc:
            sample.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            error_count += 1
        samples.append(sample)

    report = {
        "schema_version": "canonical_video_render_report_v1",
        "canonical_root": str(canonical_root),
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
        "layout": layout,
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
        "# Canonical SMPL Mesh 视频预览报告",
        "",
        "## 总体结论",
        "",
        f"- canonical_root: `{report['canonical_root']}`",
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
            lines.append(f"- thumbnails: `{sample.get('thumbnails')}`")
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
