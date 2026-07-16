from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from holosmpl.visualization.body_model import BodyModelCache
from holosmpl.visualization.canonical_loader import (
    evenly_spaced_indices,
    load_canonical_clip,
)
from holosmpl.visualization.geometry import compute_geometry_summary
from holosmpl.visualization.render import render_contact_sheet
from holosmpl.visualization.report import write_visualization_reports


def visualize_canonical_root(
    *,
    canonical_root: str | Path,
    output_root: str | Path,
    smpl_models_root: str | Path,
    num_clips: int = 8,
    num_frames: int = 8,
    geometry_max_frames: int = 120,
    batch_size: int = 32,
    seed: int = 20260703,
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
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    (output_root / "previews").mkdir(parents=True, exist_ok=True)

    selected_paths = _select_paths(clip_paths, num_clips=num_clips, seed=seed)
    body_cache = BodyModelCache(smpl_models_root)
    samples: list[dict[str, Any]] = []
    error_count = 0
    rendered_count = 0
    warning_count = 0

    for clip_path in selected_paths:
        sample: dict[str, Any] = {
            "clip_path": str(clip_path),
            "status": "pending",
        }
        try:
            clip = load_canonical_clip(clip_path)
            frame_indices = evenly_spaced_indices(clip.frame_count, num_frames)
            visual_output = body_cache.forward_clip_frames(
                clip,
                frame_indices,
                batch_size=batch_size,
            )
            preview_dir = output_root / "previews" / clip.clip_id
            contact_sheet = preview_dir / "contact_sheet.png"
            render_contact_sheet(
                clip=clip,
                body_output=visual_output,
                frame_indices=frame_indices,
                output_path=contact_sheet,
            )
            geometry = compute_geometry_summary(
                clip,
                body_cache,
                max_frames=geometry_max_frames,
                batch_size=batch_size,
            )
            if geometry["warnings"]:
                warning_count += 1
            rendered_count += 1
            sample.update(
                {
                    "status": "ok",
                    "clip_id": clip.clip_id,
                    "source_relative_path": clip.metadata.get("source_relative_path"),
                    "frame_count": clip.frame_count,
                    "visualized_frame_indices": frame_indices,
                    "contact_sheet": contact_sheet.relative_to(output_root).as_posix(),
                    "geometry": geometry,
                }
            )
        except Exception as exc:
            error_count += 1
            sample.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        samples.append(sample)

    report = {
        "schema_version": "canonical_visualization_report_v1",
        "canonical_root": str(canonical_root),
        "output_root": str(output_root),
        "smpl_models_root": str(smpl_models_root),
        "seed": int(seed),
        "num_clips_requested": int(num_clips),
        "num_frames_requested": int(num_frames),
        "geometry_max_frames": int(geometry_max_frames),
        "batch_size": int(batch_size),
        "validation": {
            "available_clip_count": len(clip_paths),
            "selected_clip_count": len(selected_paths),
            "rendered_clip_count": rendered_count,
            "warning_clip_count": warning_count,
            "error_count": error_count,
            "all_ok": error_count == 0,
        },
        "samples": samples,
    }
    write_visualization_reports(output_root, report)
    return report


def resolve_smpl_models_root(
    *,
    explicit_root: str | Path | None,
    env_config: str | Path | None,
    project_root: str | Path,
) -> Path:
    from holosmpl.core.config import load_config

    if explicit_root is not None:
        return Path(explicit_root).expanduser()

    config_paths: list[Path] = []
    if env_config is not None:
        config_paths.append(Path(env_config))
    default_apex = Path(project_root) / "configs" / "env" / "apex.yaml"
    if default_apex.exists():
        config_paths.append(default_apex)

    for config_path in config_paths:
        config = load_config(config_path)
        value = (config.get("paths") or {}).get("smpl_models_root")
        if value:
            return Path(str(value)).expanduser()

    raise ValueError(
        "SMPL models root is not configured. Pass --smpl-models-root or --env-config."
    )


def _select_paths(clip_paths: list[Path], *, num_clips: int, seed: int) -> list[Path]:
    if num_clips <= 0:
        raise ValueError(f"num_clips must be positive, got {num_clips}")
    if len(clip_paths) <= num_clips:
        return clip_paths
    rng = random.Random(seed)
    return sorted(rng.sample(clip_paths, num_clips))
