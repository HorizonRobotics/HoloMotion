from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from holosmpl.core.validation.formal_h5_check import check_formal_h5_root


def combine_formal_h5_roots(
    *,
    inputs: list[str | Path],
    output_root: str | Path,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    mode: str = "copy",
    overwrite: bool = False,
    progress_interval: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    input_roots = [Path(value) for value in inputs]
    output_root = Path(output_root)
    if not input_roots:
        raise ValueError("at least one --input formal_h5 root is required")
    mode = str(mode).lower().strip()
    if mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError(f"unsupported combine mode: {mode}")
    for root in input_roots:
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"missing formal_h5 manifest: {root / 'manifest.json'}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    shards_root = output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    combined_manifest: dict[str, Any] | None = None
    combined_shards: list[dict[str, Any]] = []
    combined_samples: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    input_summaries: list[dict[str, Any]] = []
    total_clips = 0
    total_frames = 0

    for input_idx, input_root in enumerate(input_roots):
        manifest = _read_json(input_root / "manifest.json")
        if combined_manifest is None:
            combined_manifest = dict(manifest)
        shard_id_map: dict[str, tuple[str, str]] = {}
        for old_shard_idx, shard in enumerate(manifest.get("shards") or []):
            old_shard_id = str(shard.get("shard_id") or f"shard_{old_shard_idx:06d}")
            old_rel = str(shard["path"])
            old_path = input_root / old_rel
            if not old_path.is_file():
                raise FileNotFoundError(old_path)
            new_shard_index = len(combined_shards)
            new_shard_id = f"shard_{new_shard_index:06d}"
            new_rel = f"shards/{new_shard_id}.h5"
            _materialize(old_path, output_root / new_rel, mode=mode)
            new_shard = dict(shard)
            new_shard["shard_id"] = new_shard_id
            new_shard["path"] = new_rel
            new_shard["source_formal_h5_root"] = str(input_root)
            new_shard["source_shard_id"] = old_shard_id
            new_shard["source_shard_path"] = old_rel
            new_shard["clip_start_index"] = len(combined_samples)
            new_shard["clip_end_index"] = len(combined_samples) + int(
                new_shard.get("clip_count", 0)
            )
            combined_shards.append(new_shard)
            shard_id_map[old_shard_id] = (new_shard_id, new_rel)
            total_frames += int(new_shard.get("frame_count", 0))
            total_clips += int(new_shard.get("clip_count", 0))

        for sample in manifest.get("samples") or []:
            clip_id = str(sample["clip_id"])
            if clip_id in seen_clip_ids:
                raise ValueError(f"duplicate clip_id across formal_h5 roots: {clip_id}")
            seen_clip_ids.add(clip_id)
            old_shard_id = str(sample.get("shard_id") or "shard_000000")
            if old_shard_id not in shard_id_map:
                raise KeyError(f"{input_root}: sample {clip_id} references {old_shard_id}")
            new_shard_id, new_rel = shard_id_map[old_shard_id]
            new_sample = dict(sample)
            new_sample["index"] = len(combined_samples)
            new_sample["shard_id"] = new_shard_id
            new_sample["shard_path"] = new_rel
            new_sample["source_formal_h5_root"] = str(input_root)
            combined_samples.append(new_sample)

        input_summaries.append(
            {
                "root": str(input_root),
                "clip_count": int(manifest.get("clip_count", 0)),
                "frame_count": int(manifest.get("frame_count", 0)),
                "shard_count": len(manifest.get("shards") or []),
            }
        )

    assert combined_manifest is not None
    combined_manifest["output_root"] = str(output_root)
    combined_manifest["clip_count"] = len(combined_samples)
    combined_manifest["frame_count"] = total_frames
    combined_manifest["shards"] = combined_shards
    combined_manifest["samples"] = combined_samples
    combined_manifest["errors"] = []
    combined_manifest["combined_from"] = {
        "schema_version": "formal_h5_roots_combined_v1",
        "mode": mode,
        "input_count": len(input_roots),
        "inputs": input_summaries,
    }
    _write_json(output_root / "manifest.json", combined_manifest)
    (output_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    if validate:
        report = check_formal_h5_root(
            output_root,
            expected_clip_count=len(combined_samples),
            report_json=report_json,
            report_md=report_md,
            progress_interval=progress_interval,
        )
        if not report["validation"].get("all_ok", False):
            raise RuntimeError(f"combined formal H5 validation failed: {report['validation']}")
        validation = report["validation"]
        summary = report["summary"]
    else:
        report = {
            "validation": {
                "all_ok": True,
                "skipped": True,
                "clip_count": len(combined_samples),
                "frame_count": total_frames,
            },
            "summary": {},
        }
        validation = report["validation"]
        summary = report["summary"]
    return {
        "manifest": combined_manifest,
        "validation": validation,
        "summary": summary,
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "input_summaries": input_summaries,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _materialize(src: Path, dst: Path, *, mode: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    if mode == "symlink":
        target = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        os.symlink(target, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
