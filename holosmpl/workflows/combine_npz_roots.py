from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from holosmpl.core.validation.canonical_check import check_canonical_root
from holosmpl.core.validation.formal_check import check_formal_root


COMBINE_MODES = {"symlink", "hardlink", "copy"}


def combine_canonical_roots(
    *,
    inputs: list[str | Path],
    output_root: str | Path,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    target_fps: float = 50.0,
    mode: str = "symlink",
    overwrite: bool = False,
    progress_interval: int | None = None,
) -> dict[str, Any]:
    input_roots = _validate_inputs(inputs, expected="canonical")
    output_root = _prepare_output(output_root, overwrite=overwrite)
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    combined_samples: list[dict[str, Any]] = []
    combined_rejected: list[dict[str, Any]] = []
    combined_errors: list[dict[str, Any]] = []
    input_summaries: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    manifest_template: dict[str, Any] | None = None

    for input_idx, input_root in enumerate(input_roots):
        manifest = _read_json(input_root / "manifest.json")
        rejected_manifest = (
            _read_json(input_root / "rejected_manifest.json")
            if (input_root / "rejected_manifest.json").is_file()
            else {"rejected_samples": []}
        )
        if manifest_template is None:
            manifest_template = dict(manifest)
        for sample in manifest.get("samples") or []:
            clip_id = str(sample["clip_id"])
            if clip_id in seen_clip_ids:
                raise ValueError(f"duplicate canonical clip_id across inputs: {clip_id}")
            seen_clip_ids.add(clip_id)
            src_rel = str(sample["canonical_path"])
            src = input_root / src_rel
            dst_rel = f"clips/{Path(src_rel).name}"
            dst = output_root / dst_rel
            _materialize(src, dst, mode=mode)
            new_sample = dict(sample)
            new_sample["index"] = len(combined_samples)
            new_sample["canonical_path"] = dst_rel
            new_sample["source_canonical_root"] = str(input_root)
            new_sample["source_canonical_relative_path"] = src_rel
            combined_samples.append(new_sample)
        for sample in rejected_manifest.get("rejected_samples") or []:
            new_sample = dict(sample)
            new_sample["index"] = len(combined_rejected)
            new_sample["source_canonical_root"] = str(input_root)
            combined_rejected.append(new_sample)
        for sample in manifest.get("errors") or []:
            new_sample = dict(sample)
            new_sample["index"] = len(combined_errors)
            new_sample["source_canonical_root"] = str(input_root)
            combined_errors.append(new_sample)
        input_summaries.append(
            {
                "root": str(input_root),
                "source_count": int(manifest.get("source_count", 0)),
                "total_source_count": manifest.get("total_source_count"),
                "sample_count": int(manifest.get("sample_count", 0)),
                "rejected_count": int(manifest.get("rejected_count", 0)),
                "error_count": int(manifest.get("error_count", 0)),
            }
        )

    assert manifest_template is not None
    source_count = sum(item["source_count"] for item in input_summaries)
    total_source_values = {
        int(item["total_source_count"])
        for item in input_summaries
        if item.get("total_source_count") is not None
    }
    total_source_count = (
        total_source_values.pop() if len(total_source_values) == 1 else source_count
    )
    manifest = dict(manifest_template)
    manifest.update(
        {
            "output_root": str(output_root),
            "source_count": int(source_count),
            "total_source_count": int(total_source_count),
            "sample_count": len(combined_samples),
            "rejected_count": len(combined_rejected),
            "error_count": len(combined_errors),
            "shard_index": None,
            "shard_count": len(input_roots),
            "samples": combined_samples,
            "rejected_samples": combined_rejected,
            "errors": combined_errors,
            "combined_from": {
                "schema_version": "canonical_roots_combined_v1",
                "mode": mode,
                "input_count": len(input_roots),
                "inputs": input_summaries,
            },
        }
    )
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "rejected_manifest.json",
        {
            "schema_version": "canonical_rejected_manifest_v1",
            "dataset": manifest.get("dataset"),
            "input_root": manifest.get("input_root"),
            "output_root": str(output_root),
            "source_count": int(source_count),
            "total_source_count": int(total_source_count),
            "rejected_count": len(combined_rejected),
            "rejected_samples": combined_rejected,
            "combined_from": manifest["combined_from"],
        },
    )
    (output_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    report = check_canonical_root(
        output_root,
        expected_dataset=str(manifest.get("dataset")),
        expected_count=len(combined_samples),
        target_fps=target_fps,
        report_json=report_json,
        report_md=report_md,
        progress_interval=progress_interval,
    )
    if not report["validation"].get("all_ok", False):
        raise RuntimeError(f"combined canonical validation failed: {report['validation']}")
    return _result(manifest, report, report_json, report_md, input_summaries)


def combine_formal_npz_roots(
    *,
    inputs: list[str | Path],
    output_root: str | Path,
    report_json: str | Path | None = None,
    report_md: str | Path | None = None,
    mode: str = "symlink",
    overwrite: bool = False,
    progress_interval: int | None = None,
) -> dict[str, Any]:
    input_roots = _validate_inputs(inputs, expected="formal_npz")
    output_root = _prepare_output(output_root, overwrite=overwrite)
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    combined_samples: list[dict[str, Any]] = []
    combined_errors: list[dict[str, Any]] = []
    input_summaries: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    manifest_template: dict[str, Any] | None = None

    for input_root in input_roots:
        manifest = _read_json(input_root / "manifest.json")
        if manifest_template is None:
            manifest_template = dict(manifest)
        for sample in manifest.get("samples") or []:
            clip_id = str(sample["clip_id"])
            if clip_id in seen_clip_ids:
                raise ValueError(f"duplicate formal clip_id across inputs: {clip_id}")
            seen_clip_ids.add(clip_id)
            src_rel = str(sample["formal_path"])
            src = input_root / src_rel
            dst_rel = f"clips/{Path(src_rel).name}"
            dst = output_root / dst_rel
            _materialize(src, dst, mode=mode)
            new_sample = dict(sample)
            new_sample["index"] = len(combined_samples)
            new_sample["formal_path"] = dst_rel
            new_sample["source_formal_npz_root"] = str(input_root)
            new_sample["source_formal_relative_path"] = src_rel
            combined_samples.append(new_sample)
        for sample in manifest.get("errors") or []:
            new_sample = dict(sample)
            new_sample["index"] = len(combined_errors)
            new_sample["source_formal_npz_root"] = str(input_root)
            combined_errors.append(new_sample)
        input_summaries.append(
            {
                "root": str(input_root),
                "source_count": int(manifest.get("source_count", 0)),
                "sample_count": int(manifest.get("sample_count", 0)),
                "error_count": int(manifest.get("error_count", 0)),
            }
        )

    assert manifest_template is not None
    manifest = dict(manifest_template)
    manifest.update(
        {
            "output_root": str(output_root),
            "source_count": sum(item["source_count"] for item in input_summaries),
            "sample_count": len(combined_samples),
            "error_count": len(combined_errors),
            "samples": combined_samples,
            "errors": combined_errors,
            "combined_from": {
                "schema_version": "formal_npz_roots_combined_v1",
                "mode": mode,
                "input_count": len(input_roots),
                "inputs": input_summaries,
            },
        }
    )
    _write_json(output_root / "manifest.json", manifest)
    (output_root / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    report = check_formal_root(
        output_root,
        expected_dataset=str(manifest.get("dataset")),
        expected_count=len(combined_samples),
        report_json=report_json,
        report_md=report_md,
        progress_interval=progress_interval,
    )
    if not report["validation"].get("all_ok", False):
        raise RuntimeError(f"combined formal NPZ validation failed: {report['validation']}")
    return _result(manifest, report, report_json, report_md, input_summaries)


def _validate_inputs(inputs: list[str | Path], *, expected: str) -> list[Path]:
    roots = [Path(value) for value in inputs]
    if not roots:
        raise ValueError(f"at least one --input {expected} root is required")
    for root in roots:
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"missing manifest: {root / 'manifest.json'}")
        if not (root / "clips").is_dir():
            raise FileNotFoundError(f"missing clips dir: {root / 'clips'}")
    return roots


def _prepare_output(output_root: str | Path, *, overwrite: bool) -> Path:
    output = Path(output_root)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _materialize(src: Path, dst: Path, *, mode: str) -> None:
    mode = str(mode).lower().strip()
    if mode not in COMBINE_MODES:
        raise ValueError(f"unsupported combine mode: {mode}")
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


def _result(
    manifest: dict[str, Any],
    report: dict[str, Any],
    report_json: str | Path | None,
    report_md: str | Path | None,
    input_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest": manifest,
        "validation": report["validation"],
        "summary": report["summary"],
        "report_json": None if report_json is None else str(report_json),
        "report_md": None if report_md is None else str(report_md),
        "input_summaries": input_summaries,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
