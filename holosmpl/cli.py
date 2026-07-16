from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from holosmpl.core.config import load_config
from holosmpl.sources import source_table
from holosmpl.workflows.inspect_smoke_data import inspect_smoke_data
from holosmpl.workflows.raw_to_holosmpl import raw_to_holosmpl
from holosmpl.core.schema.canonical import canonical_schema_summary
from holosmpl.core.schema.formal import formal_schema_summary
from holosmpl.core.schema.joints import smpl_joint_summary
from holosmpl.workflows.convert_dataset import convert_dataset_to_canonical
from holosmpl.workflows.convert_formal import convert_canonical_to_formal_npz
from holosmpl.workflows.combine_formal_h5 import combine_formal_h5_roots
from holosmpl.workflows.combine_npz_roots import (
    combine_canonical_roots,
    combine_formal_npz_roots,
)
from holosmpl.workflows.pack_formal_h5 import pack_formal_npz_to_h5
from holosmpl.visualization.render_canonical_video import render_canonical_video_root
from holosmpl.visualization.render_formal_video import render_formal_video_root
from holosmpl.visualization.visualize_canonical import (
    resolve_smpl_models_root,
    visualize_canonical_root,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLOSMPL_TARGET_FPS = 50.0


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_list_sources(args: argparse.Namespace) -> int:
    rows = source_table()
    if args.group:
        rows = [row for row in rows if row["group"] == args.group]
    if args.json:
        _print_json(rows)
        return 0
    print("Available HoloSMPL sources:")
    for row in rows:
        aliases = f" aliases={','.join(row['aliases'])}" if row["aliases"] else ""
        notes = f" | {row['notes']}" if row["notes"] else ""
        print(
            f"- {row['group']}/{row['key']}: {row['display_name']} "
            f"({row['input_hint']}){aliases}{notes}"
        )
    return 0


def cmd_schema(_: argparse.Namespace) -> int:
    _print_json(
        {
            "canonical": canonical_schema_summary(),
            "formal": formal_schema_summary(),
            "joints": smpl_joint_summary(),
        }
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan_path = PROJECT_ROOT / "docs" / "项目规划.md"
    if args.path:
        print(str(plan_path))
    else:
        print(plan_path.read_text(encoding="utf-8"))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.env_config) if args.env_config else {}
    checks: list[dict[str, Any]] = []
    for label, raw_path in (config.get("paths") or {}).items():
        path = Path(str(raw_path)).expanduser()
        checks.append(
            {
                "name": str(label),
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
        )
    _print_json(
        {
            "project_root": str(PROJECT_ROOT),
            "env_config": None if args.env_config is None else str(args.env_config),
            "checks": checks,
            "schema": {
                "canonical": canonical_schema_summary(),
                "formal": formal_schema_summary(),
            },
        }
    )
    return 0


def cmd_inspect_smoke(args: argparse.Namespace) -> int:
    report = inspect_smoke_data(
        smoke_root=args.smoke_root,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    _print_json(
        {
            "report_json": str(args.report_json),
            "report_md": str(args.report_md),
            "validation": report["validation"],
            "summary": report["summary"],
        }
    )
    return 0


def cmd_convert_canonical(args: argparse.Namespace) -> int:
    result = convert_dataset_to_canonical(
        dataset=args.dataset,
        input_root=args.input_root,
        output_root=args.output_root,
        target_fps=HOLOSMPL_TARGET_FPS,
        report_json=args.report_json,
        report_md=args.report_md,
        run_config_json=args.run_config_json,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        progress_interval=args.progress_interval,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    _print_json(
        {
            "dataset": args.dataset,
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "run_config_json": result["run_config_json"],
        }
    )
    return 0


def cmd_convert_formal_npz(args: argparse.Namespace) -> int:
    result = convert_canonical_to_formal_npz(
        canonical_root=args.canonical_root,
        output_root=args.output_root,
        report_json=args.report_json,
        report_md=args.report_md,
        run_config_json=args.run_config_json,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "canonical_root": str(args.canonical_root),
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "run_config_json": result["run_config_json"],
        }
    )
    return 0


def cmd_visualize_canonical(args: argparse.Namespace) -> int:
    smpl_models_root = resolve_smpl_models_root(
        explicit_root=args.smpl_models_root,
        env_config=args.env_config,
        project_root=PROJECT_ROOT,
    )
    report = visualize_canonical_root(
        canonical_root=args.canonical_root,
        output_root=args.output_root,
        smpl_models_root=smpl_models_root,
        num_clips=args.num_clips,
        num_frames=args.num_frames,
        geometry_max_frames=args.geometry_max_frames,
        batch_size=args.batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "canonical_root": str(args.canonical_root),
            "output_root": str(args.output_root),
            "smpl_models_root": str(smpl_models_root),
            "validation": report["validation"],
            "geometry_report_json": str(args.output_root / "geometry_report.json"),
            "geometry_report_md": str(args.output_root / "geometry_report.md"),
        }
    )
    return 0


def cmd_render_canonical_video(args: argparse.Namespace) -> int:
    smpl_models_root = resolve_smpl_models_root(
        explicit_root=args.smpl_models_root,
        env_config=args.env_config,
        project_root=PROJECT_ROOT,
    )
    report = render_canonical_video_root(
        canonical_root=args.canonical_root,
        output_root=args.output_root,
        smpl_models_root=smpl_models_root,
        num_clips=args.num_clips,
        video_fps=args.video_fps,
        max_seconds=args.max_seconds,
        width=args.width,
        height=args.height,
        floor_policy=args.floor_policy,
        camera_mode=args.camera_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        layout=args.layout,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "canonical_root": str(args.canonical_root),
            "output_root": str(args.output_root),
            "smpl_models_root": str(smpl_models_root),
            "validation": report["validation"],
            "video_report_json": str(args.output_root / "video_report.json"),
            "video_report_md": str(args.output_root / "video_report.md"),
        }
    )
    return 0


def cmd_render_formal_video(args: argparse.Namespace) -> int:
    smpl_models_root = resolve_smpl_models_root(
        explicit_root=args.smpl_models_root,
        env_config=args.env_config,
        project_root=PROJECT_ROOT,
    )
    report = render_formal_video_root(
        formal_root=args.formal_root,
        output_root=args.output_root,
        smpl_models_root=smpl_models_root,
        num_clips=args.num_clips,
        video_fps=args.video_fps,
        max_seconds=args.max_seconds,
        width=args.width,
        height=args.height,
        floor_policy=args.floor_policy,
        camera_mode=args.camera_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "formal_root": str(args.formal_root),
            "output_root": str(args.output_root),
            "smpl_models_root": str(smpl_models_root),
            "validation": report["validation"],
            "video_report_json": str(args.output_root / "video_report.json"),
            "video_report_md": str(args.output_root / "video_report.md"),
        }
    )
    return 0


def cmd_pack_formal_h5(args: argparse.Namespace) -> int:
    result = pack_formal_npz_to_h5(
        formal_npz_root=args.formal_npz_root,
        output_root=args.output_root,
        report_json=args.report_json,
        report_md=args.report_md,
        run_config_json=args.run_config_json,
        compression=args.compression,
        overwrite=args.overwrite,
        shard_target_gb=args.shard_target_gb,
        shard_target_bytes=args.shard_target_bytes,
        shard_target_mode=args.shard_target_mode,
        shard_target_clips=args.shard_target_clips,
        shard_target_frames=args.shard_target_frames,
        chunks_t=args.chunks_t,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "formal_npz_root": str(args.formal_npz_root),
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "run_config_json": result["run_config_json"],
        }
    )
    return 0


def cmd_retarget_holoretarget_h5(args: argparse.Namespace) -> int:
    from holomotion.src.training.data_production.robot_h5 import (
        retarget_holosmpl_h5_to_robot_h5,
    )

    manifest = retarget_holosmpl_h5_to_robot_h5(
        holosmpl_h5_root=args.holosmpl_h5_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        compression=args.compression,
        chunks_t=args.chunks_t,
        shard_target_frames=args.shard_target_frames,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "holosmpl_h5_root": str(args.holosmpl_h5_root),
            "output_root": str(args.output_root),
            "clip_count": len(manifest["clips"]),
            "shard_count": len(manifest["hdf5_shards"]),
            "array_names": manifest["array_names"],
        }
    )
    return 0


def cmd_combine_formal_h5_roots(args: argparse.Namespace) -> int:
    result = combine_formal_h5_roots(
        inputs=args.input,
        output_root=args.output_root,
        report_json=args.report_json,
        report_md=args.report_md,
        mode=args.mode,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "inputs": [str(value) for value in args.input],
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "input_summaries": result["input_summaries"],
        }
    )
    return 0


def cmd_combine_canonical_roots(args: argparse.Namespace) -> int:
    result = combine_canonical_roots(
        inputs=args.input,
        output_root=args.output_root,
        report_json=args.report_json,
        report_md=args.report_md,
        target_fps=args.target_fps,
        mode=args.mode,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "inputs": [str(value) for value in args.input],
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "input_summaries": result["input_summaries"],
        }
    )
    return 0


def cmd_combine_formal_npz_roots(args: argparse.Namespace) -> int:
    result = combine_formal_npz_roots(
        inputs=args.input,
        output_root=args.output_root,
        report_json=args.report_json,
        report_md=args.report_md,
        mode=args.mode,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "inputs": [str(value) for value in args.input],
            "output_root": str(args.output_root),
            "validation": result["validation"],
            "summary": result["summary"],
            "report_json": result["report_json"],
            "report_md": result["report_md"],
            "input_summaries": result["input_summaries"],
        }
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    result = raw_to_holosmpl(
        source=args.source,
        input_root=args.input_root,
        output_root=args.output_root,
        target_fps=HOLOSMPL_TARGET_FPS,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        progress_interval=args.progress_interval,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        write_formal_h5=not args.skip_h5,
        compression=args.compression,
        h5_shard_target_gb=args.shard_target_gb,
        h5_chunks_t=args.chunks_t,
    )
    _print_json(
        {
            "source": args.source,
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "canonical_root": result["canonical_root"],
            "formal_npz_root": result["formal_npz_root"],
            "formal_h5_root": result["formal_h5_root"],
            "canonical_summary": result["canonical"]["summary"],
            "formal_npz_summary": result["formal_npz"]["summary"],
            "formal_h5_summary": None
            if result["formal_h5"] is None
            else result["formal_h5"]["summary"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holosmpl")
    sub = parser.add_subparsers(dest="command", required=True)

    list_sources = sub.add_parser("list-sources", help="List supported raw data sources")
    list_sources.add_argument("--group", choices=("datasets", "devices"))
    list_sources.add_argument("--json", action="store_true")
    list_sources.set_defaults(func=cmd_list_sources)

    schema = sub.add_parser("schema", help="Print canonical/formal schema summary")
    schema.set_defaults(func=cmd_schema)

    plan = sub.add_parser("plan", help="Print the project plan")
    plan.add_argument("--path", action="store_true", help="Print only the plan path")
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor", help="Check configured local paths")
    doctor.add_argument("--env-config", type=Path)
    doctor.set_defaults(func=cmd_doctor)

    inspect_smoke = sub.add_parser("inspect-smoke", help="Inspect smoke_data samples")
    inspect_smoke.add_argument("--smoke-root", type=Path, default=PROJECT_ROOT / "smoke_data")
    inspect_smoke.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "smoke_data_inspection.json",
    )
    inspect_smoke.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "smoke_data_inspection.md",
    )
    inspect_smoke.set_defaults(func=cmd_inspect_smoke)

    convert_canonical = sub.add_parser(
        "convert-canonical",
        help="Convert a dataset root into standardized canonical NPZ clips",
    )
    convert_canonical.add_argument("--dataset", required=True)
    convert_canonical.add_argument("--input-root", type=Path, required=True)
    convert_canonical.add_argument("--output-root", type=Path, required=True)
    convert_canonical.add_argument("--target-fps", type=float, default=50.0)
    convert_canonical.add_argument("--report-json", type=Path)
    convert_canonical.add_argument("--report-md", type=Path)
    convert_canonical.add_argument("--run-config-json", type=Path)
    convert_canonical.add_argument("--overwrite", action="store_true")
    convert_canonical.add_argument("--num-workers", type=int, default=1)
    convert_canonical.add_argument("--progress-interval", type=int, default=100)
    convert_canonical.add_argument("--shard-index", type=int, default=0)
    convert_canonical.add_argument("--shard-count", type=int, default=1)
    convert_canonical.set_defaults(func=cmd_convert_canonical)

    convert_formal_npz = sub.add_parser(
        "convert-formal-npz",
        help="Convert canonical NPZ clips into formal human-side NPZ clips",
    )
    convert_formal_npz.add_argument("--canonical-root", type=Path, required=True)
    convert_formal_npz.add_argument("--output-root", type=Path, required=True)
    convert_formal_npz.add_argument("--report-json", type=Path)
    convert_formal_npz.add_argument("--report-md", type=Path)
    convert_formal_npz.add_argument("--run-config-json", type=Path)
    convert_formal_npz.add_argument("--overwrite", action="store_true")
    convert_formal_npz.add_argument("--num-workers", type=int, default=1)
    convert_formal_npz.add_argument("--progress-interval", type=int, default=100)
    convert_formal_npz.set_defaults(func=cmd_convert_formal_npz)

    visualize_canonical = sub.add_parser(
        "visualize-canonical",
        help="Render SMPL/SMPL-X mesh+skeleton contact sheets for canonical NPZ clips",
    )
    visualize_canonical.add_argument("--canonical-root", type=Path, required=True)
    visualize_canonical.add_argument("--output-root", type=Path, required=True)
    visualize_canonical.add_argument("--smpl-models-root", type=Path)
    visualize_canonical.add_argument("--env-config", type=Path)
    visualize_canonical.add_argument("--num-clips", type=int, default=8)
    visualize_canonical.add_argument("--num-frames", type=int, default=8)
    visualize_canonical.add_argument("--geometry-max-frames", type=int, default=120)
    visualize_canonical.add_argument("--batch-size", type=int, default=32)
    visualize_canonical.add_argument("--seed", type=int, default=20260703)
    visualize_canonical.add_argument("--overwrite", action="store_true")
    visualize_canonical.set_defaults(func=cmd_visualize_canonical)

    render_video = sub.add_parser(
        "render-canonical-video",
        help="Render 50Hz MP4 previews for canonical SMPL/SMPL-X mesh clips",
    )
    render_video.add_argument("--canonical-root", type=Path, required=True)
    render_video.add_argument("--output-root", type=Path, required=True)
    render_video.add_argument("--smpl-models-root", type=Path)
    render_video.add_argument("--env-config", type=Path)
    render_video.add_argument("--num-clips", type=int, default=8)
    render_video.add_argument("--video-fps", type=float, default=50.0)
    render_video.add_argument("--max-seconds", type=float, default=10.0)
    render_video.add_argument("--width", type=int, default=960)
    render_video.add_argument("--height", type=int, default=720)
    render_video.add_argument(
        "--floor-policy",
        choices=("first5_min", "first5_percentile"),
        default="first5_min",
    )
    render_video.add_argument(
        "--camera-mode",
        choices=("fixed_3quarter",),
        default="fixed_3quarter",
    )
    render_video.add_argument("--batch-size", type=int, default=32)
    render_video.add_argument("--seed", type=int, default=20260703)
    render_video.add_argument("--layout", choices=("nested", "flat"), default="nested")
    render_video.add_argument("--overwrite", action="store_true")
    render_video.set_defaults(func=cmd_render_canonical_video)

    render_formal_video = sub.add_parser(
        "render-formal-video",
        help="Render 50Hz MP4 previews for formal human-side NPZ clips",
    )
    render_formal_video.add_argument("--formal-root", type=Path, required=True)
    render_formal_video.add_argument("--output-root", type=Path, required=True)
    render_formal_video.add_argument("--smpl-models-root", type=Path)
    render_formal_video.add_argument("--env-config", type=Path)
    render_formal_video.add_argument("--num-clips", type=int, default=20)
    render_formal_video.add_argument("--video-fps", type=float, default=50.0)
    render_formal_video.add_argument("--max-seconds", type=float, default=4.0)
    render_formal_video.add_argument("--width", type=int, default=960)
    render_formal_video.add_argument("--height", type=int, default=720)
    render_formal_video.add_argument(
        "--floor-policy",
        choices=("first5_min", "first5_percentile"),
        default="first5_min",
    )
    render_formal_video.add_argument(
        "--camera-mode",
        choices=("fixed_3quarter",),
        default="fixed_3quarter",
    )
    render_formal_video.add_argument("--batch-size", type=int, default=32)
    render_formal_video.add_argument("--seed", type=int, default=20260703)
    render_formal_video.add_argument("--overwrite", action="store_true")
    render_formal_video.set_defaults(func=cmd_render_formal_video)

    pack_formal_h5 = sub.add_parser(
        "pack-formal-h5",
        help="Pack formal NPZ clips into frame-major formal H5 shard(s)",
    )
    pack_formal_h5.add_argument("--formal-npz-root", type=Path, required=True)
    pack_formal_h5.add_argument("--output-root", type=Path, required=True)
    pack_formal_h5.add_argument("--report-json", type=Path)
    pack_formal_h5.add_argument("--report-md", type=Path)
    pack_formal_h5.add_argument("--run-config-json", type=Path)
    pack_formal_h5.add_argument("--compression", default="gzip")
    pack_formal_h5.add_argument("--shard-target-gb", type=float, default=2.0)
    pack_formal_h5.add_argument("--shard-target-bytes", type=int)
    pack_formal_h5.add_argument(
        "--shard-target-mode",
        choices=(
            "uncompressed_nbytes",
            "nbytes",
            "uncompressed",
            "npz_filesize",
            "npz_size",
            "npz_bytes",
            "h5_filesize",
            "h5_size",
            "output_filesize",
            "disk",
        ),
        default="uncompressed_nbytes",
    )
    pack_formal_h5.add_argument("--shard-target-clips", type=int, default=0)
    pack_formal_h5.add_argument("--shard-target-frames", type=int, default=0)
    pack_formal_h5.add_argument("--chunks-t", type=int, default=1024)
    pack_formal_h5.add_argument("--progress-interval", type=int, default=100)
    pack_formal_h5.add_argument("--overwrite", action="store_true")
    pack_formal_h5.set_defaults(func=cmd_pack_formal_h5)

    retarget_h5 = sub.add_parser(
        "retarget-holoretarget-h5",
        help="Run HoloSMPL H5 through HoloRetarget and write robot training H5 v2",
    )
    retarget_h5.add_argument("--holosmpl-h5-root", type=Path, required=True)
    retarget_h5.add_argument("--output-root", type=Path, required=True)
    retarget_h5.add_argument("--compression", default="lzf")
    retarget_h5.add_argument("--chunks-t", type=int, default=1024)
    retarget_h5.add_argument("--shard-target-frames", type=int, default=250_000)
    retarget_h5.add_argument("--progress-interval", type=int, default=10)
    retarget_h5.add_argument("--overwrite", action="store_true")
    retarget_h5.set_defaults(func=cmd_retarget_holoretarget_h5)

    combine_formal_h5 = sub.add_parser(
        "combine-formal-h5-roots",
        help="Combine multiple formal_h5 roots by copying shards and merging manifests",
    )
    combine_formal_h5.add_argument("--input", action="append", type=Path, required=True)
    combine_formal_h5.add_argument("--output-root", type=Path, required=True)
    combine_formal_h5.add_argument("--report-json", type=Path)
    combine_formal_h5.add_argument("--report-md", type=Path)
    combine_formal_h5.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="copy",
        help="How to materialize H5 shard files into the combined root",
    )
    combine_formal_h5.add_argument("--overwrite", action="store_true")
    combine_formal_h5.add_argument("--progress-interval", type=int, default=1)
    combine_formal_h5.set_defaults(func=cmd_combine_formal_h5_roots)

    combine_canonical = sub.add_parser(
        "combine-canonical-roots",
        help="Combine multiple canonical roots by materializing clips and merging manifests",
    )
    combine_canonical.add_argument("--input", action="append", type=Path, required=True)
    combine_canonical.add_argument("--output-root", type=Path, required=True)
    combine_canonical.add_argument("--report-json", type=Path)
    combine_canonical.add_argument("--report-md", type=Path)
    combine_canonical.add_argument("--target-fps", type=float, default=50.0)
    combine_canonical.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize canonical NPZ clips into the combined root",
    )
    combine_canonical.add_argument("--overwrite", action="store_true")
    combine_canonical.add_argument("--progress-interval", type=int, default=1000)
    combine_canonical.set_defaults(func=cmd_combine_canonical_roots)

    combine_formal_npz = sub.add_parser(
        "combine-formal-npz-roots",
        help="Combine multiple formal_npz roots by materializing clips and merging manifests",
    )
    combine_formal_npz.add_argument("--input", action="append", type=Path, required=True)
    combine_formal_npz.add_argument("--output-root", type=Path, required=True)
    combine_formal_npz.add_argument("--report-json", type=Path)
    combine_formal_npz.add_argument("--report-md", type=Path)
    combine_formal_npz.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize formal NPZ clips into the combined root",
    )
    combine_formal_npz.add_argument("--overwrite", action="store_true")
    combine_formal_npz.add_argument("--progress-interval", type=int, default=1000)
    combine_formal_npz.set_defaults(func=cmd_combine_formal_npz_roots)

    convert = sub.add_parser(
        "convert",
        help="Convert a raw source root into canonical/formal HoloSMPL artifacts",
    )
    convert.add_argument("--source", required=True, help="Source key from `holosmpl list-sources`")
    convert.add_argument("--input-root", type=Path, required=True)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--num-workers", type=int, default=1)
    convert.add_argument("--progress-interval", type=int, default=100)
    convert.add_argument("--shard-index", type=int, default=0)
    convert.add_argument("--shard-count", type=int, default=1)
    convert.add_argument("--skip-h5", action="store_true", help="Stop after formal NPZ output")
    convert.add_argument("--compression", default="gzip")
    convert.add_argument("--shard-target-gb", type=float, default=2.0)
    convert.add_argument("--chunks-t", type=int, default=1024)
    convert.set_defaults(func=cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
