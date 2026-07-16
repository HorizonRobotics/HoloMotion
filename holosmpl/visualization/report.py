from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_visualization_reports(output_root: str | Path, report: dict[str, Any]) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "geometry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "geometry_report.md").write_text(render_report_md(report), encoding="utf-8")


def render_report_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# SMPL 可视化与几何检查报告",
        "",
        "## 总体结论",
        "",
        f"- canonical_root: `{report['canonical_root']}`",
        f"- output_root: `{report['output_root']}`",
        f"- selected_clip_count: {validation['selected_clip_count']}",
        f"- rendered_clip_count: {validation['rendered_clip_count']}",
        f"- warning_clip_count: {validation['warning_clip_count']}",
        f"- error_count: {validation['error_count']}",
        f"- all_ok: {validation['all_ok']}",
        "",
        "## Clips",
        "",
    ]
    for sample in report["samples"]:
        title = sample.get("clip_id") or Path(str(sample.get("clip_path", "unknown"))).stem
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- status: {sample['status']}")
        lines.append(f"- source: `{sample.get('source_relative_path')}`")
        lines.append(f"- contact_sheet: `{sample.get('contact_sheet')}`")
        geometry = sample.get("geometry") or {}
        if geometry:
            lines.append(f"- model: {geometry.get('model_type')} / {geometry.get('gender')}")
            lines.append(f"- mesh: {geometry.get('mesh_vertex_count')} vertices, {geometry.get('mesh_face_count')} faces")
            lines.append(f"- joints: {geometry.get('joint_count')}")
            lines.append(f"- body_height_median: {geometry.get('body_height_median')}")
            lines.append(f"- root_height_min/max: {geometry.get('root_height_min')} / {geometry.get('root_height_max')}")
            lines.append(f"- lowest_vertex_z_min/max: {geometry.get('lowest_vertex_z_min')} / {geometry.get('lowest_vertex_z_max')}")
            lines.append(f"- root_speed_max: {geometry.get('root_speed_max')}")
            warnings = geometry.get("warnings") or []
            lines.append(f"- warnings: {warnings if warnings else '[]'}")
        if sample.get("error"):
            lines.append(f"- error: `{sample['error']}`")
        lines.append("")
    return "\n".join(lines)
