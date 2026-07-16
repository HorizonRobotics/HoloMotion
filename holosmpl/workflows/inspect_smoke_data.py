from __future__ import annotations

import ast
import csv
import gzip
import json
import pickletools
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile


KNOWN_PICKLE_FIELD_NAMES = (
    "pose_aa",
    "transl",
    "trans",
    "smpl_joints",
    "joints",
    "betas",
    "gender",
    "fps",
    "mocap_framerate",
    "root_orient",
    "pose_body",
    "human_pose_aa",
    "human_shape_beta",
    "human_gravity_projection",
    "human_root_height",
    "human_root_trans",
)


@dataclass(frozen=True)
class SmokeManifestSample:
    split: str
    dataset: str
    local_path: Path
    raw_key: str
    source_uri: str
    suffix: str


def inspect_smoke_data(
    smoke_root: Path,
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    smoke_root = smoke_root.resolve()
    summary = _read_json(smoke_root / "summary.json")
    samples = list(_iter_manifest_samples(smoke_root))
    inspected = [_inspect_sample(smoke_root, sample) for sample in samples]
    report = _build_report(smoke_root, summary, inspected)
    _write_json(report_json, report)
    _write_text(report_md, _render_markdown(report))
    return report


def _iter_manifest_samples(smoke_root: Path) -> list[SmokeManifestSample]:
    manifest_root = smoke_root / "manifests"
    samples: list[SmokeManifestSample] = []
    for manifest_path in sorted(manifest_root.rglob("*.json")):
        manifest = _read_json(manifest_path)
        for item in manifest.get("samples", []):
            local_rel = Path(str(item["local_path"]))
            samples.append(
                SmokeManifestSample(
                    split=str(item["split"]),
                    dataset=str(item["dataset"]),
                    local_path=smoke_root / local_rel,
                    raw_key=str(item["raw_key"]),
                    source_uri=str(item["source_uri"]),
                    suffix=str(item["suffix"]).lower(),
                )
            )
    return samples


def _inspect_sample(smoke_root: Path, sample: SmokeManifestSample) -> dict[str, Any]:
    path = sample.local_path
    base: dict[str, Any] = {
        "split": sample.split,
        "dataset": sample.dataset,
        "raw_key": sample.raw_key,
        "source_uri": sample.source_uri,
        "local_path": path.relative_to(smoke_root).as_posix(),
        "suffix": sample.suffix,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        base["status"] = "missing"
        base["error"] = "local_path does not exist"
        return base

    try:
        if sample.suffix == ".npz":
            payload = _inspect_npz(path)
        elif sample.suffix == ".bvh":
            payload = _inspect_bvh(path)
        elif sample.suffix == ".csv":
            payload = _inspect_csv(path)
        elif sample.suffix == ".pkl":
            payload = _inspect_pickle_like(path)
        else:
            payload = {"format": "unknown"}
        base.update(payload)
        base["status"] = "ok"
    except Exception as exc:
        base["status"] = "error"
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _inspect_npz(path: Path) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                arrays[Path(member).stem] = _read_npy_header(handle)

    fields = sorted(arrays)
    first_dims = [
        value["shape"][0]
        for value in arrays.values()
        if value.get("shape") and isinstance(value["shape"][0], int)
    ]
    sequence_length = Counter(first_dims).most_common(1)[0][0] if first_dims else None
    smpl_relevant = {
        name: arrays[name]
        for name in fields
        if name
        in {
            "root_orient",
            "pose_body",
            "trans",
            "transl",
            "betas",
            "gender",
            "mocap_framerate",
            "human_pose_aa",
            "human_shape_beta",
            "human_gravity_projection",
            "human_root_height",
            "human_root_trans",
        }
    }
    return {
        "format": "npz",
        "fields": fields,
        "array_count": len(arrays),
        "arrays": arrays,
        "sequence_length_guess": sequence_length,
        "smpl_relevant_fields": smpl_relevant,
    }


def _read_npy_header(handle: Any) -> dict[str, Any]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("invalid npy magic")
    major = handle.read(1)[0]
    minor = handle.read(1)[0]
    if major == 1:
        header_len = int.from_bytes(handle.read(2), "little")
    elif major in (2, 3):
        header_len = int.from_bytes(handle.read(4), "little")
    else:
        raise ValueError(f"unsupported npy version: {major}.{minor}")
    header_text = handle.read(header_len).decode("latin1").strip()
    header = ast.literal_eval(header_text)
    return {
        "dtype": str(header["descr"]),
        "fortran_order": bool(header["fortran_order"]),
        "shape": list(header["shape"]),
        "npy_version": [major, minor],
    }


def _inspect_bvh(path: Path) -> dict[str, Any]:
    frames = None
    frame_time = None
    joint_names: list[str] = []
    channel_count = 0
    first_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if line_idx < 40:
                first_lines.append(stripped)
            if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2:
                    joint_names.append(parts[1])
            elif stripped.startswith("CHANNELS "):
                parts = stripped.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    channel_count += int(parts[1])
            elif stripped.startswith("Frames:"):
                frames = _parse_int_after_colon(stripped)
            elif stripped.startswith("Frame Time:"):
                frame_time = _parse_float_after_colon(stripped)
    fps = None if not frame_time else 1.0 / frame_time
    return {
        "format": "bvh",
        "joint_count": len(joint_names),
        "joint_names_first20": joint_names[:20],
        "channel_count": channel_count,
        "frames": frames,
        "frame_time": frame_time,
        "fps_guess": fps,
        "first_lines": first_lines,
    }


def _inspect_csv(path: Path) -> dict[str, Any]:
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    header: list[str] = []
    first_rows: list[list[str]] = []
    used_encoding = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="strict", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                for _, row in zip(range(3), reader):
                    first_rows.append(row[:12])
            used_encoding = encoding
            break
        except UnicodeError:
            continue
    if used_encoding is None:
        text = path.read_bytes()[:8192].decode("utf-8", errors="replace")
        rows = list(csv.reader(text.splitlines()))
        header = rows[0] if rows else []
        first_rows = [row[:12] for row in rows[1:4]]
        used_encoding = "utf-8-replace"

    row_count = 0
    with path.open("rb") as handle:
        for row_count, _ in enumerate(handle, start=1):
            pass
    data_row_count = max(row_count - 1, 0) if header else row_count
    lower_header = [item.lower() for item in header]
    return {
        "format": "csv",
        "encoding": used_encoding,
        "column_count": len(header),
        "columns_first40": header[:40],
        "data_row_count": data_row_count,
        "first_rows_first12_columns": first_rows,
        "time_like_columns": [
            header[idx]
            for idx, name in enumerate(lower_header)
            if "time" in name or "timestamp" in name or name in {"t", "frame"}
        ][:20],
    }


def _inspect_pickle_like(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload, compression = _maybe_decompress(raw)
    pickle_protocol = payload[1] if len(payload) >= 2 and payload[0] == 0x80 else None
    strings, opcode_error = _collect_pickle_strings(payload)
    ascii_tokens = _collect_ascii_tokens(payload)
    known_hits = sorted(
        {
            field
            for field in KNOWN_PICKLE_FIELD_NAMES
            if field in strings or field.encode("utf-8") in ascii_tokens
        }
    )
    return {
        "format": "pickle_like",
        "compression": compression,
        "payload_size_bytes": len(payload),
        "pickle_protocol_guess": pickle_protocol,
        "pickle_string_tokens_first80": strings[:80],
        "pickle_opcode_scan_error": opcode_error,
        "known_field_hits": known_hits,
        "ascii_tokens_first120": [
            token.decode("utf-8", errors="replace")
            for token in sorted(ascii_tokens)[:120]
        ],
    }


def _maybe_decompress(raw: bytes) -> tuple[bytes, str]:
    if raw[:2] in {b"x\x9c", b"x^", b"x\xda"}:
        return zlib.decompress(raw), "zlib"
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw), "gzip"
    return raw, "none"


def _collect_pickle_strings(payload: bytes) -> tuple[list[str], str | None]:
    strings: list[str] = []
    error = None
    try:
        for opcode, arg, _ in pickletools.genops(payload):
            if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                strings.append(str(arg))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return strings, error


def _collect_ascii_tokens(payload: bytes) -> set[bytes]:
    tokens = set(re.findall(rb"[A-Za-z_][A-Za-z0-9_]{1,63}", payload))
    return {token for token in tokens if len(token) <= 64}


def _build_report(
    smoke_root: Path,
    summary: dict[str, Any],
    inspected: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    suffix_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    dataset_fields: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_shapes: dict[str, dict[str, list[list[int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in inspected:
        split = str(item["split"])
        dataset = str(item["dataset"])
        dataset_counts[split][dataset] += 1
        suffix_counts[str(item["suffix"])] += 1
        status_counts[str(item["status"])] += 1
        if item.get("format") == "npz":
            for field in item.get("fields", []):
                dataset_fields[f"{split}/{dataset}"][str(field)] += 1
            for field, meta in item.get("arrays", {}).items():
                shapes = dataset_shapes[f"{split}/{dataset}"][str(field)]
                shape = list(meta.get("shape", []))
                if shape not in shapes and len(shapes) < 8:
                    shapes.append(shape)

    manifest_sample_count = len(inspected)
    summary_sample_count = summary.get("total_selected_files")
    validation = {
        "summary_total_selected_files": summary_sample_count,
        "manifest_sample_count": manifest_sample_count,
        "inspected_sample_count": len(inspected),
        "all_counts_match": summary_sample_count
        == manifest_sample_count
        == len(inspected),
        "error_count": status_counts.get("error", 0) + status_counts.get("missing", 0),
    }

    return {
        "schema_version": "human_smpl_smoke_inspection_v1",
        "smoke_root": str(smoke_root),
        "validation": validation,
        "summary": {
            "dataset_counts": {
                split: dict(counter) for split, counter in sorted(dataset_counts.items())
            },
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "npz_field_counts_by_dataset": {
            dataset: dict(counter.most_common())
            for dataset, counter in sorted(dataset_fields.items())
        },
        "npz_shape_examples_by_dataset": {
            dataset: dict(fields) for dataset, fields in sorted(dataset_shapes.items())
        },
        "samples": inspected,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# smoke_data 检查报告",
        "",
        "## 总体结论",
        "",
    ]
    validation = report["validation"]
    lines.extend(
        [
            f"- summary 总数: {validation['summary_total_selected_files']}",
            f"- manifest 样本数: {validation['manifest_sample_count']}",
            f"- 实际检查样本数: {validation['inspected_sample_count']}",
            f"- 三者是否一致: {validation['all_counts_match']}",
            f"- error/missing 数量: {validation['error_count']}",
            "",
            "## 数据集样本数",
            "",
        ]
    )
    for split, datasets in report["summary"]["dataset_counts"].items():
        lines.append(f"### {split}")
        lines.append("")
        for dataset, count in sorted(datasets.items()):
            lines.append(f"- {dataset}: {count}")
        lines.append("")

    lines.extend(["## 后缀统计", ""])
    for suffix, count in report["summary"]["suffix_counts"].items():
        lines.append(f"- {suffix}: {count}")
    lines.append("")

    lines.extend(["## NPZ 字段概览", ""])
    for dataset, fields in report["npz_field_counts_by_dataset"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        field_text = ", ".join(f"{name}({count})" for name, count in fields.items())
        lines.append(field_text or "无")
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- `.npz` 文件通过 zip/npy header 读取字段、shape、dtype，不依赖 numpy。",
            "- `.pkl` 文件不强制反序列化，只做压缩探测、pickle opcode/string token 扫描和已知字段命中检查。",
            "- `.bvh` 文件统计 ROOT/JOINT、CHANNELS、Frames、Frame Time。",
            "- `.csv` 文件统计表头、行数和疑似时间列。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_int_after_colon(text: str) -> int | None:
    try:
        return int(text.split(":", 1)[1].strip())
    except Exception:
        return None


def _parse_float_after_colon(text: str) -> float | None:
    try:
        return float(text.split(":", 1)[1].strip())
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
