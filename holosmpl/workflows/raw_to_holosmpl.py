"""End-to-end raw source to HoloSMPL pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from holosmpl.workflows.convert_dataset import convert_dataset_to_canonical
from holosmpl.workflows.convert_formal import convert_canonical_to_formal_npz
from holosmpl.workflows.pack_formal_h5 import pack_formal_npz_to_h5


def raw_to_holosmpl(
    *,
    source: str,
    input_root: str | Path,
    output_root: str | Path,
    target_fps: float = 50.0,
    overwrite: bool = False,
    num_workers: int = 1,
    progress_interval: int = 100,
    shard_index: int = 0,
    shard_count: int = 1,
    write_formal_h5: bool = True,
    compression: str = "gzip",
    h5_shard_target_gb: float = 2.0,
    h5_chunks_t: int = 1024,
) -> dict[str, Any]:
    """Convert one raw source root into canonical/formal HoloSMPL artifacts."""

    output_root = Path(output_root)
    canonical_root = output_root / "canonical"
    formal_npz_root = output_root / "formal_npz"
    formal_h5_root = output_root / "formal_h5"
    run_config_root = output_root / "run_configs"
    report_root = output_root / "reports"
    run_config_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    canonical = convert_dataset_to_canonical(
        dataset=source,
        input_root=input_root,
        output_root=canonical_root,
        target_fps=target_fps,
        report_json=report_root / "canonical_validation.json",
        report_md=report_root / "canonical_validation.md",
        run_config_json=run_config_root / "canonical_run_config.json",
        overwrite=overwrite,
        num_workers=num_workers,
        progress_interval=progress_interval,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    formal_npz = convert_canonical_to_formal_npz(
        canonical_root=canonical_root,
        output_root=formal_npz_root,
        report_json=report_root / "formal_npz_validation.json",
        report_md=report_root / "formal_npz_validation.md",
        run_config_json=run_config_root / "formal_npz_run_config.json",
        overwrite=overwrite,
        num_workers=num_workers,
        progress_interval=progress_interval,
    )
    formal_h5 = None
    if write_formal_h5:
        formal_h5 = pack_formal_npz_to_h5(
            formal_npz_root=formal_npz_root,
            output_root=formal_h5_root,
            report_json=report_root / "formal_h5_validation.json",
            report_md=report_root / "formal_h5_validation.md",
            run_config_json=run_config_root / "formal_h5_run_config.json",
            compression=compression,
            overwrite=overwrite,
            shard_target_gb=h5_shard_target_gb,
            chunks_t=h5_chunks_t,
            progress_interval=progress_interval,
        )
    return {
        "source": source,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "canonical_root": str(canonical_root),
        "formal_npz_root": str(formal_npz_root),
        "formal_h5_root": str(formal_h5_root) if write_formal_h5 else None,
        "canonical": canonical,
        "formal_npz": formal_npz,
        "formal_h5": formal_h5,
    }
