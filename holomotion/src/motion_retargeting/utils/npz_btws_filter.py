#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterworth filtering utility for NPZ motion data.
- Loads all configuration via Hydra from config/motion_retargeting/filter_config.yaml
- Applies Butterworth low-pass filters to configured keys in NPZ files
- No face-front selection, no duration/time-based selection, no other filters
"""

from pathlib import Path
from typing import Dict

import numpy as np
from scipy.signal import butter, filtfilt
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def apply_butterworth_filter(data: np.ndarray, filter_config: Dict) -> np.ndarray:
    """Apply a Butterworth low-pass filter to data.

    Args:
        data: Input array, shape can be (T,), (T, D), or (T, J, D)
        filter_config: dict with keys {enabled, order, cutoff, sample_rate}

    Returns:
        Filtered data, same shape as input
    """
    if not filter_config.get("enabled", False):
        return data

    order = int(filter_config["order"])  # filter order
    cutoff = float(filter_config["cutoff"])  # Hz
    fs = float(filter_config["sample_rate"])  # Hz

    nyquist = 0.5 * fs
    normal_cutoff = min(max(cutoff / nyquist, 1e-6), 0.99)

    # Design Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    filtered_data = data.copy()

    if data.ndim == 1:
        filtered_data = filtfilt(b, a, filtered_data)
    elif data.ndim == 2:
        # Apply per feature
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
    elif data.ndim == 3:
        # Apply per joint and per dim
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                filtered_data[:, i, j] = filtfilt(b, a, data[:, i, j])
    else:
        # Unsupported ndim; return original
        return data

    return filtered_data


def apply_butterworth_filters_to_npz(data_dict: Dict[str, np.ndarray], config: Dict) -> Dict[str, np.ndarray]:
    """Apply Butterworth filters to a NPZ data dict according to config."""
    filtered_data = data_dict.copy()

    for key, value in data_dict.items():
        if key in config:
            filter_config = config[key]
            if filter_config.get("enabled", False):
                try:
                    filtered_data[key] = apply_butterworth_filter(value, filter_config)
                except Exception:
                    filtered_data[key] = value
        # keys without config are left unchanged

    return filtered_data


def process_npz_file(in_path: Path, out_path: Path, config: Dict) -> bool:
    try:
        data = np.load(in_path)
        data_dict = {k: data[k].copy() for k in data.files}

        processed = apply_butterworth_filters_to_npz(data_dict, config)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **processed)
        return True
    except Exception:
        return False


def _resolve_io_dirs(cfg: DictConfig) -> (Path, Path):
    """Resolve input/output directories from cfg with fallbacks."""
    in_dir = None
    out_dir = None

    for key in ["input_dir", "in_dir", "npz_root"]:
        if key in cfg:
            in_dir = Path(str(cfg[key]))
            break
        if "data" in cfg and key in cfg.data:
            in_dir = Path(str(cfg.data[key]))
            break

    for key in ["output_dir", "out_dir", "dump_dir"]:
        if key in cfg:
            out_dir = Path(str(cfg[key]))
            break
        if "data" in cfg and key in cfg.data:
            out_dir = Path(str(cfg.data[key]))
            break

    if in_dir is None or out_dir is None:
        raise ValueError("input_dir/output_dir must be specified in filter_config.yaml")

    return in_dir, out_dir


def _resolve_btws_config(cfg: DictConfig) -> Dict[str, Dict]:
    """Resolve per-key Butterworth config mapping from cfg."""
    for path in [
        ("butterworth",),
        ("filter", "butterworth"),
        ("filters", "butterworth"),
    ]:
        node = cfg
        ok = True
        for p in path:
            if p in node:
                node = node[p]
            else:
                ok = False
                break
        if ok:
            return OmegaConf.to_container(node, resolve=True)  # type: ignore

    return {}


def _print_btws_config(btws_config: Dict[str, Dict]) -> None:
    print("Butterworth filter settings:")
    print("-" * 40)
    if not btws_config:
        print("(empty config)")
        return
    for key, cfg in btws_config.items():
        enabled = cfg.get("enabled", False)
        print(f"{key}: {'ENABLED' if enabled else 'disabled'}")
        if enabled:
            print(
                f"  order={cfg.get('order')} cutoff={cfg.get('cutoff')}Hz fs={cfg.get('sample_rate')}Hz"
            )


@hydra.main(
    version_base=None,
    config_path="../../../config/motion_retargeting",
    config_name="filter_config",
)
def main(cfg: DictConfig) -> None:
    input_dir, output_dir = _resolve_io_dirs(cfg)
    btws_config = _resolve_btws_config(cfg)

    # Print settings first
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    _print_btws_config(btws_config)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return

    npz_files = sorted(list(input_dir.glob("*.npz")))
    if not npz_files:
        print(f"No NPZ files found in: {input_dir}")
        return

    success = 0
    with tqdm(total=len(npz_files), desc="Filtering NPZ") as pbar:
        for npz_path in npz_files:
            ok = process_npz_file(npz_path, output_dir / npz_path.name, btws_config)
            success += int(ok)
            pbar.update(1)

    print(f"Done. Success: {success}/{len(npz_files)}")


if __name__ == "__main__":
    main() 