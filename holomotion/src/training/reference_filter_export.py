# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import json
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
from omegaconf import DictConfig, ListConfig
from scipy.spatial.transform import Rotation as Rotation3D

from holomotion.src.training.h5_dataloader import (
    MotionClipSample,
    build_motion_datasets_from_cfg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy(array_like) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=np.float32)


def _require_tensor(
    tensors: Mapping[str, object], tensor_name: str, error_message: str
) -> np.ndarray:
    if tensor_name not in tensors:
        raise ValueError(error_message)
    return _to_numpy(tensors[tensor_name])


def _quat_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    flat = quat_xyzw.reshape(-1, 4)
    euler = Rotation3D.from_quat(flat).as_euler("xyz", degrees=False)
    return euler.reshape(*quat_xyzw.shape[:-1], 3).astype(
        np.float32, copy=False
    )


def _write_npz(output_path: Path, payload: Mapping[str, np.ndarray]) -> None:
    np.savez(str(output_path), **payload)


def _plot_series_groups(
    output_path: Path,
    title: str,
    groups: Sequence[tuple[str, np.ndarray, np.ndarray]],
    axis_labels: Sequence[str] = ("x", "y", "z"),
) -> None:
    nrows = len(groups)
    ncols = len(axis_labels)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 2.8 * max(1, nrows)),
        squeeze=False,
    )
    plot_steps = np.arange(groups[0][1].shape[0], dtype=np.int32)
    for row_idx, (group_name, ref_values, ft_values) in enumerate(groups):
        for col_idx, axis_name in enumerate(axis_labels):
            ax = axes[row_idx, col_idx]
            ax.plot(
                plot_steps,
                ref_values[:, col_idx],
                label="raw",
                linewidth=1.4,
            )
            ax.plot(
                plot_steps,
                ft_values[:, col_idx],
                label="filtered",
                linewidth=1.2,
            )
            ax.set_title(f"{group_name} {axis_name}")
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dof_matrix(
    output_path: Path,
    title: str,
    dof_names: Sequence[str],
    ref_values: np.ndarray,
    ft_values: np.ndarray,
) -> None:
    num_dofs = len(dof_names)
    fig, axes = plt.subplots(
        nrows=num_dofs,
        ncols=1,
        figsize=(14.0, max(2.8 * num_dofs, 3.5)),
        squeeze=False,
    )
    plot_steps = np.arange(ref_values.shape[0], dtype=np.int32)
    for idx, dof_name in enumerate(dof_names):
        ax = axes[idx, 0]
        ax.plot(plot_steps, ref_values[:, idx], label="raw", linewidth=1.4)
        ax.plot(plot_steps, ft_values[:, idx], label="filtered", linewidth=1.2)
        ax.set_title(dof_name)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_reference_filter_debug_artifacts(
    *,
    sample: MotionClipSample,
    output_dir: str | Path,
    body_names: Sequence[str],
    dof_names: Sequence[str],
    selected_body_links: Sequence[str],
) -> Path:
    tensors = sample.tensors
    if "ft_ref_rg_pos" not in tensors or "ft_ref_dof_pos" not in tensors:
        raise ValueError(
            "Filtered reference tensors are unavailable. Ensure online filtering "
            "is enabled and ft_ref_* tensors are materialized."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_root_pos = _require_tensor(
        tensors,
        "ref_root_pos",
        "Missing ref_root_pos tensor in sampled clip.",
    )
    ft_root_pos = _require_tensor(
        tensors,
        "ft_ref_root_pos",
        "Missing ft_ref_root_pos tensor in sampled clip.",
    )
    ref_root_rot = _require_tensor(
        tensors,
        "ref_root_rot",
        "Missing ref_root_rot tensor in sampled clip.",
    )
    ft_root_rot = _require_tensor(
        tensors,
        "ft_ref_root_rot",
        "Missing ft_ref_root_rot tensor in sampled clip.",
    )
    ref_root_vel = _require_tensor(
        tensors,
        "ref_root_vel",
        "Missing ref_root_vel tensor in sampled clip.",
    )
    ft_root_vel = _require_tensor(
        tensors,
        "ft_ref_root_vel",
        "Missing ft_ref_root_vel tensor in sampled clip.",
    )
    ref_root_ang_vel = _require_tensor(
        tensors,
        "ref_root_ang_vel",
        "Missing ref_root_ang_vel tensor in sampled clip.",
    )
    ft_root_ang_vel = _require_tensor(
        tensors,
        "ft_ref_root_ang_vel",
        "Missing ft_ref_root_ang_vel tensor in sampled clip.",
    )
    ref_rg_pos = _require_tensor(
        tensors,
        "ref_rg_pos",
        "Missing ref_rg_pos tensor in sampled clip.",
    )
    ft_ref_rg_pos = _require_tensor(
        tensors,
        "ft_ref_rg_pos",
        "Missing ft_ref_rg_pos tensor in sampled clip.",
    )
    ref_body_vel = _require_tensor(
        tensors,
        "ref_body_vel",
        "Missing ref_body_vel tensor in sampled clip.",
    )
    ft_ref_body_vel = _require_tensor(
        tensors,
        "ft_ref_body_vel",
        "Missing ft_ref_body_vel tensor in sampled clip.",
    )
    ref_body_ang_vel = _require_tensor(
        tensors,
        "ref_body_ang_vel",
        "Missing ref_body_ang_vel tensor in sampled clip.",
    )
    ft_ref_body_ang_vel = _require_tensor(
        tensors,
        "ft_ref_body_ang_vel",
        "Missing ft_ref_body_ang_vel tensor in sampled clip.",
    )
    ref_dof_pos = _require_tensor(
        tensors,
        "ref_dof_pos",
        "Missing ref_dof_pos tensor in sampled clip.",
    )
    ft_ref_dof_pos = _require_tensor(
        tensors,
        "ft_ref_dof_pos",
        "Missing ft_ref_dof_pos tensor in sampled clip.",
    )
    ref_dof_vel = _require_tensor(
        tensors,
        "ref_dof_vel",
        "Missing ref_dof_vel tensor in sampled clip.",
    )
    ft_ref_dof_vel = _require_tensor(
        tensors,
        "ft_ref_dof_vel",
        "Missing ft_ref_dof_vel tensor in sampled clip.",
    )

    body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}
    missing_links = [
        link_name
        for link_name in selected_body_links
        if link_name not in body_name_to_idx
    ]
    if missing_links:
        raise ValueError(
            f"Requested body links are missing from robot.body_names: {missing_links}"
        )

    ref_root_rpy = _quat_xyzw_to_rpy(ref_root_rot)
    ft_root_rpy = _quat_xyzw_to_rpy(ft_root_rot)

    root_payload = {
        "ref_global_pos": ref_root_pos,
        "ft_ref_global_pos": ft_root_pos,
        "ref_rpy": ref_root_rpy,
        "ft_ref_rpy": ft_root_rpy,
        "ref_lin_vel": ref_root_vel,
        "ft_ref_lin_vel": ft_root_vel,
        "ref_ang_vel": ref_root_ang_vel,
        "ft_ref_ang_vel": ft_root_ang_vel,
    }
    _write_npz(output_dir / "root_signals.npz", root_payload)

    body_payload: dict[str, np.ndarray] = {}
    for link_name in selected_body_links:
        body_idx = body_name_to_idx[link_name]
        body_payload[f"{link_name}__ref_global_pos"] = ref_rg_pos[
            :, body_idx, :
        ]
        body_payload[f"{link_name}__ft_ref_global_pos"] = ft_ref_rg_pos[
            :, body_idx, :
        ]
        body_payload[f"{link_name}__ref_lin_vel"] = ref_body_vel[
            :, body_idx, :
        ]
        body_payload[f"{link_name}__ft_ref_lin_vel"] = ft_ref_body_vel[
            :, body_idx, :
        ]
        body_payload[f"{link_name}__ref_ang_vel"] = ref_body_ang_vel[
            :, body_idx, :
        ]
        body_payload[f"{link_name}__ft_ref_ang_vel"] = ft_ref_body_ang_vel[
            :, body_idx, :
        ]
    _write_npz(output_dir / "bodylink_signals.npz", body_payload)

    dof_payload = {
        "ref_dof_pos": ref_dof_pos,
        "ft_ref_dof_pos": ft_ref_dof_pos,
        "ref_dof_vel": ref_dof_vel,
        "ft_ref_dof_vel": ft_ref_dof_vel,
    }
    _write_npz(output_dir / "dof_signals.npz", dof_payload)

    filter_cutoff_tensor = tensors.get("filter_cutoff_hz")
    filter_cutoff_hz = None
    if filter_cutoff_tensor is not None:
        cutoff_values = _to_numpy(filter_cutoff_tensor).reshape(-1)
        if cutoff_values.size > 0:
            filter_cutoff_hz = float(cutoff_values[0])

    metadata = {
        "motion_key": sample.motion_key,
        "raw_motion_key": sample.raw_motion_key,
        "window_index": int(sample.window_index),
        "length": int(sample.length),
        "filter_cutoff_hz": filter_cutoff_hz,
        "selected_body_links": list(selected_body_links),
        "body_names": list(body_names),
        "dof_names": list(dof_names),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _plot_series_groups(
        output_dir / "root_comparison.png",
        title="Root Raw vs Filtered Reference Signals",
        groups=[
            ("global_pos", ref_root_pos, ft_root_pos),
            ("rpy", ref_root_rpy, ft_root_rpy),
            ("lin_vel", ref_root_vel, ft_root_vel),
            ("ang_vel", ref_root_ang_vel, ft_root_ang_vel),
        ],
    )

    for link_name in selected_body_links:
        _plot_series_groups(
            output_dir / f"{link_name}_comparison.png",
            title=f"{link_name} Raw vs Filtered Reference Signals",
            groups=[
                (
                    "global_pos",
                    body_payload[f"{link_name}__ref_global_pos"],
                    body_payload[f"{link_name}__ft_ref_global_pos"],
                ),
                (
                    "lin_vel",
                    body_payload[f"{link_name}__ref_lin_vel"],
                    body_payload[f"{link_name}__ft_ref_lin_vel"],
                ),
                (
                    "ang_vel",
                    body_payload[f"{link_name}__ref_ang_vel"],
                    body_payload[f"{link_name}__ft_ref_ang_vel"],
                ),
            ],
        )

    _plot_dof_matrix(
        output_dir / "dof_pos_comparison.png",
        title="DOF Position Raw vs Filtered",
        dof_names=dof_names,
        ref_values=ref_dof_pos,
        ft_values=ft_ref_dof_pos,
    )
    _plot_dof_matrix(
        output_dir / "dof_vel_comparison.png",
        title="DOF Velocity Raw vs Filtered",
        dof_names=dof_names,
        ref_values=ref_dof_vel,
        ft_values=ft_ref_dof_vel,
    )

    return output_dir


def _to_plain_sequence(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (ListConfig, tuple, list)):
        return [str(v) for v in values]
    return [str(values)]


def export_reference_filter_artifacts_from_config(config) -> Path:
    debug_cfg = getattr(config, "debug_reference_filter_export", None)
    if debug_cfg is None or not bool(debug_cfg.get("enabled", False)):
        raise ValueError("debug_reference_filter_export.enabled must be true.")

    motion_cfg = config.robot.motion
    online_filter_cfg = motion_cfg.get("online_filter", {})
    if not bool(online_filter_cfg.get("enabled", False)):
        raise ValueError(
            "Reference filter debug export requires robot.motion.online_filter.enabled=true."
        )

    output_dir = debug_cfg.get("output_dir", None)
    if output_dir in (None, ""):
        output_dir = tempfile.mkdtemp(prefix="motrack-ref-filter-")

    train_dataset, _, _ = build_motion_datasets_from_cfg(
        motion_cfg=motion_cfg,
        max_frame_length=int(motion_cfg.max_frame_length),
        min_window_length=int(motion_cfg.min_frame_length),
        world_frame_normalization=bool(
            motion_cfg.get("world_frame_normalization", True)
        ),
    )
    sample = train_dataset[0]

    return export_reference_filter_debug_artifacts(
        sample=sample,
        output_dir=Path(str(output_dir)),
        body_names=_to_plain_sequence(config.robot.body_names),
        dof_names=_to_plain_sequence(config.robot.dof_names),
        selected_body_links=_to_plain_sequence(
            debug_cfg.get("selected_body_links", [])
        ),
    )
