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


from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import csv
import json
import os
import re
from glob import glob
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import welch
from scipy.spatial.transform import Rotation as sRot
from tabulate import tabulate
from tqdm import tqdm


DEFAULT_ROBOT_CONTROL_DT = 1.0 / 50.0
TORQUE_JUMP_RATIO_EPS = 1e-6
MIN_WELCH_SAMPLES = 8
STABILITY_BURST_WINDOW_SECONDS = 0.5
TOUCHDOWN_WINDOW_SECONDS = 0.05
ROOT_BODY_INDEX = 0
PROBABILITY_EPS = 1e-12


def quat_inv(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    xyz = q[:, None, :3]
    w = q[:, None, 3:4]

    t = 2.0 * np.cross(xyz, v, axis=-1)
    return v + w * t + np.cross(xyz, t, axis=-1)


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Procrustes-aligned MPJPE between predicted and ground truth.

    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    # Per-frame SVD with graceful handling for non-convergence: mark those frames as NaN
    batch_size = int(h.shape[0])
    jdim = int(h.shape[1])
    u = np.empty((batch_size, jdim, jdim), dtype=h.dtype)
    s = np.empty((batch_size, jdim), dtype=h.dtype)
    vt = np.empty((batch_size, jdim, jdim), dtype=h.dtype)
    for i in range(batch_size):
        try:
            ui, si, vti = np.linalg.svd(h[i])
            u[i] = ui
            s[i] = si
            vt[i] = vti
        except np.linalg.LinAlgError:
            u[i].fill(np.nan)
            s[i].fill(np.nan)
            vt[i].fill(np.nan)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Corrected rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    predicted_aligned = a * np.matmul(predicted, r) + t

    return np.linalg.norm(
        predicted_aligned - target, axis=len(target.shape) - 1
    )


def _parse_clip_len_from_name(filename: str) -> Optional[int]:
    """Extract clip length from filename suffix '__start_XXX_len_N'."""
    m = re.search(r"__start_\d+_len_(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else None


def _parse_metadata_entry(raw_metadata) -> Dict[str, object]:
    if raw_metadata is None:
        return {}

    parsed = raw_metadata
    if isinstance(parsed, np.ndarray):
        if parsed.shape != ():
            return {}
        parsed = parsed.item()

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, bytes):
        parsed = parsed.decode("utf-8")

    if isinstance(parsed, str):
        try:
            obj = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    return {}


def _extract_robot_control_dt(
    metadata: Dict[str, object], raw_data: Dict[str, np.ndarray]
) -> float:
    if "robot_low_level_torque_dt" in raw_data:
        raw_dt = np.asarray(raw_data["robot_low_level_torque_dt"]).item()
    else:
        raw_dt = metadata.get(
            "robot_low_level_torque_dt",
            metadata.get("robot_control_dt", DEFAULT_ROBOT_CONTROL_DT),
        )
    try:
        robot_control_dt = float(raw_dt)
    except (TypeError, ValueError):
        return DEFAULT_ROBOT_CONTROL_DT

    if not np.isfinite(robot_control_dt) or robot_control_dt <= 0.0:
        return DEFAULT_ROBOT_CONTROL_DT
    return robot_control_dt


def _extract_low_level_contact_dt(
    metadata: Dict[str, object],
    raw_data: Dict[str, np.ndarray],
    robot_control_dt: float,
) -> float:
    if "robot_low_level_contact_dt" in raw_data:
        raw_dt = np.asarray(raw_data["robot_low_level_contact_dt"]).item()
    else:
        raw_dt = metadata.get(
            "robot_low_level_contact_dt",
            metadata.get(
                "robot_low_level_torque_dt",
                metadata.get("robot_control_dt", robot_control_dt),
            ),
        )
    try:
        contact_dt = float(raw_dt)
    except (TypeError, ValueError):
        return robot_control_dt

    if not np.isfinite(contact_dt) or contact_dt <= 0.0:
        return robot_control_dt
    return contact_dt


def _aggregate_sample_metric_to_frames(
    sample_metric: np.ndarray, num_frames: int
) -> np.ndarray:
    if int(sample_metric.shape[0]) == num_frames:
        return sample_metric.astype(float, copy=False)
    if num_frames <= 0:
        return np.empty((0,), dtype=float)

    aggregated = np.full((num_frames,), np.nan, dtype=float)
    for frame_idx, chunk in enumerate(
        np.array_split(sample_metric, num_frames)
    ):
        if chunk.size == 0:
            continue
        if np.all(np.isnan(chunk)):
            continue
        aggregated[frame_idx] = float(np.nanmean(chunk))
    return aggregated


def _compute_torque_jump_series(
    torque_samples: np.ndarray, torque_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = int(torque_samples.shape[0])
    torque_jump_norm = np.full((num_samples,), np.nan, dtype=float)
    torque_jump_ratio = np.full((num_samples,), np.nan, dtype=float)
    if num_samples <= 1:
        return torque_jump_norm, torque_jump_ratio

    torque_mag = np.linalg.norm(torque_samples, axis=1)
    torque_delta_norm = np.linalg.norm(
        torque_samples[1:] - torque_samples[:-1], axis=1
    )
    torque_jump_norm[1:] = torque_delta_norm / torque_dt
    torque_scale = np.maximum(
        np.maximum(torque_mag[1:], torque_mag[:-1]), TORQUE_JUMP_RATIO_EPS
    )
    torque_jump_ratio[1:] = torque_delta_norm / torque_scale
    return torque_jump_norm, torque_jump_ratio


def _safe_nanpercentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def _safe_nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _compute_rolling_nanmean_max(
    values: np.ndarray, window_size: int
) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    if window_size <= 1:
        return float(np.nanmax(arr))

    best = float("nan")
    max_start = int(arr.size) - int(window_size) + 1
    if max_start <= 0:
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    for start in range(max_start):
        window = arr[start : start + window_size]
        if np.all(np.isnan(window)):
            continue
        mean_value = float(np.nanmean(window))
        if np.isnan(best) or mean_value > best:
            best = mean_value
    return best


def _integrate_psd_band(
    frequencies: np.ndarray,
    power_density: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    if (
        not np.isfinite(low_hz)
        or not np.isfinite(high_hz)
        or high_hz <= low_hz
    ):
        return float("nan")
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        return float("nan")
    band_freq = frequencies[band_mask]
    band_power = power_density[band_mask]
    if band_freq.size == 1:
        return float(band_power[0])
    return float(np.trapz(band_power, band_freq))


def _compute_psd_high_frequency_ratio(
    signal_values: np.ndarray,
    sample_dt: float,
    *,
    high_band_low_hz: float,
    band_high_hz: float,
    band_low_hz: float = 0.5,
) -> float:
    samples = np.asarray(signal_values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < MIN_WELCH_SAMPLES:
        return float("nan")

    sample_rate_hz = 1.0 / float(sample_dt)
    max_band_hz = min(float(band_high_hz), 0.45 * sample_rate_hz)
    if max_band_hz <= max(float(band_low_hz), float(high_band_low_hz)):
        return float("nan")

    nperseg = min(int(samples.size), 256)
    frequencies, power_density = welch(
        samples,
        fs=sample_rate_hz,
        nperseg=nperseg,
        detrend="constant",
        average="mean",
    )
    total_power = _integrate_psd_band(
        frequencies, power_density, float(band_low_hz), max_band_hz
    )
    high_power = _integrate_psd_band(
        frequencies, power_density, float(high_band_low_hz), max_band_hz
    )
    if (
        not np.isfinite(total_power)
        or total_power <= 0.0
        or not np.isfinite(high_power)
    ):
        return float("nan")
    return float(high_power / total_power)


def _compute_torque_chatter_hf_ratio(
    low_level_torque: np.ndarray, low_level_dt: float
) -> float:
    torque_samples = np.asarray(low_level_torque, dtype=float)
    if torque_samples.ndim != 2 or torque_samples.shape[0] < MIN_WELCH_SAMPLES:
        return float("nan")

    ratios = []
    for joint_idx in range(int(torque_samples.shape[1])):
        ratio = _compute_psd_high_frequency_ratio(
            torque_samples[:, joint_idx],
            low_level_dt,
            high_band_low_hz=10.0,
            band_high_hz=40.0,
        )
        if np.isfinite(ratio):
            ratios.append(ratio)
    if len(ratios) == 0:
        return float("nan")
    return float(np.mean(ratios))


def _compute_torso_roll_pitch_stability_metrics(
    robot_global_angular_velocity: np.ndarray,
    robot_control_dt: float,
) -> Dict[str, float]:
    angular_velocity = np.asarray(robot_global_angular_velocity, dtype=float)
    if angular_velocity.ndim != 3 or angular_velocity.shape[0] == 0:
        return {
            "torso_rp_hf_ratio": float("nan"),
            "torso_rp_angacc_p95": float("nan"),
        }

    torso_roll_pitch_vel = angular_velocity[:, ROOT_BODY_INDEX, :2]
    torso_roll_pitch_speed = np.linalg.norm(torso_roll_pitch_vel, axis=1)
    hf_ratio = _compute_psd_high_frequency_ratio(
        torso_roll_pitch_speed,
        robot_control_dt,
        high_band_low_hz=5.0,
        band_high_hz=20.0,
    )

    if torso_roll_pitch_vel.shape[0] <= 1:
        angacc_p95 = float("nan")
    else:
        roll_pitch_angacc = np.diff(torso_roll_pitch_vel, axis=0) / float(
            robot_control_dt
        )
        roll_pitch_angacc_mag = np.linalg.norm(roll_pitch_angacc, axis=1)
        angacc_p95 = _safe_nanpercentile(roll_pitch_angacc_mag, 95.0)

    return {
        "torso_rp_hf_ratio": hf_ratio,
        "torso_rp_angacc_p95": angacc_p95,
    }


def _compute_expert_switching_js_div(
    robot_moe_expert_logits: np.ndarray | None,
) -> float:
    if robot_moe_expert_logits is None:
        return float("nan")

    logits = np.asarray(robot_moe_expert_logits, dtype=float)
    if logits.ndim != 3 or logits.shape[0] <= 1 or logits.shape[-1] <= 1:
        return float("nan")

    if not np.all(np.isfinite(logits)):
        return float("nan")

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(shifted_logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)

    prev_probs = np.clip(probs[:-1], PROBABILITY_EPS, 1.0)
    next_probs = np.clip(probs[1:], PROBABILITY_EPS, 1.0)
    mixture = 0.5 * (prev_probs + next_probs)

    kl_prev = np.sum(
        prev_probs * (np.log(prev_probs) - np.log(mixture)), axis=-1
    )
    kl_next = np.sum(
        next_probs * (np.log(next_probs) - np.log(mixture)), axis=-1
    )
    js_divergence = 0.5 * (kl_prev + kl_next) / np.log(2.0)
    return _safe_nanmean(js_divergence)


def _compute_contact_stability_metrics(
    foot_contact_samples: np.ndarray | None,
    foot_normal_force_samples: np.ndarray | None,
    foot_tangent_speed_samples: np.ndarray | None,
    contact_dt: float,
) -> Dict[str, float]:
    metrics = {
        "foot_contact_toggle_rate": float("nan"),
        "foot_impact_force_p95": float("nan"),
        "stance_slip_speed_p95": float("nan"),
    }
    if (
        foot_contact_samples is None
        or foot_normal_force_samples is None
        or foot_tangent_speed_samples is None
    ):
        return metrics

    contact = np.asarray(foot_contact_samples, dtype=float)
    normal_force = np.asarray(foot_normal_force_samples, dtype=float)
    tangent_speed = np.asarray(foot_tangent_speed_samples, dtype=float)
    if (
        contact.shape != normal_force.shape
        or contact.shape != tangent_speed.shape
        or contact.ndim != 2
        or contact.shape[1] != 2
    ):
        return metrics

    finite_contact = np.isfinite(contact)
    if not np.any(finite_contact):
        return metrics

    contact_binary = np.where(contact >= 0.5, 1.0, 0.0)
    valid_pair_mask = finite_contact[1:] & finite_contact[:-1]
    toggle_count = int(
        np.sum(
            np.abs(contact_binary[1:] - contact_binary[:-1]) * valid_pair_mask
        )
    )
    clip_duration_seconds = float(contact.shape[0]) * float(contact_dt)
    if clip_duration_seconds > 0.0:
        metrics["foot_contact_toggle_rate"] = (
            float(toggle_count) / clip_duration_seconds
        )

    touchdown_window = max(
        1, int(round(TOUCHDOWN_WINDOW_SECONDS / float(contact_dt)))
    )
    touchdown_peaks = []
    for foot_idx in range(2):
        foot_contact = contact_binary[:, foot_idx]
        foot_force = normal_force[:, foot_idx]
        onset_mask = np.zeros_like(foot_contact, dtype=bool)
        onset_mask[0] = foot_contact[0] >= 0.5
        onset_mask[1:] = (foot_contact[1:] >= 0.5) & (foot_contact[:-1] < 0.5)
        for onset_idx in np.flatnonzero(onset_mask):
            window = foot_force[onset_idx : onset_idx + touchdown_window]
            if window.size == 0 or np.all(~np.isfinite(window)):
                continue
            touchdown_peaks.append(float(np.nanmax(window)))
    metrics["foot_impact_force_p95"] = _safe_nanpercentile(
        np.asarray(touchdown_peaks, dtype=float), 95.0
    )

    stance_slip_mask = (contact_binary >= 0.5) & np.isfinite(tangent_speed)
    if np.any(stance_slip_mask):
        metrics["stance_slip_speed_p95"] = _safe_nanpercentile(
            tangent_speed[stance_slip_mask], 95.0
        )
    return metrics


def _compute_clip_stability_summary(
    data: Dict[str, np.ndarray],
    robot_control_dt: float,
    low_level_contact_dt: float,
) -> Dict[str, float]:
    robot_low_level_dof_torque = (
        np.asarray(data["robot_low_level_dof_torque"])
        if "robot_low_level_dof_torque" in data
        else None
    )
    if robot_low_level_dof_torque is None and "robot_dof_torque" in data:
        robot_low_level_dof_torque = np.asarray(data["robot_dof_torque"])

    if robot_low_level_dof_torque is None:
        torque_chatter_hf_ratio = float("nan")
        torque_jump_burst_max = float("nan")
    else:
        torque_chatter_hf_ratio = _compute_torque_chatter_hf_ratio(
            robot_low_level_dof_torque, low_level_contact_dt
        )
        _, torque_jump_ratio = _compute_torque_jump_series(
            robot_low_level_dof_torque, low_level_contact_dt
        )
        torque_jump_window = max(
            1,
            int(
                round(
                    STABILITY_BURST_WINDOW_SECONDS
                    / float(low_level_contact_dt)
                )
            ),
        )
        torque_jump_burst_max = _compute_rolling_nanmean_max(
            torque_jump_ratio[1:], torque_jump_window
        )

    torso_metrics = _compute_torso_roll_pitch_stability_metrics(
        np.asarray(data["robot_global_angular_velocity"]),
        robot_control_dt,
    )
    contact_metrics = _compute_contact_stability_metrics(
        np.asarray(data["robot_low_level_foot_contact"])
        if "robot_low_level_foot_contact" in data
        else None,
        np.asarray(data["robot_low_level_foot_normal_force"])
        if "robot_low_level_foot_normal_force" in data
        else None,
        np.asarray(data["robot_low_level_foot_tangent_speed"])
        if "robot_low_level_foot_tangent_speed" in data
        else None,
        low_level_contact_dt,
    )
    expert_switching_js_div = _compute_expert_switching_js_div(
        np.asarray(data["robot_moe_expert_logits"])
        if "robot_moe_expert_logits" in data
        else None
    )
    return {
        "torque_chatter_hf_ratio": torque_chatter_hf_ratio,
        "torque_jump_burst_max": torque_jump_burst_max,
        "expert_switching_js_div": expert_switching_js_div,
        **torso_metrics,
        **contact_metrics,
    }


def _compute_clip_torque_jump_summary(
    data: Dict[str, np.ndarray],
    dof_mode: str,
    torque_dt: float,
) -> Dict[str, float]:
    robot_dof_torque = (
        np.asarray(data["robot_dof_torque"])
        if "robot_dof_torque" in data
        else None
    )
    robot_low_level_dof_torque = (
        np.asarray(data["robot_low_level_dof_torque"])
        if "robot_low_level_dof_torque" in data
        else None
    )

    if dof_mode == "23" and robot_dof_torque is not None:
        total_dofs_in_file = int(robot_dof_torque.shape[1])
        if total_dofs_in_file == 29:
            idx_23_in_29_dof = list(range(19)) + list(range(22, 26))
            robot_dof_torque = robot_dof_torque[:, idx_23_in_29_dof]
            if (
                robot_low_level_dof_torque is not None
                and int(robot_low_level_dof_torque.shape[1])
                == total_dofs_in_file
            ):
                robot_low_level_dof_torque = robot_low_level_dof_torque[
                    :, idx_23_in_29_dof
                ]

    chatter_torque = robot_low_level_dof_torque
    if chatter_torque is None:
        chatter_torque = robot_dof_torque

    if chatter_torque is None or int(chatter_torque.shape[0]) <= 1:
        return {
            "mean_torque_jump_norm": float("nan"),
            "p95_torque_jump_norm": float("nan"),
            "mean_torque_jump_ratio": float("nan"),
            "p95_torque_jump_ratio": float("nan"),
        }

    torque_jump_norm, torque_jump_ratio = _compute_torque_jump_series(
        chatter_torque, torque_dt
    )
    return {
        "mean_torque_jump_norm": float(np.nanmean(torque_jump_norm)),
        "p95_torque_jump_norm": float(
            np.nanpercentile(torque_jump_norm[1:], 95)
        ),
        "mean_torque_jump_ratio": float(np.nanmean(torque_jump_ratio)),
        "p95_torque_jump_ratio": float(
            np.nanpercentile(torque_jump_ratio[1:], 95)
        ),
    }


def _per_frame_metrics_from_npz(
    motion_key: str,
    data: Dict[str, np.ndarray],
    dof_mode: str = "29",
    robot_control_dt: float = DEFAULT_ROBOT_CONTROL_DT,
) -> pd.DataFrame:
    """Compute per-frame metrics for a single motion clip from loaded npz arrays.

    Expects the following keys in `data` (URDF order):
    - dof_pos, robot_dof_pos
    - global_translation, robot_global_translation
    - global_rotation_quat, robot_global_rotation_quat (xyzw)
    """
    # Required arrays
    jpos_gt = np.asarray(data["ref_global_translation"])  # (T, J, 3)
    jpos_pred = np.asarray(data["robot_global_translation"])  # (T, J, 3)
    rot_gt = np.asarray(data["ref_global_rotation_quat"])  # (T, J, 4) xyzw
    rot_pred = np.asarray(data["robot_global_rotation_quat"])  # (T, J, 4)
    dof_gt = np.asarray(data["ref_dof_pos"])  # (T, D)
    dof_pred = np.asarray(data["robot_dof_pos"])  # (T, D)
    robot_dof_vel = (
        np.asarray(data["robot_dof_vel"]) if "robot_dof_vel" in data else None
    )
    robot_dof_acc = (
        np.asarray(data["robot_dof_acc"]) if "robot_dof_acc" in data else None
    )
    robot_dof_torque = (
        np.asarray(data["robot_dof_torque"])
        if "robot_dof_torque" in data
        else None
    )
    robot_low_level_dof_torque = (
        np.asarray(data["robot_low_level_dof_torque"])
        if "robot_low_level_dof_torque" in data
        else None
    )
    robot_action_rate = (
        np.asarray(data["robot_action_rate"])
        if "robot_action_rate" in data
        else None
    )

    total_dofs_in_file = int(dof_gt.shape[1])
    IDX_23_IN_29_DOF = list(range(19)) + list(range(22, 26))
    IDX_23_IN_29_BODY = [0] + [i + 1 for i in IDX_23_IN_29_DOF]

    if dof_mode == "23":
        if total_dofs_in_file == 29:
            dof_gt = dof_gt[:, IDX_23_IN_29_DOF]
            dof_pred = dof_pred[:, IDX_23_IN_29_DOF]
            if (
                robot_dof_vel is not None
                and int(robot_dof_vel.shape[1]) == total_dofs_in_file
            ):
                robot_dof_vel = robot_dof_vel[:, IDX_23_IN_29_DOF]
            if (
                robot_dof_acc is not None
                and int(robot_dof_acc.shape[1]) == total_dofs_in_file
            ):
                robot_dof_acc = robot_dof_acc[:, IDX_23_IN_29_DOF]
            if (
                robot_dof_torque is not None
                and int(robot_dof_torque.shape[1]) == total_dofs_in_file
            ):
                robot_dof_torque = robot_dof_torque[:, IDX_23_IN_29_DOF]
            if (
                robot_low_level_dof_torque is not None
                and int(robot_low_level_dof_torque.shape[1])
                == total_dofs_in_file
            ):
                robot_low_level_dof_torque = robot_low_level_dof_torque[
                    :, IDX_23_IN_29_DOF
                ]

            jpos_gt = jpos_gt[:, IDX_23_IN_29_BODY, :]
            jpos_pred = jpos_pred[:, IDX_23_IN_29_BODY, :]

            rot_gt = rot_gt[:, IDX_23_IN_29_BODY, :]
            rot_pred = rot_pred[:, IDX_23_IN_29_BODY, :]

    assert jpos_gt.shape == jpos_pred.shape
    assert rot_gt.shape == rot_pred.shape
    assert dof_gt.shape == dof_pred.shape

    num_frames = int(jpos_gt.shape[0])

    # Global MPJPE [mm]
    mpjpe_g = (
        np.mean(np.linalg.norm(jpos_gt - jpos_pred, axis=2), axis=1) * 1000.0
    )

    # Per-frame maximum body-link position error [m] (used for failure criterion)
    # per_joint_err = np.linalg.norm(jpos_pred - jpos_gt, axis=2)
    # frame_max_body_pos_err = np.max(per_joint_err, axis=1)
    frame_max_body_pos_err = np.abs(jpos_pred[:, 0, 2] - jpos_gt[:, 0, 2])

    # Localize by root (index 0)
    jpos_gt_local = jpos_gt - jpos_gt[:, [0]]
    jpos_pred_local = jpos_pred - jpos_pred[:, [0]]
    ref_body_pos_root_rel = quat_apply(
        quat_inv(rot_gt[:, 0, :]),
        jpos_gt - jpos_gt[:, [0]],
    )
    robot_body_pos_root_rel = quat_apply(
        quat_inv(rot_pred[:, 0, :]),
        jpos_pred - jpos_pred[:, [0]],
    )

    mpjpe_l = (
        np.mean(
            np.linalg.norm(
                robot_body_pos_root_rel - ref_body_pos_root_rel, axis=2
            ),
            axis=1,
        )
        * 1000.0
    )

    # Procrustes-aligned MPJPE [mm]
    pa_per_joint = p_mpjpe(jpos_pred_local, jpos_gt_local)
    mpjpe_pa = np.mean(pa_per_joint, axis=1) * 1000.0

    # Velocity/acceleration errors from positions (discrete frame diffs) [mm/frame],[mm/frame^2]
    vel_gt = jpos_gt[1:] - jpos_gt[:-1]
    vel_pred = jpos_pred[1:] - jpos_pred[:-1]
    vel_dist = (
        np.mean(np.linalg.norm(vel_pred - vel_gt, axis=2), axis=1) * 1000.0
    )

    acc_gt = jpos_gt[:-2] - 2 * jpos_gt[1:-1] + jpos_gt[2:]
    acc_pred = jpos_pred[:-2] - 2 * jpos_pred[1:-1] + jpos_pred[2:]
    accel_dist = (
        np.mean(np.linalg.norm(acc_pred - acc_gt, axis=2), axis=1) * 1000.0
    )

    # DOF angle errors [radians] — whole body average
    dof_err = np.abs(dof_pred - dof_gt)
    whole_body_joints_dist = np.mean(dof_err, axis=1)

    # Root orientation errors [radians] — handle zero-norm/invalid quaternions by NaN
    q_gt_root = rot_gt[:, 0, :]
    q_pred_root = rot_pred[:, 0, :]
    norms_gt = np.linalg.norm(q_gt_root, axis=1)
    norms_pred = np.linalg.norm(q_pred_root, axis=1)
    valid_mask = (
        (norms_gt > 0.0)
        & (norms_pred > 0.0)
        & np.isfinite(norms_gt)
        & np.isfinite(norms_pred)
    )

    root_r_error = np.full((num_frames,), np.nan, dtype=float)
    root_p_error = np.full((num_frames,), np.nan, dtype=float)
    root_y_error = np.full((num_frames,), np.nan, dtype=float)

    if np.any(valid_mask):
        q_gt_valid = q_gt_root[valid_mask] / norms_gt[valid_mask, None]
        q_pred_valid = q_pred_root[valid_mask] / norms_pred[valid_mask, None]
        rel_valid = sRot.from_quat(q_gt_valid).inv() * sRot.from_quat(
            q_pred_valid
        )
        euler_xyz = rel_valid.as_euler("xyz", degrees=False)
        root_r_error[valid_mask] = np.abs(euler_xyz[:, 0])
        root_p_error[valid_mask] = np.abs(euler_xyz[:, 1])
        root_y_error[valid_mask] = np.abs(euler_xyz[:, 2])

    # Root velocity error [m/frame]
    root_pos_gt = jpos_gt[:, 0, :]
    root_pos_pred = jpos_pred[:, 0, :]
    root_vel_err = np.linalg.norm(
        (root_pos_pred[1:] - root_pos_pred[:-1])
        - (root_pos_gt[1:] - root_pos_gt[:-1]),
        axis=1,
    )

    # Root height error [m]
    root_height_error = np.abs(root_pos_pred[:, 2] - root_pos_gt[:, 2])

    # Robot low-level magnitudes (optional)
    mean_dof_vel = np.full((num_frames,), np.nan, dtype=float)
    if robot_dof_vel is not None:
        if int(robot_dof_vel.shape[0]) != num_frames:
            raise ValueError(
                "robot_dof_vel frame length mismatch: "
                f"{robot_dof_vel.shape[0]} vs {num_frames}"
            )
        mean_dof_vel = np.linalg.norm(robot_dof_vel, axis=1)

    mean_dof_acc = np.full((num_frames,), np.nan, dtype=float)
    if robot_dof_acc is not None:
        if int(robot_dof_acc.shape[0]) != num_frames:
            raise ValueError(
                "robot_dof_acc frame length mismatch: "
                f"{robot_dof_acc.shape[0]} vs {num_frames}"
            )
        mean_dof_acc = np.linalg.norm(robot_dof_acc, axis=1)

    mean_dof_torque = np.full((num_frames,), np.nan, dtype=float)
    mean_torque_jump_norm = np.full((num_frames,), np.nan, dtype=float)
    mean_torque_jump_ratio = np.full((num_frames,), np.nan, dtype=float)
    if robot_dof_torque is not None:
        if int(robot_dof_torque.shape[0]) != num_frames:
            raise ValueError(
                "robot_dof_torque frame length mismatch: "
                f"{robot_dof_torque.shape[0]} vs {num_frames}"
            )
        mean_dof_torque = np.linalg.norm(robot_dof_torque, axis=1)
    chatter_torque = robot_low_level_dof_torque
    if chatter_torque is None:
        chatter_torque = robot_dof_torque
    if chatter_torque is not None and int(chatter_torque.shape[0]) > 1:
        torque_jump_norm, torque_jump_ratio = _compute_torque_jump_series(
            chatter_torque, robot_control_dt
        )
        mean_torque_jump_norm = _aggregate_sample_metric_to_frames(
            torque_jump_norm, num_frames
        )
        mean_torque_jump_ratio = _aggregate_sample_metric_to_frames(
            torque_jump_ratio, num_frames
        )

    mean_action_rate = np.full((num_frames,), np.nan, dtype=float)
    if robot_action_rate is not None:
        flat_action_rate = robot_action_rate.reshape(-1)
        if int(flat_action_rate.shape[0]) != num_frames:
            raise ValueError(
                "robot_action_rate frame length mismatch: "
                f"{flat_action_rate.shape[0]} vs {num_frames}"
            )
        mean_action_rate = flat_action_rate

    # Frame DataFrame (align lengths by padding NaN at the start where needed)
    def pad_front(x: np.ndarray, pad: int) -> np.ndarray:
        if pad <= 0:
            return x
        return np.concatenate(
            [np.full((pad,), np.nan, dtype=float), x], axis=0
        )

    df = pd.DataFrame(
        {
            "motion_key": [motion_key] * num_frames,
            "frame_idx": np.arange(num_frames, dtype=int),
            "mpjpe_g": mpjpe_g,
            "mpjpe_l": mpjpe_l,
            "mpjpe_pa": mpjpe_pa,
            "vel_dist": pad_front(vel_dist, 1),
            "accel_dist": pad_front(accel_dist, 2),
            "frame_max_body_pos_err": frame_max_body_pos_err,
            "whole_body_joints_dist": whole_body_joints_dist,
            "root_r_error": root_r_error,
            "root_p_error": root_p_error,
            "root_y_error": root_y_error,
            "root_vel_error": pad_front(root_vel_err, 1),
            "root_height_error": root_height_error,
            "mean_dof_vel": mean_dof_vel,
            "mean_dof_acc": mean_dof_acc,
            "mean_dof_torque": mean_dof_torque,
            "mean_torque_jump_norm": mean_torque_jump_norm,
            "mean_torque_jump_ratio": mean_torque_jump_ratio,
            "mean_action_rate": mean_action_rate,
        }
    )
    return df


def offline_evaluate_dumped_npzs(
    npz_dir: str,
    output_json_path: str,
    failure_pos_err_thresh_m: float = 0.25,
    metric_calculation: str = "per_clip",
    dof_mode: str = "29",
    threadpool_max_workers: Optional[int] = None,
) -> Dict[str, dict]:
    """Evaluate dumped NPZs in `npz_dir` and write a JSON summary to `output_dir`.

    The function produces dataset-wide averages and per-clip averages across frames.
    """
    npz_dir_abs = Path(npz_dir).resolve()
    os.makedirs(npz_dir_abs, exist_ok=True)

    # Add file handler for logging to metric.log
    metric_log_path = npz_dir_abs / "metric.log"
    logger.add(
        str(metric_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )

    logger.info(f"Input NPZ directory (absolute): {npz_dir_abs}")

    # Gather NPZ files
    files = sorted(glob(os.path.join(npz_dir_abs, "*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found in: {npz_dir_abs}")

    # Accumulate per-frame metrics
    frame_tables: List[pd.DataFrame] = []
    clip_meta: Dict[str, dict] = {}

    skipped_files_count = 0

    required_keys = [
        "ref_dof_pos",
        "ref_dof_vel",
        "ref_global_translation",
        "ref_global_rotation_quat",
        "ref_global_velocity",
        "ref_global_angular_velocity",
        "robot_dof_pos",
        "robot_dof_vel",
        "robot_global_translation",
        "robot_global_rotation_quat",
        "robot_global_velocity",
        "robot_global_angular_velocity",
    ]
    optional_keys = [
        "robot_dof_acc",
        "robot_dof_torque",
        "robot_low_level_dof_torque",
        "robot_low_level_torque_dt",
        "robot_low_level_foot_contact",
        "robot_low_level_foot_normal_force",
        "robot_low_level_foot_tangent_speed",
        "robot_low_level_contact_dt",
        "robot_action_rate",
        "robot_moe_expert_indices",
        "robot_moe_expert_logits",
    ]

    def _compute_metrics_from_file(fpath: str):
        try:
            with np.load(fpath, allow_pickle=True) as npz_data:
                # Extract arrays and metadata
                data = {k: npz_data[k] for k in required_keys}
                for k in optional_keys:
                    if k in npz_data.files:
                        data[k] = npz_data[k]

                metadata = _parse_metadata_entry(npz_data.get("metadata"))
                robot_control_dt = _extract_robot_control_dt(metadata, data)
                low_level_contact_dt = _extract_low_level_contact_dt(
                    metadata, data, robot_control_dt
                )

            motion_key = os.path.splitext(os.path.basename(fpath))[0]
            clip_len_from_name = _parse_clip_len_from_name(fpath)

            df_frames = _per_frame_metrics_from_npz(
                motion_key=motion_key,
                data=data,
                dof_mode=dof_mode,
                robot_control_dt=robot_control_dt,
            )
            chatter_summary = _compute_clip_torque_jump_summary(
                data=data, dof_mode=dof_mode, torque_dt=robot_control_dt
            )
            stability_summary = _compute_clip_stability_summary(
                data=data,
                robot_control_dt=robot_control_dt,
                low_level_contact_dt=low_level_contact_dt,
            )

            # Clip-level info and failure criterion (max body-link pos error > threshold)
            num_frames_clip = int(df_frames.shape[0])
            clip_length = int(
                metadata.get(
                    "clip_length", clip_len_from_name or num_frames_clip
                )
            )
            max_body_err = float(
                np.nanmax(df_frames["frame_max_body_pos_err"].to_numpy())
            )
            success = 1.0 if max_body_err <= failure_pos_err_thresh_m else 0.0
            clip_meta_entry = {
                "motion_key": motion_key,
                "num_frames": num_frames_clip,
                "clip_length": clip_length,
                "success": success,
                "max_body_pos_err": max_body_err,
                "failure_threshold_m": float(failure_pos_err_thresh_m),
                **chatter_summary,
                **stability_summary,
            }
            return fpath, df_frames, motion_key, clip_meta_entry, None
        except (ValueError, KeyError, BadZipFile, EOFError, OSError) as e:
            return fpath, None, None, None, e

    if threadpool_max_workers is None:
        max_workers = max(1, min(len(files), 24))
        requested_workers = None
    else:
        requested_workers = int(threadpool_max_workers)
        if requested_workers <= 0:
            raise ValueError("threadpool_max_workers must be > 0")
        max_workers = min(requested_workers, len(files))
        if max_workers <= 0:
            max_workers = 1
    logger.info(
        f"Metric ThreadPoolExecutor max_workers={max_workers} "
        f"(requested={requested_workers}, num_npz_files={len(files)})"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_metrics_from_file, fpath): file_idx
            for file_idx, fpath in enumerate(files)
        }
        processed_results = [None] * len(files)
        for future in tqdm(
            as_completed(futures.keys()),
            total=len(files),
            desc="Compute metrics from NPZs",
        ):
            processed_results[futures[future]] = future.result()

    for result in processed_results:
        (
            fpath,
            df_frames,
            motion_key,
            clip_meta_entry,
            file_error,
        ) = result
        if file_error is not None:
            logger.warning(f"\nCaught an error while processing file: {fpath}")
            logger.warning(f"Error type: {type(file_error).__name__}")
            logger.warning(f"Error message: {file_error}")
            logger.warning("This file will be SKIPPED.")
            skipped_files_count += 1
            continue
        frame_tables.append(df_frames)
        clip_meta[motion_key] = clip_meta_entry

    if skipped_files_count > 0:
        logger.info(
            f"\nFinished processing. Skipped a total of {skipped_files_count} files due to errors."
        )

    # If all files were skipped, there's nothing to process further.
    if not frame_tables:
        logger.error(
            "No valid NPZ files could be processed. Aborting evaluation."
        )
        return {}

    # Concatenate per-frame metrics
    all_frames = pd.concat(frame_tables, ignore_index=True)

    # Per-clip averages
    frame_metric_cols = [
        "mpjpe_g",
        "mpjpe_l",
        "whole_body_joints_dist",
        "root_vel_error",
        "root_r_error",
        "root_p_error",
        "root_y_error",
        "root_height_error",
        "mean_dof_vel",
        "mean_dof_acc",
        "mean_dof_torque",
        "mean_torque_jump_norm",
        "mean_torque_jump_ratio",
        "mean_action_rate",
    ]
    percentile_metric_cols = [
        "mean_torque_jump_norm",
        "mean_torque_jump_ratio",
    ]
    percentile_rename_map = {
        "mean_torque_jump_norm": "p95_torque_jump_norm",
        "mean_torque_jump_ratio": "p95_torque_jump_ratio",
    }
    metric_cols = frame_metric_cols + list(percentile_rename_map.values())
    clip_only_metric_cols = [
        "torque_chatter_hf_ratio",
        "torque_jump_burst_max",
        "expert_switching_js_div",
        "torso_rp_hf_ratio",
        "torso_rp_angacc_p95",
        "foot_contact_toggle_rate",
        "foot_impact_force_p95",
        "stance_slip_speed_p95",
    ]
    metric_cols += clip_only_metric_cols
    # Metric display configuration: metric_key -> (display_name, unit)
    metric_display_map = {
        "mpjpe_g": ("Global Bodylink Mean Position Error", "mm"),
        "mpjpe_l": ("Local Bodylink Mean Position Error", "mm"),
        "whole_body_joints_dist": ("DOF Position Error", "rad"),
        "root_vel_error": ("Root Velocity Error", "m/s"),
        "root_r_error": ("Root Roll Error", "rad"),
        "root_p_error": ("Root Pitch Error", "rad"),
        "root_y_error": ("Root Yaw Error", "rad"),
        "root_height_error": ("Root Height Error", "mm"),
        "mean_dof_vel": ("Mean DOF Velocity", "rad/s"),
        "mean_dof_acc": ("Mean DOF Acceleration", "rad/s^2"),
        "mean_dof_torque": ("Mean DOF Torque", "N*m"),
        "mean_torque_jump_norm": ("Mean Torque Jump Norm", "N*m/s"),
        "p95_torque_jump_norm": ("P95 Torque Jump Norm", "N*m/s"),
        "mean_torque_jump_ratio": ("Mean Torque Jump Ratio", "ratio"),
        "p95_torque_jump_ratio": ("P95 Torque Jump Ratio", "ratio"),
        "mean_action_rate": ("Mean Action Rate", "1/s"),
        "torque_chatter_hf_ratio": ("Torque Chatter HF Ratio", "ratio"),
        "torque_jump_burst_max": ("Torque Jump Burst Max", "ratio"),
        "expert_switching_js_div": ("Expert Switching JS Div", "bits"),
        "torso_rp_hf_ratio": ("Torso RP HF Ratio", "ratio"),
        "torso_rp_angacc_p95": ("Torso RP Angular Accel P95", "rad/s^2"),
        "foot_contact_toggle_rate": ("Foot Contact Toggle Rate", "1/s"),
        "foot_impact_force_p95": ("Foot Impact Force P95", "N"),
        "stance_slip_speed_p95": ("Stance Slip Speed P95", "m/s"),
    }

    per_clip_mean = (
        all_frames.groupby("motion_key")[frame_metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    per_clip_p95 = (
        all_frames.groupby("motion_key")[percentile_metric_cols]
        .quantile(0.95)
        .reset_index()
        .rename(columns=percentile_rename_map)
    )
    per_clip_summary = per_clip_mean.merge(
        per_clip_p95, on="motion_key", how="left"
    )
    for metric_key in (
        "mean_torque_jump_norm",
        "p95_torque_jump_norm",
        "mean_torque_jump_ratio",
        "p95_torque_jump_ratio",
    ):
        per_clip_summary[metric_key] = per_clip_summary["motion_key"].map(
            {mk: clip_meta[mk].get(metric_key, np.nan) for mk in clip_meta}
        )
    for metric_key in clip_only_metric_cols:
        per_clip_summary[metric_key] = per_clip_summary["motion_key"].map(
            {mk: clip_meta[mk].get(metric_key, np.nan) for mk in clip_meta}
        )

    # Merge with success flags
    per_clip_records = []
    for _, row in per_clip_summary.iterrows():
        mk = row["motion_key"]
        rec = {**row.to_dict(), **clip_meta.get(mk, {})}
        per_clip_records.append(rec)

    # Persist per-clip metrics as a tabular CSV for easier downstream analysis.
    per_clip_df = pd.DataFrame(per_clip_records)
    output_csv_path = str(npz_dir_abs / "per_clip_metrics.csv")
    per_clip_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved per-clip metrics CSV to: {output_csv_path}")

    dataset_means = {}
    dataset_medians = {}
    if metric_calculation == "per_frame":
        agg_source = all_frames
        agg_desc = "PER-FRAME"
    else:
        agg_source = per_clip_summary
        agg_desc = "PER-CLIP"
    for k in metric_cols:
        if k in agg_source.columns:
            arr = agg_source[k].to_numpy()
        else:
            arr = per_clip_summary[k].to_numpy()
        dataset_means[k] = _safe_nanmean(arr)
        dataset_medians[k] = _safe_nanmedian(arr)

    success_rate = float(
        np.mean([clip_meta[mk]["success"] for mk in clip_meta])
        if len(clip_meta) > 0
        else 0.0
    )
    dataset_means["success_rate"] = success_rate

    # Compose result and write
    result = {
        "dataset": {
            "calculation_mode": metric_calculation,
            "mean": dataset_means,
            "median": dataset_medians,
            "success_rate": success_rate,
        },
        "num_clips": int(len(clip_meta)),
        "per_clip": per_clip_records,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Conversion factors for unit conversion (assuming 50Hz)
    frame_rate_hz = 50.0
    unit_conversions = {
        "root_height_error": 1000.0,  # m to mm
        "root_vel_error": frame_rate_hz,  # m/frame to m/s
    }

    table_data = []
    # Iterate through metric_display_map to preserve order
    for key in metric_display_map.keys():
        if key not in dataset_means:
            continue

        val_mean = dataset_means[key]
        val_median = dataset_medians[key]
        display_name, unit = metric_display_map[key]

        # Apply unit conversion if needed
        if key in unit_conversions:
            factor = unit_conversions[key]
            val_mean = val_mean * factor
            val_median = val_median * factor

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        table_data.append([display_name, fmt(val_mean), fmt(val_median), unit])

    table_headers = ["Metric", "Mean", "Median", "Unit"]
    output_tsv_path = str(npz_dir_abs / "whole_dataset_metrics.tsv")
    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(table_headers)
        writer.writerows(table_data)
    logger.info(f"Saved whole-dataset metrics TSV to: {output_tsv_path}")

    table_str = tabulate(
        table_data,
        headers=table_headers,
        tablefmt="simple_outline",
        colalign=("left", "left", "left", "left"),
    )
    logger.info(
        "\n"
        + "=" * 80
        + f"\nDATASET-WISE METRICS ({agg_desc})\n"
        + "=" * 80
        + f"\n\n{table_str}\n"
        + "=" * 80
        + "\n"
    )

    return result


def parse_ckpt_and_dataset_from_eval_dirname(
    eval_dir_name: str, dataset_suffix: str
):
    VALID_PREFIXES = ["isaaclab_eval_output_", "mujoco_eval_output_"]

    matched_prefix = None
    for prefix in VALID_PREFIXES:
        if eval_dir_name.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        return None, None

    rest = eval_dir_name[len(matched_prefix) :]
    if not rest.endswith(dataset_suffix):
        return None, None

    model_part = rest[: -len(dataset_suffix)]
    if model_part.endswith("_"):
        model_part = model_part[:-1]

    m = re.search(r"model_(\d+)$", model_part)
    if not m:
        return None, dataset_suffix

    return m.group(1), dataset_suffix


def run_evaluation(
    npz_dir: str,
    dataset_suffix: str,
    failure_pos_err_thresh_m: float = 0.25,
    metric_calculation: str = "per_clip",
    dof_mode: str = "29",
    threadpool_max_workers: Optional[int] = None,
):
    """
    Main function to run evaluation. It scans a root directory, runs evaluation
    for each found subdirectory, and generates a final summary report.

    Args:
        npz_dir (str): Top-level directory containing all model evaluation results (e.g., 'logs/test').
        output_dir (str): Directory to store all generated JSON files and logs.
        failure_pos_err_thresh_m (float): The position error threshold in meters to determine a failure.
    """
    root_path = Path(npz_dir)

    logger.info(f"Starting batch evaluation. Root directory: '{root_path}'")
    logger.info(
        f"Searching for directories matching pattern: '{dataset_suffix}'"
    )

    def has_npz_files(path: Path) -> bool:
        return path.is_dir() and any(path.glob("*.npz"))

    is_single_eval_dir = (
        root_path.is_dir()
        and (
            root_path.name.startswith("isaaclab_eval_output_")
            or root_path.name.startswith("mujoco_eval_output_")
        )
        and has_npz_files(root_path)
    )

    if is_single_eval_dir:
        output_path = root_path
    else:
        output_path = root_path / f"metrics_output_{dataset_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    if is_single_eval_dir:
        logger.info(
            f"Detected '{root_path}' as a single evaluation directory. "
            "Running offline evaluation only for this directory."
        )
        model_name = root_path.parent.name

        ckpt_str, ds = parse_ckpt_and_dataset_from_eval_dirname(
            root_path.name, dataset_suffix
        )
        if ckpt_str is None:
            logger.warning(
                f"Could not parse checkpoint/dataset from directory name '{root_path.name}'. "
                "Using 'checkpoint_unknown' in output filename."
            )
            ckpt_str = "checkpoint_unknown"
            ds = dataset_suffix

        output_json_name = f"{model_name}_{ckpt_str}_{dof_mode}dof.json"
        output_json_path = output_path / output_json_name

        offline_evaluate_dumped_npzs(
            npz_dir=str(root_path),
            output_json_path=str(output_json_path),
            failure_pos_err_thresh_m=failure_pos_err_thresh_m,
            metric_calculation=metric_calculation,
            dof_mode=dof_mode,
            threadpool_max_workers=threadpool_max_workers,
        )
        logger.success(
            f"Finished single-directory evaluation: model='{model_name}', checkpoint={ckpt_str}"
        )
        return
    logger.info(
        f"Treating '{root_path}' as root directory for batch evaluation."
    )
    # Find all directories matching the evaluation output pattern.
    eval_dirs = sorted(
        p
        for p in root_path.glob(f"**/*eval_output_*_{dataset_suffix}")
        if p.is_dir()
    )
    if not eval_dirs:
        logger.error(
            f"No directories matching the pattern '{dataset_suffix}' found under '{root_path}'. "
            "Please check the path and pattern."
        )
        return

    all_results = []

    # Process each found evaluation directory.
    for eval_dir in tqdm(eval_dirs, desc="Overall Progress"):
        # Extract model name from the parent directory.
        model_name = eval_dir.parent.name
        # Parse the checkpoint number from the directory name.
        ckpt_str, ds = parse_ckpt_and_dataset_from_eval_dirname(
            eval_dir.name, dataset_suffix
        )
        if ckpt_str is None:
            logger.warning(
                f"Could not parse ckpt/dataset from '{eval_dir.name}'. Skipping."
            )
            continue

        checkpoint = int(ckpt_str)

        logger.info(
            f"\n--- Processing: model='{model_name}', dataset='{ds}', checkpoint={checkpoint} ---"
        )

        # Construct a unique output JSON filename.
        output_json_name = f"{model_name}_{checkpoint}.json"
        output_json_path = output_path / output_json_name

        # Call the evaluation function for the current directory.
        result = offline_evaluate_dumped_npzs(
            npz_dir=str(eval_dir),
            output_json_path=str(output_json_path),
            failure_pos_err_thresh_m=failure_pos_err_thresh_m,
            metric_calculation=metric_calculation,
            dof_mode=dof_mode,
            threadpool_max_workers=threadpool_max_workers,
        )

        if result and "dataset" in result:
            # Collect dataset-level average metrics for the final summary.
            flat_result = {
                "model": model_name,
                "checkpoint": checkpoint,
                **result["dataset"],
            }
            all_results.append(flat_result)
            logger.success(
                f"--- Finished processing: model='{model_name}', checkpoint={checkpoint} ---"
            )
        else:
            logger.error(
                f"--- Failed to process: model='{model_name}', checkpoint={checkpoint} ---"
            )

    if not all_results:
        logger.error(
            "No evaluations succeeded. Cannot generate a summary report."
        )
        return

    logger.info("\n" + "=" * 80)
    logger.info("Batch evaluation finished successfully.")
    logger.info(f"Total successful evaluations: {len(all_results)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--npz_dir", type=str, required=True)
    argument_parser.add_argument(
        "--dataset_suffix",
        type=str,
        required=True,
    )
    argument_parser.add_argument(
        "--failure_pos_err_thresh_m", type=float, default=0.25
    )
    argument_parser.add_argument(
        "--metric_calculation",
        type=str,
        choices=["per_clip", "per_frame"],
        default="per_clip",
        help="Calculation mode for dataset metrics. 'per_clip' averages clip means (Macro). 'per_frame' averages all frames (Micro).",
    )
    argument_parser.add_argument(
        "--dof_mode",
        type=str,
        choices=["29", "23"],
        default="29",
        help="Compute metrics for full 29 DoF or reduced 23 DoF (excluding hands).",
    )
    argument_parser.add_argument(
        "--threadpool_max_workers",
        type=int,
        default=None,
        help="Max workers for per-NPZ ThreadPoolExecutor. "
        "Default: None (auto = min(num_files, 24)).",
    )
    args = argument_parser.parse_args()

    run_evaluation(
        npz_dir=args.npz_dir,
        dataset_suffix=args.dataset_suffix,
        failure_pos_err_thresh_m=args.failure_pos_err_thresh_m,
        metric_calculation=args.metric_calculation,
        dof_mode=args.dof_mode,
        threadpool_max_workers=args.threadpool_max_workers,
    )
