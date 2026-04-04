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


from typing import Dict, Mapping, Tuple

import numpy as np

# This module keeps the offline preprocess filtering path and the online
# root/DoF-before-FK path aligned while still exposing helpers tailored to
# each tensor family.


def _reshape_time_flat(a: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    shape = a.shape
    t = shape[0]
    return a.reshape(t, -1), shape


def _butterworth_lowpass_smooth_time(
    a: np.ndarray, fps: float, cutoff_hz: float, order: int
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    t = a.shape[0]
    if t < 3:
        return a.astype(np.float32, copy=True)
    if fps <= 0.0 or cutoff_hz <= 0.0:
        return a.astype(np.float32, copy=True)
    nyquist = 0.5 * float(fps)
    wn = float(cutoff_hz) / nyquist
    if wn >= 1.0:
        wn = 0.999
    if wn <= 0.0:
        return a.astype(np.float32, copy=True)
    flat, shape = _reshape_time_flat(a.astype(np.float64, copy=False))
    b, a_coefs = butter(int(order), wn, btype="low", analog=False)
    maxlen = max(len(b), len(a_coefs))
    padlen_required = max(3 * (maxlen - 1), 3 * maxlen)
    if t <= padlen_required:
        return a.astype(np.float32, copy=True)
    filtered = filtfilt(b, a_coefs, flat, axis=0, method="pad")
    return filtered.reshape(shape).astype(np.float32, copy=False)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (q / norm).astype(np.float32, copy=False)


def _quat_hemisphere_align(q: np.ndarray) -> np.ndarray:
    if q.shape[0] == 0:
        return q
    aligned = q.copy()
    prev = aligned[0]
    for t in range(1, aligned.shape[0]):
        dots = np.sum(prev * aligned[t], axis=-1)
        mask = dots < 0.0
        if np.any(mask):
            aligned[t, mask] = -aligned[t, mask]
        prev = aligned[t]
    return aligned


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    conj = q.copy()
    conj[..., :3] = -conj[..., :3]
    return conj


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av = a[..., :3]
    aw = a[..., 3:4]
    bv = b[..., :3]
    bw = b[..., 3:4]
    cross = np.cross(av, bv)
    vec = aw * bv + bw * av + cross
    scalar = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    return np.concatenate([vec, scalar], axis=-1)


def _finite_difference_time(a: np.ndarray, dt: float) -> np.ndarray:
    t = a.shape[0]
    if t < 2 or dt <= 0.0:
        return np.zeros_like(a, dtype=np.float32)
    deriv = np.gradient(
        a.astype(np.float64, copy=False),
        dt,
        axis=0,
        edge_order=2 if t >= 3 else 1,
    )
    return deriv.astype(np.float32, copy=False)


def _angular_velocity_from_quat(
    q: np.ndarray, q_dot: np.ndarray
) -> np.ndarray:
    q_conj = _quat_conjugate(q)
    prod = _quat_multiply(q_conj, q_dot)
    omega = 2.0 * prod[..., :3]
    return omega.astype(np.float32, copy=False)


def butterworth_filter_ref_arrays(
    arrays: Mapping[str, np.ndarray],
    fps: float,
    cutoff_hz: float,
    order: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    dt = 1.0 / float(fps) if float(fps) > 0.0 else 0.0
    if "ref_dof_pos" in arrays:
        dof_pos = arrays["ref_dof_pos"].astype(np.float32, copy=True)
        smooth_dof_pos = _butterworth_lowpass_smooth_time(
            dof_pos, fps, cutoff_hz, order
        )
        dof_vel = _finite_difference_time(smooth_dof_pos, dt)
        out["ft_ref_dof_pos"] = smooth_dof_pos
        out["ft_ref_dof_vel"] = dof_vel
    if "ref_global_translation" in arrays:
        body_pos = arrays["ref_global_translation"].astype(
            np.float32, copy=True
        )
        smooth_body_pos = _butterworth_lowpass_smooth_time(
            body_pos, fps, cutoff_hz, order
        )
        body_vel = _finite_difference_time(smooth_body_pos, dt)
        out["ft_ref_global_translation"] = smooth_body_pos
        out["ft_ref_global_velocity"] = body_vel
    if "ref_global_rotation_quat" in arrays:
        body_rot = arrays["ref_global_rotation_quat"].astype(
            np.float32, copy=True
        )
        body_rot = _quat_normalize(body_rot)
        body_rot = _quat_hemisphere_align(body_rot)
        smooth_body_rot = _butterworth_lowpass_smooth_time(
            body_rot, fps, cutoff_hz, order
        )
        smooth_body_rot = _quat_normalize(smooth_body_rot)
        body_rot_dot = _finite_difference_time(smooth_body_rot, dt)
        out["ft_ref_global_rotation_quat"] = _quat_normalize(smooth_body_rot)
        out["ft_ref_global_angular_velocity"] = _angular_velocity_from_quat(
            smooth_body_rot, body_rot_dot
        )
    return out


def butterworth_filter_root_dof_arrays(
    arrays: Mapping[str, np.ndarray],
    fps: float,
    cutoff_hz: float,
    order: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if "ref_root_pos" in arrays:
        root_pos = arrays["ref_root_pos"].astype(np.float32, copy=True)
        out["ft_ref_root_pos"] = _butterworth_lowpass_smooth_time(
            root_pos, fps, cutoff_hz, order
        )
    if "ref_root_rot" in arrays:
        root_rot = arrays["ref_root_rot"].astype(np.float32, copy=True)
        root_rot = _quat_normalize(root_rot)
        root_rot = _quat_hemisphere_align(root_rot)
        smooth_root_rot = _butterworth_lowpass_smooth_time(
            root_rot, fps, cutoff_hz, order
        )
        out["ft_ref_root_rot"] = _quat_normalize(smooth_root_rot)
    if "ref_dof_pos" in arrays:
        dof_pos = arrays["ref_dof_pos"].astype(np.float32, copy=True)
        out["ft_ref_dof_pos"] = _butterworth_lowpass_smooth_time(
            dof_pos, fps, cutoff_hz, order
        )
    return out
