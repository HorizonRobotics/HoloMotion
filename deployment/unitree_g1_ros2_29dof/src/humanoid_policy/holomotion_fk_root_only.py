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


from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Sequence

import numpy as np
import torch


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., 1:4] *= -1.0
    return out


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]
    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _standardize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    return np.where(q[..., 0:1] < 0.0, -q, q)


def _axis_angle_from_wxyz(q: np.ndarray) -> np.ndarray:
    q = _standardize_quaternion_wxyz(q)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1.0e-9)
    quat_w = q[..., 0]
    quat_xyz = q[..., 1:4]
    mag = np.linalg.norm(quat_xyz, axis=-1)
    half_angle = np.arctan2(mag, quat_w)
    angle = 2.0 * half_angle
    use_taylor = np.abs(angle) <= 1.0e-6
    angle_safe = np.where(use_taylor, 1.0, angle)
    sin_half_over_angle = np.where(
        use_taylor,
        0.5 - angle * angle / 48.0,
        np.sin(half_angle) / angle_safe,
    )
    return quat_xyz / sin_half_over_angle[..., None]


def _grad_t(x: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError(f"Invalid dt: {dt}")
    if x.shape[1] < 2:
        return np.zeros_like(x)
    grad = np.empty_like(x)
    inv_dt = 1.0 / dt
    grad[:, 0] = (x[:, 1] - x[:, 0]) * inv_dt
    grad[:, -1] = (x[:, -1] - x[:, -2]) * inv_dt
    if x.shape[1] > 2:
        grad[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * (0.5 * inv_dt)
    return grad


class HoloMotionFKRootOnly(torch.nn.Module):
    """Root-only online FK.

    This lightweight variant is intended for policy-time VR reference building when
    only the root body pose/velocity are consumed by observation terms.
    """

    def __init__(
        self,
        dof_names: Sequence[str],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        timing_logger_enabled: bool = False,
        timing_log_interval_sec: float = 5.0,
        timing_log_per_call: bool = False,
        timing_name: str = "HoloMotionFKRootOnly",
        timing_log_fn: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.body_names = ["root"]
        self.dof_names = list(dof_names)
        self.num_bodies = 1
        self.num_dof = len(self.dof_names)
        if self.num_dof <= 0:
            raise ValueError("dof_names must not be empty")
        self._device = torch.device(device)
        self._dtype = dtype
        if self._dtype == torch.float64:
            self._np_dtype = np.float64
        else:
            self._np_dtype = np.float32
        self._timing_logger_enabled = bool(timing_logger_enabled)
        self._timing_log_interval_sec = float(timing_log_interval_sec)
        self._timing_log_per_call = bool(timing_log_per_call)
        self._timing_name = str(timing_name)
        self._timing_logger = logging.getLogger(__name__)
        self._timing_log_fn = timing_log_fn
        self._timing_last_log_time = None
        self._timing_count = 0
        self._timing_sum_ms = {}
        self._timing_max_ms = {}
        self.last_timing_ms = {}
        self._gaussian_kernel_cache: Dict[tuple[float, str], np.ndarray] = {}

    def set_timing_logger(
        self,
        enabled: bool,
        interval_sec: float | None = None,
        per_call: bool | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._timing_logger_enabled = bool(enabled)
        if interval_sec is not None:
            self._timing_log_interval_sec = float(interval_sec)
        if per_call is not None:
            self._timing_log_per_call = bool(per_call)
        if log_fn is not None:
            self._timing_log_fn = log_fn

    def _timing_ms(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _to_numpy(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=self._np_dtype)
        if not isinstance(x, torch.Tensor):
            return np.asarray(x, dtype=self._np_dtype)
        if x.device.type != "cpu" or x.dtype != self._dtype:
            x = x.detach().to(device="cpu", dtype=self._dtype)
        else:
            x = x.detach()
        return x.numpy()

    def _to_output_tensor(self, x: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        if self._device.type != "cpu" or tensor.dtype != self._dtype:
            tensor = tensor.to(device=self._device, dtype=self._dtype)
        return tensor

    def _get_gaussian_kernel(self, sigma: float) -> np.ndarray | None:
        if sigma <= 0.0:
            return None
        key = (float(sigma), np.dtype(self._np_dtype).str)
        kernel = self._gaussian_kernel_cache.get(key, None)
        if kernel is not None:
            return kernel
        radius = int(4.0 * sigma + 0.5)
        kernel_x = np.arange(-radius, radius + 1, dtype=self._np_dtype)
        kernel = np.exp(-0.5 * np.square(kernel_x / sigma)).astype(
            self._np_dtype, copy=False
        )
        kernel /= kernel.sum(dtype=self._np_dtype)
        self._gaussian_kernel_cache[key] = kernel
        return kernel

    def _gaussian_filter_time(self, x: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
        if kernel is None or x.shape[1] < 2:
            return x
        radius = kernel.shape[0] // 2
        padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=kernel.shape[0], axis=1
        )
        return np.tensordot(windows, kernel, axes=([-1], [0])).astype(
            x.dtype, copy=False
        )

    def _log_timing_message(self, message: str) -> None:
        if self._timing_log_fn is not None:
            self._timing_log_fn(message)
        else:
            self._timing_logger.info(message)

    def _record_timing(self, sample: Dict[str, float]) -> None:
        self.last_timing_ms = dict(sample)
        if not self._timing_logger_enabled:
            return

        self._timing_count += 1
        for key, value in sample.items():
            v = float(value)
            self._timing_sum_ms[key] = self._timing_sum_ms.get(key, 0.0) + v
            self._timing_max_ms[key] = max(self._timing_max_ms.get(key, v), v)
        if self._timing_log_per_call:
            self._log_timing_message(
                (
                    f"[{self._timing_name}][Timing] "
                    f"total={sample['total_ms']:.3f}ms "
                    f"input={sample['input_ms']:.3f}ms "
                    f"quat={sample['quat_ms']:.3f}ms "
                    f"linvel={sample['linvel_ms']:.3f}ms "
                    f"angvel={sample['angvel_ms']:.3f}ms "
                    f"smooth={sample['smooth_ms']:.3f}ms "
                    f"output={sample['output_ms']:.3f}ms"
                )
            )

        now = time.time()
        if self._timing_last_log_time is None:
            self._timing_last_log_time = now
            return
        if now - self._timing_last_log_time < self._timing_log_interval_sec:
            return
        if self._timing_count == 0:
            self._timing_last_log_time = now
            return

        keys = [
            "total_ms",
            "input_ms",
            "quat_ms",
            "linvel_ms",
            "angvel_ms",
            "smooth_ms",
            "output_ms",
        ]
        self._log_timing_message(
            f"[{self._timing_name}][Timing-Agg] "
            + " ".join(
                f"{key}=mean:{self._timing_sum_ms.get(key, 0.0) / self._timing_count:.3f}ms/"
                f"max:{self._timing_max_ms.get(key, 0.0):.3f}ms"
                for key in keys
            )
            + f" n={self._timing_count}"
        )
        self._timing_count = 0
        self._timing_sum_ms.clear()
        self._timing_max_ms.clear()
        self._timing_last_log_time = now

    @torch.inference_mode()
    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        dof_pos: torch.Tensor,
        fps: float,
        quat_format: str = "xyzw",
        sub_batch_size: int = 64,
        vel_smoothing_sigma: float = 2.0,
        compute_velocity: bool = True,
    ) -> Dict[str, torch.Tensor]:
        t_total = time.perf_counter()
        del sub_batch_size
        del compute_velocity  # kept for call-site compatibility

        if fps <= 0.0:
            raise ValueError(f"Invalid fps: {fps}")
        if root_pos.ndim != 3 or root_quat.ndim != 3 or dof_pos.ndim != 3:
            raise ValueError("Inputs must be (B, T, ...)")
        if (
            root_pos.shape[:2] != root_quat.shape[:2]
            or root_pos.shape[:2] != dof_pos.shape[:2]
        ):
            raise ValueError("Mismatched batch/time shapes among inputs")
        if root_pos.shape[-1] != 3 or root_quat.shape[-1] != 4:
            raise ValueError(
                "root_pos must be (B,T,3) and root_quat must be (B,T,4)"
            )
        if dof_pos.shape[-1] != self.num_dof:
            raise ValueError(
                f"dof_pos last dim {dof_pos.shape[-1]} does not match {self.num_dof}"
            )

        t_input = time.perf_counter()
        root_pos_np = self._to_numpy(root_pos)
        root_quat_np = self._to_numpy(root_quat)
        dof_pos_np = self._to_numpy(dof_pos)
        input_ms = self._timing_ms(t_input)

        t_quat = time.perf_counter()
        if quat_format == "xyzw":
            root_quat_xyzw_np = root_quat_np
            root_quat_wxyz_np = _xyzw_to_wxyz(root_quat_np)
        elif quat_format == "wxyz":
            root_quat_wxyz_np = root_quat_np
            root_quat_xyzw_np = _wxyz_to_xyzw(root_quat_np)
        else:
            raise ValueError(f"Unsupported quat_format: {quat_format}")
        quat_ms = self._timing_ms(t_quat)

        dt = 1.0 / fps
        kernel = self._get_gaussian_kernel(float(vel_smoothing_sigma))
        t_linvel = time.perf_counter()
        root_vel_np = _grad_t(root_pos_np, dt)
        linvel_ms = self._timing_ms(t_linvel)

        t_angvel = time.perf_counter()
        root_angvel_np = np.zeros_like(root_pos_np)
        if root_quat_wxyz_np.shape[1] >= 2:
            q1 = root_quat_wxyz_np[:, 1:]
            q0_inv = _quat_conjugate_wxyz(root_quat_wxyz_np[:, :-1])
            q_rel = _quat_mul_wxyz(q1, q0_inv)
            root_angvel_np[:, :-1] = _axis_angle_from_wxyz(q_rel) / dt
        angvel_ms = self._timing_ms(t_angvel)

        t_smooth = time.perf_counter()
        if kernel is not None and root_pos_np.shape[1] >= 2:
            vel_and_ang_np = np.concatenate([root_vel_np, root_angvel_np], axis=-1)
            vel_and_ang_np = self._gaussian_filter_time(vel_and_ang_np, kernel)
            root_vel_np = vel_and_ang_np[..., :3]
            root_angvel_np = vel_and_ang_np[..., 3:6]
        smooth_ms = self._timing_ms(t_smooth)

        t_output = time.perf_counter()
        out = {
            "global_translation": self._to_output_tensor(root_pos_np[:, :, None, :]),
            "global_rotation_quat": self._to_output_tensor(
                root_quat_xyzw_np[:, :, None, :]
            ),
            "global_velocity": self._to_output_tensor(root_vel_np[:, :, None, :]),
            "global_angular_velocity": self._to_output_tensor(
                root_angvel_np[:, :, None, :]
            ),
            "dof_pos": self._to_output_tensor(dof_pos_np),
            "dof_vel": self._to_output_tensor(np.zeros_like(dof_pos_np)),
        }
        output_ms = self._timing_ms(t_output)
        self._record_timing(
            {
                "total_ms": self._timing_ms(t_total),
                "input_ms": input_ms,
                "quat_ms": quat_ms,
                "linvel_ms": linvel_ms,
                "angvel_ms": angvel_ms,
                "smooth_ms": smooth_ms,
                "output_ms": output_ms,
            }
        )
        return out
