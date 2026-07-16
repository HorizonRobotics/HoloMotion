"""Shared reference kinematics used by training and deployed policies.

HoloRetarget emits only robot qpos. This module derives the temporal quantities
required by the motion-tracking policy without depending on the retargeter.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from holoretarget.schema import DOF_POS_DIM, QPOS_DIM


REFERENCE_DOF_DIM = DOF_POS_DIM
REFERENCE_QPOS_DIM = QPOS_DIM
ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_SLICE = slice(3, 7)
DOF_POS_SLICE = slice(7, REFERENCE_QPOS_DIM)


@dataclass(frozen=True)
class ReferenceKinematics:
    dof_vel: Any
    root_linvel_world: Any
    root_angvel_world: Any
    root_linvel_local: Any
    root_angvel_local: Any
    projected_gravity: Any

    def sliced(self, index: Any) -> "ReferenceKinematics":
        return ReferenceKinematics(
            dof_vel=self.dof_vel[index],
            root_linvel_world=self.root_linvel_world[index],
            root_angvel_world=self.root_angvel_world[index],
            root_linvel_local=self.root_linvel_local[index],
            root_angvel_local=self.root_angvel_local[index],
            projected_gravity=self.projected_gravity[index],
        )


def pack_reference_qpos(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    leading = root_pos.shape[:-1]
    if root_pos.shape != (*leading, 3):
        raise ValueError(
            f"root_pos must end in 3 values, got {root_pos.shape}"
        )
    if root_quat_wxyz.shape != (*leading, 4):
        raise ValueError(
            f"root_quat_wxyz must match root_pos and end in 4, got {root_quat_wxyz.shape}"
        )
    if dof_pos.shape != (*leading, REFERENCE_DOF_DIM):
        raise ValueError(
            f"dof_pos must match root_pos and end in {REFERENCE_DOF_DIM}, got {dof_pos.shape}"
        )
    result = np.empty((*leading, REFERENCE_QPOS_DIM), dtype=np.float32)
    result[..., ROOT_POS_SLICE] = root_pos
    result[..., ROOT_QUAT_SLICE] = root_quat_wxyz
    result[..., DOF_POS_SLICE] = dof_pos
    return result


@wp.func
def _quat_rotate_inverse_wxyz(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(-q[1], -q[2], -q[3])
    t = 2.0 * wp.cross(qv, v)
    return v + q[0] * t + wp.cross(qv, t)


@wp.func
def _angular_velocity_wxyz(q0: wp.vec4, q1: wp.vec4, dt: float) -> wp.vec3:
    rw = q1[0] * q0[0] + q1[1] * q0[1] + q1[2] * q0[2] + q1[3] * q0[3]
    rx = -q1[0] * q0[1] + q1[1] * q0[0] - q1[2] * q0[3] + q1[3] * q0[2]
    ry = -q1[0] * q0[2] + q1[1] * q0[3] + q1[2] * q0[0] - q1[3] * q0[1]
    rz = -q1[0] * q0[3] - q1[1] * q0[2] + q1[2] * q0[1] + q1[3] * q0[0]
    if rw < 0.0:
        rw = -rw
        rx = -rx
        ry = -ry
        rz = -rz
    norm = wp.sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
    inv_norm = 1.0 / wp.max(norm, 1.0e-9)
    rw = rw * inv_norm
    rx = rx * inv_norm
    ry = ry * inv_norm
    rz = rz * inv_norm
    mag = wp.sqrt(rx * rx + ry * ry + rz * rz)
    scale = 2.0
    if mag > 1.0e-7:
        scale = 2.0 * wp.atan2(mag, rw) / mag
    return wp.vec3(rx * scale / dt, ry * scale / dt, rz * scale / dt)


@wp.kernel
def _reference_root_kinematics_kernel(
    qpos: wp.array3d(dtype=float),
    sample_time: wp.array2d(dtype=float),
    sequence_length: int,
    root_linvel_world: wp.array3d(dtype=float),
    root_angvel_world: wp.array3d(dtype=float),
    root_linvel_local: wp.array3d(dtype=float),
    root_angvel_local: wp.array3d(dtype=float),
    projected_gravity: wp.array3d(dtype=float),
):
    index = wp.tid()
    batch = index // sequence_length
    frame = index - batch * sequence_length

    lo = frame - 1
    hi = frame + 1
    if frame == 0:
        lo = 0
    if frame == sequence_length - 1:
        hi = sequence_length - 1

    linear_dt = sample_time[batch, hi] - sample_time[batch, lo]
    linear_dt = wp.max(linear_dt, 1.0e-6)
    linear_world = wp.vec3(
        (qpos[batch, hi, 0] - qpos[batch, lo, 0]) / linear_dt,
        (qpos[batch, hi, 1] - qpos[batch, lo, 1]) / linear_dt,
        (qpos[batch, hi, 2] - qpos[batch, lo, 2]) / linear_dt,
    )

    quat = wp.vec4(
        qpos[batch, frame, 3],
        qpos[batch, frame, 4],
        qpos[batch, frame, 5],
        qpos[batch, frame, 6],
    )
    lo_quat = wp.vec4(
        qpos[batch, lo, 3],
        qpos[batch, lo, 4],
        qpos[batch, lo, 5],
        qpos[batch, lo, 6],
    )
    hi_quat = wp.vec4(
        qpos[batch, hi, 3],
        qpos[batch, hi, 4],
        qpos[batch, hi, 5],
        qpos[batch, hi, 6],
    )
    angular_world = _angular_velocity_wxyz(lo_quat, hi_quat, linear_dt)

    linear_local = _quat_rotate_inverse_wxyz(quat, linear_world)
    angular_local = _quat_rotate_inverse_wxyz(quat, angular_world)
    gravity = _quat_rotate_inverse_wxyz(quat, wp.vec3(0.0, 0.0, -1.0))
    for axis in range(3):
        root_linvel_world[batch, frame, axis] = linear_world[axis]
        root_angvel_world[batch, frame, axis] = angular_world[axis]
        root_linvel_local[batch, frame, axis] = linear_local[axis]
        root_angvel_local[batch, frame, axis] = angular_local[axis]
        projected_gravity[batch, frame, axis] = gravity[axis]


@wp.kernel
def _reference_dof_velocity_kernel(
    qpos: wp.array3d(dtype=float),
    sample_time: wp.array2d(dtype=float),
    sequence_length: int,
    dof_vel: wp.array3d(dtype=float),
):
    index = wp.tid()
    dof = index % REFERENCE_DOF_DIM
    frame_batch = index // REFERENCE_DOF_DIM
    batch = frame_batch // sequence_length
    frame = frame_batch - batch * sequence_length

    lo = frame - 1
    hi = frame + 1
    if frame == 0:
        lo = 0
    if frame == sequence_length - 1:
        hi = sequence_length - 1
    dt = wp.max(sample_time[batch, hi] - sample_time[batch, lo], 1.0e-6)
    dof_vel[batch, frame, dof] = (
        qpos[batch, hi, 7 + dof] - qpos[batch, lo, 7 + dof]
    ) / dt


def _validate_qpos_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 3 or shape[-1] != REFERENCE_QPOS_DIM:
        raise ValueError(
            f"reference_qpos must be [B,T,{REFERENCE_QPOS_DIM}], got {shape}"
        )
    if shape[1] <= 0:
        raise ValueError(
            "reference_qpos sequence must contain at least one frame"
        )


def _uniform_sample_times_numpy(
    batch: int, frames: int, fps: float
) -> np.ndarray:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    one = np.arange(frames, dtype=np.float32) / np.float32(fps)
    return np.broadcast_to(one[None], (batch, frames)).copy()


def derive_reference_kinematics_numpy(
    reference_qpos: np.ndarray,
    *,
    sample_time: np.ndarray | None = None,
    fps: float = 50.0,
    device: str = "cpu",
) -> ReferenceKinematics:
    wp.init()
    qpos = np.asarray(reference_qpos, dtype=np.float32)
    squeeze = qpos.ndim == 2
    if squeeze:
        qpos = qpos[None]
    qpos = np.ascontiguousarray(qpos)
    _validate_qpos_shape(qpos.shape)
    batch, frames, _ = qpos.shape
    if sample_time is None:
        times = _uniform_sample_times_numpy(batch, frames, fps)
    else:
        times = np.asarray(sample_time, dtype=np.float32)
        if times.ndim == 1:
            times = times[None]
        if times.shape != (batch, frames):
            raise ValueError(
                f"sample_time must have shape {(batch, frames)}, got {times.shape}"
            )
        times = np.ascontiguousarray(times)

    qpos_wp = wp.from_numpy(qpos, dtype=wp.float32, device=device)
    times_wp = wp.from_numpy(times, dtype=wp.float32, device=device)
    output_shapes = [(batch, frames, REFERENCE_DOF_DIM)] + [
        (batch, frames, 3)
    ] * 5
    outputs = [
        wp.empty(shape, dtype=wp.float32, device=device)
        for shape in output_shapes
    ]
    wp.launch(
        _reference_root_kinematics_kernel,
        dim=batch * frames,
        inputs=[qpos_wp, times_wp, frames, *outputs[1:]],
        device=device,
    )
    wp.launch(
        _reference_dof_velocity_kernel,
        dim=batch * frames * REFERENCE_DOF_DIM,
        inputs=[qpos_wp, times_wp, frames, outputs[0]],
        device=device,
    )
    wp.synchronize_device(device)
    arrays = [output.numpy() for output in outputs]
    if squeeze:
        arrays = [array[0] for array in arrays]
    return ReferenceKinematics(*arrays)


def derive_reference_kinematics_torch(
    reference_qpos,
    *,
    sample_time=None,
    fps: float = 50.0,
) -> ReferenceKinematics:
    import torch

    wp.init()
    qpos = reference_qpos
    squeeze = qpos.ndim == 2
    if squeeze:
        qpos = qpos.unsqueeze(0)
    if qpos.dtype != torch.float32 or not qpos.is_contiguous():
        qpos = qpos.to(dtype=torch.float32).contiguous()
    _validate_qpos_shape(tuple(qpos.shape))
    batch, frames, _ = qpos.shape
    if sample_time is None:
        one = torch.arange(
            frames, dtype=torch.float32, device=qpos.device
        ) / float(fps)
        times = one.unsqueeze(0).expand(batch, -1).contiguous()
    else:
        times = sample_time
        if times.ndim == 1:
            times = times.unsqueeze(0)
        times = times.to(device=qpos.device, dtype=torch.float32).contiguous()
        if tuple(times.shape) != (batch, frames):
            raise ValueError(
                f"sample_time must have shape {(batch, frames)}, got {tuple(times.shape)}"
            )

    outputs = [
        torch.empty((batch, frames, REFERENCE_DOF_DIM), device=qpos.device),
        *[
            torch.empty((batch, frames, 3), device=qpos.device)
            for _ in range(5)
        ],
    ]
    device = wp.device_from_torch(qpos.device)
    qpos_wp = wp.from_torch(qpos, dtype=wp.float32)
    times_wp = wp.from_torch(times, dtype=wp.float32)
    output_wp = [wp.from_torch(output, dtype=wp.float32) for output in outputs]
    wp.launch(
        _reference_root_kinematics_kernel,
        dim=batch * frames,
        inputs=[qpos_wp, times_wp, frames, *output_wp[1:]],
        device=device,
    )
    wp.launch(
        _reference_dof_velocity_kernel,
        dim=batch * frames * REFERENCE_DOF_DIM,
        inputs=[qpos_wp, times_wp, frames, output_wp[0]],
        device=device,
    )
    if squeeze:
        outputs = [output[0] for output in outputs]
    return ReferenceKinematics(*outputs)


def derive_reference_kinematics_warp(
    qpos,
    sample_time,
    *,
    outputs: ReferenceKinematics | None = None,
    device: str | None = None,
) -> ReferenceKinematics:
    """Derive reference kinematics directly on existing Warp arrays."""

    wp.init()
    _validate_qpos_shape(tuple(qpos.shape))
    batch, frames, _ = qpos.shape
    if tuple(sample_time.shape) != (batch, frames):
        raise ValueError(
            f"sample_time must have shape {(batch, frames)}, got {sample_time.shape}"
        )
    launch_device = qpos.device if device is None else device
    if outputs is None:
        outputs = ReferenceKinematics(
            dof_vel=wp.empty(
                (batch, frames, REFERENCE_DOF_DIM),
                dtype=wp.float32,
                device=launch_device,
            ),
            root_linvel_world=wp.empty(
                (batch, frames, 3), dtype=wp.float32, device=launch_device
            ),
            root_angvel_world=wp.empty(
                (batch, frames, 3), dtype=wp.float32, device=launch_device
            ),
            root_linvel_local=wp.empty(
                (batch, frames, 3), dtype=wp.float32, device=launch_device
            ),
            root_angvel_local=wp.empty(
                (batch, frames, 3), dtype=wp.float32, device=launch_device
            ),
            projected_gravity=wp.empty(
                (batch, frames, 3), dtype=wp.float32, device=launch_device
            ),
        )
    wp.launch(
        _reference_root_kinematics_kernel,
        dim=batch * frames,
        inputs=[
            qpos,
            sample_time,
            frames,
            outputs.root_linvel_world,
            outputs.root_angvel_world,
            outputs.root_linvel_local,
            outputs.root_angvel_local,
            outputs.projected_gravity,
        ],
        device=launch_device,
    )
    wp.launch(
        _reference_dof_velocity_kernel,
        dim=batch * frames * REFERENCE_DOF_DIM,
        inputs=[qpos, sample_time, frames, outputs.dof_vel],
        device=launch_device,
    )
    return outputs


__all__ = [
    "DOF_POS_SLICE",
    "REFERENCE_DOF_DIM",
    "REFERENCE_QPOS_DIM",
    "ROOT_POS_SLICE",
    "ROOT_QUAT_SLICE",
    "ReferenceKinematics",
    "derive_reference_kinematics_numpy",
    "derive_reference_kinematics_torch",
    "derive_reference_kinematics_warp",
    "pack_reference_qpos",
]
