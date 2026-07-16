"""GPU target writer for the online HoloRetarget path."""

from __future__ import annotations

import ctypes
import time
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .schema import QPOS_DIM

if TYPE_CHECKING:
    from .online import HoloRetargeter


@wp.func
def _qmul_wxyz(
    aw: float,
    ax: float,
    ay: float,
    az: float,
    bw: float,
    bx: float,
    by: float,
    bz: float,
):
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@wp.func
def _qnorm_wxyz(qw: float, qx: float, qy: float, qz: float):
    norm = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    inv_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-8))
    return qw * inv_norm, qx * inv_norm, qy * inv_norm, qz * inv_norm


@wp.kernel
def _pico_to_newton_targets_kernel(
    body: wp.array(dtype=wp.float32),
    target_human_indices: wp.array(dtype=wp.int32),
    target_scales: wp.array(dtype=wp.float32),
    target_pos_offsets: wp.array(dtype=wp.float32),
    target_rot_offsets: wp.array(dtype=wp.float32),
    position_target_def_indices: wp.array(dtype=wp.int32),
    rotation_target_def_indices: wp.array(dtype=wp.int32),
    ground_target_mask: wp.array(dtype=wp.int32),
    target_count: int,
    position_count: int,
    rotation_count: int,
    root_human_index: int,
    root_scale: float,
    root_seed_target_index: int,
    ground_buffer: wp.array(dtype=wp.float32),
    ground_count: wp.array(dtype=wp.int32),
    ground_frames: int,
    ground_height: float,
    ground_lift_only: int,
    body_pos: wp.array(dtype=wp.float32),
    body_quat: wp.array(dtype=wp.float32),
    target_pos_all: wp.array(dtype=wp.float32),
    target_quat_xyzw_all: wp.array(dtype=wp.float32),
    objective_pos: wp.array(dtype=wp.vec3),
    objective_rot: wp.array(dtype=wp.vec4),
    joint_q: wp.array2d(dtype=wp.float32),
) -> None:
    if wp.tid() != 0:
        return

    # Unity/Pico global pose -> HoloRetarget's right-handed target frame.
    rw = wp.float32(0.7071067811865476)
    rx = wp.float32(0.7071067811865476)
    ry = wp.float32(0.0)
    rz = wp.float32(0.0)
    yw = wp.float32(0.0)
    yx = wp.float32(0.0)
    yy = wp.float32(1.0)
    yz = wp.float32(0.0)

    for j in range(24):
        base = j * 7
        bw = body[base + 6]
        bx = body[base + 3]
        by = body[base + 4]
        bz = body[base + 5]
        n = wp.sqrt(bw * bw + bx * bx + by * by + bz * bz)
        inv_n = wp.float32(1.0) / wp.max(n, wp.float32(1.0e-8))
        bw *= inv_n
        bx *= inv_n
        by *= inv_n
        bz *= inv_n

        tw = rw * bw - rx * bx - ry * by - rz * bz
        tx = rw * bx + rx * bw + ry * bz - rz * by
        ty = rw * by - rx * bz + ry * bw + rz * bx
        tz = rw * bz + rx * by - ry * bx + rz * bw

        qw = tw * yw - tx * yx - ty * yy - tz * yz
        qx = tw * yx + tx * yw + ty * yz - tz * yy
        qy = tw * yy - tx * yz + ty * yw + tz * yx
        qz = tw * yz + tx * yy - ty * yx + tz * yw
        qn = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qinv = wp.float32(1.0) / wp.max(qn, wp.float32(1.0e-8))
        qb = j * 4
        body_quat[qb + 0] = qw * qinv
        body_quat[qb + 1] = qx * qinv
        body_quat[qb + 2] = qy * qinv
        body_quat[qb + 3] = qz * qinv

        pb = j * 3
        body_pos[pb + 0] = body[base + 0]
        body_pos[pb + 1] = -body[base + 2]
        body_pos[pb + 2] = body[base + 1]

    rb = root_human_index * 3
    root_x = body_pos[rb + 0]
    root_y = body_pos[rb + 1]
    root_z = body_pos[rb + 2]
    scaled_root_x = root_scale * root_x
    scaled_root_y = root_scale * root_y
    scaled_root_z = root_scale * root_z

    min_z = wp.float32(3.4028234663852886e38)
    for t in range(target_count):
        h = target_human_indices[t]
        hb = h * 3
        qb = h * 4
        scale = target_scales[t]
        sx = (body_pos[hb + 0] - root_x) * scale + scaled_root_x
        sy = (body_pos[hb + 1] - root_y) * scale + scaled_root_y
        sz = (body_pos[hb + 2] - root_z) * scale + scaled_root_z

        aw = body_quat[qb + 0]
        ax = body_quat[qb + 1]
        ay = body_quat[qb + 2]
        az = body_quat[qb + 3]
        rob = t * 4
        bw = target_rot_offsets[rob + 0]
        bx = target_rot_offsets[rob + 1]
        by = target_rot_offsets[rob + 2]
        bz = target_rot_offsets[rob + 3]
        uw = aw * bw - ax * bx - ay * by - az * bz
        ux = aw * bx + ax * bw + ay * bz - az * by
        uy = aw * by - ax * bz + ay * bw + az * bx
        uz = aw * bz + ax * by - ay * bx + az * bw
        un = wp.sqrt(uw * uw + ux * ux + uy * uy + uz * uz)
        uinv = wp.float32(1.0) / wp.max(un, wp.float32(1.0e-8))
        uw *= uinv
        ux *= uinv
        uy *= uinv
        uz *= uinv

        pob = t * 3
        vx = target_pos_offsets[pob + 0]
        vy = target_pos_offsets[pob + 1]
        vz = target_pos_offsets[pob + 2]
        tx = wp.float32(2.0) * (uy * vz - uz * vy)
        ty = wp.float32(2.0) * (uz * vx - ux * vz)
        tz = wp.float32(2.0) * (ux * vy - uy * vx)
        px = sx + vx + uw * tx + (uy * tz - uz * ty)
        py = sy + vy + uw * ty + (uz * tx - ux * tz)
        pz = sz + vz + uw * tz + (ux * ty - uy * tx)
        target_pos_all[pob + 0] = px
        target_pos_all[pob + 1] = py
        target_pos_all[pob + 2] = pz
        target_quat_xyzw_all[rob + 0] = ux
        target_quat_xyzw_all[rob + 1] = uy
        target_quat_xyzw_all[rob + 2] = uz
        target_quat_xyzw_all[rob + 3] = uw
        if ground_target_mask[t] != 0:
            min_z = wp.min(min_z, pz)

    ground_offset = wp.float32(0.0)
    if ground_frames > 0:
        c = ground_count[0]
        ground_buffer[c % ground_frames] = min_z
        c2 = c + 1
        ground_count[0] = c2
        active = c2
        if active > ground_frames:
            active = ground_frames
        window_min = wp.float32(3.4028234663852886e38)
        for i in range(128):
            if i < active and i < ground_frames:
                window_min = wp.min(window_min, ground_buffer[i])
        ground_offset = ground_height - window_min
        if ground_lift_only != 0 and ground_offset < wp.float32(0.0):
            ground_offset = wp.float32(0.0)

    for i in range(position_count):
        t = position_target_def_indices[i]
        b = t * 3
        objective_pos[i] = wp.vec3(
            target_pos_all[b + 0],
            target_pos_all[b + 1],
            target_pos_all[b + 2] + ground_offset,
        )
    for i in range(rotation_count):
        t = rotation_target_def_indices[i]
        b = t * 4
        objective_rot[i] = wp.vec4(
            target_quat_xyzw_all[b + 0],
            target_quat_xyzw_all[b + 1],
            target_quat_xyzw_all[b + 2],
            target_quat_xyzw_all[b + 3],
        )

    if root_seed_target_index >= 0:
        tb = root_seed_target_index * 3
        qb = root_seed_target_index * 4
        joint_q[0, 0] = target_pos_all[tb + 0]
        joint_q[0, 1] = target_pos_all[tb + 1]
        joint_q[0, 2] = target_pos_all[tb + 2] + ground_offset
        joint_q[0, 3] = target_quat_xyzw_all[qb + 0]
        joint_q[0, 4] = target_quat_xyzw_all[qb + 1]
        joint_q[0, 5] = target_quat_xyzw_all[qb + 2]
        joint_q[0, 6] = target_quat_xyzw_all[qb + 3]


@wp.kernel
def _pack_reference_qpos_kernel(
    joint_q: wp.array2d(dtype=wp.float32),
    limit_lower: wp.array(dtype=wp.float32),
    limit_upper: wp.array(dtype=wp.float32),
    limit_mask: wp.array(dtype=wp.int32),
    limit_mid: wp.array(dtype=wp.float32),
    previous_dof: wp.array(dtype=wp.float32),
    previous_root: wp.array(dtype=wp.float32),
    has_previous: wp.array(dtype=wp.int32),
    max_joint_step: float,
    reference_qpos: wp.array(dtype=wp.float32),
) -> None:
    if wp.tid() != 0:
        return

    had_previous = has_previous[0] != 0
    two_pi = wp.float32(6.283185307179586)
    for i in range(29):
        q = joint_q[0, 7 + i]
        projected = q
        if limit_mask[i] != 0:
            ref = previous_dof[i] if had_previous else limit_mid[i]
            nearest_k = wp.round((ref - q) / two_pi)
            min_k = wp.ceil((limit_lower[i] - q) / two_pi)
            max_k = wp.floor((limit_upper[i] - q) / two_pi)
            chosen_k = nearest_k
            if min_k <= max_k:
                chosen_k = wp.min(wp.max(nearest_k, min_k), max_k)
            projected = wp.min(
                wp.max(q + chosen_k * two_pi, limit_lower[i]),
                limit_upper[i],
            )
        if had_previous and max_joint_step > wp.float32(0.0):
            delta = wp.min(
                wp.max(
                    projected - previous_dof[i],
                    -max_joint_step,
                ),
                max_joint_step,
            )
            projected = previous_dof[i] + delta
            if limit_mask[i] != 0:
                projected = wp.min(
                    wp.max(projected, limit_lower[i]),
                    limit_upper[i],
                )
        previous_dof[i] = projected
        joint_q[0, 7 + i] = projected
        reference_qpos[7 + i] = projected

    reference_qpos[0] = joint_q[0, 0]
    reference_qpos[1] = joint_q[0, 1]
    reference_qpos[2] = joint_q[0, 2]
    qx = joint_q[0, 3]
    qy = joint_q[0, 4]
    qz = joint_q[0, 5]
    qw = joint_q[0, 6]
    norm = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    inv_norm = wp.float32(1.0) / wp.max(norm, wp.float32(1.0e-8))
    qw *= inv_norm
    qx *= inv_norm
    qy *= inv_norm
    qz *= inv_norm
    if had_previous:
        dot = (
            qw * previous_root[0]
            + qx * previous_root[1]
            + qy * previous_root[2]
            + qz * previous_root[3]
        )
        if dot < wp.float32(0.0):
            qw = -qw
            qx = -qx
            qy = -qy
            qz = -qz
    previous_root[0] = qw
    previous_root[1] = qx
    previous_root[2] = qy
    previous_root[3] = qz
    reference_qpos[3] = qw
    reference_qpos[4] = qx
    reference_qpos[5] = qy
    reference_qpos[6] = qz
    has_previous[0] = 1


class HoloPicoGpuTargetRunner:
    """Directly write online Pico targets into Newton GPU buffers."""

    def __init__(self, owner: HoloRetargeter) -> None:
        self.owner = owner
        self.runner = owner._runner
        self.solver = self.runner.newton_solver
        self.wp = self.solver.wp
        self.device = self.solver.joint_q.device
        if self.solver.src_human != "smplx":
            raise RuntimeError("GPU Pico target path requires the 24-joint target schema")
        if not self.device.is_cuda:
            raise RuntimeError("GPU Pico target path requires a CUDA Warp device")
        if self.solver.ground_calibration_mode != "sliding_min":
            raise RuntimeError("GPU Pico target path currently supports sliding_min ground calibration")
        if self.solver.ground_calibration_frames > 128:
            raise RuntimeError("GPU Pico target path supports at most 128 ground calibration frames")
        if self.solver.root_seed_mode not in {"off", "pelvis"}:
            raise RuntimeError("GPU Pico target path currently supports off/pelvis root seeding")

        self.body_host_wp = wp.empty(shape=(24 * 7,), dtype=wp.float32, device="cpu", pinned=True)
        self.body_host = self.body_host_wp.numpy()
        self.body_dev = wp.empty(shape=(24 * 7,), dtype=wp.float32, device=self.device)
        self.target_human_dev = wp.array(
            self.solver._target_human_indices.astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.target_scales_dev = wp.array(
            self.solver._target_scales32.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.target_pos_offsets_dev = wp.array(
            self.solver._target_pos_offsets32.astype(np.float32).reshape(-1),
            dtype=wp.float32,
            device=self.device,
        )
        self.target_rot_offsets_dev = wp.array(
            self.solver._target_rot_offsets_wxyz32.astype(np.float32).reshape(-1),
            dtype=wp.float32,
            device=self.device,
        )
        self.position_def_dev = wp.array(
            np.asarray(self.solver._position_target_def_indices, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.rotation_def_dev = wp.array(
            np.asarray(self.solver._rotation_target_def_indices, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        if self.solver.ground_target_scope == "all":
            ground_mask = np.ones(len(self.solver.target_defs), dtype=np.int32)
        else:
            ground_mask = self.solver._target_foot_mask.astype(np.int32)
            if not np.any(ground_mask):
                ground_mask = np.ones(len(self.solver.target_defs), dtype=np.int32)
        self.ground_mask_dev = wp.array(ground_mask, dtype=wp.int32, device=self.device)

        self.body_pos_dev = wp.empty(shape=(24 * 3,), dtype=wp.float32, device=self.device)
        self.body_quat_dev = wp.empty(shape=(24 * 4,), dtype=wp.float32, device=self.device)
        self.target_pos_all_dev = wp.empty(
            shape=(len(self.solver.target_defs) * 3,),
            dtype=wp.float32,
            device=self.device,
        )
        self.target_quat_all_dev = wp.empty(
            shape=(len(self.solver.target_defs) * 4,),
            dtype=wp.float32,
            device=self.device,
        )
        self.objective_pos_dev = wp.empty(
            shape=(len(self.solver.position_objectives),),
            dtype=wp.vec3,
            device=self.device,
        )
        self.objective_rot_dev = wp.empty(
            shape=(len(self.solver.rotation_objectives),),
            dtype=wp.vec4,
            device=self.device,
        )
        ground_frames = max(1, int(self.solver.ground_calibration_frames))
        self.ground_buffer_dev = wp.zeros(shape=(ground_frames,), dtype=wp.float32, device=self.device)
        self.ground_count_dev = wp.zeros(shape=(1,), dtype=wp.int32, device=self.device)
        self._ground_zero_host_wp = wp.zeros(shape=(ground_frames,), dtype=wp.float32, device="cpu")
        self._ground_count_zero_host_wp = wp.zeros(shape=(1,), dtype=wp.int32, device="cpu")
        self.limit_lower_dev = wp.array(
            self.solver._dof_limit_lower.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.limit_upper_dev = wp.array(
            self.solver._dof_limit_upper.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.limit_mask_dev = wp.array(
            self.solver._dof_limit_mask.astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.limit_mid_dev = wp.array(
            self.solver._dof_limit_mid.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.previous_dof_dev = wp.zeros(
            shape=(29,), dtype=wp.float32, device=self.device
        )
        self.previous_root_dev = wp.zeros(
            shape=(4,), dtype=wp.float32, device=self.device
        )
        self.has_previous_dev = wp.zeros(
            shape=(1,), dtype=wp.int32, device=self.device
        )
        self._previous_dof_zero_host_wp = wp.zeros(
            shape=(29,), dtype=wp.float32, device="cpu"
        )
        self._previous_root_zero_host_wp = wp.zeros(
            shape=(4,), dtype=wp.float32, device="cpu"
        )
        self._has_previous_zero_host_wp = wp.zeros(
            shape=(1,), dtype=wp.int32, device="cpu"
        )
        self.reference_qpos_dev = wp.empty(
            shape=(QPOS_DIM,), dtype=wp.float32, device=self.device
        )
        self.reference_qpos_host_wp = wp.empty(
            shape=(QPOS_DIM,), dtype=wp.float32, device="cpu", pinned=True
        )
        self.reference_qpos_host = self.reference_qpos_host_wp.numpy()

        self.root_seed_target_index = int(
            self.solver._root_seed_target_index
            if self.solver._root_seed_target_index is not None and self.solver.root_seed_mode != "off"
            else -1
        )
        self._init_objective_copy()
        self._warmup()

    def _init_objective_copy(self) -> None:
        count = len(self.solver.position_objectives) + len(self.solver.rotation_objectives)
        void_p_array = ctypes.c_void_p * count
        size_array = ctypes.c_size_t * count
        dst_ptrs = [obj.target_positions.ptr for obj in self.solver.position_objectives]
        dst_ptrs.extend(obj.target_rotations.ptr for obj in self.solver.rotation_objectives)
        pos_stride = 3 * np.dtype(np.float32).itemsize
        rot_stride = 4 * np.dtype(np.float32).itemsize
        src_ptrs = [
            self.objective_pos_dev.ptr + i * pos_stride
            for i in range(len(self.solver.position_objectives))
        ]
        src_ptrs.extend(
            self.objective_rot_dev.ptr + i * rot_stride
            for i in range(len(self.solver.rotation_objectives))
        )
        sizes = [pos_stride] * len(self.solver.position_objectives)
        sizes.extend([rot_stride] * len(self.solver.rotation_objectives))
        self._copy_dsts = void_p_array(*dst_ptrs)
        self._copy_srcs = void_p_array(*src_ptrs)
        self._copy_sizes = size_array(*sizes)
        self._copy_count = ctypes.c_size_t(count)
        self._copy_context = self.solver.joint_q.device.context
        self._copy_core = self.wp._src.context.runtime.core

    def _warmup(self) -> None:
        body = np.zeros((24, 7), dtype=np.float32)
        body[:, 6] = 1.0
        self.retarget_qpos_from_body_poses(body)
        self.solver.reset_sequence()
        self.runner._last_root_quat_wxyz = None
        self.reset_sequence()

    def reset_sequence(self) -> None:
        self.wp.copy(self.ground_buffer_dev, self._ground_zero_host_wp)
        self.wp.copy(self.ground_count_dev, self._ground_count_zero_host_wp)
        self.wp.copy(self.previous_dof_dev, self._previous_dof_zero_host_wp)
        self.wp.copy(self.previous_root_dev, self._previous_root_zero_host_wp)
        self.wp.copy(self.has_previous_dev, self._has_previous_zero_host_wp)

    def _write_targets(self, body_poses: np.ndarray) -> None:
        np.copyto(self.body_host, np.asarray(body_poses, dtype=np.float32).reshape(-1))
        wp.copy(self.body_dev, self.body_host_wp)
        wp.launch(
            _pico_to_newton_targets_kernel,
            dim=1,
            inputs=[
                self.body_dev,
                self.target_human_dev,
                self.target_scales_dev,
                self.target_pos_offsets_dev,
                self.target_rot_offsets_dev,
                self.position_def_dev,
                self.rotation_def_dev,
                self.ground_mask_dev,
                len(self.solver.target_defs),
                len(self.solver.position_objectives),
                len(self.solver.rotation_objectives),
                int(self.solver._root_human_index),
                float(self.solver._root_scale32),
                self.root_seed_target_index,
                self.ground_buffer_dev,
                self.ground_count_dev,
                int(self.solver.ground_calibration_frames),
                float(self.solver.ground_height),
                int(bool(self.solver.ground_lift_only)),
                self.body_pos_dev,
                self.body_quat_dev,
                self.target_pos_all_dev,
                self.target_quat_all_dev,
                self.objective_pos_dev,
                self.objective_rot_dev,
                self.solver.joint_q,
            ],
            device=self.device,
        )
        stream = self.wp.get_stream(self.solver.joint_q.device)
        ok = self._copy_core.wp_memcpy_batch(
            self._copy_context,
            self._copy_dsts,
            self._copy_srcs,
            self._copy_sizes,
            self._copy_count,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if not ok:
            raise RuntimeError("HoloRetarget GPU target copy failed")

    def retarget_qpos_from_body_poses(
        self,
        body_poses: np.ndarray,
    ) -> np.ndarray:
        reference_qpos_dev = self.retarget_qpos_device_from_body_poses(body_poses)
        self.wp.copy(self.reference_qpos_host_wp, reference_qpos_dev)
        self.wp.synchronize_device(self.device)
        reference_qpos = self.reference_qpos_host.copy()
        self.solver.last_output = reference_qpos
        return reference_qpos

    def retarget_qpos_device_from_body_poses(
        self,
        body_poses: np.ndarray,
    ):
        """Return qpos in Warp CUDA memory without a device-to-host sync."""
        t0 = time.perf_counter()
        self._write_targets(body_poses)
        self.solver._sync_if_profile()
        t_targets = time.perf_counter()

        if self.solver.graph_capture is not None:
            self.wp.capture_launch(self.solver.graph_capture)
        else:
            self.solver._single_step()
        self.solver._sync_if_profile()
        t_solve = time.perf_counter()

        wp.launch(
            _pack_reference_qpos_kernel,
            dim=1,
            inputs=[
                self.solver.joint_q,
                self.limit_lower_dev,
                self.limit_upper_dev,
                self.limit_mask_dev,
                self.limit_mid_dev,
                self.previous_dof_dev,
                self.previous_root_dev,
                self.has_previous_dev,
                float(self.solver.max_joint_step),
                self.reference_qpos_dev,
            ],
            device=self.device,
        )
        self.solver._sync_if_profile()
        t_output = time.perf_counter()
        t_project = t_output
        t_pack = time.perf_counter()

        self.solver.last_targets = None
        self.solver.last_output = None
        self.solver.last_ground_offset_z = 0.0
        self.solver.last_timing = {
            "holoretarget.pico_targets": t_targets - t0,
            "holoretarget.newton_targets": 0.0,
            "holoretarget.newton_set_targets": 0.0,
            "holoretarget.newton_root_seed": 0.0,
            "holoretarget.solve": t_solve - t_targets,
            "holoretarget.output": t_output - t_solve,
            "holoretarget.newton_project": t_project - t_output,
            "holoretarget.qpos_pack": t_pack - t_project,
            "holoretarget.total": t_pack - t0,
        }
        self.runner.last_timing = dict(self.solver.last_timing)
        return self.reference_qpos_dev
