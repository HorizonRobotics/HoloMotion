"""Convert HoloSMPL frames to the Pico-like HoloRetarget input schema."""

from __future__ import annotations

import numpy as np
import warp as wp

from holoretarget._gpu_targets import _qmul_wxyz, _qnorm_wxyz


@wp.func
def _axis_angle_to_quat_wxyz(x: float, y: float, z: float):
    theta2 = x * x + y * y + z * z
    if theta2 < wp.float32(1.0e-8):
        return (
            wp.float32(1.0),
            wp.float32(0.5) * x,
            wp.float32(0.5) * y,
            wp.float32(0.5) * z,
        )
    theta = wp.sqrt(theta2)
    half_theta = wp.float32(0.5) * theta
    scale = wp.sin(half_theta) / theta
    return wp.cos(half_theta), x * scale, y * scale, z * scale


@wp.func
def _qrot_wxyz(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    vx: float,
    vy: float,
    vz: float,
):
    tx = wp.float32(2.0) * (qy * vz - qz * vy)
    ty = wp.float32(2.0) * (qz * vx - qx * vz)
    tz = wp.float32(2.0) * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


@wp.func
def _match_matrix_to_quat_sign(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
):
    m00 = wp.float32(1.0) - wp.float32(2.0) * (qy * qy + qz * qz)
    m11 = wp.float32(1.0) - wp.float32(2.0) * (qx * qx + qz * qz)
    m22 = wp.float32(1.0) - wp.float32(2.0) * (qx * qx + qy * qy)
    trace = m00 + m11 + m22
    flip = wp.float32(1.0)
    if trace > wp.float32(0.0):
        if qw < wp.float32(0.0):
            flip = wp.float32(-1.0)
    elif m00 > m11 and m00 > m22:
        if qx < wp.float32(0.0):
            flip = wp.float32(-1.0)
    elif m11 > m22:
        if qy < wp.float32(0.0):
            flip = wp.float32(-1.0)
    else:
        if qz < wp.float32(0.0):
            flip = wp.float32(-1.0)
    return qw * flip, qx * flip, qy * flip, qz * flip


@wp.kernel
def _smpl_to_pico_body_kernel(
    transl: wp.array(dtype=wp.float32),
    pose_aa: wp.array(dtype=wp.float32),
    smpl_parents: wp.array(dtype=wp.int32),
    smpl_offsets: wp.array(dtype=wp.float32),
    smpl_pos: wp.array(dtype=wp.float32),
    smpl_quat: wp.array(dtype=wp.float32),
    body_out: wp.array(dtype=wp.float32),
) -> None:
    if wp.tid() != 0:
        return

    for joint in range(24):
        aa = joint * 3
        lw, lx, ly, lz = _axis_angle_to_quat_wxyz(
            pose_aa[aa],
            pose_aa[aa + 1],
            pose_aa[aa + 2],
        )
        if joint == 0:
            smpl_pos[0] = transl[0]
            smpl_pos[1] = transl[1]
            smpl_pos[2] = transl[2]
            smpl_quat[0] = lw
            smpl_quat[1] = lx
            smpl_quat[2] = ly
            smpl_quat[3] = lz
        else:
            parent = smpl_parents[joint]
            parent_quat = parent * 4
            pw = smpl_quat[parent_quat]
            px = smpl_quat[parent_quat + 1]
            py = smpl_quat[parent_quat + 2]
            pz = smpl_quat[parent_quat + 3]
            gw, gx, gy, gz = _qmul_wxyz(
                pw, px, py, pz, lw, lx, ly, lz
            )
            gw, gx, gy, gz = _qnorm_wxyz(gw, gx, gy, gz)
            quat = joint * 4
            smpl_quat[quat] = gw
            smpl_quat[quat + 1] = gx
            smpl_quat[quat + 2] = gy
            smpl_quat[quat + 3] = gz

            offset = joint * 3
            ox, oy, oz = _qrot_wxyz(
                pw,
                px,
                py,
                pz,
                smpl_offsets[offset],
                smpl_offsets[offset + 1],
                smpl_offsets[offset + 2],
            )
            parent_pos = parent * 3
            pos = joint * 3
            smpl_pos[pos] = smpl_pos[parent_pos] + ox
            smpl_pos[pos + 1] = smpl_pos[parent_pos + 1] + oy
            smpl_pos[pos + 2] = smpl_pos[parent_pos + 2] + oz

    inv_uw = wp.float32(0.7071067811865476)
    inv_ux = wp.float32(-0.7071067811865476)
    y180_w = wp.float32(0.0)
    y180_y = wp.float32(1.0)
    zero = wp.float32(0.0)
    for joint in range(24):
        pos = joint * 3
        out = joint * 7
        body_out[out] = smpl_pos[pos]
        body_out[out + 1] = smpl_pos[pos + 2]
        body_out[out + 2] = -smpl_pos[pos + 1]

        quat = joint * 4
        tw, tx, ty, tz = _qmul_wxyz(
            inv_uw,
            inv_ux,
            zero,
            zero,
            smpl_quat[quat],
            smpl_quat[quat + 1],
            smpl_quat[quat + 2],
            smpl_quat[quat + 3],
        )
        rw, rx, ry, rz = _qmul_wxyz(
            tw, tx, ty, tz, y180_w, zero, y180_y, zero
        )
        rw, rx, ry, rz = _qnorm_wxyz(rw, rx, ry, rz)
        rw, rx, ry, rz = _match_matrix_to_quat_sign(rw, rx, ry, rz)
        body_out[out + 3] = rx
        body_out[out + 4] = ry
        body_out[out + 5] = rz
        body_out[out + 6] = rw


class SmplToBodyPosesConverter:
    """Convert one SMPL frame into Pico-like ``body_poses[24, 7]``."""

    def __init__(self, retargeter) -> None:
        if retargeter._smpl_adapter is None:
            from holoretarget._engine_impl import SmplSkeletonAdapter

            retargeter._smpl_adapter = SmplSkeletonAdapter()
        self.adapter = retargeter._smpl_adapter
        self.device = retargeter._runner.newton_solver.joint_q.device
        if not self.device.is_cuda:
            raise RuntimeError("GPU SMPL body conversion requires CUDA Warp")

        self.transl_host_wp = wp.empty(
            shape=(3,), dtype=wp.float32, device="cpu", pinned=True
        )
        self.pose_host_wp = wp.empty(
            shape=(72,), dtype=wp.float32, device="cpu", pinned=True
        )
        self.body_host_wp = wp.empty(
            shape=(24 * 7,), dtype=wp.float32, device="cpu", pinned=True
        )
        self.transl_host = self.transl_host_wp.numpy()
        self.pose_host = self.pose_host_wp.numpy()
        self.body_host = self.body_host_wp.numpy()
        self.transl_dev = wp.empty(
            shape=(3,), dtype=wp.float32, device=self.device
        )
        self.pose_dev = wp.empty(
            shape=(72,), dtype=wp.float32, device=self.device
        )
        self.smpl_pos_dev = wp.empty(
            shape=(24 * 3,), dtype=wp.float32, device=self.device
        )
        self.smpl_quat_dev = wp.empty(
            shape=(24 * 4,), dtype=wp.float32, device=self.device
        )
        self.body_dev = wp.empty(
            shape=(24 * 7,), dtype=wp.float32, device=self.device
        )
        self.parents_dev = wp.array(
            self.adapter.smpl_parents.astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.offsets_dev = wp.array(
            self.adapter.smpl_rest_parent_offsets.astype(np.float32).reshape(-1),
            dtype=wp.float32,
            device=self.device,
        )

    def __call__(
        self,
        *,
        transl: np.ndarray,
        global_orient_aa: np.ndarray,
        body_pose_aa: np.ndarray,
    ) -> np.ndarray:
        body_pose = np.asarray(body_pose_aa, dtype=np.float32).reshape(-1)
        if body_pose.shape[0] not in {63, 69}:
            raise ValueError(
                "SMPL body pose must contain 63 or 69 values, got "
                f"{body_pose.shape[0]}"
            )
        np.copyto(
            self.transl_host,
            np.asarray(transl, dtype=np.float32).reshape(3),
        )
        self.pose_host.fill(0.0)
        self.pose_host[:3] = np.asarray(
            global_orient_aa, dtype=np.float32
        ).reshape(3)
        self.pose_host[3 : 3 + body_pose.shape[0]] = body_pose
        wp.copy(self.transl_dev, self.transl_host_wp)
        wp.copy(self.pose_dev, self.pose_host_wp)
        wp.launch(
            _smpl_to_pico_body_kernel,
            dim=1,
            inputs=[
                self.transl_dev,
                self.pose_dev,
                self.parents_dev,
                self.offsets_dev,
                self.smpl_pos_dev,
                self.smpl_quat_dev,
                self.body_dev,
            ],
            device=self.device,
        )
        wp.copy(self.body_host_wp, self.body_dev)
        wp.synchronize()
        return self.body_host.reshape(24, 7).copy()


__all__ = ["SmplToBodyPosesConverter"]
