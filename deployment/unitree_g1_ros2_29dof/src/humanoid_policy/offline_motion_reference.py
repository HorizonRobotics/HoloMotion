"""Offline motion reference observation helpers for the 29DOF policy node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from humanoid_policy.motion_clip_library import LoadedMotionClip


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float32)
    if q_xyzw.shape[-1] != 4:
        raise ValueError(
            f"xyzw_to_wxyz expects (...,4) but got shape {q_xyzw.shape}"
        )
    w = q_xyzw[..., 3:4]
    xyz = q_xyzw[..., 0:3]
    return np.concatenate([w, xyz], axis=-1)


def standardize_quaternion_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    if q_wxyz.shape[-1] != 4:
        raise ValueError(
            f"standardize_quaternion_wxyz expects (...,4) but got shape {q_wxyz.shape}"
        )
    return np.where(q_wxyz[..., 0:1] < 0.0, -q_wxyz, q_wxyz)


def yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    qw = q_wxyz[..., 0]
    qx = q_wxyz[..., 1]
    qy = q_wxyz[..., 2]
    qz = q_wxyz[..., 3]
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ).astype(np.float32)


def gravity_orientation_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    qw = q_wxyz[..., 0]
    qx = q_wxyz[..., 1]
    qy = q_wxyz[..., 2]
    qz = q_wxyz[..., 3]
    gravity = np.zeros(q_wxyz.shape[:-1] + (3,), dtype=np.float32)
    gravity[..., 0] = 2.0 * (-qz * qx + qw * qy)
    gravity[..., 1] = -2.0 * (qz * qy + qw * qx)
    gravity[..., 2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def quat_rotate_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qvec = q_wxyz[..., 1:4]
    w = q_wxyz[..., 0:1]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def quat_rotate_inv_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    q_conj = np.empty_like(q_wxyz)
    q_conj[..., 0] = q_wxyz[..., 0]
    q_conj[..., 1:4] = -q_wxyz[..., 1:4]
    return quat_rotate_wxyz(q_conj, v)


@dataclass
class OfflineMotionReference:
    """Active offline motion clip reference calculations.

    The class owns only offline reference computations. VR latest_obs queues and
    FK-derived references stay in the policy node until the separate VR phase.
    """

    n_fut_frames: int
    num_actions: int
    ref_to_onnx: Sequence[int]
    root_body_idx: int = 0
    reference_dof_count: int | None = None

    def __post_init__(self) -> None:
        self.n_fut_frames = max(int(self.n_fut_frames), 0)
        self.num_actions = int(self.num_actions)
        self.ref_to_onnx = np.asarray(list(self.ref_to_onnx), dtype=np.int64)
        if self.reference_dof_count is None:
            self.reference_dof_count = self.num_actions
        self.reference_dof_count = int(self.reference_dof_count)
        self.clip: LoadedMotionClip | None = None

        self._future_frame_offsets = np.arange(
            1, self.n_fut_frames + 1, dtype=np.int64
        )
        self._future_frame_indices_buffer = np.zeros(
            self.n_fut_frames, dtype=np.int64
        )
        self._future_root_quat_wxyz_buffer = np.zeros(
            (self.n_fut_frames, 4), dtype=np.float32
        )
        self._future_yaw_delta_sin_cos_buffer = np.zeros(
            (self.n_fut_frames, 2), dtype=np.float32
        )
        self._pos_fut_buffer = np.zeros(
            (self.reference_dof_count, self.n_fut_frames),
            dtype=np.float32,
        )
        self._dof_pos_onnx_buffer = np.zeros(
            self.num_actions, dtype=np.float32
        )
        self._dof_vel_onnx_buffer = np.zeros(
            self.num_actions, dtype=np.float32
        )
        self._dof_pos_fut_onnx_buffer = np.zeros(
            self.n_fut_frames * self.num_actions,
            dtype=np.float32,
        )
        self._h_fut_buffer = np.zeros((1, self.n_fut_frames), dtype=np.float32)
        self._root_pos_fut_buffer = np.zeros(
            (self.n_fut_frames, 3), dtype=np.float32
        )
        self._root_pos_cur_buffer = np.zeros(3, dtype=np.float32)
        self._root_height_cur_buffer = np.zeros(1, dtype=np.float32)
        self._gravity_fut_buffer = np.zeros(
            (self.n_fut_frames, 3), dtype=np.float32
        )
        self._base_linvel_fut_buffer = np.zeros(
            (self.n_fut_frames, 3), dtype=np.float32
        )
        self._base_angvel_fut_buffer = np.zeros(
            (self.n_fut_frames, 3), dtype=np.float32
        )
        self._keybody_rel_pos_fut_buffer = np.zeros(
            (self.n_fut_frames, 0, 3), dtype=np.float32
        )
        self._root_quat_wxyz_buffer = np.zeros(4, dtype=np.float32)
        self._gravity_cur_buffer = np.zeros(3, dtype=np.float32)
        self._base_linvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._base_angvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._motion_states_buffer = np.zeros(
            self.num_actions * 2, dtype=np.float32
        )
        max_t = max(1, self.n_fut_frames)
        self._vel_fut_T6 = np.zeros((max_t, 6), dtype=np.float32)
        self._rot_t_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._rot_cross_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._cache_motion_frame_idx: int | None = None

    def set_clip(self, clip: LoadedMotionClip) -> None:
        self.clip = clip
        self._cache_motion_frame_idx = None

    @property
    def has_clip(self) -> bool:
        return self.clip is not None

    @property
    def n_motion_frames(self) -> int:
        if self.clip is None:
            return 0
        return int(self.clip.n_frames)

    @property
    def last_valid_frame_idx(self) -> int:
        return max(self.n_motion_frames - 1, 0)

    def current_frame_idx(self, motion_frame_idx: int) -> int:
        return min(int(motion_frame_idx), self.last_valid_frame_idx)

    def future_frame_indices(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return self._future_frame_indices_buffer
        np.minimum(
            self.current_frame_idx(motion_frame_idx)
            + self._future_frame_offsets,
            self.last_valid_frame_idx,
            out=self._future_frame_indices_buffer,
        )
        return self._future_frame_indices_buffer

    def ref_dof_pos_raw(self, motion_frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        return clip.dof_pos[self.current_frame_idx(motion_frame_idx)]

    def ref_dof_vel_raw(self, motion_frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        return clip.dof_vel[self.current_frame_idx(motion_frame_idx)]

    def ref_dof_pos_onnx_order(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._dof_pos_onnx_buffer

    def ref_dof_vel_onnx_order(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._dof_vel_onnx_buffer

    def ref_root_pos_raw(self, motion_frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        return np.asarray(
            clip.global_translation[
                self.current_frame_idx(motion_frame_idx), self.root_body_idx
            ],
            dtype=np.float32,
        )

    def obs_ref_motion_states(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._motion_states_buffer

    def obs_ref_dof_pos_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._dof_pos_fut_onnx_buffer

    def obs_ref_root_height_cur(self, motion_frame_idx: int) -> np.float32:
        self.prepare_frame(motion_frame_idx)
        return self._root_height_cur_buffer[0]

    def obs_ref_root_height_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._h_fut_buffer.reshape(-1)

    def obs_ref_root_pos_cur(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._root_pos_cur_buffer

    def obs_ref_root_pos_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._root_pos_fut_buffer.reshape(-1)

    def obs_ref_gravity_projection_cur(
        self, motion_frame_idx: int
    ) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._gravity_cur_buffer

    def obs_ref_gravity_projection_fut(
        self, motion_frame_idx: int
    ) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._gravity_fut_buffer.reshape(-1)

    def obs_ref_base_linvel_cur(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._base_linvel_cur_buffer

    def obs_ref_base_linvel_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._base_linvel_fut_buffer.reshape(-1)

    def obs_ref_base_angvel_cur(self, motion_frame_idx: int) -> np.ndarray:
        self.prepare_frame(motion_frame_idx)
        return self._base_angvel_cur_buffer

    def obs_ref_base_angvel_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        self.prepare_frame(motion_frame_idx)
        return self._base_angvel_fut_buffer.reshape(-1)

    def ref_root_quat_wxyz_cur(self, motion_frame_idx: int) -> np.ndarray:
        frame_idx = self.current_frame_idx(motion_frame_idx)
        return self._root_quat_wxyz(frame_idx)

    def ref_root_quat_wxyz_fut(self, motion_frame_idx: int) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        return self._future_root_quat_wxyz(motion_frame_idx)

    def obs_ref_future_yaw_delta_sin_cos(
        self,
        motion_frame_idx: int,
    ) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        q_cur = self.ref_root_quat_wxyz_cur(motion_frame_idx)
        q_fut = self.ref_root_quat_wxyz_fut(motion_frame_idx)
        yaw_delta = yaw_from_quat_wxyz(q_fut) - yaw_from_quat_wxyz(q_cur)
        self._future_yaw_delta_sin_cos_buffer[:, 0] = np.sin(yaw_delta)
        self._future_yaw_delta_sin_cos_buffer[:, 1] = np.cos(yaw_delta)
        return self._future_yaw_delta_sin_cos_buffer.reshape(-1)

    def obs_ref_keybody_rel_pos_cur(
        self,
        motion_frame_idx: int,
        keybody_idxs: np.ndarray,
    ) -> np.ndarray:
        clip = self._require_clip()
        keybody_idxs = np.asarray(keybody_idxs, dtype=np.int64)
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.current_frame_idx(motion_frame_idx)
        ref_body_global_pos = np.asarray(
            clip.global_translation[frame_idx], dtype=np.float32
        )
        ref_root_global_pos = ref_body_global_pos[self.root_body_idx]
        q_root_wxyz = self._root_quat_wxyz(frame_idx)
        rel_pos_w = (
            ref_body_global_pos[keybody_idxs] - ref_root_global_pos[None, :]
        )
        rel_pos_root = quat_rotate_inv_wxyz(q_root_wxyz, rel_pos_w)
        return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

    def obs_ref_keybody_rel_pos_fut(
        self,
        motion_frame_idx: int,
        keybody_idxs: np.ndarray,
    ) -> np.ndarray:
        if self.n_fut_frames <= 0:
            return np.zeros(0, dtype=np.float32)
        clip = self._require_clip()
        keybody_idxs = np.asarray(keybody_idxs, dtype=np.int64)
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros((self.n_fut_frames, 0), dtype=np.float32).reshape(
                -1
            )
        fut_idx = self.future_frame_indices(motion_frame_idx)
        q_root_wxyz = self._future_root_quat_wxyz(motion_frame_idx)
        ref_body_global_pos = np.asarray(
            clip.global_translation[fut_idx], dtype=np.float32
        )
        ref_root_global_pos = ref_body_global_pos[:, self.root_body_idx, :]
        rel_pos_w = (
            ref_body_global_pos[:, keybody_idxs, :]
            - ref_root_global_pos[:, None, :]
        )
        if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
            self._keybody_rel_pos_fut_buffer = np.zeros(
                (self.n_fut_frames, n_keybodies, 3),
                dtype=np.float32,
            )
        self._keybody_rel_pos_fut_buffer[:, :, :] = quat_rotate_inv_wxyz(
            q_root_wxyz[:, None, :],
            rel_pos_w,
        )
        return self._keybody_rel_pos_fut_buffer.reshape(-1).astype(np.float32)

    def _future_root_quat_wxyz(self, motion_frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        fut_idx = self.future_frame_indices(motion_frame_idx)
        q_root_xyzw = np.asarray(
            clip.global_rotation_quat[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        self._future_root_quat_wxyz_buffer[:, 0] = q_root_xyzw[:, 3]
        self._future_root_quat_wxyz_buffer[:, 1] = q_root_xyzw[:, 0]
        self._future_root_quat_wxyz_buffer[:, 2] = q_root_xyzw[:, 1]
        self._future_root_quat_wxyz_buffer[:, 3] = q_root_xyzw[:, 2]
        neg_mask = self._future_root_quat_wxyz_buffer[:, 0] < 0.0
        self._future_root_quat_wxyz_buffer[neg_mask] *= -1.0
        return self._future_root_quat_wxyz_buffer

    def _root_quat_wxyz(self, frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        q_root_xyzw = clip.global_rotation_quat[frame_idx, self.root_body_idx]
        q_root_wxyz = xyzw_to_wxyz(q_root_xyzw)
        return standardize_quaternion_wxyz(q_root_wxyz)

    def prepare_frame(self, motion_frame_idx: int) -> None:
        motion_frame_idx = int(motion_frame_idx)
        if self._cache_motion_frame_idx == motion_frame_idx:
            return

        clip = self._require_clip()
        frame_idx = self.current_frame_idx(motion_frame_idx)
        self._dof_pos_onnx_buffer[:] = clip.dof_pos[
            frame_idx, self.ref_to_onnx
        ]
        self._dof_vel_onnx_buffer[:] = clip.dof_vel[
            frame_idx, self.ref_to_onnx
        ]
        self._motion_states_buffer[: self.num_actions] = (
            self._dof_pos_onnx_buffer
        )
        self._motion_states_buffer[self.num_actions :] = (
            self._dof_vel_onnx_buffer
        )
        self._root_pos_cur_buffer[:] = clip.global_translation[
            frame_idx,
            self.root_body_idx,
        ]
        self._root_height_cur_buffer[0] = self._root_pos_cur_buffer[2]

        q_cur = self._root_quat_wxyz_into(frame_idx)
        self._fill_gravity(q_cur, self._gravity_cur_buffer)
        self._quat_rotate_inv_single_into(
            q_cur,
            clip.global_velocity[frame_idx, self.root_body_idx],
            self._base_linvel_cur_buffer,
        )
        self._quat_rotate_inv_single_into(
            q_cur,
            clip.global_angular_velocity[frame_idx, self.root_body_idx],
            self._base_angvel_cur_buffer,
        )

        T = self.n_fut_frames
        if T > 0:
            fut_idx = self.future_frame_indices(motion_frame_idx)
            self._pos_fut_buffer[:, :] = clip.dof_pos[fut_idx].T
            self._dof_pos_fut_onnx_buffer[:] = (
                self._pos_fut_buffer[self.ref_to_onnx, :]
                .transpose(1, 0)
                .reshape(-1)
            )
            self._root_pos_fut_buffer[:, :] = clip.global_translation[
                fut_idx,
                self.root_body_idx,
                :,
            ]
            self._h_fut_buffer[0, :] = self._root_pos_fut_buffer[:, 2]
            q_fut = self._future_root_quat_wxyz_buffer
            q_root_xyzw = clip.global_rotation_quat[
                fut_idx, self.root_body_idx
            ]
            q_fut[:, 0] = q_root_xyzw[:, 3]
            q_fut[:, 1:4] = q_root_xyzw[:, 0:3]
            neg_mask = q_fut[:, 0] < 0.0
            q_fut[neg_mask] *= -1.0
            v_fut = clip.global_velocity[fut_idx, self.root_body_idx]
            w_fut = clip.global_angular_velocity[fut_idx, self.root_body_idx]
            if T <= 16:
                self._fill_future_root_terms_small(T, q_fut, v_fut, w_fut)
            else:
                self._fill_gravity_batch(q_fut, self._gravity_fut_buffer)

                vel_T6 = self._vel_fut_T6[:T]
                vel_T6[:, :3] = v_fut
                vel_T6[:, 3:6] = w_fut
                self._quat_rotate_inv_batch_into(
                    q_fut,
                    vel_T6[:, :3],
                    self._base_linvel_fut_buffer,
                )
                self._quat_rotate_inv_batch_into(
                    q_fut,
                    vel_T6[:, 3:6],
                    self._base_angvel_fut_buffer,
                )

        self._cache_motion_frame_idx = motion_frame_idx

    def _root_quat_wxyz_into(self, frame_idx: int) -> np.ndarray:
        clip = self._require_clip()
        q_root_xyzw = clip.global_rotation_quat[frame_idx, self.root_body_idx]
        out = self._root_quat_wxyz_buffer
        out[0] = q_root_xyzw[3]
        out[1] = q_root_xyzw[0]
        out[2] = q_root_xyzw[1]
        out[3] = q_root_xyzw[2]
        if out[0] < 0.0:
            out *= -1.0
        return out

    @staticmethod
    def _fill_gravity(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        qw = q_wxyz[0]
        qx = q_wxyz[1]
        qy = q_wxyz[2]
        qz = q_wxyz[3]
        out[0] = 2.0 * (-qz * qx + qw * qy)
        out[1] = -2.0 * (qz * qy + qw * qx)
        out[2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    @staticmethod
    def _fill_gravity_batch(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        qw = q_wxyz[:, 0]
        qx = q_wxyz[:, 1]
        qy = q_wxyz[:, 2]
        qz = q_wxyz[:, 3]
        out[:, 0] = 2.0 * (-qz * qx + qw * qy)
        out[:, 1] = -2.0 * (qz * qy + qw * qx)
        out[:, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    def _fill_future_root_terms_small(
        self,
        T: int,
        q_wxyz: np.ndarray,
        linvel_w: np.ndarray,
        angvel_w: np.ndarray,
    ) -> None:
        for i in range(T):
            qw = float(q_wxyz[i, 0])
            qx = float(q_wxyz[i, 1])
            qy = float(q_wxyz[i, 2])
            qz = float(q_wxyz[i, 3])
            self._gravity_fut_buffer[i, 0] = 2.0 * (-qz * qx + qw * qy)
            self._gravity_fut_buffer[i, 1] = -2.0 * (qz * qy + qw * qx)
            self._gravity_fut_buffer[i, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)

            self._quat_rotate_inv_values_into(
                qw,
                qx,
                qy,
                qz,
                float(linvel_w[i, 0]),
                float(linvel_w[i, 1]),
                float(linvel_w[i, 2]),
                self._base_linvel_fut_buffer[i],
            )
            self._quat_rotate_inv_values_into(
                qw,
                qx,
                qy,
                qz,
                float(angvel_w[i, 0]),
                float(angvel_w[i, 1]),
                float(angvel_w[i, 2]),
                self._base_angvel_fut_buffer[i],
            )

    @staticmethod
    def _quat_rotate_inv_values_into(
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        vx: float,
        vy: float,
        vz: float,
        out: np.ndarray,
    ) -> None:
        qx = -qx
        qy = -qy
        qz = -qz
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        out[0] = vx + qw * tx + (qy * tz - qz * ty)
        out[1] = vy + qw * ty + (qz * tx - qx * tz)
        out[2] = vz + qw * tz + (qx * ty - qy * tx)

    def _quat_rotate_inv_single_into(
        self, q_wxyz: np.ndarray, v: np.ndarray, out: np.ndarray
    ) -> None:
        qx = -q_wxyz[1]
        qy = -q_wxyz[2]
        qz = -q_wxyz[3]
        qw = q_wxyz[0]
        tx = 2.0 * (qy * v[2] - qz * v[1])
        ty = 2.0 * (qz * v[0] - qx * v[2])
        tz = 2.0 * (qx * v[1] - qy * v[0])
        out[0] = v[0] + qw * tx + (qy * tz - qz * ty)
        out[1] = v[1] + qw * ty + (qz * tx - qx * tz)
        out[2] = v[2] + qw * tz + (qx * ty - qy * tx)

    def _quat_rotate_inv_batch_into(
        self, q_wxyz: np.ndarray, v: np.ndarray, out: np.ndarray
    ) -> None:
        qx = -q_wxyz[:, 1]
        qy = -q_wxyz[:, 2]
        qz = -q_wxyz[:, 3]
        qw = q_wxyz[:, 0]
        t = self._rot_t_buffer[: q_wxyz.shape[0]]
        c = self._rot_cross_buffer[: q_wxyz.shape[0]]
        t[:, 0] = 2.0 * (qy * v[:, 2] - qz * v[:, 1])
        t[:, 1] = 2.0 * (qz * v[:, 0] - qx * v[:, 2])
        t[:, 2] = 2.0 * (qx * v[:, 1] - qy * v[:, 0])
        c[:, 0] = qy * t[:, 2] - qz * t[:, 1]
        c[:, 1] = qz * t[:, 0] - qx * t[:, 2]
        c[:, 2] = qx * t[:, 1] - qy * t[:, 0]
        out[:, 0] = v[:, 0] + qw * t[:, 0] + c[:, 0]
        out[:, 1] = v[:, 1] + qw * t[:, 1] + c[:, 1]
        out[:, 2] = v[:, 2] + qw * t[:, 2] + c[:, 2]

    def _require_clip(self) -> LoadedMotionClip:
        if self.clip is None:
            raise RuntimeError(
                "Offline motion reference has no active motion clip"
            )
        return self.clip
