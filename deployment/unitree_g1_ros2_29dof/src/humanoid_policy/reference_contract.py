"""Reference behavior contracts for the 29DOF policy node.

This module captures Phase 3B reference semantics in pure numpy helpers. It is
not wired into the runtime yet; it exists to lock behavior before retrying the
Phase 3C motion-reference extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LATEST_OBS_DIM = 65
REFERENCE_DOF_DIM = 29


@dataclass
class LatestObsReferenceQueues:
    n_fut_frames: int
    num_actions: int = REFERENCE_DOF_DIM
    expected_dim: int = LATEST_OBS_DIM

    def __post_init__(self) -> None:
        self.n_fut_frames = int(self.n_fut_frames)
        self.num_actions = int(self.num_actions)
        self.expected_dim = int(self.expected_dim)
        self.latest_obs: np.ndarray | None = None
        self.received = False
        self.last_obs_time: float | None = None
        self.seen_frames = 0
        self.prev_dof_pos: np.ndarray | None = None
        self.prev_dof_vel: np.ndarray | None = None
        self.prev_root_pos: np.ndarray | None = None
        self.prev_root_rot: np.ndarray | None = None
        self.prev_frame_idx: int | None = None
        if self.n_fut_frames > 0:
            self.dof_pos_queue = np.zeros(
                (self.n_fut_frames, self.num_actions), dtype=np.float32
            )
            self.dof_vel_queue = np.zeros(
                (self.n_fut_frames, self.num_actions), dtype=np.float32
            )
            self.root_pos_queue = np.zeros((self.n_fut_frames, 3), dtype=np.float32)
            self.root_rot_queue = np.zeros((self.n_fut_frames, 4), dtype=np.float32)
            self.frame_idx_queue = np.full((self.n_fut_frames,), -1, dtype=np.int32)
        else:
            self.dof_pos_queue = None
            self.dof_vel_queue = None
            self.root_pos_queue = None
            self.root_rot_queue = None
            self.frame_idx_queue = None

    def is_ready_for_motion(self, enable_teleop_reference: bool, delay_frames: int) -> bool:
        if not bool(enable_teleop_reference):
            return False
        if not (self.received and self.latest_obs is not None):
            return False
        if self.n_fut_frames <= 0:
            return True
        needed = self.n_fut_frames + max(int(delay_frames), 0) + 1
        return int(self.seen_frames) >= needed

    def store(
        self,
        arr: np.ndarray,
        current_time: float,
        frame_index: int | None = None,
    ) -> bool:
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.expected_dim:
            return False

        clipped = arr[:, : self.expected_dim].astype(np.float32, copy=False)
        self.latest_obs = clipped
        self.received = True
        self.last_obs_time = float(current_time)
        self.seen_frames += 1

        if self.n_fut_frames <= 0 or self.dof_pos_queue is None:
            return True

        latest_root_pos = clipped[0, 58:61]
        latest_root_rot = clipped[0, 61:65]
        latest_dof_pos = clipped[0, :REFERENCE_DOF_DIM]
        latest_dof_vel = clipped[0, REFERENCE_DOF_DIM : 2 * REFERENCE_DOF_DIM]
        try:
            latest_frame_idx = int(frame_index) if frame_index is not None else -1
        except Exception:
            latest_frame_idx = -1

        if self.prev_dof_pos is None:
            self.prev_dof_pos = np.empty_like(self.dof_pos_queue[0])
            self.prev_dof_vel = np.empty_like(self.dof_vel_queue[0])
            self.prev_root_pos = np.empty_like(self.root_pos_queue[0])
            if self.root_rot_queue is not None:
                self.prev_root_rot = np.empty_like(self.root_rot_queue[0])

        np.copyto(self.prev_dof_pos, self.dof_pos_queue[0])
        np.copyto(self.prev_dof_vel, self.dof_vel_queue[0])
        np.copyto(self.prev_root_pos, self.root_pos_queue[0])
        if self.root_rot_queue is not None:
            np.copyto(self.prev_root_rot, self.root_rot_queue[0])
        if self.frame_idx_queue is not None:
            try:
                self.prev_frame_idx = int(self.frame_idx_queue[0])
            except Exception:
                self.prev_frame_idx = -1

        self.dof_pos_queue[:-1] = self.dof_pos_queue[1:]
        self.dof_pos_queue[-1] = latest_dof_pos
        self.dof_vel_queue[:-1] = self.dof_vel_queue[1:]
        self.dof_vel_queue[-1] = latest_dof_vel
        self.root_pos_queue[:-1] = self.root_pos_queue[1:]
        self.root_pos_queue[-1] = latest_root_pos
        if self.root_rot_queue is not None:
            self.root_rot_queue[:-1] = self.root_rot_queue[1:]
            self.root_rot_queue[-1] = latest_root_rot
        if self.frame_idx_queue is not None:
            self.frame_idx_queue[:-1] = self.frame_idx_queue[1:]
            self.frame_idx_queue[-1] = latest_frame_idx
        return True


def future_frame_indices(frame_idx: int, n_motion_frames: int, n_fut_frames: int) -> np.ndarray:
    offsets = np.arange(1, int(n_fut_frames) + 1, dtype=np.int64)
    last_valid = int(n_motion_frames) - 1
    return np.minimum(int(frame_idx) + offsets, last_valid).astype(np.int64)


def vr_current_dof_pos(
    queues: LatestObsReferenceQueues,
    offline_ref_dof_pos: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    if queues.n_fut_frames > 0 and queues.dof_pos_queue is not None:
        if queues.prev_dof_pos is not None:
            return queues.prev_dof_pos
        return queues.dof_pos_queue[0]
    if queues.latest_obs is None:
        return offline_ref_dof_pos[frame_idx]
    return queues.latest_obs[0, :REFERENCE_DOF_DIM]


def vr_current_dof_vel(
    queues: LatestObsReferenceQueues,
    offline_ref_dof_vel: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    if queues.n_fut_frames > 0 and queues.dof_vel_queue is not None:
        if queues.prev_dof_vel is not None:
            return queues.prev_dof_vel
        return queues.dof_vel_queue[0]
    if queues.latest_obs is None:
        return offline_ref_dof_vel[frame_idx]
    return queues.latest_obs[0, REFERENCE_DOF_DIM : 2 * REFERENCE_DOF_DIM]


def vr_current_root_pos(queues: LatestObsReferenceQueues) -> np.ndarray:
    if queues.n_fut_frames > 0 and queues.root_pos_queue is not None:
        if queues.prev_root_pos is not None:
            return queues.prev_root_pos.astype(np.float32)
        return queues.root_pos_queue[0].astype(np.float32)
    if queues.latest_obs is None:
        return np.zeros(3, dtype=np.float32)
    return queues.latest_obs[0, 58:61].astype(np.float32)


def flatten_future_dof_pos_onnx(
    ref_dof_pos: np.ndarray,
    frame_idx: int,
    n_motion_frames: int,
    n_fut_frames: int,
    ref_to_onnx: list[int],
) -> np.ndarray:
    fut_idx = future_frame_indices(frame_idx, n_motion_frames, n_fut_frames)
    pos_fut = ref_dof_pos[fut_idx].T
    pos_fut_onnx = pos_fut[ref_to_onnx, :].transpose(1, 0)
    return pos_fut_onnx.reshape(-1).astype(np.float32)


def flatten_vr_future_dof_pos_onnx(
    queues: LatestObsReferenceQueues,
    n_fut_frames: int,
    ref_to_onnx: list[int],
) -> np.ndarray:
    if queues.dof_pos_queue is None or queues.dof_pos_queue.shape[0] < n_fut_frames:
        return np.zeros(REFERENCE_DOF_DIM * int(n_fut_frames), dtype=np.float32)
    pos_fut = queues.dof_pos_queue[:n_fut_frames].T
    pos_fut_onnx = pos_fut[ref_to_onnx, :].transpose(1, 0)
    return pos_fut_onnx.reshape(-1).astype(np.float32)
