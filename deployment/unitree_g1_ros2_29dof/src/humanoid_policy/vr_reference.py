"""VR/latest_obs reference queue state for the 29DOF policy node."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LATEST_OBS_DIM = 65
REFERENCE_DOF_DIM = 29


@dataclass
class VrLatestObsReference:
    """Maintain Phase 3B latest_obs current/previous/future queues."""

    n_fut_frames: int
    num_actions: int = REFERENCE_DOF_DIM
    expected_dim: int = LATEST_OBS_DIM

    def __post_init__(self) -> None:
        self.n_fut_frames = max(int(self.n_fut_frames), 0)
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
                (self.n_fut_frames, self.num_actions),
                dtype=np.float32,
            )
            self.dof_vel_queue = np.zeros(
                (self.n_fut_frames, self.num_actions),
                dtype=np.float32,
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

    @property
    def has_latest_obs(self) -> bool:
        return self.received and self.latest_obs is not None

    def data_age(self, current_time: float) -> float:
        if self.last_obs_time is None:
            return float("inf")
        return float(current_time) - self.last_obs_time

    def has_future_sequence(self, n_frames: int | None = None) -> bool:
        n_frames = self._future_count(n_frames)
        if n_frames <= 0:
            return False
        return (
            self.dof_pos_queue is not None
            and self.dof_vel_queue is not None
            and self.root_pos_queue is not None
            and self.root_rot_queue is not None
            and self.dof_pos_queue.shape[0] >= n_frames
            and self.dof_vel_queue.shape[0] >= n_frames
            and self.root_pos_queue.shape[0] >= n_frames
            and self.root_rot_queue.shape[0] >= n_frames
        )

    def _future_count(self, n_frames: int | None = None) -> int:
        if n_frames is None:
            return self.n_fut_frames
        return max(int(n_frames), 0)

    def store(
        self,
        arr: np.ndarray,
        *,
        current_time: float,
        frame_index: int | None = None,
    ) -> bool:
        """Store one latest_obs packet and advance future queues."""
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
        latest_dof_pos = clipped[0, : self.num_actions]
        latest_dof_vel = clipped[0, self.num_actions : 2 * self.num_actions]
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

    def latest_dof_pos(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, : self.num_actions]

    def latest_root_pos(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 58:61].astype(np.float32)

    def latest_root_rot(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 61:65].astype(np.float32)

    def current_dof_pos(self, offline_fallback: np.ndarray | None = None) -> np.ndarray:
        if self.n_fut_frames > 0 and self.dof_pos_queue is not None:
            if self.prev_dof_pos is not None:
                return self.prev_dof_pos
            return self.dof_pos_queue[0]
        if self.latest_obs is None:
            if offline_fallback is None:
                return np.zeros(self.num_actions, dtype=np.float32)
            return offline_fallback
        return self.latest_obs[0, : self.num_actions]

    def current_dof_vel(self, offline_fallback: np.ndarray | None = None) -> np.ndarray:
        if self.n_fut_frames > 0 and self.dof_vel_queue is not None:
            if self.prev_dof_vel is not None:
                return self.prev_dof_vel
            return self.dof_vel_queue[0]
        if self.latest_obs is None:
            if offline_fallback is None:
                return np.zeros(self.num_actions, dtype=np.float32)
            return offline_fallback
        return self.latest_obs[0, self.num_actions : 2 * self.num_actions]

    def current_root_pos(self) -> np.ndarray:
        if self.n_fut_frames > 0 and self.root_pos_queue is not None:
            if self.prev_root_pos is not None:
                return self.prev_root_pos.astype(np.float32)
            return self.root_pos_queue[0].astype(np.float32)
        if self.latest_obs is None:
            return np.zeros(3, dtype=np.float32)
        return self.latest_obs[0, 58:61].astype(np.float32)

    def current_root_rot(self) -> np.ndarray | None:
        if self.prev_root_rot is not None:
            return self.prev_root_rot
        if self.root_rot_queue is not None:
            return self.root_rot_queue[0].astype(np.float32)
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 61:65].astype(np.float32)

    def obs_ref_dof_pos_fut(
        self,
        *,
        ref_to_onnx: np.ndarray,
        pos_fut_buffer: np.ndarray,
        n_frames: int | None = None,
    ) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(self.num_actions * T, dtype=np.float32)
        pos_fut_buffer[:, :T] = self.dof_pos_queue[:T].T
        pos_fut_onnx = pos_fut_buffer[ref_to_onnx, :T].transpose(1, 0)
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def obs_ref_root_height_fut(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(T, dtype=np.float32)
        root_pos_fut = self.root_pos_queue[:T, 2].astype(np.float32)
        return root_pos_fut.reshape(-1)

    def obs_ref_root_pos_fut(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(3 * T, dtype=np.float32)
        return self.root_pos_queue[:T].astype(np.float32).reshape(-1)

    def copy_fk_sequence_inputs(
        self,
        *,
        root_pos_seq: np.ndarray,
        root_rot_seq: np.ndarray,
        dof_pos_seq: np.ndarray,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray,
        n_frames: int | None = None,
    ) -> bool:
        T = self._future_count(n_frames)
        if T <= 0 or not self.has_future_sequence(T):
            return False
        np.copyto(root_pos_seq[0, 0], cur_root_pos)
        np.copyto(root_rot_seq[0, 0], cur_root_rot)
        np.copyto(dof_pos_seq[0, 0], cur_dof_pos)
        np.copyto(root_pos_seq[0, 1 : 1 + T], self.root_pos_queue[:T])
        np.copyto(root_rot_seq[0, 1 : 1 + T], self.root_rot_queue[:T])
        np.copyto(dof_pos_seq[0, 1 : 1 + T], self.dof_pos_queue[:T])
        return True
