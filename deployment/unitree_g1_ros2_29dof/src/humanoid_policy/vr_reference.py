"""Online reference pose queue for the 29DOF motion-tracking policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from holomotion.src.motion_tracking.reference_observation import (
    REFERENCE_DOF_DIM,
    REFERENCE_QPOS_DIM,
    ReferenceKinematics,
    derive_reference_kinematics_numpy,
)


@dataclass
class VrReference:
    """Maintain the current and future qpos frames used by policy observations."""

    n_fut_frames: int
    num_actions: int = REFERENCE_DOF_DIM
    expected_dim: int = REFERENCE_QPOS_DIM

    def __post_init__(self) -> None:
        self.n_fut_frames = max(int(self.n_fut_frames), 0)
        self.num_actions = int(self.num_actions)
        self.expected_dim = int(self.expected_dim)
        if self.num_actions != REFERENCE_DOF_DIM:
            raise ValueError(
                f"expected {REFERENCE_DOF_DIM} actions, got {self.num_actions}"
            )
        if self.expected_dim != REFERENCE_QPOS_DIM:
            raise ValueError(
                f"expected qpos dim {REFERENCE_QPOS_DIM}, got {self.expected_dim}"
            )

        self.latest_qpos: np.ndarray | None = None
        self.latest_sample_time: float | None = None
        self.received = False
        self.last_receive_time: float | None = None
        self.seen_frames = 0
        self.previous_latest_qpos: np.ndarray | None = None
        self.previous_latest_sample_time: float | None = None

        self.past_qpos: np.ndarray | None = None
        self.current_qpos_buffer: np.ndarray | None = None
        self.past_sample_time: float | None = None
        self.current_sample_time: float | None = None
        self.prev_frame_idx: int | None = None
        self.kinematics: ReferenceKinematics | None = None
        self.queue_capacity = self.n_fut_frames + 1 if self.n_fut_frames > 0 else 0

        if self.n_fut_frames > 0:
            self.qpos_queue = np.zeros(
                (self.queue_capacity, REFERENCE_QPOS_DIM), dtype=np.float32
            )
            self.sample_time_queue = np.zeros(
                self.queue_capacity, dtype=np.float64
            )
            self.frame_idx_queue = np.full(
                self.queue_capacity, -1, dtype=np.int32
            )
        else:
            self.qpos_queue = None
            self.sample_time_queue = None
            self.frame_idx_queue = None

    def is_ready_for_motion(
        self, enable_teleop_reference: bool, delay_frames: int
    ) -> bool:
        if not bool(enable_teleop_reference) or not self.has_reference:
            return False
        if self.n_fut_frames <= 0:
            return True
        needed = self.n_fut_frames + max(int(delay_frames), 0) + 3
        return int(self.seen_frames) >= needed

    @property
    def has_reference(self) -> bool:
        return self.received and self.latest_qpos is not None

    def data_age(self, current_time: float) -> float:
        if self.last_receive_time is None:
            return float("inf")
        return float(current_time) - self.last_receive_time

    def has_future_sequence(
        self,
        n_frames: int | None = None,
        *,
        include_derivative_tail: bool = False,
    ) -> bool:
        count = self._future_count(n_frames)
        required = count + int(bool(include_derivative_tail))
        return (
            count > 0
            and self.qpos_queue is not None
            and self.sample_time_queue is not None
            and self.qpos_queue.shape[0] >= required
            and self.sample_time_queue.shape[0] >= required
            and self.current_qpos_buffer is not None
        )

    def _future_count(self, n_frames: int | None = None) -> int:
        return self.n_fut_frames if n_frames is None else max(int(n_frames), 0)

    def store(
        self,
        arr: np.ndarray,
        *,
        current_time: float,
        sample_time: float | None = None,
        frame_index: int | None = None,
    ) -> bool:
        qpos = np.asarray(arr, dtype=np.float32)
        if qpos.ndim == 2:
            qpos = qpos[0]
        if qpos.shape != (self.expected_dim,):
            return False
        timestamp = float(current_time if sample_time is None else sample_time)
        if not np.isfinite(timestamp):
            timestamp = float(current_time)

        self.previous_latest_qpos = self.latest_qpos
        self.previous_latest_sample_time = (
            None if self.latest_qpos is None else self.latest_sample_time
        )
        self.latest_qpos = qpos.copy()
        self.latest_sample_time = timestamp
        self.received = True
        self.last_receive_time = float(current_time)
        self.seen_frames += 1
        self.kinematics = None

        if self.n_fut_frames <= 0 or self.qpos_queue is None:
            return True

        self.past_qpos = self.current_qpos_buffer
        self.past_sample_time = self.current_sample_time
        self.current_qpos_buffer = self.qpos_queue[0].copy()
        self.current_sample_time = float(self.sample_time_queue[0])
        if self.frame_idx_queue is not None:
            self.prev_frame_idx = int(self.frame_idx_queue[0])

        self.qpos_queue[:-1] = self.qpos_queue[1:]
        self.qpos_queue[-1] = qpos
        self.sample_time_queue[:-1] = self.sample_time_queue[1:]
        self.sample_time_queue[-1] = timestamp
        if self.frame_idx_queue is not None:
            self.frame_idx_queue[:-1] = self.frame_idx_queue[1:]
            self.frame_idx_queue[-1] = (
                int(frame_index) if frame_index is not None else -1
            )
        return True

    def update_kinematics(
        self,
        *,
        n_frames: int | None = None,
        fps: float = 50.0,
    ) -> ReferenceKinematics:
        count = self._future_count(n_frames)
        if count > 0 and self.has_future_sequence(
            count, include_derivative_tail=True
        ):
            sequence, times = self.observation_sequence(
                n_frames=count,
                fps=fps,
                include_derivative_tail=True,
            )
            all_kinematics = derive_reference_kinematics_numpy(
                sequence,
                sample_time=times,
                fps=fps,
                device="cpu",
            )
            self.kinematics = all_kinematics.sliced(slice(1, count + 2))
            return self.kinematics

        if self.latest_qpos is None:
            raise RuntimeError("reference qpos is unavailable")
        previous = (
            self.latest_qpos
            if self.previous_latest_qpos is None
            else self.previous_latest_qpos
        )
        sequence = np.stack([previous, self.latest_qpos])
        dt = 1.0 / float(fps)
        times = np.asarray([0.0, dt], dtype=np.float32)
        all_kinematics = derive_reference_kinematics_numpy(
            sequence, sample_time=times, fps=fps, device="cpu"
        )
        self.kinematics = all_kinematics.sliced(slice(-1, None))
        return self.kinematics

    def observation_sequence(
        self,
        *,
        n_frames: int | None = None,
        fps: float = 50.0,
        include_derivative_tail: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return past/current/future qpos and relative sample times."""

        count = self._future_count(n_frames)
        if not self.has_future_sequence(
            count,
            include_derivative_tail=include_derivative_tail,
        ):
            raise RuntimeError("reference future sequence is unavailable")
        tail = int(bool(include_derivative_tail))
        current = self.current_qpos()
        past = current if self.past_qpos is None else self.past_qpos
        future_count = count + tail
        future = self.qpos_queue[:future_count]
        sequence = np.concatenate([past[None], current[None], future], axis=0)
        current_time = (
            float(self.sample_time_queue[0]) - 1.0 / float(fps)
            if self.current_sample_time is None
            else float(self.current_sample_time)
        )
        past_time = (
            current_time - 1.0 / float(fps)
            if self.past_sample_time is None
            else float(self.past_sample_time)
        )
        times64 = np.concatenate(
            [
                np.asarray([past_time, current_time], dtype=np.float64),
                self.sample_time_queue[:future_count],
            ]
        )
        return sequence, (times64 - times64[0]).astype(np.float32)

    def latest_dof_pos(self) -> np.ndarray | None:
        return None if self.latest_qpos is None else self.latest_qpos[7:]

    def latest_root_pos(self) -> np.ndarray | None:
        return None if self.latest_qpos is None else self.latest_qpos[:3]

    def latest_root_rot(self) -> np.ndarray | None:
        return None if self.latest_qpos is None else self.latest_qpos[3:7]

    def current_qpos(self) -> np.ndarray:
        if self.n_fut_frames > 0 and self.current_qpos_buffer is not None:
            return self.current_qpos_buffer
        if self.latest_qpos is None:
            return np.zeros(REFERENCE_QPOS_DIM, dtype=np.float32)
        return self.latest_qpos

    def current_dof_pos(
        self, offline_fallback: np.ndarray | None = None
    ) -> np.ndarray:
        if self.has_reference:
            return self.current_qpos()[7:]
        if offline_fallback is not None:
            return offline_fallback
        return np.zeros(self.num_actions, dtype=np.float32)

    def current_dof_vel(
        self, offline_fallback: np.ndarray | None = None
    ) -> np.ndarray:
        if self.kinematics is not None:
            return self.kinematics.dof_vel[0]
        if offline_fallback is not None:
            return offline_fallback
        return np.zeros(self.num_actions, dtype=np.float32)

    def current_root_pos(self) -> np.ndarray:
        return self.current_qpos()[:3]

    def current_root_rot(self) -> np.ndarray | None:
        return None if not self.has_reference else self.current_qpos()[3:7]

    @property
    def dof_pos_queue(self) -> np.ndarray | None:
        if self.qpos_queue is None:
            return None
        return self.qpos_queue[: self.n_fut_frames, 7:]

    @property
    def root_pos_queue(self) -> np.ndarray | None:
        if self.qpos_queue is None:
            return None
        return self.qpos_queue[: self.n_fut_frames, :3]

    @property
    def root_rot_queue(self) -> np.ndarray | None:
        if self.qpos_queue is None:
            return None
        return self.qpos_queue[: self.n_fut_frames, 3:7]

    def obs_ref_dof_pos_fut(
        self,
        *,
        ref_to_onnx: np.ndarray,
        pos_fut_buffer: np.ndarray,
        n_frames: int | None = None,
    ) -> np.ndarray:
        count = self._future_count(n_frames)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(count):
            return np.zeros(self.num_actions * count, dtype=np.float32)
        pos_fut_buffer[:, :count] = self.dof_pos_queue[:count].T
        return (
            pos_fut_buffer[ref_to_onnx, :count]
            .transpose(1, 0)
            .reshape(-1)
            .astype(np.float32)
        )

    def obs_ref_root_height_fut(
        self, n_frames: int | None = None
    ) -> np.ndarray:
        count = self._future_count(n_frames)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(count):
            return np.zeros(count, dtype=np.float32)
        return self.root_pos_queue[:count, 2].astype(np.float32)

    def obs_ref_root_pos_fut(self, n_frames: int | None = None) -> np.ndarray:
        count = self._future_count(n_frames)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(count):
            return np.zeros(3 * count, dtype=np.float32)
        return self.root_pos_queue[:count].astype(np.float32).reshape(-1)


__all__ = ["VrReference"]
