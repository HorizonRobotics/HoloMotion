"""CUDA reference queue for the existing VR application's root/joint frames.

Buffering and readiness are separate from the source-independent policy
reference API. Conversion to link poses belongs to VRJointReferenceAdapter.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np
import warp as wp

from holomotion.src.motion_tracking.reference_observation import REFERENCE_QPOS_DIM


@wp.kernel
def _store_reference_frame_kernel(
    source: wp.array(dtype=wp.float32),
    ring: wp.array2d(dtype=wp.float32),
    slot: int,
) -> None:
    column = wp.tid()
    ring[slot, column] = source[column]


@wp.kernel
def _gather_reference_ring_kernel(
    ring: wp.array2d(dtype=wp.float32),
    oldest_slot: int,
    capacity: int,
    output: wp.array3d(dtype=wp.float32),
) -> None:
    index = wp.tid()
    frame = index // REFERENCE_QPOS_DIM
    column = index - frame * REFERENCE_QPOS_DIM
    source_frame = (oldest_slot + frame) % capacity
    output[0, frame, column] = ring[source_frame, column]


class ReferenceQueue:
    """Fixed-rate device queue for local Retarget and policy Observation."""

    def __init__(
        self,
        *,
        n_future_frames: int,
        device: str = "cuda:0",
        fps: float = 50.0,
    ) -> None:
        self.n_fut_frames = max(int(n_future_frames), 0)
        self.sequence_frames = self.n_fut_frames + 3
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError("reference fps must be > 0")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")

        self._ring = wp.empty(
            (self.sequence_frames, REFERENCE_QPOS_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        self._sequence = wp.empty(
            (1, self.sequence_frames, REFERENCE_QPOS_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        relative_time = (
            np.arange(self.sequence_frames, dtype=np.float32) / self.fps
        ).reshape(1, self.sequence_frames)
        self._sample_time = wp.from_numpy(
            relative_time,
            dtype=wp.float32,
            device=self.device,
        )
        self._current_host = wp.empty(
            REFERENCE_QPOS_DIM,
            dtype=wp.float32,
            device="cpu",
            pinned=True,
        )
        host_buffer = (ctypes.c_float * REFERENCE_QPOS_DIM).from_address(
            self._current_host.ptr
        )
        self._current_host_numpy = np.ctypeslib.as_array(host_buffer)

        self._write_slot = 0
        self._record_meta_ring = [None] * self.sequence_frames
        self.seen_frames = 0
        self.received = False
        self.last_receive_time: float | None = None
        self.latest_sample_time: float | None = None
        self.latest_frame_index: int | None = None
        self.latest_device_qpos = None
        self.kinematics = None

    def warmup(self):
        """Compile queue kernels before control; synthetic data is not a live reference."""
        source = wp.zeros(REFERENCE_QPOS_DIM, dtype=wp.float32, device=self.device)
        for index in range(self.sequence_frames):
            self.store_device(source, frame_index=index)
        wp.synchronize_device(self.device)
        self._write_slot = 0
        self._record_meta_ring = [None] * self.sequence_frames
        self.seen_frames = 0
        self.received = False
        self.last_receive_time = None
        self.latest_sample_time = None
        self.latest_frame_index = None
        self.latest_device_qpos = None
        self.kinematics = None

    @property
    def has_reference(self) -> bool:
        return self.received and self.latest_device_qpos is not None

    @property
    def latest_qpos(self):
        # Deliberately avoid an implicit device-to-host copy in debug paths.
        return None

    @property
    def root_rot_queue(self):
        return None

    def data_age(self, current_time: float) -> float:
        if self.last_receive_time is None:
            return float("inf")
        return float(current_time) - self.last_receive_time

    def is_ready_for_motion(
        self, enable_teleop_reference: bool, delay_frames: int
    ) -> bool:
        if not bool(enable_teleop_reference) or not self.has_reference:
            return False
        needed = self.n_fut_frames + max(int(delay_frames), 0) + 3
        return self.seen_frames >= needed

    def has_future_sequence(
        self,
        n_frames: int | None = None,
        *,
        include_derivative_tail: bool = False,
    ) -> bool:
        count = self.n_fut_frames if n_frames is None else max(int(n_frames), 0)
        required = count + 2 + int(bool(include_derivative_tail))
        return (
            count > 0
            and required <= self.sequence_frames
            and self.seen_frames >= self.sequence_frames
        )

    def store_device(
        self,
        qpos,
        *,
        current_time: float | None = None,
        sample_time: float | None = None,
        frame_index: int | None = None,
        source_meta: dict | None = None,
    ) -> bool:
        if tuple(qpos.shape) != (REFERENCE_QPOS_DIM,):
            return False
        if wp.get_device(qpos.device) != self.device:
            raise ValueError(
                f"reference qpos is on {qpos.device}, expected {self.device}"
            )

        slot = self._write_slot
        self._record_meta_ring[slot] = {
            "frame_index": -1 if frame_index is None else int(frame_index),
            "retarget_realtime": float(time.time() if current_time is None else current_time),
            "retarget_monotonic": time.monotonic(),
            "source": source_meta,
        }
        wp.launch(
            _store_reference_frame_kernel,
            dim=REFERENCE_QPOS_DIM,
            inputs=[qpos, self._ring, slot],
            device=self.device,
        )
        self._write_slot = (slot + 1) % self.sequence_frames
        self.seen_frames += 1
        self.received = True
        self.last_receive_time = float(
            time.time() if current_time is None else current_time
        )
        self.latest_sample_time = float(
            self.last_receive_time if sample_time is None else sample_time
        )
        self.latest_frame_index = (
            None if frame_index is None else int(frame_index)
        )
        self.latest_device_qpos = qpos
        self.kinematics = None

        if self.seen_frames >= self.sequence_frames:
            wp.launch(
                _gather_reference_ring_kernel,
                dim=self.sequence_frames * REFERENCE_QPOS_DIM,
                inputs=[
                    self._ring,
                    self._write_slot,
                    self.sequence_frames,
                    self._sequence,
                ],
                device=self.device,
            )
        return True

    def sequence_record_metadata(self):
        """CPU metadata in the same oldest-to-newest device-ring order."""
        ordered = [self._record_meta_ring[(self._write_slot + i) % self.sequence_frames]
                   for i in range(self.sequence_frames)]
        if any(item is None for item in ordered):
            raise ValueError("Recording reference metadata queue is not ready")
        return {
            "consumed_reference_frame_ids": np.array(
                [item["frame_index"] for item in ordered], dtype=np.int64),
            "consumed_reference_retarget_realtime": np.array(
                [item["retarget_realtime"] for item in ordered], dtype=np.float64),
            "consumed_reference_retarget_monotonic": np.array(
                [item["retarget_monotonic"] for item in ordered], dtype=np.float64),
            "consumed_pico_sdk_timestamp_ns": np.array(
                [(item["source"] or {}).get("timestamp_ns", -1) for item in ordered], dtype=np.int64),
            "consumed_pico_received_realtime": np.array(
                [(item["source"] or {}).get("timestamp_realtime", np.nan) for item in ordered], dtype=np.float64),
            "consumed_pico_received_monotonic": np.array(
                [(item["source"] or {}).get("timestamp_monotonic", np.nan) for item in ordered], dtype=np.float64),
            "consumed_pico_body_poses": np.stack([
                (item["source"] or {}).get("body_poses", np.full((24, 7), np.nan, dtype=np.float32))
                for item in ordered]).astype(np.float32, copy=False),
            "consumed_reference_relative_times": np.arange(
                self.sequence_frames, dtype=np.float32) / self.fps,
        }

    def device_observation_sequence(
        self,
        *,
        n_frames: int | None = None,
        fps: float = 50.0,
        include_derivative_tail: bool = True,
    ):
        count = (
            self.n_fut_frames if n_frames is None else max(int(n_frames), 0)
        )
        if count != self.n_fut_frames:
            raise ValueError(
                f"device queue was allocated for {self.n_fut_frames} future frames, got {count}"
            )
        if abs(float(fps) - self.fps) > 1.0e-5:
            raise ValueError(f"device queue fps={self.fps}, requested fps={fps}")
        if not include_derivative_tail:
            raise ValueError("device Observation requires the derivative tail")
        if not self.has_future_sequence(count, include_derivative_tail=True):
            raise RuntimeError("reference future sequence is unavailable")
        return self._sequence, self._sample_time

    def current_qpos(self) -> np.ndarray:
        if self.seen_frames < self.sequence_frames:
            return np.zeros(REFERENCE_QPOS_DIM, dtype=np.float32)
        current_slot = (self._write_slot + 1) % self.sequence_frames
        wp.copy(
            self._current_host,
            self._ring,
            src_offset=current_slot * REFERENCE_QPOS_DIM,
            count=REFERENCE_QPOS_DIM,
        )
        wp.synchronize_device(self.device)
        return self._current_host_numpy.copy()

    def current_root_rot(self) -> np.ndarray | None:
        if not self.has_reference:
            return None
        return self.current_qpos()[3:7]


__all__ = ["ReferenceQueue"]
