"""Warp CUDA staging adapter for the shared motion actor Observation kernel."""

from __future__ import annotations

import ctypes
import time

import numpy as np
import warp as wp

from holomotion.src.motion_tracking.actor_observation import (
    launch_motion_actor_observation_warp,
    motion_actor_observation_dim,
)
from holomotion.src.motion_tracking.reference_observation import (
    REFERENCE_DOF_DIM,
    REFERENCE_QPOS_DIM,
    ReferenceKinematics,
    derive_reference_kinematics_warp,
)


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


class GpuReferenceQueue:
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
        self.seen_frames = 0
        self.received = False
        self.last_receive_time: float | None = None
        self.latest_sample_time: float | None = None
        self.latest_frame_index: int | None = None
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
    ) -> bool:
        if tuple(qpos.shape) != (REFERENCE_QPOS_DIM,):
            return False
        if wp.get_device(qpos.device) != self.device:
            raise ValueError(
                f"reference qpos is on {qpos.device}, expected {self.device}"
            )

        slot = self._write_slot
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


class WarpCudaObservation:
    """CUDA-buffer protocol consumed by ONNX Runtime I/O binding."""

    is_cuda = True

    def __init__(self, array) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def buffer_ptr(self) -> int:
        return int(self.array.ptr)

    def synchronize(self) -> None:
        wp.synchronize_device(self.array.device)


class GpuMotionObservationBuilder:
    """Copy raw state once, then build the actor input with Warp CUDA."""

    def __init__(
        self,
        *,
        n_future_frames: int,
        reference_dof_indices,
        default_dof_pos,
        device: str = "cuda:0",
        fps: float = 50.0,
    ) -> None:
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")
        self.n_future_frames = int(n_future_frames)
        self.sequence_frames = self.n_future_frames + 3
        self.fps = float(fps)

        widths = {
            "qpos": self.sequence_frames * 36,
            "sample_time": self.sequence_frames,
            "robot_quat": 4,
            "robot_angvel": 3,
            "robot_dof_pos": 29,
            "robot_dof_vel": 29,
            "last_action": 29,
            "yaw_alignment": 4,
        }
        self._slices = {}
        offset = 0
        for name, width in widths.items():
            self._slices[name] = slice(offset, offset + width)
            offset += width
        self._host = wp.empty(
            offset, dtype=wp.float32, device="cpu", pinned=True
        )
        host_buffer = (ctypes.c_float * offset).from_address(self._host.ptr)
        self._host_numpy = np.ctypeslib.as_array(host_buffer)
        self._host_numpy.fill(0.0)
        self._device_staging = wp.empty(
            offset, dtype=wp.float32, device=self.device
        )
        self._device_views = {
            name: self._device_staging[field_slice].reshape(shape)
            for (name, shape), field_slice in zip(
                (
                    ("qpos", (1, self.sequence_frames, REFERENCE_QPOS_DIM)),
                    ("sample_time", (1, self.sequence_frames)),
                    ("robot_quat", (1, 4)),
                    ("robot_angvel", (1, 3)),
                    ("robot_dof_pos", (1, REFERENCE_DOF_DIM)),
                    ("robot_dof_vel", (1, REFERENCE_DOF_DIM)),
                    ("last_action", (1, REFERENCE_DOF_DIM)),
                    ("yaw_alignment", (1, 4)),
                ),
                self._slices.values(),
                strict=True,
            )
        }
        indices = np.ascontiguousarray(reference_dof_indices, dtype=np.int32)
        defaults = np.ascontiguousarray(default_dof_pos, dtype=np.float32)
        if indices.size != REFERENCE_DOF_DIM:
            raise ValueError("reference_dof_indices must contain 29 indices")
        if defaults.size != REFERENCE_DOF_DIM:
            raise ValueError("default_dof_pos must contain 29 values")
        self._reference_dof_indices = wp.from_numpy(
            indices, dtype=wp.int32, device=self.device
        )
        self._default_dof_pos = wp.from_numpy(
            defaults.reshape(1, REFERENCE_DOF_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        self._kinematics = ReferenceKinematics(
            dof_vel=wp.empty(
                (1, self.sequence_frames, REFERENCE_DOF_DIM),
                dtype=wp.float32,
                device=self.device,
            ),
            **{
                name: wp.empty(
                    (1, self.sequence_frames, 3),
                    dtype=wp.float32,
                    device=self.device,
                )
                for name in (
                    "root_linvel_world",
                    "root_angvel_world",
                    "root_linvel_local",
                    "root_angvel_local",
                    "projected_gravity",
                )
            },
        )
        self._output = wp.empty(
            (1, motion_actor_observation_dim(self.n_future_frames)),
            dtype=wp.float32,
            device=self.device,
        )
        self._graph = None
        self._capture_graph()

    def _host_view(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return self._host_numpy[self._slices[name]].reshape(shape)

    def build(
        self,
        *,
        vr_reference,
        robot_root_quat_wxyz,
        robot_root_angvel_local,
        robot_dof_pos,
        robot_dof_vel,
        last_action,
        reference_yaw_alignment_wxyz=None,
    ):
        device_reference = None
        if hasattr(vr_reference, "device_observation_sequence"):
            device_reference = vr_reference.device_observation_sequence(
                n_frames=self.n_future_frames,
                fps=self.fps,
                include_derivative_tail=True,
            )
        else:
            sequence, sample_time = vr_reference.observation_sequence(
                n_frames=self.n_future_frames,
                fps=self.fps,
                include_derivative_tail=True,
            )
            np.copyto(
                self._host_view("qpos", (self.sequence_frames, 36)),
                np.asarray(sequence, dtype=np.float32),
            )
            np.copyto(
                self._host_view("sample_time", (self.sequence_frames,)),
                np.asarray(sample_time, dtype=np.float32),
            )
        np.copyto(
            self._host_view("robot_quat", (4,)),
            np.asarray(robot_root_quat_wxyz, dtype=np.float32),
        )
        np.copyto(
            self._host_view("robot_angvel", (3,)),
            np.asarray(robot_root_angvel_local, dtype=np.float32),
        )
        np.copyto(
            self._host_view("robot_dof_pos", (29,)),
            np.asarray(robot_dof_pos, dtype=np.float32),
        )
        np.copyto(
            self._host_view("robot_dof_vel", (29,)),
            np.asarray(robot_dof_vel, dtype=np.float32),
        )
        np.copyto(
            self._host_view("last_action", (29,)),
            np.asarray(last_action, dtype=np.float32),
        )
        alignment = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
            if reference_yaw_alignment_wxyz is None
            else reference_yaw_alignment_wxyz,
            dtype=np.float32,
        )
        np.copyto(self._host_view("yaw_alignment", (4,)), alignment)

        if device_reference is None:
            self._launch_pipeline()
        else:
            self._launch_pipeline_direct(*device_reference)
        return WarpCudaObservation(self._output)

    def _launch_pipeline_direct(
        self,
        reference_qpos=None,
        reference_sample_time=None,
    ) -> None:
        wp.copy(self._device_staging, self._host)
        if reference_qpos is not None:
            wp.copy(self._device_views["qpos"], reference_qpos)
            wp.copy(
                self._device_views["sample_time"], reference_sample_time
            )
        derive_reference_kinematics_warp(
            self._device_views["qpos"],
            self._device_views["sample_time"],
            outputs=self._kinematics,
            device=self.device,
        )
        launch_motion_actor_observation_warp(
            qpos_wp=self._device_views["qpos"],
            kinematics=(
                self._kinematics.dof_vel,
                self._kinematics.root_linvel_local,
                self._kinematics.root_angvel_local,
                self._kinematics.projected_gravity,
            ),
            robot_root_quat_wp=self._device_views["robot_quat"],
            robot_root_angvel_wp=self._device_views["robot_angvel"],
            robot_dof_pos_wp=self._device_views["robot_dof_pos"],
            robot_dof_vel_wp=self._device_views["robot_dof_vel"],
            last_action_wp=self._device_views["last_action"],
            default_dof_pos_wp=self._default_dof_pos,
            reference_dof_indices_wp=self._reference_dof_indices,
            reference_yaw_alignment_wp=self._device_views["yaw_alignment"],
            use_yaw_alignment=True,
            current_index=1,
            num_future_frames=self.n_future_frames,
            output_wp=self._output,
            device=self.device,
        )

    def _launch_pipeline(self) -> None:
        if self._graph is None:
            self._launch_pipeline_direct()
        else:
            wp.capture_launch(self._graph)

    def _capture_graph(self) -> None:
        self._launch_pipeline_direct()
        wp.synchronize_device(self.device)
        capture_started = False
        try:
            wp.capture_begin(
                device=self.device,
                force_module_load=False,
            )
            capture_started = True
            self._launch_pipeline_direct()
            self._graph = wp.capture_end(device=self.device)
            capture_started = False
        except RuntimeError:
            if capture_started:
                try:
                    wp.capture_end(device=self.device)
                except RuntimeError:
                    pass
            self._graph = None


__all__ = [
    "GpuMotionObservationBuilder",
    "GpuReferenceQueue",
    "WarpCudaObservation",
]
