"""CUDA queue and asynchronous snapshot tests; never start robot control."""

import time
import unittest

import numpy as np
import warp as wp
from humanoid_policy.reference_queue import ReferenceQueue
from humanoid_policy.local_retarget import AsyncDeviceQposSnapshotter
from humanoid_policy.vr_reference import VrReference


def _qpos(frame: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [0.1 * frame, -0.2 * frame, 0.8 + 0.01 * frame]
    yaw = 0.03 * frame
    qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    qpos[7:] = frame + np.arange(29, dtype=np.float32) * 0.01
    return qpos



class ReferenceQueueTests(unittest.TestCase):
    def test_device_reference_queue_matches_fixed_rate_cpu_queue(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("Warp CUDA is unavailable")

        future = 3
        gpu_queue = ReferenceQueue(
            n_future_frames=future,
            device="cuda:0",
            fps=50.0,
        )
        cpu_queue = VrReference(n_fut_frames=future)
        for frame in range(future + 3):
            qpos = _qpos(float(frame))
            timestamp = 10.0 + frame / 50.0
            gpu_queue.store_device(
                wp.from_numpy(qpos, device="cuda:0"),
                current_time=timestamp,
                sample_time=timestamp,
                frame_index=frame,
            )
            cpu_queue.store(
                qpos,
                current_time=timestamp,
                sample_time=timestamp,
                frame_index=frame,
            )

        gpu_sequence, gpu_time = gpu_queue.device_observation_sequence(
            n_frames=future,
            fps=50.0,
            include_derivative_tail=True,
        )
        cpu_sequence, cpu_time = cpu_queue.observation_sequence(
            n_frames=future,
            fps=50.0,
            include_derivative_tail=True,
        )
        wp.synchronize_device("cuda:0")

        np.testing.assert_array_equal(gpu_sequence.numpy()[0], cpu_sequence)
        np.testing.assert_allclose(gpu_time.numpy()[0], cpu_time, atol=1.0e-7)
        np.testing.assert_array_equal(
            gpu_queue.current_qpos(), cpu_queue.current_qpos()
        )

    def test_async_snapshot_does_not_require_control_stream_sync(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("Warp CUDA is unavailable")

        snapshotter = AsyncDeviceQposSnapshotter(
            device="cuda:0",
            max_hz=50.0,
        )
        expected = _qpos(7.0)
        qpos = wp.from_numpy(expected, device="cuda:0")
        self.assertTrue(
            snapshotter.offer(
                qpos,
                frame_index=7,
                sample_meta={"timestamp_ns": 123},
            )
        )

        completed = []
        deadline = time.monotonic() + 1.0
        while not completed and time.monotonic() < deadline:
            completed = snapshotter.poll_completed()
            if not completed:
                time.sleep(0.001)
        self.assertEqual(len(completed), 1)
        actual, frame_index, sample_meta = completed[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(frame_index, 7)
        self.assertEqual(sample_meta["timestamp_ns"], 123)
