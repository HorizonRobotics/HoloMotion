import unittest
import time
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from holomotion.src.motion_tracking.actor_observation import (
    MotionActorObservationInput,
    build_motion_actor_observation_torch,
)
from humanoid_policy.gpu_motion_observation import GpuMotionObservationBuilder
from humanoid_policy.gpu_motion_observation import GpuReferenceQueue
from humanoid_policy.local_retarget import AsyncDeviceQposSnapshotter
from humanoid_policy.observation_evaluator import PolicyObservationEvaluator
from humanoid_policy.vr_reference import VrReference


def _qpos(frame: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [0.1 * frame, -0.2 * frame, 0.8 + 0.01 * frame]
    yaw = 0.03 * frame
    qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    qpos[7:] = frame + np.arange(29, dtype=np.float32) * 0.01
    return qpos


class GpuMotionObservationBuilderTests(unittest.TestCase):
    def test_device_reference_queue_matches_fixed_rate_cpu_queue(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("Warp CUDA is unavailable")

        future = 3
        gpu_queue = GpuReferenceQueue(
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

        indices = np.arange(29, dtype=np.int32)
        default = np.zeros(29, dtype=np.float32)
        builder = GpuMotionObservationBuilder(
            n_future_frames=future,
            reference_dof_indices=indices,
            default_dof_pos=default,
            device="cuda:0",
        )
        build_kwargs = {
            "robot_root_quat_wxyz": np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "robot_root_angvel_local": np.zeros(3, dtype=np.float32),
            "robot_dof_pos": np.zeros(29, dtype=np.float32),
            "robot_dof_vel": np.zeros(29, dtype=np.float32),
            "last_action": np.zeros(29, dtype=np.float32),
        }
        cpu_observation = builder.build(
            vr_reference=cpu_queue,
            **build_kwargs,
        )
        cpu_observation.synchronize()
        cpu_value = cpu_observation.array.numpy().copy()
        gpu_observation = builder.build(
            vr_reference=gpu_queue,
            **build_kwargs,
        )
        gpu_observation.synchronize()
        np.testing.assert_allclose(
            gpu_observation.array.numpy(),
            cpu_value,
            atol=1.0e-6,
            rtol=1.0e-6,
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

    def test_gpu_matches_deployment_cpu_terms(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("Warp CUDA is unavailable")

        future = 10
        names = [f"joint_{index}" for index in range(29)]
        onnx_names = list(reversed(names))
        default = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
        real_pos = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        real_vel = np.linspace(2.0, -2.0, 29, dtype=np.float32)
        robot_yaw = 0.35
        lowstate = SimpleNamespace(
            imu_state=SimpleNamespace(
                quaternion=[
                    np.cos(robot_yaw / 2.0),
                    0.0,
                    0.0,
                    np.sin(robot_yaw / 2.0),
                ],
                gyroscope=[0.1, -0.2, 0.3],
            ),
            motor_state=[
                SimpleNamespace(q=float(q), dq=float(dq))
                for q, dq in zip(real_pos, real_vel, strict=True)
            ],
        )

        class Logger:
            def info(self, message):
                del message

            def warn(self, message):
                raise AssertionError(str(message))

        node = SimpleNamespace(
            real_dof_names=names,
            dof_names_ref_motion=names,
            velocity_dof_names_onnx=onnx_names,
            motion_dof_names_onnx=onnx_names,
            velocity_default_angles_onnx=default,
            motion_default_angles_onnx=default,
            n_fut_frames=future,
            num_actions=29,
            actions_dim=29,
            current_policy_mode="motion",
            reference_stream_active=True,
            motion_frame_idx=0,
            n_motion_frames=1,
            actions_onnx=np.linspace(0.5, -0.5, 29, dtype=np.float32),
            actor_place_holder_ndim=0,
            root_body_idx=0,
            _lowstate_msg=lowstate,
            get_logger=lambda: Logger(),
        )
        reference = VrReference(n_fut_frames=future)
        for frame in range(future + 4):
            sample_time = 20.0 + 0.018 * frame + 0.0002 * frame * frame
            reference.store(
                _qpos(float(frame)),
                current_time=sample_time,
                sample_time=sample_time,
                frame_index=frame,
            )
        node._vr_reference = reference

        evaluator = PolicyObservationEvaluator(node)
        evaluator.initialize_observation_state()
        evaluator.cache_lowstate(lowstate, force=True)
        reference.update_kinematics(n_frames=future)
        evaluator.begin_motion_yaw_alignment()

        cpu_terms = (
            evaluator._get_obs_ref_gravity_projection_cur(),
            evaluator._get_obs_ref_base_linvel_cur(),
            evaluator._get_obs_ref_base_angvel_cur(),
            evaluator._get_obs_ref_dof_pos_cur(),
            evaluator._get_obs_ref_root_height_cur(),
            evaluator._get_obs_ref_robot_yaw_error_sin_cos(),
            evaluator._get_obs_projected_gravity(),
            evaluator._get_obs_rel_robot_root_ang_vel(),
            evaluator._get_obs_dof_pos(),
            evaluator._get_obs_dof_vel(),
            evaluator._get_obs_last_action(),
            evaluator._get_obs_ref_dof_pos_fut(),
            evaluator._get_obs_ref_root_height_fut(),
            evaluator._get_obs_ref_gravity_projection_fut(),
            evaluator._get_obs_ref_base_linvel_fut(),
            evaluator._get_obs_ref_base_angvel_fut(),
            evaluator._get_obs_ref_future_yaw_delta_sin_cos(),
            evaluator._get_obs_ref_future_root_ori_robot_frame_6d(),
        )
        expected = np.concatenate(
            [np.asarray(term, dtype=np.float32).reshape(-1) for term in cpu_terms]
        )

        builder = GpuMotionObservationBuilder(
            n_future_frames=future,
            reference_dof_indices=evaluator.ref_to_onnx,
            default_dof_pos=default,
            device="cuda:0",
        )
        motion_real_indices = evaluator.motion_dof_real_indices_np
        actual = builder.build(
            vr_reference=reference,
            robot_root_quat_wxyz=evaluator.robot_root_rot_quat_wxyz,
            robot_root_angvel_local=evaluator.robot_root_ang_vel,
            robot_dof_pos=real_pos[motion_real_indices],
            robot_dof_vel=real_vel[motion_real_indices],
            last_action=node.actions_onnx,
            reference_yaw_alignment_wxyz=(
                evaluator._motion_ref_yaw_alignment_quat_wxyz
            ),
        )
        actual.synchronize()

        self.assertEqual(expected.shape, (604,))
        np.testing.assert_allclose(
            actual.array.numpy()[0], expected, atol=1.0e-5, rtol=1.0e-5
        )

    def test_staging_matches_direct_shared_builder(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        future = 3
        reference = VrReference(n_fut_frames=future)
        sample_times = [10.00, 10.02, 10.05, 10.09, 10.14, 10.20, 10.27]
        for frame, sample_time in enumerate(sample_times):
            reference.store(
                _qpos(float(frame)),
                current_time=sample_time,
                sample_time=sample_time,
                frame_index=frame,
            )

        indices = np.arange(28, -1, -1, dtype=np.int32)
        default = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
        robot_quat = np.asarray([0.98, 0.0, 0.0, 0.2], dtype=np.float32)
        robot_quat /= np.linalg.norm(robot_quat)
        robot_angvel = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        robot_dof_pos = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        robot_dof_vel = np.linspace(1.0, -1.0, 29, dtype=np.float32)
        last_action = np.linspace(0.5, -0.5, 29, dtype=np.float32)
        alignment = np.asarray([0.99, 0.0, 0.0, 0.1], dtype=np.float32)
        alignment /= np.linalg.norm(alignment)

        builder = GpuMotionObservationBuilder(
            n_future_frames=future,
            reference_dof_indices=indices,
            default_dof_pos=default,
            device="cuda:0",
        )
        actual = builder.build(
            vr_reference=reference,
            robot_root_quat_wxyz=robot_quat,
            robot_root_angvel_local=robot_angvel,
            robot_dof_pos=robot_dof_pos,
            robot_dof_vel=robot_dof_vel,
            last_action=last_action,
            reference_yaw_alignment_wxyz=alignment,
        )

        sequence, sample_time = reference.observation_sequence(
            n_frames=future,
            include_derivative_tail=True,
        )
        expected = build_motion_actor_observation_torch(
            MotionActorObservationInput(
                reference_qpos=torch.from_numpy(sequence).cuda().unsqueeze(0),
                reference_sample_time=(
                    torch.from_numpy(sample_time).cuda().unsqueeze(0)
                ),
                robot_root_quat_wxyz=torch.from_numpy(robot_quat).cuda(),
                robot_root_angvel_local=torch.from_numpy(robot_angvel).cuda(),
                robot_dof_pos=torch.from_numpy(robot_dof_pos).cuda(),
                robot_dof_vel=torch.from_numpy(robot_dof_vel).cuda(),
                last_action=torch.from_numpy(last_action).cuda(),
                default_dof_pos=torch.from_numpy(default).cuda(),
                reference_dof_indices=torch.from_numpy(indices).cuda(),
                reference_yaw_alignment_wxyz=torch.from_numpy(alignment).cuda(),
            ),
            current_index=1,
            num_future_frames=future,
        )
        actual.synchronize()
        torch.cuda.synchronize()

        self.assertEqual(tuple(actual.shape), (1, 275))
        np.testing.assert_array_equal(actual.array.numpy(), expected.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
