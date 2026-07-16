import unittest
from types import SimpleNamespace

import numpy as np

from humanoid_policy.policy_runtime import PolicyRuntime
from humanoid_policy.utils.remote_controller_filter import KeyMap


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errors = []

    def info(self, msg):
        self.infos.append(str(msg))

    def warn(self, msg):
        self.warns.append(str(msg))

    def error(self, msg):
        self.errors.append(str(msg))


class FakeRemote:
    def __init__(self):
        self.button = [0] * 16
        self.velocity = (0.1, -0.2, 0.3)

    def set(self, data):
        del data

    def get_velocity_commands(self):
        return self.velocity


class FakeVrReference:
    has_reference = True

    def __init__(self, age):
        self.age = float(age)
        self.seen_frames = 16

    def data_age(self, current_time):
        del current_time
        return self.age


class FakePort:
    def __init__(self):
        self.logger = FakeLogger()
        self.remote_controller = FakeRemote()
        self.num_actions = 3
        self.use_kv_cache = False
        self.motion_kv_cache = None
        self.enable_teleop_reference = True
        self.max_data_age = 0.5
        self.dt = 0.02
        self.velocity_default_angles_onnx = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.motion_default_angles_onnx = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.all_motion_data = [object(), object(), object()]
        self.motion_file_names = ["clip0.npz", "clip1.npz", "clip2.npz"]
        self._lowstate_msg = None
        self._vr_reference = None
        self._use_fk_vr = False
        self.vr_ready = False
        self.publish_control_count = 0
        self.load_current_motion_count = 0
        self.published_action = None
        self.policy_mode_publish_count = 0

    def get_logger(self):
        return self.logger

    def _publish_control_params(self):
        self.publish_control_count += 1

    def _is_vr_ready_for_motion(self):
        return self.vr_ready

    def _load_current_motion(self):
        self.load_current_motion_count += 1

    def clear_vr_reference_cache(self):
        self._use_fk_vr = False

    def _publish_action_target(self, target):
        self.published_action = None if target is None else np.asarray(target)

    def _publish_policy_mode(self):
        self.policy_mode_publish_count += 1


class FakeObservationEvaluator:
    def __init__(self):
        self._use_fk_vr = True

    def clear_vr_reference_cache(self):
        self._use_fk_vr = False


class PolicyRuntimeTest(unittest.TestCase):
    def test_local_retarget_observation_tracking_and_telemetry_are_serial(self):
        port = FakePort()
        port._setup_completed = True
        port.reference_source = "pico_local"
        port._timing_ms = lambda start: 0.0
        port._record_timing_sample = lambda sample: None
        order = []
        port._poll_local_retarget_reference = lambda: order.append("retarget")
        port._poll_zmq_reference = lambda: order.append("zmq")
        port._publish_reference = lambda: order.append("debug")
        port._flush_reference_telemetry = (
            lambda **kwargs: order.append("telemetry")
        )
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime._log_vr_status = lambda now: None
        runtime._log_vr_ready = lambda now: None
        runtime._log_policy_latency = lambda elapsed: None
        runtime._record_tracking_rate = lambda now, completed: None
        runtime.run_policy_step = lambda: (
            order.append("observation_tracking")
            or {"policy_total_ms": 1.0}
        )

        runtime.run()

        self.assertEqual(
            order,
            ["retarget", "observation_tracking", "debug", "telemetry"],
        )

    def test_a_button_enables_velocity_without_vr_reference(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.set_robot_state("MOVE_TO_DEFAULT")

        port.remote_controller.button[KeyMap.A] = 1
        runtime.handle_low_state(SimpleNamespace(wireless_remote=b""))

        self.assertTrue(runtime.state.policy_enabled)
        self.assertEqual(runtime.state.current_policy_mode, "velocity")
        self.assertFalse(runtime.state.reference_stream_active)
        np.testing.assert_allclose(
            runtime.state.target_dof_pos_onnx,
            port.velocity_default_angles_onnx,
        )
        self.assertEqual(port.publish_control_count, 1)
        self.assertEqual((runtime.state.vx, runtime.state.vy, runtime.state.vyaw), port.remote_controller.velocity)

    def test_b_button_refuses_motion_until_vr_ready_when_teleop_enabled(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.set_robot_state("MOVE_TO_DEFAULT")
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "velocity"
        port._vr_reference = FakeVrReference(age=0.0)
        port.vr_ready = False

        port.remote_controller.button[KeyMap.B] = 1
        runtime.handle_low_state(SimpleNamespace(wireless_remote=b""))

        self.assertEqual(runtime.state.current_policy_mode, "velocity")
        self.assertFalse(runtime.state.reference_stream_active)
        self.assertEqual(port.load_current_motion_count, 0)
        self.assertTrue(
            any("VR queue is not ready" in msg for msg in port.logger.warns)
        )

    def test_b_button_switches_to_vr_motion_when_ready(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.set_robot_state("MOVE_TO_DEFAULT")
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "velocity"
        port._vr_reference = FakeVrReference(age=0.0)
        port.vr_ready = True

        port.remote_controller.button[KeyMap.B] = 1
        runtime.handle_low_state(SimpleNamespace(wireless_remote=b""))

        self.assertEqual(runtime.state.current_policy_mode, "motion")
        self.assertTrue(runtime.state.reference_stream_active)
        self.assertTrue(runtime.state.motion_in_progress)
        np.testing.assert_allclose(
            runtime.state.target_dof_pos_onnx,
            port.motion_default_angles_onnx,
        )
        self.assertEqual(port.load_current_motion_count, 1)

    def test_vr_queue_ready_log_does_not_enable_vr_reference_in_velocity_mode(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)

        runtime.mark_vr_queue_ready()

        self.assertFalse(runtime.state.reference_stream_active)
        self.assertTrue(runtime.state.vr_reference_started_logged)

    def test_stale_vr_motion_falls_back_to_velocity(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "motion"
        runtime.state.reference_stream_active = True
        port._lowstate_msg = object()
        port._vr_reference = FakeVrReference(age=1.0)

        result = runtime.run_policy_step()

        self.assertIsNone(result)
        self.assertEqual(runtime.state.current_policy_mode, "velocity")
        self.assertFalse(runtime.state.reference_stream_active)
        self.assertTrue(
            any("VR reference_qpos stale" in msg for msg in port.logger.infos)
        )

    def test_switch_to_velocity_clears_reference_cache(self):
        port = FakePort()
        port.observation_evaluator = FakeObservationEvaluator()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)

        runtime.switch_to_velocity_mode(reason="test")

        self.assertFalse(port.observation_evaluator._use_fk_vr)

    def test_motion_clip_selection_lives_in_runtime(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.set_robot_state("MOVE_TO_DEFAULT")
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "velocity"

        port.remote_controller.button[KeyMap.down] = 1
        runtime.handle_low_state(SimpleNamespace(wireless_remote=b""))

        self.assertEqual(runtime.state.current_motion_clip_index, 1)
        self.assertTrue(
            any("Selected next motion clip" in msg for msg in port.logger.infos)
        )

    def test_action_mapping_lives_in_runtime(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=2)

        first = runtime.apply_policy_output(
            np.array([1.0, 3.0], dtype=np.float32),
            action_scale=np.array([2.0, 2.0], dtype=np.float32),
            default_angles=np.array([0.5, 0.5], dtype=np.float32),
            onnx_to_real=[1, 0],
            is_motion=True,
        )
        second = runtime.apply_policy_output(
            np.array([3.0, 5.0], dtype=np.float32),
            action_scale=np.array([2.0, 2.0], dtype=np.float32),
            default_angles=np.array([0.5, 0.5], dtype=np.float32),
            onnx_to_real=[1, 0],
            is_motion=True,
        )

        np.testing.assert_allclose(first, np.array([6.5, 2.5], dtype=np.float32))
        np.testing.assert_allclose(second, np.array([10.5, 6.5], dtype=np.float32))

    def test_tracking_rate_counts_completed_policy_steps(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "motion"
        runtime.state.tracking_rate_window_start = 10.0
        runtime.state.tracking_completed_in_window = 49

        runtime._record_tracking_rate(15.0, completed=True)

        self.assertTrue(
            any(
                "[Tracking] mode=motion actual=10.0Hz target=50.0Hz" in msg
                for msg in port.logger.infos
            )
        )
        self.assertEqual(runtime.state.tracking_completed_in_window, 0)

    def test_emergency_stop_disables_policy_inference(self):
        port = FakePort()
        runtime = PolicyRuntime(port, num_actions=port.num_actions)
        runtime.state.robot_state_ready = True
        runtime.state.policy_enabled = True
        runtime.state.current_policy_mode = "motion"
        runtime.state.motion_in_progress = True
        runtime.state.motion_uses_vr_reference = True
        runtime.state.vx = 1.0
        runtime.state.tracking_rate_window_start = 10.0
        runtime.state.tracking_completed_in_window = 50
        port._lowstate_msg = object()

        runtime.set_robot_state("EMERGENCY_STOP")

        self.assertFalse(runtime.state.robot_state_ready)
        self.assertFalse(runtime.state.policy_enabled)
        self.assertFalse(runtime.state.motion_in_progress)
        self.assertFalse(runtime.state.reference_stream_active)
        self.assertEqual((runtime.state.vx, runtime.state.vy, runtime.state.vyaw), (0.0, 0.0, 0.0))
        self.assertIsNone(runtime.state.tracking_rate_window_start)
        self.assertEqual(runtime.state.tracking_completed_in_window, 0)
        self.assertIsNone(runtime.run_policy_step())
        self.assertTrue(
            any("Policy inference stopped" in msg for msg in port.logger.infos)
        )


if __name__ == "__main__":
    unittest.main()
