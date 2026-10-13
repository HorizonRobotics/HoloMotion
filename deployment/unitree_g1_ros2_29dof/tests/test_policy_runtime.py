import unittest
from types import SimpleNamespace

import numpy as np
from humanoid_policy.policy_runtime import (
    PolicyRuntime,
    load_policy_after_reference_validation,
    run_initial_setup_once,
    validate_v2_motion_action,
)
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
        self.velocity_reset_count = 0
        self.velocity_obs_builder = SimpleNamespace(reset=self.reset_velocity_history)
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
        self.motion_policy_contract = {"modes": {"Keypoint-5": []}}
        self.publish_control_count = 0
        self.load_current_motion_count = 0
        self.published_action = None
        self.policy_mode_publish_count = 0

    def reset_velocity_history(self):
        self.velocity_reset_count += 1

    def get_logger(self):
        return self.logger

    def get_parameter(self, name):
        assert name == "link_control_mode"
        return SimpleNamespace(value="Keypoint-5")

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
            ["observation_tracking", "retarget", "debug", "telemetry"],
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
        self.assertEqual(port.velocity_reset_count, 1)
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
        self.assertEqual(port.load_current_motion_count, 0)

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

    def test_v2_action_guard_rejects_nonfinite_and_wrong_shape(self):
        good = np.zeros((1, 29), dtype=np.float32)
        np.testing.assert_array_equal(validate_v2_motion_action(good), good[0])
        for bad in (
            np.zeros((1, 28), dtype=np.float32),
            np.full((1, 29), np.nan, dtype=np.float32),
            np.full((1, 29), np.inf, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                validate_v2_motion_action(bad)



def _setup_test_port(command_fps=50):
    port = FakePort()
    port.motion_config = {
        "link_observation": {"schema_version": "g1_link_xz6d_v1", "frame_offsets": [1, 2, 3, 4, 5]},
        "obs": {"target_fps": 50},
        "env": {
            "config": {
                "commands": {
                    "ref_motion": {"params": {"target_fps": command_fps}}
                }
            }
        },
    }
    port.robot_config = SimpleNamespace(policy_freq=50)
    port.runtime = PolicyRuntime(port, num_actions=port.num_actions)
    cancellations = []
    port._setup_timer = SimpleNamespace(cancel=lambda: cancellations.append(True))
    port.velocity_policy_session = None
    port.motion_policy_session = None
    return port, cancellations


def test_bad_v2_frequency_never_creates_sessions_or_publishes_action():
    port, cancellations = _setup_test_port(command_fps=60)
    attempts = []
    sessions = []
    port.load_policy = lambda: sessions.append(True)

    def setup():
        attempts.append(True)
        load_policy_after_reference_validation(port)

    port.setup = setup
    port.runtime.state.policy_enabled = True
    port._lowstate_msg = object()
    assert run_initial_setup_once(port) is False
    assert run_initial_setup_once(port) is False
    port.runtime.run()
    assert len(attempts) == 1
    assert sessions == []
    assert cancellations == [True]
    assert port._setup_failed is True
    assert port.velocity_policy_session is None
    port.runtime.state.robot_state_ready = True
    port.remote_controller.button[KeyMap.A] = 1
    low_state = SimpleNamespace(wireless_remote=b"")
    port.runtime.handle_low_state(low_state)
    assert port._lowstate_msg is low_state
    assert port.motion_policy_session is None
    assert port.runtime.state.policy_enabled is False
    assert port.published_action is None
    assert port.publish_control_count == 0


def test_partial_setup_failure_is_single_shot_and_releases_sessions():
    port, cancellations = _setup_test_port()
    attempts = []

    def setup():
        attempts.append(True)
        port.velocity_policy_session = object()
        port.motion_policy_session = object()
        raise RuntimeError("simulated setup failure")

    port.setup = setup
    assert run_initial_setup_once(port) is False
    assert run_initial_setup_once(port) is False
    assert len(attempts) == 1
    assert cancellations == [True]
    assert port.velocity_policy_session is None
    assert port.motion_policy_session is None
    assert not getattr(port, "_setup_completed", False)


def test_valid_v2_setup_loads_once_and_cancels_timer():
    port, cancellations = _setup_test_port()
    attempts = []
    sessions = []
    port.load_policy = lambda: sessions.append(True)

    def setup():
        attempts.append(True)
        load_policy_after_reference_validation(port)

    port.setup = setup
    assert run_initial_setup_once(port) is True
    assert run_initial_setup_once(port) is True
    assert len(attempts) == 1
    assert sessions == [True]
    assert cancellations == [True]
    assert port._setup_completed is True


def test_legacy_motion_model_is_rejected_before_either_session_is_loaded():
    port, cancellations = _setup_test_port()
    port.motion_config = {"obs": {"n_fut_frames": 3}}
    sessions = []
    port.load_policy = lambda: sessions.append(True)
    port.setup = lambda: load_policy_after_reference_validation(port)
    assert run_initial_setup_once(port) is False
    assert sessions == []
    assert cancellations == [True]
    assert port._setup_failed is True
    assert any("only HoloMotion v2.0" in msg for msg in port.logger.errors)



def _live_motion_runtime():
    port = FakePort()
    port._lowstate_msg = object()
    port._vr_reference = FakeVrReference(age=0.0)
    port._timing_ms = lambda start: 0.0
    port.motion_obs_builder = object()
    port.velocity_obs_builder = SimpleNamespace(reset=port.reset_velocity_history)
    port.motion_action_scale_onnx = np.ones(port.num_actions, dtype=np.float32)
    port.motion_onnx_to_real = list(range(port.num_actions))
    port.observation_evaluator = SimpleNamespace(
        cache_lowstate=lambda msg, force: None,
        clear_vr_reference_cache=lambda: None,
    )
    observation = object()
    calls = []
    port._build_motion_observation = lambda: calls.append("observation") or observation
    runtime = PolicyRuntime(port, num_actions=port.num_actions)
    runtime.state.policy_enabled = True
    runtime.state.current_policy_mode = "motion"
    runtime.state.motion_uses_vr_reference = True
    runtime._run_current_policy = lambda obs: (
        calls.append("inference") or [np.zeros((1, port.num_actions), dtype=np.float32)]
    )
    return port, runtime, calls


def test_motion_step_uses_only_the_single_observation_interface():
    port, runtime, calls = _live_motion_runtime()
    assert runtime.run_policy_step() is not None
    assert calls == ["observation", "inference"]
    np.testing.assert_array_equal(port.published_action, port.motion_default_angles_onnx)
    assert runtime.state.current_policy_mode == "motion"


def test_unavailable_pico_causes_only_one_fallback():
    port, runtime, calls = _live_motion_runtime()
    port.reference_source = "pico_local"
    port._local_retarget_source = SimpleNamespace(ready=False)
    assert runtime.run_policy_step() is None
    assert calls == []
    assert port.publish_control_count == 1
    assert port.velocity_reset_count == 1
    assert runtime.state.current_policy_mode == "velocity"
    assert not any("motion requires" in msg for msg in port.logger.infos)


def test_policy_hold_stops_inference_and_requires_default_transition():
    port, runtime, _ = _live_motion_runtime()
    runtime.state.vx, runtime.state.vy, runtime.state.vyaw = 1., 2., 3.
    runtime.set_robot_state("POLICY_HOLD")
    assert not runtime.state.policy_enabled
    assert not runtime.state.robot_state_ready
    assert not runtime.state.motion_uses_vr_reference
    assert runtime.state.current_policy_mode == "velocity"
    assert (runtime.state.vx, runtime.state.vy, runtime.state.vyaw) == (0., 0., 0.)
    port.remote_controller.button[KeyMap.A] = 1
    runtime.handle_low_state(SimpleNamespace(wireless_remote=b""))
    assert not runtime.state.policy_enabled


def test_recording_freezes_consumed_inputs_before_next_reference_overwrite():
    port = FakePort()
    port._setup_completed = True
    port.reference_source = "pico_local"
    port._timing_ms = lambda start: 0.
    port._record_timing_sample = lambda sample: None
    port._teleop_recorder = object()
    reference = [10]
    recorded = []
    port._poll_local_retarget_reference = lambda: reference.__setitem__(0, 20)
    port._publish_reference = lambda: None
    port._flush_reference_telemetry = lambda **kwargs: None
    port._flush_policy_recording = lambda **kwargs: recorded.append(
        (reference[0], port._teleop_record_inputs)
    )
    runtime = PolicyRuntime(port, num_actions=port.num_actions)
    runtime._log_vr_status = runtime._log_vr_ready = lambda now: None
    runtime._log_policy_latency = lambda elapsed: None

    def infer():
        port._teleop_record_inputs = reference[0]
        return {"policy_total_ms": 1.}

    runtime.run_policy_step = infer
    runtime.run()
    assert recorded == [(10, 10)]
    assert reference[0] == 20


def test_invalid_motion_reference_falls_back_without_inference_or_publication():
    port, runtime, calls = _live_motion_runtime()

    def invalid_reference():
        calls.append("observation")
        raise ValueError("invalid test reference")

    port._build_motion_observation = invalid_reference
    assert runtime.run_policy_step() is None
    assert calls == ["observation"]
    assert port.published_action is None
    assert runtime.state.current_policy_mode == "velocity"


def test_invalid_motion_action_falls_back_without_publication():
    port, runtime, calls = _live_motion_runtime()
    runtime._run_current_policy = lambda obs: [
        np.full((1, port.num_actions), np.nan, dtype=np.float32)
    ]
    assert runtime.run_policy_step() is None
    assert calls == ["observation"]
    assert port.published_action is None
    assert runtime.state.current_policy_mode == "velocity"


def test_motion_without_live_reference_never_builds_observation():
    port, runtime, calls = _live_motion_runtime()
    runtime.state.motion_uses_vr_reference = False
    assert runtime.run_policy_step() is None
    assert calls == []
    assert port.published_action is None
    assert runtime.state.current_policy_mode == "velocity"


def test_invalid_control_mode_does_not_enter_motion_or_publish_parameters():
    port = FakePort()
    port.vr_ready = True
    port._vr_reference = FakeVrReference(age=0.0)
    port.get_parameter = lambda name: SimpleNamespace(value="unsupported-mode")
    runtime = PolicyRuntime(port, num_actions=port.num_actions)
    assert runtime.switch_to_motion_mode() is False
    assert runtime.state.current_policy_mode == "velocity"
    assert port.publish_control_count == 0


if __name__ == "__main__":
    unittest.main()
