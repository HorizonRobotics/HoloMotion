"""Runtime state machine and policy loop for the 29DOF policy node."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from humanoid_policy.motion_clip_library import select_motion_clip_index
from humanoid_policy.utils.remote_controller_filter import KeyMap


def _default_button_states() -> dict[int, int]:
    return {
        KeyMap.up: 0,
        KeyMap.down: 0,
        KeyMap.left: 0,
        KeyMap.right: 0,
        KeyMap.A: 0,
        KeyMap.B: 0,
        KeyMap.Y: 0,
    }


@dataclass
class PolicyRuntimeState:
    """Mutable policy runtime state kept outside the ROS node."""

    num_actions: int = 29
    policy_enabled: bool = False
    robot_state_ready: bool = False
    current_policy_mode: str = "velocity"
    motion_uses_vr_reference: bool = False
    motion_frame_idx: int = 0
    motion_step_idx: int = 0
    motion_in_progress: bool = False
    current_motion_clip_index: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    target_dof_pos_real: np.ndarray | None = None
    actions_onnx: np.ndarray = field(init=False)
    target_dof_pos_onnx: np.ndarray = field(init=False)
    target_dof_pos_real_buffers: list[np.ndarray] = field(init=False)
    target_dof_pos_real_buffer: np.ndarray = field(init=False)
    target_dof_pos_real_buffer_idx: int = field(init=False)
    motion_filtered_actions_onnx: np.ndarray | None = None
    last_button_states: dict[int, int] = field(default_factory=_default_button_states)
    last_vr_status_log_time: float | None = None
    vr_queue_ready_logged: bool = False
    vr_fk_started_logged: bool = False
    vr_cold_start_logged: bool = False
    policy_slow_count: int = 0

    def __post_init__(self) -> None:
        self.resize_actions(self.num_actions)

    @property
    def latest_obs_flag(self) -> bool:
        """Compatibility alias for old node code.

        In Phase 3D this means "motion mode uses live VR reference"; it no longer
        means that the VR queue has merely become ready while still in velocity
        mode.
        """

        return bool(self.motion_uses_vr_reference)

    @latest_obs_flag.setter
    def latest_obs_flag(self, value: bool) -> None:
        self.motion_uses_vr_reference = bool(value)

    def resize_actions(self, num_actions: int) -> None:
        self.num_actions = int(num_actions)
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_real_buffers = [
            np.zeros(self.num_actions, dtype=np.float32),
            np.zeros(self.num_actions, dtype=np.float32),
        ]
        self.target_dof_pos_real_buffer_idx = 0
        self.target_dof_pos_real_buffer = self.target_dof_pos_real_buffers[0]
        self.target_dof_pos_real = self.target_dof_pos_real_buffer
        self.motion_filtered_actions_onnx = None


class PolicyRuntime:
    """Owns mode transitions, VR gating, and policy-step orchestration."""

    def __init__(self, port: Any, num_actions: int = 29):
        self.port = port
        self.state = PolicyRuntimeState(num_actions=int(num_actions))
        self._velocity_input_feed: dict[str, np.ndarray] | None = None
        self._velocity_output_names: list[str] | None = None
        self._motion_input_feed: dict[str, np.ndarray] | None = None
        self._motion_output_names: list[str] | None = None
        self._motion_step_idx_array: np.ndarray | None = None

    def resize_actions(self, num_actions: int) -> None:
        self.state.resize_actions(int(num_actions))

    def _obs_evaluator(self):
        return getattr(self.port, "observation_evaluator", self.port)

    def set_robot_state(self, robot_state: str) -> None:
        if robot_state == "MOVE_TO_DEFAULT":
            self.state.robot_state_ready = True
        elif robot_state in {"ZERO_TORQUE", "EMERGENCY_STOP"}:
            self.state.robot_state_ready = False

    def handle_low_state(self, ls_msg: Any) -> None:
        """Process low-state input and remote-controller mode transitions."""

        self.port._lowstate_msg = ls_msg
        self.port.remote_controller.set(ls_msg.wireless_remote)

        if self._is_button_pressed(KeyMap.A) and self.state.robot_state_ready:
            self.enable_velocity_policy()

        if (
            self._is_button_pressed(KeyMap.B)
            and self.state.robot_state_ready
            and self.state.policy_enabled
            and self.state.current_policy_mode == "velocity"
        ):
            self.switch_to_motion_mode()

        if (
            self._is_button_pressed(KeyMap.Y)
            and self.state.robot_state_ready
            and self.state.policy_enabled
            and self.state.current_policy_mode == "motion"
        ):
            self.switch_to_velocity_mode()

        if self.state.current_policy_mode == "velocity":
            self.state.vx, self.state.vy, self.state.vyaw = (
                self.port.remote_controller.get_velocity_commands()
            )
        else:
            self.state.vx, self.state.vy, self.state.vyaw = 0.0, 0.0, 0.0

        if (
            self.state.current_policy_mode == "velocity"
            and self.state.policy_enabled
            and self.state.robot_state_ready
        ):
            self._handle_motion_clip_selection()

    def enable_velocity_policy(self) -> None:
        self.state.policy_enabled = True
        self.state.current_policy_mode = "velocity"
        self.state.motion_uses_vr_reference = False
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = (
            self.port.velocity_default_angles_onnx.copy()
        )
        self.port._publish_control_params()
        self.port.get_logger().info(
            f"Policy enabled in {self.state.current_policy_mode} tracking mode"
        )

    def switch_to_velocity_mode(self, reason: str = "") -> None:
        self.state.current_policy_mode = "velocity"
        self.state.motion_uses_vr_reference = False
        self.state.motion_in_progress = False
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "clear_vr_fk_cache"):
            obs_eval.clear_vr_fk_cache()
        else:
            obs_eval._fk_vr_out = None
            obs_eval._use_fk_vr = False
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = (
            self.port.velocity_default_angles_onnx.copy()
        )
        self.port._publish_control_params()
        if reason:
            self.port.get_logger().info(f"Switched to velocity tracking mode ({reason})")
        else:
            self.port.get_logger().info("Switched to velocity tracking mode")

    def switch_to_motion_mode(self) -> bool:
        vr_data_available = self._vr_reference_available()
        vr_ready = self.port._is_vr_ready_for_motion()
        if self.port.enable_teleop_reference and not vr_ready:
            self.port.get_logger().warn(
                "VR teleoperation is enabled but the VR queue is not ready yet; staying in velocity mode."
            )
            return False

        if getattr(self.port, "all_motion_data", None):
            self.port._load_current_motion()

        self.state.current_policy_mode = "motion"
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        if self.port.use_kv_cache:
            self.port.get_logger().info("Motion KV-Cache reset.")
        self.port.get_logger().info("Motion Step Index reset to 0.")
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = self.port.motion_default_angles_onnx.copy()
        self.port._publish_control_params()

        self.state.motion_uses_vr_reference = bool(
            self.port.enable_teleop_reference and vr_data_available
        )
        source_mode = (
            "ZMQ latest_obs"
            if self.state.motion_uses_vr_reference
            else "offline motion"
        )
        self.port.get_logger().info(
            f"Switched to motion tracking mode ({source_mode}) - "
            f"motion clip index: {self.state.current_motion_clip_index}"
        )
        if self.state.motion_uses_vr_reference:
            self.port.get_logger().info("[VR] Reference trajectory source: ZMQ latest_obs")
            self._obs_evaluator()._warmup_fk_for_vr()
        self.state.motion_in_progress = True
        return True

    def reset_counters(self) -> None:
        self.state.motion_frame_idx = 0
        self.state.motion_step_idx = 0
        if self.port.use_kv_cache and self.port.motion_kv_cache is not None:
            self.port.motion_kv_cache.fill(0)

    def reset_motion_action_ema_filter(self) -> None:
        self.state.motion_filtered_actions_onnx = None

    def apply_motion_action_ema_filter(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
        if not self.port.motion_action_ema_filter_enabled:
            return raw_actions.copy()

        if self.state.motion_filtered_actions_onnx is None:
            self.state.motion_filtered_actions_onnx = raw_actions.copy()
            return self.state.motion_filtered_actions_onnx

        alpha = float(self.port.motion_action_ema_filter_alpha)
        filtered_actions = self.state.motion_filtered_actions_onnx
        filtered_actions *= 1.0 - alpha
        filtered_actions += alpha * raw_actions
        return filtered_actions

    def mark_vr_queue_ready(self) -> None:
        if self.state.vr_fk_started_logged:
            return
        self.port.get_logger().info(
            "[VR] ZMQ data is ready; the main thread will build the reference trajectory from live ZMQ input."
        )
        self.state.vr_fk_started_logged = True

    def run(self) -> None:
        if not getattr(self.port, "_setup_completed", False):
            return

        t_loop_start = time.perf_counter()
        now = time.time()
        t_io = time.perf_counter()
        self._consume_ros_latest_obs()
        self.port._poll_zmq_latest_obs()
        self._log_vr_status(now)
        self._log_vr_ready(now)
        self.port._publish_latest_obs()
        io_ms = self.port._timing_ms(t_io)

        policy_timing = self.run_policy_step()
        run_elapsed = 0.0
        if policy_timing is not None:
            run_elapsed = float(policy_timing.get("policy_total_ms", 0.0)) / 1000.0
        self._log_policy_latency(run_elapsed)

        if policy_timing is not None:
            sample = dict(policy_timing)
            sample["io_ms"] = io_ms
            sample["loop_total_ms"] = self.port._timing_ms(t_loop_start)
            self.port._record_timing_sample(sample)

    def run_policy_step(self) -> dict[str, float] | None:
        if self.port._lowstate_msg is None or not self.state.policy_enabled:
            return None

        timing_info = {
            "policy_total_ms": 0.0,
            "fk_ms": 0.0,
            "obs_ms": 0.0,
            "onnx_ms": 0.0,
            "post_ms": 0.0,
        }
        t_policy_start = time.perf_counter()

        if self.state.current_policy_mode == "motion":
            if self.state.motion_uses_vr_reference:
                current_time = time.time()
                if self.port._vr_reference is None:
                    data_age = float("inf")
                else:
                    data_age = self.port._vr_reference.data_age(current_time)

                if data_age > self.port.max_data_age:
                    self.port.get_logger().warn(
                        f"ZMQ latest_obs is stale: age={data_age*1000:.1f}ms > "
                        f"{self.port.max_data_age*1000:.1f}ms; switching to velocity tracking mode for safety."
                    )
                    self.switch_to_velocity_mode(reason="VR latest_obs stale")
                    return None

            if not self.state.motion_uses_vr_reference and (
                not hasattr(self.port, "n_motion_frames")
                or not hasattr(self.port, "ref_dof_pos")
            ):
                self.port.get_logger().warn(
                    "Motion data not loaded, skipping policy execution"
                )
                return None

            if (
                self.state.motion_uses_vr_reference
                and self.port._vr_reference is not None
            ):
                obs_eval = self._obs_evaluator()
                try:
                    n_fut = int(getattr(self.port, "n_fut_frames", 0))
                    if (
                        n_fut > 0
                        and getattr(obs_eval, "fk", None) is not None
                        and self.port._vr_reference.has_future_sequence(n_fut)
                    ):
                        t_fk = time.perf_counter()
                        cur_root_pos = obs_eval.ref_root_pos_raw.astype(np.float32)
                        cur_root_rot = self.port._vr_reference.current_root_rot()
                        if cur_root_rot is None:
                            obs_eval.clear_vr_fk_cache()
                            return None
                        if hasattr(obs_eval, "_compute_and_cache_vr_root_fk"):
                            obs_eval._compute_and_cache_vr_root_fk(
                                vr_reference=self.port._vr_reference,
                                cur_root_pos=cur_root_pos,
                                cur_root_rot=cur_root_rot,
                                n_fut=n_fut,
                                fps=float(1.0 / self.port.dt),
                            )
                        else:
                            cur_dof_pos = obs_eval.ref_dof_pos_raw.astype(np.float32)
                            root_pos_tensor, root_rot_tensor, dof_pos_tensor = (
                                obs_eval._prepare_vr_fk_tensors(
                                    vr_reference=self.port._vr_reference,
                                    cur_root_pos=cur_root_pos,
                                    cur_root_rot=cur_root_rot,
                                    cur_dof_pos=cur_dof_pos,
                                    n_fut=n_fut,
                                )
                            )
                            fk_out = obs_eval.fk(
                                root_pos=root_pos_tensor,
                                root_quat=root_rot_tensor,
                                dof_pos=dof_pos_tensor,
                                fps=float(1.0 / self.port.dt),
                                quat_format="wxyz",
                                vel_smoothing_sigma=0.0,
                                compute_velocity=False,
                            )
                            obs_eval._fk_vr_out = {
                                k: v.detach().cpu().numpy() for k, v in fk_out.items()
                            }
                        timing_info["fk_ms"] = self.port._timing_ms(t_fk)
                    else:
                        obs_eval.clear_vr_fk_cache()
                except Exception as exc:
                    self.port.get_logger().error(
                        f"VR FK computation failed; falling back to offline reference: {exc}"
                    )
                    obs_eval.clear_vr_fk_cache()

            self.port.obs_builder = self.port.motion_obs_builder
            current_action_scale = self.port.motion_action_scale_onnx
            current_default_angles = self.port.motion_default_angles_onnx
            current_onnx_to_real = self.port.motion_onnx_to_real
        else:
            self.port.obs_builder = self.port.velocity_obs_builder
            current_action_scale = self.port.velocity_action_scale_onnx
            current_default_angles = self.port.velocity_default_angles_onnx
            current_onnx_to_real = self.port.velocity_onnx_to_real

        t_obs = time.perf_counter()
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "cache_lowstate"):
            obs_eval.cache_lowstate(self.port._lowstate_msg, force=True)
        if self.state.current_policy_mode == "motion":
            if not getattr(obs_eval, "_fk_vr_cache_ready", False):
                obs_eval._cache_fk_vr_for_obs()
        policy_obs_base = self.port.obs_builder.build_policy_obs()
        if hasattr(self.port.obs_builder, "batch_view"):
            policy_obs_np = self.port.obs_builder.batch_view()
        else:
            policy_obs_np = policy_obs_base[None, :].astype(np.float32, copy=False)
        timing_info["obs_ms"] = self.port._timing_ms(t_obs)

        t_onnx = time.perf_counter()
        onnx_output = self._run_current_policy(policy_obs_np)
        timing_info["onnx_ms"] = self.port._timing_ms(t_onnx)

        t_post = time.perf_counter()
        raw_actions_onnx = np.asarray(onnx_output[0], dtype=np.float32).reshape(-1)
        self.apply_policy_output(
            raw_actions_onnx,
            action_scale=current_action_scale,
            default_angles=current_default_angles,
            onnx_to_real=current_onnx_to_real,
            is_motion=self.state.current_policy_mode == "motion",
        )
        self.publish_current_action()
        self._mark_offline_motion_complete()
        self.port._publish_policy_mode()
        timing_info["post_ms"] = self.port._timing_ms(t_post)
        timing_info["policy_total_ms"] = self.port._timing_ms(t_policy_start)
        return timing_info

    def apply_policy_output(
        self,
        raw_actions_onnx: np.ndarray,
        *,
        action_scale: np.ndarray,
        default_angles: np.ndarray,
        onnx_to_real: Any,
        is_motion: bool,
    ) -> np.ndarray:
        if is_motion:
            if self.port.motion_action_ema_filter_enabled:
                actions = self.apply_motion_action_ema_filter(raw_actions_onnx)
            else:
                actions = np.asarray(raw_actions_onnx, dtype=np.float32).reshape(-1)
            np.copyto(self.state.actions_onnx, actions)
        else:
            actions = np.asarray(raw_actions_onnx, dtype=np.float32).reshape(-1)
            np.copyto(self.state.actions_onnx, actions)

        np.multiply(
            self.state.actions_onnx,
            action_scale,
            out=self.state.target_dof_pos_onnx,
        )
        np.add(
            self.state.target_dof_pos_onnx,
            default_angles,
            out=self.state.target_dof_pos_onnx,
        )
        self.state.target_dof_pos_real_buffer_idx = (
            1 - self.state.target_dof_pos_real_buffer_idx
        )
        target_real = self.state.target_dof_pos_real_buffers[
            self.state.target_dof_pos_real_buffer_idx
        ]
        np.take(self.state.target_dof_pos_onnx, onnx_to_real, out=target_real)
        self.state.target_dof_pos_real_buffer = target_real
        self.state.target_dof_pos_real = target_real
        return self.state.target_dof_pos_real

    def publish_current_action(self) -> None:
        self.port._publish_action_target(self.state.target_dof_pos_real)
        self.state.motion_frame_idx += 1

    def _run_current_policy(self, policy_obs_np: np.ndarray):
        if self.state.current_policy_mode == "velocity":
            if self._velocity_input_feed is None:
                self._velocity_input_feed = {self.port.velocity_input_name: policy_obs_np}
            else:
                self._velocity_input_feed[self.port.velocity_input_name] = policy_obs_np
            if self._velocity_output_names is None:
                self._velocity_output_names = [self.port.velocity_output_name]
            return self.port.velocity_policy_session.run(
                self._velocity_output_names,
                self._velocity_input_feed,
            )

        if self.port.use_kv_cache:
            if self.port.motion_kv_cache is None:
                shape = [
                    dim if isinstance(dim, int) else 1
                    for dim in self.port.motion_kv_shape
                ]
                self.port.motion_kv_cache = np.zeros(
                    shape, dtype=self.port.motion_kv_dtype
                )

            if self._motion_input_feed is None:
                self._motion_input_feed = {
                    self.port.motion_input_name: policy_obs_np,
                    self.port.motion_kv_input_name: self.port.motion_kv_cache,
                }
            else:
                self._motion_input_feed[self.port.motion_input_name] = policy_obs_np
                self._motion_input_feed[self.port.motion_kv_input_name] = (
                    self.port.motion_kv_cache
                )
            if self.port.motion_step_idx_input_name is not None:
                if self._motion_step_idx_array is None:
                    self._motion_step_idx_array = np.zeros(1, dtype=np.int64)
                self._motion_step_idx_array[0] = self.state.motion_step_idx
                self._motion_input_feed[self.port.motion_step_idx_input_name] = (
                    self._motion_step_idx_array
                )

            if self._motion_output_names is None:
                self._motion_output_names = [self.port.motion_output_name]
                if self.port.motion_kv_output_name:
                    self._motion_output_names.append(self.port.motion_kv_output_name)
            onnx_output = self.port.motion_policy_session.run(
                self._motion_output_names,
                self._motion_input_feed,
            )
            if len(onnx_output) > 1:
                self.port.motion_kv_cache = onnx_output[1]
            self.state.motion_step_idx += 1
            return onnx_output

        if self._motion_input_feed is None:
            self._motion_input_feed = {self.port.motion_input_name: policy_obs_np}
        else:
            self._motion_input_feed[self.port.motion_input_name] = policy_obs_np
        if self._motion_output_names is None:
            self._motion_output_names = [self.port.motion_output_name]
        return self.port.motion_policy_session.run(
            self._motion_output_names,
            self._motion_input_feed,
        )

    def _is_button_pressed(self, button_key: int) -> bool:
        current_state = self.port.remote_controller.button[button_key]
        last_state = self.state.last_button_states[button_key]
        self.state.last_button_states[button_key] = current_state
        return current_state == 1 and last_state == 0

    def _handle_motion_clip_selection(self) -> None:
        if self._is_button_pressed(KeyMap.up):
            self._select_motion_clip("previous", "Selected previous motion clip")
        elif self._is_button_pressed(KeyMap.down):
            self._select_motion_clip("next", "Selected next motion clip")
        elif self._is_button_pressed(KeyMap.left):
            self._select_motion_clip("first", "Selected first motion clip")
        elif self._is_button_pressed(KeyMap.right):
            self._select_motion_clip("last", "Selected last motion clip")

    def _select_motion_clip(self, command: str, log_prefix: str) -> None:
        if not getattr(self.port, "all_motion_data", None):
            return
        self.state.current_motion_clip_index = select_motion_clip_index(
            self.state.current_motion_clip_index,
            len(self.port.all_motion_data),
            command,
        )
        self.port.get_logger().info(
            f"{log_prefix}: "
            f"{self.port.motion_file_names[self.state.current_motion_clip_index]}"
        )

    def _vr_reference_available(self) -> bool:
        return bool(
            self.port.enable_teleop_reference
            and self.port._vr_reference is not None
            and self.port._vr_reference.has_latest_obs
        )

    def _consume_ros_latest_obs(self) -> None:
        buf = getattr(self.port, "_ros_latest_obs_buffer", None)
        if buf is None:
            return
        self.port._ros_latest_obs_buffer = None
        frame_idx, obs_arr = buf
        if frame_idx is not None:
            self.port._npz_replay_frame_index = frame_idx
        self.port._store_vr_latest_obs(obs_arr[None, :])

    def _log_vr_status(self, now: float) -> None:
        if self.state.current_policy_mode != "motion":
            return
        if self.state.last_vr_status_log_time is None:
            self.state.last_vr_status_log_time = now
            return
        if now - self.state.last_vr_status_log_time < 5.0:
            return

        vr_available = bool(
            self.port._vr_reference is not None and self.port._vr_reference.has_latest_obs
        )
        queue_stats = self.port._latest_obs_buffer.get_queue_stats()
        if vr_available:
            freq = queue_stats.get("expected_freq")
            if freq:
                details = (
                    f"buffer_size={queue_stats['queue_size']} "
                    f"expected_freq={freq:.1f}Hz"
                )
            else:
                details = (
                    f"buffer_size={queue_stats['queue_size']} "
                    "expected_freq=unknown"
                )
            self.port.get_logger().info(
                "[VR-STATUS] ZMQ latest_obs streaming | " + details
            )
        else:
            self.port.get_logger().warn(
                "[VR-STATUS] No new ZMQ latest_obs received in the last 5 seconds; "
                "using offline reference or the last buffered VR state."
            )
        self.state.last_vr_status_log_time = now

    def _log_vr_ready(self, now: float) -> None:
        del now
        if (
            not self.port.enable_teleop_reference
            or not self.state.policy_enabled
            or self.state.vr_queue_ready_logged
            or not self.port._is_vr_ready_for_motion()
        ):
            return
        seen_frames = (
            self.port._vr_reference.seen_frames
            if self.port._vr_reference is not None
            else 0
        )
        self.port.get_logger().info(
            f"[VR] VR queue is ready for motion mode (seen_frames={int(seen_frames)}, "
            f"n_fut={int(getattr(self.port, 'n_fut_frames', 0) or 0)}, "
            f"delay={int(getattr(self.port, 'zmq_jitter_delay_frames', 0) or 0)})"
        )
        self.state.vr_queue_ready_logged = True

    def _log_policy_latency(self, run_elapsed: float) -> None:
        if not (
            self.state.current_policy_mode == "motion"
            and self.state.motion_uses_vr_reference
        ):
            return
        if run_elapsed > 0.5 and not self.state.vr_cold_start_logged:
            self.state.vr_cold_start_logged = True
            self.port.get_logger().info(
                "[VR] The first motion step is a cold start (FK/ONNX initialization) and may take about 1 second."
            )
        if run_elapsed > 1.15 * self.port.dt and run_elapsed <= 0.5:
            self.state.policy_slow_count += 1
            if (
                self.state.policy_slow_count == 1
                or self.state.policy_slow_count % 50 == 0
            ):
                self.port.get_logger().warn(
                    f"[VR] Policy step latency {run_elapsed*1000:.1f} ms exceeds the target "
                    f"{self.port.dt*1000:.1f} ms. Estimated /humanoid/action rate: "
                    f"{1.0/run_elapsed:.1f} Hz (target {1.0/self.port.dt:.0f} Hz). "
                    "The main bottleneck is usually FK or ONNX inference; if the system settles near 30 Hz, consider setting policy_freq to 30."
                )

    def _mark_offline_motion_complete(self) -> None:
        if self.state.current_policy_mode != "motion":
            return
        if (
            not self.state.motion_uses_vr_reference
            and self.state.motion_frame_idx >= self.port.n_motion_frames
            and self.state.motion_in_progress
        ):
            self.switch_to_velocity_mode(reason="offline motion completed")
