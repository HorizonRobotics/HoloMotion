#! /your_dir/miniconda3/envs/holomotion_deploy/bin/python
"""
HoloMotion Policy Node

This module implements the main policy execution node for the HoloMotion humanoid robot system using ZMQ latest_obs transport.
It handles neural network policy inference, motion sequence management, remote controller input,
and robot state coordination for humanoid behaviors including velocity tracking and motion tracking.

The policy node serves as the high-level decision maker that:
- Processes sensor observations and builds state representations
- Executes trained neural network policies for motion generation (velocity tracking and motion tracking)
- Manages multiple motion sequences (motion clips) loaded from offline files
- Handles remote controller input for motion selection
- Coordinates with the main control node for safe operation

Key Features:
- Dual policy support: velocity tracking and motion tracking
- Offline motion file loading (.npz format)
- Runtime policy switching with button controls
- Separate hyperparameters (kps, kds, action_scale, default_angles) for each model

Author: HoloMotion Team
License: See project LICENSE file
"""
import os
import time
from collections import deque

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState

from humanoid_policy.config import DeploymentConfig
from humanoid_policy.config import RobotConfig
from humanoid_policy.config import format_config_for_log
from humanoid_policy.cpu_affinity import parse_cpu_affinity
from humanoid_policy.cpu_affinity import set_thread_cpu_affinity
from humanoid_policy.latest_obs_transport import DEFAULT_ZMQ_TOPIC
from humanoid_policy.latest_obs_transport import LatestObsBuffer
from humanoid_policy.latest_obs_transport import ZmqLatestObsSubscriber
from humanoid_policy.latest_obs_transport import decode_zmq_topic
from humanoid_policy.motion_clip_library import list_motion_clip_files
from humanoid_policy.motion_clip_library import load_motion_clips
from humanoid_policy.motion_clip_library import validate_loaded_motion_clip
from humanoid_policy.onnx_policy import load_dual_policy_bundle
from humanoid_policy.onnx_policy import read_onnx_metadata
from humanoid_policy.onnx_policy import warmup_motion_policy
from humanoid_policy.onnx_policy import warmup_policy_session
from humanoid_policy.obs_builder import PolicyObsBuilder
from humanoid_policy.observation_evaluator import PolicyObservationEvaluator
from humanoid_policy.policy_runtime import PolicyRuntime
from humanoid_policy.vr_reference import VrLatestObsReference
from humanoid_policy.utils.remote_controller_filter import RemoteController


def _coerce_config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


class HoloMotionPolicyNode(Node):
    """Main policy execution node for HoloMotion humanoid robot control with dual policy support.

    This node implements the high-level control logic for a humanoid robot capable of
    performing both velocity tracking and motion sequence execution. It supports two
    neural network policies and allows runtime switching between them.

    Key Features:
    - Dual neural network policy inference (velocity + motion) using ONNX Runtime
    - Runtime policy switching with A/B/Y button controls
    - Velocity tracking mode with joystick control
    - Motion tracking mode with motion clip sequence selection
    - Safety-aware state machine with motion prerequisites
    - Real-time observation processing and action generation

    Policy Control:
    - A button: Enable policy (defaults to velocity mode)
    - B button: Switch from velocity to motion mode
    - Y button: Switch from motion back to velocity mode

    Input Controls:
    - Motion mode:  B button (for mode switch)
    - Velocity mode: Y button (for mode switch) + Joystick +UP/DOWN/LEFT/RIGHT (for motion selection)

    State Machine:
    - ZERO_TORQUE: Initial safe state, waiting for activation
    - MOVE_TO_DEFAULT: Ready state, allows policy operations
    - Policy execution with mode switching
    - Emergency stop handling
    """

    def __init__(self):
        """Initialize the policy node with configuration, models, and ROS2 interfaces.

        Sets up the complete policy execution pipeline including:
        - Configuration loading from YAML file
        - Neural network model initialization
        - Motion data loading for all sequences
        - ROS2 publishers, subscribers, and timers
        - State machine initialization

        The node starts in a safe state and waits for proper robot state
        before allowing motion execution.
        """
        super().__init__("policy_node")
        self.runtime = PolicyRuntime(self, num_actions=29)
        self.observation_evaluator = PolicyObservationEvaluator(self)

        self.deployment_config = DeploymentConfig.from_node(self)
        self.robot_config = RobotConfig.load(self.deployment_config.robot_config_path)
        self.get_logger().info(
            "Deployment config:\n"
            + format_config_for_log(self.deployment_config.to_log_dict())
        )
        self.get_logger().info(
            "Robot config:\n" + format_config_for_log(self.robot_config.to_log_dict())
        )
        policy_freq = self.robot_config.policy_freq
        self.dt = 1.0 / policy_freq
        self.get_logger().info(f"Policy frequency set to: {policy_freq} Hz (dt = {self.dt:.4f} s)")
        # Initialize basic parameters - will be updated after config loading
        self.actions_dim = 29  # Default value, will be updated from config
        self.real_dof_names = []  # Will be loaded from config
        self.current_motion_clip_index = 0  # Current motion clip index
        # Safety check related flags
        self.policy_enabled = False  # Controls whether policy is enabled
        # Robot state related flags
        self.robot_state_ready = False  # Marks whether MOVE_TO_DEFAULT state is received, allowing key operations
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_timers()
        # Initialize variables for dual policy
        self.velocity_policy_session = None
        self.motion_policy_session = None
        self.use_kv_cache = False
        self.motion_kv_cache = None
        self.motion_kv_input_name = None
        self.motion_kv_output_name = None
        self.motion_step_idx_input_name = None
        self.current_policy_mode = "velocity"
        self.velocity_config = None
        self.motion_config = None
        self.motion_frame_idx = 0
        self.ref_dof_pos = None
        self.ref_dof_vel = None
        self.ref_raw_bodylink_pos = None
        self.ref_raw_bodylink_rot = None
        self.n_motion_frames = 0

        self._latest_sender_timestamp = None
        self.latest_obs_flag = False
        self.latest_obs_expected_dim = 65
        self.max_data_age = self.deployment_config.max_data_age
        self.stale_data_warning_count = 0
        self.last_poll_time = None
        self.latest_obs_zmq_uri = self.deployment_config.latest_obs_zmq_uri
        self.latest_obs_zmq_topic = self.deployment_config.latest_obs_zmq_topic
        self.latest_obs_zmq_mode = self.deployment_config.latest_obs_zmq_mode
        self.latest_obs_zmq_conflate = self.deployment_config.latest_obs_zmq_conflate
        self.zmq_jitter_delay_frames = self.deployment_config.zmq_jitter_delay_frames
        self.enable_teleop_reference = self.deployment_config.enable_teleop_reference
        self.motion_rope_max_seq_len = self.deployment_config.motion_rope_max_seq_len
        self.motion_rope_reset_margin = self.deployment_config.motion_rope_reset_margin
        self._cpu_affinity_main_str = self.deployment_config.cpu_affinity_main
        self._cpu_affinity_zmq_sub_str = self.deployment_config.cpu_affinity_zmq_sub
        self.timing_debug_enabled = self.deployment_config.timing_debug_enabled
        self.timing_debug_log_interval_sec = (
            self.deployment_config.timing_debug_log_interval_sec
        )
        self.timing_debug_log_per_loop = self.deployment_config.timing_debug_log_per_loop
        self._timing_debug_last_log_time = None
        self._timing_debug_samples = deque(maxlen=500)
        self._root_only_fk_keybody_warned = False
        self._ros_latest_obs_buffer = None
        self._npz_replay_frame_index = None

        self._latest_obs_buffer = LatestObsBuffer()
        self._latest_obs_zmq_topic_bytes = decode_zmq_topic(self.latest_obs_zmq_topic)
        if str(self.latest_obs_zmq_mode).strip().lower() == "connect":
            uri_str = str(self.latest_obs_zmq_uri)
            if "*" in uri_str or "0.0.0.0" in uri_str:
                self.get_logger().warn(
                    "[ZMQ] connect mode requires a concrete peer address. "
                    "Do not use '*' or '0.0.0.0'; use the sender IP instead, "
                    "for example tcp://192.168.124.29:6001."
                )
        zmq_cpu_affinity = parse_cpu_affinity(self._cpu_affinity_zmq_sub_str)
        self._zmq_subscriber = ZmqLatestObsSubscriber(
            uri=self.latest_obs_zmq_uri,
            topic=self._latest_obs_zmq_topic_bytes,
            mode=self.latest_obs_zmq_mode,
            conflate=bool(self.latest_obs_zmq_conflate),
            buffer=self._latest_obs_buffer,
            logger=self.get_logger(),
            cpu_affinity=zmq_cpu_affinity if zmq_cpu_affinity else None,
        )
        self._zmq_subscriber.start()
        self.get_logger().info(
            f"ZMQ latest_obs subscriber started: mode={self.latest_obs_zmq_mode}, "
            f"uri={self.latest_obs_zmq_uri}, topic={self.latest_obs_zmq_topic}, "
            f"jitter_delay={self.zmq_jitter_delay_frames}"
        )

        self.dof_names_ref_motion = []
        self.num_actions = 29
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)

        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.runtime.resize_actions(self.num_actions)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self._lowstate_msg = None
        self.target_dof_pos_real = None
        self.motion_in_progress = False
        self._keybody_indices_by_term_name = {}
        self.motion_action_ema_filter_enabled = False
        self.motion_action_ema_filter_alpha = 1.0
        self._vr_reference = None

    def _is_vr_ready_for_motion(self) -> bool:
        """Return whether the ZMQ reference stream is ready for motion mode."""
        if self._vr_reference is None:
            return False
        return self._vr_reference.is_ready_for_motion(
            enable_teleop_reference=getattr(self, "enable_teleop_reference", True),
            delay_frames=getattr(self, "zmq_jitter_delay_frames", 0),
        )

    def _runtime_state(self):
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return None
        return runtime.state

    @property
    def policy_enabled(self):
        state = self._runtime_state()
        return state.policy_enabled if state is not None else getattr(self, "_policy_enabled", False)

    @policy_enabled.setter
    def policy_enabled(self, value):
        state = self._runtime_state()
        if state is None:
            self._policy_enabled = bool(value)
        else:
            state.policy_enabled = bool(value)

    @property
    def robot_state_ready(self):
        state = self._runtime_state()
        return state.robot_state_ready if state is not None else getattr(self, "_robot_state_ready", False)

    @robot_state_ready.setter
    def robot_state_ready(self, value):
        state = self._runtime_state()
        if state is None:
            self._robot_state_ready = bool(value)
        else:
            state.robot_state_ready = bool(value)

    @property
    def current_policy_mode(self):
        state = self._runtime_state()
        return state.current_policy_mode if state is not None else getattr(self, "_current_policy_mode", "velocity")

    @current_policy_mode.setter
    def current_policy_mode(self, value):
        state = self._runtime_state()
        if state is None:
            self._current_policy_mode = str(value)
        else:
            state.current_policy_mode = str(value)

    @property
    def latest_obs_flag(self):
        state = self._runtime_state()
        return state.latest_obs_flag if state is not None else getattr(self, "_latest_obs_flag", False)

    @latest_obs_flag.setter
    def latest_obs_flag(self, value):
        state = self._runtime_state()
        if state is None:
            self._latest_obs_flag = bool(value)
        else:
            state.latest_obs_flag = bool(value)

    @property
    def motion_frame_idx(self):
        state = self._runtime_state()
        return state.motion_frame_idx if state is not None else getattr(self, "_motion_frame_idx", 0)

    @motion_frame_idx.setter
    def motion_frame_idx(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_frame_idx = int(value)
        else:
            state.motion_frame_idx = int(value)

    @property
    def motion_step_idx(self):
        state = self._runtime_state()
        return state.motion_step_idx if state is not None else getattr(self, "_motion_step_idx", 0)

    @motion_step_idx.setter
    def motion_step_idx(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_step_idx = int(value)
        else:
            state.motion_step_idx = int(value)

    @property
    def motion_in_progress(self):
        state = self._runtime_state()
        return state.motion_in_progress if state is not None else getattr(self, "_motion_in_progress", False)

    @motion_in_progress.setter
    def motion_in_progress(self, value):
        state = self._runtime_state()
        if state is None:
            self._motion_in_progress = bool(value)
        else:
            state.motion_in_progress = bool(value)

    @property
    def current_motion_clip_index(self):
        state = self._runtime_state()
        return state.current_motion_clip_index if state is not None else getattr(self, "_current_motion_clip_index", 0)

    @current_motion_clip_index.setter
    def current_motion_clip_index(self, value):
        state = self._runtime_state()
        if state is None:
            self._current_motion_clip_index = int(value)
        else:
            state.current_motion_clip_index = int(value)

    @property
    def vx(self):
        state = self._runtime_state()
        return state.vx if state is not None else getattr(self, "_vx", 0.0)

    @vx.setter
    def vx(self, value):
        state = self._runtime_state()
        if state is None:
            self._vx = float(value)
        else:
            state.vx = float(value)

    @property
    def vy(self):
        state = self._runtime_state()
        return state.vy if state is not None else getattr(self, "_vy", 0.0)

    @vy.setter
    def vy(self, value):
        state = self._runtime_state()
        if state is None:
            self._vy = float(value)
        else:
            state.vy = float(value)

    @property
    def vyaw(self):
        state = self._runtime_state()
        return state.vyaw if state is not None else getattr(self, "_vyaw", 0.0)

    @vyaw.setter
    def vyaw(self, value):
        state = self._runtime_state()
        if state is None:
            self._vyaw = float(value)
        else:
            state.vyaw = float(value)

    @property
    def actions_onnx(self):
        state = self._runtime_state()
        return state.actions_onnx if state is not None else getattr(self, "_actions_onnx", None)

    @actions_onnx.setter
    def actions_onnx(self, value):
        state = self._runtime_state()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if state is None:
            self._actions_onnx = arr.copy()
        else:
            state.actions_onnx = arr.copy()

    @property
    def target_dof_pos_onnx(self):
        state = self._runtime_state()
        return state.target_dof_pos_onnx if state is not None else getattr(self, "_target_dof_pos_onnx", None)

    @target_dof_pos_onnx.setter
    def target_dof_pos_onnx(self, value):
        state = self._runtime_state()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if state is None:
            self._target_dof_pos_onnx = arr.copy()
        else:
            state.target_dof_pos_onnx = arr.copy()

    @property
    def target_dof_pos_real(self):
        state = self._runtime_state()
        return state.target_dof_pos_real if state is not None else getattr(self, "_target_dof_pos_real", None)

    @target_dof_pos_real.setter
    def target_dof_pos_real(self, value):
        state = self._runtime_state()
        arr = None if value is None else np.asarray(value, dtype=np.float32).reshape(-1)
        if state is None:
            self._target_dof_pos_real = None if arr is None else arr.copy()
        else:
            state.target_dof_pos_real = None if arr is None else arr.copy()


    def _init_obs_buffers(self):
        """Initialize observation builders for both velocity and motion policies.

        Each obs_builder uses its own model's dof_names_onnx and default_angles_onnx
        to ensure correct observation computation for each policy.
        """
        # Use velocity model's parameters for velocity obs_builder
        self.velocity_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.velocity_dof_names_onnx,
            default_angles_onnx=self.velocity_default_angles_onnx,
            evaluator=self.observation_evaluator,
            obs_policy_cfg=self.observation_evaluator._get_policy_atomic_obs_list(
                self.velocity_config
            ),
        )

        # Use motion model's parameters for motion obs_builder
        self.motion_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.motion_dof_names_onnx,
            default_angles_onnx=self.motion_default_angles_onnx,
            evaluator=self.observation_evaluator,
            obs_policy_cfg=self.observation_evaluator._get_policy_atomic_obs_list(
                self.motion_config
            ),
        )

        n_fut = int(self.n_fut_frames) if hasattr(self, "n_fut_frames") else 0
        self._vr_reference = VrLatestObsReference(
            n_fut_frames=n_fut,
            num_actions=self.num_actions,
            expected_dim=self.latest_obs_expected_dim,
        )
        self.observation_evaluator.initialize_vr_fk_buffers(n_fut, self.num_actions)

        # Set default obs_builder to velocity mode
        self.obs_builder = self.velocity_obs_builder

    def load_policy(self):
        """Load both velocity and motion policy models using ONNX Runtime."""
        self.get_logger().info(
            "Loading dual policies "
            f"(backend={self.deployment_config.inference_backend})..."
        )
        motion_max_context_len = int(
            self.motion_config.get("algo", {})
            .get("config", {})
            .get("num_steps_per_env", 0)
        )
        bundle = load_dual_policy_bundle(
            package_share_dir=get_package_share_directory("humanoid_control"),
            velocity_model_folder=self.robot_config.velocity_tracking_model_folder,
            motion_model_folder=self.robot_config.motion_tracking_model_folder,
            intra_op_threads=self.robot_config.onnx_intra_op_threads,
            motion_max_context_len=motion_max_context_len,
            inference_backend=self.deployment_config.inference_backend,
        )

        self.velocity_policy_session = bundle.velocity_session
        self.motion_policy_session = bundle.motion_session
        self.velocity_onnx_path = bundle.velocity_onnx_path
        self.motion_onnx_path = bundle.motion_onnx_path
        self.velocity_input_name = bundle.velocity_input_name
        self.velocity_output_name = bundle.velocity_output_name
        self.get_logger().info(
            f"Velocity policy loaded from {self.velocity_onnx_path} using: "
            f"{self.velocity_policy_session.get_providers()}"
        )
        self.get_logger().info(
            f"Motion policy loaded from {self.motion_onnx_path} using: "
            f"{self.motion_policy_session.get_providers()}"
        )

        motion_io = bundle.motion_io
        self.motion_input_name = motion_io.input_name
        self.motion_output_name = motion_io.output_name
        self.motion_kv_input_name = motion_io.kv_input_name
        self.motion_kv_output_name = motion_io.kv_output_name
        self.motion_kv_shape = motion_io.kv_shape
        self.motion_step_idx_input_name = motion_io.step_idx_input_name
        self.motion_kv_dtype = motion_io.kv_dtype

        self.get_logger().info(
            f"Velocity policy - Input: {self.velocity_input_name}, "
            f"Output: {self.velocity_output_name}"
        )
        self.get_logger().info(
            f"Motion policy - Input: {self.motion_input_name}, "
            f"Output: {self.motion_output_name}"
        )
        self.get_logger().info("Initializing KV-Cache for Motion Policy...")

        for node in self.motion_policy_session.get_inputs():
            self.get_logger().info(
                f"Motion policy input: name={node.name}, shape={node.shape}, type={node.type}"
            )
        for node in self.motion_policy_session.get_outputs():
            self.get_logger().info(
                f"Motion policy output: name={node.name}, shape={node.shape}, type={node.type}"
            )
        if self.motion_kv_input_name is not None and self.motion_kv_output_name is None:
            self.get_logger().warn(
                "Motion policy has past_key_values input but no present_key_values output was found. "
                "KV cache will not update and transformer performance will degrade."
            )

        kv_cache = bundle.motion_kv_cache
        self.motion_kv_cache = kv_cache.cache
        self.motion_max_context_len = motion_max_context_len
        self.motion_model_context_len = kv_cache.model_context_len
        self.motion_effective_context_len = kv_cache.effective_context_len
        self.use_kv_cache = kv_cache.enabled
        if self.use_kv_cache:
            self.get_logger().info(
                f"KV-Cache initialized with shape {kv_cache.shape} "
                f"(model_ctx={self.motion_model_context_len}, "
                f"effective_ctx={self.motion_effective_context_len})"
            )
        else:
            self.get_logger().warn("No KV-Cache inputs found in Motion Policy model!")
        self.get_logger().info("Dual policies loaded successfully")

    def load_model_config(self):
        """Load config.yaml from both velocity and motion model folders."""
        # Load velocity model config
        velocity_model_folder = self.robot_config.velocity_tracking_model_folder
        velocity_config_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            velocity_model_folder,
        )
        # Try different config file names for velocity model
        config_names = ["config.yaml"]
        velocity_config_path = None

        for config_name in config_names:
            potential_path = os.path.join(velocity_config_dir, config_name)
            if os.path.exists(potential_path):
                velocity_config_path = potential_path
                break

        if velocity_config_path is None:
            raise FileNotFoundError(
                f"No config file found in {velocity_config_dir}. Tried: {config_names}"
            )

        self.get_logger().info(
            f"Loading velocity model config from {velocity_config_path}"
        )
        self.velocity_config = OmegaConf.load(velocity_config_path)

        # Load motion model config
        motion_model_folder = self.robot_config.motion_tracking_model_folder
        motion_config_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            motion_model_folder,
        )
        # Try different config file names for motion model
        motion_config_path = None

        for config_name in config_names:
            potential_path = os.path.join(motion_config_dir, config_name)
            if os.path.exists(potential_path):
                motion_config_path = potential_path
                break

        if motion_config_path is None:
            raise FileNotFoundError(
                f"No config file found in {motion_config_dir}. Tried: {config_names}"
            )

        self.get_logger().info(f"Loading motion model config from {motion_config_path}")
        self.motion_config = OmegaConf.load(motion_config_path)
        self._load_motion_action_ema_filter_cfg()
        self.actor_place_holder_ndim = (
            self.observation_evaluator._find_actor_place_holder_ndim()
        )
        self.n_fut_frames = int(self.motion_config.obs.n_fut_frames)
        self.torso_body_idx = self.motion_config.robot.body_names.index("torso_link")
        self.get_logger().info("Both model configs loaded successfully")

    def _load_motion_action_ema_filter_cfg(self) -> None:
        actuator_cfg = self.motion_config.get("robot", {}).get("actuators", {})
        enabled_raw = actuator_cfg.get("ema_filter_enabled", None)
        alpha_raw = actuator_cfg.get("ema_filter_alpha", None)

        if enabled_raw is None or alpha_raw is None:
            self.motion_action_ema_filter_enabled = False
            self.motion_action_ema_filter_alpha = 1.0
            self.get_logger().info(
                "[Motion EMA] ema_filter_enabled/ema_filter_alpha not found in motion config; EMA disabled."
            )
            return

        self.motion_action_ema_filter_enabled = _coerce_config_bool(
            enabled_raw, default=False
        )
        self.motion_action_ema_filter_alpha = float(alpha_raw)
        if not 0.0 <= self.motion_action_ema_filter_alpha <= 1.0:
            raise ValueError(
                "motion_config robot.actuators.ema_filter_alpha must be within [0, 1], "
                f"got {self.motion_action_ema_filter_alpha}."
            )
        self.get_logger().info(
            "[Motion EMA] Loaded from motion config: "
            f"enabled={self.motion_action_ema_filter_enabled}, "
            f"alpha={self.motion_action_ema_filter_alpha:.4f}"
        )

    def _warmup_motion_policy(self, num_iters: int = 2) -> None:
        if self.velocity_policy_session is not None:
            velocity_obs_dim = None
            try:
                velocity_obs_dim = int(
                    self.velocity_obs_builder.build_policy_obs().shape[0]
                )
            except Exception:
                velocity_obs_dim = None
            try:
                velocity_iterations = warmup_policy_session(
                    session=self.velocity_policy_session,
                    input_name=self.velocity_input_name,
                    output_name=self.velocity_output_name,
                    obs_dim=velocity_obs_dim,
                    num_iters=num_iters,
                )
                self.get_logger().info(
                    f"[Warmup] Velocity policy warmup completed ({velocity_iterations} iterations)."
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"[Warmup] Velocity policy warmup skipped: {exc}"
                )

        if self.motion_policy_session is None:
            return

        motion_obs_dim = None
        try:
            motion_obs_dim = int(self.motion_obs_builder.build_policy_obs().shape[0])
        except Exception:
            motion_obs_dim = None

        try:
            iterations = warmup_motion_policy(
                session=self.motion_policy_session,
                input_name=self.motion_input_name,
                output_name=self.motion_output_name,
                kv_input_name=self.motion_kv_input_name,
                kv_output_name=self.motion_kv_output_name,
                step_idx_input_name=self.motion_step_idx_input_name,
                use_kv_cache=self.use_kv_cache,
                kv_cache=self.motion_kv_cache,
                kv_shape=self.motion_kv_shape,
                kv_dtype=self.motion_kv_dtype,
                obs_dim=motion_obs_dim,
                num_iters=num_iters,
            )
            if self.motion_kv_cache is not None:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            self.get_logger().info(
                f"[Warmup] Motion policy warmup completed ({iterations} iterations, KV cache kept clean)."
            )
        except Exception as exc:
            if self.motion_kv_cache is not None:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            self.get_logger().warn(f"[Warmup] Motion policy warmup skipped: {exc}")

    def update_config_parameters(self):
        """Update configuration parameters from loaded configs."""
        # Check if both models have the same basic parameters
        velocity_actions_dim = self.velocity_config.get("robot", {}).get("actions_dim", 29)
        motion_actions_dim = self.motion_config.get("robot", {}).get("actions_dim", 29)

        velocity_dof_names = self.velocity_config.get("robot", {}).get("dof_names", [])
        motion_dof_names = self.motion_config.get("robot", {}).get("dof_names", [])

        # Verify that both models have compatible configurations
        if velocity_actions_dim != motion_actions_dim:
            self.get_logger().warn(
                f"Different actions_dim: velocity={velocity_actions_dim}, "
                f"motion={motion_actions_dim}"
            )

        if velocity_dof_names != motion_dof_names:
            self.get_logger().warn(f"Different dof_names between models")
            self.get_logger().warn(f"Velocity dof_names: {len(velocity_dof_names)} items")
            self.get_logger().warn(f"Motion dof_names: {len(motion_dof_names)} items")

        # Use velocity config as the primary source for basic parameters
        config = self.velocity_config
        # Update basic parameters
        self.actions_dim = config.get("robot", {}).get("actions_dim", 29)
        self.real_dof_names = config.get("robot", {}).get("dof_names", [])
        self.dof_names_ref_motion = list(config.robot.dof_names)
        self.num_actions = len(self.dof_names_ref_motion)

        # Update arrays with correct sizes
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)
        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self.get_logger().info(
            f"Updated config parameters: actions_dim={self.actions_dim}, "
            f"dof_names={len(self.real_dof_names)}"
        )

    def load_motion_data(self):
        """Load motion clip data from .npz files."""
        motion_clips_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            self.robot_config.motion_clip_dir,
        )

        self.get_logger().info(f"Looking for motion clip data in: {motion_clips_dir}")
        self.get_logger().info(f"Directory exists: {os.path.exists(motion_clips_dir)}")

        if not os.path.exists(motion_clips_dir):
            self.get_logger().warn(f"Motion clips directory not found: {motion_clips_dir}")
            return

        # Only collect .npz files
        motion_clip_files = list_motion_clip_files(motion_clips_dir)
        self.get_logger().info(
            f"Found {len(motion_clip_files)} motion clip files (.npz): {motion_clip_files}"
        )
        if not motion_clip_files:
            self.get_logger().warn(
                f"No motion clip files (.npz) found in directory: {motion_clips_dir}"
            )
            return

        # Load each .npz file
        self.all_motion_data = load_motion_clips(motion_clips_dir, motion_clip_files)
        self.motion_file_names = list(motion_clip_files)

        if not self.all_motion_data:
            self.get_logger().error("Failed to load any motion clip files")
            return

        # Initialize with the first motion clip
        self.current_motion_clip_index = 0
        self._load_current_motion()

        self.get_logger().info(f"Loaded {len(self.all_motion_data)} motion clips successfully")
        self.get_logger().info(
            f"Current motion clip: {self.motion_file_names[self.current_motion_clip_index]}"
        )

    def _load_current_motion(self):
        """Load the current selected motion clip data."""
        if not self.all_motion_data:
            return

        self.motion_frame_idx = 0
        current_motion = validate_loaded_motion_clip(
            self.all_motion_data[self.current_motion_clip_index],
            expected_dof_count=len(self.dof_names_ref_motion),
            expected_body_count=len(self.motion_config.robot.body_names),
        )
        self.ref_dof_pos = current_motion.dof_pos
        self.ref_dof_vel = current_motion.dof_vel
        self.ref_raw_bodylink_pos = current_motion.global_translation
        self.ref_raw_bodylink_rot = current_motion.global_rotation_quat
        self.ref_global_velocity = current_motion.global_velocity
        self.ref_global_angular_velocity = current_motion.global_angular_velocity
        self.n_motion_frames = current_motion.n_frames
        self.observation_evaluator._offline_reference.set_clip(current_motion)

        self.motion_in_progress = True
        self.get_logger().info(
            f"Loaded motion clip {self.current_motion_clip_index}: "
            f"{self.motion_file_names[self.current_motion_clip_index]} ({self.n_motion_frames} frames)"
        )

    def _setup_subscribers(self):
        """Set up ROS2 subscribers for robot state and remote controller input."""
        self.remote_controller = RemoteController()
        self.low_state_sub = self.create_subscription(
            LowState,
            self.robot_config.lowstate_topic,
            self._low_state_callback,
            QoSProfile(depth=10),
        )

        # Add robot_state topic subscription
        self.robot_state_sub = self.create_subscription(
            String,
            "/robot_state",
            self._robot_state_callback,
            QoSProfile(depth=10),
        )

        self.latest_obs_ros_sub = self.create_subscription(
            Float32MultiArray,
            "latest_obs_ros",
            self._latest_obs_ros_callback,
            QoSProfile(depth=10),
        )

    def _latest_obs_ros_callback(self, msg: Float32MultiArray):
        """Receive replayed latest_obs_ros messages for offline validation."""
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size == 66:
            frame_idx = int(data[0])
            obs = data[1:66]
            self._ros_latest_obs_buffer = (frame_idx, obs)
        elif data.size >= 65:
            self._ros_latest_obs_buffer = (None, data[:65])

    def _setup_publishers(self):
        """Set up ROS2 publishers for action commands and status information."""
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.robot_config.action_topic,
            QoSProfile(depth=10),
        )
        # Add publishers for kps and kds parameters
        self.kps_pub = self.create_publisher(
            Float32MultiArray,
            "/humanoid/kps",
            QoSProfile(depth=10),
        )
        self.kds_pub = self.create_publisher(
            Float32MultiArray,
            "/humanoid/kds",
            QoSProfile(depth=10),
        )
        # Add publisher for policy mode status
        self.policy_mode_pub = self.create_publisher(
            String,
            "policy_mode",
            QoSProfile(depth=10),
        )
        self.latest_obs_pub = self.create_publisher(
            Float32MultiArray,
            "latest_obs",
            QoSProfile(depth=10),
        )
        self._action_msg = Float32MultiArray()
        self._latest_obs_msg = Float32MultiArray()
        self._policy_mode_msg = String()
        self._last_policy_mode_msg_data = None
        self._last_policy_mode_publish_time = 0.0
        self._policy_mode_publish_interval_sec = 1.0

    def _setup_timers(self):
        """Set up ROS2 timer for main execution loop."""
        # Create a one-time timer to call setup after ROS2 initialization
        self.create_timer(0.1, self._delayed_setup)
        self.create_timer(self.dt, self.run)


    def _delayed_setup(self):
        """Call setup after ROS2 initialization is complete."""
        if not hasattr(self, '_setup_completed'):
            self.get_logger().info("Starting policy node setup...")
            try:
                self.setup()
                self._setup_completed = True
                self.get_logger().info("Policy node setup completed successfully")
            except Exception as e:
                self.get_logger().error(f"Policy node setup failed: {e}")
                # Cancel the timer to avoid repeated attempts
                return


    def _robot_state_callback(self, msg: String):
        """Handle robot state messages for safety coordination.

        Processes robot state updates from the main control node to ensure
        safe operation. Button operations are only allowed when the robot
        is in MOVE_TO_DEFAULT state.

        Args:
            msg: String message containing robot state information
                Valid states: ZERO_TORQUE, MOVE_TO_DEFAULT, EMERGENCY_STOP, POLICY
        """
        robot_state = msg.data
        self.runtime.set_robot_state(robot_state)

    def _low_state_callback(self, ls_msg: LowState):
        """Forward low-level state and remote-controller input to runtime."""
        self.runtime.handle_low_state(ls_msg)

    def run(self):
        """Main execution loop for policy inference and action publication."""
        self.runtime.run()

    def _store_vr_latest_obs(self, arr: np.ndarray):
        """Store latest_obs and maintain the current/future frame queues."""
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.latest_obs_expected_dim:
            self.get_logger().warn(
                f"Received latest_obs dim={arr.shape[1]}, expected >= {self.latest_obs_expected_dim}"
            )
            return
        current_time = time.time()
        raw_idx = getattr(self, "_npz_replay_frame_index", None)
        if self._vr_reference is None:
            self._vr_reference = VrLatestObsReference(
                n_fut_frames=getattr(self, "n_fut_frames", 0),
                num_actions=self.num_actions,
                expected_dim=self.latest_obs_expected_dim,
            )
        stored = self._vr_reference.store(
            arr,
            current_time=current_time,
            frame_index=raw_idx,
        )
        if not stored:
            self.get_logger().warn(
                f"Received latest_obs dim={arr.shape[1]}, expected >= {self.latest_obs_expected_dim}"
            )
            return

    def _poll_zmq_latest_obs(self):
        """Poll the ZMQ latest_obs buffer with stale-data checks and delay."""
        current_time = time.time()

        data, timestamp, is_stale, frame_index, sender_timestamp = self._latest_obs_buffer.get_with_age_and_delay(
            max_age=self.max_data_age,
            delay_steps=int(getattr(self, "zmq_jitter_delay_frames", 0)),
        )

        if data is None:
            return

        if frame_index is not None:
            self._npz_replay_frame_index = int(frame_index)
        self._latest_sender_timestamp = sender_timestamp

        if is_stale:
            self.stale_data_warning_count += 1
            if self.stale_data_warning_count % 50 == 0:
                age_ms = (current_time - timestamp) * 1000.0
                self.get_logger().warn(
                    f"ZMQ latest_obs is stale: age={age_ms:.1f}ms "
                    f"(threshold={self.max_data_age*1000:.1f}ms), "
                    f"stale_count={self.stale_data_warning_count}"
                )
                queue_stats = self._latest_obs_buffer.get_queue_stats()
                if queue_stats.get("expected_freq"):
                    self.get_logger().warn(
                        f"latest_obs buffer: size={queue_stats['queue_size']}, "
                        f"avg_interval={queue_stats['avg_interval']*1000:.1f}ms, "
                        f"expected_freq={queue_stats['expected_freq']:.1f}Hz"
                    )
        else:
            if self.stale_data_warning_count > 0:
                self.stale_data_warning_count = 0

        if self.last_poll_time is not None:
            poll_interval = current_time - self.last_poll_time
            if poll_interval > 0.03:
                self.get_logger().debug(
                    f"Policy poll interval {poll_interval*1000:.1f}ms (>30ms)"
                )
        self.last_poll_time = current_time

        self._store_vr_latest_obs(np.asarray(data, dtype=np.float32))

        if (
            getattr(self, "enable_teleop_reference", True)
            and self._is_vr_ready_for_motion()
        ):
            self.runtime.mark_vr_queue_ready()

    def _publish_latest_obs(self):
        """Publish the latest_obs topic for debugging or reuse."""
        if self._vr_reference is None or self._vr_reference.latest_obs is None:
            return
        try:
            msg = self._latest_obs_msg
            msg.data = self._vr_reference.latest_obs[0].tolist()
            self.latest_obs_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish latest_obs: {e}")

    def _apply_onnx_metadata(self):
        """Apply PD/scale/defaults from ONNX metadata as authoritative values.
        Load separate metadata for velocity and motion models."""
        # Load velocity model metadata
        velocity_meta = read_onnx_metadata(self.velocity_onnx_path)
        self.velocity_dof_names_onnx = velocity_meta["joint_names"]
        self.velocity_action_scale_onnx = velocity_meta["action_scale"].astype(np.float32)
        self.velocity_kps_onnx = velocity_meta["kps"].astype(np.float32)
        self.velocity_kds_onnx = velocity_meta["kds"].astype(np.float32)
        self.velocity_default_angles_onnx = velocity_meta["default_joint_pos"].astype(np.float32)

        # Load motion model metadata
        motion_meta = read_onnx_metadata(self.motion_onnx_path)
        self.motion_dof_names_onnx = motion_meta["joint_names"]
        self.motion_action_scale_onnx = motion_meta["action_scale"].astype(np.float32)
        self.motion_kps_onnx = motion_meta["kps"].astype(np.float32)
        self.motion_kds_onnx = motion_meta["kds"].astype(np.float32)
        self.motion_default_angles_onnx = motion_meta["default_joint_pos"].astype(np.float32)
        configured_rope_max_seq_len = int(
            getattr(self.deployment_config, "motion_rope_max_seq_len", 0) or 0
        )
        artifact_rope_max_seq_len = motion_meta.get("rope_max_seq_len", None)
        if configured_rope_max_seq_len > 0:
            if (
                artifact_rope_max_seq_len is not None
                and int(artifact_rope_max_seq_len) != configured_rope_max_seq_len
            ):
                self.get_logger().warn(
                    "motion_rope_max_seq_len profile override "
                    f"({configured_rope_max_seq_len}) differs from motion ONNX "
                    f"artifact metadata ({artifact_rope_max_seq_len})."
                )
            self.motion_rope_max_seq_len = configured_rope_max_seq_len
        elif artifact_rope_max_seq_len is not None:
            self.motion_rope_max_seq_len = int(artifact_rope_max_seq_len)
            self.get_logger().info(
                "Motion RoPE max sequence length loaded from ONNX artifact: "
                f"{self.motion_rope_max_seq_len}"
            )
        elif self.motion_step_idx_input_name is not None:
            raise RuntimeError(
                "Motion policy uses step_idx/KV-cache but ONNX artifact does not "
                "provide rope_max_seq_len metadata. Re-export the motion policy "
                "with current code, or set motion_rope_max_seq_len explicitly in "
                "the launch profile for a legacy artifact."
            )
        else:
            self.motion_rope_max_seq_len = 0

        # Use velocity model metadata as default (for backward compatibility)
        self.dof_names_onnx = self.velocity_dof_names_onnx
        self.action_scale_onnx = self.velocity_action_scale_onnx
        self.kps_onnx = self.velocity_kps_onnx
        self.kds_onnx = self.velocity_kds_onnx
        self.default_angles_onnx = self.velocity_default_angles_onnx
        self.default_angles_dict = {
            name: float(self.default_angles_onnx[idx])
            for idx, name in enumerate(self.dof_names_onnx)
        }

    def _build_dof_mappings(self):
        # Map ONNX <-> MJCF for control

        # Check if all ONNX names exist in real_dof_names (use velocity as reference)
        missing_names = [name for name in self.velocity_dof_names_onnx if name not in self.real_dof_names]
        if missing_names:
            self.get_logger().warn(f"Missing names in real_dof_names: {missing_names}")

        # Build mappings for velocity model
        self.velocity_onnx_to_real = [
            self.velocity_dof_names_onnx.index(name) for name in self.real_dof_names
        ]
        self.velocity_kps_real = self.velocity_kps_onnx[self.velocity_onnx_to_real].astype(np.float32)
        self.velocity_kds_real = self.velocity_kds_onnx[self.velocity_onnx_to_real].astype(np.float32)

        # Build mappings for motion model
        self.motion_onnx_to_real = [
            self.motion_dof_names_onnx.index(name) for name in self.real_dof_names
        ]
        self.motion_kps_real = self.motion_kps_onnx[self.motion_onnx_to_real].astype(np.float32)
        self.motion_kds_real = self.motion_kds_onnx[self.motion_onnx_to_real].astype(np.float32)

        # Use velocity model mappings as default (for backward compatibility)
        self.onnx_to_real = self.velocity_onnx_to_real
        self.kps_real = self.velocity_kps_real
        self.kds_real = self.velocity_kds_real
        self.default_angles_mu = self.velocity_default_angles_onnx[self.velocity_onnx_to_real].astype(np.float32)
        self.action_scale_mu = self.velocity_action_scale_onnx[self.velocity_onnx_to_real].astype(np.float32)

        self.observation_evaluator.initialize_observation_state()

        # Publish kps and kds parameters (use velocity as default)
        self._publish_control_params()

    def _publish_control_params(self):
        """Publish kps and kds control parameters based on current policy mode.

        Called during initialization and mode switching to ensure control node
        receives the correct parameters for the current policy mode.
        """
        try:
            # Use appropriate parameters based on current policy mode
            if self.current_policy_mode == "motion":
                current_kps = self.motion_kps_real
                current_kds = self.motion_kds_real
            else:  # velocity mode
                current_kps = self.velocity_kps_real
                current_kds = self.velocity_kds_real

            # Publish kps
            kps_msg = Float32MultiArray()
            kps_msg.data = current_kps.tolist()
            self.kps_pub.publish(kps_msg)

            # Publish kds
            kds_msg = Float32MultiArray()
            kds_msg.data = current_kds.tolist()
            self.kds_pub.publish(kds_msg)

            self.get_logger().info(
                f"Published control parameters ({self.current_policy_mode} mode): "
                f"kps={len(current_kps)}, kds={len(current_kds)}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to publish control parameters: {e}")

    def _publish_policy_mode(self, force: bool = False):
        """Publish current policy mode status."""
        try:
            data = f"{self.current_policy_mode}_{'enabled' if self.policy_enabled else 'disabled'}"
            now = time.time()
            if (
                not force
                and data == self._last_policy_mode_msg_data
                and now - self._last_policy_mode_publish_time
                < self._policy_mode_publish_interval_sec
            ):
                return
            self._policy_mode_msg.data = data
            self.policy_mode_pub.publish(self._policy_mode_msg)
            self._last_policy_mode_msg_data = data
            self._last_policy_mode_publish_time = now
        except Exception as e:
            self.get_logger().error(f"Failed to publish policy mode: {e}")

    def _timing_ms(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _record_timing_sample(self, sample: dict):
        if not getattr(self, "timing_debug_enabled", False):
            return
        self._timing_debug_samples.append(sample)
        if getattr(self, "timing_debug_log_per_loop", False):
            self.get_logger().info(
                "[Timing] "
                f"loop_total={sample['loop_total_ms']:.2f}ms "
                f"io={sample['io_ms']:.2f}ms "
                f"policy_total={sample['policy_total_ms']:.2f}ms "
                f"fk={sample['fk_ms']:.2f}ms "
                f"obs={sample['obs_ms']:.2f}ms "
                f"onnx={sample['onnx_ms']:.2f}ms "
                f"post={sample['post_ms']:.2f}ms"
            )

        now = time.time()
        last = getattr(self, "_timing_debug_last_log_time", None)
        interval = float(getattr(self, "timing_debug_log_interval_sec", 5.0))
        if last is None:
            self._timing_debug_last_log_time = now
            return
        if now - last < interval:
            return
        if len(self._timing_debug_samples) == 0:
            self._timing_debug_last_log_time = now
            return

        keys = [
            "loop_total_ms",
            "io_ms",
            "policy_total_ms",
            "fk_ms",
            "obs_ms",
            "onnx_ms",
            "post_ms",
        ]
        stats = {}
        for key in keys:
            vals = np.array(
                [float(s.get(key, 0.0)) for s in self._timing_debug_samples],
                dtype=np.float64,
            )
            stats[key] = (float(np.mean(vals)), float(np.max(vals)))
        self.get_logger().info(
            "[Timing-Agg] "
            + " ".join(
                f"{key}=mean:{stats[key][0]:.2f}ms/max:{stats[key][1]:.2f}ms"
                for key in keys
            )
            + f" n={len(self._timing_debug_samples)}"
        )
        self._timing_debug_samples.clear()
        self._timing_debug_last_log_time = now

    def _publish_action_target(self, target_dof_pos_real):
        """Publish action commands prepared by PolicyRuntime."""
        if target_dof_pos_real is None:
            return
        action_msg = self._action_msg
        action_msg.data = target_dof_pos_real.tolist()

        # Check for NaN values
        if np.isnan(target_dof_pos_real).any():
            self.get_logger().error("Action contains NaN values")

        self.action_pub.publish(action_msg)

    def setup(self):
        """Set up the evaluator by loading all required components."""
        main_affinity = parse_cpu_affinity(
            getattr(self, "_cpu_affinity_main_str", "") or ""
        )
        if main_affinity and set_thread_cpu_affinity(main_affinity):
            self.get_logger().info(f"[Policy] main thread pinned to CPUs {main_affinity}")
        self.load_model_config()  # Load config first
        self.update_config_parameters()  # Update parameters from config
        # Initialize FK for online VR reference reconstruction
        self.observation_evaluator.initialize_fk()
        self.load_policy()        # Then load policies
        self._apply_onnx_metadata()
        self._init_obs_buffers()
        self._build_dof_mappings()
        self._warmup_motion_policy()
        self.observation_evaluator._init_keybody_indices_cache()
        # Always load motion data since we support both modes
        self.load_motion_data()
        self.get_logger().info("Synchronous root-only policy setup completed")

    def destroy_node(self):
        try:
            if getattr(self, "_zmq_subscriber", None) is not None:
                self._zmq_subscriber.stop()
        except Exception:
            pass

        super().destroy_node()


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = HoloMotionPolicyNode()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
