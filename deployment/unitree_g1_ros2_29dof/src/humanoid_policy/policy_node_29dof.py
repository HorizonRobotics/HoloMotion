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
import torch
import time
import json
import threading
from collections import deque

import easydict
import numpy as np
import onnx
import onnxruntime
import rclpy
import zmq
import yaml
from ament_index_python.packages import get_package_share_directory
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState

from humanoid_policy.obs_builder import PolicyObsBuilder
from humanoid_policy.utils.remote_controller_filter import KeyMap, RemoteController
from humanoid_policy.holomotion_fk_root_only import HoloMotionFKRootOnly


def _parse_cpu_affinity_str(s):
    """Parse '0,1' or '2' -> [0,1] or [2]. Empty/invalid -> []."""
    s = str(s).strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out


def set_thread_cpu_affinity(cpu_ids):
    """Pin current thread to given CPU core IDs (Linux only).
    cpu_ids: list of int, e.g. [0], [0,1]. Returns True if set successfully."""
    if not cpu_ids:
        return False
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        CPU_SETSIZE = 1024
        ncpubits = 8 * ctypes.sizeof(ctypes.c_ulong)
        nlongs = (CPU_SETSIZE + ncpubits - 1) // ncpubits
        class CpuSetT(ctypes.Structure):
            _fields_ = [("__bits", ctypes.c_ulong * nlongs)]
        libc.pthread_self.restype = ctypes.c_ulong
        libc.pthread_setaffinity_np.argtypes = [
            ctypes.c_ulong, ctypes.c_size_t, ctypes.POINTER(CpuSetT)
        ]
        cs = CpuSetT()
        for i in range(nlongs):
            cs.__bits[i] = 0
        for c in cpu_ids:
            if 0 <= c < CPU_SETSIZE:
                idx = c // ncpubits
                bit = c % ncpubits
                cs.__bits[idx] |= 1 << bit
        tid = libc.pthread_self()
        sz = ctypes.sizeof(CpuSetT)
        ret = libc.pthread_setaffinity_np(tid, sz, ctypes.byref(cs))
        return ret == 0
    except Exception:
        return False


HEADER_SIZE = 1280
DEFAULT_ZMQ_TOPIC = b"obs65"
_DTYPE_BY_NAME = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "bool": np.bool_,
}


def _decode_zmq_topic(topic_value) -> bytes:
    if isinstance(topic_value, bytes):
        return topic_value
    return str(topic_value).encode("utf-8")


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


def _infer_onnx_dim(dim, default: int = 1) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return int(default)


def _infer_numpy_dtype_from_onnx_type(type_str: str):
    type_str = str(type_str).lower()
    if "float16" in type_str:
        return np.float16
    if "float64" in type_str or "double" in type_str:
        return np.float64
    if "int64" in type_str:
        return np.int64
    if "int32" in type_str:
        return np.int32
    if "bool" in type_str:
        return np.bool_
    return np.float32


def unpack_numpy_message(packet: bytes, expected_topic: bytes | None = None) -> dict:
    if expected_topic is not None:
        if not packet.startswith(expected_topic):
            raise ValueError("ZMQ packet topic prefix mismatch")
        packet = packet[len(expected_topic) :]

    if len(packet) < HEADER_SIZE:
        raise ValueError(f"ZMQ packet too short: {len(packet)} < {HEADER_SIZE}")

    header_bytes = packet[:HEADER_SIZE].rstrip(b"\x00")
    if not header_bytes:
        raise ValueError("ZMQ packet has empty header")
    header = json.loads(header_bytes.decode("utf-8"))

    payload = memoryview(packet)[HEADER_SIZE:]
    result = {}
    offset = 0
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        shape = tuple(int(x) for x in field.get("shape", []))
        if dtype_name not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in ZMQ packet: {dtype_name}")

        dtype = np.dtype(_DTYPE_BY_NAME[dtype_name]).newbyteorder("<")
        count = int(np.prod(shape, dtype=np.int64)) if len(shape) > 0 else 1
        nbytes = count * dtype.itemsize
        end = offset + nbytes
        if end > len(payload):
            raise ValueError(
                f"ZMQ packet field '{name}' exceeds payload size: end={end}, payload={len(payload)}"
            )
        arr = np.frombuffer(payload[offset:end], dtype=dtype, count=count)
        if len(shape) > 0:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())
        result[name] = np.array(arr, copy=True)
        offset = end
    return result


class LatestObsBuffer:
    """Thread-safe buffer for delayed latest_obs access."""

    def __init__(self, max_queue_size: int = 20):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp = None
        self._sender_timestamp = None
        self._frame_index = None
        self._data_queue = deque(maxlen=max_queue_size)
        self._timestamp_queue = deque(maxlen=max_queue_size)
        self._sender_timestamp_queue = deque(maxlen=max_queue_size)
        self._frame_index_queue = deque(maxlen=max_queue_size)

    def set(
        self,
        arr: np.ndarray,
        sender_timestamp: float | None = None,
        frame_index: int | None = None,
    ):
        with self._lock:
            current_time = time.time()
            arr_copy = np.asarray(arr, dtype=np.float32).copy()
            self._data = arr_copy
            self._timestamp = current_time
            self._sender_timestamp = sender_timestamp
            self._frame_index = frame_index
            self._data_queue.append(arr_copy)
            self._timestamp_queue.append(current_time)
            self._sender_timestamp_queue.append(sender_timestamp)
            self._frame_index_queue.append(frame_index)

    def get_with_age_and_delay(self, max_age: float = 0.1, delay_steps: int = 0):
        """Return a delayed frame and report whether it is stale."""
        with self._lock:
            if len(self._data_queue) == 0:
                if self._data is None or self._timestamp is None:
                    return None, None, True, None, None
                current_time = time.time()
                age = current_time - self._timestamp
                return (
                    self._data,
                    self._timestamp,
                    age > max_age,
                    self._frame_index,
                    self._sender_timestamp,
                )

            if delay_steps < 0:
                delay_steps = 0
            idx = len(self._data_queue) - 1 - delay_steps
            if idx < 0:
                idx = 0

            data = self._data_queue[idx]
            ts = self._timestamp_queue[idx]
            frame_index = self._frame_index_queue[idx]
            sender_timestamp = self._sender_timestamp_queue[idx]

        current_time = time.time()
        age = current_time - ts
        is_stale = age > max_age
        return data, ts, is_stale, frame_index, sender_timestamp

    def get_queue_stats(self):
        with self._lock:
            if len(self._data_queue) < 2:
                return {"queue_size": len(self._data_queue), "avg_interval": None}
            intervals = []
            for i in range(1, len(self._timestamp_queue)):
                interval = self._timestamp_queue[i] - self._timestamp_queue[i - 1]
                intervals.append(interval)
            avg_interval = float(np.mean(intervals)) if intervals else None
            return {
                "queue_size": len(self._data_queue),
                "avg_interval": avg_interval,
                "expected_freq": 1.0 / avg_interval if avg_interval and avg_interval > 0 else None,
            }


class ZmqLatestObsSubscriber:
    """Background ZMQ SUB receiver for latest_obs packets."""

    def __init__(
        self,
        uri: str,
        topic: bytes,
        buffer: LatestObsBuffer,
        logger,
        mode: str = "connect",
        cpu_affinity=None,
        conflate: bool = True,
    ):
        self.uri = uri
        self.topic = topic
        self.buffer = buffer
        self.logger = logger
        self.mode = str(mode).strip().lower()
        self.cpu_affinity = cpu_affinity or []
        self.conflate = bool(conflate)

        self._thread = None
        self._stop_event = threading.Event()
        self._context = None
        self._socket = None
        self._poller = None
        self._recv_count = 0

    def _process_packet(self, packet: bytes):
        payload = unpack_numpy_message(packet, expected_topic=self.topic)
        latest_obs = payload.get("latest_obs", None)
        if latest_obs is None:
            raise ValueError("ZMQ packet missing latest_obs field")

        frame_index = payload.get("frame_index", None)
        if frame_index is not None:
            frame_index = int(np.asarray(frame_index).reshape(-1)[0])

        sender_timestamp = payload.get("timestamp_realtime", None)
        if sender_timestamp is not None:
            sender_timestamp = float(np.asarray(sender_timestamp).reshape(-1)[0])

        self.buffer.set(
            np.asarray(latest_obs, dtype=np.float32),
            sender_timestamp=sender_timestamp,
            frame_index=frame_index,
        )
        self._recv_count += 1
        if self._recv_count == 1:
            self.logger.info(
                f"[ZMQ] first latest_obs packet received from {self.uri}, "
                f"topic={self.topic.decode('utf-8', errors='ignore')}"
            )

    def _run(self):
        if self.cpu_affinity and set_thread_cpu_affinity(self.cpu_affinity):
            self.logger.info(f"[ZMQ] subscriber thread pinned to CPUs {self.cpu_affinity}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        if self.conflate and hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1)

        if self.mode == "bind":
            self._socket.bind(self.uri)
        elif self.mode == "connect":
            self._socket.connect(self.uri)
        else:
            raise ValueError("latest_obs_zmq_mode must be 'bind' or 'connect'")

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self.logger.info(
            f"[ZMQ] latest_obs subscriber ready: mode={self.mode}, uri={self.uri}, "
            f"topic={self.topic.decode('utf-8', errors='ignore')}, conflate={self.conflate}"
        )

        try:
            while not self._stop_event.is_set():
                events = dict(self._poller.poll(50))
                if self._socket not in events:
                    continue
                try:
                    packet = self._socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                self._process_packet(packet)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.logger.error(f"[ZMQ] subscriber loop failed: {exc}")
        finally:
            try:
                if self._poller is not None and self._socket is not None:
                    self._poller.unregister(self._socket)
            except Exception:
                pass
            try:
                if self._socket is not None:
                    self._socket.close(0)
            except Exception:
                pass
            try:
                if self._context is not None:
                    self._context.term()
            except Exception:
                pass
            self._socket = None
            self._context = None
            self._poller = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.logger.info("[ZMQ] subscriber thread started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.logger.info("[ZMQ] subscriber thread stopped")


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

        # Get config path from ROS parameter
        config_path = self.declare_parameter("config_path", "").value
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config_yaml = easydict.EasyDict(yaml.safe_load(config_file))
        # Read policy frequency from config, default to 50 Hz if not specified
        policy_freq = self.config_yaml.get("policy_freq", 50)
        self.dt = 1.0 / policy_freq
        self.get_logger().info(f"Policy frequency set to: {policy_freq} Hz (dt = {self.dt:.4f} s)")
        # Initialize basic parameters - will be updated after config loading
        self.actions_dim = 29  # Default value, will be updated from config
        self.real_dof_names = []  # Will be loaded from config
        self.current_motion_clip_index = 0  # Current motion clip index
        # Button state tracking for preventing multiple triggers
        self.last_button_states = {
            KeyMap.up: 0,
            KeyMap.down: 0,
            KeyMap.left: 0,
            KeyMap.right: 0,
            KeyMap.A: 0,
            KeyMap.B: 0,
            KeyMap.Y: 0,
        }
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

        self.external_latest_obs = None
        self.external_obs_received = False
        self.last_external_obs_time = None
        self._latest_sender_timestamp = None
        self.latest_obs_flag = False
        self.latest_obs_expected_dim = 65
        self.external_fut_dof_pos_queue = None
        self.external_fut_dof_vel_queue = None
        self.external_fut_root_pos_queue = None
        self.external_fut_root_rot_queue = None
        self.external_fut_frame_idx_queue = None
        self._prev_external_dof_pos = None
        self._prev_external_dof_vel = None
        self._prev_external_root_pos = None
        self._prev_external_root_rot = None
        self._prev_external_frame_idx = None
        self.max_data_age = 0.6
        self.stale_data_warning_count = 0
        self.last_poll_time = None
        self._last_vr_status_log_time = None
        self.latest_obs_zmq_uri = self.declare_parameter(
            "latest_obs_zmq_uri", "tcp://192.168.124.29:6001"
        ).value
        self.latest_obs_zmq_topic = self.declare_parameter(
            "latest_obs_zmq_topic", DEFAULT_ZMQ_TOPIC.decode("utf-8")
        ).value
        self.latest_obs_zmq_mode = self.declare_parameter(
            "latest_obs_zmq_mode", "connect"
        ).value
        self.latest_obs_zmq_conflate = self.declare_parameter(
            "latest_obs_zmq_conflate", True
        ).value
        self.zmq_jitter_delay_frames = self.declare_parameter(
            "zmq_jitter_delay_frames", 5
        ).value
        self.require_vr_data_for_motion = self.declare_parameter(
            "require_vr_data_for_motion", True
        ).value
        self.enable_teleop_reference = self.declare_parameter(
            "enable_teleop_reference", True
        ).value
        self._cpu_affinity_main_str = self.declare_parameter(
            "cpu_affinity_main", ""
        ).value
        self._cpu_affinity_zmq_sub_str = self.declare_parameter(
            "cpu_affinity_zmq_sub", ""
        ).value
        self.timing_debug_enabled = self.declare_parameter(
            "timing_debug_enabled", True
        ).value
        self.timing_debug_log_interval_sec = self.declare_parameter(
            "timing_debug_log_interval_sec", 5.0
        ).value
        self.timing_debug_log_per_loop = self.declare_parameter(
            "timing_debug_log_per_loop", False
        ).value
        self._timing_debug_last_log_time = None
        self._timing_debug_samples = deque(maxlen=500)
        self._root_only_fk_keybody_warned = False

        _vr = getattr(self.config_yaml, "vr", None) or {}
        if _vr:
            self.latest_obs_zmq_uri = str(_vr.get("latest_obs_zmq_uri", self.latest_obs_zmq_uri))
            self.latest_obs_zmq_topic = str(
                _vr.get("latest_obs_zmq_topic", self.latest_obs_zmq_topic)
            )
            self.latest_obs_zmq_mode = str(
                _vr.get("latest_obs_zmq_mode", self.latest_obs_zmq_mode)
            )
            self.latest_obs_zmq_conflate = bool(
                _vr.get("latest_obs_zmq_conflate", self.latest_obs_zmq_conflate)
            )
            self.zmq_jitter_delay_frames = int(
                _vr.get("zmq_jitter_delay_frames", self.zmq_jitter_delay_frames)
            )
            self.max_data_age = float(_vr.get("max_data_age", self.max_data_age))
            self.require_vr_data_for_motion = bool(
                _vr.get("require_vr_data_for_motion", self.require_vr_data_for_motion)
            )
            self.enable_teleop_reference = bool(
                _vr.get("enable_teleop_reference", self.enable_teleop_reference)
            )
            self.timing_debug_enabled = bool(
                _vr.get("timing_debug_enabled", self.timing_debug_enabled)
            )
            self.timing_debug_log_interval_sec = float(
                _vr.get(
                    "timing_debug_log_interval_sec",
                    self.timing_debug_log_interval_sec,
                )
            )
            self.timing_debug_log_per_loop = bool(
                _vr.get("timing_debug_log_per_loop", self.timing_debug_log_per_loop)
            )

        self._cpu_affinity_main_str = str(
            getattr(self.config_yaml, "cpu_affinity_main", self._cpu_affinity_main_str)
        )
        self._cpu_affinity_zmq_sub_str = str(
            getattr(
                self.config_yaml,
                "cpu_affinity_zmq_sub",
                self._cpu_affinity_zmq_sub_str,
            )
        )
        self._ros_latest_obs_buffer = None
        self._npz_replay_frame_index = None
        self._external_seen_frames = 0
        self._vr_ready_logged = False

        self._latest_obs_buffer = LatestObsBuffer()
        self._latest_obs_zmq_topic_bytes = _decode_zmq_topic(self.latest_obs_zmq_topic)
        if str(self.latest_obs_zmq_mode).strip().lower() == "connect":
            uri_str = str(self.latest_obs_zmq_uri)
            if "*" in uri_str or "0.0.0.0" in uri_str:
                self.get_logger().warn(
                    "[ZMQ] connect mode requires a concrete peer address. "
                    "Do not use '*' or '0.0.0.0'; use the sender IP instead, "
                    "for example tcp://192.168.124.29:6001."
                )
        zmq_cpu_affinity = _parse_cpu_affinity_str(self._cpu_affinity_zmq_sub_str)
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
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self._lowstate_msg = None
        self.target_dof_pos_real = None
        self.motion_in_progress = False
        self._keybody_indices_by_term_name = {}
        self.fk = None
        self.fk_initialized = False
        self.motion_action_ema_filter_enabled = False
        self.motion_action_ema_filter_alpha = 1.0
        self._motion_filtered_actions_onnx = None

    def _is_vr_ready_for_motion(self) -> bool:
        """Return whether the ZMQ reference stream is ready for motion mode."""
        if not getattr(self, "enable_teleop_reference", True):
            return False
        if not (
            getattr(self, "external_obs_received", False)
            and getattr(self, "external_latest_obs", None) is not None
        ):
            return False
        n_fut = int(getattr(self, "n_fut_frames", 0) or 0)
        if n_fut <= 0:
            return True
        delay = int(getattr(self, "zmq_jitter_delay_frames", 0) or 0)
        needed = n_fut + max(delay, 0) + 1
        return int(getattr(self, "_external_seen_frames", 0)) >= needed

    
    def _init_keybody_indices_cache(self):
        if self.motion_config is None:
            raise ValueError("motion_config is not loaded; cannot init keybody index cache")

        atomic_list = self._get_policy_atomic_obs_list(self.motion_config)["atomic_obs_list"]
        body_names = [str(name) for name in self.motion_config.robot.body_names]
        body_name_to_idx = {body_name: idx for idx, body_name in enumerate(body_names)}

        cache = {}
        for term_dict in atomic_list:
            term_name = str(list(term_dict.keys())[0])
            term_cfg = term_dict[term_name]
            params = {}
            if isinstance(term_cfg, dict):
                params = term_cfg.get("params", {}) or {}
                if not isinstance(params, dict):
                    raise ValueError(
                        f"Observation term '{term_name}' params must be a dict, got {type(params)}"
                    )
            needs_keybody = ("keybody" in term_name) or ("keybody_names" in params)
            if not needs_keybody:
                continue

            keybody_names = params.get("keybody_names", None)
            if keybody_names is None:
                keybody_idxs = np.arange(len(body_names), dtype=np.int64)
            else:
                keybody_names = [str(name) for name in keybody_names]
                missing_names = [
                    name for name in keybody_names if name not in body_name_to_idx
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Unknown keybody_names in '{term_name}': {missing_names}. "
                        f"Available body names: {body_names}"
                    )
                keybody_idxs = np.asarray(
                    [body_name_to_idx[name] for name in keybody_names],
                    dtype=np.int64,
                )

            cache[term_name] = keybody_idxs

        self._keybody_indices_by_term_name = cache

    def _get_policy_atomic_obs_list(self, config):
        """Resolve the atomic obs list used to build the ONNX policy input.

        Aligns with MuJoCo sim2sim eval ordering by honoring modules.actor.obs_schema
        when available, to guarantee the policy input term order matches training/export.
        """

        def _to_plain_obs_cfg(cfg):
            if OmegaConf.is_config(cfg):
                plain_cfg = OmegaConf.to_container(cfg, resolve=True)
            elif cfg is None:
                plain_cfg = {}
            else:
                plain_cfg = dict(cfg)
            if plain_cfg is None:
                plain_cfg = {}
            if not isinstance(plain_cfg, dict):
                raise ValueError(
                    f"Observation term config must be a mapping, got {type(plain_cfg)}"
                )
            return plain_cfg

        def _get_actor_atomic_obs_entries():
            obs_cfg = config.get("obs", None)
            if obs_cfg is None:
                raise ValueError("Missing config.obs for policy obs")
            obs_groups = obs_cfg.get("obs_groups", None)
            if obs_groups is None:
                raise ValueError("Missing config.obs.obs_groups for policy obs")

            if obs_groups.get("policy", None) is not None:
                entries = []
                for term_dict in obs_groups.policy.atomic_obs_list:
                    term_name = str(list(term_dict.keys())[0])
                    entries.append(
                        (
                            "policy",
                            term_name,
                            _to_plain_obs_cfg(term_dict[term_name]),
                        )
                    )
                return entries

            if obs_groups.get("unified", None) is not None:
                entries = []
                for term_dict in obs_groups.unified.atomic_obs_list:
                    term_name = str(list(term_dict.keys())[0])
                    if term_name.startswith("actor_"):
                        entries.append(
                            (
                                "unified",
                                term_name,
                                _to_plain_obs_cfg(term_dict[term_name]),
                            )
                        )
                if not entries:
                    raise ValueError(
                        "obs_groups.unified found but contains no actor_* terms."
                    )
                return entries

            raise ValueError(
                "Unsupported obs config : expected obs_groups.policy or obs_groups.unified."
            )

        def _get_actor_obs_schema_terms():
            modules_cfg = config.get("modules", None)
            if modules_cfg is None:
                return []
            actor_cfg = modules_cfg.get("actor", None)
            if actor_cfg is None:
                return []
            obs_schema = actor_cfg.get("obs_schema", None)
            if obs_schema is None:
                return []

            if OmegaConf.is_config(obs_schema):
                obs_schema_plain = OmegaConf.to_container(obs_schema, resolve=True)
            else:
                obs_schema_plain = obs_schema
            if not isinstance(obs_schema_plain, dict):
                return []

            ordered_terms = []

            def _collect_terms(node):
                if node is None:
                    return
                if isinstance(node, dict):
                    if "terms" in node and isinstance(node["terms"], list):
                        ordered_terms.extend(str(term) for term in node["terms"])
                        return
                    for v in node.values():
                        _collect_terms(v)
                    return
                if isinstance(node, list):
                    for v in node:
                        _collect_terms(v)
                    return

            _collect_terms(obs_schema_plain)
            return ordered_terms

        actor_atomic_entries = _get_actor_atomic_obs_entries()
        schema_terms = _get_actor_obs_schema_terms()

        if len(schema_terms) == 0:
            return {
                "atomic_obs_list": [
                    {term_name: term_cfg}
                    for _, term_name, term_cfg in actor_atomic_entries
                ]
            }

        by_full_key = {}
        by_leaf_key = {}
        ambiguous_leaf_keys = set()
        for group_name, term_name, term_cfg in actor_atomic_entries:
            full_key = f"{group_name}/{term_name}"
            by_full_key[full_key] = (term_name, term_cfg)
            if term_name in by_leaf_key:
                ambiguous_leaf_keys.add(term_name)
            else:
                by_leaf_key[term_name] = (term_name, term_cfg)

        ordered_atomic_list = []
        for schema_term in schema_terms:
            schema_term_key = str(schema_term)
            if schema_term_key in by_full_key:
                term_name, term_cfg = by_full_key[schema_term_key]
                ordered_atomic_list.append({term_name: term_cfg})
                continue

            leaf_key = schema_term_key.split("/")[-1]
            if leaf_key in ambiguous_leaf_keys:
                raise ValueError(
                    f"Ambiguous obs_schema term '{schema_term_key}': "
                    f"multiple atomic obs share leaf key '{leaf_key}'."
                )
            if leaf_key not in by_leaf_key:
                available = sorted(list(by_leaf_key.keys()))
                raise ValueError(
                    f"obs_schema term '{schema_term_key}' not found in atomic_obs_list. "
                    f"Available terms: {available}"
                )
            term_name, term_cfg = by_leaf_key[leaf_key]
            ordered_atomic_list.append({term_name: term_cfg})

        return {"atomic_obs_list": ordered_atomic_list}

    def _find_actor_place_holder_ndim(self):
        n_dim = 0
        atomic_list = self._get_policy_atomic_obs_list(self.motion_config)[
            "atomic_obs_list"
        ]
        for obs_dict in atomic_list:
            name = str(list(obs_dict.keys())[0])
            if name == "place_holder" or name == "actor_place_holder":
                cfg = obs_dict[name]
                params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
                n_dim = int(params.get("n_dim", 0))
        return n_dim

    def _init_obs_buffers(self):
        """Initialize observation builders for both velocity and motion policies.
        
        Each obs_builder uses its own model's dof_names_onnx and default_angles_onnx
        to ensure correct observation computation for each policy.
        """
        # Use velocity model's parameters for velocity obs_builder
        self.velocity_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.velocity_dof_names_onnx,
            default_angles_onnx=self.velocity_default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=self._get_policy_atomic_obs_list(self.velocity_config),
        )

        # Use motion model's parameters for motion obs_builder
        self.motion_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.motion_dof_names_onnx,
            default_angles_onnx=self.motion_default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=self._get_policy_atomic_obs_list(self.motion_config),
        )

        if hasattr(self, "n_fut_frames") and int(self.n_fut_frames) > 0:
            n_fut = int(self.n_fut_frames)
            self.external_fut_dof_pos_queue = np.zeros((n_fut, self.num_actions), dtype=np.float32)
            self.external_fut_dof_vel_queue = np.zeros((n_fut, self.num_actions), dtype=np.float32)
            self.external_fut_root_pos_queue = np.zeros((n_fut, 3), dtype=np.float32)
            self.external_fut_root_rot_queue = np.zeros((n_fut, 4), dtype=np.float32)
            self._fk_root_pos_seq_np = np.zeros((1, n_fut + 1, 3), dtype=np.float32)
            self._fk_root_rot_seq_np = np.zeros((1, n_fut + 1, 4), dtype=np.float32)
            self._fk_dof_pos_seq_np = np.zeros(
                (1, n_fut + 1, self.num_actions), dtype=np.float32
            )
            self._fk_root_pos_seq_tensor = torch.from_numpy(self._fk_root_pos_seq_np)
            self._fk_root_rot_seq_tensor = torch.from_numpy(self._fk_root_rot_seq_np)
            self._fk_dof_pos_seq_tensor = torch.from_numpy(self._fk_dof_pos_seq_np)
            self.external_fut_frame_idx_queue = np.full((n_fut,), -1, dtype=np.int32)
            self.get_logger().info(
                f"Initialized VR future frame queues: n_fut_frames={n_fut}, num_actions={self.num_actions}"
            )
        else:
            self.external_fut_dof_pos_queue = None
            self.external_fut_dof_vel_queue = None
            self.external_fut_root_pos_queue = None
            self.external_fut_root_rot_queue = None
            self.external_fut_frame_idx_queue = None
            self._fk_root_pos_seq_np = None
            self._fk_root_rot_seq_np = None
            self._fk_dof_pos_seq_np = None
            self._fk_root_pos_seq_tensor = None
            self._fk_root_rot_seq_tensor = None
            self._fk_dof_pos_seq_tensor = None

        # Set default obs_builder to velocity mode
        self.obs_builder = self.velocity_obs_builder

    def _reset_counter(self):
        """Reset motion timing counters to start of sequence."""
        self.motion_frame_idx = 0
        self.motion_step_idx = 0
        if self.use_kv_cache and self.motion_kv_cache is not None:
            self.motion_kv_cache.fill(0)

    def _switch_to_velocity_mode(self, reason: str = ""):
        """Switch to velocity tracking mode and clear action cache.
        
        Uses velocity model's default_angles_onnx to ensure correct initialization.
        Also publishes velocity model's control parameters (kps/kds).
        """
        self.current_policy_mode = "velocity"
        self.latest_obs_flag = False
        self.motion_in_progress = False
        self._fk_vr_out = None
        self._use_fk_vr = False
        self._reset_motion_action_ema_filter()
        self._reset_counter()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        # Use velocity model's default angles
        self.target_dof_pos_onnx = self.velocity_default_angles_onnx.copy()
        # Publish velocity model's control parameters
        self._publish_control_params()
        if reason:
            self.get_logger().info(f"Switched to velocity tracking mode ({reason})")
        else:
            self.get_logger().info("Switched to velocity tracking mode")

    def _is_button_pressed(self, button_key):
        """Check if button was just pressed (rising edge detection)."""
        current_state = self.remote_controller.button[button_key]
        last_state = self.last_button_states[button_key]  
        # Update the last state
        self.last_button_states[button_key] = current_state
        # Return True only on rising edge (0 -> 1)
        return current_state == 1 and last_state == 0

    def load_policy(self):
        """Load both velocity and motion policy models using ONNX Runtime."""
        self.get_logger().info("Loading dual policies...")
        
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                },
            ),
            "CPUExecutionProvider",
        ]
        onnx_threads = int(self.config_yaml.get("onnx_intra_op_threads", 2))
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = onnx_threads
        sess_options.inter_op_num_threads = 1
        # Load velocity policy from model folder
        velocity_model_folder = self.config_yaml.velocity_tracking_model_folder
        velocity_model_path = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            velocity_model_folder,
            "exported",
        )
        # Find ONNX file in exported folder
        velocity_onnx_files = [f for f in os.listdir(velocity_model_path) if f.endswith('.onnx')]
        if not velocity_onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {velocity_model_path}")
        
        velocity_onnx_path = os.path.join(velocity_model_path, velocity_onnx_files[0])
        self.get_logger().info(f"Loading velocity policy from {velocity_onnx_path}")
        
        self.velocity_policy_session = onnxruntime.InferenceSession(
            str(velocity_onnx_path), sess_options=sess_options, providers=providers
        )
        self.get_logger().info(
            f"Velocity policy loaded successfully using: "
            f"{self.velocity_policy_session.get_providers()}"
        )
        # Load motion policy from model folder
        motion_model_folder = self.config_yaml.motion_tracking_model_folder
        motion_model_path = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            motion_model_folder,
            "exported",
        )
        # Find ONNX file in exported folder
        motion_onnx_files = [f for f in os.listdir(motion_model_path) if f.endswith('.onnx')]
        if not motion_onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {motion_model_path}")
        
        motion_onnx_path = os.path.join(motion_model_path, motion_onnx_files[0])
        self.get_logger().info(f"Loading motion policy from {motion_onnx_path}")
        
        self.motion_policy_session = onnxruntime.InferenceSession(
            str(motion_onnx_path), sess_options=sess_options, providers=providers
        )
        self.get_logger().info(
            f"Motion policy loaded successfully using: "
            f"{self.motion_policy_session.get_providers()}"
        )
        # Set input/output names for both policies
        self.velocity_input_name = self.velocity_policy_session.get_inputs()[0].name
        self.velocity_output_name = self.velocity_policy_session.get_outputs()[0].name
        self.motion_input_name = self.motion_policy_session.get_inputs()[0].name
        self.motion_output_name = self.motion_policy_session.get_outputs()[0].name
        
        self.get_logger().info(
            f"Velocity policy - Input: {self.velocity_input_name}, "
            f"Output: {self.velocity_output_name}"
        )
        self.get_logger().info(
            f"Motion policy - Input: {self.motion_input_name}, "
            f"Output: {self.motion_output_name}"
        )
        # Store ONNX paths for metadata reading
        self.velocity_onnx_path = velocity_onnx_path
        self.motion_onnx_path = motion_onnx_path
        self.get_logger().info("Initializing KV-Cache for Motion Policy...")
        
        self.motion_kv_input_name = None
        self.motion_kv_output_name = None
        self.motion_kv_shape = None
        self.motion_step_idx_input_name = None
        self.motion_kv_dtype = np.float32
        
        for node in self.motion_policy_session.get_inputs():
            name = node.name
            shape = node.shape
            node_type = node.type
            self.get_logger().info(f"Motion policy input: name={name}, shape={shape}, type={node_type}")
            if "obs" in name:
                self.motion_input_name = name
            elif "past_key_values" in name:
                self.motion_kv_input_name = name
                self.motion_kv_shape = shape
                if isinstance(node_type, str) and "float16" in node_type:
                    self.motion_kv_dtype = np.float16
            elif "step_idx" in name or name == "step_idx":
                self.motion_step_idx_input_name = name
            elif "current_pos" in name or name == "current_pos":
                self.motion_step_idx_input_name = name
            elif (
                self.motion_step_idx_input_name is None
                and isinstance(node_type, str)
                and "int64" in node_type
                and name not in (self.motion_input_name, self.motion_kv_input_name)
            ):
                self.motion_step_idx_input_name = name

        motion_outputs = self.motion_policy_session.get_outputs()
        action_output_name = None
        kv_output_name = None
        for node in motion_outputs:
            self.get_logger().info(f"Motion policy output: name={node.name}, shape={node.shape}, type={node.type}")
            if "present_key_values" in node.name:
                kv_output_name = node.name
            elif "actions" in node.name:
                action_output_name = node.name
        if action_output_name is None:
            for node in motion_outputs:
                if kv_output_name is not None and node.name == kv_output_name:
                    continue
                action_output_name = node.name
                break
        if action_output_name is None:
            action_output_name = motion_outputs[0].name
        self.motion_output_name = action_output_name
        self.motion_kv_output_name = kv_output_name
        if self.motion_kv_input_name is not None and self.motion_kv_output_name is None:
            self.get_logger().warn(
                "Motion policy has past_key_values input but no present_key_values output was found. "
                "KV cache will not update and transformer performance will degrade."
            )

        if self.motion_kv_input_name and self.motion_kv_shape:
            shape = [d if isinstance(d, int) else 1 for d in self.motion_kv_shape]
            
            self.motion_kv_cache = np.zeros(shape, dtype=self.motion_kv_dtype)
            self.motion_model_context_len = int(shape[3]) if len(shape) > 3 else 0
            self.motion_max_context_len = int(
                self.motion_config.get("algo", {})
                .get("config", {})
                .get("num_steps_per_env", 0)
            )
            if self.motion_max_context_len > 0 and self.motion_model_context_len > 0:
                self.motion_effective_context_len = min(
                    self.motion_max_context_len, self.motion_model_context_len
                )
            else:
                self.motion_effective_context_len = self.motion_model_context_len
            self.use_kv_cache = True
            self.get_logger().info(
                f"KV-Cache initialized with shape {shape} "
                f"(model_ctx={self.motion_model_context_len}, "
                f"effective_ctx={self.motion_effective_context_len})"
            )
        else:
            self.use_kv_cache = False
            self.motion_kv_cache = None
            self.motion_model_context_len = 0
            self.motion_effective_context_len = 0
            self.get_logger().warn("No KV-Cache inputs found in Motion Policy model!")
        self.get_logger().info("Dual policies loaded successfully")

    def load_model_config(self):
        """Load config.yaml from both velocity and motion model folders."""
        # Load velocity model config
        velocity_model_folder = self.config_yaml.velocity_tracking_model_folder
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
        motion_model_folder = self.config_yaml.motion_tracking_model_folder
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
        self.actor_place_holder_ndim = self._find_actor_place_holder_ndim()
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

    def _reset_motion_action_ema_filter(self) -> None:
        self._motion_filtered_actions_onnx = None

    def _apply_motion_action_ema_filter(
        self, raw_actions: np.ndarray
    ) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
        if not self.motion_action_ema_filter_enabled:
            return raw_actions.copy()

        if self._motion_filtered_actions_onnx is None:
            self._motion_filtered_actions_onnx = raw_actions.copy()
            return self._motion_filtered_actions_onnx.copy()

        alpha = float(self.motion_action_ema_filter_alpha)
        filtered_actions = (
            alpha * raw_actions
            + (1.0 - alpha) * self._motion_filtered_actions_onnx
        ).astype(np.float32, copy=False)
        self._motion_filtered_actions_onnx = filtered_actions.copy()
        return self._motion_filtered_actions_onnx.copy()

    def _build_dummy_input_from_onnx_node(self, node, fallback_last_dim: int | None = None):
        shape = list(getattr(node, "shape", []) or [])
        if not shape:
            shape = [1]
        inferred_shape = [_infer_onnx_dim(dim, default=1) for dim in shape]
        if fallback_last_dim is not None and len(inferred_shape) >= 2:
            last_dim = shape[-1]
            if not isinstance(last_dim, int) or last_dim <= 0:
                inferred_shape[-1] = int(fallback_last_dim)
        dtype = _infer_numpy_dtype_from_onnx_type(getattr(node, "type", "tensor(float)"))
        return np.zeros(inferred_shape, dtype=dtype)

    def _warmup_motion_policy(self, num_iters: int = 2) -> None:
        if self.motion_policy_session is None:
            return

        try:
            input_nodes = {node.name: node for node in self.motion_policy_session.get_inputs()}
            obs_node = input_nodes.get(self.motion_input_name, None)
            if obs_node is None:
                raise ValueError(
                    f"Motion warmup failed to find obs input '{self.motion_input_name}'."
                )

            motion_obs_dim = None
            try:
                motion_obs_dim = int(
                    self.motion_obs_builder.build_policy_obs().shape[0]
                )
            except Exception:
                motion_obs_dim = None

            obs_dummy = self._build_dummy_input_from_onnx_node(
                obs_node, fallback_last_dim=motion_obs_dim
            )
            output_names = [self.motion_output_name]
            if self.motion_kv_output_name:
                output_names.append(self.motion_kv_output_name)

            local_kv_cache = None
            if self.use_kv_cache and self.motion_kv_input_name is not None:
                if self.motion_kv_cache is not None:
                    local_kv_cache = np.zeros_like(self.motion_kv_cache)
                else:
                    shape = [
                        _infer_onnx_dim(dim, default=1)
                        for dim in (self.motion_kv_shape or [])
                    ]
                    local_kv_cache = np.zeros(shape, dtype=self.motion_kv_dtype)

            for warmup_step in range(max(1, int(num_iters))):
                input_feed = {self.motion_input_name: obs_dummy}
                if self.use_kv_cache and self.motion_kv_input_name is not None:
                    input_feed[self.motion_kv_input_name] = local_kv_cache
                if self.motion_step_idx_input_name is not None:
                    step_node = input_nodes.get(self.motion_step_idx_input_name, None)
                    step_dtype = np.int64
                    if step_node is not None:
                        step_dtype = _infer_numpy_dtype_from_onnx_type(
                            getattr(step_node, "type", "tensor(int64)")
                        )
                    input_feed[self.motion_step_idx_input_name] = np.array(
                        [warmup_step], dtype=step_dtype
                    )

                warmup_output = self.motion_policy_session.run(output_names, input_feed)
                if (
                    local_kv_cache is not None
                    and self.motion_kv_output_name
                    and len(warmup_output) > 1
                ):
                    local_kv_cache = warmup_output[1]

            if self.motion_kv_cache is not None:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            self.get_logger().info(
                f"[Warmup] Motion policy warmup completed ({max(1, int(num_iters))} iterations, KV cache kept clean)."
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
            self.config_yaml.motion_clip_dir,
        )
        
        self.get_logger().info(f"Looking for motion clip data in: {motion_clips_dir}")
        self.get_logger().info(f"Directory exists: {os.path.exists(motion_clips_dir)}")

        if not os.path.exists(motion_clips_dir):
            self.get_logger().warn(f"Motion clips directory not found: {motion_clips_dir}")
            return

        # Only collect .npz files
        motion_clip_files = [f for f in os.listdir(motion_clips_dir) if f.endswith(".npz")]
        motion_clip_files.sort()
        self.get_logger().info(
            f"Found {len(motion_clip_files)} motion clip files (.npz): {motion_clip_files}"
        )
        if not motion_clip_files:
            self.get_logger().warn(
                f"No motion clip files (.npz) found in directory: {motion_clips_dir}"
            )
            return

        # Load each .npz file
        self.all_motion_data = []
        self.motion_file_names = []
        for motion_clip_file in motion_clip_files:
            motion_path = os.path.join(motion_clips_dir, motion_clip_file)
            motion_data_dict = dict(np.load(motion_path, allow_pickle=True))

            self.all_motion_data.append(
                {
                    "dof_pos": motion_data_dict["ref_dof_pos"],
                    "dof_vel": motion_data_dict["ref_dof_vel"],
                    "global_translation": motion_data_dict[
                        "ref_global_translation"
                    ],
                    "global_rotation_quat": motion_data_dict[
                        "ref_global_rotation_quat"
                    ],
                    "global_velocity": motion_data_dict["ref_global_velocity"],
                    "global_angular_velocity": motion_data_dict["ref_global_angular_velocity"],
                    "n_frames": motion_data_dict["ref_dof_pos"].shape[0],
                }
            )
            self.motion_file_names.append(motion_clip_file)
        
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
        current_motion = self.all_motion_data[self.current_motion_clip_index]
        self.ref_dof_pos = current_motion["dof_pos"]
        self.ref_dof_vel = current_motion["dof_vel"]
        self.ref_raw_bodylink_pos = current_motion["global_translation"]
        self.ref_raw_bodylink_rot = current_motion["global_rotation_quat"]
        self.ref_global_velocity = current_motion["global_velocity"]
        self.ref_global_angular_velocity = current_motion["global_angular_velocity"]

        self.n_motion_frames = current_motion["n_frames"]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            raise ValueError("Motion clip is missing ref_dof_pos/ref_dof_vel arrays")
        if self.ref_raw_bodylink_pos is None or self.ref_raw_bodylink_rot is None:
            raise ValueError(
                "Motion clip is missing ref_global_translation/ref_global_rotation_quat arrays"
            )
        if int(self.ref_dof_pos.shape[1]) != int(len(self.dof_names_ref_motion)):
            raise ValueError(
                "ref_dof_pos DOF dimension mismatch: "
                f"ref_dof_pos.shape[1]={int(self.ref_dof_pos.shape[1])} "
                f"but len(dof_names_ref_motion)={int(len(self.dof_names_ref_motion))}"
            )
        if int(self.ref_raw_bodylink_pos.shape[1]) != int(
            len(self.motion_config.robot.body_names)
        ):
            raise ValueError(
                "ref_global_translation body dimension mismatch: "
                f"ref_raw_bodylink_pos.shape[1]={int(self.ref_raw_bodylink_pos.shape[1])} "
                f"but len(motion_config.robot.body_names)={int(len(self.motion_config.robot.body_names))}"
            )

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
            self.config_yaml.lowstate_topic,
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
            self.config_yaml.action_topic,
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
        # Only allow button operations when robot state is MOVE_TO_DEFAULT
        if robot_state == "MOVE_TO_DEFAULT":
            self.robot_state_ready = True
        elif robot_state == "ZERO_TORQUE":
            self.robot_state_ready = False
        elif robot_state == "EMERGENCY_STOP":
            self.robot_state_ready = False

    # =========== Properties ===========

    @property
    def robot_root_rot_quat_wxyz(self):
        return np.array(self._lowstate_msg.imu_state.quaternion, dtype=np.float32)

    @property
    def robot_root_ang_vel(self):
        return np.array(self._lowstate_msg.imu_state.gyroscope, dtype=np.float32)

    @property
    def robot_dof_pos_by_name(self):
        """Get DOF positions by name."""
        if self._lowstate_msg is None:
            return {}
        return {
            self.real_dof_names[i]: float(self._lowstate_msg.motor_state[i].q)
            for i in range(self.actions_dim)
        }

    @property
    def robot_dof_vel_by_name(self):
        """Get DOF velocities by name."""
        if self._lowstate_msg is None:
            return {}
        return {
            self.real_dof_names[i]: float(self._lowstate_msg.motor_state[i].dq)
            for i in range(self.actions_dim)
        }

    @property
    def ref_motion_frame_idx(self):
        return min(self.motion_frame_idx, self.n_motion_frames - 1)

    @property
    def ref_dof_pos_raw(self):
        if not self.latest_obs_flag:
            return self.ref_dof_pos[self.ref_motion_frame_idx]
        if self.n_fut_frames > 0 and self.external_fut_dof_pos_queue is not None:
            if self._prev_external_dof_pos is not None:
                return self._prev_external_dof_pos
            return self.external_fut_dof_pos_queue[0]
        if self.external_latest_obs is None:
            return self.ref_dof_pos[self.ref_motion_frame_idx]
        return self.external_latest_obs[0, :29]

    @property
    def ref_dof_vel_raw(self):
        if not self.latest_obs_flag:
            return self.ref_dof_vel[self.ref_motion_frame_idx]
        if self.n_fut_frames > 0 and self.external_fut_dof_pos_queue is not None:
            if self._prev_external_dof_vel is not None:
                return self._prev_external_dof_vel
            return self.external_fut_dof_vel_queue[0]
        if self.external_latest_obs is None:
            return self.ref_dof_vel[self.ref_motion_frame_idx]
        return self.external_latest_obs[0, 29:58]

    @property
    def ref_dof_pos_onnx_order(self):
        return self.ref_dof_pos_raw[self.ref_to_onnx]

    @property
    def ref_dof_vel_onnx_order(self):
        return self.ref_dof_vel_raw[self.ref_to_onnx]

    @property
    def ref_root_pos_raw(self):
        if not self.latest_obs_flag:
            return np.asarray(
                self.ref_raw_bodylink_pos[self.ref_motion_frame_idx, self.root_body_idx],
                dtype=np.float32,
            )
        if self.n_fut_frames > 0 and self.external_fut_root_pos_queue is not None:
            if self._prev_external_root_pos is not None:
                return self._prev_external_root_pos.astype(np.float32)
            return self.external_fut_root_pos_queue[0].astype(np.float32)
        if self.external_latest_obs is None:
            return np.zeros(3, dtype=np.float32)
        return self.external_latest_obs[0, 58:61].astype(np.float32)

    @property
    def root_body_idx(self):
        return 0

    @property
    def last_valid_ref_motion_frame_idx(self):
        return self.n_motion_frames - 1

    # =========== Policy Obeservation Methods ===========
    def _xyzw_to_wxyz(self, q_xyzw: np.ndarray) -> np.ndarray:
        """Convert quaternions from xyzw to wxyz order."""
        q_xyzw = np.asarray(q_xyzw, dtype=np.float32)
        if q_xyzw.shape[-1] != 4:
            raise ValueError(f"_xyzw_to_wxyz expects (...,4) but got shape {q_xyzw.shape}")
        # q_xyzw[..., 0:3] -> xyz, q_xyzw[..., 3:4] -> w
        w = q_xyzw[..., 3:4]
        xyz = q_xyzw[..., 0:3]
        return np.concatenate([w, xyz], axis=-1)

    def _standardize_quaternion_wxyz(self, q_wxyz: np.ndarray) -> np.ndarray:
        """Standardize quaternion sign so that w >= 0."""
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        if q_wxyz.shape[-1] != 4:
            raise ValueError(f"_standardize_quaternion_wxyz expects (...,4) but got shape {q_wxyz.shape}")
        mask = q_wxyz[..., 0:1] < 0.0
        q_wxyz = np.where(mask, -q_wxyz, q_wxyz)
        return q_wxyz

    def _quat_rotate_wxyz(self, q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        qvec = q_wxyz[..., 1:4]
        w = q_wxyz[..., 0:1]
        t = 2.0 * np.cross(qvec, v)
        return v + w * t + np.cross(qvec, t)

    def _quat_rotate_inv_wxyz(self, q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        n = int(np.prod(q_wxyz.shape[:-1])) if q_wxyz.ndim > 1 else 1
        q_conj = self._q_conj_buffer[:n].reshape(q_wxyz.shape)
        q_conj[..., 0] = q_wxyz[..., 0]
        q_conj[..., 1:4] = -q_wxyz[..., 1:4]
        return self._quat_rotate_wxyz(q_conj, v)

    def _quat_rotate_inv_wxyz_single(
        self, q_wxyz: np.ndarray, v: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        """Rotate one 3D vector by the inverse quaternion into a preallocated output."""
        q_conj = self._q_conj_buffer[0]
        q_conj[0] = q_wxyz[0]
        q_conj[1] = -q_wxyz[1]
        q_conj[2] = -q_wxyz[2]
        q_conj[3] = -q_wxyz[3]
        qvec = q_conj[1:4]
        w = q_conj[0]
        self._cross_t_buffer[:] = np.cross(qvec, v)
        self._cross_t_buffer *= 2.0
        out[:] = v + w * self._cross_t_buffer
        self._cross_t_buffer[:] = np.cross(qvec, self._cross_t_buffer)
        out += self._cross_t_buffer
        return out

    def _get_future_frame_indices(self) -> np.ndarray:
        frame_idx = self.ref_motion_frame_idx
        last_valid = self.last_valid_ref_motion_frame_idx
        np.minimum(
            frame_idx + self._future_frame_offsets,
            last_valid,
            out=self._future_frame_indices_buffer,
        )
        return self._future_frame_indices_buffer

    def _cache_fk_vr_for_obs(self):
        """Cache FK outputs used repeatedly during observation construction."""
        fk = getattr(self, "_fk_vr_out", None)
        if not getattr(self, "latest_obs_flag", False) or fk is None:
            self._use_fk_vr = False
            return
        self._use_fk_vr = True
        T = self.n_fut_frames_int
        rb = self.root_body_idx
        np.copyto(self._fk_vel_0_root, fk["global_velocity"][0, 0, rb])
        np.copyto(self._fk_angvel_0_root, fk["global_angular_velocity"][0, 0, rb])
        np.copyto(self._fk_quat_0_root, fk["global_rotation_quat"][0, 0, rb])
        self._fk_quat_0_root_wxyz[0] = self._fk_quat_0_root[3]
        self._fk_quat_0_root_wxyz[1:4] = self._fk_quat_0_root[:3]
        if self._fk_quat_0_root_wxyz[0] < 0.0:
            self._fk_quat_0_root_wxyz *= -1.0
        trans_0 = fk["global_translation"][0, 0]
        if self._fk_trans_0 is None or self._fk_trans_0.shape != trans_0.shape:
            self._fk_trans_0 = np.empty_like(trans_0)
        np.copyto(self._fk_trans_0, trans_0)
        if T > 0:
            np.copyto(self._fk_vel_fut[:T], fk["global_velocity"][0, 1 : 1 + T, rb])
            np.copyto(self._fk_angvel_fut[:T], fk["global_angular_velocity"][0, 1 : 1 + T, rb])
            np.copyto(self._fk_quat_fut[:T], fk["global_rotation_quat"][0, 1 : 1 + T, rb])
            self._fk_quat_fut_wxyz[:T, 0] = self._fk_quat_fut[:T, 3]
            self._fk_quat_fut_wxyz[:T, 1:4] = self._fk_quat_fut[:T, :3]
            neg = self._fk_quat_fut_wxyz[:T, 0] < 0.0
            self._fk_quat_fut_wxyz[:T][neg] *= -1.0
            trans_fut = fk["global_translation"][0, 1 : 1 + T]
            if self._fk_trans_fut is None or self._fk_trans_fut.shape != trans_fut.shape:
                self._fk_trans_fut = np.empty_like(trans_fut)
            np.copyto(self._fk_trans_fut, trans_fut)
            self._fill_vr_base_linvel_angvel_fut()

    def _fill_vr_base_linvel_angvel_fut(self):
        """Rotate future linear and angular velocity buffers in one pass."""
        T = self.n_fut_frames_int
        if T <= 0:
            return
        vel_T6 = self._vel_fut_T6[:T]
        vel_T6[:, :3] = self._fk_vel_fut[:T]
        vel_T6[:, 3:6] = self._fk_angvel_fut[:T]
        q = self._fk_quat_fut_wxyz[:T]
        q_conj = self._q_conj_buffer[:T].reshape(T, 4)
        q_conj[:, 0] = q[:, 0]
        q_conj[:, 1:4] = -q[:, 1:4]
        qvec = q_conj[:, 1:4]
        w = q_conj[:, 0:1]
        rt = self._rot_t_buffer[:T]
        rc = self._rot_cross_buffer[:T]
        rt[:] = np.cross(qvec, vel_T6[:, :3])
        rt *= 2.0
        rc[:] = np.cross(qvec, rt)
        self._base_linvel_fut_buffer[:T] = vel_T6[:, :3] + w * rt + rc
        rt[:] = np.cross(qvec, vel_T6[:, 3:6])
        rt *= 2.0
        rc[:] = np.cross(qvec, rt)
        self._base_angvel_fut_buffer[:T] = vel_T6[:, 3:6] + w * rt + rc

    def _prepare_vr_fk_tensors(
        self,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray,
        n_fut: int,
    ):
        """Fill preallocated FK input buffers and return torch views without reallocation."""
        if (
            n_fut <= 0
            or self._fk_root_pos_seq_np is None
            or self._fk_root_rot_seq_np is None
            or self._fk_dof_pos_seq_np is None
        ):
            raise ValueError("VR FK sequence buffers are not initialized")

        np.copyto(self._fk_root_pos_seq_np[0, 0], cur_root_pos)
        np.copyto(self._fk_root_rot_seq_np[0, 0], cur_root_rot)
        np.copyto(self._fk_dof_pos_seq_np[0, 0], cur_dof_pos)
        np.copyto(
            self._fk_root_pos_seq_np[0, 1 : 1 + n_fut],
            self.external_fut_root_pos_queue[:n_fut],
        )
        np.copyto(
            self._fk_root_rot_seq_np[0, 1 : 1 + n_fut],
            self.external_fut_root_rot_queue[:n_fut],
        )
        np.copyto(
            self._fk_dof_pos_seq_np[0, 1 : 1 + n_fut],
            self.external_fut_dof_pos_queue[:n_fut],
        )
        return (
            self._fk_root_pos_seq_tensor,
            self._fk_root_rot_seq_tensor,
            self._fk_dof_pos_seq_tensor,
        )

    def _get_future_root_quat_wxyz(self) -> np.ndarray:
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable; future_root_quat_wxyz will return zeros."
            )
            return self._future_root_quat_wxyz_buffer

        fut_idx = self._get_future_frame_indices()
        q_root_xyzw = np.asarray(
            self.ref_raw_bodylink_rot[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        q_root_wxyz = self._future_root_quat_wxyz_buffer
        q_root_wxyz[:, 0] = q_root_xyzw[:, 3]
        q_root_wxyz[:, 1] = q_root_xyzw[:, 0]
        q_root_wxyz[:, 2] = q_root_xyzw[:, 1]
        q_root_wxyz[:, 3] = q_root_xyzw[:, 2]
        neg_mask = q_root_wxyz[:, 0] < 0.0
        q_root_wxyz[neg_mask] *= -1.0
        return self._future_root_quat_wxyz_buffer

    def _get_ref_keybody_indices(self, term_name: str) -> np.ndarray:
        keybody_idxs = self._keybody_indices_by_term_name.get(term_name, None)
        if keybody_idxs is None:
            raise ValueError(
                f"Keybody indices for term '{term_name}' were not cached. "
                "Ensure the term exists in motion policy obs and cache is initialized."
            )
        return keybody_idxs

    def _get_obs_actor_velocity_command(self):
        return self._get_obs_velocity_command()

    def _get_obs_actor_projected_gravity(self):
        return self._get_obs_projected_gravity()

    def _get_obs_actor_rel_robot_root_ang_vel(self):
        return self._get_obs_rel_robot_root_ang_vel()

    def _get_obs_actor_dof_pos(self):
        return self._get_obs_dof_pos()

    def _get_obs_actor_dof_vel(self):
        return self._get_obs_dof_vel()
        
    def _get_obs_actor_last_action(self):
        return self._get_obs_last_action()

    def _get_obs_actor_ref_gravity_projection_cur(self):
        return self._get_obs_ref_gravity_projection_cur()

    def _get_obs_actor_ref_gravity_projection_fut(self):
        return self._get_obs_ref_gravity_projection_fut()

    def _get_obs_actor_ref_base_linvel_cur(self):
        return self._get_obs_ref_base_linvel_cur()

    def _get_obs_actor_ref_base_linvel_fut(self):
        return self._get_obs_ref_base_linvel_fut()

    def _get_obs_actor_ref_base_angvel_cur(self):
        return self._get_obs_ref_base_angvel_cur()

    def _get_obs_actor_ref_base_angvel_fut(self):
        return self._get_obs_ref_base_angvel_fut()

    def _get_obs_actor_ref_dof_pos_cur(self):
        return self._get_obs_ref_dof_pos_cur()

    def _get_obs_actor_ref_dof_pos_fut(self):
        return self._get_obs_ref_dof_pos_fut()

    def _get_obs_actor_ref_root_height_cur(self):
        return self._get_obs_ref_root_height_cur()

    def _get_obs_actor_ref_root_height_fut(self):
        return self._get_obs_ref_root_height_fut()

    def _get_obs_actor_ref_keybody_rel_pos_cur(self):
        return self._get_obs_ref_keybody_rel_pos_cur()

    def _get_obs_actor_ref_keybody_rel_pos_fut(self):
        return self._get_obs_ref_keybody_rel_pos_fut()



    def _get_obs_velocity_command(self):
        """Get velocity command observation (reuses pre-allocated array)."""
        self._velocity_cmd_obs[1] = self.vx
        self._velocity_cmd_obs[2] = self.vy
        self._velocity_cmd_obs[3] = self.vyaw
        self._velocity_cmd_obs[0] = float(
            np.linalg.norm(self._velocity_cmd_obs[1:4]) > 0.1
        )
        return self._velocity_cmd_obs

    def _get_obs_projected_gravity(self):
        return get_gravity_orientation(self.robot_root_rot_quat_wxyz)

    def _get_obs_rel_robot_root_ang_vel(self):
        return self.robot_root_ang_vel

    def _get_obs_dof_pos(self):
        """Get DOF position observation (pre-allocated buffer + index lookup, no dict/list)."""
        if self._lowstate_msg is None:
            return self._dof_pos_obs_buffer[: len(self.motion_dof_names_onnx)]
        if self.current_policy_mode == "motion":
            buf = self._dof_pos_obs_buffer
            ms = self._lowstate_msg.motor_state
            def_angles = self.motion_default_angles_onnx
            for i, ri in enumerate(self.motion_dof_real_indices):
                buf[i] = ms[ri].q - def_angles[i]
            return buf[: len(self.motion_dof_names_onnx)]
        def_angles = self.velocity_default_angles_onnx
        for i, ri in enumerate(self.velocity_dof_real_indices):
            self._dof_pos_obs_buffer[i] = (
                self._lowstate_msg.motor_state[ri].q - def_angles[i]
            )
        return self._dof_pos_obs_buffer[: len(self.velocity_dof_names_onnx)]

    def _get_obs_dof_vel(self):
        """Get DOF velocity observation (pre-allocated buffer + index lookup, no dict/list)."""
        if self._lowstate_msg is None:
            return self._dof_vel_obs_buffer[: len(self.motion_dof_names_onnx)]
        if self.current_policy_mode == "motion":
            buf = self._dof_vel_obs_buffer
            ms = self._lowstate_msg.motor_state
            for i, ri in enumerate(self.motion_dof_real_indices):
                buf[i] = ms[ri].dq
            return buf[: len(self.motion_dof_names_onnx)]
        for i, ri in enumerate(self.velocity_dof_real_indices):
            self._dof_vel_obs_buffer[i] = self._lowstate_msg.motor_state[ri].dq
        return self._dof_vel_obs_buffer[: len(self.velocity_dof_names_onnx)]

    def _get_obs_last_action(self):
        return self.actions_onnx.copy()

    def _get_obs_ref_motion_states(self):
        return np.concatenate(
            [self.ref_dof_pos_onnx_order, self.ref_dof_vel_onnx_order]
        )

    def _get_obs_ref_dof_pos_fut(self):
        """Get future DOF position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "latest_obs_flag", False):
            if (
                getattr(self, "external_fut_dof_pos_queue", None) is not None
                and self.external_fut_dof_pos_queue.shape[0] >= T
            ):
                pos_fut = self._pos_fut_buffer
                pos_fut[:, :] = self.external_fut_dof_pos_queue[:T].T
                pos_fut_onnx = pos_fut[self.ref_to_onnx, :].transpose(1, 0)  # [N, T]
                return pos_fut_onnx.reshape(-1).astype(np.float32)
            return np.zeros(self.num_actions * T, dtype=np.float32)
        if not hasattr(self, "ref_dof_pos") or self.ref_dof_pos is None:
            self.get_logger().warn(
                "[VR] ref_dof_pos is unavailable and latest_obs is not active; returning zeros for ref_dof_pos_fut."
            )
            return np.zeros(self.num_actions * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        pos_fut = self._pos_fut_buffer
        pos_fut[:, :] = self.ref_dof_pos[fut_idx].T
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :].transpose(1, 0)  # [N, T]
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_height_fut(self):
        """Get future root height observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.latest_obs_flag and self.external_fut_root_pos_queue is not None:
            root_pos_fut = self.external_fut_root_pos_queue[:, 2].astype(np.float32)
            return root_pos_fut.reshape(-1)
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_root_height_fut."
            )
            return np.zeros(T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        h_fut = self._h_fut_buffer
        h_fut[0, :] = self.ref_raw_bodylink_pos[fut_idx, self.root_body_idx, 2]
        return h_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_pos_fut(self):
        """Get future root position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.latest_obs_flag and self.external_fut_root_pos_queue is not None:
            pos_fut = self.external_fut_root_pos_queue.astype(np.float32)
            return pos_fut.reshape(-1).astype(np.float32)
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_root_pos_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        pos_fut = self._root_pos_fut_buffer
        pos_fut[:, :] = self.ref_raw_bodylink_pos[fut_idx, self.root_body_idx, :]
        return pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_cur(self):
        return self.ref_dof_pos_onnx_order

    def _get_obs_ref_dof_vel_cur(self):
        return self.ref_dof_vel_onnx_order

    def _get_obs_ref_root_height_cur(self):
        if not self.latest_obs_flag:
            return self.ref_raw_bodylink_pos[
                self.ref_motion_frame_idx, self.root_body_idx, 2
            ]
        return float(self.ref_root_pos_raw[2])

    def _get_obs_ref_root_pos_cur(self):
        return self.ref_root_pos_raw.astype(np.float32)

    def _get_obs_ref_gravity_projection_cur(self):
        if getattr(self, "_use_fk_vr", False):
            return get_gravity_orientation(self._fk_quat_0_root_wxyz)
        if getattr(self, "latest_obs_flag", False) and getattr(
            self, "external_latest_obs", None
        ) is not None:
            q_root_wxyz = self.external_latest_obs[0, 61:65].astype(np.float32)
            q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
            return get_gravity_orientation(q_root_wxyz)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for gravity_projection_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        return get_gravity_orientation(q_root_wxyz)

    def _get_obs_ref_gravity_projection_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            q_root_wxyz = self._fk_quat_fut_wxyz[:T]
            gravity_fut = self._gravity_fut_buffer
            qw = q_root_wxyz[:, 0]
            qx = q_root_wxyz[:, 1]
            qy = q_root_wxyz[:, 2]
            qz = q_root_wxyz[:, 3]
            gravity_fut[:, 0] = 2.0 * (-qz * qx + qw * qy)
            gravity_fut[:, 1] = -2.0 * (qz * qy + qw * qx)
            gravity_fut[:, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)
            return gravity_fut.reshape(-1).astype(np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_gravity_projection_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        q_root_wxyz = self._get_future_root_quat_wxyz()
        gravity_fut = self._gravity_fut_buffer
        qw = q_root_wxyz[:, 0]
        qx = q_root_wxyz[:, 1]
        qy = q_root_wxyz[:, 2]
        qz = q_root_wxyz[:, 3]
        gravity_fut[:, 0] = 2.0 * (-qz * qx + qw * qy)
        gravity_fut[:, 1] = -2.0 * (qz * qy + qw * qx)
        gravity_fut[:, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return gravity_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_linvel_cur(self):
        if getattr(self, "_use_fk_vr", False):
            self._quat_rotate_inv_wxyz_single(
                self._fk_quat_0_root_wxyz, self._fk_vel_0_root, self._rotated_3vec_buffer
            )
            return self._rotated_3vec_buffer
        if getattr(self, "latest_obs_flag", False) and getattr(
            self, "external_latest_obs", None
        ) is not None:
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_global_velocity") or self.ref_global_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        v_root_w = np.asarray(
            self.ref_global_velocity[self.ref_motion_frame_idx, self.root_body_idx],
            dtype=np.float32,
        )
        v_root = self._quat_rotate_inv_wxyz(q_root_wxyz, v_root_w)
        return np.asarray(v_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_linvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            return self._base_linvel_fut_buffer[:T].reshape(-1).astype(np.float32)

        if not hasattr(self, "ref_global_velocity") or self.ref_global_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        v_root_w = np.asarray(
            self.ref_global_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_linvel_fut = self._base_linvel_fut_buffer
        base_linvel_fut[:, :] = self._quat_rotate_inv_wxyz(q_root_wxyz, v_root_w)
        return base_linvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_angvel_cur(self):
        if getattr(self, "_use_fk_vr", False):
            self._quat_rotate_inv_wxyz_single(
                self._fk_quat_0_root_wxyz,
                self._fk_angvel_0_root,
                self._rotated_angvel_cur_buffer,
            )
            return self._rotated_angvel_cur_buffer
        if getattr(self, "latest_obs_flag", False) and getattr(
            self, "external_latest_obs", None
        ) is not None:
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_global_angular_velocity") or self.ref_global_angular_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[self.ref_motion_frame_idx, self.root_body_idx],
            dtype=np.float32,
        )
        w_root = self._quat_rotate_inv_wxyz(q_root_wxyz, w_root_w)
        return np.asarray(w_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_angvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            return self._base_angvel_fut_buffer[:T].reshape(-1).astype(np.float32)

        if not hasattr(self, "ref_global_angular_velocity") or self.ref_global_angular_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_angvel_fut = self._base_angvel_fut_buffer
        base_angvel_fut[:, :] = self._quat_rotate_inv_wxyz(q_root_wxyz, w_root_w)
        return base_angvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_keybody_rel_pos_cur(self):
        if getattr(self, "_use_fk_vr", False) and self._fk_trans_0 is not None:
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            if not self._root_only_fk_has_required_keybodies(keybody_idxs):
                return np.zeros(3 * n_keybodies, dtype=np.float32)
            root_pos = self._fk_trans_0[self.root_body_idx]
            keybody_pos = self._fk_trans_0[keybody_idxs]
            rel_pos_w = keybody_pos - root_pos[None, :]
            rel_pos_root = self._quat_rotate_inv_wxyz(self._fk_quat_0_root_wxyz, rel_pos_w)
            return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

        if getattr(self, "latest_obs_flag", False) and getattr(
            self, "external_latest_obs", None
        ) is not None:
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros(0, dtype=np.float32)

        frame_idx = self.ref_motion_frame_idx
        ref_body_global_pos = np.asarray(self.ref_raw_bodylink_pos[frame_idx], dtype=np.float32)
        ref_root_global_pos = ref_body_global_pos[self.root_body_idx]
        q_root_xyzw = self.ref_raw_bodylink_rot[frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)

        rel_pos_w = ref_body_global_pos[keybody_idxs] - ref_root_global_pos[None, :]
        rel_pos_root = self._quat_rotate_inv_wxyz(q_root_wxyz, rel_pos_w)
        return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

    def _get_obs_ref_keybody_rel_pos_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False) and self._fk_trans_fut is not None:
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_fut")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            if not self._root_only_fk_has_required_keybodies(keybody_idxs):
                return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)
            ref_body = self._fk_trans_fut[:T]  # (T, num_bodies, 3)
            ref_root = ref_body[:, self.root_body_idx, :]  # (T, 3)
            if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
                self._keybody_rel_pos_fut_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
                self._keybody_rel_pos_w_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
            elif (
                self._keybody_rel_pos_w_buffer is None
                or self._keybody_rel_pos_w_buffer.shape[0] < T
                or self._keybody_rel_pos_w_buffer.shape[1] != n_keybodies
            ):
                self._keybody_rel_pos_w_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
            rel_pos_fut = self._keybody_rel_pos_fut_buffer
            np.subtract(
                ref_body[:, keybody_idxs, :],
                ref_root[:, None, :],
                out=self._keybody_rel_pos_w_buffer[:T, :n_keybodies, :],
            )
            rel_pos_fut[:, :, :] = self._quat_rotate_inv_wxyz(
                self._fk_quat_fut_wxyz[:T, None, :],
                self._keybody_rel_pos_w_buffer[:T, :n_keybodies, :],
            )
            return rel_pos_fut.reshape(-1).astype(np.float32)
        keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_fut")
        n_keybodies = int(keybody_idxs.shape[0])
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)

        if n_keybodies == 0:
            return np.zeros((T, 0), dtype=np.float32).reshape(-1)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        ref_body_global_pos = np.asarray(self.ref_raw_bodylink_pos[fut_idx], dtype=np.float32)
        ref_root_global_pos = ref_body_global_pos[:, self.root_body_idx, :]
        rel_pos_w = ref_body_global_pos[:, keybody_idxs, :] - ref_root_global_pos[:, None, :]
        if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
            self._keybody_rel_pos_fut_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
        rel_pos_fut = self._keybody_rel_pos_fut_buffer
        rel_pos_fut[:, :, :] = self._quat_rotate_inv_wxyz(q_root_wxyz[:, None, :], rel_pos_w)
        return rel_pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_place_holder(self):
        return np.zeros(self.actor_place_holder_ndim, dtype=np.float32)

    # =========== Policy Obeservation Methods ===========

    def _warmup_fk_for_vr(self):
        """Run one FK warmup step when entering VR motion mode."""
        try:
            if (
                getattr(self, "fk", None) is None
                or not getattr(self, "fk_initialized", False)
            ):
                return
            if getattr(self, "external_latest_obs", None) is None:
                return
            if getattr(self, "external_fut_dof_pos_queue", None) is None:
                return

            n_fut = int(getattr(self, "n_fut_frames", 0))
            if (
                n_fut <= 0
                or self.external_fut_root_pos_queue is None
                or self.external_fut_root_rot_queue is None
            ):
                return

            latest = self.external_latest_obs[0]
            cur_root_pos = latest[58:61]
            cur_root_rot = latest[61:65]
            cur_dof_pos = latest[0:29]
            root_pos_tensor, root_rot_tensor, dof_pos_tensor = (
                self._prepare_vr_fk_tensors(
                    cur_root_pos=cur_root_pos,
                    cur_root_rot=cur_root_rot,
                    cur_dof_pos=cur_dof_pos,
                    n_fut=n_fut,
                )
            )

            fk_out = self.fk(
                root_pos=root_pos_tensor,
                root_quat=root_rot_tensor,
                dof_pos=dof_pos_tensor,
                fps=float(1.0 / self.dt),
                quat_format="wxyz",
                vel_smoothing_sigma=0.0,
                compute_velocity=False,
            )
            self._fk_vr_out = {
                k: v.detach().cpu().numpy() for k, v in fk_out.items()
            }
        except Exception as e:
            self.get_logger().warn(f"[VR] FK warmup failed, fallback to zeros: {e}")

    def _low_state_callback(self, ls_msg: LowState):
        """Process low-level robot state and remote controller input.

        Main callback that handles:
        - Remote controller input processing
        - Motion selection based on button presses
        - Safety state checking
        - Velocity command extraction

        Motion Button Mapping:
        - A button: Enable policy (defaults to velocity mode)
        - B button: Switch from velocity to motion mode
        - Y button: Switch from motion back to velocity mode
        - UP/DOWN/LEFT/RIGHT: Motion clip selection (only in velocity tracking mode)

        Args:
            ls_msg: LowState message containing robot sensor data and remote controller input
        """
        self._lowstate_msg = ls_msg
        self.remote_controller.set(ls_msg.wireless_remote)

        # A button: Toggle policy enable state (default to velocity mode)
        if (
            self._is_button_pressed(KeyMap.A) and self.robot_state_ready
        ):
            self.policy_enabled = True
            self.current_policy_mode = "velocity"  # Default to velocity mode
            self.latest_obs_flag = False
            self._reset_motion_action_ema_filter()
            self._reset_counter()
            if hasattr(self, "use_kv_cache") and self.use_kv_cache:
                self.motion_kv_cache.fill(0)
            self.motion_step_idx = 0
            # Initialize with velocity model's default angles
            self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
            self.target_dof_pos_onnx = self.velocity_default_angles_onnx.copy()
            # Publish velocity model's control parameters (kps/kds)
            self._publish_control_params()
            self.get_logger().info(
                f"Policy enabled in {self.current_policy_mode} tracking mode"
            )

        # B button: Switch to motion tracking mode (only when policy is enabled)
        if (
            self._is_button_pressed(KeyMap.B)
            and self.robot_state_ready
            and self.policy_enabled
            and self.current_policy_mode == "velocity"  # Only allow switch from velocity mode
        ):
            vr_data_available = bool(
                getattr(self, "enable_teleop_reference", True)
                and getattr(self, "external_obs_received", False)
                and getattr(self, "external_latest_obs", None) is not None
            )
            vr_ready = self._is_vr_ready_for_motion()
            if (
                self.enable_teleop_reference
                and self.require_vr_data_for_motion
                and not vr_ready
            ):
                self.get_logger().warn(
                    "require_vr_data_for_motion=True but the VR queue is not ready yet; staying in velocity mode."
                )
            else:
                # Don't automatically switch to next motion clip - keep current selection
                if hasattr(self, "all_motion_data") and self.all_motion_data:
                    # Load the current motion clip data (don't change current_motion_clip_index)
                    self._load_current_motion()

                self.current_policy_mode = "motion"
                self._reset_motion_action_ema_filter()
                self._reset_counter()
                if hasattr(self, "use_kv_cache") and self.use_kv_cache:
                    self.motion_kv_cache.fill(0)
                    self.get_logger().info("Motion KV-Cache reset.")

                self.motion_step_idx = 0
                self.get_logger().info("Motion Step Index reset to 0.")

                # Clear any pending actions to prevent conflicts between policies
                # Use motion model's default angles
                self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
                self.target_dof_pos_onnx = self.motion_default_angles_onnx.copy()

                # Publish motion model's control parameters (kps/kds)
                self._publish_control_params()

                self.latest_obs_flag = bool(vr_data_available)
                source_mode = "ZMQ latest_obs" if self.latest_obs_flag else "offline motion"

                self.get_logger().info(
                    f"Switched to motion tracking mode ({source_mode}) - motion clip index: {self.current_motion_clip_index}"
                )
                if self.latest_obs_flag:
                    self.get_logger().info("[VR] Reference trajectory source: ZMQ latest_obs")
                    self._warmup_fk_for_vr()
                self.motion_in_progress = True

        if (
            self._is_button_pressed(KeyMap.Y)
            and self.robot_state_ready
            and self.policy_enabled
            and self.current_policy_mode == "motion"  # Only allow switch from motion mode
        ):
            self._switch_to_velocity_mode()

        # Get velocity commands only in velocity tracking mode
        if self.current_policy_mode == "velocity":
            self.vx, self.vy, self.vyaw = self.remote_controller.get_velocity_commands()
        else:
            # In motion tracking mode, ignore joystick input
            self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0

        # Handle motion clip selection in velocity tracking mode (UP/DOWN/LEFT/RIGHT)
        if (
            self.current_policy_mode == "velocity"
            and self.policy_enabled
            and self.robot_state_ready
        ):
            # Handle motion clip selection with UP/DOWN/LEFT/RIGHT buttons
            if self._is_button_pressed(KeyMap.up):
                # Switch to previous motion clip
                if hasattr(self, "all_motion_data") and self.all_motion_data:
                    self.current_motion_clip_index = (
                        self.current_motion_clip_index - 1
                    ) % len(self.all_motion_data)
                    self.get_logger().info(
                        f"Selected previous motion clip: "
                        f"{self.motion_file_names[self.current_motion_clip_index]}"
                    )
            elif self._is_button_pressed(KeyMap.down):
                # Switch to next motion clip
                if hasattr(self, "all_motion_data") and self.all_motion_data:
                    self.current_motion_clip_index = (
                        self.current_motion_clip_index + 1
                    ) % len(self.all_motion_data)
                    self.get_logger().info(
                        f"Selected next motion clip: "
                        f"{self.motion_file_names[self.current_motion_clip_index]}"
                    )
            elif self._is_button_pressed(KeyMap.left):
                # Select first motion clip
                if hasattr(self, "all_motion_data") and self.all_motion_data:
                    self.current_motion_clip_index = 0
                    self.get_logger().info(
                        f"Selected first motion clip: "
                        f"{self.motion_file_names[self.current_motion_clip_index]}"
                    )
            elif self._is_button_pressed(KeyMap.right):
                # Select last motion clip
                if hasattr(self, "all_motion_data") and self.all_motion_data:
                    self.current_motion_clip_index = len(self.all_motion_data) - 1
                    self.get_logger().info(
                        f"Selected last motion clip: "
                        f"{self.motion_file_names[self.current_motion_clip_index]}"
                    )

    def run(self):
        """Main execution loop for policy inference and action publication."""
        # Only run if setup is completed
        if not hasattr(self, '_setup_completed') or not self._setup_completed:
            return
        t_loop_start = time.perf_counter()
        now = time.time()
        t_io = time.perf_counter()
        buf = getattr(self, "_ros_latest_obs_buffer", None)
        if buf is not None:
            self._ros_latest_obs_buffer = None
            frame_idx, obs_arr = buf
            if frame_idx is not None:
                self._npz_replay_frame_index = frame_idx
            self._store_external_latest_obs(obs_arr[None, :])
        self._poll_zmq_latest_obs()
        if getattr(self, "current_policy_mode", None) == "motion":
            if self._last_vr_status_log_time is None:
                self._last_vr_status_log_time = now
            elif now - self._last_vr_status_log_time >= 5.0:
                vr_available = bool(
                    getattr(self, "external_obs_received", False)
                    and getattr(self, "external_latest_obs", None) is not None
                )
                queue_stats = self._latest_obs_buffer.get_queue_stats()
                freq = queue_stats.get("expected_freq")
                if vr_available:
                    self.get_logger().info(
                        "[VR-STATUS] ZMQ latest_obs streaming | "
                        f"buffer_size={queue_stats['queue_size']} "
                        f"expected_freq={freq:.1f}Hz" if freq else
                        f"buffer_size={queue_stats['queue_size']} expected_freq=unknown"
                    )
                else:
                    self.get_logger().warn(
                        "[VR-STATUS] No new ZMQ latest_obs received in the last 5 seconds; using offline reference or the last buffered VR state."
                    )
                self._last_vr_status_log_time = now
        if (
            getattr(self, "require_vr_data_for_motion", False)
            and getattr(self, "policy_enabled", False)
            and not getattr(self, "_vr_ready_logged", False)
            and self._is_vr_ready_for_motion()
        ):
            self.get_logger().info(
                f"[VR] VR queue is ready for motion mode (seen_frames={int(getattr(self, '_external_seen_frames', 0))}, "
                f"n_fut={int(getattr(self, 'n_fut_frames', 0) or 0)}, "
                f"delay={int(getattr(self, 'zmq_jitter_delay_frames', 0) or 0)})"
            )
            self._vr_ready_logged = True
        self._publish_latest_obs()
        io_ms = self._timing_ms(t_io)
        policy_timing = self._run_without_profiling()
        _run_elapsed = 0.0
        if policy_timing is not None:
            _run_elapsed = float(policy_timing.get("policy_total_ms", 0.0)) / 1000.0
        if (
            getattr(self, "current_policy_mode", None) == "motion"
            and getattr(self, "latest_obs_flag", False)
            and _run_elapsed > 0.5
            and not getattr(self, "_vr_cold_start_logged", False)
        ):
            self._vr_cold_start_logged = True
            self.get_logger().info(
                "[VR] The first motion step is a cold start (FK/ONNX initialization) and may take about 1 second."
            )
        if (
            getattr(self, "current_policy_mode", None) == "motion"
            and getattr(self, "latest_obs_flag", False)
            and _run_elapsed > 1.15 * self.dt
            and _run_elapsed <= 0.5
        ):
            self._policy_slow_count = getattr(self, "_policy_slow_count", 0) + 1
            if self._policy_slow_count == 1 or self._policy_slow_count % 50 == 0:
                self.get_logger().warn(
                    f"[VR] Policy step latency {_run_elapsed*1000:.1f} ms exceeds the target {self.dt*1000:.1f} ms. "
                    f"Estimated /humanoid/action rate: {1.0/_run_elapsed:.1f} Hz (target {1.0/self.dt:.0f} Hz). "
                    "The main bottleneck is usually FK or ONNX inference; if the system settles near 30 Hz, consider setting policy_freq to 30."
                )
        if policy_timing is not None:
            sample = dict(policy_timing)
            sample["io_ms"] = io_ms
            sample["loop_total_ms"] = self._timing_ms(t_loop_start)
            self._record_timing_sample(sample)

    def _read_onnx_metadata(self, onnx_model_path: str) -> dict:
        """Read model metadata from ONNX file and parse into Python types."""
        model = onnx.load(str(onnx_model_path))
        meta = {p.key: p.value for p in model.metadata_props}

        def _parse_floats(csv_str: str):
            return np.array(
                [float(x) for x in csv_str.split(",") if x != ""],
                dtype=np.float32,
            )

        result = {}
        result["action_scale"] = _parse_floats(meta["action_scale"])
        result["kps"] = _parse_floats(meta["joint_stiffness"])
        result["kds"] = _parse_floats(meta["joint_damping"])
        result["default_joint_pos"] = _parse_floats(meta["default_joint_pos"])
        result["joint_names"] = [x for x in meta["joint_names"].split(",") if x != ""]
        return result

    def _store_external_latest_obs(self, arr: np.ndarray):
        """Store latest_obs and maintain the current/future frame queues."""
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.latest_obs_expected_dim:
            self.get_logger().warn(
                f"Received latest_obs dim={arr.shape[1]}, expected >= {self.latest_obs_expected_dim}"
            )
            return
        clipped = arr[:, : self.latest_obs_expected_dim].astype(np.float32, copy=False)
        current_time = time.time()
        self.external_latest_obs = clipped
        self.external_obs_received = True
        self.last_external_obs_time = current_time
        self._external_seen_frames = int(getattr(self, "_external_seen_frames", 0)) + 1

        latest_root_pos = clipped[0, 58:61]
        latest_root_rot = clipped[0, 61:65]
        latest_dof_pos = clipped[0, :29]
        latest_dof_vel = clipped[0, 29:58]

        if self.n_fut_frames > 0 and self.external_fut_dof_pos_queue is not None:
            raw_idx = getattr(self, "_npz_replay_frame_index", None)
            try:
                latest_frame_idx = int(raw_idx) if raw_idx is not None else -1
            except Exception:
                latest_frame_idx = -1

            if self._prev_external_dof_pos is None:
                self._prev_external_dof_pos = np.empty_like(self.external_fut_dof_pos_queue[0])
                self._prev_external_dof_vel = np.empty_like(self.external_fut_dof_vel_queue[0])
                self._prev_external_root_pos = np.empty_like(self.external_fut_root_pos_queue[0])
                if self.external_fut_root_rot_queue is not None:
                    self._prev_external_root_rot = np.empty_like(
                        self.external_fut_root_rot_queue[0]
                    )
            np.copyto(self._prev_external_dof_pos, self.external_fut_dof_pos_queue[0])
            np.copyto(self._prev_external_dof_vel, self.external_fut_dof_vel_queue[0])
            np.copyto(self._prev_external_root_pos, self.external_fut_root_pos_queue[0])
            if self.external_fut_root_rot_queue is not None:
                np.copyto(self._prev_external_root_rot, self.external_fut_root_rot_queue[0])
            if self.external_fut_frame_idx_queue is not None:
                try:
                    self._prev_external_frame_idx = int(self.external_fut_frame_idx_queue[0])
                except Exception:
                    self._prev_external_frame_idx = -1

            self.external_fut_dof_pos_queue[:-1] = self.external_fut_dof_pos_queue[1:]
            self.external_fut_dof_pos_queue[-1] = latest_dof_pos
            self.external_fut_dof_vel_queue[:-1] = self.external_fut_dof_vel_queue[1:]
            self.external_fut_dof_vel_queue[-1] = latest_dof_vel
            self.external_fut_root_pos_queue[:-1] = self.external_fut_root_pos_queue[1:]
            self.external_fut_root_pos_queue[-1] = latest_root_pos
            if self.external_fut_root_rot_queue is not None:
                self.external_fut_root_rot_queue[:-1] = self.external_fut_root_rot_queue[1:]
                self.external_fut_root_rot_queue[-1] = latest_root_rot
            if self.external_fut_frame_idx_queue is not None:
                self.external_fut_frame_idx_queue[:-1] = self.external_fut_frame_idx_queue[1:]
                self.external_fut_frame_idx_queue[-1] = latest_frame_idx

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

        self._store_external_latest_obs(np.asarray(data, dtype=np.float32))

        if (
            getattr(self, "enable_teleop_reference", True)
            and getattr(self, "require_vr_data_for_motion", False)
            and not getattr(self, "latest_obs_flag", False)
            and self._is_vr_ready_for_motion()
        ):
            self.latest_obs_flag = True
            if not getattr(self, "_vr_fk_started_logged", False):
                self.get_logger().info(
                    "[VR] ZMQ data is ready; the main thread will build the reference trajectory from live ZMQ input."
                )
                self._vr_fk_started_logged = True

    def _publish_latest_obs(self):
        """Publish the latest_obs topic for debugging or reuse."""
        if self.external_latest_obs is None:
            return
        try:
            msg = Float32MultiArray()
            msg.data = self.external_latest_obs[0].tolist()
            self.latest_obs_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish latest_obs: {e}")

    def _apply_onnx_metadata(self):
        """Apply PD/scale/defaults from ONNX metadata as authoritative values.
        Load separate metadata for velocity and motion models."""
        # Load velocity model metadata
        velocity_meta = self._read_onnx_metadata(self.velocity_onnx_path)
        self.velocity_dof_names_onnx = velocity_meta["joint_names"]
        self.velocity_action_scale_onnx = velocity_meta["action_scale"].astype(np.float32)
        self.velocity_kps_onnx = velocity_meta["kps"].astype(np.float32)
        self.velocity_kds_onnx = velocity_meta["kds"].astype(np.float32)
        self.velocity_default_angles_onnx = velocity_meta["default_joint_pos"].astype(np.float32)
        
        # Load motion model metadata
        motion_meta = self._read_onnx_metadata(self.motion_onnx_path)
        self.motion_dof_names_onnx = motion_meta["joint_names"]
        self.motion_action_scale_onnx = motion_meta["action_scale"].astype(np.float32)
        self.motion_kps_onnx = motion_meta["kps"].astype(np.float32)
        self.motion_kds_onnx = motion_meta["kds"].astype(np.float32)
        self.motion_default_angles_onnx = motion_meta["default_joint_pos"].astype(np.float32)
        
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
        
        # Build ref_to_onnx mapping (for motion model)
        self.ref_to_onnx = [
            self.dof_names_ref_motion.index(name) for name in self.motion_dof_names_onnx
        ]
        
        # Pre-compute default angles dictionaries for efficient observation building
        self.velocity_default_angles_dict = {
            name: float(self.velocity_default_angles_onnx[idx])
            for idx, name in enumerate(self.velocity_dof_names_onnx)
        }
        self.motion_default_angles_dict = {
            name: float(self.motion_default_angles_onnx[idx])
            for idx, name in enumerate(self.motion_dof_names_onnx)
        }
        
        # Pre-compute dof_names_onnx arrays for each mode (avoid repeated selection)
        self.velocity_dof_names_onnx_array = np.array(self.velocity_dof_names_onnx)
        self.motion_dof_names_onnx_array = np.array(self.motion_dof_names_onnx)
        self.motion_dof_real_indices = [
            self.real_dof_names.index(n) for n in self.motion_dof_names_onnx
        ]
        self.velocity_dof_real_indices = [
            self.real_dof_names.index(n) for n in self.velocity_dof_names_onnx
        ]
        n_dof = max(len(self.motion_dof_names_onnx), len(self.velocity_dof_names_onnx))
        self._dof_pos_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        self._dof_vel_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        
        # Pre-allocate arrays for future frame observations
        if hasattr(self, "n_fut_frames") and self.n_fut_frames is not None:
            self.n_fut_frames_int = int(self.n_fut_frames)
            if self.n_fut_frames_int > 0:
                self._pos_fut_buffer = np.zeros(
                    (len(self.dof_names_ref_motion), self.n_fut_frames_int), dtype=np.float32
                )
                self._h_fut_buffer = np.zeros((1, self.n_fut_frames_int), dtype=np.float32)
                self._root_pos_fut_buffer = np.zeros((self.n_fut_frames_int, 3), dtype=np.float32)
            else:
                self.n_fut_frames_int = 0
        else:
            self.n_fut_frames_int = 0

        self._future_frame_offsets = np.arange(1, self.n_fut_frames_int + 1, dtype=np.int64)
        self._future_frame_indices_buffer = np.zeros(self.n_fut_frames_int, dtype=np.int64)
        self._future_root_quat_wxyz_buffer = np.zeros((self.n_fut_frames_int, 4), dtype=np.float32)
        self._gravity_fut_buffer = np.zeros((self.n_fut_frames_int, 3), dtype=np.float32)
        self._base_linvel_fut_buffer = np.zeros((self.n_fut_frames_int, 3), dtype=np.float32)
        self._base_angvel_fut_buffer = np.zeros((self.n_fut_frames_int, 3), dtype=np.float32)
        self._keybody_rel_pos_fut_buffer = np.zeros((self.n_fut_frames_int, 0, 3), dtype=np.float32)
        self._keybody_rel_pos_w_buffer = None
        max_t = max(1, self.n_fut_frames_int)
        self._vel_fut_T6 = np.zeros((max_t, 6), dtype=np.float32)
        self._rot_t_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._rot_cross_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._use_fk_vr = False
        self._fk_vel_0_root = np.zeros(3, dtype=np.float32)
        self._fk_angvel_0_root = np.zeros(3, dtype=np.float32)
        self._fk_quat_0_root = np.zeros(4, dtype=np.float32)
        self._fk_trans_0 = None
        max_t = max(1, self.n_fut_frames_int)
        self._fk_vel_fut = np.zeros((max_t, 3), dtype=np.float32)
        self._fk_angvel_fut = np.zeros((max_t, 3), dtype=np.float32)
        self._fk_quat_fut = np.zeros((max_t, 4), dtype=np.float32)
        self._fk_trans_fut = None
        self._q_conj_buffer = np.zeros((max_t + 1, 4), dtype=np.float32)
        self._rotated_3vec_buffer = np.zeros(3, dtype=np.float32)
        self._rotated_angvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._cross_t_buffer = np.zeros(3, dtype=np.float32)
        self._fk_quat_0_root_wxyz = np.zeros(4, dtype=np.float32)
        self._fk_quat_fut_wxyz = np.zeros((max_t, 4), dtype=np.float32)
        
        # Pre-allocate velocity command observation array
        self._velocity_cmd_obs = np.zeros(4, dtype=np.float32)
        
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

    def _publish_policy_mode(self):
        """Publish current policy mode status."""
        try:
            mode_msg = String()
            mode_msg.data = f"{self.current_policy_mode}_{'enabled' if self.policy_enabled else 'disabled'}"
            self.policy_mode_pub.publish(mode_msg)
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

    def _root_only_fk_has_required_keybodies(self, keybody_idxs: np.ndarray) -> bool:
        if keybody_idxs.size == 0:
            return True
        available_bodies = 0 if self._fk_trans_0 is None else int(self._fk_trans_0.shape[0])
        if available_bodies <= int(np.max(keybody_idxs)):
            if not self._root_only_fk_keybody_warned:
                self.get_logger().warn(
                    "[RootOnlyFK] FK output only contains root body, but obs schema still "
                    "requests non-root keybody positions. Returning zeros for keybody obs."
                )
                self._root_only_fk_keybody_warned = True
            return False
        return True

    def _run_without_profiling(self):
        """Run the main loop without performance profiling."""
        if self._lowstate_msg is None or not self.policy_enabled:
            return None

        timing_info = {
            "policy_total_ms": 0.0,
            "fk_ms": 0.0,
            "obs_ms": 0.0,
            "onnx_ms": 0.0,
            "post_ms": 0.0,
        }
        _t_policy_start = time.perf_counter()

        if self.current_policy_mode == "motion":
            if self.latest_obs_flag:
                current_time = time.time()
                if self.last_external_obs_time is None:
                    data_age = float("inf")
                else:
                    data_age = current_time - self.last_external_obs_time

                if data_age > self.max_data_age:
                    self.get_logger().warn(
                        f"ZMQ latest_obs is stale: age={data_age*1000:.1f}ms > {self.max_data_age*1000:.1f}ms; "
                        "switching to velocity tracking mode for safety."
                    )
                    self._switch_to_velocity_mode(reason="VR latest_obs stale")
                    return None

            if not self.latest_obs_flag and (
                not hasattr(self, "n_motion_frames") or not hasattr(self, "ref_dof_pos")
            ):
                self.get_logger().warn("Motion data not loaded, skipping policy execution")
                return None

            if (
                self.latest_obs_flag
                and self.fk is not None
                and self.external_fut_dof_pos_queue is not None
            ):
                try:
                    n_fut = int(getattr(self, "n_fut_frames", 0))
                    if (
                        n_fut > 0
                        and self.external_fut_root_pos_queue is not None
                        and self.external_fut_root_rot_queue is not None
                    ):
                        t_fk = time.perf_counter()
                        cur_root_pos = self.ref_root_pos_raw.astype(np.float32)
                        cur_root_rot = (
                            self._prev_external_root_rot
                            if self._prev_external_root_rot is not None
                            else self.external_fut_root_rot_queue[0].astype(np.float32)
                        )
                        cur_dof_pos = self.ref_dof_pos_raw.astype(np.float32)
                        root_pos_tensor, root_rot_tensor, dof_pos_tensor = (
                            self._prepare_vr_fk_tensors(
                                cur_root_pos=cur_root_pos,
                                cur_root_rot=cur_root_rot,
                                cur_dof_pos=cur_dof_pos,
                                n_fut=n_fut,
                            )
                        )
                        fk_out = self.fk(
                            root_pos=root_pos_tensor,
                            root_quat=root_rot_tensor,
                            dof_pos=dof_pos_tensor,
                            fps=float(1.0 / self.dt),
                            quat_format="wxyz",
                            vel_smoothing_sigma=0.0,
                            compute_velocity=False,
                        )
                        self._fk_vr_out = {
                            k: v.detach().cpu().numpy() for k, v in fk_out.items()
                        }
                        timing_info["fk_ms"] = self._timing_ms(t_fk)
                    else:
                        self._fk_vr_out = None
                except Exception as e:
                    self.get_logger().error(
                        f"VR FK computation failed; falling back to offline reference: {e}"
                    )
                    self._fk_vr_out = None

            self.obs_builder = self.motion_obs_builder
            # Use motion model metadata
            current_action_scale = self.motion_action_scale_onnx
            current_default_angles = self.motion_default_angles_onnx
            current_onnx_to_real = self.motion_onnx_to_real
        else:  # velocity mode
            self.obs_builder = self.velocity_obs_builder
            # Use velocity model metadata
            current_action_scale = self.velocity_action_scale_onnx
            current_default_angles = self.velocity_default_angles_onnx
            current_onnx_to_real = self.velocity_onnx_to_real

        t_obs = time.perf_counter()
        if self.current_policy_mode == "motion":
            self._cache_fk_vr_for_obs()
        policy_obs_np = self.obs_builder.build_policy_obs()[None, :].astype(
            np.float32, copy=False
        )
        timing_info["obs_ms"] = self._timing_ms(t_obs)
        # Run ONNX inference with the appropriate policy session and correct input/output names
        t_onnx = time.perf_counter()
        if self.current_policy_mode == "velocity":
            input_feed = {self.velocity_input_name: policy_obs_np}
            onnx_output = self.velocity_policy_session.run([self.velocity_output_name], input_feed)
        else:  # motion mode
            if self.use_kv_cache:
                if self.motion_kv_cache is None:
                    shape = [
                        d if isinstance(d, int) else 1
                        for d in self.motion_kv_shape
                    ]
                    self.motion_kv_cache = np.zeros(shape, dtype=self.motion_kv_dtype)
                # if (
                #     self.motion_effective_context_len > 0
                #     and self.motion_step_idx > 0
                #     and self.motion_step_idx % self.motion_effective_context_len == 0
                # ):
                #     self.motion_kv_cache.fill(0.0)

                input_feed = {
                    self.motion_input_name: policy_obs_np,
                    self.motion_kv_input_name: self.motion_kv_cache,
                }
                if self.motion_step_idx_input_name is not None:
                    step_idx = self.motion_step_idx
                    # if self.motion_effective_context_len > 0:
                    #     step_idx = (
                    #         self.motion_step_idx
                    #         % self.motion_effective_context_len
                    #     )
                    input_feed[self.motion_step_idx_input_name] = np.array(
                        [step_idx], dtype=np.int64
                    )

                output_names = [self.motion_output_name]
                if self.motion_kv_output_name:
                    output_names.append(self.motion_kv_output_name)
                onnx_output = self.motion_policy_session.run(
                    output_names, input_feed
                )
                if len(onnx_output) > 1:
                    self.motion_kv_cache = onnx_output[1]
                self.motion_step_idx += 1
            else:
                input_feed = {self.motion_input_name: policy_obs_np}
                onnx_output = self.motion_policy_session.run(
                    [self.motion_output_name], input_feed
                )
        timing_info["onnx_ms"] = self._timing_ms(t_onnx)

        t_post = time.perf_counter()
        raw_actions_onnx = np.asarray(onnx_output[0], dtype=np.float32).reshape(-1)
        if self.current_policy_mode == "motion":
            self.actions_onnx = self._apply_motion_action_ema_filter(raw_actions_onnx)
        else:
            self.actions_onnx = raw_actions_onnx.copy()
        # Use the appropriate metadata based on current policy mode
        self.target_dof_pos_onnx = (
            self.actions_onnx * current_action_scale + current_default_angles
        )
        self.target_dof_pos_real = self.target_dof_pos_onnx[current_onnx_to_real]
        # Action processing and publishing
        self._process_and_publish_actions()
        if self.current_policy_mode == "motion":
            if (
                not getattr(self, "latest_obs_flag", False)
                and self.motion_frame_idx >= self.n_motion_frames
                and self.motion_in_progress
            ):
                self.get_logger().info("Motion action completed (offline reference)")
                self.motion_in_progress = False

        # Publish policy mode status
        self._publish_policy_mode()
        timing_info["post_ms"] = self._timing_ms(t_post)
        timing_info["policy_total_ms"] = self._timing_ms(_t_policy_start)
        return timing_info
    
    def _process_and_publish_actions(self):
        """Process and publish action commands."""
        if self.target_dof_pos_real is not None:
            action_msg = Float32MultiArray()
            action_msg.data = self.target_dof_pos_real.tolist()

            # Check for NaN values
            if np.isnan(self.target_dof_pos_real).any():
                self.get_logger().error("Action contains NaN values")

            self.action_pub.publish(action_msg)

        self.motion_frame_idx += 1

    def setup(self):
        """Set up the evaluator by loading all required components."""
        main_affinity = _parse_cpu_affinity_str(
            getattr(self, "_cpu_affinity_main_str", "") or ""
        )
        if main_affinity and set_thread_cpu_affinity(main_affinity):
            self.get_logger().info(f"[Policy] main thread pinned to CPUs {main_affinity}")
        self.load_model_config()  # Load config first
        self.update_config_parameters()  # Update parameters from config
        # Initialize FK for online VR reference reconstruction
        self._init_fk()
        self.load_policy()        # Then load policies
        self._apply_onnx_metadata()
        self._init_obs_buffers()
        self._build_dof_mappings()
        self._warmup_motion_policy()
        self._init_keybody_indices_cache()
        # Always load motion data since we support both modes
        self.load_motion_data()
        self.get_logger().info("Synchronous root-only policy setup completed")

    def _init_fk(self):
        """Initialize lightweight root-only FK for synchronous VR reference updates."""
        try:
            self.get_logger().info(
                "Initializing root-only FK (no URDF, sync main-thread mode)"
            )
            self.fk = HoloMotionFKRootOnly(
                dof_names=self.dof_names_ref_motion,
                device="cpu",
                timing_logger_enabled=True,
                timing_log_interval_sec=5.0,
                timing_log_per_call=False,
                timing_name="FKRootOnlyVR",
                timing_log_fn=self.get_logger().info,
            )
            try:
                ndof = len(self.fk.dof_names)
                root_pos_dummy = torch.zeros((1, 4, 3), dtype=torch.float32)
                root_quat_dummy = torch.zeros((1, 4, 4), dtype=torch.float32)
                root_quat_dummy[..., 0] = 1.0
                dof_pos_dummy = torch.zeros((1, 4, ndof), dtype=torch.float32)
                _ = self.fk(
                    root_pos=root_pos_dummy,
                    root_quat=root_quat_dummy,
                    dof_pos=dof_pos_dummy,
                    fps=float(1.0 / self.dt),
                    quat_format="wxyz",
                    vel_smoothing_sigma=0.0,
                    compute_velocity=False,
                )
                self.get_logger().info("[FK] Root-only warmup completed (B=1,T=4)")
            except Exception as e_dummy:
                self.get_logger().warn(f"[FK] Root-only warmup failed (ignored): {e_dummy}")

            self.fk_initialized = True
            self.get_logger().info(
                f"Root-only FK initialized successfully with {len(self.fk.dof_names)} dofs"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize root-only FK: {e}")
            self.fk = None
            self.fk_initialized = False

    def destroy_node(self):
        try:
            if getattr(self, "_zmq_subscriber", None) is not None:
                self._zmq_subscriber.stop()
        except Exception:
            pass

        super().destroy_node()

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Calculate gravity orientation from quaternion.

    Args:
        quaternion: Array-like [w, x, y, z]

    Returns:
        np.ndarray of shape (3,) representing gravity projection.
    """
    qw = float(quaternion[0])
    qx = float(quaternion[1])
    qy = float(quaternion[2])
    qz = float(quaternion[3])

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = HoloMotionPolicyNode()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
