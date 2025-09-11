#! /home/unitree/miniconda3/envs/holomotion_deploy/bin/python
"""
HoloMotion Policy Node

This module implements the main policy execution node for the HoloMotion humanoid robot system.
It handles neural network policy inference, motion sequence management, remote controller input,
and robot state coordination for complex humanoid behaviors including dancing and locomotion.

The policy node serves as the high-level decision maker that:
- Processes sensor observations and builds state representations
- Executes trained neural network policies for motion generation
- Manages multiple motion sequences (stand, squat, dance moves)
- Handles remote controller input for motion selection
- Coordinates with the main control node for safe operation

Author: HoloMotion Team
License: See project LICENSE file
"""
# 0812 record frequence
import time
import numpy as np
import rclpy
import torch
import onnxruntime
from rclpy.node import Node
from rclpy.qos import QoSProfile
import yaml
import easydict
import os
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory
from unitree_hg.msg import (
    LowState,
    LowCmd,
)
import joblib

from humanoid_policy.utils.command_helper import (
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_hg,
    init_cmd_go,
    MotorMode,
)
from humanoid_policy.utils.rotation_helper import (
    get_gravity_orientation,
    transform_imu_data,
)
from humanoid_policy.utils.remote_controller_filter import RemoteController, KeyMap
from humanoid_policy.utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    get_euler_xyz,
    wrap_to_pi,
    quat_inverse,
    quat_rotate,
)

from typing import List


@torch.compile
def quat_to_tan_norm(q: torch.Tensor, w_last: bool) -> torch.Tensor:
    """Converts quaternion to tangent-normal representation.

    Args:
        q: Input quaternion tensor of shape [..., 4]
        w_last: Whether the quaternion format is [x,y,z,w] (True) or [w,x,y,z] (False)

    Returns:
        Tangent-normal representation tensor of shape [..., 6] containing
        tangent (first 3 elements) and normal (last 3 elements) vectors
    """
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan, w_last)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm, w_last)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.compile
def remove_yaw_component(
    quat_raw,
    quat_init,
    w_last: bool = True,
):
    """Remove yaw component from quaternion while keeping roll and pitch.

    This function extracts the yaw component from the initial quaternion and uses
    it to normalize the raw quaternion, effectively removing the initial heading
    offset while preserving roll and pitch components.

    Args:
        quat_raw: Current quaternion from IMU, shape (..., 4)
        quat_init: Initial quaternion (contains the yaw to be removed), shape (..., 4)
        w_last: If True, quaternion format is (x, y, z, w).
                If False, quaternion format is (w, x, y, z). Default: True.

    Returns:
        Quaternion with initial yaw component removed, same shape as input.
        The resulting quaternion represents roll and pitch relative to the
        heading-aligned coordinate frame.

    Example:
        >>> # Initial robot orientation (roll=0°, pitch=0°, yaw=45°)
        >>> quat_init = quat_from_euler_xyz(
        ...     torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.7854)
        ... )
        >>> # Current IMU reading (roll=10°, pitch=20°, yaw=60°)
        >>> quat_raw = quat_from_euler_xyz(
        ...     torch.tensor(0.1745),
        ...     torch.tensor(0.3491),
        ...     torch.tensor(1.0472),
        ... )
        >>> quat_norm = remove_yaw_component(quat_raw, quat_init)
        >>> # quat_norm contains roll=10°, pitch=20°, with initial yaw offset removed
    """
    # Extract quaternion components based on format
    if w_last:
        q_w = quat_init[..., -1]
        q_vec = quat_init[..., :3]
    else:
        q_w = quat_init[..., 0]
        q_vec = quat_init[..., 1:]

    # Calculate heading by rotating x-axis with quaternion
    # ref_dir = [1, 0, 0] (x-axis)
    ref_dir = torch.zeros_like(q_vec)
    ref_dir[..., 0] = 1.0

    # Quaternion rotation: v' = v + 2 * w * (q_vec × v) + 2 * q_vec × (q_vec × v)
    cross1 = torch.cross(q_vec, ref_dir, dim=-1)
    cross2 = torch.cross(q_vec, cross1, dim=-1)
    rot_dir = ref_dir + 2.0 * q_w.unsqueeze(-1) * cross1 + 2.0 * cross2

    # Extract heading angle from rotated x-axis
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])

    # Create inverse heading quaternion (rotation about negative z-axis)
    half_heading = (-heading) * 0.5
    heading_q_inv = torch.zeros_like(quat_init)

    if w_last:
        heading_q_inv[..., 0] = 0.0  # x
        heading_q_inv[..., 1] = 0.0  # y
        heading_q_inv[..., 2] = torch.sin(half_heading)  # z
        heading_q_inv[..., 3] = torch.cos(half_heading)  # w
    else:
        heading_q_inv[..., 0] = torch.cos(half_heading)  # w
        heading_q_inv[..., 1] = 0.0  # x
        heading_q_inv[..., 2] = 0.0  # y
        heading_q_inv[..., 3] = torch.sin(half_heading)  # z

    # Quaternion multiplication: heading_q_inv * quat_raw
    shape = quat_raw.shape
    a = heading_q_inv.reshape(-1, 4)
    b = quat_raw.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    # Quaternion multiplication formula
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat_result = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat_result = torch.stack([w, x, y, z], dim=-1).view(shape)

    # Normalize the result quaternion
    norm = torch.norm(quat_result, p=2, dim=-1, keepdim=True)
    quat_norm = quat_result / norm.clamp(min=1e-8)

    return quat_norm

class ObsSeqSerializer:
    def __init__(self, schema_list: List[dict]):
        self.schema_list = schema_list
        self.obs_dim_dict = self._build_obs_dim_dict()
        self.obs_seq_len_dict = self._build_obs_seq_len_dict()
        self.obs_flat_dim = self._build_obs_flat_dim()

    def _build_obs_dim_dict(self):
        obs_dim_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            feat_dim = schema["feat_dim"]
            obs_dim_dict[obs_name] = feat_dim
        return obs_dim_dict

    def _build_obs_seq_len_dict(self):
        obs_seq_len_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            obs_seq_len_dict[obs_name] = seq_len
        return obs_seq_len_dict

    def _build_obs_flat_dim(self):
        obs_flat_dim = 0
        for schema in self.schema_list:
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_flat_dim += seq_len * feat_dim
        return obs_flat_dim

    def serialize(self, obs_seq_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(obs_seq_list) == len(self.schema_list)
        B = obs_seq_list[0].shape[0]
        output_tensor = []
        for schema, obs_seq in zip(self.schema_list, obs_seq_list):
            assert obs_seq.ndim == 3
            assert obs_seq.shape[0] == B
            assert obs_seq.shape[1] == schema["seq_len"]
            assert obs_seq.shape[2] == schema["feat_dim"]
            output_tensor.append(obs_seq.reshape(B, -1))
        return torch.cat(output_tensor, dim=-1)

    def deserialize(self, obs_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        assert obs_seq_tensor.ndim == 2
        output_dict = {}
        array_start_idx = 0
        B = obs_seq_tensor.shape[0]
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_size = seq_len * feat_dim
            array_end_idx = array_start_idx + obs_size
            output_dict[obs_name] = obs_seq_tensor[:, array_start_idx:array_end_idx].reshape(B, seq_len, feat_dim)
            array_start_idx = array_end_idx

        return output_dict


class PolicyNodeJustDance(Node):
    """Main policy execution node for HoloMotion humanoid robot control.

    This node implements the high-level control logic for a humanoid robot capable of
    performing various motion sequences including standing, squatting, and dance moves.
    It uses trained neural network policies for motion generation and coordinates with
    hardware control nodes for safe execution.

    Key Features:
    - Neural network policy inference using ONNX Runtime
    - Multi-motion sequence management (6 different motions)
    - Remote controller input processing for motion selection
    - Safety-aware state machine with motion prerequisites
    - Real-time observation processing and action generation

    State Machine:
    - ZERO_TORQUE: Initial safe state, waiting for activation
    - MOVE_TO_DEFAULT: Ready state, allows motion selection
    - Motion execution with completion tracking
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
        self.config = easydict.EasyDict(yaml.safe_load(open(config_path)))

        self.device = self.config.device
        self.dt = 1.0 / self.config.policy_freq

        self.current_motion = 0  # 0: stand motion, 1: squat motion
        self.current_dance_index = 0  # Current dance motion index
        self.state_mode = 0  # 0: standby, 1: start dance
        
        # Initialize motion initial quaternion (will be set from low state in __init__)
        self.cur_motion_init_quat = None

        # Safety check related flags
        self.policy_enabled = False  # Controls whether policy is enabled
        self.stand_completed = False  # Marks whether stand motion is completed
        self.squat_state = True  # Marks whether robot is in squat state
        self.walking_mode = False  # Controls whether walking mode is executed
        self.can_activate_walking = False  # Whether walking mode can be activated

        # Robot state related flags
        self.robot_state_ready = False  # Marks whether MOVE_TO_DEFAULT state is received, allowing key operations

        self.num_bodies_extend = self.config.num_bodies_extend

        self.policy_context_length = self.config.context_length
        self.max_context_length = self.policy_context_length
        self.performance_completed = False  # [T, 4]
        self.on_action = False
        self.context_length = self.config.context_length

        self.obs_serializer = ObsSeqSerializer(schema_list=self.config.serialization_schema)


        self._setup_data()
        self._setup_models()
        self._setup_motion_data()
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_timers()
        self._reset_policy()
        
        # Publish initial motion type (default is motion_file)
        self._publish_motion_type(0)
        
        # Set initial motion quaternion from low state if available
        if self.low_state_buffer_ is not None:
            self.cur_motion_init_quat = self.low_state_buffer_.imu_state.quaternion
            self.get_logger().info("Set initial motion quaternion from low state")
        else:
            self.cur_motion_init_quat = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion as fallback
            self.get_logger().warn("No low state available, set initial motion quaternion to identity quaternion")

        # Music playback removed

    def _setup_data(self):
        """Initialize data structures and parameters for observation processing.

        Sets up:
        - Buffer for low-level robot state
        - Action scaling parameters
        - Default joint positions from configuration
        - Control decimation settings
        - Velocity command variables
        - History validation queues
        """
        self.low_state_buffer_ = None
        self.latest_obs = None

        self.action_scale = self.config.action_scale
        self.default_dof_pos = np.array(
            [self.config.default_joint_angles[dof_name] for dof_name in self.config.policy_dof_order]
        )
        self.control_decimation = self.config.control_decimation
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.hist_valid_queue = np.zeros(self.max_context_length, dtype=bool)

        self.actions = np.zeros(self.config.num_actions, dtype=np.float32)
        self.obs_queues = {key: None for key in self.config.history_obs.keys()}
        self._update_obs_queue()

    def _reset_policy(self):
        """Reset policy state to initial conditions.

        Clears:
        - Action outputs
        - Observation queues and history
        - Motion counters and phase tracking

        Used when initializing the policy.
        """
        self.actions = np.zeros(self.config.num_actions, dtype=np.float32)
        self.obs_queues = {key: None for key in self.config.history_obs.keys()}
        self._update_obs_queue()
        self.hist_policy_obs = np.zeros(
            sum(self.config.obs_dims[key] * self.config.context_length for key in self.config.history_obs.keys()),
            dtype=np.float32,
        )
        self.counter = 0
        self.current_phase = 0.0

    def _reset_counter(self):
        """Reset motion timing counters to start of sequence."""
        self.counter = 0
        self.current_phase = 0.0

    def _setup_models(self):
        """Load and initialize the neural network policy model.

        Sets up ONNX Runtime inference session with GPU acceleration support.
        The model is loaded from the configured path and optimized for real-time
        inference with CUDA backend when available.

        Raises:
            RuntimeError: If model loading fails or ONNX Runtime setup fails
        """
        # Old model (existing)
        self.get_logger().info(f"Loading OLD policy from {self.config.policy_path} ...")
        old_model_path = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            self.config.policy_path,
        )

        # New model (preferred). Support either 'new_policy_path' or 'policy_path_new'
        new_policy_key = "new_policy_path" if hasattr(self.config, "new_policy_path") else ("policy_path_new" if hasattr(self.config, "policy_path_new") else None)
        new_model_path = None
        if new_policy_key is not None:
            new_model_path = os.path.join(
                get_package_share_directory("humanoid_control"),
                "models",
                getattr(self.config, new_policy_key),
            )

        # Setup ONNX Runtime sessions
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        try:
            # Load old model
            self.policy_session_old = onnxruntime.InferenceSession(old_model_path, providers=providers)
            self.policy_input_name_old = self.policy_session_old.get_inputs()[0].name
            self.policy_output_name_old = self.policy_session_old.get_outputs()[0].name
            self.get_logger().info(
                f"OLD policy loaded. Providers: {self.policy_session_old.get_providers()} | IO: {self.policy_input_name_old} -> {self.policy_output_name_old}"
            )

            # Load new model if provided; if not, fall back to old
            if new_model_path is not None:
                self.get_logger().info(f"Loading NEW policy from {getattr(self.config, new_policy_key)} ...")
                self.policy_session_new = onnxruntime.InferenceSession(new_model_path, providers=providers)
                self.policy_input_name_new = self.policy_session_new.get_inputs()[0].name
                self.policy_output_name_new = self.policy_session_new.get_outputs()[0].name
                self.get_logger().info(
                    f"NEW policy loaded. Providers: {self.policy_session_new.get_providers()} | IO: {self.policy_input_name_new} -> {self.policy_output_name_new}"
                )
            else:
                self.policy_session_new = None
                self.get_logger().warn("No new policy path provided (expected 'new_policy_path' or 'policy_path_new'). Using OLD policy only.")

        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model(s): {e}")
            raise

        # Select active session: prefer NEW if available
        if self.policy_session_new is not None:
            self._set_active_session_internal(use_new=True)
            self.get_logger().info("Active policy set to NEW model by default.")
        else:
            self._set_active_session_internal(use_new=False)
            self.get_logger().info("Active policy set to OLD model by default.")

    def _set_active_session_internal(self, use_new: bool):
        self.use_new_model = bool(use_new)
        if self.use_new_model:
            self.active_session = self.policy_session_new
            self.active_input_name = self.policy_input_name_new
            self.active_output_name = self.policy_output_name_new
        else:
            self.active_session = self.policy_session_old
            self.active_input_name = self.policy_input_name_old
            self.active_output_name = self.policy_output_name_old

    def _setup_motion_data(self):
        """Load and prepare motion sequence data for basic motions and dance motions.

        Loads motion data for:
        - B button: Stand up motion
        - X button: Squat motion
        - UP/DOWN/LEFT/RIGHT buttons: Dance motions from dance_motions_dir

        Each motion contains:
        - Joint positions and velocities over time
        - Root body poses and velocities
        - Body link positions and orientations
        - Timing and synchronization data
        """
        # Load basic motion files (stand and squat)
        self.basic_motion_cfg_files = [
            self.config.motion_file_squat2stand,      # stand motion
            self.config.motion_file_stand2squat,    # squat motion
        ]

        basic_motion_paths = [
            os.path.join(
                get_package_share_directory("humanoid_control"),
                "motion_data",
                motion_file,
            )
            for motion_file in self.basic_motion_cfg_files
        ]

        # Load basic motions
        self.basic_motions = []
        self.basic_motion_frames = []
        for path in basic_motion_paths:
            motion = joblib.load(path)
            motion_name = list(motion.keys())[0]
            motion_data = motion[motion_name]

            # transform the np arrays to torch tensors
            for k, v in motion_data.items():
                if isinstance(v, np.ndarray):
                    motion_data[k] = torch.from_numpy(v)

            self.basic_motions.append(motion_data)
            self.basic_motion_frames.append(motion_data["dof_pos"].shape[0])

        # Load dance motions from directory
        self._load_dance_motions()

        # Initialize with first basic motion (stand)
        self.motion_data = self.basic_motions[0]
        self.num_motion_frames = self.basic_motion_frames[0]
        self.motion_fps = self.motion_data["fps"]
        self.num_fut_frames = self.config.num_fut_frames

        self.ref_initial_base_rot_quat = self.motion_data["root_rot"][0:1, :]  # [1, 4]
        self.ref_initial_heading_inv_quat = calc_heading_quat_inv(
            self.ref_initial_base_rot_quat,
            w_last=True,
        )  # [1, 4]
        self.ref_initial_heading_inv_quat_expanded = self.ref_initial_heading_inv_quat.repeat(
            self.num_fut_frames,
            1,
        ).reshape(self.num_fut_frames, 4)

    def _load_dance_motions(self):
        """Load all dance motion files from the configured directory."""
        dance_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            "motion_data",
            self.config.dance_motions_dir,
        )
        
        if not os.path.exists(dance_dir):
            self.get_logger().warn(f"Dance motions directory not found: {dance_dir}")
            self.dance_motions = []
            self.dance_motion_frames = []
            self.dance_motion_files = []
            return

        # Get all .pkl files in the dance directory
        dance_files = [f for f in os.listdir(dance_dir) if f.endswith('.pkl')]
        dance_files.sort()  # Sort for consistent ordering
        
        if not dance_files:
            self.get_logger().warn(f"No .pkl files found in dance directory: {dance_dir}")
            self.dance_motions = []
            self.dance_motion_frames = []
            self.dance_motion_files = []
            return

        self.get_logger().info(f"Loading {len(dance_files)} dance motions from {dance_dir}")
        
        self.dance_motions = []
        self.dance_motion_frames = []
        self.dance_motion_files = []
        
        for dance_file in dance_files:
            dance_path = os.path.join(dance_dir, dance_file)
            try:
                motion = joblib.load(dance_path)
                motion_name = list(motion.keys())[0]
                motion_data = motion[motion_name]

                # transform the np arrays to torch tensors
                for k, v in motion_data.items():
                    if isinstance(v, np.ndarray):
                        motion_data[k] = torch.from_numpy(v)

                self.dance_motions.append(motion_data)
                self.dance_motion_frames.append(motion_data["dof_pos"].shape[0])
                self.dance_motion_files.append(dance_file)
                
                self.get_logger().info(f"Loaded dance motion: {dance_file} ({motion_data['dof_pos'].shape[0]} frames)")
                
            except Exception as e:
                self.get_logger().error(f"Failed to load dance motion {dance_file}: {e}")

        self.get_logger().info(f"Successfully loaded {len(self.dance_motions)} dance motions")

    # Music helpers removed

    def _setup_subscribers(self):
        """Set up ROS2 subscribers for robot state and remote controller input."""
        self.remote_controller = RemoteController()
        self.low_state_sub = self.create_subscription(
            LowState,
            self.config.lowstate_topic,
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

    def _setup_publishers(self):
        """Set up ROS2 publishers for action commands and status information."""
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.config.action_topic,
            QoSProfile(depth=10),
        )
        self.counter_pub = self.create_publisher(
            Int32,
            "current_count",
            QoSProfile(depth=10),
        )
        # Add new publisher for total frames
        self.total_frames_pub = self.create_publisher(
            Int32,
            "total_motion_frames",
            QoSProfile(depth=10),
        )
        # Add new publisher for joystick data
        self.joystick_pub = self.create_publisher(
            Float32MultiArray,
            "joystick_data",
            QoSProfile(depth=10),
        )
        # Add new publisher for velocity commands
        self.velocity_cmd_pub = self.create_publisher(
            Float32MultiArray,
            "velocity_commands",
            QoSProfile(depth=10),
        )
        # Add new publisher for inference time
        self.inference_time_pub = self.create_publisher(
            Float32,
            "inference_time",
            QoSProfile(depth=10),
        )
        # Add new publisher for motion type
        self.motion_type_pub = self.create_publisher(
            String,
            "motion_type",
            QoSProfile(depth=10),
        )

    def _setup_timers(self):
        """Set up ROS2 timer for main execution loop."""
        self.create_timer(self.dt, self.run)

    def _update_obs_queue(self):
        cur_q = np.zeros(len(self.config.policy_dof_order))
        cur_dq = np.zeros(len(self.config.policy_dof_order))
        ang_vel = np.zeros(3, dtype=np.float32)
        gravity_orientation = np.zeros(3, dtype=np.float32)
        if self.low_state_buffer_ is not None:
            motor_state = self.low_state_buffer_.motor_state
            for i, dof_name in enumerate(self.config.policy_dof_order):
                motor_idx = self.config.dof2motor_idx_mapping[dof_name]
                cur_q[i] = motor_state[motor_idx].q - self.config.default_joint_angles[dof_name]
                cur_dq[i] = motor_state[motor_idx].dq
            quat = self.low_state_buffer_.imu_state.quaternion
            ang_vel = np.array([self.low_state_buffer_.imu_state.gyroscope], dtype=np.float32).reshape(-1)
            gravity_orientation = get_gravity_orientation(quat)

        if self.state_mode == 0:
            # self.counter = 351
            self.counter = 1
        elif self.state_mode == 1:
            # self.counter = self.num_motion_frames - self.num_fut_frames - 1
            pass

        current_obs_dict = {
            "actions": self.actions.astype(np.float32),
            "base_ang_vel": ang_vel.astype(np.float32),
            "dof_pos": cur_q.astype(np.float32),
            "dof_vel": cur_dq.astype(np.float32),
            "projected_gravity": gravity_orientation.astype(np.float32),
        }
        for key in self.obs_queues.keys():
            obs_item = current_obs_dict[key]
            if self.obs_queues[key] is None:
                self.obs_queues[key] = np.zeros((self.context_length, obs_item.shape[0]), dtype=np.float32)
            else:
                self.obs_queues[key][1:] = self.obs_queues[key][:-1].copy()
                self.obs_queues[key][0] = obs_item * self.config.obs_scales[key]

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
        # self.get_logger().info(f"Received robot state: {robot_state}")

        # Only allow button operations when robot state is MOVE_TO_DEFAULT
        if robot_state == "MOVE_TO_DEFAULT":
            self.robot_state_ready = True
            # self.get_logger().info("Robot state ready: Button operations enabled")
        elif robot_state == "ZERO_TORQUE":
            self.robot_state_ready = False
            # self.get_logger().info("Robot state not ready: Button operations disabled")
        elif robot_state == "EMERGENCY_STOP":
            self.robot_state_ready = False
            # self.get_logger().warn("Emergency stop detected: Button operations disabled")

    def _low_state_callback(self, ls_msg: LowState):
        """Process low-level robot state and remote controller input.

        Main callback that handles:
        - Remote controller input processing
        - Motion selection based on button presses
        - Safety state checking
        - Velocity command extraction

        Motion Button Mapping:
        - A: Enable/disable policy execution
        - B: Stand up motion (always available when policy enabled)
        - X: Squat motion (only after standing)
        - UP/DOWN/LEFT/RIGHT: Dance motions (only after standing)

        Args:
            ls_msg: LowState message containing robot sensor data and remote controller input
        """
        self.low_state_buffer_ = ls_msg
        self.remote_controller.set(ls_msg.wireless_remote)

        # A or start button: Toggle policy enable state
        if self.remote_controller.button[KeyMap.A] == 1 and (not self.on_action) and self.robot_state_ready:
            # if self.remote_controller.button[KeyMap.A] == 1:
            # self.on_action = True
            if self.low_state_buffer_ is not None:
                self.cur_motion_init_quat = self.low_state_buffer_.imu_state.quaternion
            self.policy_enabled = True
            self.stand_completed = False  # Reset stand state
            self.squat_state = True  # Reset squat state
            self.state_mode = 0
            self._reset_counter()
            self.motion_data = self.basic_motions[0]
            self.num_motion_frames = self.basic_motion_frames[0]
            # Reset walking related flags
            self.walking_mode = False
            self.can_activate_walking = False
            self.get_logger().info(f"Policy {'enabled' if self.policy_enabled else 'disabled'}")

        # Hot-swap models at runtime via controller
        # Y -> switch to OLD model; R1 -> switch back to NEW model
        if self.policy_enabled:
            if self.remote_controller.button[KeyMap.Y] == 1:
                if hasattr(self, "policy_session_old") and self.active_session is not self.policy_session_old:
                    self._set_active_session_internal(use_new=False)
                    self.get_logger().info("Switched ACTIVE policy to OLD model via Y button")
            if self.remote_controller.button[KeyMap.R1] == 1:
                if hasattr(self, "policy_session_new") and self.policy_session_new is not None and self.active_session is not self.policy_session_new:
                    self._set_active_session_internal(use_new=True)
                    self.get_logger().info("Switched ACTIVE policy to NEW model via R1 button")

        # Only process other buttons when policy is enabled and robot state is ready
        if self.policy_enabled and self.robot_state_ready:
            # B button (stand) is always available when policy is enabled
            if self.remote_controller.button[KeyMap.B] == 1 and (not self.on_action):
                self._start_motion(0)
                self.stand_completed = False
                self.on_action = True
                # self.can_activate_walking = True  # After B button pressed, allow walking mode activation after full stand
                self.get_logger().info("Starting stand up motion")

            # If in squat state, only B button is available
            elif self.squat_state:
                return

            # Other motions are only available after standing is completed and not in squat state
            elif self.stand_completed:
                if self.remote_controller.button[KeyMap.X] == 1 and self.performance_completed:
                    self._start_motion(1)
                    self.walking_mode = False
                    self.can_activate_walking = False
                    self.get_logger().info("Starting squat motion")
                    self.performance_completed = False
                    self.on_action = True
                elif self.remote_controller.button[KeyMap.up] == 1 and self.performance_completed:
                    # Previous dance motion (index -1)
                    self._select_dance_motion(-1)
                elif self.remote_controller.button[KeyMap.down] == 1 and self.performance_completed:
                    # Next dance motion (index +1)
                    self._select_dance_motion(1)
                elif self.remote_controller.button[KeyMap.right] == 1 and self.performance_completed:
                    # First dance motion (index 0)
                    self._select_dance_motion(0)
                elif self.remote_controller.button[KeyMap.left] == 1 and self.performance_completed:
                    # Last dance motion (index -1)
                    self._select_dance_motion(len(self.dance_motions) - 1)

        # If policy is enabled but robot state is not ready, show warning
        elif self.policy_enabled and not self.robot_state_ready:
            # Check if any buttons are pressed, if so show warning
            buttons_pressed = any(
                [
                    self.remote_controller.button[KeyMap.B],
                    self.remote_controller.button[KeyMap.X],
                    self.remote_controller.button[KeyMap.up],
                    self.remote_controller.button[KeyMap.down],
                    self.remote_controller.button[KeyMap.left],
                    self.remote_controller.button[KeyMap.right],
                ]
            )
            if buttons_pressed:
                self.get_logger().warn("Robot not in MOVE_TO_DEFAULT state. Button operations disabled.")

        self.vx, self.vy, self.vyaw = self.remote_controller.get_velocity_commands()

    def _select_dance_motion(self, target_index: int):
        """Select and start a dance motion based on the target index.
        
        Args:
            target_index: Target dance motion index
                -1: Previous dance motion (current - 1)
                0: First dance motion
                positive: Next dance motion (current + 1)
                last_index: Last dance motion
        """
        if not self.dance_motions:
            self.get_logger().warn("No dance motions available")
            return
        
        if target_index == -1:  # Previous dance motion
            self.current_dance_index = (self.current_dance_index - 1) % len(self.dance_motions)
        elif target_index == 0:  # First dance motion
            self.current_dance_index = 0
        elif target_index == len(self.dance_motions) - 1:  # Last dance motion
            self.current_dance_index = len(self.dance_motions) - 1
        else:  # Next dance motion (target_index == 1)
            self.current_dance_index = (self.current_dance_index + 1) % len(self.dance_motions)
        
        # Start the selected dance motion (motion_index = 2 + dance_index)
        motion_index = 2 + self.current_dance_index
        self._start_motion(motion_index)
        self.walking_mode = False
        self.performance_completed = False
        self.on_action = True
        
        dance_name = self.dance_motion_files[self.current_dance_index] if self.dance_motion_files else f"dance_{self.current_dance_index}"
        self.get_logger().info(f"Starting dance motion {self.current_dance_index}: {dance_name}")

    def _start_motion(self, motion_index: int):
        """Start execution of a specific motion sequence.

        Initializes motion playback including:
        - Loading motion data and resetting counters
        - Publishing motion metadata

        Args:
            motion_index: Index of motion to execute
                0: Stand up, 1: Squat, 2+: Dance motions
        """
        self.state_mode = 1
        self.current_motion = motion_index
        
        # Load motion data based on type
        if motion_index < 2:  # Basic motions (stand/squat)
            self.motion_data = self.basic_motions[motion_index]
            self.num_motion_frames = self.basic_motion_frames[motion_index]
            motion_name = "basic"
        else:  # Dance motions
            dance_index = motion_index - 2
            if dance_index < len(self.dance_motions):
                self.motion_data = self.dance_motions[dance_index]
                self.num_motion_frames = self.dance_motion_frames[dance_index]
                motion_name = "dance"
            else:
                self.get_logger().error(f"Dance motion index {dance_index} out of range")
                return
        
        self._reset_counter()

        # Update motion initial quaternion when starting new motion
        if self.low_state_buffer_ is not None:
            self.cur_motion_init_quat = self.low_state_buffer_.imu_state.quaternion
            self.get_logger().info(f"Updated motion initial quaternion for motion {motion_index}")
        else:
            self.get_logger().warn("No low state buffer available, cannot update motion initial quaternion")

        # Publish total frames when starting new motion
        total_frames_msg = Int32()
        total_frames_msg.data = self.num_motion_frames
        self.total_frames_pub.publish(total_frames_msg)

        # Publish motion type based on motion index
        self._publish_motion_type(motion_index)

        # Music playback removed

    def _publish_motion_type(self, motion_index: int):
        """Publish motion type based on motion index.
        
        Args:
            motion_index: Index of motion to execute
                0: Stand up, 1: Squat, 2+: Dance motions
        """
        motion_type_msg = String()
        if motion_index == 0:  # motion_file (B button - stand)
            motion_type_msg.data = "motion_file_squat2stand"
        elif motion_index == 1:  # motion_file_3 (X button - squat)
            motion_type_msg.data = "motion_file_stand2squat"
        else:  # Dance motions (UP/DOWN/LEFT/RIGHT buttons)
            motion_type_msg.data = "dance_motion"
        self.motion_type_pub.publish(motion_type_msg)
        self.get_logger().info(f"Published motion type: {motion_type_msg.data} for motion index {motion_index}")

    def _get_obs_noisy_dof_with_history_seq(self):
        hist_len = self.config.context_length
        hist_dof_pos = self.obs_queues["dof_pos"][:hist_len]

        cur_q = np.zeros(len(self.config.policy_dof_order))
        cur_dq = np.zeros(len(self.config.policy_dof_order))
        motor_state = self.low_state_buffer_.motor_state
        for i, dof_name in enumerate(self.config.policy_dof_order):
            motor_idx = self.config.dof2motor_idx_mapping[dof_name]
            # cur_q[i] = motor_state[motor_idx].q - self.config.default_joint_angles[dof_name]
            cur_q[i] = motor_state[motor_idx].q
            cur_dq[i] = motor_state[motor_idx].dq

        cur_dof_pos = torch.tensor(cur_q, dtype=torch.float32)  # [num_dofs]
        hist_dof_pos = torch.tensor(hist_dof_pos, dtype=torch.float32)
        dof_pos_seq = torch.cat([cur_dof_pos[None, :], hist_dof_pos], dim=0)  # [num_envs, hist_len + 1, num_dofs]

        cur_dof_vel = torch.tensor(cur_dq, dtype=torch.float32)  # [num_dofs]
        hist_dof_vel = torch.tensor(self.obs_queues["dof_vel"][:hist_len], dtype=torch.float32)
        dof_vel_seq = torch.cat([cur_dof_vel[None, :], hist_dof_vel], dim=0)  # [num_envs, hist_len + 1, num_dofs]

        dof_seq = torch.cat([dof_pos_seq, dof_vel_seq], dim=-1)  # [hist_len + 1, 2 * num_dofs]
        return dof_seq  # [hist_len + 1, 2 * num_dofs]

    def _get_obs_noisy_imu_with_history_seq(self):
        hist_len = self.config.context_length
        quat = self.low_state_buffer_.imu_state.quaternion
        ang_vel = np.array([self.low_state_buffer_.imu_state.gyroscope], dtype=np.float32).reshape(-1)
        gravity_orientation = get_gravity_orientation(quat)

        current_base_ang_vel = torch.tensor(ang_vel, dtype=torch.float32)  # [3]
        current_base_proj_gravity = torch.tensor(gravity_orientation, dtype=torch.float32)  # [3]
        hist_base_ang_vel = torch.tensor(self.obs_queues["base_ang_vel"][:hist_len], dtype=torch.float32)
        hist_base_proj_gravity = torch.tensor(self.obs_queues["projected_gravity"][:hist_len], dtype=torch.float32)

        imu_seq = torch.cat(
            [
                torch.cat(
                    [
                        current_base_ang_vel[None, :],
                        hist_base_ang_vel,
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        current_base_proj_gravity[None, :],
                        hist_base_proj_gravity,
                    ],
                    dim=0,
                ),
            ],
            dim=-1,
        )  # [num_envs, hist_len + 1, 6]
        return imu_seq  # [num_envs, hist_len + 1, 6]

    def _get_obs_action_with_history_seq(self):
        hist_len = self.config.context_length
        last_actions = torch.tensor(self.actions.copy(), dtype=torch.float32)  # [num_dofs]
        hist_actions = torch.tensor(self.obs_queues["actions"][:hist_len], dtype=torch.float32)  # [hist_len, num_dofs]
        action_seq = torch.cat(
            [last_actions[None, :], hist_actions],
            dim=0,
        )  # [hist_len + 1, num_dofs]
        return action_seq  # [hist_len + 1, num_dofs]

    def _get_obs_fut_ref_root_rel_teacher_v2(self, motion_init_quat):
        """
        This observation is used for obtaining the future reference motion state
        for training the teacher policy. Notice that the future bodylink properties
        are expressed in the **current** root reference frame.

        - Root roll and pitch: in future per-frame heading-aligned frame
        - Root linear and angular velocity: in the **current** root reference frame
        - DoF position and velocity: in the absolute frame
        - Bodylink position, rotation, linear and angular velocity: in the **current** root reference frame
        """
        NB = self.num_bodies_extend
        FT = self.num_fut_frames
        target_ref_frame_id = self.counter + 1
        max_target_ref_frame = self.num_motion_frames - self.num_fut_frames
        if target_ref_frame_id > max_target_ref_frame:
            target_ref_frame_id = max_target_ref_frame

        quat_raw = self.low_state_buffer_.imu_state.quaternion
        quat_raw = torch.tensor(quat_raw, dtype=torch.float32)
        quat_raw = quat_raw[..., [1, 2, 3, 0]]  # wxyz -> xyzw
        motion_init_quat = torch.tensor(motion_init_quat, dtype=torch.float32)
        motion_init_quat = motion_init_quat[..., [1, 2, 3, 0]]  # wxyz -> xyzw
        quat = remove_yaw_component(quat_raw, motion_init_quat, w_last=True)

        raw_roll, raw_pitch, raw_yaw = get_euler_xyz(quat_raw[None, :], w_last=True)
        normed_roll, normed_pitch, normed_yaw = get_euler_xyz(quat[None, :], w_last=True)
        motion_init_roll, motion_init_pitch, motion_init_yaw = get_euler_xyz(motion_init_quat[None, :], w_last=True)

        cur_root_quat = quat

        # cur_root_quat = torch.tensor(quat, dtype=torch.float32)
        # wxyz -> xyzw
        # cur_root_quat = cur_root_quat[..., [1, 2, 3, 0]]
        cur_root_quat_inv = quat_inverse(cur_root_quat, w_last=True)
        cur_root_quat_inv_fut_flat = cur_root_quat_inv[None, :].repeat(FT, 1).view(-1, 4)  # [T, 4]

        ref_fut_base_rot_quat = self.motion_data["root_rot"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, 4]
        ref_fut_root_global_lin_vel = self.motion_data["root_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, 3]
        ref_fut_root_global_ang_vel = self.motion_data["root_ang_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, 3]
        ref_fut_dof_pos = self.motion_data["dof_pos"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_actions]
        ref_fut_dof_vel = self.motion_data["dof_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_actions]
        ref_fut_bodylink_pos = self.motion_data["rg_pos_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_bodies_extend, 3]
        ref_fut_bodylink_rot = self.motion_data["rg_rot_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_bodies_extend, 4]
        ref_fut_bodylink_vel = self.motion_data["body_vel_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_bodies_extend, 3]
        ref_fut_bodylink_ang_vel = self.motion_data["body_ang_vel_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, num_bodies_extend, 3]
        ref_cur_root_pos = self.motion_data["root_pos"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, 3]

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            ref_fut_base_rot_quat,
            w_last=True,
        )  # [T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            ref_fut_base_rot_quat,
            w_last=True,
        )  # [T, 4]

        # --- calculate the global roll and pitch of the future heading-aligned frame ---
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll)  # [T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch)  # [T, 1]
        ref_fut_rp = torch.cat([ref_fut_roll, ref_fut_pitch], dim=-1)  # [T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(-1)  # [T * 2]
        # ---

        # --- calculate the relative root linear and angular velocity to the current root ---
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            cur_root_quat_inv_fut_flat,  # [T, 4]
            ref_fut_root_global_lin_vel,  # [T, 3]
            w_last=True,
        ).reshape(-1)  # [T * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            cur_root_quat_inv_fut_flat,  # [T, 4]
            ref_fut_root_global_ang_vel,  # [T, 3]
            w_last=True,
        ).reshape(-1)  # [T * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_rel_dof_pos_flat = ref_fut_dof_pos.reshape(-1)  # [T * num_actions]
        fut_ref_rel_dof_vel_flat = ref_fut_dof_vel.reshape(-1)  # [T * num_actions]
        # ---

        # --- calculate the relative bodylink pos in the current root reference frame ---
        ref_fut_cur_root_quat_inv_body_flat = (
            cur_root_quat_inv_fut_flat[:, None, :].repeat(1, NB, 1).reshape(-1, 4)
        )  # [T * NB, 4]

        fut_root_rel_ref_bodylink_pos_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            (ref_fut_bodylink_pos - ref_cur_root_pos[None, None, :]).reshape(-1, 3),
            w_last=True,
        ).reshape(-1)  # [T * NB * 3]
        fut_root_rel_ref_bodylink_rot_tannorm = quat_mul(
            ref_fut_cur_root_quat_inv_body_flat,
            ref_fut_bodylink_rot.reshape(-1, 4),
            w_last=True,
        ).reshape(-1, 4)  # [T * NB * 4]
        fut_root_rel_ref_bodylink_rot_tannorm_flat = quat_to_tan_norm(
            fut_root_rel_ref_bodylink_rot_tannorm,
            w_last=True,
        ).reshape(-1)  # [T * NB * 6]
        fut_root_rel_ref_bodylink_vel_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            ref_fut_bodylink_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(-1)  # [T * NB * 3]
        fut_root_rel_ref_bodylink_ang_vel_flat = quat_rotate(
            ref_fut_cur_root_quat_inv_body_flat,
            ref_fut_bodylink_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(-1)  # [T * NB * 3]
        # ---
        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(FT, -1),  # [T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(FT, -1),  # [T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(FT, -1),  # [T, 3]
                fut_ref_rel_dof_pos_flat.reshape(FT, -1),  # [T, num_dofs]
                fut_ref_rel_dof_vel_flat.reshape(FT, -1),  # [T, num_dofs]
                fut_root_rel_ref_bodylink_pos_flat.reshape(FT, -1),  # [T, num_bodies_extend * 3]
                fut_root_rel_ref_bodylink_rot_tannorm_flat.reshape(FT, -1),  # [T, num_bodies_extend * 6]
                fut_root_rel_ref_bodylink_vel_flat.reshape(FT, -1),  # [T, num_bodies_extend * 3]
                fut_root_rel_ref_bodylink_ang_vel_flat.reshape(FT, -1),  # [T, num_bodies_extend * 3]
            ],
            dim=-1,
        )  # [T, 2 + 3 + 3 + num_dofs * 2 + num_bodies_extend * (3 + 6 + 3 + 3)]
        return rel_fut_ref_motion_state_seq.flatten()

    def _get_obs_priocep_with_fut_ref_v7_student(self):
        target_ref_frame_id = int(self.counter / self.control_decimation) + 1
        max_target_ref_frame = self.num_motion_frames - self.num_fut_frames
        if target_ref_frame_id > max_target_ref_frame:
            target_ref_frame_id = max_target_ref_frame

        valid_frames_left = min(self.num_motion_frames - target_ref_frame_id, self.num_fut_frames)
        ref_fut_valid_mask = torch.zeros(self.num_fut_frames)
        ref_fut_valid_mask[:valid_frames_left] = 1
        ref_fut_valid_mask = ref_fut_valid_mask.flatten()

        dof_seq = self._get_obs_noisy_dof_with_history_seq()  # [HT + 1, num_dofs]
        imu_seq = self._get_obs_noisy_imu_with_history_seq()  # [HT + 1, 6]
        action_seq = self._get_obs_action_with_history_seq()  # [HT + 1, num_actions]

        hist_seq = torch.cat([dof_seq, imu_seq, action_seq], dim=-1)

        # Pass the motion initial quaternion to teacher_v2 function
        fut_ref = self._get_obs_fut_ref_root_rel_teacher_v2(self.cur_motion_init_quat)
        fut_ref_valid_mask = ref_fut_valid_mask[:, None]
        fut_ref = fut_ref * fut_ref_valid_mask.float()

        return (
            self.obs_serializer.serialize(
                [
                    hist_seq[None, ...],
                    fut_ref[None, ...],
                    fut_ref_valid_mask[None, ...],
                ]
            )
            .cpu()
            .numpy()
        )

    def _process_observation(self):
        """Process robot sensors and build observation for policy inference.

        Constructs the complete observation vector that includes:
        - Current joint positions and velocities (relative to defaults)
        - IMU orientation and angular velocity
        - Reference motion state for trajectory following
        - Action history for temporal consistency

        The observation is built according to the configured observation space
        and properly scaled for neural network input.

        Returns:
            None: Updates self.latest_obs with processed observation vector
        """
        if self.low_state_buffer_ is None:
            return None
        self.latest_obs = self._get_obs_priocep_with_fut_ref_v7_student()
        self._update_obs_queue()

    def run(self):
        """Main execution loop for policy inference and action publication.

        Executes the complete policy pipeline at the configured frequency:
        1. Process observations from robot sensors
        2. Run neural network inference for action generation
        3. Scale and publish actions to control system
        4. Publish status information and telemetry
        5. Update motion counters and check for completion

        This method is called by the ROS2 timer at the policy frequency
        (typically 50Hz) and coordinates all aspects of policy execution.
        """
        start_obs_time = time.time()
        self._process_observation()
        end_obs_time = time.time()
        # self.get_logger().info(
        #     f"Observation processing time: {end_obs_time - start_obs_time} seconds"
        # )
        if self.latest_obs is None:
            return

        # Convert to numpy for ONNX inference
        policy_obs_np = self.latest_obs.astype(np.float32)
        input_feed = {self.active_input_name: policy_obs_np}

        # Run ONNX inference
        start_infer_time = time.time()
        onnx_output = self.active_session.run([self.active_output_name], input_feed)
        end_infer_time = time.time()
        inference_time = (end_infer_time - start_infer_time) * 1000  # Convert to milliseconds
        self.actions = onnx_output[0][0]
        
        # Publish inference time
        inference_time_msg = Float32()
        inference_time_msg.data = inference_time
        self.inference_time_pub.publish(inference_time_msg)


        if self.actions is not None:
            action_msg = Float32MultiArray()
            target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
            action_msg.data = target_dof_pos.tolist()

            if np.isnan(target_dof_pos).any():
                self.get_logger().error("Action contains NaN values")
            self.action_pub.publish(action_msg)


        self.counter += 1

        # Check if stand up motion is completed
        if (
            self.current_motion == 0  # Stand up motion completed
            and self.counter == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.stand_completed = True
            self.squat_state = False
            self.can_activate_walking = True
            self.get_logger().info("Stand up completed, other motions now available")
            self.performance_completed = True
            self.on_action = False

        # Check if squat motion is completed
        elif (
            self.current_motion == 1  # Squat motion completed
            and self.counter == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.squat_state = True
            self.stand_completed = False
            self.walking_mode = False
            self.can_activate_walking = False
            self.on_action = False
            self.get_logger().info("Squat completed, only stand up motion available")
        elif (
            self.current_motion >= 2  # Dance motions
            and self.counter == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.performance_completed = True
            self.on_action = False
            dance_index = self.current_motion - 2
            dance_name = self.dance_motion_files[dance_index] if dance_index < len(self.dance_motion_files) else f"dance_{dance_index}"
            self.get_logger().info(f"Dance motion {dance_index} ({dance_name}) completed, other motions now available")


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = PolicyNodeJustDance()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
