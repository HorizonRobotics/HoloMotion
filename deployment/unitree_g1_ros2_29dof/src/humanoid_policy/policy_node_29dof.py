#! /your_dir/miniconda3/envs/holomotion_deploy/bin/python
"""
HoloMotion Policy Node

This module implements the main policy execution node for the HoloMotion humanoid robot system.
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

import easydict
import numpy as np
import onnx
import onnxruntime
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from omegaconf import OmegaConf
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState

from humanoid_policy.obs_builder import PolicyObsBuilder
from humanoid_policy.utils.remote_controller_filter import KeyMap, RemoteController


class PolicyNodeJustDance(Node):
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
        self.config_yaml = easydict.EasyDict(yaml.safe_load(open(config_path)))
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
        self.current_policy_mode = "velocity"  # "velocity" or "motion"     
        # Separate configs for each model
        self.velocity_config = None
        self.motion_config = None
        # Motion data
        self.motion_frame_idx = 0

        # Extract configuration parameters
        # These will be updated after config loading
        self.dof_names_ref_motion = []
        self.num_actions = 29  # Default value
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)

        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)

        self.counter = 0
        self._lowstate_msg = None
        # Desired target positions keyed by DOF name (updated after each policy step)
        self.target_dof_pos_by_name = {}
        # Don't call setup() here - it will be called after ROS2 initialization
        self.motion_in_progress = False
        self.motion_mode_first_entry = True

    def _find_actor_place_holder_ndim(self):
        n_dim = 0
        for obs_dict in self.motion_config.obs.obs_groups.policy.atomic_obs_list:
            if list(obs_dict.keys())[0] == "place_holder":
                n_dim = obs_dict["place_holder"].params.n_dim
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
            obs_policy_cfg=self.velocity_config.obs.obs_groups.policy,
        )

        # Use motion model's parameters for motion obs_builder
        self.motion_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.motion_dof_names_onnx,
            default_angles_onnx=self.motion_default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=self.motion_config.obs.obs_groups.policy,
        )

        # Set default obs_builder to velocity mode
        self.obs_builder = self.velocity_obs_builder

    def _reset_counter(self):
        """Reset motion timing counters to start of sequence."""
        self.counter = 0
        self.motion_frame_idx = 0

    def _switch_to_velocity_mode(self, reason: str = ""):
        """Switch to velocity tracking mode and clear action cache.
        
        Uses velocity model's default_angles_onnx to ensure correct initialization.
        Also publishes velocity model's control parameters (kps/kds).
        """
        self.current_policy_mode = "velocity"
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
            str(velocity_onnx_path), providers=providers
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
            str(motion_onnx_path), providers=providers
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
        self.actor_place_holder_ndim = self._find_actor_place_holder_ndim()
        self.n_fut_frames = int(self.motion_config.obs.n_fut_frames)
        self.torso_body_idx = self.motion_config.robot.body_names.index("torso_link")
        self.get_logger().info("Both model configs loaded successfully")

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
            try:
                motion_data_dict = dict(np.load(motion_path, allow_pickle=True))
                try:
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
                            "n_frames": motion_data_dict["ref_dof_pos"].shape[0],
                        }
                    )
                    self.motion_file_names.append(motion_clip_file)
                except:
                    self.all_motion_data.append(
                        {
                            "dof_pos": motion_data_dict["dof_pos"],
                            "dof_vel": motion_data_dict["dof_vels"],
                            "global_translation": motion_data_dict[
                                "global_translation"
                            ],
                            "global_rotation_quat": motion_data_dict[
                                "global_rotation_quat"
                            ],
                            "n_frames": motion_data_dict["dof_pos"].shape[0],
                        }
                    )
                    self.motion_file_names.append(motion_clip_file)
            except Exception as e:
                self.get_logger().warn(f"Failed to load motion data from {motion_clip_file}: {e}")
                continue
        
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

        self.n_motion_frames = current_motion["n_frames"]

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
        return self.ref_dof_pos[self.ref_motion_frame_idx]

    @property
    def ref_dof_vel_raw(self):
        return self.ref_dof_vel[self.ref_motion_frame_idx]

    @property
    def ref_dof_pos_onnx_order(self):
        return self.ref_dof_pos_raw[self.ref_to_onnx]

    @property
    def ref_dof_vel_onnx_order(self):
        return self.ref_dof_vel_raw[self.ref_to_onnx]

    @property
    def ref_root_pos_raw(self):
        return np.asarray(
            self.ref_raw_bodylink_pos[self.ref_motion_frame_idx, self.root_body_idx],
            dtype=np.float32,
        )

    @property
    def root_body_idx(self):
        return 0

    @property
    def last_valid_ref_motion_frame_idx(self):
        return self.n_motion_frames - 1

    # =========== Policy Obeservation Methods ===========

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
        """Get DOF position observation (uses pre-computed dictionaries)."""
        dof_pos_by_name = self.robot_dof_pos_by_name
        # Use pre-computed default angles dictionary based on current policy mode
        if self.current_policy_mode == "motion":
            default_angles_dict = self.motion_default_angles_dict
            dof_names_onnx = self.motion_dof_names_onnx
        else:
            default_angles_dict = self.velocity_default_angles_dict
            dof_names_onnx = self.velocity_dof_names_onnx
        
        dof_pos_onnx = np.array(
            [
                dof_pos_by_name[name] - default_angles_dict[name]
                for name in dof_names_onnx
            ],
            dtype=np.float32,
        )
        return dof_pos_onnx

    def _get_obs_dof_vel(self):
        """Get DOF velocity observation (uses mode-specific DOF names)."""
        dof_vel_by_name = self.robot_dof_vel_by_name
        # Use mode-specific DOF names
        if self.current_policy_mode == "motion":
            dof_names_onnx = self.motion_dof_names_onnx
        else:
            dof_names_onnx = self.velocity_dof_names_onnx
        
        dof_vel_onnx = np.array(
            [dof_vel_by_name[name] for name in dof_names_onnx],
            dtype=np.float32,
        )
        return dof_vel_onnx

    def _get_obs_last_action(self):
        return self.actions_onnx.copy()

    def _get_obs_ref_motion_states(self):
        return np.concatenate(
            [self.ref_dof_pos_onnx_order, self.ref_dof_vel_onnx_order]
        )

    def _get_obs_ref_dof_pos_fut(self):
        """Get future DOF position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0 or self.ref_dof_pos is None:
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.ref_motion_frame_idx
        # Reuse pre-allocated buffer
        pos_fut = self._pos_fut_buffer
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                pos_fut[:, i] = self.ref_dof_pos[idx]
            else:
                pos_fut[:, i] = self.ref_dof_pos[self.last_valid_ref_motion_frame_idx]
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :].transpose(1, 0)  # [N, T]
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_height_fut(self):
        """Get future root height observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.ref_raw_bodylink_pos is None:
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.ref_motion_frame_idx
        # Reuse pre-allocated buffer
        h_fut = self._h_fut_buffer
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                h_fut[0, i] = self.ref_raw_bodylink_pos[
                    idx,
                    self.root_body_idx,
                    2,
                ]
            else:
                h_fut[0, i] = self.ref_raw_bodylink_pos[
                    self.last_valid_ref_motion_frame_idx,
                    self.root_body_idx,
                    2,
                ]
        return h_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_pos_fut(self):
        """Get future root position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.ref_raw_bodylink_pos is None:
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.ref_motion_frame_idx
        # Reuse pre-allocated buffer
        pos_fut = self._root_pos_fut_buffer
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                pos_fut[i] = self.ref_raw_bodylink_pos[
                    idx,
                    self.root_body_idx,
                    :
                ]
            else:
                pos_fut[i] = self.ref_raw_bodylink_pos[
                    self.last_valid_ref_motion_frame_idx,
                    self.root_body_idx,
                    :
                ]
        return pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_cur(self):
        return self.ref_dof_pos_onnx_order

    def _get_obs_ref_dof_vel_cur(self):
        return self.ref_dof_vel_onnx_order

    def _get_obs_ref_root_height_cur(self):
        return self.ref_raw_bodylink_pos[
            self.ref_motion_frame_idx, self.root_body_idx, 2
        ]

    def _get_obs_ref_root_pos_cur(self):
        return self.ref_root_pos_raw.astype(np.float32)

    def _get_obs_place_holder(self):
        return np.zeros(self.actor_place_holder_ndim, dtype=np.float32)

    # =========== Policy Obeservation Methods ===========

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
            self._reset_counter()
            self.motion_mode_first_entry = True   # reset flag
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
            # Don't automatically switch to next motion clip - keep current selection
            if hasattr(self, "all_motion_data") and self.all_motion_data:
                # Load the current motion clip data (don't change current_motion_clip_index)
                self._load_current_motion()

            self.motion_mode_first_entry = False
            self.current_policy_mode = "motion"
            self._reset_counter()

            # Clear any pending actions to prevent conflicts between policies
            # Use motion model's default angles
            self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
            self.target_dof_pos_onnx = self.motion_default_angles_onnx.copy()
            
            # Publish motion model's control parameters (kps/kds)
            self._publish_control_params()

            self.get_logger().info(
                f"Switched to motion tracking mode - motion clip index: {self.current_motion_clip_index}"
            )
            self.motion_in_progress = True

        # Y button: Switch back to velocity tracking mode (only when policy is enabled)
        if (
            self._is_button_pressed(KeyMap.Y)
            and self.robot_state_ready
            and self.policy_enabled
            and self.current_policy_mode == "motion"  # Only allow switch from motion mode
        ):
            self._switch_to_velocity_mode()
            # Don't reset motion_mode_first_entry here - we want to advance to next motion clip

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
        self._run_without_profiling()

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

    def _run_without_profiling(self):
        """Run the main loop without performance profiling."""
        if self._lowstate_msg is None or not self.policy_enabled:
            return None

        if self.current_policy_mode == "motion":
            # Check if motion data is loaded
            if not hasattr(self, "n_motion_frames") or not hasattr(self, "ref_dof_pos"):
                self.get_logger().warn("Motion data not loaded, skipping policy execution")
                return None
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

        policy_obs_np = self.obs_builder.build_policy_obs()[None, :]

        # Run ONNX inference with the appropriate policy session and correct input/output names
        if self.current_policy_mode == "velocity":
            input_feed = {self.velocity_input_name: policy_obs_np}
            onnx_output = self.velocity_policy_session.run([self.velocity_output_name], input_feed)
        else:  # motion mode
            input_feed = {self.motion_input_name: policy_obs_np}
            onnx_output = self.motion_policy_session.run(
                [self.motion_output_name], input_feed
            )

        self.actions_onnx = onnx_output[0].reshape(-1)
        
        # Use the appropriate metadata based on current policy mode
        self.target_dof_pos_onnx = (
            self.actions_onnx * current_action_scale + current_default_angles
        )
        self.target_dof_pos_real = self.target_dof_pos_onnx[current_onnx_to_real]
        # Update named targets for each actuator DOF
        for i, dof_name in enumerate(self.real_dof_names):
            self.target_dof_pos_by_name[dof_name] = float(self.target_dof_pos_real[i])
        # Action processing and publishing
        self._process_and_publish_actions()
        if self.current_policy_mode == "motion":
            if self.motion_frame_idx >= self.n_motion_frames and self.motion_in_progress:
                self.get_logger().info("Motion action completed")
                self.motion_in_progress = False

        # Publish policy mode status
        self._publish_policy_mode()

    def _process_and_publish_actions(self):
        """Process and publish action commands."""
        if hasattr(self, "target_dof_pos_by_name") and self.target_dof_pos_by_name:
            action_msg = Float32MultiArray()

            action_msg.data = list(self.target_dof_pos_by_name.values())

            # Check for NaN values
            target_dof_pos = np.array(list(self.target_dof_pos_by_name.values()))
            if np.isnan(target_dof_pos).any():
                self.get_logger().error("Action contains NaN values")

            self.action_pub.publish(action_msg)

        self.counter += 1
        self.motion_frame_idx += 1

    def setup(self):
        """Set up the evaluator by loading all required components."""
        self.load_model_config()  # Load config first
        self.update_config_parameters()  # Update parameters from config
        self.load_policy()        # Then load policies
        self._apply_onnx_metadata()
        self._init_obs_buffers()
        self._build_dof_mappings()
        # Always load motion data since we support both modes
        self.load_motion_data()

    def destroy_node(self):
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
    policy_node = PolicyNodeJustDance()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
