#! /your_dir/miniconda3/envs/holomotion_deploy/bin/python
"""
HoloMotion Policy Node

This module implements the main policy execution node for the HoloMotion humanoid robot system.
It handles neural network policy inference, motion sequence management, remote controller input,
and robot state coordination for complex humanoid behaviors including dancing and locomotion.

The policy node serves as the high-level decision maker that:
- Processes sensor observations and builds state representations
- Executes trained neural network policies for motion generation
- Manages multiple motion sequences (dance moves)
- Handles remote controller input for motion selection
- Coordinates with the main control node for safe operation

Author: HoloMotion Team
License: See project LICENSE file
"""
import numpy as np
import rclpy
import torch
import onnxruntime
import onnx
from rclpy.node import Node
from rclpy.qos import QoSProfile
import yaml
import easydict
import os
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory
from unitree_hg.msg import (
    LowState,
    LowCmd,
)
from humanoid_policy.utils.remote_controller_filter import RemoteController, KeyMap
from humanoid_policy.obs_builder import PolicyObsBuilder

# Set up logger
import logging
logger = logging.getLogger(__name__)


class PolicyNodeJustDance(Node):
    """Main policy execution node for HoloMotion humanoid robot control with dual policy support.

    This node implements the high-level control logic for a humanoid robot capable of
    performing both velocity tracking and motion sequence execution. It supports two
    neural network policies and allows runtime switching between them.

    Key Features:
    - Dual neural network policy inference (velocity + motion) using ONNX Runtime
    - Runtime policy switching with A/B/Y button controls
    - Velocity tracking mode with joystick control
    - Motion tracking mode with dance sequence selection
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
        # Initialize config - will be loaded from model folders later
        self.config = None
        self.device = self._get_device()
        self.dt = 1.0 / 50
        # Initialize basic parameters - will be updated after config loading
        self.actions_dim = 29  # Default value, will be updated from config
        self.real_dof_names = []  # Will be loaded from config
        self.current_dance_index = 0  # Current dance motion index       
        # Button state tracking for preventing multiple triggers
        self.last_button_states = {
            KeyMap.up: 0,
            KeyMap.down: 0,
            KeyMap.left: 0,
            KeyMap.right: 0,
            KeyMap.A: 0,
            KeyMap.B: 0,
            KeyMap.Y: 0
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
        self.motion_data = None
        self.motion_frame_idx = 0
        # Extract configuration parameters
        # These will be updated after config loading
        self.dof_names_ref_motion = []
        self.num_actions = 29  # Default value
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)
        self.context_length = 10  # Default value

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
        self.command_mode = "velocity_tracking"  # Keep for backward compatibility
        self.motion_in_progress = False
        self.motion_mode_first_entry = True


    def _get_device(self):
        """Get the device to use for computation."""
        # Use device from main config, fallback to CUDA/CPU
        device = self.config_yaml.get("device", None)
        if device:
            # If config specifies cuda but torch doesn't have CUDA support, fallback to cpu
            if device == "cuda":
                try:
                    if torch.cuda.is_available():
                        return "cuda"
                    else:
                        self.get_logger().warn("Config specifies CUDA but torch doesn't have CUDA support, using CPU instead")
                        return "cpu"
                except Exception as e:
                    self.get_logger().warn(f"Error checking CUDA availability: {e}, using CPU instead")
                    return "cpu"
            return device
        else:
            # Try to use CUDA if available, otherwise use CPU
            try:
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except Exception as e:
                self.get_logger().warn(f"Error checking CUDA availability: {e}, using CPU instead")
                return "cpu"


    def _init_obs_buffers(self):
        # Get context_length from each model's config
        velocity_context_length = self.velocity_config.get("obs", {}).get("context_length", 10)
        motion_context_length = self.motion_config.get("obs", {}).get("context_length", 10)
        
        self.get_logger().info(f"Velocity model context_length: {velocity_context_length}")
        self.get_logger().info(f"Motion model context_length: {motion_context_length}")
        # Create separate observation builders for each mode with their own config
        self.velocity_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.dof_names_onnx,
            default_angles_onnx=self.default_angles_onnx,
            context_length=velocity_context_length,
            device=self.device,
            command_mode="velocity_tracking",
        )
        
        self.motion_obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.dof_names_onnx,
            default_angles_onnx=self.default_angles_onnx,
            context_length=motion_context_length,
            device=self.device,
            command_mode="motion_tracking",
        )
        
        # Set default obs_builder to velocity mode
        self.obs_builder = self.velocity_obs_builder


    def _reset_counter(self):
        """Reset motion timing counters to start of sequence."""
        self.counter = 0
        self.motion_frame_idx = 0
    

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
                    "device_id": 0,  # Or determine from self.device if needed
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
            f"Velocity policy loaded successfully using: {self.velocity_policy_session.get_providers()}"
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
            f"Motion policy loaded successfully using: {self.motion_policy_session.get_providers()}"
        )
        # Set input/output names for both policies
        self.velocity_input_name = self.velocity_policy_session.get_inputs()[0].name
        self.velocity_output_name = self.velocity_policy_session.get_outputs()[0].name
        self.motion_input_name = self.motion_policy_session.get_inputs()[0].name
        self.motion_output_name = self.motion_policy_session.get_outputs()[0].name
        
        self.get_logger().info(
            f"Velocity policy - Input: {self.velocity_input_name}, Output: {self.velocity_output_name}"
        )
        self.get_logger().info(
            f"Motion policy - Input: {self.motion_input_name}, Output: {self.motion_output_name}"
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
            raise FileNotFoundError(f"No config file found in {velocity_config_dir}. Tried: {config_names}")
        
        self.get_logger().info(f"Loading velocity model config from {velocity_config_path}")
        self.velocity_config = easydict.EasyDict(yaml.safe_load(open(velocity_config_path)))
        
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
            raise FileNotFoundError(f"No config file found in {motion_config_dir}. Tried: {config_names}")
        
        self.get_logger().info(f"Loading motion model config from {motion_config_path}")
        self.motion_config = easydict.EasyDict(yaml.safe_load(open(motion_config_path)))      
        # Set the main config to velocity config for backward compatibility
        self.config = self.velocity_config
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
            self.get_logger().warn(f"Different actions_dim: velocity={velocity_actions_dim}, motion={motion_actions_dim}")
        
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
        # Note: context_length is now handled separately for each obs_builder
        
        # Update arrays with correct sizes
        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)
        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        
        self.get_logger().info(f"Updated config parameters: actions_dim={self.actions_dim}, dof_names={len(self.real_dof_names)}")


    def load_motion_data(self):
        """ Load motion data from .npz files."""
        dance_dir = os.path.join(
            get_package_share_directory("humanoid_control"),
            self.config_yaml.dance_motions_dir,
        )
        
        self.get_logger().info(f"Looking for motion data in: {dance_dir}")
        self.get_logger().info(f"Directory exists: {os.path.exists(dance_dir)}")

        if not os.path.exists(dance_dir):
            self.get_logger().warn(f"Dance motions directory not found: {dance_dir}")
            return

        # Only collect .npz files
        dance_files = [f for f in os.listdir(dance_dir) if f.endswith('.npz')]
        dance_files.sort()
        self.get_logger().info(f"Found {len(dance_files)} motion files (.npz): {dance_files}")
        if not dance_files:
            self.get_logger().warn(
                f"No motion files (.npz) found in dance directory: {dance_dir}"
            )
            return

        # Load each .npz file
        self.all_motion_data = []
        self.motion_file_names = []
        for dance_file in dance_files:
            motion_path = os.path.join(dance_dir, dance_file)
            try:
                ref_dof_pos, ref_dof_vel = self._load_motion_file(motion_path)
                if ref_dof_pos is None or ref_dof_vel is None:
                    self.get_logger().warn(f"Unsupported or invalid motion file (missing keys): {dance_file}")
                    continue

                ref_dof_pos = ref_dof_pos.astype(np.float32)
                ref_dof_vel = ref_dof_vel.astype(np.float32)

                if ref_dof_pos.shape[0] != ref_dof_vel.shape[0]:
                    self.get_logger().warn(
                        f"Frame mismatch in {dance_file}: pos {ref_dof_pos.shape}, vel {ref_dof_vel.shape}"
                    )
                    n_min = min(ref_dof_pos.shape[0], ref_dof_vel.shape[0])
                    ref_dof_pos = ref_dof_pos[:n_min]
                    ref_dof_vel = ref_dof_vel[:n_min]

                self.all_motion_data.append({
                    'dof_pos': ref_dof_pos,
                    'dof_vel': ref_dof_vel,
                    'n_frames': ref_dof_pos.shape[0]
                })
                self.motion_file_names.append(dance_file)
            except Exception as e:
                self.get_logger().warn(f"Failed to load motion data from {dance_file}: {e}")
                continue
        
        if not self.all_motion_data:
            self.get_logger().error("Failed to load any motion data files")
            return

        # Initialize with the first motion
        self.current_dance_index = 0
        self._load_current_motion()
        
        self.get_logger().info(f"Loaded {len(self.all_motion_data)} dance motions successfully")
        self.get_logger().info(f"Current dance: {self.motion_file_names[self.current_dance_index]}")

    def _load_motion_file(self, motion_path: str):
        """Load a single .npz file, return (dof_pos [T,N], dof_vel [T,N]). Return (None, None) on failure."""
        try:
            if not motion_path.endswith('.npz'):
                return None, None
            with np.load(motion_path, allow_pickle=True) as npz:
                keys = list(npz.keys())
                naming_pairs = [
                    ("dof_pos", "dof_vel"),
                    ("dof_pos", "dof_vels"),
                    ("ref_dof_pos", "ref_dof_vel"),
                ]
                for pos_key, vel_key in naming_pairs:
                    if pos_key in npz and vel_key in npz:
                        pos = np.array(npz[pos_key]).astype(np.float32)
                        vel = np.array(npz[vel_key]).astype(np.float32)
                        if pos.shape[0] != vel.shape[0]:
                            min_T = min(pos.shape[0], vel.shape[0])
                            pos = pos[:min_T]
                            vel = vel[:min_T]
                        return pos, vel
                if len(keys) == 1:
                    arr = npz[keys[0]]
                    if getattr(arr, 'dtype', None) == object:
                        obj = arr.item() if arr.size == 1 else arr
                        if isinstance(obj, dict):
                            if 'dof_pos' in obj and 'dof_vel' in obj:
                                return np.array(obj['dof_pos']).astype(np.float32), np.array(obj['dof_vel']).astype(np.float32)
                            if 'dof_pos' in obj and 'dof_vels' in obj:
                                return np.array(obj['dof_pos']).astype(np.float32), np.array(obj['dof_vels']).astype(np.float32)
                return None, None
        except Exception as load_err:
            raise load_err


    def _load_current_motion(self):
        """Load the current selected motion data."""
        if not self.all_motion_data:
            return
            
        self.motion_frame_idx = 0
        current_motion = self.all_motion_data[self.current_dance_index]
        self.ref_dof_pos = current_motion['dof_pos']
        self.ref_dof_vel = current_motion['dof_vel']
        self.n_motion_frames = current_motion['n_frames']
        
        self.motion_in_progress = True
        self.get_logger().info(f"Loaded motion {self.current_dance_index}: {self.motion_file_names[self.current_dance_index]} ({self.n_motion_frames} frames)")
       

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
        - UP/DOWN/LEFT/RIGHT: Dance motions (only in velocity mode)

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
            self.get_logger().info(
                f"Policy enabled in {self.current_policy_mode} mode"
            )

        # B button: Switch to motion policy mode (only when policy is enabled)
        if (
            self._is_button_pressed(KeyMap.B) and 
            self.robot_state_ready and 
            self.policy_enabled and
            self.current_policy_mode == "velocity"  # Only allow switch from velocity mode
        ):
            # Don't automatically switch to next dance - keep current selection
            if hasattr(self, 'all_motion_data') and self.all_motion_data:
                # Load the current motion data (don't change current_dance_index)
                self._load_current_motion()
            
            self.motion_mode_first_entry = False
            self.current_policy_mode = "motion"
            self._reset_counter()
            
            # Clear any pending actions to prevent conflicts between policies
            self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
            self.target_dof_pos_onnx = self.default_angles_onnx.copy()
            
            self.get_logger().info(f"Switched to motion policy mode - dance index: {self.current_dance_index}")
            self.motion_in_progress = True

        # Y button: Switch back to velocity policy mode (only when policy is enabled)
        if (
            self._is_button_pressed(KeyMap.Y) and 
            self.robot_state_ready and 
            self.policy_enabled and
            self.current_policy_mode == "motion"  # Only allow switch from motion mode
        ):
            self.current_policy_mode = "velocity"
            self._reset_counter()
            
            # Clear any pending actions to prevent conflicts between policies
            self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
            self.target_dof_pos_onnx = self.default_angles_onnx.copy()
            
            # Don't reset motion_mode_first_entry here - we want to advance to next dance
            self.get_logger().info(f"Switched to velocity policy mode")

        # Get velocity commands only in velocity mode
        if self.current_policy_mode == "velocity":
            self.vx, self.vy, self.vyaw = self.remote_controller.get_velocity_commands()
        else:
            # In motion mode, ignore joystick input
            self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0

        # Handle motion selection in velocity mode (UP/DOWN/LEFT/RIGHT for dance selection)
        if (self.current_policy_mode == "velocity" and 
            self.policy_enabled and 
            self.robot_state_ready):
            
            # Handle dance motion selection with UP/DOWN/LEFT/RIGHT buttons
            if self._is_button_pressed(KeyMap.up):
                # Switch to previous dance
                if hasattr(self, 'all_motion_data') and self.all_motion_data:
                    self.current_dance_index = (self.current_dance_index - 1) % len(self.all_motion_data)
                    self.get_logger().info(f"Selected previous dance: {self.motion_file_names[self.current_dance_index]}")
            elif self._is_button_pressed(KeyMap.down):
                # Switch to next dance
                if hasattr(self, 'all_motion_data') and self.all_motion_data:
                    self.current_dance_index = (self.current_dance_index + 1) % len(self.all_motion_data)
                    self.get_logger().info(f"Selected next dance: {self.motion_file_names[self.current_dance_index]}")
            elif self._is_button_pressed(KeyMap.left):
                # Select first dance
                if hasattr(self, 'all_motion_data') and self.all_motion_data:
                    self.current_dance_index = 0
                    self.get_logger().info(f"Selected first dance: {self.motion_file_names[self.current_dance_index]}")
            elif self._is_button_pressed(KeyMap.right):
                # Select last dance
                if hasattr(self, 'all_motion_data') and self.all_motion_data:
                    self.current_dance_index = len(self.all_motion_data) - 1
                    self.get_logger().info(f"Selected last dance: {self.motion_file_names[self.current_dance_index]}")


    def run(self):
        """Main execution loop for policy inference and action publication."""
        # Only run if setup is completed
        if not hasattr(self, '_setup_completed') or not self._setup_completed:
            return
        self._run_without_profiling()


    def _read_onnx_metadata(self) -> dict:
        """Read model metadata from ONNX file and parse into Python types."""
        # Use velocity policy metadata as the primary source (assuming both policies have same structure)
        onnx_model_path = self.velocity_onnx_path
            
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
        """Apply PD/scale/defaults from ONNX metadata as authoritative values."""
        meta = self._read_onnx_metadata()
        self.dof_names_onnx = meta["joint_names"]
        self.action_scale_onnx = meta["action_scale"].astype(np.float32)
        self.kps_onnx = meta["kps"].astype(np.float32)
        self.kds_onnx = meta["kds"].astype(np.float32)
        self.default_angles_onnx = meta["default_joint_pos"].astype(np.float32)
    
    
    def _build_dof_mappings(self):
        # Map ONNX <-> MJCF for control
        
        # Check if all ONNX names exist in real_dof_names
        missing_names = [name for name in self.dof_names_onnx if name not in self.real_dof_names]
        if missing_names:
            self.get_logger().warn(f"Missing names in real_dof_names: {missing_names}")
            # Use dof_names_onnx as the authoritative source
            # self.real_dof_names = self.dof_names_onnx.copy()
        
        self.onnx_to_real = [
            self.dof_names_onnx.index(name) for name in self.real_dof_names
        ]
        self.ref_to_onnx = [
            self.dof_names_ref_motion.index(name) for name in self.dof_names_onnx
        ]

        self.kps_real = self.kps_onnx[self.onnx_to_real].astype(np.float32)
        self.kds_real = self.kds_onnx[self.onnx_to_real].astype(np.float32)
        self.default_angles_mu = self.default_angles_onnx[self.onnx_to_real].astype(
            np.float32
        )
        self.action_scale_mu = self.action_scale_onnx[self.onnx_to_real].astype(
            np.float32
        )
        
        # Publish kps and kds parameters
        self._publish_control_params()


    def _publish_control_params(self):
        """Publish kps and kds control parameters."""
        try:
            # Publish kps
            kps_msg = Float32MultiArray()
            kps_msg.data = self.kps_real.tolist()
            self.kps_pub.publish(kps_msg)
            
            # Publish kds
            kds_msg = Float32MultiArray()
            kds_msg.data = self.kds_real.tolist()
            self.kds_pub.publish(kds_msg)
            
            self.get_logger().info(f"Published control parameters: kps={len(self.kps_real)}, kds={len(self.kds_real)}")
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
        q_by_name = {
            self.real_dof_names[i]: float(self._lowstate_msg.motor_state[i].q)
            for i in range(self.actions_dim)
        }
        dq_by_name = {
            self.real_dof_names[i]: float(self._lowstate_msg.motor_state[i].dq)
            for i in range(self.actions_dim)
        }
        imu_quat = np.array(self._lowstate_msg.imu_state.quaternion, dtype=np.float32)
        imu_gyro = np.array(self._lowstate_msg.imu_state.gyroscope, dtype=np.float32)

        if self.current_policy_mode == "motion":
            # Check if motion data is loaded
            if not hasattr(self, 'n_motion_frames') or not hasattr(self, 'ref_dof_pos'):
                self.get_logger().warn("Motion data not loaded, skipping policy execution")
                return None
            frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
            ref_dof_pos_raw = self.ref_dof_pos[frame_idx]
            ref_dof_vel_raw = self.ref_dof_vel[frame_idx]

            # Use motion obs_builder
            self.obs_builder = self.motion_obs_builder
            
            hist_obs = self.obs_builder.update_from_lowstate(
                q_by_name=q_by_name,
                dq_by_name=dq_by_name,
                imu_quat=imu_quat,
                imu_gyro=imu_gyro,
                last_action=self.actions_onnx,
                ref_dof_pos_mu=ref_dof_pos_raw,
                ref_dof_vel_mu=ref_dof_vel_raw,
                ref_to_onnx=self.ref_to_onnx,
            )
        else:  # velocity mode
            # Get velocity command directly from keyboard handler
            velocity_cmd = np.array([self.vx, self.vy, self.vyaw], dtype=np.float32)

            # Use velocity obs_builder
            self.obs_builder = self.velocity_obs_builder

            hist_obs = self.obs_builder.update_from_lowstate(
                q_by_name=q_by_name,
                dq_by_name=dq_by_name,
                imu_quat=imu_quat,
                imu_gyro=imu_gyro,
                last_action=self.actions_onnx,
                velocity_command=velocity_cmd,
            )

        policy_obs_np = hist_obs[None, :]

        # Run ONNX inference with the appropriate policy session and correct input/output names
        if self.current_policy_mode == "velocity":
            input_feed = {self.velocity_input_name: policy_obs_np}
            onnx_output = self.velocity_policy_session.run([self.velocity_output_name], input_feed)
        else:  # motion mode
            input_feed = {self.motion_input_name: policy_obs_np}
            onnx_output = self.motion_policy_session.run([self.motion_output_name], input_feed)
            
        self.actions_onnx = onnx_output[0].reshape(-1)
        
        self.target_dof_pos_onnx = (
            self.actions_onnx * self.action_scale_onnx + self.default_angles_onnx
        )
        self.target_dof_pos_real = self.target_dof_pos_onnx[self.onnx_to_real]
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
        if hasattr(self, 'target_dof_pos_by_name') and self.target_dof_pos_by_name:
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


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = PolicyNodeJustDance()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
