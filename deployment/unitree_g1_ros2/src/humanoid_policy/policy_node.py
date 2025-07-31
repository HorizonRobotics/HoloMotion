#! /home/dpx/miniconda3/envs/humanoid_deploy/bin/python
# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""HoloMotion Policy Node.

This module implements the main policy execution node for the HoloMotion
humanoid robot system. It handles neural network policy inference, motion
sequence management, remote controller input, and robot state coordination
for complex humanoid behaviors including dancing and locomotion.

The policy node serves as the high-level decision maker that:
- Processes sensor observations and builds state representations
- Executes trained neural network policies for motion generation
- Manages multiple motion sequences (stand, squat, dance moves)
- Handles remote controller input for motion selection
- Coordinates with the main control node for safe operation

Author: HoloMotion Team
License: See project LICENSE file
"""

import os

import easydict
import joblib
import numpy as np
import onnxruntime
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, Int32, String
from unitree_hg.msg import (
    LowState,
)

from humanoid_policy.utils.remote_controller_filter import (
    KeyMap,
    RemoteController,
)
from humanoid_policy.utils.rotation_helper import (
    get_gravity_orientation,
)
from humanoid_policy.utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    quat_mul,
    quat_rotate,
    wrap_to_pi,
)


@torch.compile
def quat_to_tan_norm(q: torch.Tensor, w_last: bool) -> torch.Tensor:
    """Converts quaternion to tangent-normal representation.

    Args:
        q: Input quaternion tensor of shape [..., 4]
        w_last: Whether the quaternion format is [x,y,z,w] (True)
            or [w,x,y,z] (False)

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


class PolicyNodeJustDance(Node):
    """Main policy execution node for HoloMotion humanoid robot control.

    This node implements the high-level control logic for a humanoid robot
    capable of performing various motion sequences including standing,
    squatting, and dance moves. It uses trained neural network policies for
    motion generation and coordinates with hardware control nodes for safe
    execution.

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
        """Initialize the policy node with configuration, models, and ROS2.

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

        self.music_flag = False
        self.current_motion = 0  # 0: first motion, 1: second motion
        self.state_mode = 0  # 0: standby, 1: start dance

        # Safety check related flags
        self.policy_enabled = False  # Controls whether policy is enabled
        self.stand_completed = False  # Marks whether stand motion is completed
        self.squat_state = True  # Marks whether robot is in squat state
        self.walking_mode = False  # Controls whether walking mode is executed
        self.can_activate_walking = (
            False  # Whether walking mode can be activated
        )

        # Robot state related flags
        # Marks whether MOVE_TO_DEFAULT state is received, allowing key
        # operations
        self.robot_state_ready = False

        self.num_bodies_extend = self.config.num_bodies_extend

        self.policy_context_length = self.config.context_length
        self.max_context_length = self.policy_context_length
        self.performance_completed = False  # [T, 4]
        self.on_action = False

        self._setup_data()
        self._setup_models()
        self._setup_motion_data()
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_timers()
        self._reset_policy()

    def _setup_data(self):
        """Initialize data structures and parameters for observation.

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
            [
                self.config.default_joint_angles[dof_name]
                for dof_name in self.config.policy_dof_order
            ]
        )
        self.control_decimation = self.config.control_decimation
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.hist_valid_queue = np.zeros(
            self.max_context_length, dtype=np.bool
        )

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
        self.hist_policy_obs = np.zeros(
            sum(
                self.config.obs_dims[key] * self.config.context_length
                for key in self.config.history_obs.keys()
            ),
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
        The model is loaded from the configured path and optimized for
        real-time inference with CUDA backend when available.

        Raises:
            RuntimeError: If model loading fails or ONNX Runtime setup fails
        """
        self.get_logger().info(
            f"Loading policy from {self.config.policy_path} ..."
        )
        model_path = os.path.join(
            get_package_share_directory("humanoid_control"),
            "models",
            self.config.policy_path,
        )

        # Setup ONNX Runtime session
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

        try:
            self.policy_session = onnxruntime.InferenceSession(
                model_path, providers=providers
            )
            self.get_logger().info(
                f"ONNX Runtime session created using: "
                f"{self.policy_session.get_providers()}"
            )
            self.policy_input_name = self.policy_session.get_inputs()[0].name
            self.policy_output_name = self.policy_session.get_outputs()[0].name
            self.get_logger().info(
                f"Policy ONNX Input: {self.policy_input_name}, "
                f"Output: {self.policy_output_name}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            raise

        self.get_logger().info("Policy loaded successfully.")

    def _setup_motion_data(self):
        """Load and prepare motion sequence data for all available motions.

        Loads motion data for 6 different sequences:
        - B button: Stand up motion
        - X button: Squat motion
        - UP button: jog in place
        - DOWN button: practice martial arts
        - RIGHT button: stretch body
        - LEFT button: bow

        Each motion contains:
        - Joint positions and velocities over time
        - Root body poses and velocities
        - Body link positions and orientations
        - Timing and synchronization data
        """
        # Load all six motion files
        motion_paths = [
            os.path.join(
                get_package_share_directory("humanoid_control"),
                "motion_data",
                motion_file,
            )
            for motion_file in [
                self.config.motion_file,  # B button
                self.config.motion_file_3,  # X button
                self.config.motion_file_4,  # UP button
                self.config.motion_file_5,  # DOWN button
                self.config.motion_file_6,  # RIGHT button
                self.config.motion_file_7,  # LEFT button
            ]
        ]

        self.motions = []
        self.motion_frames = []  # Store number of frames for each motion
        for path in motion_paths:
            motion = joblib.load(path)
            motion_name = list(motion.keys())[0]
            motion_data = motion[motion_name]

            # transform the np arrays to torch tensors
            for k, v in motion_data.items():
                if isinstance(v, np.ndarray):
                    motion_data[k] = torch.from_numpy(v)

            self.motions.append(motion_data)
            self.motion_frames.append(
                motion_data["dof_pos"].shape[0]
            )  # Store frame count

        # Initialize with first motion
        self.motion_data = self.motions[0]
        self.num_motion_frames = self.motion_frames[0]
        self.motion_fps = self.motion_data["fps"]
        self.num_fut_frames = self.config.num_fut_frames

        self.ref_initial_base_rot_quat = self.motion_data["root_rot"][
            0:1, :
        ]  # [1, 4]
        self.ref_initial_heading_inv_quat = calc_heading_quat_inv(
            self.ref_initial_base_rot_quat,
            w_last=True,
        )  # [1, 4]
        self.ref_initial_heading_inv_quat_expanded = (
            self.ref_initial_heading_inv_quat.repeat(
                self.num_fut_frames,
                1,
            ).reshape(self.num_fut_frames, 4)
        )

    def _setup_subscribers(self):
        """Set up ROS2 subscribers for robot state and remote controller."""
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
        """Set up ROS2 publishers for action commands and status."""

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

    def _setup_timers(self):
        """Set up ROS2 timer for main execution loop."""
        self.create_timer(self.dt, self.run)

    def _update_obs_queue(self, current_obs_dict: dict):
        """Update observation history queues with current observations.

        Args:
            current_obs_dict: Dictionary containing current observation data
                with keys matching configured observation types
        """
        for key in self.obs_queues.keys():
            obs_item = current_obs_dict[key]
            if self.obs_queues[key] is None:
                self.obs_queues[key] = np.zeros(
                    (self.max_context_length, obs_item.shape[0]),
                    dtype=np.float32,
                )
            else:
                self.obs_queues[key][1:] = self.obs_queues[key][:-1].copy()
                self.obs_queues[key][0] = (
                    obs_item * self.config.obs_scales[key]
                )
        self.hist_valid_queue[1:] = self.hist_valid_queue[:-1].copy()
        self.hist_valid_queue[0] = True

        self.hist_policy_obs = np.concatenate(
            [
                np.array(
                    self.obs_queues[key][: self.policy_context_length]
                ).reshape(-1)
                for key in sorted(self.config.history_obs.keys())
            ],
            axis=0,
        )

    def _robot_state_callback(self, msg: String):
        """Handle robot state messages for safety coordination.

        Processes robot state updates from the main control node to ensure
        safe operation. Button operations are only allowed when the robot
        is in MOVE_TO_DEFAULT state.

        Args:
            msg: String message containing robot state information
                Valid states: ZERO_TORQUE, MOVE_TO_DEFAULT, EMERGENCY_STOP,
                POLICY
        """
        robot_state = msg.data
        # self.get_logger().info(f"Received robot state: {robot_state}")

        # Only allow button operations when robot state is MOVE_TO_DEFAULT
        if robot_state == "MOVE_TO_DEFAULT":
            self.robot_state_ready = True
            # self.get_logger().info(
            #     "Robot state ready: Button operations enabled"
            # )
        elif robot_state == "ZERO_TORQUE":
            self.robot_state_ready = False
            # self.get_logger().info(
            #     "Robot state not ready: Button operations disabled"
            # )
        elif robot_state == "EMERGENCY_STOP":
            self.robot_state_ready = False
            # self.get_logger().warn(
            #     "Emergency stop detected: Button operations disabled"
            # )

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
            ls_msg: LowState message containing robot sensor data and remote
                controller input
        """
        self.low_state_buffer_ = ls_msg
        self.remote_controller.set(ls_msg.wireless_remote)

        # A or start button: Toggle policy enable state
        if (
            self.remote_controller.button[KeyMap.A] == 1
            and (not self.on_action)
            and self.robot_state_ready
        ):
            # if self.remote_controller.button[KeyMap.A] == 1:
            # self.on_action = True
            self.policy_enabled = True
            self.stand_completed = False  # Reset stand state
            self.squat_state = True  # Reset squat state
            self.state_mode = 0
            self._reset_counter()
            self.motion_data = self.motions[0]
            self.num_motion_frames = self.motion_frames[0]
            # Reset walking related flags
            self.walking_mode = False
            self.can_activate_walking = False
            self.get_logger().info(
                f"Policy {'enabled' if self.policy_enabled else 'disabled'}"
            )

        # Only process other buttons when policy is enabled and robot state is
        # ready
        if self.policy_enabled and self.robot_state_ready:
            # B button (stand) is always available when policy is enabled
            if self.remote_controller.button[KeyMap.B] == 1 and (
                not self.on_action
            ):
                self._start_motion(0, "~/music.sh")
                self.stand_completed = False
                self.on_action = True
                # self.can_activate_walking = True  # After B button pressed,
                # allow walking mode activation after full stand
                self.get_logger().info("Starting stand up motion")

            # If in squat state, only B button is available
            elif self.squat_state:
                return

            # Other motions are only available after standing is completed and
            # not in squat state
            elif self.stand_completed:
                if (
                    self.remote_controller.button[KeyMap.X] == 1
                    and self.performance_completed
                ):
                    self._start_motion(1, "~/music3.sh")
                    self.walking_mode = False
                    self.can_activate_walking = False
                    self.get_logger().info("Starting squat motion")
                    self.performance_completed = False
                    self.on_action = True
                elif (
                    self.remote_controller.button[KeyMap.up] == 1
                    and self.performance_completed
                ):
                    self._start_motion(2, "~/music4.sh")
                    self.walking_mode = False
                    self.performance_completed = False
                    self.on_action = True
                    # self.can_activate_walking = False
                elif (
                    self.remote_controller.button[KeyMap.down] == 1
                    and self.performance_completed
                ):
                    self._start_motion(3, "~/music5.sh")
                    self.walking_mode = False
                    self.performance_completed = False
                    self.on_action = True
                    # self.can_activate_walking = False
                elif (
                    self.remote_controller.button[KeyMap.right] == 1
                    and self.performance_completed
                ):
                    self._start_motion(4, "~/music6.sh")
                    self.walking_mode = False
                    self.performance_completed = False
                    self.on_action = True
                    # self.can_activate_walking = False
                elif (
                    self.remote_controller.button[KeyMap.left] == 1
                    and self.performance_completed
                ):
                    self._start_motion(5, "~/music7.sh")
                    self.walking_mode = False
                    self.performance_completed = False
                    self.on_action = True
                    # self.can_activate_walking = False

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
                self.get_logger().warn(
                    "Robot not in MOVE_TO_DEFAULT state. Button operations "
                    "disabled."
                )

        self.vx, self.vy, self.vyaw = (
            self.remote_controller.get_velocity_commands()
        )

    def _start_motion(self, motion_index: int, music_script: str):
        """Start execution of a specific motion sequence.

        Initializes motion playback including:
        - Loading motion data and resetting counters
        - Publishing motion metadata
        - Starting synchronized music playback

        Args:
            motion_index: Index of motion to execute (0-5)
                0: Stand up, 1: Squat, 2-5: Dance motions
            music_script: Path to script for synchronized music playback
        """
        self.state_mode = 1
        self.current_motion = motion_index
        self.motion_data = self.motions[motion_index]
        self.num_motion_frames = self.motion_frames[motion_index]
        self._reset_counter()

        # Publish total frames when starting new motion
        total_frames_msg = Int32()
        total_frames_msg.data = self.num_motion_frames
        self.total_frames_pub.publish(total_frames_msg)

    def _build_ref_motion_state(self):
        """Build reference motion state for policy input.

        Constructs the reference motion state observation that provides the
        policy with information about the desired future motion trajectory.
        This includes:
        - Future body orientations and positions
        - Joint position and velocity trajectories
        - Body link poses and velocities
        - All data transformed to robot's local coordinate frame

        Returns:
            numpy.ndarray: Flattened reference motion state vector for policy
                input. Contains concatenated future motion data over the
                prediction horizon
        """
        target_ref_frame_id = self.counter + 1
        max_target_ref_frame = self.num_motion_frames - self.num_fut_frames
        if target_ref_frame_id > max_target_ref_frame:
            target_ref_frame_id = max_target_ref_frame

        ref_fut_base_rot_quat = self.motion_data["root_rot"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ]  # [T, 4]
        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            ref_fut_base_rot_quat, w_last=True
        )  # [T, 4]
        ref_fut_heading_aligned_frame_quat = quat_mul(
            ref_fut_heading_quat_inv,
            ref_fut_base_rot_quat,
            w_last=True,
        )  # [T, 4]
        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_heading_aligned_frame_quat,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll)  # [T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch)  # [T, 1]

        ref_fut_in_initial_ref_heading = quat_mul(
            self.ref_initial_heading_inv_quat_expanded,
            ref_fut_base_rot_quat,
            w_last=True,
        )  # [T, 4]
        _, _, ref_fut_yaw = get_euler_xyz(
            ref_fut_in_initial_ref_heading,
            w_last=True,
        )  # [T, 1]
        ref_fut_yaw = wrap_to_pi(ref_fut_yaw)  # [T, 1]
        ref_fut_rpy = torch.cat(
            [ref_fut_roll, ref_fut_pitch, ref_fut_yaw], dim=-1
        )  # [T, 3]
        ref_fut_rpy_flat = ref_fut_rpy.reshape(1, -1)  # [1, T * 3]

        cur_ref_base_pos = self.motion_data["root_pos"][target_ref_frame_id][
            None, :
        ]  # [1, 3]
        cur_ref_base_rot_quat = self.motion_data["root_rot"][
            target_ref_frame_id
        ][None, :]  # [1, 4]
        cur_ref_base_rot_heading_quat_inv = calc_heading_quat_inv(
            cur_ref_base_rot_quat, w_last=True
        )  # [1, 4]
        cur_ref_base_rot_heading_quat_inv_body_flat = (
            cur_ref_base_rot_heading_quat_inv[:, None, None, :]
            .repeat(1, self.num_fut_frames, self.num_bodies_extend, 1)
            .reshape(-1, 4)
        )  # [T * num_bodies_extend, 4]
        fut_ref_bodylink_pos = self.motion_data["rg_pos_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...]  # [1, T, num_bodies_extend, 3]
        fut_rel_ref_bodylink_pos_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            (fut_ref_bodylink_pos - cur_ref_base_pos[:, None, None, :]).view(
                -1, 3
            ),
            w_last=True,
        ).reshape(1, -1)  # [1, T * num_bodies_extend * 3]

        fut_ref_bodylink_rot = self.motion_data["rg_rot_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...]  # [1, T, num_bodies_extend, 4]
        fut_rel_ref_bodylink_rot_quat = quat_mul(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )  # [T * num_bodies_extend ,4]
        fut_rel_ref_bodylink_rot_tannorm_flat = quat_to_tan_norm(
            fut_rel_ref_bodylink_rot_quat, w_last=True
        ).reshape(1, -1)  # [1, T * num_bodies_extend * 6]

        fut_ref_bodylink_vel = self.motion_data["body_vel_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...]  # [1, T, num_bodies_extend, 3]
        fut_rel_ref_bodylink_vel_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(1, -1)  # [1, T * num_bodies_extend * 3]

        fut_ref_bodylink_ang_vel = self.motion_data["body_ang_vel_t"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...]  # [1, T, num_bodies_extend, 3]
        fut_rel_ref_bodylink_ang_vel_flat = quat_rotate(
            cur_ref_base_rot_heading_quat_inv_body_flat,
            fut_ref_bodylink_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(1, -1)  # [1, T * num_bodies_extend * 3]

        fut_ref_dof_pos_flat = self.motion_data["dof_pos"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...].reshape(1, -1)  # [1, T * num_actions]
        fut_ref_dof_vel_flat = self.motion_data["dof_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...].reshape(1, -1)  # [1, T * num_actions]

        fut_ref_root_global_lin_vel = self.motion_data["root_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...].reshape(1, -1)  # [1, T * 3]
        fut_ref_rel_base_lin_vel = quat_rotate(
            cur_ref_base_rot_heading_quat_inv,
            fut_ref_root_global_lin_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(1, -1)  # [1, T * 3]

        fut_ref_root_global_ang_vel = self.motion_data["root_ang_vel"][
            target_ref_frame_id : target_ref_frame_id + self.num_fut_frames, :
        ][None, ...].reshape(1, -1)  # [1, T * 3]
        fut_ref_rel_base_ang_vel = quat_rotate(
            cur_ref_base_rot_heading_quat_inv,
            fut_ref_root_global_ang_vel.reshape(-1, 3),
            w_last=True,
        ).reshape(1, -1)  # [1, T * 3]

        ref_motion_state_flat = torch.cat(
            [
                ref_fut_rpy_flat,
                fut_ref_rel_base_lin_vel,
                fut_ref_rel_base_ang_vel,
                fut_ref_dof_pos_flat,
                fut_ref_dof_vel_flat,
                fut_rel_ref_bodylink_pos_flat,
                fut_rel_ref_bodylink_rot_tannorm_flat,
                fut_rel_ref_bodylink_vel_flat,
                fut_rel_ref_bodylink_ang_vel_flat,
            ],
            dim=-1,
        )
        return ref_motion_state_flat.flatten().cpu().numpy()

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

        cur_q = np.zeros(len(self.config.policy_dof_order))
        cur_dq = np.zeros(len(self.config.policy_dof_order))
        motor_state = self.low_state_buffer_.motor_state
        for i, dof_name in enumerate(self.config.policy_dof_order):
            motor_idx = self.config.dof2motor_idx_mapping[dof_name]
            cur_q[i] = (
                motor_state[motor_idx].q
                - self.config.default_joint_angles[dof_name]
            )
            cur_dq[i] = motor_state[motor_idx].dq
        quat = self.low_state_buffer_.imu_state.quaternion
        ang_vel = np.array(
            [self.low_state_buffer_.imu_state.gyroscope], dtype=np.float32
        ).reshape(-1)
        gravity_orientation = get_gravity_orientation(quat)

        if self.state_mode == 0:
            # self.counter = 351
            self.counter = 1
        elif self.state_mode == 1:
            # self.counter = self.num_motion_frames - self.num_fut_frames - 1
            self.music_flag = False

        ref_motion_state_flat = self._build_ref_motion_state()
        current_obs_dict = {
            "actions": self.actions.astype(np.float32),
            "base_ang_vel": ang_vel.astype(np.float32),
            "dof_pos": cur_q.astype(np.float32),
            "dof_vel": cur_dq.astype(np.float32),
            "projected_gravity": gravity_orientation.astype(np.float32),
            "ref_motion_state_flat": ref_motion_state_flat.astype(np.float32),
            "history_actor": self.hist_policy_obs.copy(),
        }
        self._update_obs_queue(current_obs_dict)

        self.latest_obs = np.concatenate(
            [
                current_obs_dict[key] * self.config.obs_scales[key]
                for key in sorted(self.config.current_obs)
            ],
            axis=0,
        )

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
        self._process_observation()
        if self.latest_obs is None:
            return

        # Convert to numpy for ONNX inference
        policy_obs_np = self.latest_obs[None, :].astype(np.float32)
        input_feed = {self.policy_input_name: policy_obs_np}

        # Run ONNX inference
        onnx_output = self.policy_session.run(
            [self.policy_output_name], input_feed
        )
        self.actions = onnx_output[0][0]

        if self.actions is not None:
            action_msg = Float32MultiArray()
            target_dof_pos = (
                self.actions * self.action_scale + self.default_dof_pos
            )
            action_msg.data = target_dof_pos.tolist()

            if np.isnan(target_dof_pos).any():
                self.get_logger().error("Action contains NaN values")
            self.action_pub.publish(action_msg)

        # Publish counter value
        counter_msg = Int32()
        counter_msg.data = self.counter
        self.counter_pub.publish(counter_msg)

        # Publish total frames (publish in every cycle to ensure data
        # availability)
        total_frames_msg = Int32()
        total_frames_msg.data = self.num_motion_frames
        self.total_frames_pub.publish(total_frames_msg)

        # Publish joystick data
        joystick_msg = Float32MultiArray()
        joystick_msg.data = [
            self.remote_controller.lx,
            self.remote_controller.ly,
            self.remote_controller.rx,
            self.remote_controller.ry,
        ]
        self.joystick_pub.publish(joystick_msg)

        # Get and publish velocity commands
        vx, vy, vyaw = self.remote_controller.get_velocity_commands()
        velocity_cmd_msg = Float32MultiArray()
        velocity_cmd_msg.data = [vx, vy, vyaw]
        self.velocity_cmd_pub.publish(velocity_cmd_msg)

        self.counter += 1

        # Check if stand up motion is completed
        if (
            self.current_motion == 0  # Stand up motion completed
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.stand_completed = True
            self.squat_state = False
            self.can_activate_walking = True
            self.get_logger().info(
                "Stand up completed, other motions now available"
            )
            self.performance_completed = True
            self.on_action = False

        # Check if squat motion is completed
        elif (
            self.current_motion == 1  # Squat motion completed
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.squat_state = True
            self.stand_completed = False
            self.walking_mode = False
            self.can_activate_walking = False
            self.on_action = False
            self.get_logger().info(
                "Squat completed, only stand up motion available"
            )
        elif (
            self.current_motion == 2  # Motion 1
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.performance_completed = True
            self.on_action = False
            self.get_logger().info(
                "action1 completed, other motions now available"
            )
        elif (
            self.current_motion == 3  # Motion 2
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.performance_completed = True
            self.on_action = False
            self.get_logger().info("action2, other motions now available")
        elif (
            self.current_motion == 4  # Motion 3
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.performance_completed = True
            self.on_action = False
            self.get_logger().info("action3, other motions now available")
        elif (
            self.current_motion == 5  # Motion 4
            and self.counter
            == self.num_motion_frames - self.num_fut_frames - 1
        ):
            self.performance_completed = True
            self.on_action = False
            self.get_logger().info("action4, other motions now available")


def main():
    """Main entry point for the policy node."""
    rclpy.init()
    policy_node = PolicyNodeJustDance()
    rclpy.spin(policy_node)


if __name__ == "__main__":
    main()
