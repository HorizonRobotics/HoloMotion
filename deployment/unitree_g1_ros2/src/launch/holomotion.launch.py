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

"""HoloMotion ROS2 Launch Configuration.

This module defines the ROS2 launch configuration for the HoloMotion
humanoid robot control system. It sets up a complete robotics pipeline
including robot control, motion policy execution, and data recording for
the Unitree G1 humanoid robot.

The launch file coordinates three main components:
1. Main control node (C++) - Handles low-level robot control and
   communication
2. Policy node (Python) - Executes motion policies and high-level
   decision making
3. Recording node - Captures sensor data and commands for analysis

Key Features:
- Configures network interface for robot communication
- Sets up CycloneDDS middleware with specific network interface
- Launches coordinated multi-node system with shared configuration
- Automatically records operational data with timestamped bags

Author: HoloMotion Team
License: See project LICENSE file
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the complete launch description for the HoloMotion.

    This function creates a comprehensive ROS2 launch configuration that
    coordinates multiple nodes required for humanoid robot operation.

    Network Configuration:
        - Uses specific network interface (enxf8e43ba00afd) for robot
          communication
        - Configures CycloneDDS middleware to use designated network interface
        - Ensures proper isolation and communication with the robot hardware

    Node Architecture:
        1. Main Control Node (C++):
           - Handles real-time robot control and sensor data processing
           - Manages low-level motor commands and feedback loops
           - Interfaces directly with robot hardware via configured network

        2. Policy Node (Python):
           - Executes trained motion policies for humanoid locomotion
           - Processes high-level commands and translates to robot actions
           - Handles motion planning and behavior coordination

        3. Recording Node:
           - Automatically captures all relevant system data
           - Records sensor states, commands, and system metrics
           - Creates timestamped bag files for later analysis

    Configuration:
        - Robot: Unitree G1 with 21 DOF (6+6+1+4+4) configuration
        - Config file: g1_21dof_holomotion.yaml
        - Recording format: MCAP for efficient data storage

    Recorded Topics:
        - /lowcmd: Low-level motor commands sent to robot
        - /lowstate: Robot sensor feedback and joint states
        - /humanoid/action: High-level action commands from policy
        - /current_count: Motion frame counting for synchronization
        - /joystick_data: Human operator input commands
        - /odommodestate: Robot odometry and pose information
        - /total_motion_frames: Total motion sequence length
        - /velocity_commands: Desired velocity commands

    Returns:
        LaunchDescription: Complete ROS2 launch configuration with all nodes,
            environment variables, and recording setup

    Raises:
        FileNotFoundError: If the configuration file cannot be located
        PermissionError: If unable to create recording directory

    Example:
        This launch file is typically executed via:
            $ ros2 launch humanoid_control holomotion.launch.py
    """
    network_interface = "enxf8e43ba00afd"
    config_name = "g1_21dof_holomotion.yaml"

    pkg_dir = get_package_share_directory("humanoid_control")
    config_file = os.path.join(pkg_dir, "config", config_name)

    return LaunchDescription(
        [
            # Main control node (C++)
            SetEnvironmentVariable(
                name="CYCLONEDDS_URI",
                value=f"<CycloneDDS><Domain><General><NetworkInterfaceAddress>{network_interface}</NetworkInterfaceAddress></General></Domain></CycloneDDS>",
            ),
            Node(
                package="humanoid_control",
                executable="humanoid_control",
                name="main_node",
                parameters=[{"config_path": config_file}],
                output="screen",
            ),
            # Policy node (Python)
            Node(
                package="humanoid_control",
                executable="policy_node",
                name="policy_node",
                parameters=[{"config_path": config_file}],
                output="screen",
            ),
            # Recording node
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "--storage",
                    "mcap",
                    "-o",
                    (
                        "./bag_record/"
                        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        + "_"
                        + config_name.split(".")[0]
                    ),
                    "/lowcmd",
                    "/lowstate",
                    "/humanoid/action",
                    "/current_count",
                    "/joystick_data",
                    "/odommodestate",
                    "/total_motion_frames",
                    "/velocity_commands",
                ],
                output="screen",
            ),
        ]
    )
