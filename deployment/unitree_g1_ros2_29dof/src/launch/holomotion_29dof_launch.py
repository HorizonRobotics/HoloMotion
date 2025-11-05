"""
HoloMotion ROS2 Launch Configuration

This module defines the ROS2 launch configuration for the HoloMotion humanoid robot control system.
It sets up a complete robotics pipeline including robot control, motion policy execution, and data recording
for the Unitree G1 humanoid robot.

The launch file coordinates three main components:
1. Main control node (C++) - Handles low-level robot control and communication
2. Policy node (Python) - Executes motion policies and high-level decision making  
3. Recording node - Captures sensor data and commands for analysis

Key Features:
- Configures network interface for robot communication
- Sets up CycloneDDS middleware with specific network interface
- Launches coordinated multi-node system with shared configuration
- Automatically records operational data with timestamped bags

Author: HoloMotion Team
License: See project LICENSE file
"""

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition


def generate_launch_description():
    """
    Generate the complete launch description for the HoloMotion humanoid control system.
    
    This function creates a comprehensive ROS2 launch configuration that coordinates
    multiple nodes required for humanoid robot operation. It sets up the necessary
    environment, launches control nodes, and optionally initiates data recording.
    
    Network Configuration:
        - Uses specific network interface (eth0) for robot communication
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
        
        3. Recording Node (Optional):
           - Automatically captures all relevant system data when enabled
           - Records sensor states, commands, and system metrics
           - Creates timestamped bag files for later analysis
    
    Configuration:
        - Robot: Unitree G1 with 29 DOF configuration
        - Config file: g1_29dof_holomotion.yaml
        - Recording format: MCAP for efficient data storage
        - Recording: Disabled by default, can be enabled with --record parameter
    
    Recorded Topics (when recording enabled):
        - /lowcmd: Low-level motor commands sent to robot
        - /lowstate: Robot sensor feedback and joint states  
        - /humanoid/action: High-level action commands from policy
    
    Parameters:
        - enable_recording: Boolean flag to enable/disable topic recording (default: false)
    
    Returns:
        LaunchDescription: Complete ROS2 launch configuration with all nodes,
                          environment variables, and optional recording setup
    
    Raises:
        FileNotFoundError: If the configuration file cannot be located
        PermissionError: If unable to create recording directory
    
    Example:
        Launch without recording (default):
        $ ros2 launch humanoid_control holomotion_29dof.launch.py
        
        Launch with recording enabled:
        $ ros2 launch humanoid_control holomotion_29dof.launch.py enable_recording:=true
        
        Or using the shell script:
        $ ./launch_holomotion_29dof.sh --record
    """
    # Declare launch arguments
    enable_recording_arg = DeclareLaunchArgument(
        'enable_recording',
        default_value='false',
        description='Enable topic recording (true/false)'
    )
    
    network_interface = "eth0"
    config_name = "g1_29dof_holomotion.yaml" 
    
    pkg_dir = get_package_share_directory("humanoid_control")
    config_file = os.path.join(pkg_dir, "config", config_name)
    python_exec = os.environ.get("PYTHON_EXECUTABLE", "python3")
    close_ai = ExecuteProcess(
        cmd=[
            python_exec,
            os.path.join(pkg_dir, "scripts", "close_ai_sport.py"),
            network_interface,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            # Declare launch arguments
            enable_recording_arg,
            
            # Main control node (C++)
            SetEnvironmentVariable(
                name="CYCLONEDDS_URI",
                value=f"<CycloneDDS><Domain><General><NetworkInterfaceAddress>{network_interface}</NetworkInterfaceAddress></General></Domain></CycloneDDS>",
            ),
            close_ai,
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
                executable="policy_node_29dof",
                name="policy_node",
                parameters=[{"config_path": config_file}],
                output="screen",
            ),
            # Recording node (conditional)
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
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration('enable_recording')),
            ),
        ]
    )
