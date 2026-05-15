"""Launch HoloMotion 29DOF from a deployment profile."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from humanoid_policy.launch_profile import as_bool
from humanoid_policy.launch_profile import find_profile_roots
from humanoid_policy.launch_profile import load_effective_profile
from humanoid_policy.launch_profile import policy_node_parameters
from humanoid_policy.launch_profile import print_effective_profile
from humanoid_policy.launch_profile import resolve_existing_path
from humanoid_policy.launch_profile import resolve_python_executable
from humanoid_policy.launch_profile import unique_paths
from humanoid_policy.launch_profile import validate_robot_config_file
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_profile",
                default_value="launch_profiles/orin_docker.yaml",
                description="Path to a HoloMotion launch profile YAML file.",
            ),
            DeclareLaunchArgument(
                "profile_override",
                default_value="",
                description=(
                    "Optional override as a YAML file, YAML/JSON mapping string, "
                    "or comma-separated dotted key=value pairs."
                ),
            ),
            DeclareLaunchArgument(
                "enable_recording",
                default_value="",
                description=(
                    "Legacy recording override. Prefer "
                    "profile_override:=recording.enabled=true."
                ),
            ),
            OpaqueFunction(function=_launch_from_profile),
        ]
    )


def _launch_from_profile(context, *args, **kwargs):
    del args, kwargs
    pkg_dir = Path(get_package_share_directory("humanoid_control")).resolve()
    profile_roots = unique_paths([Path.cwd(), *find_profile_roots(pkg_dir), pkg_dir])

    profile_arg = LaunchConfiguration("launch_profile").perform(context)
    override_arg = LaunchConfiguration("profile_override").perform(context).strip()
    legacy_recording_arg = LaunchConfiguration("enable_recording").perform(context).strip()

    profile_path, profile = load_effective_profile(
        profile_arg,
        override_arg=override_arg,
        legacy_recording_arg=legacy_recording_arg,
        roots=profile_roots,
    )
    config_file = resolve_existing_path(
        str(profile["robot"]["config_file"]),
        unique_paths([Path.cwd(), pkg_dir, *find_profile_roots(pkg_dir)]),
    )
    python_executable = resolve_python_executable(
        profile["runtime"].get("python_executable", "")
    )
    network_interface = str(profile["robot"]["network_interface"])
    validate_robot_config_file(config_file)

    print_effective_profile(
        profile_path,
        override_arg,
        legacy_recording_arg,
        profile,
        config_file,
    )

    actions = [
        SetEnvironmentVariable(
            name="CYCLONEDDS_URI",
            value=(
                "<CycloneDDS><Domain><General>"
                f"<NetworkInterfaceAddress>{network_interface}</NetworkInterfaceAddress>"
                "</General></Domain></CycloneDDS>"
            ),
        ),
        Node(
            package="humanoid_control",
            executable="humanoid_control",
            name="main_node",
            parameters=[{"config_path": str(config_file)}],
            output="screen",
        ),
        Node(
            package="humanoid_control",
            executable="policy_node_29dof",
            name="policy_node",
            parameters=[policy_node_parameters(profile, config_file)],
            output="screen",
            prefix=python_executable,
        ),
    ]

    if as_bool(profile["recording"].get("enabled", False)):
        actions.append(_recording_process(profile, config_file))

    return actions


def _recording_process(profile: dict, robot_config_file: Path) -> ExecuteProcess:
    recording = profile["recording"]
    bag_dir = Path(str(recording.get("bag_dir", "./bag_record"))).expanduser()
    bag_dir.mkdir(parents=True, exist_ok=True)
    bag_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{robot_config_file.stem}"
    topics = [str(topic) for topic in recording.get("topics", [])]
    if not topics:
        raise ValueError("recording.topics must contain at least one topic when recording is enabled")

    return ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--storage",
            str(recording.get("storage", "mcap")),
            "-o",
            str(bag_dir / bag_name),
            *topics,
        ],
        output="screen",
    )
