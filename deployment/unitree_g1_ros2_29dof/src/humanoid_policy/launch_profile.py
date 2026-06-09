"""Shared launch profile parsing utilities for HoloMotion deployment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shlex
import sys

import yaml


DEFAULT_PROFILE = {
    "runtime": {
        "conda_setup": "",
        "ros_setup": "/opt/ros/humble/setup.sh",
        "unitree_setup": "/root/unitree_ros2/setup.sh",
        "deploy_env": "../../deploy.env",
        "build_before_launch": True,
        "python_executable": "${Deploy_CONDA_PREFIX}/bin/python",
        "cyclonedds_home": "",
        "extra_ld_library_paths": [],
    },
    "robot": {
        "config_file": "config/g1_29dof_holomotion.yaml",
        "network_interface": "eth0",
    },
    "recording": {
        "enabled": False,
        "storage": "mcap",
        "bag_dir": "./bag_record",
        "topics": ["/lowcmd", "/lowstate", "/humanoid/action"],
    },
    "policy": {
        "latest_obs_zmq_uri": "tcp://192.168.124.29:6001",
        "latest_obs_zmq_topic": "obs65",
        "latest_obs_zmq_mode": "connect",
        "latest_obs_zmq_conflate": True,
        "zmq_jitter_delay_frames": 5,
        "max_data_age": 0.6,
        "enable_teleop_reference": False,
        "inference_backend": "onnx",
        "motion_rope_max_seq_len": 0,
        "motion_rope_reset_margin": 64,
        "timing_debug_enabled": False,
        "timing_debug_log_interval_sec": 5.0,
        "timing_debug_log_per_loop": False,
        "cpu_affinity_main": "",
        "cpu_affinity_zmq_sub": "",
    },
}

ROBOT_CONFIG_RUNTIME_FIELDS = (
    "vr",
    "latest_obs_zmq_uri",
    "latest_obs_zmq_topic",
    "latest_obs_zmq_mode",
    "latest_obs_zmq_conflate",
    "zmq_jitter_delay_frames",
    "max_data_age",
    "enable_teleop_reference",
    "inference_backend",
    "motion_rope_max_seq_len",
    "motion_rope_reset_margin",
    "timing_debug_enabled",
    "timing_debug_log_interval_sec",
    "timing_debug_log_per_loop",
    "cpu_affinity_main",
    "cpu_affinity_zmq_sub",
    "network_interface",
    "python_executable",
    "enable_recording",
    "bag_dir",
    "recording",
)


def load_effective_profile(
    profile_arg: str,
    override_arg: str = "",
    legacy_recording_arg: str = "",
    roots: list[Path] | None = None,
) -> tuple[Path, dict]:
    search_roots = unique_paths(roots or [Path.cwd()])
    profile_path = resolve_existing_path(profile_arg, search_roots)
    profile = load_profile(profile_path)

    if override_arg:
        profile = deep_merge(
            profile,
            load_override(override_arg, [profile_path.parent, *search_roots]),
        )
    if legacy_recording_arg:
        profile = deep_merge(
            profile,
            {"recording": {"enabled": as_bool(legacy_recording_arg)}},
        )

    profile = expand_env(profile)
    drop_legacy_policy_fields(profile)
    return profile_path, profile


def drop_legacy_policy_fields(profile: dict) -> None:
    policy = profile.setdefault("policy", {})
    policy.pop("require_vr_data_for_motion", None)
    policy.pop("inference_device_id", None)
    policy.pop("tensorrt_fp16_enable", None)
    policy.pop("tensorrt_engine_cache_enable", None)
    policy.pop("tensorrt_engine_cache_path", None)


def load_profile(profile_path: Path) -> dict:
    with profile_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Launch profile must be a mapping: {profile_path}")
    return deep_merge(deepcopy(DEFAULT_PROFILE), loaded)


def load_override(value: str, roots: list[Path]) -> dict:
    if ";" in value:
        merged: dict = {}
        for item in value.split(";"):
            item = item.strip()
            if item:
                merged = deep_merge(merged, load_single_override(item, roots))
        return merged
    return load_single_override(value, roots)


def load_single_override(value: str, roots: list[Path]) -> dict:
    maybe_path = expand_string(value)
    if "=" not in value:
        path = try_resolve_existing_path(maybe_path, roots)
        if path is not None:
            with path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Profile override must be a mapping: {path}")
            return loaded

    loaded = yaml.safe_load(value)
    if isinstance(loaded, dict):
        return loaded

    return parse_dotted_overrides(value)


def parse_dotted_overrides(value: str) -> dict:
    result: dict = {}
    separator = ";" if ";" in value else ","
    for item in value.split(separator):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "profile_override entries must use dotted key=value syntax; "
                f"invalid entry: {item}"
            )
        key, raw = item.split("=", 1)
        target = result
        parts = [part.strip() for part in key.split(".") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid profile_override key: {key}")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    return result


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def expand_env(value):
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str):
        return expand_string(value)
    return value


def expand_string(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def find_profile_roots(pkg_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for seed in [pkg_dir, Path(__file__).resolve()]:
        for parent in [seed, *seed.parents]:
            if (parent / "launch_profiles").is_dir():
                roots.append(parent)
    return roots


def unique_paths(paths) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def resolve_existing_path(path_value: str, roots: list[Path]) -> Path:
    resolved = try_resolve_existing_path(path_value, roots)
    if resolved is None:
        candidates = ", ".join(str(path) for path in candidate_paths(path_value, roots))
        raise FileNotFoundError(f"Could not resolve path '{path_value}'. Tried: {candidates}")
    return resolved


def try_resolve_existing_path(path_value: str, roots: list[Path]) -> Path | None:
    for candidate in candidate_paths(path_value, roots):
        if candidate.exists():
            return candidate.resolve()
    return None


def candidate_paths(path_value: str, roots: list[Path]) -> list[Path]:
    expanded = Path(expand_string(path_value))
    if expanded.is_absolute():
        return [expanded]
    return [root / expanded for root in roots]


def resolve_python_executable(value: str) -> str:
    expanded = expand_string(value)
    if expanded and "$" not in expanded and Path(expanded).exists():
        return expanded

    deploy_prefix = os.environ.get("Deploy_CONDA_PREFIX", "")
    if deploy_prefix:
        candidate = Path(deploy_prefix) / "bin" / "python"
        if candidate.exists():
            print(
                "[holomotion_29dof_launch] runtime.python_executable was not usable; "
                f"falling back to {candidate}"
            )
            return str(candidate)

    print(
        "[holomotion_29dof_launch] runtime.python_executable was not usable; "
        f"falling back to current interpreter {sys.executable}"
    )
    return sys.executable


def load_robot_config_mapping(config_file: Path) -> dict:
    with config_file.open("r", encoding="utf-8") as stream:
        robot_config = yaml.safe_load(stream) or {}
    if not isinstance(robot_config, dict):
        raise ValueError(f"Robot config must be a mapping: {config_file}")
    return robot_config


def validate_robot_config_ownership(robot_config: dict, config_file: Path | str) -> None:
    forbidden = [key for key in ROBOT_CONFIG_RUNTIME_FIELDS if key in robot_config]
    if forbidden:
        fields = ", ".join(forbidden)
        raise ValueError(
            "Runtime/deployment fields are not allowed in robot config "
            f"{config_file}: {fields}. Move these fields to launch_profiles/orin_docker.yaml "
            "or the active launch profile."
        )


def validate_robot_config_file(config_file: Path) -> None:
    validate_robot_config_ownership(load_robot_config_mapping(config_file), config_file)


def policy_node_parameters(profile: dict, robot_config_file: Path) -> dict:
    params = {"config_path": str(robot_config_file)}
    for key, value in profile.get("policy", {}).items():
        if value is not None:
            params[key] = value
    return params


def print_effective_profile(
    profile_path: Path,
    override_arg: str,
    legacy_recording_arg: str,
    profile: dict,
    robot_config_file: Path,
) -> None:
    printable = deepcopy(profile)
    printable["_meta"] = {
        "launch_profile": str(profile_path),
        "profile_override": override_arg,
        "legacy_enable_recording": legacy_recording_arg,
        "robot_config": str(robot_config_file),
    }
    print(
        "[holomotion_29dof_launch] Configuration source rules:\n"
        "  Runtime/deployment fields come from launch profile.\n"
        f"  Robot/model fields come from {robot_config_file.name}."
    )
    print(
        "[holomotion_29dof_launch] Effective launch profile:\n"
        + yaml.safe_dump(printable, sort_keys=False)
    )


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def runtime_shell_assignments(
    profile_path: Path,
    profile: dict,
    roots: list[Path] | None = None,
) -> dict[str, str]:
    runtime = profile.get("runtime", {})
    profile_dir = profile_path.parent
    search_roots = unique_paths([profile_dir, *(roots or [])])

    def runtime_path(key: str) -> str:
        value = str(runtime.get(key, "") or "").strip()
        if not value:
            return ""
        path = Path(expand_string(value))
        if path.is_absolute():
            return str(path)
        for root in search_roots:
            candidate = root / path
            if candidate.exists():
                return str(candidate.resolve())
        return str((profile_dir / path).resolve())

    extra_paths = runtime.get("extra_ld_library_paths", []) or []
    if isinstance(extra_paths, str):
        extra_paths = [extra_paths]

    resolved_extra_paths = []
    for item in extra_paths:
        item_str = str(item).strip()
        if not item_str:
            continue
        path = Path(expand_string(item_str))
        if path.is_absolute():
            resolved_extra_paths.append(str(path))
        else:
            for root in search_roots:
                candidate = root / path
                if candidate.exists():
                    resolved_extra_paths.append(str(candidate.resolve()))
                    break
            else:
                resolved_extra_paths.append(str((profile_dir / path).resolve()))

    return {
        "HLM_RUNTIME_CONDA_SETUP": runtime_path("conda_setup"),
        "HLM_RUNTIME_ROS_SETUP": runtime_path("ros_setup"),
        "HLM_RUNTIME_UNITREE_SETUP": runtime_path("unitree_setup"),
        "HLM_RUNTIME_DEPLOY_ENV": runtime_path("deploy_env"),
        "HLM_RUNTIME_BUILD_BEFORE_LAUNCH": (
            "true" if as_bool(runtime.get("build_before_launch", True)) else "false"
        ),
        "HLM_RUNTIME_CYCLONEDDS_HOME": runtime_path("cyclonedds_home"),
        "HLM_RUNTIME_EXTRA_LD_LIBRARY_PATHS": ":".join(resolved_extra_paths),
    }


def print_shell_assignments(assignments: dict[str, str]) -> None:
    for key, value in assignments.items():
        print(f"{key}={shlex.quote(str(value))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a HoloMotion launch profile.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_parser = subparsers.add_parser(
        "runtime-shell",
        help="Print shell assignments for runtime fields.",
    )
    runtime_parser.add_argument("--profile", required=True)
    runtime_parser.add_argument("--override", default="")
    runtime_parser.add_argument("--root", action="append", default=[])

    args = parser.parse_args(argv)
    if args.command == "runtime-shell":
        roots = unique_paths([Path(root) for root in args.root] + [Path.cwd()])
        profile_path, profile = load_effective_profile(
            args.profile,
            override_arg=args.override,
            roots=roots,
        )
        print_shell_assignments(runtime_shell_assignments(profile_path, profile, roots=roots))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
