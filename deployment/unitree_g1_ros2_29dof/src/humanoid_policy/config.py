"""Configuration ownership helpers for the 29DOF policy runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from humanoid_policy.launch_profile import validate_robot_config_ownership


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _declare_value(node, name: str, default: Any) -> Any:
    return node.declare_parameter(name, default).value


def _required(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required robot config field '{key}' in {path}")
    return raw[key]


def _as_str(raw: dict[str, Any], key: str, path: Path, default: str | None = None) -> str:
    value = raw.get(key, default) if default is not None else _required(raw, key, path)
    value = str(value).strip()
    if not value:
        raise ValueError(f"Robot config field '{key}' must be a non-empty string")
    return value


def _as_int(raw: dict[str, Any], key: str, path: Path, default: int | None = None) -> int:
    value = raw.get(key, default) if default is not None else _required(raw, key, path)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Robot config field '{key}' must be an integer") from exc
    return parsed


def _as_float(raw: dict[str, Any], key: str, path: Path, default: float | None = None) -> float:
    value = raw.get(key, default) if default is not None else _required(raw, key, path)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Robot config field '{key}' must be a number") from exc
    return parsed


def _as_string_list(raw: dict[str, Any], key: str, path: Path) -> list[str]:
    value = _required(raw, key, path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Robot config field '{key}' must be a non-empty list")
    return [str(item) for item in value]


def _as_int_map(raw: dict[str, Any], key: str, path: Path) -> dict[str, int]:
    value = _required(raw, key, path)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Robot config field '{key}' must be a non-empty mapping")
    return {str(item_key): int(item_value) for item_key, item_value in value.items()}


def _as_float_map(raw: dict[str, Any], key: str, path: Path) -> dict[str, float]:
    value = _required(raw, key, path)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Robot config field '{key}' must be a non-empty mapping")
    return {str(item_key): float(item_value) for item_key, item_value in value.items()}


def _as_mapping(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = _required(raw, key, path)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Robot config field '{key}' must be a non-empty mapping")
    return value


@dataclass(frozen=True)
class DeploymentConfig:
    """Runtime/deployment fields supplied by launch profile ROS parameters."""

    robot_config_path: Path
    latest_obs_zmq_uri: str = "tcp://192.168.124.29:6001"
    latest_obs_zmq_topic: str = "obs65"
    latest_obs_zmq_mode: str = "connect"
    latest_obs_zmq_conflate: bool = True
    zmq_jitter_delay_frames: int = 5
    max_data_age: float = 0.6
    enable_teleop_reference: bool = False
    inference_backend: str = "onnx"
    timing_debug_enabled: bool = False
    timing_debug_log_interval_sec: float = 5.0
    timing_debug_log_per_loop: bool = False
    cpu_affinity_main: str = ""
    cpu_affinity_zmq_sub: str = ""

    @classmethod
    def from_node(cls, node) -> "DeploymentConfig":
        config_path = str(_declare_value(node, "config_path", "") or "").strip()
        if not config_path:
            raise ValueError(
                "config_path ROS parameter is required. Launch with a profile or pass "
                "config_path:=/path/to/g1_29dof_holomotion.yaml."
            )

        config = cls(
            robot_config_path=Path(config_path).expanduser(),
            latest_obs_zmq_uri=str(
                _declare_value(node, "latest_obs_zmq_uri", cls.latest_obs_zmq_uri)
            ),
            latest_obs_zmq_topic=str(
                _declare_value(node, "latest_obs_zmq_topic", cls.latest_obs_zmq_topic)
            ),
            latest_obs_zmq_mode=str(
                _declare_value(node, "latest_obs_zmq_mode", cls.latest_obs_zmq_mode)
            ),
            latest_obs_zmq_conflate=_as_bool(
                _declare_value(
                    node,
                    "latest_obs_zmq_conflate",
                    cls.latest_obs_zmq_conflate,
                ),
                cls.latest_obs_zmq_conflate,
            ),
            zmq_jitter_delay_frames=int(
                _declare_value(
                    node,
                    "zmq_jitter_delay_frames",
                    cls.zmq_jitter_delay_frames,
                )
            ),
            max_data_age=float(_declare_value(node, "max_data_age", cls.max_data_age)),
            enable_teleop_reference=_as_bool(
                _declare_value(
                    node,
                    "enable_teleop_reference",
                    cls.enable_teleop_reference,
                ),
                cls.enable_teleop_reference,
            ),
            inference_backend=str(
                _declare_value(node, "inference_backend", cls.inference_backend)
            ),
            timing_debug_enabled=_as_bool(
                _declare_value(
                    node,
                    "timing_debug_enabled",
                    cls.timing_debug_enabled,
                ),
                cls.timing_debug_enabled,
            ),
            timing_debug_log_interval_sec=float(
                _declare_value(
                    node,
                    "timing_debug_log_interval_sec",
                    cls.timing_debug_log_interval_sec,
                )
            ),
            timing_debug_log_per_loop=_as_bool(
                _declare_value(
                    node,
                    "timing_debug_log_per_loop",
                    cls.timing_debug_log_per_loop,
                ),
                cls.timing_debug_log_per_loop,
            ),
            cpu_affinity_main=str(
                _declare_value(node, "cpu_affinity_main", cls.cpu_affinity_main) or ""
            ),
            cpu_affinity_zmq_sub=str(
                _declare_value(
                    node,
                    "cpu_affinity_zmq_sub",
                    cls.cpu_affinity_zmq_sub,
                )
                or ""
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        mode = self.latest_obs_zmq_mode.strip().lower()
        if mode not in {"bind", "connect"}:
            raise ValueError("latest_obs_zmq_mode must be 'bind' or 'connect'")
        if self.zmq_jitter_delay_frames < 0:
            raise ValueError("zmq_jitter_delay_frames must be >= 0")
        if self.max_data_age <= 0.0:
            raise ValueError("max_data_age must be > 0")
        if self.inference_backend.strip().lower() not in {"onnx", "tensorrt"}:
            raise ValueError("inference_backend must be 'onnx' or 'tensorrt'")
        if self.timing_debug_log_interval_sec <= 0.0:
            raise ValueError("timing_debug_log_interval_sec must be > 0")

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "robot_config_path": str(self.robot_config_path),
            "latest_obs_zmq_uri": self.latest_obs_zmq_uri,
            "latest_obs_zmq_topic": self.latest_obs_zmq_topic,
            "latest_obs_zmq_mode": self.latest_obs_zmq_mode,
            "latest_obs_zmq_conflate": self.latest_obs_zmq_conflate,
            "zmq_jitter_delay_frames": self.zmq_jitter_delay_frames,
            "max_data_age": self.max_data_age,
            "enable_teleop_reference": self.enable_teleop_reference,
            "inference_backend": self.inference_backend,
            "timing_debug_enabled": self.timing_debug_enabled,
            "timing_debug_log_interval_sec": self.timing_debug_log_interval_sec,
            "timing_debug_log_per_loop": self.timing_debug_log_per_loop,
            "cpu_affinity_main": self.cpu_affinity_main,
            "cpu_affinity_zmq_sub": self.cpu_affinity_zmq_sub,
        }


@dataclass(frozen=True)
class RobotConfig:
    """Robot/model fields supplied by g1_29dof_holomotion.yaml."""

    path: Path
    device: str
    policy_freq: int
    control_freq: float
    lowstate_topic: str
    action_topic: str
    velocity_tracking_model_folder: str
    motion_tracking_model_folder: str
    motion_clip_dir: str
    onnx_intra_op_threads: int
    complete_dof_order: list[str]
    policy_dof_order: list[str]
    dof2motor_idx_mapping: dict[str, int]
    default_joint_angles: dict[str, float]
    joint_limits: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "RobotConfig":
        config_path = Path(path).expanduser()
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Robot config must be a mapping: {config_path}")
        validate_robot_config_ownership(raw, config_path)
        config = cls(
            path=config_path,
            device=_as_str(raw, "device", config_path, default="cuda"),
            policy_freq=_as_int(raw, "policy_freq", config_path, default=50),
            control_freq=_as_float(raw, "control_freq", config_path),
            lowstate_topic=_as_str(raw, "lowstate_topic", config_path),
            action_topic=_as_str(raw, "action_topic", config_path),
            velocity_tracking_model_folder=_as_str(
                raw,
                "velocity_tracking_model_folder",
                config_path,
            ),
            motion_tracking_model_folder=_as_str(
                raw,
                "motion_tracking_model_folder",
                config_path,
            ),
            motion_clip_dir=_as_str(raw, "motion_clip_dir", config_path),
            onnx_intra_op_threads=_as_int(
                raw,
                "onnx_intra_op_threads",
                config_path,
                default=2,
            ),
            complete_dof_order=_as_string_list(raw, "complete_dof_order", config_path),
            policy_dof_order=_as_string_list(raw, "policy_dof_order", config_path),
            dof2motor_idx_mapping=_as_int_map(
                raw,
                "dof2motor_idx_mapping",
                config_path,
            ),
            default_joint_angles=_as_float_map(raw, "default_joint_angles", config_path),
            joint_limits=_as_mapping(raw, "joint_limits", config_path),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.policy_freq <= 0:
            raise ValueError("Robot config field 'policy_freq' must be > 0")
        if self.control_freq <= 0.0:
            raise ValueError("Robot config field 'control_freq' must be > 0")
        if self.onnx_intra_op_threads <= 0:
            raise ValueError("Robot config field 'onnx_intra_op_threads' must be > 0")
        if len(self.complete_dof_order) != len(self.policy_dof_order):
            raise ValueError(
                "Robot config complete_dof_order and policy_dof_order must have the same length"
            )
        missing_angles = [
            name for name in self.complete_dof_order if name not in self.default_joint_angles
        ]
        if missing_angles:
            raise ValueError(
                "Robot config default_joint_angles missing joints: "
                + ", ".join(missing_angles)
            )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": self.device,
            "policy_freq": self.policy_freq,
            "control_freq": self.control_freq,
            "lowstate_topic": self.lowstate_topic,
            "action_topic": self.action_topic,
            "velocity_tracking_model_folder": self.velocity_tracking_model_folder,
            "motion_tracking_model_folder": self.motion_tracking_model_folder,
            "motion_clip_dir": self.motion_clip_dir,
            "onnx_intra_op_threads": self.onnx_intra_op_threads,
            "complete_dof_count": len(self.complete_dof_order),
            "policy_dof_count": len(self.policy_dof_order),
        }


def format_config_for_log(mapping: dict[str, Any]) -> str:
    return yaml.safe_dump(mapping, sort_keys=False).strip()
