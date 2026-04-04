# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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


import os
import csv
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from threading import Thread

import cv2
import hydra
import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime
import torch
from loguru import logger
from omegaconf import ListConfig, OmegaConf, open_dict
from tqdm import tqdm
import glob
import re

import ray
from holomotion.src.evaluation.metrics import run_evaluation

try:
    from horizon_tc_ui.hb_runtime import HBRuntime
except ImportError:
    HB_ONNXRuntime = None
    logger.warning("HB_ONNXRuntime not available!")

ONNX_IO_DUMP_DIRNAME = "onnx_io_npy"

try:
    import pynput.keyboard as pynput_kb

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    if "headless" in sys.argv and "false" in sys.argv:
        logger.warning("pynput not available, keyboard control disabled")

from holomotion.src.evaluation.obs import PolicyObsBuilder
from holomotion.src.utils.torch_utils import (
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
    quat_normalize_wxyz,
    matrix_from_quat,
    xyzw_to_wxyz,
    quat_mul,
    quat_from_euler_xyz,
)
from holomotion.src.motion_retargeting.utils.rotation_conversions import (
    standardize_quaternion,
)

DEFAULT_FEET_GEOM_NAMES = {
    "left": ["left_foot"],
    "right": ["right_foot"],
}
DEFAULT_FEET_BODY_NAMES = {
    "left": ["left_ankle_roll_link"],
    "right": ["right_ankle_roll_link"],
}


def _coerce_config_bool(value, default: bool = False) -> bool:
    """Interpret config booleans without treating non-empty strings as truthy."""
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


class OffscreenRenderer:
    """Minimal offscreen renderer for MuJoCo frames."""

    def __init__(
        self,
        model,
        height: int,
        width: int,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
    ):
        self.model = model
        self.height = height
        self.width = width

        self._overlay_callback = None

        self._gl_ctx = mujoco.GLContext(width, height)
        self._gl_ctx.make_current()

        self._scene = mujoco.MjvScene(model, maxgeom=1000)
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self._cam)
        self.set_align_view(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )

        self._con = mujoco.MjrContext(
            model,
            mujoco.mjtFontScale.mjFONTSCALE_100,
        )
        self._rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._viewport = mujoco.MjrRect(0, 0, width, height)

    def set_overlay_callback(self, callback) -> None:
        """Register a callback to draw custom geoms into the scene each frame."""
        self._overlay_callback = callback

    def render(self, data) -> np.ndarray:
        mujoco.mjv_updateScene(
            self.model,
            data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        if self._overlay_callback is not None:
            self._overlay_callback(self._scene)
        mujoco.mjr_render(self._viewport, self._scene, self._con)
        mujoco.mjr_readPixels(self._rgb, None, self._viewport, self._con)
        return np.flipud(self._rgb)

    def set_align_view(
        self,
        lookat: np.ndarray | None = None,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
    ):
        """Set camera to 'align' preset view (default azimuth=60, elevation=-20).

        Args:
            lookat: Optional lookat point [x, y, z]. If None, uses current lookat.
            distance: Optional camera distance from lookat point. If None, uses current distance.
        """
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if azimuth is None:
            self._cam.azimuth = 60.0  # Side view (looking along Y-axis)
        else:
            self._cam.azimuth = float(azimuth)
        if elevation is None:
            self._cam.elevation = -20.0  # Slight downward angle
        else:
            self._cam.elevation = float(elevation)
        if lookat is not None:
            self._cam.lookat = np.asarray(lookat, dtype=np.float32)
        if distance is not None:
            self._cam.distance = float(distance)

    def close(self):
        self._gl_ctx.free()


class VelocityKeyboardHandler:
    """Keyboard handler for interactive velocity commands using WASD and JL keys."""

    def __init__(
        self,
        vx_increment: float = 0.1,
        vy_increment: float = 0.1,
        vyaw_increment: float = 0.05,
        vx_limits: tuple = (-0.5, 1.0),
        vy_limits: tuple = (-0.3, 0.3),
        vyaw_limits: tuple = (-0.5, 0.5),
    ):
        self.vx_increment = vx_increment
        self.vy_increment = vy_increment
        self.vyaw_increment = vyaw_increment

        # Velocity limits from training config
        self.vx_min, self.vx_max = vx_limits
        self.vy_min, self.vy_max = vy_limits
        self.vyaw_min, self.vyaw_max = vyaw_limits

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        self._listener = None
        self._lock = threading.Lock()

    def start_listener(self):
        """Start keyboard listener thread (requires pynput)."""
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not available, keyboard control disabled")
            return

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char:
                    self._handle_key(key.char)
            except AttributeError:
                pass

        self._listener = pynput_kb.Listener(on_press=on_press)
        self._listener.start()
        logger.info(
            f"Keyboard listener started. Velocity limits: "
            f"vx=[{self.vx_min:.1f},{self.vx_max:.1f}], "
            f"vy=[{self.vy_min:.1f},{self.vy_max:.1f}], "
            f"vyaw=[{self.vyaw_min:.1f},{self.vyaw_max:.1f}]"
        )

    def stop_listener(self):
        """Stop keyboard listener thread."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_velocity_command(self) -> np.ndarray:
        """Get velocity command [vx, vy, vyaw].

        Returns:
            Velocity command [vx, vy, vyaw]
        """
        with self._lock:
            return np.array([self.vx, self.vy, self.vyaw], dtype=np.float32)

    def _handle_key(self, char: str):
        """Handle keyboard press events."""
        with self._lock:
            # W/S for vx (forward/backward)
            if char in ["W", "w"]:
                self.vx = np.clip(
                    self.vx + self.vx_increment, self.vx_min, self.vx_max
                )
                logger.info(
                    f"[W] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            elif char in ["S", "s"]:
                self.vx = np.clip(
                    self.vx - self.vx_increment, self.vx_min, self.vx_max
                )
                logger.info(
                    f"[S] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            # A/D for vy (left/right)
            elif char in ["A", "a"]:
                self.vy = np.clip(
                    self.vy + self.vy_increment, self.vy_min, self.vy_max
                )
                logger.info(
                    f"[A] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            elif char in ["D", "d"]:
                self.vy = np.clip(
                    self.vy - self.vy_increment, self.vy_min, self.vy_max
                )
                logger.info(
                    f"[D] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            # J/L for vyaw (turn left/right)
            elif char in ["J", "j"]:
                self.vyaw = np.clip(
                    self.vyaw + self.vyaw_increment,
                    self.vyaw_min,
                    self.vyaw_max,
                )
                logger.info(
                    f"[J] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            elif char in ["L", "l"]:
                self.vyaw = np.clip(
                    self.vyaw - self.vyaw_increment,
                    self.vyaw_min,
                    self.vyaw_max,
                )
                logger.info(
                    f"[L] vx={self.vx:.2f}, vy={self.vy:.2f}, vyaw={self.vyaw:.2f}"
                )
            # Space to reset all
            elif char == " ":
                self.vx = 0.0
                self.vy = 0.0
                self.vyaw = 0.0
                logger.info("[Space] Command reset to zero")
            # X to stop (emergency brake)
            elif char in ["X", "x"]:
                self.vx = 0.0
                self.vy = 0.0
                self.vyaw = 0.0
                logger.info("[X] Emergency stop - all velocities set to zero")


class MujocoEvaluator:
    """Class to handle MuJoCo simulation for policy evaluation."""

    def __init__(self, config):
        """Initialize the MuJoCo evaluator.

        Args:
            config: Configuration object with simulation parameters.
        """
        self.config = config

        # Initialize variables
        self.policy_session = None
        self.motion_encoding = None
        self.m = None  # MuJoCo model
        self.d = None  # MuJoCo data

        # Determine command mode from config
        self.command_mode = self._detect_command_mode()
        if "motion_npz_dir" not in config:
            logger.info(f"Command mode: {self.command_mode}")

        # Motion data
        self.ref_dof_pos = None
        self.ref_dof_vel = None
        self.filter_cutoff_hz = None
        self.n_motion_frames = 0
        self.motion_frame_idx = 0

        # Velocity command (for velocity tracking mode)
        self.velocity_command = np.zeros(3, dtype=np.float32)  # [vx, vy, vyaw]
        self.target_heading = 0.0  # Target heading for velocity tracking
        self.keyboard_handler = (
            None  # Will be initialized if velocity_tracking
        )

        # Extract configuration parameters
        self.simulation_dt = 1 / 200
        self.policy_dt = 1 / 50
        self.control_decimation = 4
        self.dof_names_ref_motion = list(config.robot.dof_names)
        self.num_actions = len(self.dof_names_ref_motion)

        self.action_scale_onnx = np.ones(self.num_actions, dtype=np.float32)

        self.kps_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.kds_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = self.default_angles_onnx.copy()
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.n_fut_frames = int(config.obs.n_fut_frames)
        self.actor_place_holder_ndim = self._find_actor_place_holder_ndim()

        self.use_kv_cache = False
        self.policy_kv_cache = None
        self.policy_kv_input_name = None
        self.policy_kv_output_name = None
        self.policy_kv_shape = None
        self.policy_model_context_len = 0
        algo_cfg = self.config.get("algo", None)
        if algo_cfg is None:
            raise ValueError("Missing config.algo for MuJoCo evaluation.")
        algo_config = algo_cfg.get("config", None)
        if algo_config is None:
            raise ValueError(
                "Missing config.algo.config for MuJoCo evaluation."
            )
        max_context_len_cfg = algo_config.get("num_steps_per_env", None)
        if max_context_len_cfg is None:
            raise ValueError(
                "Missing config.algo.config.num_steps_per_env for MuJoCo evaluation."
            )
        self.max_context_len = int(max_context_len_cfg)
        if self.max_context_len <= 0:
            raise ValueError(
                "config.algo.config.num_steps_per_env must be > 0, "
                f"got {self.max_context_len}"
            )
        self.policy_effective_context_len = 0

        self.counter = 0
        self.tau_hist = []
        # Latest Unitree lowstate message (populated when using Unitree bridge)
        # self._lowstate_msg = None
        # Desired target positions keyed by DOF name (updated after each policy step)
        self.target_dof_pos_by_name = {}

        # Video/recording related
        self._video_writer = None
        self._offscreen = None
        self._frame_interval = None
        self._last_frame_time = 0.0
        # Reference(global)->Simulation(global) rigid transform (computed at init)
        self._ref_to_sim_ready = False
        self._ref_to_sim_q_wxyz = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self._ref_to_sim_t = np.zeros(3, dtype=np.float32)
        # Optional offset between reference globals and dataset body names (e.g., world body at index 0)
        # Robot state recording buffers for offline NPZ dumping
        self._robot_dof_pos_seq: list[np.ndarray] = []
        self._robot_dof_vel_seq: list[np.ndarray] = []
        self._robot_dof_acc_seq: list[np.ndarray] = []
        self._robot_dof_torque_seq: list[np.ndarray] = []
        self._robot_low_level_dof_torque_seq: list[np.ndarray] = []
        self._robot_low_level_foot_contact_seq: list[np.ndarray] = []
        self._robot_low_level_foot_normal_force_seq: list[np.ndarray] = []
        self._robot_low_level_foot_tangent_speed_seq: list[np.ndarray] = []
        self._robot_actions_seq: list[np.ndarray] = []
        self._robot_action_rate_seq: list[np.float32] = []
        self._robot_global_translation_seq: list[np.ndarray] = []
        self._robot_global_rotation_quat_seq: list[np.ndarray] = []
        self._robot_global_velocity_seq: list[np.ndarray] = []
        self._robot_global_angular_velocity_seq: list[np.ndarray] = []
        self._robot_moe_expert_indices_seq: list[np.ndarray] = []
        self._robot_moe_expert_logits_seq: list[np.ndarray] = []
        self._prev_recorded_dof_vel_ref: np.ndarray | None = None
        self._prev_actions_onnx: np.ndarray | None = None
        (
            self.action_ema_filter_enabled,
            self.action_ema_filter_alpha,
        ) = self._get_action_ema_filter_cfg()
        self._filtered_actions_onnx: np.ndarray | None = None
        (
            self.policy_action_delay_step,
            self.action_delay_type,
        ) = self._get_action_delay_cfg()
        self._policy_action_delay_buffer: deque[np.ndarray] = deque(
            maxlen=max(1, self.policy_action_delay_step + 1)
        )
        self._current_policy_action_delay_step = 0
        self._reset_action_delay_randomization()
        # Camera config (viewer + offscreen)
        self._camera_tracking_enabled = bool(
            self.config.get("camera_tracking", True)
        )
        self._camera_height_offset = float(
            self.config.get("camera_height_offset", 0.3)
        )
        self._camera_distance = float(self.config.get("camera_distance", 4.0))
        self._camera_azimuth = float(self.config.get("camera_azimuth", 60.0))
        self._camera_elevation = float(
            self.config.get("camera_elevation", -20.0)
        )
        self._root_body_id = -1
        self._foot_contact_logging_enabled = False
        self._foot_geom_id_groups: list[list[int]] = [[], []]
        self._foot_geom_id_to_side: dict[int, int] = {}
        self._prev_low_level_foot_geom_centers: np.ndarray | None = None
        self.dump_onnx_io_npy = bool(
            self.config.get("dump_onnx_io_npy", False)
        )
        self.policy_moe_layer_output_names: list[tuple[int, str, str]] = []
        self._reset_onnx_io_dump_buffers()

    def _reset_onnx_io_dump_buffers(self):
        self._onnx_io_input_names: list[str] = []
        self._onnx_io_output_names: list[str] = []
        self._onnx_io_inputs: dict[str, list[np.ndarray]] = {}
        self._onnx_io_outputs: dict[str, list[np.ndarray]] = {}

    def _get_action_ema_filter_cfg(self) -> tuple[bool, float]:
        actuator_cfg = self.config.get("robot", {}).get("actuators", {})
        actuator_type = actuator_cfg.get("actuator_type", "unitree")
        if actuator_type != "unitree_erfi":
            return False, 1.0

        enabled = _coerce_config_bool(
            actuator_cfg.get("ema_filter_enabled", False), default=False
        )
        alpha = float(actuator_cfg.get("ema_filter_alpha", 1.0))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "robot.actuators.ema_filter_alpha must be within [0, 1], "
                f"got {alpha}."
            )
        return enabled, alpha

    def _reset_action_ema_filter(self) -> None:
        self._filtered_actions_onnx = None

    def _apply_action_ema_filter(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        if not self.action_ema_filter_enabled:
            return raw_actions.copy()

        if self._filtered_actions_onnx is None:
            self._filtered_actions_onnx = raw_actions.copy()
            return self._filtered_actions_onnx.copy()

        # self.action_ema_filter_alpha = 0.7
        filtered_actions = (
            self.action_ema_filter_alpha * raw_actions
            + (1.0 - self.action_ema_filter_alpha)
            * self._filtered_actions_onnx
        ).astype(np.float32, copy=False)
        self._filtered_actions_onnx = filtered_actions.copy()
        return self._filtered_actions_onnx.copy()

    def _get_action_delay_cfg(self) -> tuple[int, str]:
        max_delay_step = int(self.config.get("policy_action_delay_step", 0))
        if max_delay_step < 0:
            raise ValueError(
                "policy_action_delay_step must be non-negative, "
                f"got {max_delay_step}."
            )

        delay_type = (
            str(self.config.get("action_delay_type", "episode"))
            .strip()
            .lower()
        )
        if delay_type not in {"step", "episode"}:
            raise ValueError(
                "action_delay_type must be one of {'step', 'episode'}, "
                f"got {delay_type!r}."
            )
        return max_delay_step, delay_type

    def _sample_policy_action_delay_step(self) -> int:
        if self.policy_action_delay_step <= 0:
            return 0
        return int(np.random.randint(0, self.policy_action_delay_step + 1))

    def _reset_action_delay_randomization(self) -> None:
        self._policy_action_delay_buffer = deque(
            maxlen=max(1, self.policy_action_delay_step + 1)
        )
        if self.policy_action_delay_step <= 0:
            self._current_policy_action_delay_step = 0
            return
        if self.action_delay_type == "episode":
            self._current_policy_action_delay_step = (
                self._sample_policy_action_delay_step()
            )
        else:
            self._current_policy_action_delay_step = 0

    def _apply_action_delay(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        if self.policy_action_delay_step <= 0:
            return raw_actions.copy()

        expected_buffer_len = max(1, self.policy_action_delay_step + 1)
        if (
            not hasattr(self, "_policy_action_delay_buffer")
            or self._policy_action_delay_buffer.maxlen != expected_buffer_len
        ):
            self._reset_action_delay_randomization()

        if self.action_delay_type == "step":
            self._current_policy_action_delay_step = (
                self._sample_policy_action_delay_step()
            )

        self._policy_action_delay_buffer.append(raw_actions.copy())
        if self._current_policy_action_delay_step >= len(
            self._policy_action_delay_buffer
        ):
            return self._policy_action_delay_buffer[-1].copy()

        return self._policy_action_delay_buffer[
            -1 - self._current_policy_action_delay_step
        ].copy()

    @staticmethod
    def _normalize_foot_geom_name_groups(raw_spec) -> list[list[str]]:
        if raw_spec is None:
            return [[], []]

        if OmegaConf.is_config(raw_spec):
            raw_spec = OmegaConf.to_container(raw_spec, resolve=True)

        def coerce_names(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [str(name) for name in value if str(name)]
            return []

        if isinstance(raw_spec, dict):
            return [
                coerce_names(raw_spec.get("left", raw_spec.get("left_foot"))),
                coerce_names(
                    raw_spec.get("right", raw_spec.get("right_foot"))
                ),
            ]

        if isinstance(raw_spec, (list, tuple)) and len(raw_spec) == 2:
            return [coerce_names(raw_spec[0]), coerce_names(raw_spec[1])]

        logger.warning(
            "Unsupported robot.feet_geom_names format. Ignoring configured "
            "foot geom names."
        )
        return [[], []]

    @staticmethod
    def _normalize_foot_body_name_groups(raw_spec) -> list[list[str]]:
        if raw_spec is None:
            return [
                list(DEFAULT_FEET_BODY_NAMES["left"]),
                list(DEFAULT_FEET_BODY_NAMES["right"]),
            ]

        if OmegaConf.is_config(raw_spec):
            raw_spec = OmegaConf.to_container(raw_spec, resolve=True)

        def coerce_names(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [str(name) for name in value if str(name)]
            return []

        if isinstance(raw_spec, dict):
            return [
                coerce_names(raw_spec.get("left", raw_spec.get("left_foot"))),
                coerce_names(
                    raw_spec.get("right", raw_spec.get("right_foot"))
                ),
            ]

        if isinstance(raw_spec, (list, tuple)) and len(raw_spec) == 2:
            return [coerce_names(raw_spec[0]), coerce_names(raw_spec[1])]

        logger.warning(
            "Unsupported robot.feet_body_names format. Falling back to "
            f"default foot bodies: {DEFAULT_FEET_BODY_NAMES}"
        )
        return [
            list(DEFAULT_FEET_BODY_NAMES["left"]),
            list(DEFAULT_FEET_BODY_NAMES["right"]),
        ]

    def _resolve_foot_geom_ids_from_geom_names(
        self, foot_geom_name_groups: list[list[str]]
    ) -> list[list[int]]:
        foot_geom_id_groups: list[list[int]] = [[], []]
        for side_idx, geom_names in enumerate(foot_geom_name_groups):
            for geom_name in geom_names:
                geom_id = mujoco.mj_name2id(
                    self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                if geom_id == -1:
                    logger.warning(
                        f"Foot geom '{geom_name}' was not found in the MuJoCo model."
                    )
                    continue
                foot_geom_id_groups[side_idx].append(int(geom_id))
        return foot_geom_id_groups

    def _resolve_foot_geom_ids_from_body_names(
        self, foot_body_name_groups: list[list[str]]
    ) -> list[list[int]]:
        foot_geom_id_groups: list[list[int]] = [[], []]
        geom_bodyid = np.asarray(self.m.geom_bodyid, dtype=np.int32)
        geom_contype = np.asarray(self.m.geom_contype, dtype=np.int32)
        geom_conaffinity = np.asarray(self.m.geom_conaffinity, dtype=np.int32)
        collidable_mask = (geom_contype != 0) | (geom_conaffinity != 0)

        for side_idx, body_names in enumerate(foot_body_name_groups):
            resolved_geom_ids: list[int] = []
            for body_name in body_names:
                body_id = mujoco.mj_name2id(
                    self.m, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                if body_id == -1:
                    logger.warning(
                        f"Foot body '{body_name}' was not found in the MuJoCo model."
                    )
                    continue
                body_geom_ids = np.flatnonzero(geom_bodyid == int(body_id))
                if body_geom_ids.size == 0:
                    logger.warning(
                        f"Foot body '{body_name}' has no attached geoms."
                    )
                    continue
                contact_geom_ids = body_geom_ids[
                    collidable_mask[body_geom_ids]
                ]
                if contact_geom_ids.size == 0:
                    contact_geom_ids = body_geom_ids
                resolved_geom_ids.extend(contact_geom_ids.astype(int).tolist())

            # Preserve order while removing duplicates.
            deduped = list(dict.fromkeys(resolved_geom_ids))
            foot_geom_id_groups[side_idx] = deduped
        return foot_geom_id_groups

    def _init_low_level_foot_contact_logging(self) -> None:
        self._foot_geom_id_groups = [[], []]
        self._foot_geom_id_to_side = {}
        self._foot_contact_logging_enabled = False
        self._prev_low_level_foot_geom_centers = None

        foot_geom_name_groups = self._normalize_foot_geom_name_groups(
            getattr(self.config.robot, "feet_geom_names", None)
        )
        foot_body_name_groups = self._normalize_foot_body_name_groups(
            getattr(self.config.robot, "feet_body_names", None)
        )
        geom_name_groups = self._resolve_foot_geom_ids_from_geom_names(
            foot_geom_name_groups
        )
        body_name_groups = self._resolve_foot_geom_ids_from_body_names(
            foot_body_name_groups
        )

        for side_idx in range(2):
            resolved_ids = (
                geom_name_groups[side_idx]
                if len(geom_name_groups[side_idx]) > 0
                else body_name_groups[side_idx]
            )
            self._foot_geom_id_groups[side_idx] = list(resolved_ids)
            for geom_id in resolved_ids:
                self._foot_geom_id_to_side[int(geom_id)] = side_idx

        if any(len(group) == 0 for group in self._foot_geom_id_groups):
            logger.warning(
                "Low-level foot contact logging is unavailable because one or "
                "both foot geom groups could not be resolved. Contact metrics "
                "will be written as NaN."
            )
            return

        self._foot_contact_logging_enabled = True

    def _record_low_level_foot_contact_sample(self) -> None:
        foot_contact = np.full((2,), np.nan, dtype=np.float32)
        foot_normal_force = np.full((2,), np.nan, dtype=np.float32)
        foot_tangent_speed = np.full((2,), np.nan, dtype=np.float32)

        if not self._foot_contact_logging_enabled:
            self._robot_low_level_foot_contact_seq.append(foot_contact)
            self._robot_low_level_foot_normal_force_seq.append(
                foot_normal_force
            )
            self._robot_low_level_foot_tangent_speed_seq.append(
                foot_tangent_speed
            )
            return

        current_centers = np.zeros((2, 3), dtype=np.float32)
        for side_idx, geom_ids in enumerate(self._foot_geom_id_groups):
            current_centers[side_idx] = np.mean(
                self.d.geom_xpos[np.asarray(geom_ids, dtype=np.int32)],
                axis=0,
            ).astype(np.float32)

        if self._prev_low_level_foot_geom_centers is None:
            tangential_speed = np.zeros((2,), dtype=np.float32)
        else:
            foot_velocity = (
                current_centers - self._prev_low_level_foot_geom_centers
            ) / np.float32(self.simulation_dt)
            tangential_speed = np.linalg.norm(
                foot_velocity[:, :2], axis=1
            ).astype(np.float32)
        self._prev_low_level_foot_geom_centers = current_centers.copy()

        foot_contact.fill(0.0)
        foot_normal_force.fill(0.0)
        foot_tangent_speed = tangential_speed

        contact_force = np.zeros(6, dtype=np.float64)
        for contact_idx in range(int(self.d.ncon)):
            contact = self.d.contact[contact_idx]
            contact_sides = set()
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._foot_geom_id_to_side:
                contact_sides.add(self._foot_geom_id_to_side[geom1])
            if geom2 in self._foot_geom_id_to_side:
                contact_sides.add(self._foot_geom_id_to_side[geom2])
            if len(contact_sides) != 1:
                continue

            side_idx = next(iter(contact_sides))
            foot_contact[side_idx] = 1.0
            mujoco.mj_contactForce(self.m, self.d, contact_idx, contact_force)
            foot_normal_force[side_idx] += np.float32(abs(contact_force[0]))

        self._robot_low_level_foot_contact_seq.append(foot_contact)
        self._robot_low_level_foot_normal_force_seq.append(foot_normal_force)
        self._robot_low_level_foot_tangent_speed_seq.append(foot_tangent_speed)

    @staticmethod
    def _flatten_single_step_output(values, *, dtype=None) -> np.ndarray:
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim == 0:
            raise ValueError(
                "Expected at least 1D output for single-step ONNX routing dump."
            )
        return arr.reshape(-1, arr.shape[-1])[0]

    def _discover_policy_moe_outputs(self) -> None:
        self.policy_moe_layer_output_names: list[tuple[int, str, str]] = []
        routing_outputs: dict[int, dict[str, str]] = {}
        pattern = re.compile(r"^moe_layer_(\d+)_expert_(indices|logits)$")
        for node in self.policy_session.get_outputs():
            match = pattern.fullmatch(node.name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            kind = str(match.group(2))
            routing_outputs.setdefault(layer_idx, {})[kind] = node.name

        for layer_idx in sorted(routing_outputs):
            layer_outputs = routing_outputs[layer_idx]
            if "indices" not in layer_outputs or "logits" not in layer_outputs:
                logger.warning(
                    "Skipping incomplete MoE routing outputs for layer "
                    f"{layer_idx}: {sorted(layer_outputs)}"
                )
                continue
            self.policy_moe_layer_output_names.append(
                (
                    layer_idx,
                    layer_outputs["indices"],
                    layer_outputs["logits"],
                )
            )
        if self.policy_moe_layer_output_names:
            logger.info(
                "Detected MoE routing outputs for layers: "
                f"{[layer_idx for layer_idx, _, _ in self.policy_moe_layer_output_names]}"
            )

    def _get_stacked_moe_routing_tensors(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        indices_seq = getattr(self, "_robot_moe_expert_indices_seq", [])
        logits_seq = getattr(self, "_robot_moe_expert_logits_seq", [])
        if len(indices_seq) == 0 or len(logits_seq) == 0:
            return None, None
        return (
            np.stack(indices_seq, axis=0).astype(np.int64),
            np.stack(logits_seq, axis=0).astype(np.float32),
        )

    def _get_stacked_low_level_foot_contact_tensors(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        contact_seq = getattr(self, "_robot_low_level_foot_contact_seq", [])
        normal_force_seq = getattr(
            self, "_robot_low_level_foot_normal_force_seq", []
        )
        tangent_speed_seq = getattr(
            self, "_robot_low_level_foot_tangent_speed_seq", []
        )
        if contact_seq and normal_force_seq and tangent_speed_seq:
            return (
                np.stack(contact_seq, axis=0).astype(np.float32),
                np.stack(normal_force_seq, axis=0).astype(np.float32),
                np.stack(tangent_speed_seq, axis=0).astype(np.float32),
            )

        num_low_level_samples = len(
            getattr(self, "_robot_low_level_dof_torque_seq", [])
        )
        if num_low_level_samples <= 0:
            return None, None, None

        nan_array = np.full((num_low_level_samples, 2), np.nan, np.float32)
        return nan_array.copy(), nan_array.copy(), nan_array.copy()

    def _record_onnx_io_frame(self, input_feed, output_names, onnx_output):
        if not self._onnx_io_input_names:
            self._onnx_io_input_names = list(input_feed.keys())
            self._onnx_io_inputs = {
                name: [] for name in self._onnx_io_input_names
            }
        if not self._onnx_io_output_names:
            self._onnx_io_output_names = list(output_names)
            self._onnx_io_outputs = {
                name: [] for name in self._onnx_io_output_names
            }

        for name in self._onnx_io_input_names:
            if name not in input_feed:
                raise KeyError(f"Missing ONNX input tensor: {name}")
            self._onnx_io_inputs[name].append(
                np.array(input_feed[name], copy=True)
            )
        for name, value in zip(self._onnx_io_output_names, onnx_output):
            self._onnx_io_outputs[name].append(np.array(value, copy=True))

    @staticmethod
    def _stack_onnx_io_frames(
        frame_dict: dict[str, list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        stacked: dict[str, np.ndarray] = {}
        for name, frames in frame_dict.items():
            if frames:
                stacked[name] = np.stack(frames, axis=0)
            else:
                stacked[name] = np.empty((0,), dtype=np.float32)
        return stacked

    def save_onnx_io_dump(self, output_path, meta_info):
        payload = {
            "input_names": list(self._onnx_io_input_names),
            "output_names": list(self._onnx_io_output_names),
            "inputs": self._stack_onnx_io_frames(self._onnx_io_inputs),
            "outputs": self._stack_onnx_io_frames(self._onnx_io_outputs),
            "source_npz": meta_info.get(
                "source_npz", meta_info.get("source_file", "")
            ),
            "onnx_model": meta_info.get(
                "onnx_model", meta_info.get("model", "")
            ),
        }
        np.save(output_path, payload, allow_pickle=True)

    def _find_actor_place_holder_ndim(self):
        n_dim = 0
        for obs_dict in self._get_policy_atomic_obs_list():
            name = str(list(obs_dict.keys())[0])
            if name == "place_holder":
                params = obs_dict["place_holder"].get("params", {})
                n_dim = int(params.get("n_dim", 0))
            if name == "actor_place_holder":
                params = obs_dict["actor_place_holder"].get("params", {})
                n_dim = int(params.get("n_dim", 0))
        return n_dim

    def _get_actor_obs_term_params(self, term_name: str) -> dict:
        for obs_dict in self._get_policy_atomic_obs_list():
            configured_name = str(list(obs_dict.keys())[0])
            if configured_name != term_name:
                continue
            term_cfg = obs_dict[configured_name]
            if not isinstance(term_cfg, dict):
                return {}
            params = term_cfg.get("params", {})
            return dict(params) if isinstance(params, dict) else {}
        return {}

    def _get_ref_keybody_indices(self, term_name: str) -> np.ndarray:
        params = self._get_actor_obs_term_params(term_name)
        keybody_names = params.get("keybody_names", None)
        body_names = [str(name) for name in self.config.robot.body_names]
        if keybody_names is None:
            return np.arange(len(body_names), dtype=np.int64)

        keybody_names = [str(name) for name in keybody_names]
        body_name_to_idx = {
            body_name: idx for idx, body_name in enumerate(body_names)
        }
        missing_names = [
            name for name in keybody_names if name not in body_name_to_idx
        ]
        if len(missing_names) > 0:
            raise ValueError(
                f"Unknown keybody_names in '{term_name}': {missing_names}. "
                f"Available body names: {body_names}"
            )

        return np.asarray(
            [body_name_to_idx[name] for name in keybody_names],
            dtype=np.int64,
        )

    @staticmethod
    def _to_plain_obs_cfg(cfg):
        if OmegaConf.is_config(cfg):
            plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            plain_cfg = dict(cfg)
        if not isinstance(plain_cfg, dict):
            raise ValueError(
                f"Observation term config must be a mapping, got {type(plain_cfg)}"
            )
        return plain_cfg

    def _get_actor_obs_schema_terms(self) -> list[str]:
        modules_cfg = self.config.get("modules", None)
        if modules_cfg is None:
            return []
        actor_cfg = modules_cfg.get("actor", None)
        if actor_cfg is None:
            return []
        obs_schema = actor_cfg.get("obs_schema", None)
        if obs_schema is None:
            return []

        ordered_terms: list[str] = []
        for _, seq_cfg in obs_schema.items():
            seq_terms = seq_cfg.get("terms", [])
            ordered_terms.extend(str(term) for term in seq_terms)
        return ordered_terms

    def _get_actor_atomic_obs_entries(self) -> list[tuple[str, str, dict]]:
        obs_cfg = self.config.get("obs", None)
        if obs_cfg is None:
            raise ValueError("Missing config.obs for MuJoCo sim2sim")
        obs_groups = obs_cfg.get("obs_groups", None)
        if obs_groups is None:
            raise ValueError(
                "Missing config.obs.obs_groups for MuJoCo sim2sim"
            )

        if obs_groups.get("policy", None) is not None:
            entries: list[tuple[str, str, dict]] = []
            for term_dict in obs_groups.policy.atomic_obs_list:
                term_name = str(list(term_dict.keys())[0])
                entries.append(
                    (
                        "policy",
                        term_name,
                        self._to_plain_obs_cfg(term_dict[term_name]),
                    )
                )
            return entries

        if obs_groups.get("unified", None) is not None:
            entries = []
            for term_dict in obs_groups.unified.atomic_obs_list:
                term_name = str(list(term_dict.keys())[0])
                if term_name.startswith("critic_"):
                    continue
                entries.append(
                    (
                        "unified",
                        term_name,
                        self._to_plain_obs_cfg(term_dict[term_name]),
                    )
                )
            if not entries:
                raise ValueError(
                    "obs_groups.unified found but contains no non-critic terms."
                )
            return entries

        raise ValueError(
            "Unsupported obs config for MuJoCo sim2sim: expected obs_groups.policy or obs_groups.unified."
        )

    def _get_policy_atomic_obs_list(self):
        """Resolve the atomic obs list used to build the ONNX policy input.

        Supports both legacy configs (obs_groups.policy) and PULSE-stage2 configs
        that use a unified group (obs_groups.unified) with actor_/critic_ prefixes.
        """
        actor_atomic_entries = self._get_actor_atomic_obs_entries()
        schema_terms = self._get_actor_obs_schema_terms()

        if len(schema_terms) == 0:
            logger.warning(
                "modules.actor.obs_schema is unavailable; using obs_groups actor term order for MuJoCo policy input."
            )
            return [
                {term_name: cfg} for _, term_name, cfg in actor_atomic_entries
            ]

        by_full_key: dict[str, tuple[str, dict]] = {}
        by_leaf_key: dict[str, tuple[str, dict]] = {}
        ambiguous_leaf_keys: set[str] = set()
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
                    "Actor obs_schema term "
                    f"'{schema_term}' is ambiguous by leaf key '{leaf_key}'. "
                    "Use explicit group/term hierarchy in obs_schema terms."
                )
            if leaf_key not in by_leaf_key:
                raise ValueError(
                    "Actor obs_schema term "
                    f"'{schema_term}' is not present in obs_groups actor atomic obs list."
                )
            term_name, term_cfg = by_leaf_key[leaf_key]
            ordered_atomic_list.append({term_name: term_cfg})
        return ordered_atomic_list

    # ----------------- Kinematics / velocities -----------------

    # ----------------- Kinematics / velocities -----------------
    def _body_origin_world_velocity(
        self, body_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world-frame spatial velocity (v, w) of a body's frame origin.

        Returns:
            tuple: (lin_vel_w[3], ang_vel_w[3]) in world coordinates.
        """
        # World-frame Jacobians for body origin
        jacp = np.zeros((3, self.m.nv), dtype=np.float64)
        jacr = np.zeros((3, self.m.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, int(body_id))
        # qvel is float64 in MuJoCo; keep computation in float64 then cast
        lin_vel_w = jacp @ self.d.qvel
        ang_vel_w = jacr @ self.d.qvel
        return lin_vel_w.astype(np.float32), ang_vel_w.astype(np.float32)

    # ----------------- Body name/id resolution -----------------
    def _get_anchor_body_name(self) -> str:
        if not hasattr(self, "anchor_body_name"):
            self.anchor_body_name = str(
                getattr(self.config.robot, "anchor_body", "pelvis")
            )
        logger.info(f"Anchor body name: {self.anchor_body_name}")
        return self.anchor_body_name

    def _get_torso_body_name(self) -> str:
        if not hasattr(self, "torso_body_name"):
            self.torso_body_name = str(
                getattr(self.config.robot, "torso_name", "torso_link")
            )
        return self.torso_body_name

    @property
    def ref_motion_frame_idx(self):
        return self.motion_frame_idx

    @property
    def anchor_body_idx(self) -> int:
        return self.config.robot.body_names.index(
            self.config.robot.anchor_body
        )

    @property
    def root_body_idx(self) -> int:
        return 0

    @property
    def torso_body_idx(self) -> int:
        return self.config.robot.body_names.index(self.config.robot.torso_name)

    @property
    def robot_global_bodylink_pos(self):
        """World-frame positions of all robot bodies at their MuJoCo body frame origins.

        MuJoCo stores body state for a special world body at index 0, which does not
        correspond to any physical link and is always static. We slice it out and
        return `xpos[1:]` so that row 0 corresponds to the root body (e.g. pelvis)
        and the body dimension matches the HoloMotion NPZ `*_global_translation`
        arrays.

        Returns:
            np.ndarray: Array of shape [n_bodies, 3] in MuJoCo body order with the
            world body excluded.
        """
        return self.d.xpos[1:]

    @property
    def robot_global_bodylink_rot(self):
        """World-frame orientations of all robot bodies as WXYZ quaternions.

        As with positions, the MuJoCo world body at index 0 is excluded so that the
        returned array is aligned with the body dimension used in HoloMotion NPZ
        `*_global_rotation_quat` arrays (root at index 0, no world entry).

        Returns:
            np.ndarray: Array of shape [n_bodies, 4] in MuJoCo body order with the
            world body excluded.
        """
        xquat = self.d.xquat[1:]
        xquat_t = torch.as_tensor(xquat, dtype=torch.float32, device="cpu")
        xquat_t = standardize_quaternion(xquat_t)

        return xquat_t.detach().cpu().numpy()

    @property
    def robot_global_bodylink_lin_vel(self):
        """World-frame linear velocities of all robot body frame origins.

        Uses `mujoco.mj_objectVelocity` with `mjOBJ_BODY` and `flg_centered=0` to
        query the 6D spatial velocity at each body's frame origin, then slices the
        translational component. The world body (ID 0) is excluded so that the body
        dimension matches the NPZ `*_global_velocity` arrays.

        Returns:
            np.ndarray: Array of shape [n_bodies, 3] giving linear velocities in the
            MuJoCo world frame, ordered by body ID starting from the root body.
        """
        nbody = int(self.m.nbody)
        vel_6d = np.zeros((nbody, 6), dtype=np.float64)
        for bid in range(1, nbody):
            mujoco.mj_objectVelocity(
                self.m,
                self.d,
                mujoco.mjtObj.mjOBJ_BODY,
                bid,
                vel_6d[bid],
                0,
            )
        return vel_6d[1:, 3:6]

    @property
    def robot_global_bodylink_ang_vel(self):
        """World-frame angular velocities of all robot body frame origins.

        Uses the same `mujoco.mj_objectVelocity` call as
        `robot_global_bodylink_lin_vel` and slices the rotational component. The
        world body (ID 0) is dropped so that the body dimension is identical to the
        NPZ `*_global_angular_velocity` arrays and the translation/rotation/velocity
        tensors all share the same body ordering.

        Returns:
            np.ndarray: Array of shape [n_bodies, 3] giving angular velocities in
            the MuJoCo world frame, ordered by body ID starting from the root body.
        """
        nbody = int(self.m.nbody)
        vel_6d = np.zeros((nbody, 6), dtype=np.float64)
        for bid in range(1, nbody):
            mujoco.mj_objectVelocity(
                self.m,
                self.d,
                mujoco.mjtObj.mjOBJ_BODY,
                bid,
                vel_6d[bid],
                0,
            )
        return vel_6d[1:, 0:3]

    @property
    def robot_dof_pos(self):
        if hasattr(self, "actuator_qpos_indices"):
            return self.d.qpos[self.actuator_qpos_indices]
        return self.d.qpos[7:]

    @property
    def robot_dof_vel(self):
        if hasattr(self, "actuator_qvel_indices"):
            return self.d.qvel[self.actuator_qvel_indices]
        return self.d.qvel[6:]

    # ----------------- Reference->Simulation alignment -----------------

    def _ensure_ref_to_sim_transform_rigid(self):
        """Compute rigid transform (yaw + translation) from reference globals to sim globals.

        The transform is defined such that the reference **anchor body** pose at frame 0 is mapped
        onto the robot's current global anchor pose in XY translation and yaw:

        - `yaw(q_ref_to_sim * q_ref_anchor_0) = yaw(q_robot_anchor_0)`
        - `t_ref_to_sim + R(q_ref_to_sim) @ t_ref_anchor_0 = t_robot_anchor_0`

        This uses the robot's initial global pose so that arbitrary initialization offsets in
        XY position and yaw between the robot and the reference motion are absorbed into the
        reference->simulation mapping, and all subsequent reference globals are expressed in the
        same world frame as the robot.
        """
        if self._ref_to_sim_ready:
            return

        # If we don't have reference globals, fall back to identity transform.
        if getattr(self, "ref_global_translation", None) is None:
            self._ref_to_sim_q_wxyz = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            self._ref_to_sim_t = np.zeros(3, dtype=np.float32)
            self._ref_to_sim_ready = True
            logger.info(
                "No reference global translations available; using identity Ref->Sim transform."
            )
            return

        # If rotations are missing, keep the previous translation-only semantics.
        if getattr(self, "ref_global_rotation_quat_xyzw", None) is None:
            t_robot = torch.as_tensor(
                self.robot_global_bodylink_pos[self.anchor_body_idx],
                dtype=torch.float32,
                device="cpu",
            )
            t_ref = torch.as_tensor(
                self.ref_global_translation[0, self.anchor_body_idx].astype(
                    np.float32
                ),
                dtype=torch.float32,
                device="cpu",
            )
            t_ref_to_sim = t_robot - t_ref
            self._ref_to_sim_q_wxyz = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            self._ref_to_sim_t = t_ref_to_sim.detach().cpu().numpy()
            self._ref_to_sim_ready = True
            logger.info(
                "Reference rotations missing; initialized Ref->Sim as translation-only "
                f"transform. t={self._ref_to_sim_t}"
            )
            return

        # Anchor body index shared between robot globals and reference globals
        anchor_idx = self.anchor_body_idx

        # Robot anchor pose in simulation world frame (after initial state has been set)
        t_robot = torch.as_tensor(
            self.robot_global_bodylink_pos[anchor_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_robot_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[anchor_idx],
            dtype=torch.float32,
            device="cpu",
        )

        # Reference anchor pose at frame 0 in NPZ global frame
        t_ref0 = torch.as_tensor(
            self.ref_global_translation[0, anchor_idx].astype(np.float32),
            dtype=torch.float32,
            device="cpu",
        )
        q_ref0_xyzw = torch.as_tensor(
            self.ref_global_rotation_quat_xyzw[0, anchor_idx].astype(
                np.float32
            ),
            dtype=torch.float32,
            device="cpu",
        )
        q_ref0_wxyz = xyzw_to_wxyz(q_ref0_xyzw)

        # Yaw-only rotation mapping: align reference yaw to robot yaw (keep roll/pitch from reference).
        R_robot = matrix_from_quat(q_robot_wxyz)
        R_ref0 = matrix_from_quat(q_ref0_wxyz)
        yaw_robot = torch.atan2(R_robot[1, 0], R_robot[0, 0])
        yaw_ref0 = torch.atan2(R_ref0[1, 0], R_ref0[0, 0])
        yaw_delta = yaw_robot - yaw_ref0

        yaw_quat_xyzw = quat_from_euler_xyz(
            torch.tensor(0.0, dtype=torch.float32, device="cpu"),
            torch.tensor(0.0, dtype=torch.float32, device="cpu"),
            yaw_delta,
        )
        q_ref_to_sim = xyzw_to_wxyz(yaw_quat_xyzw)
        q_ref_to_sim = quat_normalize_wxyz(q_ref_to_sim)

        # Translation mapping: t_ref_to_sim + R(q_ref_to_sim) @ t_ref0 = t_robot
        t_ref0_in_sim = quat_apply(q_ref_to_sim, t_ref0)
        t_ref_to_sim = t_robot - t_ref0_in_sim

        self._ref_to_sim_q_wxyz = (
            q_ref_to_sim.detach().cpu().numpy().astype(np.float32)
        )
        self._ref_to_sim_t = (
            t_ref_to_sim.detach().cpu().numpy().astype(np.float32)
        )

        self._ref_to_sim_ready = True
        logger.info(
            "Initialized Ref->Sim rigid transform. "
            f"q={self._ref_to_sim_q_wxyz}, t={self._ref_to_sim_t}"
        )

    def _detect_command_mode(self) -> str:
        m_dir = self.config.get("motion_npz_dir") or self.config.get(
            "eval", {}
        ).get("motion_npz_dir")
        m_path = self.config.get("motion_npz_path") or self.config.get(
            "eval", {}
        ).get("motion_npz_path")
        if m_path is not None and not os.path.exists(m_path):
            raise FileNotFoundError(f"Motion file not found: {m_path}")

        if (m_dir and str(m_dir) != "") or (m_path and str(m_path) != ""):
            return "motion_tracking"
        return "velocity_tracking"

    def _init_obs_buffers(self):
        atomic_list = self._get_policy_atomic_obs_list()
        obs_policy_cfg = {"atomic_obs_list": atomic_list}
        self.obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.dof_names_onnx,
            default_angles_onnx=self.default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=obs_policy_cfg,
        )

    def load_policy(self):
        """Load the policy model using ONNX Runtime."""
        onnx_model_path = Path(self.config.ckpt_onnx_path)

        logger.info(f"Loading ONNX policy from {onnx_model_path}")

        providers = ["CPUExecutionProvider"]
        use_gpu = _coerce_config_bool(
            self.config.get("use_gpu", False), default=False
        )
        gpu_id = int(self.config.get("gpu_id", 0))

        available_providers = onnxruntime.get_available_providers()
        if use_gpu:
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {"device_id": gpu_id}
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    cuda_options["user_compute_stream"] = str(
                        torch.cuda.current_stream().cuda_stream
                    )
                providers = [
                    ("CUDAExecutionProvider", cuda_options),
                    "CPUExecutionProvider",
                ]
                logger.info(
                    f"Using CUDAExecutionProvider with gpu_id={gpu_id}"
                )
            else:
                logger.warning(
                    "use_gpu=true but CUDAExecutionProvider is unavailable; "
                    "falling back to CPUExecutionProvider."
                )

        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3

        self.policy_session = onnxruntime.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_options,
            providers=providers,
        )
        logger.info(
            f"ONNX Runtime session created successfully using: {self.policy_session.get_providers()}"
        )
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_output_name = self.policy_session.get_outputs()[0].name
        logger.info(
            f"Policy  ONNX Input: {self.policy_input_name}, Output: {self.policy_output_name}"
        )

        logger.info("Initializing KV-Cache for Policy...")

        self.policy_input_name = "obs"
        self.policy_kv_input_name = None
        self.policy_step_input_name = None
        self.policy_kv_shape = None

        for node in self.policy_session.get_inputs():
            name = node.name
            shape = node.shape
            logger.info(f"Model Input: Name={name}, Shape={shape}")

            if "obs" in name:
                self.policy_input_name = name
            elif "past_key_values" in name:
                self.policy_kv_input_name = name
                self.policy_kv_shape = shape
            elif "step_idx" in name or "step" in name or "pos" in name:
                self.policy_step_input_name = name

        self.policy_output_name = self.policy_session.get_outputs()[0].name
        self.policy_kv_output_name = None
        for node in self.policy_session.get_outputs():
            if "present_key_values" in node.name:
                self.policy_kv_output_name = node.name
        self._discover_policy_moe_outputs()

        if self.policy_kv_input_name and self.policy_kv_shape:
            shape = [
                d if isinstance(d, int) else 1 for d in self.policy_kv_shape
            ]
            self.policy_kv_cache = np.zeros(shape, dtype=np.float32)
            self.policy_model_context_len = (
                int(shape[3]) if len(shape) > 3 else 0
            )
            if self.max_context_len > 0 and self.policy_model_context_len > 0:
                self.policy_effective_context_len = min(
                    self.max_context_len, self.policy_model_context_len
                )
                logger.info(
                    "Using context window from "
                    f"algo.config.num_steps_per_env={self.max_context_len} "
                    f"(model cache len={self.policy_model_context_len}, "
                    f"effective={self.policy_effective_context_len})"
                )
            else:
                self.policy_effective_context_len = (
                    self.policy_model_context_len
                )
            self.use_kv_cache = True
            logger.info(f"KV-Cache ENABLED. Shape: {shape}")
        else:
            self.use_kv_cache = False
            self.policy_kv_cache = None
            self.policy_model_context_len = 0
            self.policy_effective_context_len = 0
            logger.warning("KV-Cache NOT found in model inputs!")
            if self.max_context_len > 0:
                logger.warning(
                    "algo.config.num_steps_per_env is set but KV-Cache is unavailable; "
                    "ignoring context window limit."
                )

        logger.info("ONNX Policy loaded successfully")

    def _read_onnx_metadata(self) -> dict:
        """Read model metadata from ONNX file and parse into Python types."""
        onnx_model_path = Path(self.config.ckpt_onnx_path)

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
        result["joint_names"] = [
            x for x in meta["joint_names"].split(",") if x != ""
        ]

        # 打印解析后的元数据
        logger.info("=== Loaded ONNX Metadata ===")
        for key, value in result.items():
            # 如果关节名称列表很长，进行格式化处理以保持整洁
            if key == "joint_names":
                logger.info(f"{key}: {', '.join(value)}")
            else:
                logger.info(f"{key}:\n{value}")
        logger.info("============================")

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
        self.onnx_to_mu = [
            self.dof_names_onnx.index(name) for name in self.mjcf_dof_names
        ]
        self.mu_to_onnx = [
            self.mjcf_dof_names.index(name) for name in self.dof_names_onnx
        ]
        self.ref_to_onnx = [
            self.dof_names_ref_motion.index(name)
            for name in self.dof_names_onnx
        ]

        # Map MuJoCo actuator DOF order -> reference DOF order used in motion npz
        self.mu_to_ref = []
        for mu_idx in range(len(self.mjcf_dof_names)):
            onnx_idx = self.onnx_to_mu[mu_idx]
            ref_idx = self.ref_to_onnx[onnx_idx]
            self.mu_to_ref.append(ref_idx)

        self.kps_mu = self.kps_onnx[self.onnx_to_mu].astype(np.float32)
        self.kds_mu = self.kds_onnx[self.onnx_to_mu].astype(np.float32)
        self.default_angles_mu = self.default_angles_onnx[
            self.onnx_to_mu
        ].astype(np.float32)
        self.action_scale_mu = self.action_scale_onnx[self.onnx_to_mu].astype(
            np.float32
        )

    @staticmethod
    def _normalize_filter_cutoff_hz(raw_values, num_frames: int) -> np.ndarray:
        num_frames = max(int(num_frames), 0)
        if num_frames == 0:
            return np.zeros((0, 1), dtype=np.float32)
        if raw_values is None:
            return np.zeros((num_frames, 1), dtype=np.float32)

        cutoff = np.asarray(raw_values, dtype=np.float32)
        if cutoff.ndim == 0:
            cutoff = np.full((num_frames, 1), float(cutoff), dtype=np.float32)
            return cutoff
        if cutoff.ndim == 1:
            cutoff = cutoff[:, None]
        else:
            cutoff = cutoff.reshape(cutoff.shape[0], -1)[:, :1]

        if cutoff.shape[0] == 0:
            return np.zeros((num_frames, 1), dtype=np.float32)
        if cutoff.shape[0] == 1 and num_frames > 1:
            cutoff = np.repeat(cutoff, num_frames, axis=0)
        elif cutoff.shape[0] < num_frames:
            pad = np.repeat(
                cutoff[-1:, :], num_frames - cutoff.shape[0], axis=0
            )
            cutoff = np.concatenate([cutoff, pad], axis=0)
        elif cutoff.shape[0] > num_frames:
            cutoff = cutoff[:num_frames]
        return cutoff.astype(np.float32, copy=False)

    def load_motion_data(self):
        """Load motion data from npz file."""
        motion_npz_path = self.config.get("motion_npz_path", None)
        if motion_npz_path is None:
            logger.warning(
                "No motion_npz_path specified in config, using zero reference motion"
            )
            return

        logger.info(f"Loading motion data from {motion_npz_path}")

        # Load npz file
        with np.load(motion_npz_path, allow_pickle=True) as npz:
            keys = list(npz.keys())
            raw_filter_cutoff_hz = (
                np.array(npz["filter_cutoff_hz"]).astype(np.float32)
                if "filter_cutoff_hz" in npz
                else None
            )

            # Try direct arrays first (dof_pos, dof_vel or variants)
            naming_pairs = [
                ("ref_dof_pos", "ref_dof_vel"),
                ("dof_pos", "dof_vels"),  # backward compat
                # ("ft_ref_pos", "ft_ref_dof_vel"),
            ]

            pos_key = None
            vel_key = None
            for pos_k, vel_k in naming_pairs:
                if pos_k in npz and vel_k in npz:
                    pos_key = pos_k
                    vel_key = vel_k
                    break

            if pos_key is not None and vel_key is not None:
                # Direct arrays found
                self.ref_dof_pos = np.array(npz[pos_key]).astype(np.float32)
                self.ref_dof_vel = np.array(npz[vel_key]).astype(np.float32)
            elif len(keys) == 1:
                # Single key - might contain nested dict
                arr = npz[keys[0]]
                if getattr(arr, "dtype", None) == object:
                    obj = arr.item() if arr.size == 1 else arr
                    if isinstance(obj, dict):
                        if (
                            raw_filter_cutoff_hz is None
                            and "filter_cutoff_hz" in obj
                        ):
                            raw_filter_cutoff_hz = np.array(
                                obj["filter_cutoff_hz"]
                            ).astype(np.float32)
                        # Try to find dof_pos/dof_vel in nested dict
                        for pos_k, vel_k in naming_pairs:
                            if pos_k in obj and vel_k in obj:
                                self.ref_dof_pos = np.array(obj[pos_k]).astype(
                                    np.float32
                                )
                                self.ref_dof_vel = np.array(obj[vel_k]).astype(
                                    np.float32
                                )
                                break
                        else:
                            raise ValueError(
                                f"Could not find dof_pos/dof_vel in nested dict. "
                                f"Available keys: {list(obj.keys())}"
                            )
                    else:
                        raise ValueError(
                            f"Single key '{keys[0]}' does not contain a dict. "
                            f"Type: {type(obj)}"
                        )
                else:
                    raise ValueError(
                        f"Single key '{keys[0]}' is not an object array. "
                        f"Available keys: {keys}"
                    )
            else:
                raise ValueError(
                    f"Could not find dof_pos/dof_vel arrays. Available keys: {keys}"
                )

            # Ensure consistent frame count
            if self.ref_dof_pos.shape[0] != self.ref_dof_vel.shape[0]:
                min_frames = min(
                    self.ref_dof_pos.shape[0], self.ref_dof_vel.shape[0]
                )
                self.ref_dof_pos = self.ref_dof_pos[:min_frames]
                self.ref_dof_vel = self.ref_dof_vel[:min_frames]
                logger.warning(
                    f"Frame count mismatch, truncated to {min_frames} frames"
                )

            self.n_motion_frames = self.ref_dof_pos.shape[0]

            # Optional: load reference global body frames as per motion spec
            ref_pos_keys = ["ref_global_translation", "global_translation"]
            ref_rot_keys = ["ref_global_rotation_quat", "global_rotation_quat"]
            ref_vel_keys = ["ref_global_velocity", "global_velocity"]
            ref_ang_vel_keys = [
                "ref_global_angular_velocity",
                "global_angular_velocity",
            ]
            self.ref_global_translation = None
            self.ref_global_rotation_quat_xyzw = None
            self.ref_global_velocity = None
            self.ref_global_angular_velocity = None
            for k in ref_pos_keys:
                if k in npz:
                    self.ref_global_translation = np.array(npz[k]).astype(
                        np.float32
                    )
                    break
            for k in ref_rot_keys:
                if k in npz:
                    self.ref_global_rotation_quat_xyzw = np.array(
                        npz[k]
                    ).astype(np.float32)
                    break
            for k in ref_vel_keys:
                if k in npz:
                    self.ref_global_velocity = np.array(npz[k]).astype(
                        np.float32
                    )
                    break
            for k in ref_ang_vel_keys:
                if k in npz:
                    self.ref_global_angular_velocity = np.array(npz[k]).astype(
                        np.float32
                    )
                    break
            if self.ref_global_translation is not None:
                # Truncate to motion frames if needed
                t_tr = min(
                    self.n_motion_frames, self.ref_global_translation.shape[0]
                )
                if t_tr < self.n_motion_frames:
                    logger.warning(
                        f"Global translation shorter than motion frames ({t_tr} < {self.n_motion_frames}), truncating motion."
                    )
                    self.n_motion_frames = t_tr
                    self.ref_dof_pos = self.ref_dof_pos[:t_tr]
                    self.ref_dof_vel = self.ref_dof_vel[:t_tr]

                self.ref_global_translation = self.ref_global_translation[
                    :t_tr
                ]
            if self.ref_global_rotation_quat_xyzw is not None:
                t_rr = min(
                    self.n_motion_frames,
                    self.ref_global_rotation_quat_xyzw.shape[0],
                )
                if t_rr < self.n_motion_frames:
                    logger.warning(
                        f"Global rotation shorter than motion frames ({t_rr} < {self.n_motion_frames}), truncating motion."
                    )
                    self.n_motion_frames = t_rr
                    self.ref_dof_pos = self.ref_dof_pos[:t_rr]
                    self.ref_dof_vel = self.ref_dof_vel[:t_rr]
                    # Also truncate previously processed globals if necessary
                    if self.ref_global_translation is not None:
                        self.ref_global_translation = (
                            self.ref_global_translation[:t_rr]
                        )

                self.ref_global_rotation_quat_xyzw = (
                    self.ref_global_rotation_quat_xyzw[:t_rr]
                )
            if self.ref_global_velocity is not None:
                t_rv = min(
                    self.n_motion_frames,
                    self.ref_global_velocity.shape[0],
                )
                if t_rv < self.n_motion_frames:
                    self.n_motion_frames = t_rv
                    self.ref_dof_pos = self.ref_dof_pos[:t_rv]
                    self.ref_dof_vel = self.ref_dof_vel[:t_rv]
                    if self.ref_global_translation is not None:
                        self.ref_global_translation = (
                            self.ref_global_translation[:t_rv]
                        )
                    if self.ref_global_rotation_quat_xyzw is not None:
                        self.ref_global_rotation_quat_xyzw = (
                            self.ref_global_rotation_quat_xyzw[:t_rv]
                        )

                self.ref_global_velocity = self.ref_global_velocity[:t_rv]
            if self.ref_global_angular_velocity is not None:
                t_ra = min(
                    self.n_motion_frames,
                    self.ref_global_angular_velocity.shape[0],
                )
                if t_ra < self.n_motion_frames:
                    self.n_motion_frames = t_ra
                    self.ref_dof_pos = self.ref_dof_pos[:t_ra]
                    self.ref_dof_vel = self.ref_dof_vel[:t_ra]
                    if self.ref_global_translation is not None:
                        self.ref_global_translation = (
                            self.ref_global_translation[:t_ra]
                        )
                    if self.ref_global_rotation_quat_xyzw is not None:
                        self.ref_global_rotation_quat_xyzw = (
                            self.ref_global_rotation_quat_xyzw[:t_ra]
                        )
                    if self.ref_global_velocity is not None:
                        self.ref_global_velocity = self.ref_global_velocity[
                            :t_ra
                        ]

                self.ref_global_angular_velocity = (
                    self.ref_global_angular_velocity[:t_ra]
                )

        self.filter_cutoff_hz = self._normalize_filter_cutoff_hz(
            raw_filter_cutoff_hz, self.n_motion_frames
        )
        logger.info(
            f"Loaded motion data with {self.n_motion_frames} frames and {self.ref_dof_pos.shape[1]} DOFs"
        )

    def load_mujoco_model(self):
        """Load the MuJoCo model."""
        xml_path = self.config.get("robot_xml_path", None)
        if xml_path is None:
            raise ValueError(
                "robot_xml_path should be specified in config !!!"
            )

        logger.info(f"Loading MuJoCo model from {xml_path}")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        logger.info(
            f"MuJoCo model loaded with {self.m.nq} position DOFs and {self.m.nu} control DOFs"
        )

    def _init_camera_config(self):
        """Initialize shared camera configuration for viewer and offscreen renderers."""
        self._root_body_id = -1
        if not self._camera_tracking_enabled:
            logger.info("Camera tracking disabled")
            return

        # Prefer anchor body from robot config, then fall back to common root names
        candidates: list[str] = []
        anchor_name = self._get_anchor_body_name()
        candidates.append(anchor_name)
        for name in ["pelvis", "torso", "base_link", "trunk", "root"]:
            if name not in candidates:
                candidates.append(name)

        for body_name in candidates:
            bid = int(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            )
            if bid != -1:
                self._root_body_id = bid
                break

        if self._root_body_id != -1:
            logger.info(
                f"Camera tracking enabled for body '{body_name}' (ID={self._root_body_id}), "
                f"lookat height offset: {self._camera_height_offset:.2f}m"
            )
        else:
            logger.warning(
                "Could not find robot root body for camera tracking; "
                "viewer and offscreen cameras will not track the robot."
            )

    def _configure_viewer_camera(self, viewer):
        """Apply shared align-view parameters to the interactive viewer camera."""
        mujoco.mjv_defaultFreeCamera(self.m, viewer.cam)
        viewer.cam.azimuth = self._camera_azimuth
        viewer.cam.elevation = self._camera_elevation
        viewer.cam.distance = self._camera_distance

    def _init_video_tools(self, tag: str):
        """Initialize video writer and offscreen renderer when recording is enabled."""
        if not bool(self.config.get("record_video", False)):
            return
        width = int(self.config.get("video_width", 1280))
        height = int(self.config.get("video_height", 720))
        fps = float(self.config.get("video_fps", 30.0))

        onnx_stem = os.path.splitext(
            os.path.basename(self.config.ckpt_onnx_path)
        )[0]
        output_dir = os.path.join(
            os.path.dirname(self.config.ckpt_onnx_path),
            f"mujoco_output_{onnx_stem}",
        )
        os.makedirs(output_dir, exist_ok=True)
        motion_npz_path = self.config.get("motion_npz_path", None)
        if motion_npz_path is not None:
            motion_stem = os.path.splitext(os.path.basename(motion_npz_path))[
                0
            ]
            out_path = os.path.join(output_dir, f"{motion_stem}.mp4")
        else:
            out_path = os.path.join(output_dir, "velocity_tracking.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            out_path, fourcc, fps, (width, height)
        )
        self._offscreen = OffscreenRenderer(
            self.m,
            height,
            width,
            distance=self._camera_distance,
            azimuth=self._camera_azimuth,
            elevation=self._camera_elevation,
        )
        self._frame_interval = 1.0 / max(fps, 1.0)
        self._last_frame_time = 0.0
        if getattr(self, "ref_global_translation", None) is not None:
            self._offscreen.set_overlay_callback(
                lambda scene: self._draw_ref_body_spheres_to_scene(
                    scene, reset_ngeom=False
                )
            )
        logger.info(f"Recording enabled. Writing to: {out_path}")

    def _dump_robot_augmented_npz(self) -> None:
        """Copy original motion npz and append robot_* states, saved next to video output.

        The output follows the holomotion offline-eval spec used in PPO:
        - robot_dof_pos, robot_dof_vel: [T, num_dofs]
        - robot_global_translation: [T, num_bodies, 3]
        - robot_global_rotation_quat: [T, num_bodies, 4] (XYZW)
        - robot_global_velocity: [T, num_bodies, 3]
        - robot_global_angular_velocity: [T, num_bodies, 3]
        """
        motion_npz_path = self.config.get("motion_npz_path", None)
        if motion_npz_path is None:
            return
        if len(self._robot_dof_pos_seq) == 0:
            return

        # Stack recorded sequences
        robot_dof_pos = np.stack(self._robot_dof_pos_seq, axis=0).astype(
            np.float32
        )
        robot_dof_vel = np.stack(self._robot_dof_vel_seq, axis=0).astype(
            np.float32
        )
        robot_dof_acc = np.stack(self._robot_dof_acc_seq, axis=0).astype(
            np.float32
        )
        robot_dof_torque = np.stack(self._robot_dof_torque_seq, axis=0).astype(
            np.float32
        )
        robot_low_level_dof_torque = None
        if len(self._robot_low_level_dof_torque_seq) > 0:
            robot_low_level_dof_torque = np.stack(
                self._robot_low_level_dof_torque_seq, axis=0
            ).astype(np.float32)
        (
            robot_low_level_foot_contact,
            robot_low_level_foot_normal_force,
            robot_low_level_foot_tangent_speed,
        ) = self._get_stacked_low_level_foot_contact_tensors()
        robot_actions = None
        if len(getattr(self, "_robot_actions_seq", [])) > 0:
            robot_actions = np.stack(self._robot_actions_seq, axis=0).astype(
                np.float32
            )
        robot_action_rate = np.asarray(
            self._robot_action_rate_seq, dtype=np.float32
        )

        robot_global_translation = np.stack(
            self._robot_global_translation_seq, axis=0
        ).astype(np.float32)
        robot_global_rotation_quat = np.stack(
            self._robot_global_rotation_quat_seq, axis=0
        ).astype(np.float32)
        robot_global_velocity = np.stack(
            self._robot_global_velocity_seq, axis=0
        ).astype(np.float32)
        robot_global_angular_velocity = np.stack(
            self._robot_global_angular_velocity_seq, axis=0
        ).astype(np.float32)
        robot_moe_expert_indices, robot_moe_expert_logits = (
            self._get_stacked_moe_routing_tensors()
        )

        # Load original motion npz
        with np.load(motion_npz_path, allow_pickle=True) as npz:
            data_dict = {k: npz[k] for k in npz.files}

        # Augment with robot_* arrays (override if already present)
        data_dict["robot_dof_pos"] = robot_dof_pos
        data_dict["robot_dof_vel"] = robot_dof_vel
        data_dict["robot_dof_acc"] = robot_dof_acc
        data_dict["robot_dof_torque"] = robot_dof_torque
        if robot_low_level_dof_torque is not None:
            data_dict["robot_low_level_dof_torque"] = (
                robot_low_level_dof_torque
            )
        if robot_low_level_foot_contact is not None:
            data_dict["robot_low_level_foot_contact"] = (
                robot_low_level_foot_contact
            )
        if robot_low_level_foot_normal_force is not None:
            data_dict["robot_low_level_foot_normal_force"] = (
                robot_low_level_foot_normal_force
            )
        if robot_low_level_foot_tangent_speed is not None:
            data_dict["robot_low_level_foot_tangent_speed"] = (
                robot_low_level_foot_tangent_speed
            )
        if robot_actions is not None:
            data_dict["robot_actions"] = robot_actions
        data_dict["robot_low_level_torque_dt"] = np.array(
            self.simulation_dt, dtype=np.float32
        )
        data_dict["robot_low_level_contact_dt"] = np.array(
            self.simulation_dt, dtype=np.float32
        )
        data_dict["robot_action_rate"] = robot_action_rate
        data_dict["robot_global_translation"] = robot_global_translation
        data_dict["robot_global_rotation_quat"] = robot_global_rotation_quat
        data_dict["robot_global_velocity"] = robot_global_velocity
        data_dict["robot_global_angular_velocity"] = (
            robot_global_angular_velocity
        )
        if robot_moe_expert_indices is not None:
            data_dict["robot_moe_expert_indices"] = robot_moe_expert_indices
        if robot_moe_expert_logits is not None:
            data_dict["robot_moe_expert_logits"] = robot_moe_expert_logits

        # Derive output directory consistent with video writer
        onnx_stem = os.path.splitext(
            os.path.basename(self.config.ckpt_onnx_path)
        )[0]
        output_dir = os.path.join(
            os.path.dirname(self.config.ckpt_onnx_path),
            f"mujoco_output_{onnx_stem}",
        )
        os.makedirs(output_dir, exist_ok=True)
        motion_stem = os.path.splitext(os.path.basename(motion_npz_path))[0]
        out_npz_path = os.path.join(output_dir, f"{motion_stem}_robot.npz")

        np.savez_compressed(out_npz_path, **data_dict)
        logger.info(
            f"Robot-augmented motion npz saved to: {out_npz_path} "
            f"(T={robot_dof_pos.shape[0]}, num_dofs={robot_dof_pos.shape[1]}, "
            f"num_bodies={robot_global_translation.shape[1]})"
        )

    def _close_video_tools(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if self._offscreen is not None:
            self._offscreen.close()
            self._offscreen = None
        self._frame_interval = None
        self._last_frame_time = 0.0

    def _update_camera_lookat(self, cam):
        """Update camera lookat to track the robot root when tracking is enabled."""
        if not self._camera_tracking_enabled:
            return
        if self._root_body_id == -1:
            return
        cam.lookat[:2] = self.d.xpos[self._root_body_id][:2]
        cam.lookat[2] = (
            self.d.xpos[self._root_body_id][2] + self._camera_height_offset
        )

    def _maybe_record_frame(self):
        if self._video_writer is None or self._offscreen is None:
            return
        now = time.time()
        if (
            self._last_frame_time == 0.0
            or (now - self._last_frame_time) >= self._frame_interval
        ):
            # Update offscreen camera lookat to track robot (if enabled)
            self._update_camera_lookat(self._offscreen._cam)

            frame_rgb = self._offscreen.render(self.d)
            # Convert RGB (MuJoCo) -> BGR (OpenCV) before writing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._video_writer.write(frame_bgr)
            self._last_frame_time = now

    def _apply_control(self, sleep: bool):
        """Apply PD targets via Unitree lowcmd, step MuJoCo, optionally sleep."""
        for _ in range(self.control_decimation):
            record_low_level_torque = (
                self.command_mode == "motion_tracking"
                and self.ref_dof_pos is not None
            )
            if record_low_level_torque:
                torque_ref = np.zeros(
                    len(self.dof_names_ref_motion), dtype=np.float32
                )
            current_dof_pos = self.robot_dof_pos
            current_dof_vel = self.robot_dof_vel
            for name, act_idx in self.actuator_name_to_index.items():
                mu_idx = self.actuator_name_to_mu_idx[name]
                joint_name = self.mjcf_dof_names[mu_idx]
                target_q = self.target_dof_pos_by_name.get(
                    joint_name,
                    float(self.default_angles_mu[mu_idx]),
                )
                target_dq = 0.0
                feedforward_tau = 0.0
                kp = self.kps_mu[mu_idx]
                kd = self.kds_mu[mu_idx]
                current_q = current_dof_pos[mu_idx]
                current_dq = current_dof_vel[mu_idx]
                tau = (
                    feedforward_tau
                    + kp * (target_q - current_q)
                    + kd * (target_dq - current_dq)
                )
                if (
                    act_idx in self.actuator_force_range
                    and self.actuator_force_range[act_idx] is not None
                ):
                    min_force, max_force = self.actuator_force_range[act_idx]
                    tau = np.clip(tau, min_force, max_force)
                self.d.ctrl[mu_idx] = tau
                if record_low_level_torque:
                    torque_ref[self.mu_to_ref[mu_idx]] = np.float32(tau)

            mujoco.mj_step(self.m, self.d)
            if record_low_level_torque:
                self._robot_low_level_dof_torque_seq.append(torque_ref)
                self._record_low_level_foot_contact_sample()
            if sleep:
                time.sleep(self.simulation_dt)

    def _compute_pd_torque_command_ref(self) -> np.ndarray:
        current_dof_pos = self.robot_dof_pos
        current_dof_vel = self.robot_dof_vel

        num_mu_dofs = len(self.mjcf_dof_names)
        torque_mu = np.zeros(num_mu_dofs, dtype=np.float32)
        for name, act_idx in self.actuator_name_to_index.items():
            mu_idx = self.actuator_name_to_mu_idx[name]
            joint_name = self.mjcf_dof_names[mu_idx]
            target_q = self.target_dof_pos_by_name.get(
                joint_name,
                float(self.default_angles_mu[mu_idx]),
            )
            target_dq = 0.0
            feedforward_tau = 0.0
            kp = self.kps_mu[mu_idx]
            kd = self.kds_mu[mu_idx]
            current_q = current_dof_pos[mu_idx]
            current_dq = current_dof_vel[mu_idx]
            tau = (
                feedforward_tau
                + kp * (target_q - current_q)
                + kd * (target_dq - current_dq)
            )
            if (
                act_idx in self.actuator_force_range
                and self.actuator_force_range[act_idx] is not None
            ):
                min_force, max_force = self.actuator_force_range[act_idx]
                tau = np.clip(tau, min_force, max_force)
            torque_mu[mu_idx] = np.float32(tau)

        num_ref_dofs = len(self.dof_names_ref_motion)
        torque_ref = np.zeros(num_ref_dofs, dtype=np.float32)
        for mu_idx, ref_idx in enumerate(self.mu_to_ref):
            torque_ref[ref_idx] = torque_mu[mu_idx]
        return torque_ref

    def _get_obs_ref_motion_states(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        ref_pos_mu = self.ref_dof_pos[frame_idx]
        ref_vel_mu = self.ref_dof_vel[frame_idx]
        # Map URDF/Mu order -> ONNX order using precomputed indices
        ref_pos_onnx = ref_pos_mu[self.ref_to_onnx].astype(np.float32)
        ref_vel_onnx = ref_vel_mu[self.ref_to_onnx].astype(np.float32)
        return np.concatenate([ref_pos_onnx, ref_vel_onnx], axis=0).astype(
            np.float32
        )

    def _get_obs_ref_motion_states_fut(self):
        # [T, 2 * num_actions] flattened, ONNX order
        T = int(self.n_fut_frames)
        if T <= 0 or self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(0, dtype=np.float32)
        N = int(self.num_actions)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        # Build future arrays in Mu order [N, T]
        pos_fut = np.zeros(
            (len(self.dof_names_ref_motion), T), dtype=np.float32
        )
        vel_fut = np.zeros(
            (len(self.dof_names_ref_motion), T), dtype=np.float32
        )
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                pos_fut[:, i] = self.ref_dof_pos[idx]
                vel_fut[:, i] = self.ref_dof_vel[idx]
            else:
                pos_fut[:, i] = self.ref_dof_pos[last_valid_frame_idx]
                vel_fut[:, i] = self.ref_dof_vel[last_valid_frame_idx]
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :]  # [N, T]
        vel_fut_onnx = vel_fut[self.ref_to_onnx, :]  # [N, T]
        fut_concat = np.concatenate(
            [pos_fut_onnx.T, vel_fut_onnx.T], axis=1
        )  # [T, 2N]
        return fut_concat.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_fut(self):
        # [T, 2 * num_actions] flattened, ONNX order
        T = int(self.n_fut_frames)
        if T <= 0 or self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        # Build future arrays in Mu order [N, T]
        pos_fut = np.zeros(
            (len(self.dof_names_ref_motion), T), dtype=np.float32
        )
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                pos_fut[:, i] = self.ref_dof_pos[idx]
            else:
                pos_fut[:, i] = self.ref_dof_pos[last_valid_frame_idx]
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :].transpose(1, 0)  # [N, T]
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_height_fut(self):
        T = int(self.n_fut_frames)
        if (
            T <= 0
            or self.ref_dof_pos is None
            or self.ref_dof_vel is None
            or getattr(self, "ref_global_translation", None) is None
        ):
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        # Build future arrays in Mu order [N, T]
        h_fut = np.zeros((1, T), dtype=np.float32)
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                h_fut[:, i] = self.ref_global_translation[
                    idx, self.root_body_idx, 2
                ]
            else:
                h_fut[:, i] = self.ref_global_translation[
                    last_valid_frame_idx, self.root_body_idx, 2
                ]
        return h_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_cur(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        ref_pos_mu = self.ref_dof_pos[self.motion_frame_idx]
        # Map URDF/Mu order -> ONNX order using precomputed indices
        ref_pos_onnx = ref_pos_mu[self.ref_to_onnx].astype(np.float32)
        return ref_pos_onnx

    def _get_obs_ref_dof_vel_cur(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        ref_vel_mu = self.ref_dof_vel[self.motion_frame_idx]
        # Map URDF/Mu order -> ONNX order using precomputed indices
        ref_vel_onnx = ref_vel_mu[self.ref_to_onnx].astype(np.float32)
        return ref_vel_onnx

    def _get_obs_ref_motion_filter_cutoff_hz(self):
        # cutoff = getattr(self, "filter_cutoff_hz", None)
        cutoff = 1.0
        if cutoff is None:
            return np.float32(0.0)
        cutoff_flat = np.asarray(cutoff, dtype=np.float32).reshape(-1)
        if cutoff_flat.size == 0:
            return np.float32(0.0)
        frame_idx = min(
            max(int(getattr(self, "motion_frame_idx", 0)), 0),
            cutoff_flat.size - 1,
        )
        return np.float32(cutoff_flat[frame_idx])

    def _get_obs_ref_root_height_cur(self):
        if getattr(self, "ref_global_translation", None) is None:
            return 0.0
        return self.ref_global_translation[
            self.motion_frame_idx, self.root_body_idx, 2
        ]

    def _get_obs_ref_gravity_projection_cur(self):
        if (
            getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_global_rotation_quat_xyzw[
            self.motion_frame_idx, self.root_body_idx
        ].astype(np.float32)
        q_root_wxyz = xyzw_to_wxyz(
            torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
        )
        q_root_wxyz = standardize_quaternion(q_root_wxyz)
        g_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device="cpu")
        g_root = quat_apply(quat_inv(q_root_wxyz), g_w)
        return g_root.detach().cpu().numpy().astype(np.float32)

    def _get_obs_ref_gravity_projection_fut(self):
        T = int(self.n_fut_frames)
        if (
            T <= 0
            or getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        g_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device="cpu")
        gravity_fut = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            idx = frame_idx + i + 1
            if idx >= self.n_motion_frames:
                idx = last_valid_frame_idx
            q_root_xyzw = self.ref_global_rotation_quat_xyzw[
                idx, self.root_body_idx
            ].astype(np.float32)
            q_root_wxyz = xyzw_to_wxyz(
                torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
            )
            q_root_wxyz = standardize_quaternion(q_root_wxyz)
            gravity_fut[i] = (
                quat_apply(quat_inv(q_root_wxyz), g_w)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return gravity_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_linvel_cur(self):
        if (
            getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or getattr(self, "ref_global_velocity", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_global_rotation_quat_xyzw[
            self.motion_frame_idx, self.root_body_idx
        ].astype(np.float32)
        q_root_wxyz = xyzw_to_wxyz(
            torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
        )
        q_root_wxyz = standardize_quaternion(q_root_wxyz)
        v_root_w = torch.as_tensor(
            self.ref_global_velocity[
                self.motion_frame_idx, self.root_body_idx
            ],
            dtype=torch.float32,
            device="cpu",
        )
        v_root = quat_apply(quat_inv(q_root_wxyz), v_root_w)
        return v_root.detach().cpu().numpy().astype(np.float32)

    def _get_obs_ref_base_linvel_fut(self):
        T = int(self.n_fut_frames)
        if (
            T <= 0
            or getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or getattr(self, "ref_global_velocity", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        base_linvel_fut = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            idx = frame_idx + i + 1
            if idx >= self.n_motion_frames:
                idx = last_valid_frame_idx
            q_root_xyzw = self.ref_global_rotation_quat_xyzw[
                idx, self.root_body_idx
            ].astype(np.float32)
            q_root_wxyz = xyzw_to_wxyz(
                torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
            )
            q_root_wxyz = standardize_quaternion(q_root_wxyz)
            v_root_w = torch.as_tensor(
                self.ref_global_velocity[idx, self.root_body_idx],
                dtype=torch.float32,
                device="cpu",
            )
            base_linvel_fut[i] = (
                quat_apply(quat_inv(q_root_wxyz), v_root_w)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return base_linvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_angvel_cur(self):
        if (
            getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or getattr(self, "ref_global_angular_velocity", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_global_rotation_quat_xyzw[
            self.motion_frame_idx, self.root_body_idx
        ].astype(np.float32)
        q_root_wxyz = xyzw_to_wxyz(
            torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
        )
        q_root_wxyz = standardize_quaternion(q_root_wxyz)
        w_root_w = torch.as_tensor(
            self.ref_global_angular_velocity[
                self.motion_frame_idx, self.root_body_idx
            ],
            dtype=torch.float32,
            device="cpu",
        )
        w_root = quat_apply(quat_inv(q_root_wxyz), w_root_w)
        return w_root.detach().cpu().numpy().astype(np.float32)

    def _get_obs_ref_base_angvel_fut(self):
        T = int(self.n_fut_frames)
        if (
            T <= 0
            or getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or getattr(self, "ref_global_angular_velocity", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(0, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        base_angvel_fut = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            idx = frame_idx + i + 1
            if idx >= self.n_motion_frames:
                idx = last_valid_frame_idx
            q_root_xyzw = self.ref_global_rotation_quat_xyzw[
                idx, self.root_body_idx
            ].astype(np.float32)
            q_root_wxyz = xyzw_to_wxyz(
                torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
            )
            q_root_wxyz = standardize_quaternion(q_root_wxyz)
            w_root_w = torch.as_tensor(
                self.ref_global_angular_velocity[idx, self.root_body_idx],
                dtype=torch.float32,
                device="cpu",
            )
            base_angvel_fut[i] = (
                quat_apply(quat_inv(q_root_wxyz), w_root_w)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return base_angvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_keybody_rel_pos_cur(self):
        keybody_idxs = self._get_ref_keybody_indices(
            "actor_ref_keybody_rel_pos_cur"
        )
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros(0, dtype=np.float32)
        if (
            getattr(self, "ref_global_translation", None) is None
            or getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros(n_keybodies * 3, dtype=np.float32)

        frame_idx = self.motion_frame_idx
        ref_body_global_pos = self.ref_global_translation[frame_idx].astype(
            np.float32
        )  # [B, 3]
        ref_root_global_pos = ref_body_global_pos[
            self.root_body_idx
        ]  # [3], world
        q_root_xyzw = self.ref_global_rotation_quat_xyzw[
            frame_idx, self.root_body_idx
        ].astype(np.float32)
        q_root_wxyz = xyzw_to_wxyz(
            torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
        )
        q_root_wxyz = standardize_quaternion(q_root_wxyz)

        rel_pos_w = (
            ref_body_global_pos[keybody_idxs] - ref_root_global_pos[None, :]
        )  # [K, 3]
        rel_pos_w_t = torch.as_tensor(
            rel_pos_w, dtype=torch.float32, device="cpu"
        )
        q_root_expand = q_root_wxyz.unsqueeze(0).expand(n_keybodies, 4)
        rel_pos_root_t = quat_apply(quat_inv(q_root_expand), rel_pos_w_t)
        return (
            rel_pos_root_t.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    def _get_obs_ref_keybody_rel_pos_fut(self):
        T = int(self.n_fut_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)

        keybody_idxs = self._get_ref_keybody_indices(
            "actor_ref_keybody_rel_pos_fut"
        )
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros((T, 0), dtype=np.float32)
        if (
            getattr(self, "ref_global_translation", None) is None
            or getattr(self, "ref_global_rotation_quat_xyzw", None) is None
            or self.n_motion_frames <= 0
        ):
            return np.zeros((T, n_keybodies * 3), dtype=np.float32)

        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        rel_pos_fut = np.zeros((T, n_keybodies, 3), dtype=np.float32)

        for i in range(T):
            idx = frame_idx + i + 1
            if idx >= self.n_motion_frames:
                idx = last_valid_frame_idx

            ref_body_global_pos = self.ref_global_translation[idx].astype(
                np.float32
            )  # [B, 3]
            ref_root_global_pos = ref_body_global_pos[
                self.root_body_idx
            ]  # [3], world
            q_root_xyzw = self.ref_global_rotation_quat_xyzw[
                idx, self.root_body_idx
            ].astype(np.float32)
            q_root_wxyz = xyzw_to_wxyz(
                torch.as_tensor(q_root_xyzw, dtype=torch.float32, device="cpu")
            )
            q_root_wxyz = standardize_quaternion(q_root_wxyz)

            rel_pos_w = (
                ref_body_global_pos[keybody_idxs]
                - ref_root_global_pos[None, :]
            )  # [K, 3]
            rel_pos_w_t = torch.as_tensor(
                rel_pos_w, dtype=torch.float32, device="cpu"
            )
            q_root_expand = q_root_wxyz.unsqueeze(0).expand(n_keybodies, 4)
            rel_pos_fut[i] = (
                quat_apply(quat_inv(q_root_expand), rel_pos_w_t)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return rel_pos_fut.reshape(T, -1).astype(np.float32)

    def _get_obs_place_holder(self):
        return np.zeros(self.actor_place_holder_ndim, dtype=np.float32)

    def _get_obs_vr_ref_motion_states(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        frame_idx = self.motion_frame_idx
        ref_pos_mu = self.ref_dof_pos[frame_idx]
        # Map URDF/Mu order -> ONNX order using precomputed indices
        ref_pos_onnx = ref_pos_mu[self.ref_to_onnx].astype(np.float32)
        return np.concatenate(
            [ref_pos_onnx, np.zeros_like(ref_pos_onnx)],
            axis=0,
        ).astype(np.float32)

    def _get_obs_vr_ref_motion_states_fut(self):
        # [T, 2 * num_actions] flattened, ONNX order
        T = int(self.n_fut_frames)
        if T <= 0 or self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(0, dtype=np.float32)
        N = int(self.num_actions)
        frame_idx = self.motion_frame_idx
        last_valid_frame_idx = self.n_motion_frames - 1
        # Build future arrays in Mu order [N, T]
        pos_fut = np.zeros(
            (len(self.dof_names_ref_motion), T), dtype=np.float32
        )
        for i in range(T):
            idx = frame_idx + i + 1
            if idx < self.n_motion_frames:
                pos_fut[:, i] = self.ref_dof_pos[idx]
            else:
                pos_fut[:, i] = self.ref_dof_pos[last_valid_frame_idx]
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :]  # [N, T]
        fut_concat = np.concatenate(
            [pos_fut_onnx.T, np.zeros_like(pos_fut_onnx.T)], axis=1
        )  # [T, 2N]
        return fut_concat.reshape(-1).astype(np.float32)

    def _get_obs_rel_robot_root_ang_vel(self):
        q_root_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[self.root_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        w_root_w = torch.as_tensor(
            self.robot_global_bodylink_ang_vel[self.root_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        w_root_b = quat_apply(quat_inv(q_root_wxyz), w_root_w)
        return w_root_b.detach().cpu().numpy().astype(np.float32)

    def _get_obs_last_action(self):
        return np.array(self.actions_onnx, dtype=np.float32).reshape(-1)

    def _get_obs_velocity_command(self):
        # Extended velocity command: [move_mask, vx, vy, vyaw]
        if (
            self.command_mode == "velocity_tracking"
            and getattr(self, "keyboard_handler", None) is not None
        ):
            cmd = np.asarray(
                self.keyboard_handler.get_velocity_command(), dtype=np.float32
            ).reshape(3)
        else:
            cmd = np.zeros(3, dtype=np.float32)
        out = np.zeros(4, dtype=np.float32)
        out[1:] = cmd
        out[0] = float(np.linalg.norm(cmd) > 0.1)
        return out

    def _get_obs_actor_ref_headling_aligned_vel_cmd(self):
        return self._get_obs_velocity_command()

    # ----------------- Actor term aliases (PULSE stage2 unified obs) -----------------
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

    def _get_obs_actor_place_holder(self):
        return self._get_obs_place_holder()

    def _get_obs_actor_ref_dof_pos_fut(self):
        return self._get_obs_ref_dof_pos_fut()

    def _get_obs_actor_ref_dof_pos_cur(self):
        return self._get_obs_ref_dof_pos_cur()

    def _get_obs_actor_ref_motion_filter_cutoff_hz(self):
        return self._get_obs_ref_motion_filter_cutoff_hz()

    def _get_obs_actor_ref_root_height_fut(self):
        return self._get_obs_ref_root_height_fut()

    def _get_obs_actor_ref_root_height_cur(self):
        return self._get_obs_ref_root_height_cur()

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

    def _get_obs_actor_ref_keybody_rel_pos_cur(self):
        return self._get_obs_ref_keybody_rel_pos_cur()

    def _get_obs_actor_ref_keybody_rel_pos_fut(self):
        return self._get_obs_ref_keybody_rel_pos_fut()

    def _get_obs_global_anchor_diff(self):
        self._ensure_ref_to_sim_transform_rigid()
        ref_pos_sim = self.ref_global_bodylink_pos
        ref_rot_sim = self.ref_global_bodylink_rot

        if ref_pos_sim is None or ref_rot_sim is None:
            return np.zeros(9, dtype=np.float32)

        t_robot = torch.as_tensor(
            self.robot_global_bodylink_pos[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_robot_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        t_ref_sim = torch.as_tensor(
            ref_pos_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_ref_sim = torch.as_tensor(
            ref_rot_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        # Use isaaclab semantics: pose of ref (frame 2) w.r.t. robot (frame 1)
        p_diff_t, q_diff_wxyz_t = subtract_frame_transforms(
            t01=t_robot,
            q01=q_robot_wxyz,
            t02=t_ref_sim,
            q02=q_ref_sim,
        )
        q_diff_wxyz_t = quat_normalize_wxyz(q_diff_wxyz_t)
        rot_diff_mat = matrix_from_quat(q_diff_wxyz_t)
        out = torch.cat(
            [p_diff_t.reshape(-1), rot_diff_mat[..., :2].reshape(-1)], dim=-1
        )
        return out.detach().cpu().numpy().astype(np.float32)

    def _get_obs_global_anchor_pos_diff(self):
        self._ensure_ref_to_sim_transform_rigid()
        ref_pos_sim = self.ref_global_bodylink_pos
        ref_rot_sim = self.ref_global_bodylink_rot

        if ref_pos_sim is None or ref_rot_sim is None:
            return np.zeros(3, dtype=np.float32)

        t_robot = torch.as_tensor(
            self.robot_global_bodylink_pos[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )  # [3], world
        q_robot_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )  # [4], wxyz

        # Transform reference anchor pose into simulation global frame
        t_ref_sim = torch.as_tensor(
            ref_pos_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_ref_sim = torch.as_tensor(
            ref_rot_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )

        pos_diff_anchor_t, _ = subtract_frame_transforms(
            t01=t_robot,
            q01=q_robot_wxyz,
            t02=t_ref_sim,
            q02=q_ref_sim,
        )

        return pos_diff_anchor_t.detach().cpu().numpy().astype(np.float32)

    def _get_obs_global_anchor_rot_diff(self):
        self._ensure_ref_to_sim_transform_rigid()
        ref_pos_sim = self.ref_global_bodylink_pos
        ref_rot_sim = self.ref_global_bodylink_rot

        if ref_pos_sim is None or ref_rot_sim is None:
            return np.zeros(6, dtype=np.float32)

        t_robot = torch.as_tensor(
            self.robot_global_bodylink_pos[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_robot_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_robot_wxyz = standardize_quaternion(q_robot_wxyz)

        t_ref_sim = torch.as_tensor(
            ref_pos_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_ref_sim = torch.as_tensor(
            ref_rot_sim[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        q_ref_sim = standardize_quaternion(q_ref_sim)
        _, q_diff_wxyz_t = subtract_frame_transforms(
            t01=t_robot,
            q01=q_robot_wxyz,
            t02=t_ref_sim,
            q02=q_ref_sim,
        )
        q_diff_wxyz_t = standardize_quaternion(q_diff_wxyz_t)

        rot_diff_mat = matrix_from_quat(q_diff_wxyz_t)

        return (
            rot_diff_mat[..., :2]
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _get_obs_global_bodylink_translation(self) -> np.ndarray:
        """Global body translations in simulator/URDF order, flattened as [num_bodies * 3].

        The body dimension excludes the MuJoCo world body and is assumed to match
        the NPZ `*_global_translation` arrays (root at index 0).
        """
        pos = self.robot_global_bodylink_pos.astype(np.float32)  # [B, 3]
        return pos.reshape(-1)

    def _get_obs_global_bodylink_rotation_quat(self) -> np.ndarray:
        """Global body rotations as XYZW quaternions in simulator/URDF order, flattened [num_bodies * 4]."""
        q_wxyz = self.robot_global_bodylink_rot  # [B, 4] in w, x, y, z
        q_xyzw = np.empty_like(q_wxyz, dtype=np.float32)
        q_xyzw[..., 0] = q_wxyz[..., 1]
        q_xyzw[..., 1] = q_wxyz[..., 2]
        q_xyzw[..., 2] = q_wxyz[..., 3]
        q_xyzw[..., 3] = q_wxyz[..., 0]
        return q_xyzw.reshape(-1)

    def _get_obs_global_bodylink_velocity(self) -> np.ndarray:
        """Global body linear velocities in world frame, flattened [num_bodies * 3]."""
        lin_vel = self.robot_global_bodylink_lin_vel.astype(
            np.float32
        )  # [B, 3]
        return lin_vel.reshape(-1)

    def _get_obs_global_bodylink_angular_velocity(self) -> np.ndarray:
        """Global body angular velocities in world frame, flattened [num_bodies * 3]."""
        ang_vel = self.robot_global_bodylink_ang_vel.astype(
            np.float32
        )  # [B, 3]
        return ang_vel.reshape(-1)

    @property
    def ref_global_bodylink_pos(self) -> np.ndarray | None:
        """Reference body positions transformed into the simulator global frame.

        Uses the yaw+translation Ref->Sim rigid transform computed from the initial robot
        global pose so that the reference motion is expressed in the same world frame as
        the robot (matching XY translation and yaw at frame 0).

        Returns:
            Array of shape [num_bodies, 3] giving reference positions in simulator world frame,
            or None if reference globals are not available.
        """
        if getattr(self, "ref_global_translation", None) is None:
            return None
        if self.n_motion_frames <= 0:
            return None

        self._ensure_ref_to_sim_transform_rigid()

        frame_idx = self.ref_motion_frame_idx
        ref_pos_world = self.ref_global_translation[frame_idx].astype(
            np.float32
        )  # [B, 3]

        pos_world_t = torch.as_tensor(
            ref_pos_world, dtype=torch.float32, device="cpu"
        )

        q_ref_to_sim = torch.as_tensor(
            self._ref_to_sim_q_wxyz, dtype=torch.float32, device="cpu"
        )
        q_ref_to_sim = q_ref_to_sim.unsqueeze(0).expand(
            pos_world_t.shape[0], 4
        )

        t_ref_to_sim = torch.as_tensor(
            self._ref_to_sim_t, dtype=torch.float32, device="cpu"
        )

        # Apply yaw rotation + translation based on initial robot state
        pos_sim_t = (
            quat_apply(q_ref_to_sim, pos_world_t) + t_ref_to_sim[None, :]
        )

        return pos_sim_t.detach().cpu().numpy().astype(np.float32)

    @property
    def ref_global_bodylink_rot(self) -> np.ndarray | None:
        """Reference body rotations transformed into the simulator global frame.

        Uses the yaw component of the Ref->Sim transform so that the reference motion's
        global yaw is aligned with the robot's initial yaw, while preserving roll/pitch
        from the motion data.

        Returns:
            Array of shape [num_bodies, 4] giving reference orientations in WXYZ format,
            or None if reference globals are not available.
        """
        if getattr(self, "ref_global_rotation_quat_xyzw", None) is None:
            return None
        if self.n_motion_frames <= 0:
            return None

        frame_idx = self.ref_motion_frame_idx
        ref_rot_xyzw = self.ref_global_rotation_quat_xyzw[frame_idx].astype(
            np.float32
        )  # [B, 4] in XYZW

        q_ref_xyzw_t = torch.as_tensor(
            ref_rot_xyzw, dtype=torch.float32, device="cpu"
        )
        q_ref_wxyz_t = xyzw_to_wxyz(q_ref_xyzw_t)
        q_ref_wxyz_t = standardize_quaternion(q_ref_wxyz_t)

        q_ref_to_sim = torch.as_tensor(
            self._ref_to_sim_q_wxyz, dtype=torch.float32, device="cpu"
        )
        q_ref_to_sim = q_ref_to_sim.unsqueeze(0).expand_as(q_ref_wxyz_t)

        q_ref_sim_wxyz_t = quat_mul(q_ref_to_sim, q_ref_wxyz_t)
        q_ref_sim_wxyz_t = standardize_quaternion(q_ref_sim_wxyz_t)

        return q_ref_sim_wxyz_t.detach().cpu().numpy().astype(np.float32)

    def _draw_ref_body_spheres_to_scene(
        self, scene, reset_ngeom: bool
    ) -> None:
        """Draw blue spheres at reference body positions into a MuJoCo scene."""
        ref_positions_sim = self.ref_global_bodylink_pos
        if ref_positions_sim is None:
            if reset_ngeom:
                scene.ngeom = 0
            return

        if reset_ngeom:
            scene.ngeom = 0

        radius = float(self.config.get("ref_marker_radius", 0.03))
        rgba = np.array([0.8, 0.0, 0.0, 1.0], dtype=np.float32)
        size = np.array([radius, 0.0, 0.0], dtype=np.float32)
        mat = np.eye(3, dtype=np.float32).reshape(-1)

        start = int(scene.ngeom)
        idx = 0
        for pos in ref_positions_sim:
            geom_id = start + idx
            if geom_id >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[geom_id],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos.astype(np.float32),
                mat,
                rgba,
            )
            idx += 1
        scene.ngeom = start + idx

    def _get_obs_rel_anchor_lin_vel(self):
        # Anchor linear velocity expressed in the anchor frame (IsaacLab semantics)
        q_anchor_wxyz = torch.as_tensor(
            self.robot_global_bodylink_rot[self.anchor_body_idx],
            dtype=torch.float32,
            device="cpu",
        )
        v_local_t = quat_apply(
            quat_inv(q_anchor_wxyz),
            torch.as_tensor(
                self.robot_global_bodylink_lin_vel[self.anchor_body_idx],
                dtype=torch.float32,
                device="cpu",
            ),
        )
        return v_local_t.detach().cpu().numpy().astype(np.float32)

    def _get_obs_projected_gravity(self):
        q = torch.as_tensor(
            self.robot_global_bodylink_rot[self.root_body_idx],
            dtype=torch.float32,
        )
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        gravity_orientation = torch.zeros(3, dtype=torch.float32, device="cpu")
        gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return gravity_orientation.detach().cpu().numpy().astype(np.float32)

    def _get_obs_dof_pos(self):
        pos_mu = self.robot_dof_pos
        pos_onnx = pos_mu[self.mu_to_onnx]
        return (pos_onnx - self.default_angles_onnx.astype(np.float32)).astype(
            np.float32
        )

    def _get_obs_dof_vel(self):
        vel_mu = self.robot_dof_vel
        vel_onnx = vel_mu[self.mu_to_onnx]
        return vel_onnx.astype(np.float32)

    def _record_robot_states(self) -> None:
        """Record current robot DOF and global body states for offline NPZ dumping.

        - DOF states are stored in reference DOF order (config.robot.dof_names).
        - Body states are stored in dataset/URDF order (config.robot.body_names).
        """
        if self.command_mode != "motion_tracking":
            return
        if self.ref_dof_pos is None or self.n_motion_frames <= 0:
            return
        if len(self._robot_dof_pos_seq) >= self.n_motion_frames:
            return

        # Joint positions/velocities from Unitree lowstate in actuator (MuJoCo) order
        pos_mu = self.robot_dof_pos
        vel_mu = self.robot_dof_vel

        # Map MuJoCo actuator order -> reference DOF order
        num_dofs = len(self.dof_names_ref_motion)
        pos_ref = np.zeros(num_dofs, dtype=np.float32)
        vel_ref = np.zeros(num_dofs, dtype=np.float32)
        for mu_idx, ref_idx in enumerate(self.mu_to_ref):
            pos_ref[ref_idx] = pos_mu[mu_idx]
            vel_ref[ref_idx] = vel_mu[mu_idx]

        self._robot_dof_pos_seq.append(pos_ref)
        self._robot_dof_vel_seq.append(vel_ref)
        if self._prev_recorded_dof_vel_ref is None:
            acc_ref = np.zeros_like(vel_ref, dtype=np.float32)
        else:
            acc_ref = (vel_ref - self._prev_recorded_dof_vel_ref) / np.float32(
                self.policy_dt
            )
        self._prev_recorded_dof_vel_ref = vel_ref.copy()
        self._robot_dof_acc_seq.append(acc_ref.astype(np.float32))

        # Global bodylink states in dataset/URDF order
        body_count = int(self.robot_global_bodylink_pos.shape[0])
        trans = self._get_obs_global_bodylink_translation().reshape(
            body_count, 3
        )
        rot = self._get_obs_global_bodylink_rotation_quat().reshape(
            body_count, 4
        )
        vel = self._get_obs_global_bodylink_velocity().reshape(body_count, 3)
        ang_vel = self._get_obs_global_bodylink_angular_velocity().reshape(
            body_count, 3
        )

        self._robot_global_translation_seq.append(trans)
        self._robot_global_rotation_quat_seq.append(rot)
        self._robot_global_velocity_seq.append(vel)
        self._robot_global_angular_velocity_seq.append(ang_vel)

    def load_specific_motion(self, npz_path):
        with np.load(npz_path, allow_pickle=True) as npz:
            self.ref_global_translation = npz["ref_global_translation"]
            self.ref_global_rotation_quat_xyzw = npz[
                "ref_global_rotation_quat"
            ]
            self.ref_global_velocity = npz["ref_global_velocity"]
            self.ref_global_angular_velocity = npz[
                "ref_global_angular_velocity"
            ]
            self.ref_dof_pos = npz["ref_dof_pos"]
            self.ref_dof_vel = npz["ref_dof_vel"]
            raw_filter_cutoff_hz = (
                np.array(npz["filter_cutoff_hz"]).astype(np.float32)
                if "filter_cutoff_hz" in npz
                else None
            )

        self.n_motion_frames = self.ref_global_translation.shape[0]
        # self.filter_cutoff_hz = self._normalize_filter_cutoff_hz(
        #     raw_filter_cutoff_hz, self.n_motion_frames
        # )
        self.filter_cutoff_hz = 1.0
        self._ref_to_sim_q_wxyz = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self._ref_to_sim_t = np.zeros(3, dtype=np.float32)
        self._ref_to_sim_ready = True

    def reset_state_teleport(self):
        self.counter = 0
        self.motion_frame_idx = 0

        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)

        has_ref_motion = (
            self.ref_dof_pos is not None
            and self.ref_dof_vel is not None
            and self.ref_global_translation is not None
            and self.ref_global_rotation_quat_xyzw is not None
            and self.ref_global_velocity is not None
            and self.ref_global_angular_velocity is not None
        )

        if has_ref_motion:
            root_pos = self.ref_global_translation[0, 0]  # (x, y, z)
            root_rot = self.ref_global_rotation_quat_xyzw[0, 0]  # XYZW
            root_vel = self.ref_global_velocity[0, 0]
            root_ang = self.ref_global_angular_velocity[0, 0]
            dof_pos = getattr(
                self, "stored_full_ref_dof_pos", self.ref_dof_pos
            )[0]
            dof_vel = getattr(
                self, "stored_full_ref_dof_vel", self.ref_dof_vel
            )[0]

            self.d.qpos[0:3] = root_pos
            self.d.qpos[3:7] = [
                root_rot[3],
                root_rot[0],
                root_rot[1],
                root_rot[2],
            ]  # XYZW -> WXYZ
            self.d.qpos[self.actuator_qpos_indices] = dof_pos[self.mu_to_ref]

            self.d.qvel[0:3] = root_vel
            self.d.qvel[3:6] = root_ang
            self.d.qvel[self.actuator_qvel_indices] = dof_vel[self.mu_to_ref]
            self.target_dof_pos_mu = dof_pos[self.mu_to_ref].astype(np.float32)
            logger.info(
                "Teleport reset initialized from reference frame 0 "
                "(root + dof pos/vel)"
            )
        else:
            self.d.qpos[self.actuator_qpos_indices] = self.default_angles_mu
            self.d.qvel[self.actuator_qvel_indices] = 0.0
            self.target_dof_pos_mu = self.default_angles_mu.astype(np.float32)
            logger.info(
                "Teleport reset initialized from ONNX default joint positions"
            )

        self.target_dof_pos_by_name = {
            self.mjcf_dof_names[i]: float(self.target_dof_pos_mu[i])
            for i in range(self.m.nu)
        }
        mujoco.mj_forward(self.m, self.d)

        if self.use_kv_cache and self.policy_kv_shape:
            shape = [
                d if isinstance(d, int) else 1 for d in self.policy_kv_shape
            ]
            self.policy_kv_cache = np.zeros(shape, dtype=np.float32)

        self._robot_dof_pos_seq = []
        self._robot_dof_vel_seq = []
        self._robot_dof_acc_seq = []
        self._robot_dof_torque_seq = []
        self._robot_low_level_dof_torque_seq = []
        self._robot_low_level_foot_contact_seq = []
        self._robot_low_level_foot_normal_force_seq = []
        self._robot_low_level_foot_tangent_speed_seq = []
        self._robot_actions_seq = []
        self._robot_action_rate_seq = []
        self._prev_recorded_dof_vel_ref = None
        self._prev_actions_onnx = None
        self._reset_action_ema_filter()
        self._reset_action_delay_randomization()
        self._prev_low_level_foot_geom_centers = None
        self._robot_global_translation_seq = []
        self._robot_global_rotation_quat_seq = []
        self._robot_global_velocity_seq = []
        self._robot_global_angular_velocity_seq = []
        self._robot_moe_expert_indices_seq = []
        self._robot_moe_expert_logits_seq = []
        self._reset_onnx_io_dump_buffers()

    def save_batch_result(self, output_path, meta_info):
        import json

        metadata = dict(meta_info)
        metadata.setdefault(
            "robot_low_level_torque_dt",
            float(getattr(self, "simulation_dt", 1.0 / 200.0)),
        )
        metadata.setdefault(
            "robot_low_level_contact_dt",
            float(getattr(self, "simulation_dt", 1.0 / 200.0)),
        )
        robot_moe_expert_indices, robot_moe_expert_logits = (
            self._get_stacked_moe_routing_tensors()
        )
        (
            robot_low_level_foot_contact,
            robot_low_level_foot_normal_force,
            robot_low_level_foot_tangent_speed,
        ) = self._get_stacked_low_level_foot_contact_tensors()

        res = {
            "robot_dof_pos": np.stack(self._robot_dof_pos_seq),
            "robot_dof_vel": np.stack(self._robot_dof_vel_seq),
            "robot_dof_acc": np.stack(self._robot_dof_acc_seq),
            "robot_dof_torque": np.stack(self._robot_dof_torque_seq),
            "robot_low_level_dof_torque": np.stack(
                self._robot_low_level_dof_torque_seq
            ),
            "robot_low_level_foot_contact": robot_low_level_foot_contact,
            "robot_low_level_foot_normal_force": (
                robot_low_level_foot_normal_force
            ),
            "robot_low_level_foot_tangent_speed": (
                robot_low_level_foot_tangent_speed
            ),
            "robot_low_level_torque_dt": np.array(
                getattr(self, "simulation_dt", 1.0 / 200.0), dtype=np.float32
            ),
            "robot_low_level_contact_dt": np.array(
                getattr(self, "simulation_dt", 1.0 / 200.0), dtype=np.float32
            ),
            "robot_action_rate": np.asarray(
                self._robot_action_rate_seq, dtype=np.float32
            ),
            "robot_global_translation": np.stack(
                self._robot_global_translation_seq
            ),
            "robot_global_rotation_quat": np.stack(
                self._robot_global_rotation_quat_seq
            ),
            "robot_global_velocity": np.stack(self._robot_global_velocity_seq),
            "robot_global_angular_velocity": np.stack(
                self._robot_global_angular_velocity_seq
            ),
            "ref_dof_pos": self.ref_dof_pos,
            "ref_dof_vel": self.ref_dof_vel,
            "ref_global_translation": self.ref_global_translation,
            "ref_global_rotation_quat": self.ref_global_rotation_quat_xyzw,
            "ref_global_velocity": self.ref_global_velocity,
            "ref_global_angular_velocity": self.ref_global_angular_velocity,
            "metadata": json.dumps(metadata),
        }
        if len(getattr(self, "_robot_actions_seq", [])) > 0:
            res["robot_actions"] = np.stack(
                self._robot_actions_seq, axis=0
            ).astype(np.float32)
        if robot_moe_expert_indices is not None:
            res["robot_moe_expert_indices"] = robot_moe_expert_indices
        if robot_moe_expert_logits is not None:
            res["robot_moe_expert_logits"] = robot_moe_expert_logits
        np.savez_compressed(output_path, **res)

    def setup(self):
        """Set up the evaluator by loading all required components."""
        self.load_mujoco_model()
        self._init_low_level_foot_contact_logging()
        self._build_mjcf_dof_names()
        self.load_policy()
        self._apply_onnx_metadata()
        self._build_actuator_qpos_indices()
        self._build_dof_mappings()
        self._build_actuator_name_map()
        self._build_actuator_force_range_map()
        self._init_camera_config()
        self._init_obs_buffers()

        # Initialize keyboard handler for velocity tracking
        if self.command_mode == "velocity_tracking":
            self.keyboard_handler = VelocityKeyboardHandler(
                vx_increment=0.1,
                vy_increment=0.05,
                vyaw_increment=0.05,
                vx_limits=(-0.5, 1.0),
                vy_limits=(-0.3, 0.3),
                vyaw_limits=(-0.5, 0.5),
            )
            logger.info(
                "Velocity tracking mode enabled. Keyboard controls:\n"
                "  W/S: forward/backward velocity\n"
                "  A/D: left/right velocity\n"
                "  J/L: turn left/right\n"
                "  Space/X: reset all\n"
                "  Keep terminal window focused for keyboard input"
            )
        elif self.command_mode == "motion_tracking":
            m_path = self.config.get("motion_npz_path", "")
            if m_path and os.path.isfile(m_path):
                self.load_motion_data()

    def _create_eval_progress_bar(self, desc: str, max_steps: int):
        if self.ref_dof_pos is not None:
            return tqdm(total=self.n_motion_frames, desc=desc, unit="frame")
        if max_steps > 0:
            return tqdm(total=max_steps, desc=desc, unit="step")
        return None

    def _advance_eval_frame(self, max_steps: int) -> bool:
        if self.ref_dof_pos is not None:
            if self.motion_frame_idx >= (self.n_motion_frames - 1):
                return False
            self.motion_frame_idx += 1
            return True
        if max_steps > 0 and self.counter >= max_steps:
            return False
        return True

    def _run_eval_step(self, max_steps: int) -> bool:
        self._update_policy()
        self.counter += 1
        self._apply_control(sleep=True)
        if self._video_writer is not None:
            self._maybe_record_frame()
        return self._advance_eval_frame(max_steps)

    def _build_mjcf_dof_names(self):
        """Build MJCF joint name lists used for control/state indexing.

        - mjcf_dof_names: joint names corresponding to each actuator (actuator order)
        """
        names = []
        for i in range(self.m.nu):
            j_id = int(self.m.actuator_trnid[i][0])
            j_name = mujoco.mj_id2name(
                self.m, mujoco._enums.mjtObj.mjOBJ_JOINT, j_id
            )
            names.append(j_name)
        self.mjcf_dof_names = names

    def _build_actuator_qpos_indices(self):
        """Build mapping from actuator index to qpos/qvel indices."""
        self.actuator_qpos_indices = np.zeros(self.m.nu, dtype=np.int32)
        self.actuator_qvel_indices = np.zeros(self.m.nu, dtype=np.int32)
        for i in range(self.m.nu):
            j_id = int(self.m.actuator_trnid[i, 0])
            self.actuator_qpos_indices[i] = self.m.jnt_qposadr[j_id]
            self.actuator_qvel_indices[i] = self.m.jnt_dofadr[j_id]

    def _build_actuator_name_map(self):
        """Build mappings from actuator name to indices and MJCF DOF indices."""
        name_to_index = {}
        actuator_name_to_mu_idx = {}
        for i in range(self.m.nu):
            act_name = mujoco.mj_id2name(
                self.m, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            name_to_index[act_name] = i
            j_id = int(self.m.actuator_trnid[i][0])
            j_name = mujoco.mj_id2name(
                self.m, mujoco._enums.mjtObj.mjOBJ_JOINT, j_id
            )
            mu_idx = self.mjcf_dof_names.index(j_name)
            actuator_name_to_mu_idx[act_name] = mu_idx
        self.actuator_name_to_index = name_to_index
        self.actuator_name_to_mu_idx = actuator_name_to_mu_idx

    def _build_actuator_force_range_map(self):
        """Build mapping from actuator index to joint actuator force range from XML."""
        self.actuator_force_range = {}
        for i in range(self.m.nu):
            j_id = int(self.m.actuator_trnid[i][0])
            has_limit = False
            min_force = 0.0
            max_force = 0.0
            if j_id >= 0 and j_id < self.m.njnt:
                if self.m.jnt_actfrclimited[j_id]:
                    min_force = float(self.m.jnt_actfrcrange[j_id][0])
                    max_force = float(self.m.jnt_actfrcrange[j_id][1])
                    if min_force != 0.0 or max_force != 0.0:
                        has_limit = True
            if not has_limit:
                if self.m.actuator_forcelimited[i]:
                    min_force = float(self.m.actuator_forcerange[i][0])
                    max_force = float(self.m.actuator_forcerange[i][1])
                    if min_force != 0.0 or max_force != 0.0:
                        has_limit = True
            if has_limit:
                self.actuator_force_range[i] = (min_force, max_force)
            else:
                self.actuator_force_range[i] = None

    def run_simulation_unitree(self):
        """Run simulation using Unitree's official threading/viewer pattern."""
        # Defer heavy deps to runtime to keep default path light

        # Ensure thirdparty simulate_python is on sys.path for imports

        self.counter = 0
        self.motion_frame_idx = 0
        self.reset_state_teleport()
        max_steps = int(self.config.get("max_policy_steps", 0))

        viewer_dt = float(self.config.get("unitree_viewer_dt", 1.0 / 60.0))

        viewer = mujoco.viewer.launch_passive(self.m, self.d)

        # Configure viewer camera to use shared align / tracking settings
        self._configure_viewer_camera(viewer)

        # Start keyboard listener for velocity tracking
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.start_listener()

        # Optional recording in viewer mode
        if bool(self.config.get("record_video", False)):
            self._init_video_tools(tag="viewer")

        pbar = self._create_eval_progress_bar("GUI eval", max_steps)

        locker = threading.Lock()
        stop_event = threading.Event()

        def simulation_thread():
            while viewer.is_running() and not stop_event.is_set():
                with locker:
                    keep_running = self._run_eval_step(max_steps)
                    if pbar is not None:
                        pbar.update(1)
                if not keep_running:
                    stop_event.set()
                    viewer.close()

        def physics_viewer_thread():
            while viewer.is_running() and not stop_event.is_set():
                with locker:
                    # Update camera lookat to track robot root (with small offset for framing)
                    self._update_camera_lookat(viewer.cam)

                    # Draw reference global bodylink positions as blue spheres when available
                    self._draw_ref_body_spheres_to_scene(
                        viewer.user_scn, reset_ngeom=True
                    )

                    viewer.sync()
                time.sleep(viewer_dt)

        viewer_thread = Thread(target=physics_viewer_thread)
        sim_thread = Thread(target=simulation_thread)

        viewer_thread.start()
        sim_thread.start()

        # Block until viewer closes
        viewer_thread.join()
        sim_thread.join()

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Stop keyboard listener
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.stop_listener()

        # Teardown recording
        self._close_video_tools()

        # Dump robot-augmented motion npz if motion tracking is enabled
        self._dump_robot_augmented_npz()

    def run_simulation_unitree_headless(self):
        """Run simulation headless (no GUI) with optional video recording."""
        # Defer heavy deps to runtime to keep default path light

        # Initialize
        self.counter = 0
        self.motion_frame_idx = 0
        self.reset_state_teleport()
        max_steps = int(self.config.get("max_policy_steps", 0))

        # Start keyboard listener for velocity tracking (even in headless mode)
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.start_listener()

        # Optional recording in headless mode
        if bool(self.config.get("record_video", False)):
            self._init_video_tools(tag="headless")

        pbar = self._create_eval_progress_bar("Headless eval", max_steps)

        running = True
        while running:
            running = self._run_eval_step(max_steps)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Stop keyboard listener
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.stop_listener()

        self._close_video_tools()

        # Dump robot-augmented motion npz if motion tracking is enabled
        self._dump_robot_augmented_npz()

    def run_simulation(self):
        if bool(self.config.get("headless", False)):
            logger.info("Running MuJoCo sim2sim headless")
            self.run_simulation_unitree_headless()
        else:
            self.run_simulation_unitree()

    def _update_policy(self):
        # Record robot states once per policy step for offline NPZ dumping
        self._record_robot_states()

        latest_obs = self.obs_builder.build_policy_obs()
        policy_obs_np = latest_obs[None, :]
        input_feed = {}
        input_feed[self.policy_input_name] = policy_obs_np

        if self.use_kv_cache:
            if self.policy_kv_cache is None:
                shape = [
                    d if isinstance(d, int) else 1
                    for d in self.policy_kv_shape
                ]
                self.policy_kv_cache = np.zeros(shape, dtype=np.float32)
            # if (
            #     self.policy_effective_context_len > 0
            #     and self.counter > 0
            #     and self.counter % self.policy_effective_context_len == 0
            # ):
            #     self.policy_kv_cache.fill(0.0)
            input_feed[self.policy_kv_input_name] = self.policy_kv_cache

        if self.policy_step_input_name is not None:
            step_idx = self.counter
            if self.use_kv_cache and self.policy_effective_context_len > 0:
                step_idx = self.counter % self.policy_effective_context_len
            step_tensor = np.array([step_idx], dtype=np.int64)
            input_feed[self.policy_step_input_name] = step_tensor

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output_names = [self.policy_output_name]
        if self.use_kv_cache and self.policy_kv_output_name:
            output_names.append(self.policy_kv_output_name)
        for _, indices_name, logits_name in self.policy_moe_layer_output_names:
            output_names.extend([indices_name, logits_name])

        onnx_output = self.policy_session.run(output_names, input_feed)
        if self.dump_onnx_io_npy:
            self._record_onnx_io_frame(input_feed, output_names, onnx_output)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        raw_actions_onnx = onnx_output[0].reshape(-1)
        filtered_actions_onnx = self._apply_action_ema_filter(raw_actions_onnx)
        self.actions_onnx = self._apply_action_delay(filtered_actions_onnx)

        if self.use_kv_cache and len(onnx_output) > 1:
            new_cache = onnx_output[1]

            self.policy_kv_cache = new_cache
        output_offset = 1 + int(
            bool(self.use_kv_cache and self.policy_kv_output_name)
        )
        if self.policy_moe_layer_output_names:
            step_indices = []
            step_logits = []
            for (
                _layer_idx,
                _indices_name,
                _logits_name,
            ) in self.policy_moe_layer_output_names:
                step_indices.append(
                    self._flatten_single_step_output(
                        onnx_output[output_offset],
                        dtype=np.int64,
                    )
                )
                output_offset += 1
                step_logits.append(
                    self._flatten_single_step_output(
                        onnx_output[output_offset],
                        dtype=np.float32,
                    )
                )
                output_offset += 1
            self._robot_moe_expert_indices_seq.append(
                np.stack(step_indices, axis=0)
            )
            self._robot_moe_expert_logits_seq.append(
                np.stack(step_logits, axis=0)
            )

        self.target_dof_pos_onnx = (
            self.actions_onnx * self.action_scale_onnx
            + self.default_angles_onnx
        )
        self.target_dof_pos_mu = self.target_dof_pos_onnx[self.onnx_to_mu]
        for i, dof_name in enumerate(self.mjcf_dof_names):
            self.target_dof_pos_by_name[dof_name] = float(
                self.target_dof_pos_mu[i]
            )

        if (
            self.command_mode == "motion_tracking"
            and self.ref_dof_pos is not None
            and len(self._robot_action_rate_seq) < len(self._robot_dof_pos_seq)
        ):
            self._robot_actions_seq.append(
                self.actions_onnx.astype(np.float32).copy()
            )
            if self._prev_actions_onnx is None:
                action_rate = np.float32(0.0)
            else:
                action_rate = np.float32(
                    np.linalg.norm(self.actions_onnx - self._prev_actions_onnx)
                    / self.policy_dt
                )
            self._prev_actions_onnx = self.actions_onnx.copy()
            self._robot_action_rate_seq.append(action_rate)
            self._robot_dof_torque_seq.append(
                self._compute_pd_torque_command_ref()
            )


def _get_config_value(config_obj, key: str):
    value = config_obj.get(key, None)
    if value is None and config_obj.get("eval", None) is not None:
        value = config_obj.eval.get(key, None)
    return value


def _normalize_ckpt_name_list(ckpt_onnx_names):
    if ckpt_onnx_names is None:
        return []
    if isinstance(ckpt_onnx_names, ListConfig):
        raw_names = list(ckpt_onnx_names)
    elif isinstance(ckpt_onnx_names, (list, tuple)):
        raw_names = list(ckpt_onnx_names)
    else:
        raise TypeError(
            "ckpt_onnx_names must be a list/tuple, "
            f"got {type(ckpt_onnx_names)}"
        )
    normalized_names = []
    for name in raw_names:
        name_str = str(name).strip()
        if name_str != "":
            normalized_names.append(name_str)
    return normalized_names


def _resolve_multi_ckpt_paths(ckpt_onnx_root_dir, ckpt_onnx_names):
    root_dir_str = str(ckpt_onnx_root_dir).strip()
    if root_dir_str == "":
        raise ValueError("ckpt_onnx_root_dir cannot be empty")
    root_dir = Path(root_dir_str)
    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"ckpt_onnx_root_dir does not exist or is not a directory: {root_dir}"
        )

    requested_names = _normalize_ckpt_name_list(ckpt_onnx_names)
    if len(requested_names) == 0:
        raise ValueError(
            "ckpt_onnx_names is empty. Please provide checkpoint names "
            'like ["model_1000.onnx", "model_2000.onnx"].'
        )

    discovered_paths = sorted(root_dir.rglob("*.onnx"))
    if len(discovered_paths) == 0:
        raise FileNotFoundError(
            f"No .onnx files found under ckpt_onnx_root_dir={root_dir}"
        )

    paths_by_name = {}
    for path in discovered_paths:
        if path.name not in paths_by_name:
            paths_by_name[path.name] = []
        paths_by_name[path.name].append(path)

    selected_paths = []
    missing_names = []
    for name in requested_names:
        candidates = paths_by_name.get(name, [])
        if len(candidates) == 0:
            missing_names.append(name)
            continue
        if len(candidates) > 1:
            logger.warning(
                f"Found {len(candidates)} ONNX files named '{name}' under "
                f"{root_dir}; selecting the first one: {candidates[0]}"
            )
        selected_paths.append(candidates[0])

    if len(missing_names) > 0:
        logger.warning(
            "Some requested checkpoints were not found under "
            f"{root_dir}: {missing_names}"
        )
    if len(selected_paths) == 0:
        raise FileNotFoundError(
            "None of the requested checkpoints were found under "
            f"{root_dir}. Requested names: {requested_names}"
        )

    return selected_paths


def _resolve_eval_ckpt_paths(config_obj):
    ckpt_onnx_root_dir = _get_config_value(config_obj, "ckpt_onnx_root_dir")
    if (
        ckpt_onnx_root_dir is not None
        and str(ckpt_onnx_root_dir).strip() != ""
    ):
        ckpt_onnx_names = _get_config_value(config_obj, "ckpt_onnx_names")
        return _resolve_multi_ckpt_paths(ckpt_onnx_root_dir, ckpt_onnx_names)

    ckpt_onnx_path = _get_config_value(config_obj, "ckpt_onnx_path")
    if ckpt_onnx_path is None or str(ckpt_onnx_path).strip() == "":
        raise ValueError(
            "No ONNX checkpoint is provided. Set ckpt_onnx_path, or set "
            "ckpt_onnx_root_dir + ckpt_onnx_names."
        )
    ckpt_path = Path(str(ckpt_onnx_path))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ONNX checkpoint not found: {ckpt_path}")
    return [ckpt_path]


def _checkpoint_tag_from_path(ckpt_path: Path) -> str:
    match = re.search(r"model_(\d+)", ckpt_path.name)
    if match:
        return f"model_{match.group(1)}"
    return ckpt_path.stem


def _build_eval_output_dir(ckpt_path: Path, dataset_name: str) -> Path:
    ckpt_tag = _checkpoint_tag_from_path(ckpt_path)
    dir_name = f"mujoco_eval_output_{ckpt_tag}_{dataset_name}"
    return ckpt_path.parent.parent / dir_name


def _build_onnx_io_dump_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / ONNX_IO_DUMP_DIRNAME


def _build_onnx_io_dump_path(
    output_dir: str | Path, source_file: str | Path
) -> Path:
    source_stem = Path(source_file).stem
    return _build_onnx_io_dump_dir(output_dir) / f"{source_stem}_onnx_io.npy"


def _build_onnx_io_dump_readme_text() -> str:
    return """# ONNX I/O 导出说明

本目录用于保存 MuJoCo sim2sim 评测过程中导出的 ONNX 输入输出数据。

## 文件组织

- 每个动作片段会生成一个 `.npy` 文件，文件名形如 `<clip_name>_onnx_io.npy`
- 每个 `.npy` 文件对应一个原始的动作片段 `.npz`
- 当前只支持默认的 `holomotion` / `MujocoEvaluator` 批量目录评测模式（`motion_npz_dir`）

## 读取方式

`.npy` 文件内部保存的是一个 Python `dict`，读取时需要开启 `allow_pickle=True`：

```python
import numpy as np

npy_path = "onnx_io_npy/example_clip_onnx_io.npy"
payload = np.load(npy_path, allow_pickle=True).item()

print(payload.keys())
print(payload["input_names"])
print(payload["output_names"])
print(payload["inputs"]["obs"].shape)
print(payload["outputs"]["action"].shape)
```

## 数据字段

- `input_names`: ONNX 实际输入张量名称列表
- `output_names`: ONNX 实际输出张量名称列表
- `inputs`: 按输入张量名称组织的字典，数组第 0 维是帧索引
- `outputs`: 按输出张量名称组织的字典，数组第 0 维是帧索引
- `source_npz`: 原始动作片段文件名
- `onnx_model`: 导出这些张量时使用的 ONNX 模型路径

## 说明

单个 `.npy` 文件只能保存一个顶层对象，因此这里使用 pickled dict 来同时保存输入名称、输出名称以及逐帧堆叠后的 numpy 数组。
如果某次导出未产生有效 ONNX I/O 数据，`inputs` 和 `outputs` 可能为空字典，读取时请先检查键是否存在。
"""


def write_onnx_io_dump_readme(output_dir: str | Path) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir_path / "README.md"
    readme_path.write_text(_build_onnx_io_dump_readme_text(), encoding="utf-8")
    return readme_path


def _allocate_actor_counts(num_checkpoints: int, total_actors: int):
    if num_checkpoints <= 0:
        raise ValueError("num_checkpoints must be > 0")
    if total_actors <= 0:
        raise ValueError("total_actors must be > 0")
    base = total_actors // num_checkpoints
    rem = total_actors % num_checkpoints
    return [base + (1 if i < rem else 0) for i in range(num_checkpoints)]


def _infer_step_from_ckpt_name(ckpt_name: str):
    match = re.search(r"model_(\d+)", ckpt_name)
    if match:
        return int(match.group(1))
    fallback = re.search(r"(\d+)", ckpt_name)
    if fallback:
        return int(fallback.group(1))
    return None


def _read_total_macro_row(tsv_path: Path):
    if not tsv_path.is_file():
        return None
    with open(tsv_path, "r", encoding="utf-8", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            dataset_value = str(row.get("Dataset", "")).strip().lower()
            if "total" in dataset_value and "macro" in dataset_value:
                return row
    return None


def _write_total_macro_summary_table(
    eval_targets, job_log_dir: Path | None = None
):
    rows_by_parent = {}
    for target in eval_targets:
        output_dir_path = Path(target["output_dir"])
        ckpt_path = target["ckpt_path"]
        tsv_path = output_dir_path / "sub_dataset_macro_mean_metrics.tsv"
        total_row = _read_total_macro_row(tsv_path)
        if total_row is None:
            logger.warning(
                "Skipping aggregated total metrics entry because "
                f"Total (Macro) row is unavailable: {tsv_path}"
            )
            continue
        parent_dir = output_dir_path.parent
        if parent_dir not in rows_by_parent:
            rows_by_parent[parent_dir] = []
        rows_by_parent[parent_dir].append(
            {
                "step": _infer_step_from_ckpt_name(ckpt_path.stem),
                "total_row": total_row,
                "ckpt_name": ckpt_path.stem,
            }
        )

    for parent_dir, entries in rows_by_parent.items():
        if len(entries) == 0:
            continue
        entries.sort(
            key=lambda item: (
                item["step"] is None,
                item["step"] if item["step"] is not None else 0,
                item["ckpt_name"],
            )
        )
        metric_columns = list(entries[0]["total_row"].keys())
        available_steps = [
            entry["step"] for entry in entries if entry["step"] is not None
        ]
        if len(available_steps) > 0:
            step_range = f"{min(available_steps)}-{max(available_steps)}"
        else:
            step_range = "na-na"
        output_name = f"mujoco_model-{step_range}_total_metrics.tsv"
        output_path = parent_dir / output_name
        generated_artifacts = [output_path]
        with open(output_path, "w", encoding="utf-8", newline="") as out_file:
            writer = csv.writer(out_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["step"] + metric_columns)
            for entry in entries:
                step_value = (
                    str(entry["step"]) if entry["step"] is not None else ""
                )
                writer.writerow(
                    [step_value]
                    + [
                        entry["total_row"].get(col, "")
                        for col in metric_columns
                    ]
                )
        logger.info(f"Saved aggregated total metrics table at: {output_path}")

        plot_metric_columns = [
            col for col in metric_columns if col != "Dataset"
        ]
        if len(plot_metric_columns) > 0:
            import matplotlib.pyplot as plt

            ncols = 4
            nrows = (len(plot_metric_columns) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(4.0 * ncols, 2.8 * nrows),
                squeeze=False,
            )

            for idx, metric_name in enumerate(plot_metric_columns):
                ax = axes[idx // ncols][idx % ncols]
                trend_pairs = []
                for entry in entries:
                    step_value = entry["step"]
                    if step_value is None:
                        continue
                    raw_metric = entry["total_row"].get(metric_name, "")
                    if str(raw_metric).strip() == "":
                        continue
                    trend_pairs.append((step_value, float(raw_metric)))

                if len(trend_pairs) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(metric_name)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    continue

                trend_pairs.sort(key=lambda pair: pair[0])
                plot_steps = [pair[0] for pair in trend_pairs]
                plot_values = [pair[1] for pair in trend_pairs]
                ax.plot(plot_steps, plot_values, marker="o", linewidth=1.2)
                ax.set_title(metric_name)
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)

            total_axes = nrows * ncols
            for idx in range(len(plot_metric_columns), total_axes):
                axes[idx // ncols][idx % ncols].axis("off")

            fig.tight_layout()
            plot_path = output_path.with_name(
                f"{output_path.stem}_all_metric_trends.pdf"
            )
            fig.savefig(plot_path, format="pdf")
            plt.close(fig)
            generated_artifacts.append(plot_path)
            logger.info(f"Saved combined metric trend plot at: {plot_path}")

        if job_log_dir is not None:
            for artifact_path in generated_artifacts:
                job_log_path = job_log_dir / artifact_path.name
                shutil.copy2(artifact_path, job_log_path)
                logger.info(f"Exported artifact to /job_log: {job_log_path}")


def process_config(override_config):
    """Process the configuration, merging with training config if available."""
    ckpt_onnx_path = _get_config_value(override_config, "ckpt_onnx_path")
    ckpt_onnx_root_dir = _get_config_value(
        override_config, "ckpt_onnx_root_dir"
    )
    if (
        (ckpt_onnx_path is None or str(ckpt_onnx_path).strip() == "")
        and ckpt_onnx_root_dir is not None
        and str(ckpt_onnx_root_dir).strip() != ""
    ):
        ckpt_onnx_names = _get_config_value(override_config, "ckpt_onnx_names")
        resolved_paths = _resolve_multi_ckpt_paths(
            ckpt_onnx_root_dir, ckpt_onnx_names
        )
        ckpt_onnx_path = str(resolved_paths[0])
        logger.info(
            "Using the first resolved checkpoint as config anchor: "
            f"{ckpt_onnx_path}"
        )

    model_type = override_config.get("model_type") or "holomotion"
    if model_type == "gmt":
        config_path = Path(
            "holomotion/config/evaluation/gmt_eval_mujoco_sim2sim.yaml"
        )
    elif model_type == "any2track":
        config_path = Path(
            "holomotion/config/evaluation/any2track_eval_mujoco_sim2sim.json"
        )
    elif model_type == "sonic":
        config_path = Path(
            "holomotion/config/evaluation/sonic_eval_mujoco_sim2sim.yaml"
        )
    else:
        if ckpt_onnx_path is None or str(ckpt_onnx_path).strip() == "":
            raise ValueError(
                "Cannot locate training config.yaml for model_type='holomotion' "
                "without an ONNX checkpoint path. Set ckpt_onnx_path, or set "
                "ckpt_onnx_root_dir + ckpt_onnx_names."
            )
        onnx_path = Path(str(ckpt_onnx_path))
        # Load training config.yaml from one level above the ONNX path (../onnx_path)
        config_path = onnx_path.parent.parent / "config.yaml"
    logger.info(f"Loading training config file from {config_path}")

    # Ensure ${eval:'...'} expressions are supported during resolution
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))

    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Merge training config with any overrides
    config = OmegaConf.merge(train_config, override_config)
    with open_dict(config):
        config.model_type = model_type

    # Resolve config values in-place
    OmegaConf.resolve(config)
    if (
        (
            config.get("ckpt_onnx_path", None) is None
            or str(config.get("ckpt_onnx_path")).strip() == ""
        )
        and ckpt_onnx_path is not None
        and str(ckpt_onnx_path).strip() != ""
    ):
        with open_dict(config):
            config.ckpt_onnx_path = str(ckpt_onnx_path)
    return config


def _create_ray_evaluator(config_dict, model_type):
    """Create evaluator from serializable config dict (used inside Ray actor)."""
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.create(config_dict)
    if model_type == "gmt":
        from holomotion.src.evaluation.gmt_sim2sim import GMTEvaluator

        return GMTEvaluator(config)
    if model_type == "any2track":
        from holomotion.src.evaluation.any2track_sim2sim import (
            Any2TrackEvaluator,
        )

        return Any2TrackEvaluator(config)
    if model_type == "sonic":
        from holomotion.src.evaluation.sonic_mujoco_sim2sim import (
            SonicEvaluator,
        )

        return SonicEvaluator(config)
    return MujocoEvaluator(config)


def run_mujoco_sim2sim_eval(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())
    config = process_config(override_config)
    is_eval_mode = False
    dataset_dir = config.get("motion_npz_dir", None)
    specific_file = config.get("motion_npz_path", None)
    calc_per_clip_metrics = bool(config.get("calc_per_clip_metrics", False))
    generate_report = bool(config.get("generate_report", False))
    dump_npzs_cfg = bool(config.get("dump_npzs", False))
    dump_onnx_io_npy = bool(config.get("dump_onnx_io_npy", False))
    dump_npzs = dump_npzs_cfg or calc_per_clip_metrics
    if calc_per_clip_metrics and not dump_npzs_cfg:
        logger.info(
            "calc_per_clip_metrics=true requires dumped NPZs; "
            "enabling dump_npzs automatically."
        )

    if (
        dataset_dir
        and os.path.isdir(str(dataset_dir))
        and (not specific_file or str(specific_file) == "")
    ):
        is_eval_mode = True

    if is_eval_mode:
        logger.info(f"Mode: EVALUATION on directory: {dataset_dir}")
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        dataset_name = Path(dataset_dir).name
        ckpt_paths = _resolve_eval_ckpt_paths(config)
        logger.info(
            f"Resolved {len(ckpt_paths)} checkpoint(s) for evaluation."
        )
        for idx, ckpt_path in enumerate(ckpt_paths):
            logger.info(f"  [{idx}] {ckpt_path}")

        eval_targets = []
        for ckpt_path in ckpt_paths:
            output_dir = _build_eval_output_dir(ckpt_path, dataset_name)
            eval_targets.append(
                {
                    "ckpt_path": ckpt_path,
                    "output_dir": str(output_dir),
                }
            )

        if dump_npzs:
            for target in eval_targets:
                os.makedirs(target["output_dir"], exist_ok=True)
                if dump_onnx_io_npy:
                    write_onnx_io_dump_readme(
                        _build_onnx_io_dump_dir(target["output_dir"])
                    )

            files = sorted(
                [
                    os.path.join(root, name)
                    for root, _, filenames in os.walk(
                        dataset_dir, followlinks=True
                    )
                    for name in filenames
                    if name.endswith(".npz")
                ]
            )
            logger.info(
                f"Found {len(files)} files for dataset_dir={dataset_dir}. "
                f"Will evaluate {len(eval_targets)} checkpoint(s)."
            )

            if len(files) == 0:
                logger.warning(
                    f"No NPZ files found under dataset_dir={dataset_dir}"
                )

            requested_use_gpu = _coerce_config_bool(
                config.get("use_gpu", True), default=True
            )
            num_available_gpus = 0
            if requested_use_gpu and torch.cuda.is_available():
                num_available_gpus = int(torch.cuda.device_count())
            if requested_use_gpu and num_available_gpus == 0:
                logger.warning(
                    "use_gpu=true but no CUDA device is detected; "
                    "Ray actors will run on CPU."
                )
            if num_available_gpus > 0:
                logger.info(
                    f"Detected {num_available_gpus} CUDA device(s). "
                    "Using Ray for batch evaluation."
                )

            ray_actors_per_gpu = int(config.get("ray_actors_per_gpu", 4))
            if ray_actors_per_gpu <= 0:
                raise ValueError("ray_actors_per_gpu must be > 0")
            ray_multi_ckpt_mode = str(
                config.get("ray_multi_ckpt_mode", "split")
            )
            if ray_multi_ckpt_mode not in ("split", "per_checkpoint"):
                raise ValueError(
                    "ray_multi_ckpt_mode must be one of: "
                    "'split', 'per_checkpoint'"
                )

            success_count = 0
            total_jobs = len(files) * len(eval_targets)
            if total_jobs > 0:
                base_config_dict = OmegaConf.to_container(config, resolve=True)
                base_config_dict.setdefault(
                    "ray_evaluator_module",
                    "holomotion.src.evaluation.eval_mujoco_sim2sim",
                )
                if not ray.is_initialized():
                    ray.init()
                from holomotion.src.evaluation.ray_evaluator_actor import (
                    RayEvaluatorActor,
                )

                if num_available_gpus > 0:
                    base_actor_count = num_available_gpus * ray_actors_per_gpu
                    gpus_per_actor = 1.0 / ray_actors_per_gpu
                    remote_actor = ray.remote(num_gpus=gpus_per_actor)(
                        RayEvaluatorActor
                    )
                else:
                    base_actor_count = max(1, ray_actors_per_gpu)
                    gpus_per_actor = 0.0
                    remote_actor = ray.remote(num_gpus=0)(RayEvaluatorActor)

                if ray_multi_ckpt_mode == "per_checkpoint":
                    actor_counts = [
                        base_actor_count for _ in range(len(eval_targets))
                    ]
                else:
                    actor_counts = _allocate_actor_counts(
                        len(eval_targets), base_actor_count
                    )
                    if min(actor_counts) <= 0:
                        raise ValueError(
                            "Not enough actor budget to assign at least one actor "
                            "per checkpoint in split mode. Reduce checkpoint count, "
                            "increase ray_actors_per_gpu, or switch to "
                            "ray_multi_ckpt_mode=per_checkpoint."
                        )

                total_actor_count = sum(actor_counts)
                logger.info(
                    f"Ray: {total_actor_count} persistent actors "
                    f"({ray_actors_per_gpu} per GPU, {gpus_per_actor} GPU each)"
                )
                logger.info(
                    "Checkpoint actor allocation: "
                    + ", ".join(
                        [
                            f"{target['ckpt_path'].name}={actor_counts[idx]}"
                            for idx, target in enumerate(eval_targets)
                        ]
                    )
                )

                refs = []
                for target_idx, target in enumerate(eval_targets):
                    target_config_dict = dict(base_config_dict)
                    target_config_dict["ckpt_onnx_path"] = str(
                        target["ckpt_path"]
                    )
                    num_actors = actor_counts[target_idx]
                    target_actors = [
                        remote_actor.remote(
                            target_config_dict, target["output_dir"]
                        )
                        for _ in range(num_actors)
                    ]
                    for file_idx, file_path in enumerate(files):
                        actor = target_actors[file_idx % len(target_actors)]
                        refs.append(actor.run_clip.remote(file_path))
                pbar = tqdm(
                    total=total_jobs,
                    desc="Batch Processing (all checkpoints)",
                    unit="job",
                    dynamic_ncols=True,
                )
                while refs:
                    done, refs = ray.wait(refs, num_returns=1)
                    for ref in done:
                        status = ray.get(ref)
                        if status == "success":
                            success_count += 1
                        pbar.update(1)
                pbar.close()
            logger.info(
                f"Batch processing done. Success: {success_count}/{total_jobs}"
            )
        else:
            logger.info("Skipping NPZ dumping because dump_npzs=false.")

        job_log_dir = Path("/job_log")
        job_log_enabled = job_log_dir.is_dir() and os.access(
            str(job_log_dir), os.W_OK
        )
        if job_log_enabled:
            logger.info(
                f"Detected writable /job_log. Will copy summary TSVs to {job_log_dir}."
            )
        else:
            logger.info(
                "/job_log is unavailable or not writable. "
                "Skipping summary TSV export."
            )

        postprocess_targets = []
        for target in eval_targets:
            output_dir = target["output_dir"]
            output_dir_path = Path(output_dir)
            if not output_dir_path.is_dir():
                logger.warning(
                    f"Output directory does not exist, skipping post-processing: {output_dir}"
                )
                continue
            postprocess_targets.append(target)

        failure_pos_err_thresh_m = float(
            config.get("failure_pos_err_thresh_m", 0.25)
        )
        metric_calculation = str(config.get("metric_calculation", "per_clip"))
        dof_mode = str(config.get("dof_mode", "29"))

        ray_parallel_metrics = bool(
            config.get(
                "ray_parallel_metrics_postprocess",
                config.get("ray_parallel_metrics", True),
            )
        )
        metrics_threadpool_max_workers = config.get(
            "metrics_threadpool_max_workers", None
        )
        should_parallelize_metrics = (
            ray_parallel_metrics
            and len(postprocess_targets) > 1
            and (calc_per_clip_metrics or generate_report or job_log_enabled)
        )
        logger.info(
            "Metrics config: "
            f"ray_parallel_metrics_postprocess={ray_parallel_metrics}, "
            f"metrics_threadpool_max_workers={metrics_threadpool_max_workers}"
        )

        if should_parallelize_metrics:
            if not ray.is_initialized():
                ray.init()
            from holomotion.src.evaluation.ray_metrics_postprocess import (
                run_metrics_postprocess_job,
            )

            ray_metrics_num_cpus_cfg = config.get(
                "ray_metrics_postprocess_num_cpus",
                config.get("ray_metrics_num_cpus", None),
            )
            if ray_metrics_num_cpus_cfg is None:
                ray_metrics_num_cpus = 0.0
            else:
                ray_metrics_num_cpus = float(ray_metrics_num_cpus_cfg)
            if ray_metrics_num_cpus < 0.0:
                raise ValueError("ray_metrics_num_cpus must be >= 0")

            metric_refs = []
            for target in postprocess_targets:
                ckpt_path = target["ckpt_path"]
                metric_refs.append(
                    run_metrics_postprocess_job.options(
                        num_cpus=ray_metrics_num_cpus
                    ).remote(
                        output_dir=target["output_dir"],
                        dataset_name=dataset_name,
                        calc_per_clip_metrics=calc_per_clip_metrics,
                        failure_pos_err_thresh_m=failure_pos_err_thresh_m,
                        metric_calculation=metric_calculation,
                        dof_mode=dof_mode,
                        metrics_threadpool_max_workers=metrics_threadpool_max_workers,
                        generate_report=generate_report,
                        job_log_dir=str(job_log_dir)
                        if job_log_enabled
                        else None,
                        ckpt_stem=ckpt_path.stem,
                    )
                )

            pbar = tqdm(
                total=len(metric_refs),
                desc="Metrics post-processing (all checkpoints)",
                unit="ckpt",
                dynamic_ncols=True,
            )
            while metric_refs:
                done, metric_refs = ray.wait(metric_refs, num_returns=1)
                for ref in done:
                    result = ray.get(ref)
                    ckpt_stem = str(result.get("ckpt_stem", "")).strip()
                    if ckpt_stem == "":
                        ckpt_stem = "unknown"
                    if calc_per_clip_metrics:
                        logger.info(
                            f"Metric calculation finished for {ckpt_stem}."
                        )
                    report_path = str(result.get("report_path", "")).strip()
                    if report_path != "":
                        logger.info(
                            f"Generated metrics report for {ckpt_stem} at: {report_path}"
                        )
                    exported_tsv = str(
                        result.get("exported_summary_tsv", "")
                    ).strip()
                    if exported_tsv != "":
                        logger.info(f"Exported summary TSV to: {exported_tsv}")
                pbar.update(1)
            pbar.close()
        else:
            mean_process_5metrics = None
            if generate_report:
                from holomotion.scripts.evaluation import mean_process_5metrics

            for target in postprocess_targets:
                output_dir = target["output_dir"]
                output_dir_path = Path(output_dir)
                ckpt_path = target["ckpt_path"]

                if calc_per_clip_metrics:
                    logger.info(
                        "Starting metric calculation for "
                        f"{ckpt_path.name}: {output_dir}"
                    )
                    run_evaluation(
                        npz_dir=output_dir,
                        dataset_suffix=dataset_name,
                        failure_pos_err_thresh_m=failure_pos_err_thresh_m,
                        metric_calculation=metric_calculation,
                        dof_mode=dof_mode,
                        threadpool_max_workers=metrics_threadpool_max_workers,
                    )
                    logger.info(
                        f"Metric calculation finished for {ckpt_path.name}."
                    )

                if generate_report:
                    report_path = mean_process_5metrics.generate_macro_mean_report_from_json_dir(
                        output_dir
                    )
                    logger.info(
                        f"Generated metrics report for {ckpt_path.name} at: {report_path}"
                    )

                if job_log_enabled:
                    sub_dataset_tsv = (
                        output_dir_path / "sub_dataset_macro_mean_metrics.tsv"
                    )
                    if sub_dataset_tsv.is_file():
                        export_name = f"{ckpt_path.stem}_sub_dataset_macro_mean_metrics.tsv"
                        export_path = job_log_dir / export_name
                        shutil.copy2(sub_dataset_tsv, export_path)
                        logger.info(f"Exported summary TSV to: {export_path}")
                    else:
                        logger.warning(
                            "Summary TSV not found (skip export): "
                            f"{sub_dataset_tsv}"
                        )

        _write_total_macro_summary_table(
            eval_targets,
            job_log_dir=job_log_dir if job_log_enabled else None,
        )

    else:
        if config.get("model_type", "holomotion") == "sonic":
            from holomotion.src.evaluation.sonic_mujoco_sim2sim import (
                SonicEvaluator,
            )

            evaluator = SonicEvaluator(config)
        else:
            evaluator = MujocoEvaluator(config)
        evaluator.setup()
        evaluator.run_simulation()


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_mujoco_sim2sim",
    version_base=None,
)
def main(override_config: OmegaConf):
    run_mujoco_sim2sim_eval(override_config)


if __name__ == "__main__":
    main()
