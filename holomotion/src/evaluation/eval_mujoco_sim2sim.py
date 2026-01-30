import os
import sys
import threading
import time
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
from omegaconf import OmegaConf
from tqdm import tqdm


try:
    import pynput.keyboard as pynput_kb

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning(
        "pynput not available, keyboard control will not work for velocity tracking"
    )

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
        logger.info(f"Command mode: {self.command_mode}")

        # Motion data
        self.ref_dof_pos = None
        self.ref_dof_vel = None
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
        self._robot_global_translation_seq: list[np.ndarray] = []
        self._robot_global_rotation_quat_seq: list[np.ndarray] = []
        self._robot_global_velocity_seq: list[np.ndarray] = []
        self._robot_global_angular_velocity_seq: list[np.ndarray] = []
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

    def _find_actor_place_holder_ndim(self):
        n_dim = 0
        for obs_dict in self.config.obs.obs_groups.policy.atomic_obs_list:
            if list(obs_dict.keys())[0] == "place_holder":
                n_dim = obs_dict["place_holder"].params.n_dim
        return n_dim

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
        """Detect command mode from config."""
        motion_npz_path = self.config.get("motion_npz_path", None)
        if motion_npz_path is None or motion_npz_path == "":
            return "velocity_tracking"
        return "motion_tracking"

    def _init_obs_buffers(self):
        # Shared observation builder (history + flattening)
        self.obs_builder = PolicyObsBuilder(
            dof_names_onnx=self.dof_names_onnx,
            default_angles_onnx=self.default_angles_onnx,
            evaluator=self,
            obs_policy_cfg=(
                self.config.get("obs", {})
                .get("obs_groups", {})
                .get("policy", {})
            ),
        )

    def load_policy(self):
        """Load the policy model using ONNX Runtime."""
        onnx_model_path = Path(self.config.ckpt_onnx_path)

        logger.info(f"Loading ONNX policy from {onnx_model_path}")

        # ONNX Runtime providers: prefer GPU for inference, fallback to CPU
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                },
            ),
            "CPUExecutionProvider",
        ]

        self.policy_session = onnxruntime.InferenceSession(
            str(onnx_model_path), providers=providers
        )
        logger.info(
            f"ONNX Runtime session created successfully using: {self.policy_session.get_providers()}"
        )
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_output_name = self.policy_session.get_outputs()[0].name
        logger.info(
            f"Policy  ONNX Input: {self.policy_input_name}, Output: {self.policy_output_name}"
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

        # Load original motion npz
        with np.load(motion_npz_path, allow_pickle=True) as npz:
            data_dict = {k: npz[k] for k in npz.files}

        # Augment with robot_* arrays (override if already present)
        data_dict["robot_dof_pos"] = robot_dof_pos
        data_dict["robot_dof_vel"] = robot_dof_vel
        data_dict["robot_global_translation"] = robot_global_translation
        data_dict["robot_global_rotation_quat"] = robot_global_rotation_quat
        data_dict["robot_global_velocity"] = robot_global_velocity
        data_dict["robot_global_angular_velocity"] = (
            robot_global_angular_velocity
        )

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

            mujoco.mj_step(self.m, self.d)
            if sleep:
                time.sleep(self.simulation_dt)

    def _get_obs_ref_motion_states(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
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
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
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

    def _get_obs_ref_root_height_cur(self):
        if getattr(self, "ref_global_translation", None) is None:
            return 0.0
        return self.ref_global_translation[
            self.motion_frame_idx, self.root_body_idx, 2
        ]

    def _get_obs_place_holder(self):
        return np.zeros(self.actor_place_holder_ndim, dtype=np.float32)

    def _get_obs_vr_ref_motion_states(self):
        # [2 * num_actions] in ONNX order: [ref_pos, ref_vel]
        if self.ref_dof_pos is None or self.ref_dof_vel is None:
            return np.zeros(2 * self.num_actions, dtype=np.float32)
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
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
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
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

    def setup(self):
        """Set up the evaluator by loading all required components."""
        self.load_policy()
        self.load_mujoco_model()
        self._apply_onnx_metadata()
        self._build_mjcf_dof_names()
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
            self.load_motion_data()

    def _set_initial_state_default_pos(self):
        """Set the robot's initial state to the default dof pos from ONNX."""
        ref_dof_pos_mu = self.default_angles_mu

        # Joint states (excluding free root)
        self.d.qpos[self.actuator_qpos_indices] = ref_dof_pos_mu

        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

        self.target_dof_pos_mu = ref_dof_pos_mu
        self.target_dof_pos_by_name = {
            self.mjcf_dof_names[i]: float(ref_dof_pos_mu[i])
            for i in range(self.m.nu)
        }
        logger.info(
            "Initial robot state set to match default dof pos from ONNX"
        )

    def _set_initial_state_ref(self):
        """Set the robot's initial state to match the first frame of motion npz."""
        if self.ref_dof_pos is not None:
            ref_dof_pos_mu = self.ref_dof_pos[0]
            # ref_dof_vel_mu = self.ref_dof_vel[0]

            # Joint states (excluding free root)
            self.d.qpos[self.actuator_qpos_indices] = ref_dof_pos_mu
            # self.d.qvel[self.actuator_qvel_indices] = ref_dof_vel_mu

            # Align free root to IsaacLab reference root if global arrays are available
            if (
                getattr(self, "ref_global_translation", None) is not None
                and getattr(self, "ref_global_rotation_quat_xyzw", None)
                is not None
            ):
                # Assume dataset body index 0 corresponds to the root ("pelvis")
                root_pos = self.ref_global_translation[
                    0, self.root_body_idx
                ].astype(np.float32)
                root_q_xyzw = self.ref_global_rotation_quat_xyzw[
                    0, self.root_body_idx
                ].astype(np.float32)
                root_q_wxyz = (
                    xyzw_to_wxyz(torch.tensor(root_q_xyzw)).cpu().numpy()
                )

                self.d.qpos[0:3] = root_pos
                self.d.qpos[3:7] = root_q_wxyz

                # Initialize root velocities from reference motion
                # if getattr(self, "ref_global_velocity", None) is not None:
                #     root_lin_vel = self.ref_global_velocity[
                #         0, self.root_body_idx
                #     ].astype(np.float32)
                #     self.d.qvel[0:3] = root_lin_vel

                # if getattr(self, "ref_global_angular_velocity", None) is not None:
                #     root_ang_vel = self.ref_global_angular_velocity[
                #         0, self.root_body_idx
                #     ].astype(np.float32)
                #     self.d.qvel[3:6] = root_ang_vel

            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)

            self.target_dof_pos_mu = ref_dof_pos_mu[self.onnx_to_mu]
            self.target_dof_pos_by_name = {
                self.mjcf_dof_names[i]: float(ref_dof_pos_mu[i])
                for i in range(self.m.nu)
            }
            logger.info(
                "Initial robot state set to match first frame of motion npz"
            )
        else:
            logger.warning("No motion data available to set initial state")

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
        # self._set_initial_state()

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

        # Progress bar for frame tracking (only for motion tracking mode)
        pbar = None
        if self.ref_dof_pos is not None:
            pbar = tqdm(
                total=self.n_motion_frames,
                desc="GUI eval",
                unit="frame",
                position=0,
                leave=True,
            )

        locker = threading.Lock()

        def simulation_thread():
            prev_frame_idx = -1
            self._set_initial_state_default_pos()
            while viewer.is_running():
                locker.acquire()
                self._update_policy()
                self.counter += 1
                if self.ref_dof_pos is not None:
                    self.motion_frame_idx = min(
                        self.motion_frame_idx + 1,
                        self.n_motion_frames - 1,
                    )
                    # Update progress bar if frame index changed
                    if (
                        pbar is not None
                        and self.motion_frame_idx != prev_frame_idx
                    ):
                        update_amount = self.motion_frame_idx - prev_frame_idx
                        pbar.update(update_amount)
                        prev_frame_idx = self.motion_frame_idx
                self._apply_control(sleep=True)
                locker.release()

        def physics_viewer_thread():
            while viewer.is_running():
                locker.acquire()

                # Update camera lookat to track robot root (with small offset for framing)
                self._update_camera_lookat(viewer.cam)

                # Draw reference global bodylink positions as blue spheres when available
                self._draw_ref_body_spheres_to_scene(
                    viewer.user_scn, reset_ngeom=True
                )

                viewer.sync()
                # Capture frame at configured FPS if recording
                if self._video_writer is not None:
                    self._maybe_record_frame()
                locker.release()
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
        self._set_initial_state()

        # Start keyboard listener for velocity tracking (even in headless mode)
        if (
            self.command_mode == "velocity_tracking"
            and self.keyboard_handler is not None
        ):
            self.keyboard_handler.start_listener()

        # Optional recording in headless mode
        if bool(self.config.get("record_video", False)):
            self._init_video_tools(tag="headless")

        # Main headless loop: run until motion frames consumed or max steps reached
        max_steps = int(self.config.get("max_policy_steps", 0))
        steps_done = 0

        # Progress bar setup
        pbar = None
        if self.ref_dof_pos is not None:
            pbar = tqdm(
                total=self.n_motion_frames, desc="Headless eval", unit="frame"
            )
        elif max_steps > 0:
            pbar = tqdm(total=max_steps, desc="Headless eval", unit="step")

        running = True
        while running:
            # Build and run policy for current frame/step
            self._update_policy()
            self.counter += 1

            self._apply_control(sleep=True)

            # Step counters and recording
            steps_done += 1
            if pbar is not None:
                pbar.update(1)
            if self._video_writer is not None:
                self._maybe_record_frame()

            # Advance or stop
            if self.ref_dof_pos is not None:
                # Stop after processing the last frame
                if self.motion_frame_idx >= (self.n_motion_frames - 1):
                    running = False
                else:
                    self.motion_frame_idx += 1
            else:
                # Stop by step budget if no motion
                if max_steps > 0 and steps_done >= max_steps:
                    running = False

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
        input_feed = {self.policy_input_name: policy_obs_np}
        onnx_output = self.policy_session.run(
            [self.policy_output_name], input_feed
        )
        self.actions_onnx = onnx_output[0].reshape(-1)
        self.target_dof_pos_onnx = (
            self.actions_onnx * self.action_scale_onnx
            + self.default_angles_onnx
        )
        self.target_dof_pos_mu = self.target_dof_pos_onnx[self.onnx_to_mu]
        for i, dof_name in enumerate(self.mjcf_dof_names):
            self.target_dof_pos_by_name[dof_name] = float(
                self.target_dof_pos_mu[i]
            )


def process_config(override_config):
    """Process the configuration, merging with training config if available."""
    # Get ONNX path from top-level or nested under eval
    ckpt_onnx_path = override_config.get("ckpt_onnx_path", None)
    if (
        ckpt_onnx_path is None
        and override_config.get("eval", None) is not None
    ):
        ckpt_onnx_path = override_config.eval.get("ckpt_onnx_path", None)
    onnx_path = Path(ckpt_onnx_path)

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

    # Resolve config values in-place
    OmegaConf.resolve(config)
    return config


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_mujoco_sim2sim",
)
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())
    config = process_config(override_config)
    evaluator = MujocoEvaluator(config)
    evaluator.setup()
    evaluator.run_simulation()


if __name__ == "__main__":
    main()
