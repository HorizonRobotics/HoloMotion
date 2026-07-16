#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal Newton/HoloRetarget adapter for robot reference poses.

The adapter exposes the minimal retarget result used by downstream consumers:

    Pico body_poses(24, 7) -> robot qpos(36)

The upstream target config supports the XRoboToolkit/PICO body format through its ``xrobot``
IK config and SMPL-family body format through its ``smplx`` IK config. This
file adapts data layout and output packing around HoloRetarget's selected IK
configuration.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import pickle
import sys
import time
import types
from typing import Any, Optional

import numpy as np

from .schema import DOF_POS_DIM, QPOS_DIM


_FILE_DIR = Path(__file__).resolve().parent
_HOLOMOTION_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _HOLOMOTION_ROOT.parent
_SMPL_NEUTRAL_MODEL_PATH = (
    _HOLOMOTION_ROOT
    / "thirdparties"
    / "smpl_models"
    / "SMPL_python_v.1.1.0"
    / "smpl"
    / "models"
    / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)


def _first_existing_path(*candidates: Path, required: str | None = None) -> Path:
    for candidate in candidates:
        if required is not None:
            if (candidate / required).exists():
                return candidate
            continue
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_RETARGET_ASSET_ROOT = _first_existing_path(
    _HOLOMOTION_ROOT / "holoretarget" / "assets",
    required="target_configs/smplx_to_g1.json",
)
XROBOT_BODY_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
SMPLX_BODY_NAMES = tuple(name.lower() for name in XROBOT_BODY_NAMES)

_UNITY_TO_RETARGET_ROT = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_UNITY_TO_RETARGET_ROT32 = _UNITY_TO_RETARGET_ROT.astype(np.float32)
_TWO_PI = 2.0 * np.pi
_TWO_PI32 = np.float32(2.0 * np.pi)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_name_from_label(label: object) -> str:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    return str(label).split("/")[-1]

def _insert_python_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"required path does not exist: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), eps, None)


def _normalize_quat_wxyz32(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True).astype(np.float32, copy=False)
    return q / np.clip(norm, np.float32(eps), None)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_mul_wxyz32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert one 3x3 matrix to a scalar-first quaternion."""
    r = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(r))
    q = np.empty(4, dtype=np.float64)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (r[2, 1] - r[1, 2]) / s
        q[2] = (r[0, 2] - r[2, 0]) / s
        q[3] = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q[0] = (r[2, 1] - r[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (r[0, 1] + r[1, 0]) / s
        q[3] = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q[0] = (r[0, 2] - r[2, 0]) / s
        q[1] = (r[0, 1] + r[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q[0] = (r[1, 0] - r[0, 1]) / s
        q[1] = (r[0, 2] + r[2, 0]) / s
        q[2] = (r[1, 2] + r[2, 1]) / s
        q[3] = 0.25 * s
    return _normalize_quat_wxyz(q)


def _enable_chumpy_compat() -> None:
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    numpy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import chumpy  # noqa: F401
    except Exception:
        chumpy_mod = types.ModuleType("chumpy")
        ch_mod = types.ModuleType("chumpy.ch")

        class Ch:
            def __setstate__(self, state: object) -> None:
                self.__dict__["_pickle_state"] = state

            def __getstate__(self) -> object:
                return self.__dict__.get("_pickle_state", {})

        ch_mod.Ch = Ch
        chumpy_mod.ch = ch_mod
        sys.modules.setdefault("chumpy", chumpy_mod)
        sys.modules.setdefault("chumpy.ch", ch_mod)


def _resolve_smpl_model_file(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"SMPL model file not found: {path}")
        return path

    env_path = os.environ.get("HOLORETARGET_SMPL_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                "HOLORETARGET_SMPL_MODEL_PATH points to a missing "
                f"SMPL model file: {path}"
            )
        return path

    if _SMPL_NEUTRAL_MODEL_PATH.exists():
        return _SMPL_NEUTRAL_MODEL_PATH
    raise FileNotFoundError(
        "licensed SMPL neutral model is missing; expected "
        f"{_SMPL_NEUTRAL_MODEL_PATH}. Download SMPL_python_v.1.1.0.zip "
        "from the official SMPL site and extract it under "
        "thirdparties/smpl_models, or set HOLORETARGET_SMPL_MODEL_PATH."
    )


_UNITY_TO_RETARGET_QUAT_WXYZ = _matrix_to_quat_wxyz(_UNITY_TO_RETARGET_ROT)
_UNITY_TO_RETARGET_QUAT_WXYZ32 = _UNITY_TO_RETARGET_QUAT_WXYZ.astype(np.float32)


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(q)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., 1:4]
    qw = q[..., 0:1]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_rotate_wxyz32(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz32(q)
    v = np.asarray(v, dtype=np.float32)
    qv = q[..., 1:4]
    qw = q[..., 0:1]
    t = np.float32(2.0) * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_yaw_xyzw(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    half = np.float32(0.5 * float(yaw))
    return np.asarray([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def pico_body_poses_to_xrobot_arrays(body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_poses = np.asarray(body_poses, dtype=np.float64)
    if body_poses.shape != (24, 7):
        raise ValueError(f"body_poses shape must be (24,7), got {body_poses.shape}")

    positions = body_poses[:, :3] @ _UNITY_TO_RETARGET_ROT.T
    quats_wxyz = np.empty((24, 4), dtype=np.float64)
    quats_wxyz[:, 0] = body_poses[:, 6]
    quats_wxyz[:, 1:4] = body_poses[:, 3:6]
    quats_wxyz = _normalize_quat_wxyz(quats_wxyz)
    quats_wxyz = _normalize_quat_wxyz(_quat_mul_wxyz(_UNITY_TO_RETARGET_QUAT_WXYZ, quats_wxyz))
    return positions, quats_wxyz


def pico_body_poses_to_xrobot_arrays32(body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_poses = np.asarray(body_poses, dtype=np.float32)
    if body_poses.shape != (24, 7):
        raise ValueError(f"body_poses shape must be (24,7), got {body_poses.shape}")

    positions = body_poses[:, :3] @ _UNITY_TO_RETARGET_ROT32.T
    quats_wxyz = np.empty((24, 4), dtype=np.float32)
    quats_wxyz[:, 0] = body_poses[:, 6]
    quats_wxyz[:, 1:4] = body_poses[:, 3:6]
    quats_wxyz = _normalize_quat_wxyz32(quats_wxyz)
    quats_wxyz = _normalize_quat_wxyz32(_quat_mul_wxyz32(_UNITY_TO_RETARGET_QUAT_WXYZ32, quats_wxyz))
    return positions, quats_wxyz


def pico_body_poses_to_xrobot(body_poses: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Convert raw XRoboToolkit Pico body poses to HoloRetarget's ``xrobot`` dict.

    Input quaternions are ``xyzw`` in Unity coordinates, matching the current
    teleop ``PicoReader`` and the saved ``pico_raw`` CSV files. HoloRetarget's xrobot
    path expects scalar-first ``wxyz`` quaternions after the same Unity-to-right-
    handed coordinate transform used in ``XRobotStreamer``.
    """
    positions, quats_wxyz = pico_body_poses_to_xrobot_arrays(body_poses)

    return {
        name: [positions[i].copy(), quats_wxyz[i].copy()]
        for i, name in enumerate(XROBOT_BODY_NAMES)
    }


class SmplSkeletonAdapter:
    """Load the SMPL rest skeleton shared by Pico and offline input adapters."""

    def __init__(self, smpl_model_path: str | os.PathLike[str] | None = None) -> None:
        _enable_chumpy_compat()
        model_file = _resolve_smpl_model_file(smpl_model_path)
        self.smpl_model_path = model_file
        with model_file.open("rb") as f:
            model_data = pickle.load(f, encoding="latin1")
        j_regressor = model_data["J_regressor"]
        if hasattr(j_regressor, "toarray"):
            j_regressor = j_regressor.toarray()
        self.smpl_rest_joints = (
            np.asarray(j_regressor, dtype=np.float32)
            @ np.asarray(model_data["v_template"], dtype=np.float32).reshape(-1, 3)
        ).astype(np.float32)
        self.smpl_parents = np.asarray(model_data["kintree_table"][0, :24], dtype=np.int32)
        self.smpl_parents[0] = -1
        self.smpl_rest_parent_offsets = np.zeros_like(self.smpl_rest_joints, dtype=np.float32)
        for joint_id in range(1, 24):
            parent = int(self.smpl_parents[joint_id])
            self.smpl_rest_parent_offsets[joint_id] = (
                self.smpl_rest_joints[joint_id] - self.smpl_rest_joints[parent]
            )
class HoloNewtonG1Retargeter:
    """HoloRetarget target generation with Newton GPU IK.

    This intentionally keeps the upstream xrobot scaling and target offset
    semantics, but solves the G1 IK with Newton/Warp. It is an approximate
    compatibility backend because Newton's G1 asset has ankle bodies but no toe
    bodies, so upstream toe targets are mapped to ankle roll links.
    """

    _BODY_REMAP = {
        "left_toe_link": "left_ankle_roll_link",
        "right_toe_link": "right_ankle_roll_link",
    }
    _BODY_LINK_OFFSETS = {
        "left_toe_link": np.array([0.1, 0.0, -0.02], dtype=np.float64),
        "right_toe_link": np.array([0.1, 0.0, -0.02], dtype=np.float64),
    }

    def __init__(
        self,
        asset_root: str | os.PathLike[str] = DEFAULT_RETARGET_ASSET_ROOT,
        src_human: str = "xrobot",
        robot: str = "unitree_g1",
        actual_human_height: Optional[float] = None,
        ik_iterations: int = 1,
        use_cuda_graph: bool = True,
        joint_limit_weight: float = 10.0,
        max_joint_step: float = 0.1,
        target_table: str = "ik_match_table2",
        robot_asset: str = "newton_builtin",
        offset_to_ground: bool = False,
        ground_calibration_frames: int = 0,
        ground_height: float = 0.0,
        ground_lift_only: bool = True,
        ground_calibration_mode: str = "initial_mean",
        ground_target_scope: str = "feet",
        root_seed_mode: str = "off",
        profile_timing: bool = False,
    ) -> None:
        if src_human not in {"xrobot", "smplx"} or robot != "unitree_g1":
            raise ValueError("Newton backend currently supports only xrobot/smplx -> unitree_g1")

        self.asset_root = _insert_python_path(asset_root or DEFAULT_RETARGET_ASSET_ROOT)
        self.src_human = src_human
        self.robot = robot
        self.ik_iterations = max(1, int(ik_iterations))
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_target_copy = _env_bool("HOLORETARGET_GRAPH_TARGET_COPY", False)
        self.native_direct_targets = _env_bool("HOLORETARGET_NATIVE_DIRECT_TARGETS", False)
        self.fused_direct_targets = _env_bool("HOLORETARGET_FUSED_DIRECT_TARGETS", True)
        self.pinned_qpos_output = _env_bool("HOLORETARGET_PINNED_QPOS_OUTPUT", True)
        self.cpu_float32 = _env_bool("HOLORETARGET_CPU_FLOAT32", False)
        self.prealloc_targets = _env_bool("HOLORETARGET_PREALLOC_TARGETS", False)
        self.robot_asset = str(robot_asset)
        if self.robot_asset not in {"newton_builtin", "holoretarget_mjcf"}:
            raise ValueError("robot_asset must be 'newton_builtin' or 'holoretarget_mjcf'")
        self.max_joint_step = max(0.0, float(max_joint_step))
        self.offset_to_ground = bool(offset_to_ground)
        self.ground_calibration_frames = max(0, int(ground_calibration_frames))
        self.ground_height = float(ground_height)
        self.ground_lift_only = bool(ground_lift_only)
        self.ground_calibration_mode = str(ground_calibration_mode or "initial_mean").lower()
        if self.ground_calibration_mode not in {"initial_mean", "sliding_min"}:
            raise ValueError("ground_calibration_mode must be 'initial_mean' or 'sliding_min'")
        self.ground_target_scope = str(ground_target_scope or "feet").lower()
        if self.ground_target_scope not in {"feet", "all"}:
            raise ValueError("ground_target_scope must be 'feet' or 'all'")
        self.root_seed_mode = str(root_seed_mode or "off").lower()
        if self.root_seed_mode not in {"off", "pelvis", "pelvis_pos", "pelvis_yaw"}:
            raise ValueError(f"unknown HoloRetarget root_seed_mode: {root_seed_mode}")
        self._ground_min_z_samples: list[float] = []
        self._ground_offset_z: float | None = None
        self.last_ground_offset_z = 0.0
        self.profile_timing = bool(profile_timing)
        self.last_timing: dict[str, float] = {}
        self.human_body_names = (
            XROBOT_BODY_NAMES if src_human == "xrobot" else SMPLX_BODY_NAMES
        )

        config_path = self.asset_root / "target_configs" / f"{src_human}_to_g1.json"
        if robot != "unitree_g1" or not config_path.exists():
            raise FileNotFoundError(
                f"Newton config not found for {src_human}->{robot}: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as f:
            self.ik_config: dict[str, Any] = json.load(f)
        if target_table.startswith("ik_match_table2_"):
            variant_tokens = target_table[len("ik_match_table2_"):].split("_")
            table2 = self.ik_config["ik_match_table2"]
            merged_table: dict[str, Any] = {}
            for robot_body, entry in table2.items():
                human_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
                body_name = str(robot_body).lower()
                is_foot = "toe" in body_name
                is_upper = any(part in body_name for part in ("shoulder", "elbow", "wrist"))
                is_lower = any(part in body_name for part in ("hip", "knee"))
                is_torso = "torso" in body_name
                is_pelvis = "pelvis" in body_name
                pos_weight = float(pos_weight)
                rot_weight = float(rot_weight)
                for token in variant_tokens:
                    if token.startswith("footrot"):
                        if is_foot:
                            raw_weight = token[len("footrot"):]
                            rot_weight = float(raw_weight) if raw_weight else rot_weight
                    elif token.startswith("foot"):
                        if is_foot:
                            raw_weight = token[len("foot"):]
                            pos_weight = float(raw_weight) if raw_weight else 200.0
                    elif token.startswith("rot"):
                        if not is_foot:
                            raw_weight = token[len("rot"):]
                            rot_weight = float(raw_weight) if raw_weight else rot_weight
                    elif token.startswith("upper"):
                        if is_upper:
                            raw_weight = token[len("upper"):]
                            pos_weight = float(raw_weight) if raw_weight else pos_weight
                    elif token.startswith("lower"):
                        if is_lower:
                            raw_weight = token[len("lower"):]
                            pos_weight = float(raw_weight) if raw_weight else pos_weight
                    elif token.startswith("limb"):
                        if is_upper or is_lower:
                            raw_weight = token[len("limb"):]
                            pos_weight = float(raw_weight) if raw_weight else pos_weight
                    elif token.startswith("torso"):
                        if is_torso:
                            raw_weight = token[len("torso"):]
                            pos_weight = float(raw_weight) if raw_weight else pos_weight
                    elif token.startswith("pelvis"):
                        if is_pelvis:
                            raw_weight = token[len("pelvis"):]
                            pos_weight = float(raw_weight) if raw_weight else pos_weight
                    elif token == "noarms":
                        if is_upper:
                            pos_weight = 0.0
                            rot_weight = 0.0
                    elif token == "nolower":
                        if is_lower:
                            pos_weight = 0.0
                            rot_weight = 0.0
                    elif token == "notorso":
                        if is_torso:
                            pos_weight = 0.0
                            rot_weight = 0.0
                    elif token == "nofootrot":
                        if is_foot:
                            rot_weight = 0.0
                    else:
                        raise ValueError(f"unknown HoloRetarget target table variant token: {token!r}")
                merged_table[robot_body] = [
                    human_name,
                    pos_weight,
                    rot_weight,
                    pos_offset,
                    rot_offset,
                ]
            self.ik_config[target_table] = merged_table
        if target_table in {
            "ik_match_table1_lower",
            "ik_match_table1_feet_torso",
            "ik_match_table1_feet_pelvis",
        }:
            table1 = self.ik_config["ik_match_table1"]
            if target_table == "ik_match_table1_lower":
                keep = {
                    "pelvis",
                    "left_hip_yaw_link",
                    "left_knee_link",
                    "left_toe_link",
                    "right_hip_yaw_link",
                    "right_knee_link",
                    "right_toe_link",
                    "torso_link",
                }
            elif target_table == "ik_match_table1_feet_torso":
                keep = {"pelvis", "left_toe_link", "right_toe_link", "torso_link"}
            else:
                keep = {"pelvis", "left_toe_link", "right_toe_link"}
            self.ik_config[target_table] = {
                robot_body: entry for robot_body, entry in table1.items() if robot_body in keep
            }
        if target_table not in self.ik_config:
            raise ValueError(f"target_table must be one of the HoloRetarget IK tables, got {target_table!r}")
        self.target_table = target_table

        ratio = 1.0
        if actual_human_height is not None:
            ratio = float(actual_human_height) / float(self.ik_config["human_height_assumption"])
        self.human_root_name = str(self.ik_config["human_root_name"])
        self.human_scale_table = {
            str(k): float(v) * ratio
            for k, v in self.ik_config["human_scale_table"].items()
        }
        self.ground = float(self.ik_config["ground_height"]) * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ground_offset = 0.0

        table = self.ik_config[target_table]
        self.target_defs: list[dict[str, Any]] = []
        for robot_body, entry in table.items():
            human_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if float(pos_weight) == 0.0 and float(rot_weight) == 0.0:
                continue
            self.target_defs.append(
                {
                    "robot_body": (
                        self._BODY_REMAP.get(str(robot_body), str(robot_body))
                        if self.robot_asset == "newton_builtin"
                        else str(robot_body)
                    ),
                    "holoretarget_robot_body": str(robot_body),
                    "link_offset": self._BODY_LINK_OFFSETS.get(
                        str(robot_body),
                        np.zeros(3, dtype=np.float64),
                    ) if self.robot_asset == "newton_builtin" else np.zeros(3, dtype=np.float64),
                    "human_name": str(human_name),
                    "human_index": self.human_body_names.index(str(human_name)),
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                    "pos_offset": np.asarray(pos_offset, dtype=np.float64) - self.ground,
                    "rot_offset_wxyz": _normalize_quat_wxyz(np.asarray(rot_offset, dtype=np.float64)),
                    "scale": float(self.human_scale_table.get(str(human_name), 1.0)),
                }
            )
        self._target_human_indices = np.asarray(
            [target["human_index"] for target in self.target_defs],
            dtype=np.int64,
        )
        self._target_scales = np.asarray(
            [target["scale"] for target in self.target_defs],
            dtype=np.float64,
        )
        self._target_scales32 = self._target_scales.astype(np.float32)
        self._target_pos_offsets = np.asarray(
            [target["pos_offset"] for target in self.target_defs],
            dtype=np.float64,
        )
        self._target_pos_offsets32 = self._target_pos_offsets.astype(np.float32)
        self._target_rot_offsets_wxyz = _normalize_quat_wxyz(
            np.asarray([target["rot_offset_wxyz"] for target in self.target_defs], dtype=np.float64)
        )
        self._target_rot_offsets_wxyz32 = self._target_rot_offsets_wxyz.astype(np.float32)
        self._target_foot_mask = np.asarray(
            ["foot" in target["human_name"].lower() for target in self.target_defs],
            dtype=bool,
        )
        self._root_human_index = self.human_body_names.index(self.human_root_name)
        self._root_scale = float(self.human_scale_table[self.human_root_name])
        self._root_scale32 = np.float32(self._root_scale)
        self._target_root_pos_work = np.empty(3, dtype=np.float64)
        self._target_body_poses_work = np.empty((len(self.target_defs), 7), dtype=np.float64)
        self._target_pose_t_work = np.empty((len(self.target_defs), 3), dtype=np.float64)
        self._target_pose_q_work = np.empty((len(self.target_defs), 4), dtype=np.float64)
        self._target_scaled_pos_work = np.empty((len(self.target_defs), 3), dtype=np.float64)
        self._target_updated_quat_work = np.empty((len(self.target_defs), 4), dtype=np.float64)
        self._target_pos_work = np.empty((len(self.target_defs), 3), dtype=np.float64)
        self._target_norm_work = np.empty(len(self.target_defs), dtype=np.float64)
        self._target_rotate_t_work = np.empty((len(self.target_defs), 3), dtype=np.float64)
        self._targets_work = np.empty((len(self.target_defs), 7), dtype=np.float32)
        self._root_seed_target_index = next(
            (
                i
                for i, target in enumerate(self.target_defs)
                if str(target["human_name"]) == self.human_root_name
            ),
            None,
        )
        self._native_direct_targets_func = None
        self._native_direct_targets_error = ""
        self._load_native_direct_targets()

        try:
            import newton  # type: ignore
            import newton.ik as ik  # type: ignore
            import warp as wp  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError(
                "failed to import Newton/Warp dependencies for --holoretarget-ik-backend newton; "
                "use the Python 3.12 deployment runtime."
            ) from exc

        self.newton = newton
        self.ik = ik
        self.wp = wp

        builder = newton.ModelBuilder()
        if self.robot_asset == "holoretarget_mjcf":
            builder.add_mjcf(self.asset_root / "unitree_g1" / "g1_mocap_29dof.xml")
        else:
            builder.add_mjcf(newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")
        self._robot_builder = builder
        self._body_names = [_get_name_from_label(label) for label in builder.body_label]
        self._body_indices = []
        for target in self.target_defs:
            body = target["robot_body"]
            if body not in self._body_names:
                raise ValueError(f"Newton G1 asset does not contain mapped body {body!r}")
            self._body_indices.append(self._body_names.index(body))

        self.num_envs = 1
        self.model = self._build_model(self.num_envs)
        self.state = self.model.state()
        self.num_dof = int(self.model.joint_coord_count - 7)
        if self.num_dof != DOF_POS_DIM:
            raise ValueError(
                f"Newton G1 model must expose {DOF_POS_DIM} qpos DoF after root, "
                f"got {self.num_dof}"
            )
        limit_lower = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float64)
        limit_upper = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float64)
        # Newton exposes floating-base limits in joint-velocity coordinates
        # (6 base coordinates), while qpos uses a 7D free-joint pose.
        dof_limit_start = 6
        dof_limit_stop = dof_limit_start + self.num_dof
        if limit_lower.shape[0] < dof_limit_stop or limit_upper.shape[0] < dof_limit_stop:
            raise ValueError(
                "Newton joint limit arrays are shorter than expected: "
                f"lower={limit_lower.shape}, upper={limit_upper.shape}, num_dof={self.num_dof}"
            )
        self._dof_limit_lower = limit_lower[dof_limit_start:dof_limit_stop].astype(np.float32)
        self._dof_limit_upper = limit_upper[dof_limit_start:dof_limit_stop].astype(np.float32)
        finite_limit_mask = (
            np.isfinite(self._dof_limit_lower)
            & np.isfinite(self._dof_limit_upper)
            & (self._dof_limit_upper > self._dof_limit_lower)
            & ((self._dof_limit_upper - self._dof_limit_lower) < 1e6)
        )
        self._dof_limit_mask = finite_limit_mask
        self._dof_limit_mid = np.zeros(self.num_dof, dtype=np.float32)
        self._dof_limit_mid[finite_limit_mask] = (
            0.5 * (
                self._dof_limit_lower[finite_limit_mask]
                + self._dof_limit_upper[finite_limit_mask]
            )
        ).astype(np.float32)
        self._previous_projected_dof: np.ndarray | None = None
        self.last_projection_delta_max = 0.0

        (
            self.position_objectives,
            self.rotation_objectives,
            self._position_target_def_indices,
            self._rotation_target_def_indices,
        ) = self._create_target_objectives()
        objectives = [*self.position_objectives, *self.rotation_objectives]
        self.joint_limit_objective = None
        if float(joint_limit_weight) > 0.0:
            self.joint_limit_objective = ik.IKObjectiveJointLimit(
                joint_limit_lower=self.model.joint_limit_lower,
                joint_limit_upper=self.model.joint_limit_upper,
                weight=float(joint_limit_weight),
            )
            objectives.append(self.joint_limit_objective)

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=self.num_envs,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.joint_q = wp.empty(shape=(self.num_envs, self.model.joint_coord_count))
        wp.copy(self.joint_q, self.model.joint_q)
        self._qpos_upload_host_wp = wp.empty(
            shape=(self.num_envs, self.model.joint_coord_count),
            dtype=wp.float32,
            device="cpu",
            pinned=True,
        )
        self._qpos_upload_host = self._qpos_upload_host_wp.numpy()
        self._init_qpos_output_copy()
        self._init_target_batch_copy()
        self._init_root_seed_copy()
        self.ik_solver.reset()
        self.graph_capture = None
        if self.use_cuda_graph and self.joint_q.device.is_cuda:
            # Newton/Warp may lazily load specialized IK kernels on the first
            # solve. Loading CUDA modules during graph capture fails on Orin, so
            # force the first solve outside capture and reset to the initial
            # state before recording the steady-state graph.
            dummy_targets = np.zeros((len(self.target_defs), 7), dtype=np.float32)
            dummy_targets[:, 6] = 1.0
            self._set_targets(dummy_targets)
            self._single_step()
            self.wp.synchronize()
            self.ik_solver.reset()
            self.wp.copy(self.joint_q, self.model.joint_q)
            with wp.ScopedCapture() as cap:
                if self.graph_target_copy:
                    self._copy_target_host_to_device()
                self._single_step()
            self.graph_capture = cap.graph
        else:
            self._single_step()

        self.last_targets: np.ndarray | None = None
        self.last_output: np.ndarray | None = None

    def set_qpos_state(self, qpos: np.ndarray, update_projection_reference: bool = False) -> None:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.model.joint_coord_count:
            raise ValueError(
                f"qpos length mismatch: got {q.shape[0]}, expected {self.model.joint_coord_count}"
            )
        np.copyto(self._qpos_upload_host[0], q)
        self.wp.copy(self.joint_q, self._qpos_upload_host_wp)
        if update_projection_reference:
            if self._previous_projected_dof is None:
                self._previous_projected_dof = np.empty(self.num_dof, dtype=np.float32)
            np.copyto(self._previous_projected_dof, q[7:7 + self.num_dof])

    def copy_qpos_state_from(self, other: "HoloNewtonG1Retargeter") -> None:
        if int(other.model.joint_coord_count) != int(self.model.joint_coord_count):
            raise ValueError(
                "qpos length mismatch: "
                f"source={other.model.joint_coord_count}, target={self.model.joint_coord_count}"
            )
        self.wp.copy(self.joint_q, other.joint_q)

    def reset_sequence(self) -> None:
        self._ground_min_z_samples.clear()
        self._ground_offset_z = None
        self.last_ground_offset_z = 0.0
        self._previous_projected_dof = None
        self.ik_solver.reset()
        self.wp.copy(self.joint_q, self.model.joint_q)
        self.last_targets = None
        self.last_output = None
        self.last_timing = {}

    def _calibrated_ground_offset(self, min_z: float) -> float:
        if self.ground_calibration_frames <= 0:
            self.last_ground_offset_z = 0.0
            return 0.0

        if self.ground_calibration_mode == "sliding_min":
            self._ground_min_z_samples.append(float(min_z))
            if len(self._ground_min_z_samples) > self.ground_calibration_frames:
                del self._ground_min_z_samples[:-self.ground_calibration_frames]
            offset = self.ground_height - float(np.min(self._ground_min_z_samples))
            if self.ground_lift_only:
                offset = max(0.0, offset)
        else:
            if self._ground_offset_z is None:
                if len(self._ground_min_z_samples) < self.ground_calibration_frames:
                    self._ground_min_z_samples.append(float(min_z))
                offset = self.ground_height - float(np.mean(self._ground_min_z_samples))
                if self.ground_lift_only:
                    offset = max(0.0, offset)
                if len(self._ground_min_z_samples) >= self.ground_calibration_frames:
                    self._ground_offset_z = offset
            else:
                offset = self._ground_offset_z

        self.last_ground_offset_z = float(offset)
        return float(offset)

    def _apply_ground_calibration(self, targets: np.ndarray, copy_targets: bool = True) -> np.ndarray:
        if self.ground_calibration_frames <= 0:
            self.last_ground_offset_z = 0.0
            return targets
        if self.ground_target_scope == "all":
            min_z = float(np.min(targets[:, 2]))
        else:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
        offset = self._calibrated_ground_offset(min_z)
        if offset == 0.0:
            return targets
        out = np.asarray(targets, dtype=np.float32)
        if copy_targets:
            out = out.copy()
        out[:, 2] += np.float32(offset)
        return out

    def _build_model(self, num_envs: int):
        builder = self.newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self._robot_builder, xform=self.wp.transform_identity())
        builder.add_ground_plane()
        return builder.finalize(requires_grad=True)

    def _sync_if_profile(self) -> None:
        if not self.profile_timing:
            return
        try:
            self.wp.synchronize()
        except Exception:
            pass

    def _single_step(self) -> None:
        self.ik_solver.step(
            self.joint_q,
            self.joint_q,
            iterations=max(1, int(self.ik_iterations)),
        )

    def _project_dofs_to_continuous_limits(self, dof_pos: np.ndarray) -> np.ndarray:
        """Keep Newton hinge coordinates in a continuous joint-limit branch.

        Newton's joint-limit objective is soft, so a hinge can occasionally be
        solved as q +/- 2*pi. That orientation is equivalent for FK but it is
        not a useful policy/robot command. Project each revolute DoF to a valid
        joint-limit branch, choosing the equivalent angle nearest the previous
        projected frame when available.
        """
        raw = np.asarray(dof_pos, dtype=np.float32)
        projected = raw.copy()
        mask = self._dof_limit_mask
        if not np.any(mask):
            self.last_projection_delta_max = 0.0
            if self._previous_projected_dof is None:
                self._previous_projected_dof = np.empty_like(raw, dtype=np.float32)
            np.copyto(self._previous_projected_dof, raw)
            return projected

        previous = self._previous_projected_dof
        raw_masked = raw[mask]
        if np.all((raw_masked >= self._dof_limit_lower[mask]) & (raw_masked <= self._dof_limit_upper[mask])):
            if previous is None or self.max_joint_step <= 0.0 or np.all(np.abs(raw - previous) <= self.max_joint_step):
                self.last_projection_delta_max = 0.0
                if previous is None:
                    self._previous_projected_dof = np.empty_like(raw, dtype=np.float32)
                    previous = self._previous_projected_dof
                np.copyto(previous, raw)
                return raw

        reference = previous if previous is not None else self._dof_limit_mid
        if self.cpu_float32:
            q = raw[mask].astype(np.float32, copy=False)
            ref = reference[mask].astype(np.float32, copy=False)
            lower = self._dof_limit_lower[mask].astype(np.float32, copy=False)
            upper = self._dof_limit_upper[mask].astype(np.float32, copy=False)
            two_pi = _TWO_PI32
        else:
            q = raw[mask].astype(np.float64)
            ref = reference[mask].astype(np.float64)
            lower = self._dof_limit_lower[mask].astype(np.float64)
            upper = self._dof_limit_upper[mask].astype(np.float64)
            two_pi = _TWO_PI

        nearest_k = np.round((ref - q) / two_pi)
        min_k = np.ceil((lower - q) / two_pi)
        max_k = np.floor((upper - q) / two_pi)
        has_equivalent_in_limit = min_k <= max_k
        chosen_k = nearest_k.copy()
        chosen_k[has_equivalent_in_limit] = np.minimum(
            np.maximum(nearest_k[has_equivalent_in_limit], min_k[has_equivalent_in_limit]),
            max_k[has_equivalent_in_limit],
        )
        candidate = q + chosen_k * two_pi
        candidate = np.minimum(np.maximum(candidate, lower), upper)

        projected[mask] = candidate.astype(np.float32)
        if previous is not None and self.max_joint_step > 0.0:
            delta = np.clip(
                projected - previous,
                -self.max_joint_step,
                self.max_joint_step,
            )
            projected = previous + delta.astype(np.float32)
            projected[mask] = np.minimum(
                np.maximum(projected[mask], self._dof_limit_lower[mask]),
                self._dof_limit_upper[mask],
            )
        self.last_projection_delta_max = float(np.max(np.abs(projected - raw)))
        if previous is None:
            self._previous_projected_dof = np.empty_like(projected, dtype=np.float32)
            previous = self._previous_projected_dof
        np.copyto(previous, projected)
        return projected

    def _project_qpos_to_continuous_limits(self, qpos: np.ndarray) -> np.ndarray:
        projected_dof = self._project_dofs_to_continuous_limits(qpos[7:7 + self.num_dof])
        if self.last_projection_delta_max <= 1e-6:
            return qpos

        qpos = qpos.copy()
        qpos[7:7 + self.num_dof] = projected_dof
        projected_q = self.wp.array(
            qpos.reshape(1, -1),
            dtype=self.wp.float32,
            device=self.joint_q.device,
        )
        self.wp.copy(self.joint_q, projected_q)
        return qpos

    def _create_target_objectives(self):
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        body_q = self.state.body_q.numpy()
        pos_objectives = []
        rot_objectives = []
        pos_def_indices = []
        rot_def_indices = []
        for target_idx, target in enumerate(self.target_defs):
            link_idx = self._body_indices[target_idx]
            if target["pos_weight"] != 0.0:
                pos_wp = self.wp.array(
                    [body_q[link_idx][0:3]],
                    dtype=self.wp.vec3,
                )
                pos_objectives.append(
                    self.ik.IKObjectivePosition(
                        link_index=link_idx,
                        link_offset=self.wp.vec3(*np.asarray(target["link_offset"], dtype=np.float32)),
                        target_positions=pos_wp,
                        weight=target["pos_weight"],
                    )
                )
                pos_def_indices.append(target_idx)
            if target["rot_weight"] != 0.0:
                rot_wp = self.wp.array(
                    [body_q[link_idx][3:7]],
                    dtype=self.wp.vec4,
                )
                rot_objectives.append(
                    self.ik.IKObjectiveRotation(
                        link_index=link_idx,
                        link_offset_rotation=self.wp.quat_identity(),
                        target_rotations=rot_wp,
                        weight=target["rot_weight"],
                    )
                )
                rot_def_indices.append(target_idx)
        return pos_objectives, rot_objectives, pos_def_indices, rot_def_indices

    def _init_target_batch_copy(self) -> None:
        self._target_batch_copy_enabled = False
        self._target_positions_host_wp = self.wp.empty(
            shape=(len(self.position_objectives), 3),
            dtype=self.wp.float32,
            device="cpu",
            pinned=True,
        )
        self._target_rotations_host_wp = self.wp.empty(
            shape=(len(self.rotation_objectives), 4),
            dtype=self.wp.float32,
            device="cpu",
            pinned=True,
        )
        self._target_positions_host = self._target_positions_host_wp.numpy()
        self._target_rotations_host = self._target_rotations_host_wp.numpy()
        if not self.joint_q.device.is_cuda:
            return

        count = len(self.position_objectives) + len(self.rotation_objectives)
        void_p_array = ctypes.c_void_p * count
        size_array = ctypes.c_size_t * count
        dst_ptrs = [obj.target_positions.ptr for obj in self.position_objectives]
        dst_ptrs.extend(obj.target_rotations.ptr for obj in self.rotation_objectives)
        src_ptrs = [self._target_positions_host[i].ctypes.data for i in range(len(self.position_objectives))]
        src_ptrs.extend(self._target_rotations_host[i].ctypes.data for i in range(len(self.rotation_objectives)))
        sizes = [self._target_positions_host[i].nbytes for i in range(len(self.position_objectives))]
        sizes.extend(self._target_rotations_host[i].nbytes for i in range(len(self.rotation_objectives)))

        self._target_copy_dsts = void_p_array(*dst_ptrs)
        self._target_copy_srcs = void_p_array(*src_ptrs)
        self._target_copy_sizes = size_array(*sizes)
        self._target_copy_count = ctypes.c_size_t(count)
        self._target_copy_context = self.joint_q.device.context
        self._target_copy_core = self.wp._src.context.runtime.core
        self._target_batch_copy_enabled = True

    def _init_qpos_output_copy(self) -> None:
        self._qpos_host_wp = None
        self._qpos_host = None
        if not self.pinned_qpos_output or not self.joint_q.device.is_cuda:
            return
        self._qpos_host_wp = self.wp.empty(
            shape=(self.num_envs, self.model.joint_coord_count),
            dtype=self.wp.float32,
            device="cpu",
            pinned=True,
        )
        self._qpos_host = self._qpos_host_wp.numpy()

    def _init_root_seed_copy(self) -> None:
        self._root_seed_wp = None
        self._root_seed_host = None
        if (
            self.root_seed_mode == "off"
            or self._root_seed_target_index is None
            or not self.joint_q.device.is_cuda
        ):
            return
        self._root_seed_wp = self.wp.empty(
            shape=(7,),
            dtype=self.wp.float32,
            device="cpu",
            pinned=True,
        )
        self._root_seed_host = self._root_seed_wp.numpy()
        self._root_seed_host[:] = self.model.joint_q.numpy()[:7]

    def _seed_root_from_targets(self, targets_xyzw: np.ndarray) -> None:
        if (
            self.root_seed_mode == "off"
            or self._root_seed_target_index is None
            or self._root_seed_wp is None
            or self._root_seed_host is None
        ):
            return
        target = np.asarray(targets_xyzw[self._root_seed_target_index], dtype=np.float32)
        seed = self._root_seed_host
        seed[:3] = target[:3]
        if self.root_seed_mode == "pelvis_pos":
            self.wp.copy(self.joint_q, self._root_seed_wp, 0, 0, 3)
            return
        if self.root_seed_mode == "pelvis_yaw":
            seed[3:7] = _yaw_quat_xyzw(_quat_yaw_xyzw(target[3:7]))
        else:
            quat = target[3:7]
            seed[3:7] = quat / np.clip(np.linalg.norm(quat), 1e-8, None)
        self.wp.copy(self.joint_q, self._root_seed_wp, 0, 0, 7)

    def _read_qpos_host(self) -> np.ndarray:
        if self._qpos_host_wp is not None and self._qpos_host is not None:
            self.wp.copy(self._qpos_host_wp, self.joint_q)
            self.wp.synchronize()
            return self._qpos_host[0]
        return self.joint_q.numpy().astype(np.float32, copy=False)[0].copy()

    def _update_target_host(self, targets_xyzw: np.ndarray) -> bool:
        if not self._target_batch_copy_enabled:
            return False

        np.copyto(self._target_positions_host, targets_xyzw[self._position_target_def_indices, :3])
        np.copyto(self._target_rotations_host, targets_xyzw[self._rotation_target_def_indices, 3:7])
        return True

    def _copy_target_host_to_device(self) -> None:
        if not self._target_batch_copy_enabled:
            return
        stream = self.wp.get_stream(self.joint_q.device)
        ok = self._target_copy_core.wp_memcpy_batch(
            self._target_copy_context,
            self._target_copy_dsts,
            self._target_copy_srcs,
            self._target_copy_sizes,
            self._target_copy_count,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if not ok:
            raise RuntimeError("Warp target batch memcpy failed")

    def _set_targets(self, targets_xyzw: np.ndarray) -> None:
        if not self._update_target_host(targets_xyzw):
            for obj_idx, target_idx in enumerate(self._position_target_def_indices):
                self.position_objectives[obj_idx].set_target_position(
                    0,
                    self.wp.vec3(*targets_xyzw[target_idx, :3]),
                )
            for obj_idx, target_idx in enumerate(self._rotation_target_def_indices):
                self.rotation_objectives[obj_idx].set_target_rotation(
                    0,
                    self.wp.quat(*targets_xyzw[target_idx, 3:7]),
                )
            return
        self._copy_target_host_to_device()

    def _targets_from_pico_arrays_prealloc(self, positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
        root_pos = positions[self._root_human_index]
        pose_t = self._target_pose_t_work
        pose_q = self._target_pose_q_work
        np.take(positions, self._target_human_indices, axis=0, out=pose_t)
        np.take(quats_wxyz, self._target_human_indices, axis=0, out=pose_q)

        scaled_pos = self._target_scaled_pos_work
        np.subtract(pose_t, root_pos[None, :], out=scaled_pos)
        scaled_pos *= self._target_scales[:, None]
        scaled_pos += (self._root_scale * root_pos)[None, :]

        updated_quat = self._target_updated_quat_work
        a = pose_q
        b = self._target_rot_offsets_wxyz
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        updated_quat[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        updated_quat[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        updated_quat[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        updated_quat[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        updated_quat /= np.clip(np.linalg.norm(updated_quat, axis=1, keepdims=True), 1e-8, None)

        target_pos = self._target_pos_work
        target_pos[:] = scaled_pos + _quat_rotate_wxyz(updated_quat, self._target_pos_offsets)
        if self.ground_offset != 0.0:
            target_pos[:, 2] -= self.ground_offset

        targets = self._targets_work
        targets[:, :3] = target_pos
        targets[:, 3] = updated_quat[:, 1]
        targets[:, 4] = updated_quat[:, 2]
        targets[:, 5] = updated_quat[:, 3]
        targets[:, 6] = updated_quat[:, 0]
        targets = self._apply_ground_calibration(targets, copy_targets=False)

        if self.offset_to_ground:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
            targets[:, 2] += np.float32(0.1 - min_z)
        return targets

    def _load_native_direct_targets(self) -> None:
        if not self.native_direct_targets:
            return
        lib_path = Path(__file__).with_name("holoretarget_fast_targets.so")
        if not lib_path.exists():
            self._native_direct_targets_error = f"missing {lib_path}"
            return
        try:
            lib = ctypes.CDLL(str(lib_path))
            func = lib.holoretarget_direct_targets_double
            func.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_float),
            ]
            func.restype = ctypes.c_int
            self._native_direct_targets_func = func
        except Exception as exc:
            self._native_direct_targets_error = repr(exc)

    def _targets_from_pico_body_poses_direct_native(self, body_poses: np.ndarray) -> np.ndarray:
        body = np.asarray(body_poses, dtype=np.float64)
        if body.shape != (24, 7):
            raise ValueError(f"body_poses shape must be (24,7), got {body.shape}")
        if not body.flags.c_contiguous:
            body = np.ascontiguousarray(body)

        targets = self._targets_work
        rc = self._native_direct_targets_func(
            body.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(len(self.target_defs)),
            self._target_human_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(self._root_human_index),
            self._target_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(self._root_scale),
            self._target_pos_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._target_rot_offsets_wxyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(self.ground_offset),
            _UNITY_TO_RETARGET_QUAT_WXYZ.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"holoretarget_fast_targets failed with code {rc}")

        targets = self._apply_ground_calibration(targets, copy_targets=False)
        if self.offset_to_ground:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
            targets[:, 2] += np.float32(0.1 - min_z)
        return targets

    def _normalize_target_quats_inplace(self, quats: np.ndarray) -> None:
        norm = self._target_norm_work
        np.multiply(quats[:, 0], quats[:, 0], out=norm)
        norm += quats[:, 1] * quats[:, 1]
        norm += quats[:, 2] * quats[:, 2]
        norm += quats[:, 3] * quats[:, 3]
        np.maximum(norm, 1e-16, out=norm)
        np.sqrt(norm, out=norm)
        np.reciprocal(norm, out=norm)
        quats[:, 0] *= norm
        quats[:, 1] *= norm
        quats[:, 2] *= norm
        quats[:, 3] *= norm

    def _targets_from_pico_body_poses_direct_fused(self, body_poses: np.ndarray) -> np.ndarray:
        body = np.asarray(body_poses, dtype=np.float64)
        if body.shape != (24, 7):
            raise ValueError(f"body_poses shape must be (24,7), got {body.shape}")

        root_body = body[self._root_human_index]
        root_pos = self._target_root_pos_work
        root_pos[0] = root_body[0]
        root_pos[1] = -root_body[2]
        root_pos[2] = root_body[1]

        body_targets = self._target_body_poses_work
        np.take(body, self._target_human_indices, axis=0, out=body_targets)

        pose_t = self._target_pose_t_work
        pose_t[:, 0] = body_targets[:, 0]
        pose_t[:, 1] = -body_targets[:, 2]
        pose_t[:, 2] = body_targets[:, 1]

        q_raw = self._target_pose_q_work
        q_raw[:, 0] = body_targets[:, 6]
        q_raw[:, 1] = body_targets[:, 3]
        q_raw[:, 2] = body_targets[:, 4]
        q_raw[:, 3] = body_targets[:, 5]
        self._normalize_target_quats_inplace(q_raw)

        q_holoretarget = self._target_updated_quat_work
        aw, ax, ay, az = _UNITY_TO_RETARGET_QUAT_WXYZ
        bw, bx, by, bz = q_raw[:, 0], q_raw[:, 1], q_raw[:, 2], q_raw[:, 3]
        q_holoretarget[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        q_holoretarget[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        q_holoretarget[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        q_holoretarget[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        self._normalize_target_quats_inplace(q_holoretarget)

        scaled_pos = self._target_scaled_pos_work
        np.subtract(pose_t, root_pos[None, :], out=scaled_pos)
        scaled_pos *= self._target_scales[:, None]
        scaled_pos += (self._root_scale * root_pos)[None, :]

        updated_quat = q_raw
        a = q_holoretarget
        b = self._target_rot_offsets_wxyz
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        updated_quat[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        updated_quat[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        updated_quat[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        updated_quat[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        self._normalize_target_quats_inplace(updated_quat)

        offset = self._target_pos_offsets
        rotate_t = self._target_rotate_t_work
        qx, qy, qz = updated_quat[:, 1], updated_quat[:, 2], updated_quat[:, 3]
        vx, vy, vz = offset[:, 0], offset[:, 1], offset[:, 2]
        rotate_t[:, 0] = 2.0 * (qy * vz - qz * vy)
        rotate_t[:, 1] = 2.0 * (qz * vx - qx * vz)
        rotate_t[:, 2] = 2.0 * (qx * vy - qy * vx)

        qw = updated_quat[:, 0]
        tx, ty, tz = rotate_t[:, 0], rotate_t[:, 1], rotate_t[:, 2]
        target_pos = self._target_pos_work
        target_pos[:, 0] = scaled_pos[:, 0] + vx + qw * tx + (qy * tz - qz * ty)
        target_pos[:, 1] = scaled_pos[:, 1] + vy + qw * ty + (qz * tx - qx * tz)
        target_pos[:, 2] = scaled_pos[:, 2] + vz + qw * tz + (qx * ty - qy * tx)
        if self.ground_offset != 0.0:
            target_pos[:, 2] -= self.ground_offset

        targets = self._targets_work
        targets[:, :3] = target_pos
        targets[:, 3] = updated_quat[:, 1]
        targets[:, 4] = updated_quat[:, 2]
        targets[:, 5] = updated_quat[:, 3]
        targets[:, 6] = updated_quat[:, 0]
        targets = self._apply_ground_calibration(targets, copy_targets=False)

        if self.offset_to_ground:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
            targets[:, 2] += np.float32(0.1 - min_z)
        return targets

    def _targets_from_pico_body_poses_direct(self, body_poses: np.ndarray) -> np.ndarray:
        if self._native_direct_targets_func is not None:
            return self._targets_from_pico_body_poses_direct_native(body_poses)
        if self.fused_direct_targets:
            return self._targets_from_pico_body_poses_direct_fused(body_poses)

        body = np.asarray(body_poses, dtype=np.float64)
        if body.shape != (24, 7):
            raise ValueError(f"body_poses shape must be (24,7), got {body.shape}")

        root_body = body[self._root_human_index]
        root_pos = self._target_root_pos_work
        root_pos[0] = root_body[0]
        root_pos[1] = -root_body[2]
        root_pos[2] = root_body[1]

        body_targets = body[self._target_human_indices]
        pose_t = self._target_pose_t_work
        pose_t[:, 0] = body_targets[:, 0]
        pose_t[:, 1] = -body_targets[:, 2]
        pose_t[:, 2] = body_targets[:, 1]

        q_raw = self._target_pose_q_work
        q_raw[:, 0] = body_targets[:, 6]
        q_raw[:, 1:4] = body_targets[:, 3:6]
        q_raw /= np.clip(np.linalg.norm(q_raw, axis=1, keepdims=True), 1e-8, None)

        q_holoretarget = self._target_updated_quat_work
        aw, ax, ay, az = _UNITY_TO_RETARGET_QUAT_WXYZ
        bw, bx, by, bz = q_raw[:, 0], q_raw[:, 1], q_raw[:, 2], q_raw[:, 3]
        q_holoretarget[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        q_holoretarget[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        q_holoretarget[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        q_holoretarget[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        q_holoretarget /= np.clip(np.linalg.norm(q_holoretarget, axis=1, keepdims=True), 1e-8, None)

        scaled_pos = self._target_scaled_pos_work
        np.subtract(pose_t, root_pos[None, :], out=scaled_pos)
        scaled_pos *= self._target_scales[:, None]
        scaled_pos += (self._root_scale * root_pos)[None, :]

        updated_quat = q_raw
        a = q_holoretarget
        b = self._target_rot_offsets_wxyz
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        updated_quat[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        updated_quat[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        updated_quat[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        updated_quat[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        updated_quat /= np.clip(np.linalg.norm(updated_quat, axis=1, keepdims=True), 1e-8, None)

        target_pos = self._target_pos_work
        target_pos[:] = scaled_pos + _quat_rotate_wxyz(updated_quat, self._target_pos_offsets)
        if self.ground_offset != 0.0:
            target_pos[:, 2] -= self.ground_offset

        targets = self._targets_work
        targets[:, :3] = target_pos
        targets[:, 3] = updated_quat[:, 1]
        targets[:, 4] = updated_quat[:, 2]
        targets[:, 5] = updated_quat[:, 3]
        targets[:, 6] = updated_quat[:, 0]
        targets = self._apply_ground_calibration(targets, copy_targets=False)

        if self.offset_to_ground:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
            targets[:, 2] += np.float32(0.1 - min_z)
        return targets

    def _targets_from_pico_arrays(self, positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
        if self.prealloc_targets and not self.cpu_float32:
            return self._targets_from_pico_arrays_prealloc(positions, quats_wxyz)
        root_pos = positions[self._root_human_index]
        pose_t = positions[self._target_human_indices]
        pose_q = quats_wxyz[self._target_human_indices]
        if self.cpu_float32:
            scaled_root_pos = self._root_scale32 * root_pos
            scaled_pos = (pose_t - root_pos[None, :]) * self._target_scales32[:, None] + scaled_root_pos[None, :]
            updated_quat = _normalize_quat_wxyz32(_quat_mul_wxyz32(pose_q, self._target_rot_offsets_wxyz32))
            target_pos = scaled_pos + _quat_rotate_wxyz32(updated_quat, self._target_pos_offsets32)
        else:
            scaled_root_pos = self._root_scale * root_pos
            scaled_pos = (pose_t - root_pos[None, :]) * self._target_scales[:, None] + scaled_root_pos[None, :]
            updated_quat = _normalize_quat_wxyz(_quat_mul_wxyz(pose_q, self._target_rot_offsets_wxyz))
            target_pos = scaled_pos + _quat_rotate_wxyz(updated_quat, self._target_pos_offsets)
        if self.ground_offset != 0.0:
            target_pos = target_pos.copy()
            target_pos[:, 2] -= self.ground_offset

        targets = np.empty((len(self.target_defs), 7), dtype=np.float32)
        targets[:, :3] = target_pos.astype(np.float32)
        targets[:, 3:7] = updated_quat[:, [1, 2, 3, 0]].astype(np.float32)
        targets = self._apply_ground_calibration(targets)

        if self.offset_to_ground:
            min_z = float(np.min(targets[self._target_foot_mask, 2] if np.any(self._target_foot_mask) else targets[:, 2]))
            targets[:, 2] += np.float32(0.1 - min_z)
        return targets

    def retarget_from_pico_arrays(self, positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        targets = self._targets_from_pico_arrays(positions, quats_wxyz)
        t_targets = time.perf_counter()
        if self.graph_capture is not None and self._target_batch_copy_enabled and self.graph_target_copy:
            self._update_target_host(targets)
        else:
            self._set_targets(targets)
        self._sync_if_profile()
        t_set = time.perf_counter()

        self._seed_root_from_targets(targets)
        self._sync_if_profile()
        t_seed = time.perf_counter()

        if self.graph_capture is not None:
            self.wp.capture_launch(self.graph_capture)
        else:
            self._single_step()
        self._sync_if_profile()
        t_solve = time.perf_counter()

        qpos = self._read_qpos_host()
        t_output = time.perf_counter()
        qpos = self._project_qpos_to_continuous_limits(qpos)
        self._sync_if_profile()
        t_project = time.perf_counter()
        self.last_targets = targets
        self.last_output = qpos
        self.last_timing = {
            "holoretarget.newton_targets": t_targets - t0,
            "holoretarget.newton_set_targets": t_set - t_targets,
            "holoretarget.newton_root_seed": t_seed - t_set,
            "holoretarget.solve": t_solve - t_seed,
            "holoretarget.output": t_output - t_solve,
            "holoretarget.newton_project": t_project - t_output,
            "holoretarget.total": t_project - t0,
        }
        return qpos

    def solve_from_pico_arrays_no_output(self, positions: np.ndarray, quats_wxyz: np.ndarray) -> None:
        t0 = time.perf_counter()
        targets = self._targets_from_pico_arrays(positions, quats_wxyz)
        t_targets = time.perf_counter()
        if self.graph_capture is not None and self._target_batch_copy_enabled and self.graph_target_copy:
            self._update_target_host(targets)
        else:
            self._set_targets(targets)
        self._sync_if_profile()
        t_set = time.perf_counter()

        self._seed_root_from_targets(targets)
        self._sync_if_profile()
        t_seed = time.perf_counter()

        if self.graph_capture is not None:
            self.wp.capture_launch(self.graph_capture)
        else:
            self._single_step()
        self._sync_if_profile()
        t_solve = time.perf_counter()
        self.last_targets = targets
        self.last_output = None
        self.last_timing = {
            "holoretarget.newton_targets": t_targets - t0,
            "holoretarget.newton_set_targets": t_set - t_targets,
            "holoretarget.newton_root_seed": t_seed - t_set,
            "holoretarget.solve": t_solve - t_seed,
            "holoretarget.output": 0.0,
            "holoretarget.newton_project": 0.0,
            "holoretarget.total": t_solve - t0,
        }

    def retarget_from_pico_body_poses(self, body_poses: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        targets = self._targets_from_pico_body_poses_direct(body_poses)
        t_targets = time.perf_counter()
        if self.graph_capture is not None and self._target_batch_copy_enabled and self.graph_target_copy:
            self._update_target_host(targets)
        else:
            self._set_targets(targets)
        self._sync_if_profile()
        t_set = time.perf_counter()

        self._seed_root_from_targets(targets)
        self._sync_if_profile()
        t_seed = time.perf_counter()

        if self.graph_capture is not None:
            self.wp.capture_launch(self.graph_capture)
        else:
            self._single_step()
        self._sync_if_profile()
        t_solve = time.perf_counter()

        qpos = self._read_qpos_host()
        t_output = time.perf_counter()
        qpos = self._project_qpos_to_continuous_limits(qpos)
        self._sync_if_profile()
        t_project = time.perf_counter()
        self.last_targets = targets
        self.last_output = qpos
        self.last_timing = {
            "holoretarget.newton_targets": t_targets - t0,
            "holoretarget.newton_set_targets": t_set - t_targets,
            "holoretarget.newton_root_seed": t_seed - t_set,
            "holoretarget.solve": t_solve - t_seed,
            "holoretarget.output": t_output - t_solve,
            "holoretarget.newton_project": t_project - t_output,
            "holoretarget.total": t_project - t0,
        }
        return qpos

    def solve_from_pico_body_poses_no_output(self, body_poses: np.ndarray) -> None:
        t0 = time.perf_counter()
        targets = self._targets_from_pico_body_poses_direct(body_poses)
        t_targets = time.perf_counter()
        if self.graph_capture is not None and self._target_batch_copy_enabled and self.graph_target_copy:
            self._update_target_host(targets)
        else:
            self._set_targets(targets)
        self._sync_if_profile()
        t_set = time.perf_counter()

        self._seed_root_from_targets(targets)
        self._sync_if_profile()
        t_seed = time.perf_counter()

        if self.graph_capture is not None:
            self.wp.capture_launch(self.graph_capture)
        else:
            self._single_step()
        self._sync_if_profile()
        t_solve = time.perf_counter()
        self.last_targets = targets
        self.last_output = None
        self.last_timing = {
            "holoretarget.newton_targets": t_targets - t0,
            "holoretarget.newton_set_targets": t_set - t_targets,
            "holoretarget.newton_root_seed": t_seed - t_set,
            "holoretarget.solve": t_solve - t_seed,
            "holoretarget.output": 0.0,
            "holoretarget.newton_project": 0.0,
            "holoretarget.total": t_solve - t0,
        }


class HoloRetargetRunner:
    """Wrap the Newton IK solver and emit root pose plus joint positions."""

    def __init__(
        self,
        asset_root: str = "",
        robot: str = "unitree_g1",
        src_human: str = "xrobot",
        offset_to_ground: bool = False,
        newton_iterations: Optional[int] = None,
        newton_cuda_graph: bool = True,
        newton_joint_limit_weight: float = 10.0,
        newton_max_joint_step: float = 0.1,
        newton_target_table: str = "ik_match_table2",
        newton_robot_asset: str = "newton_builtin",
        newton_root_seed_mode: str = "off",
        ground_calibration_frames: int = 5,
        ground_height: float = 0.0,
        ground_lift_only: bool = True,
        ground_calibration_mode: str = "initial_mean",
        ground_target_scope: str = "feet",
        profile_timing: bool = False,
    ) -> None:
        self.asset_root = _insert_python_path(asset_root or DEFAULT_RETARGET_ASSET_ROOT)
        self.cpu_float32 = _env_bool("HOLORETARGET_CPU_FLOAT32", False)
        self.fast_pico_convert = _env_bool("HOLORETARGET_FAST_PICO_CONVERT", False)
        self.direct_pico_targets = _env_bool("HOLORETARGET_DIRECT_PICO_TARGETS", True)
        self.robot = robot
        self.src_human = src_human
        self.offset_to_ground = bool(offset_to_ground)
        self.ground_calibration_frames = max(0, int(ground_calibration_frames))
        self.ground_height = float(ground_height)
        self.ground_lift_only = bool(ground_lift_only)
        self.ground_calibration_mode = str(ground_calibration_mode or "initial_mean").lower()
        if self.ground_calibration_mode not in {"initial_mean", "sliding_min"}:
            raise ValueError("ground_calibration_mode must be 'initial_mean' or 'sliding_min'")
        self.ground_target_scope = str(ground_target_scope or "feet").lower()
        if self.ground_target_scope not in {"feet", "all"}:
            raise ValueError("ground_target_scope must be 'feet' or 'all'")
        self._ground_min_z_samples: list[float] = []
        self._ground_offset_z: float | None = None
        self.last_ground_offset_z = 0.0
        self.profile_timing = bool(profile_timing)

        solve_iterations = int(newton_iterations) if newton_iterations is not None else 1
        self.newton_solver = HoloNewtonG1Retargeter(
            asset_root=self.asset_root,
            src_human=src_human,
            robot=robot,
            actual_human_height=None,
            ik_iterations=solve_iterations,
            use_cuda_graph=newton_cuda_graph,
            joint_limit_weight=newton_joint_limit_weight,
            max_joint_step=newton_max_joint_step,
            target_table=newton_target_table,
            robot_asset=newton_robot_asset,
            offset_to_ground=offset_to_ground,
            ground_calibration_frames=ground_calibration_frames,
            ground_height=ground_height,
            ground_lift_only=ground_lift_only,
            ground_calibration_mode=ground_calibration_mode,
            ground_target_scope=ground_target_scope,
            root_seed_mode=newton_root_seed_mode,
            profile_timing=profile_timing,
        )
        self.num_dof = int(self.newton_solver.num_dof)
        if self.num_dof != DOF_POS_DIM:
            raise ValueError(
                f"robot {robot!r} must expose {DOF_POS_DIM} qpos DoF after root, "
                f"got {self.num_dof}"
            )
        self.last_timing: dict[str, float] = {}
        self._last_root_quat_wxyz: np.ndarray | None = None
        self._pico_positions_work = np.empty((24, 3), dtype=np.float64)
        self._pico_quats_raw_work = np.empty((24, 4), dtype=np.float64)
        self._pico_quats_out_work = np.empty((24, 4), dtype=np.float64)

    def _pico_body_poses_to_xrobot_arrays_fast(self, body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        body = np.asarray(body_poses, dtype=np.float64)
        if body.shape != (24, 7):
            raise ValueError(f"body_poses shape must be (24,7), got {body.shape}")

        positions = self._pico_positions_work
        positions[:, 0] = body[:, 0]
        positions[:, 1] = -body[:, 2]
        positions[:, 2] = body[:, 1]

        q_raw = self._pico_quats_raw_work
        q_raw[:, 0] = body[:, 6]
        q_raw[:, 1:4] = body[:, 3:6]
        q_raw /= np.clip(np.linalg.norm(q_raw, axis=1, keepdims=True), 1e-8, None)

        q_out = self._pico_quats_out_work
        aw, ax, ay, az = _UNITY_TO_RETARGET_QUAT_WXYZ
        bw, bx, by, bz = q_raw[:, 0], q_raw[:, 1], q_raw[:, 2], q_raw[:, 3]
        q_out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
        q_out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
        q_out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
        q_out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
        q_out /= np.clip(np.linalg.norm(q_out, axis=1, keepdims=True), 1e-8, None)
        return positions, q_out

    def _make_root_quat_continuous(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat = _normalize_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float32)).astype(np.float32)
        if self._last_root_quat_wxyz is not None and float(np.dot(quat, self._last_root_quat_wxyz)) < 0.0:
            quat = -quat
        self._last_root_quat_wxyz = quat.copy()
        return quat

    def reset_sequence(self) -> None:
        self._ground_min_z_samples.clear()
        self._ground_offset_z = None
        self.last_ground_offset_z = 0.0
        self._last_root_quat_wxyz = None
        self.last_timing = {}
        self.newton_solver.reset_sequence()

    def _calibrated_ground_offset(self, min_z: float) -> float:
        if self.ground_calibration_frames <= 0:
            self.last_ground_offset_z = 0.0
            return 0.0

        if self.ground_calibration_mode == "sliding_min":
            self._ground_min_z_samples.append(float(min_z))
            if len(self._ground_min_z_samples) > self.ground_calibration_frames:
                del self._ground_min_z_samples[:-self.ground_calibration_frames]
            offset = self.ground_height - float(np.min(self._ground_min_z_samples))
            if self.ground_lift_only:
                offset = max(0.0, offset)
        else:
            if self._ground_offset_z is None:
                if len(self._ground_min_z_samples) < self.ground_calibration_frames:
                    self._ground_min_z_samples.append(float(min_z))
                offset = self.ground_height - float(np.mean(self._ground_min_z_samples))
                if self.ground_lift_only:
                    offset = max(0.0, offset)
                if len(self._ground_min_z_samples) >= self.ground_calibration_frames:
                    self._ground_offset_z = offset
            else:
                offset = self._ground_offset_z

        self.last_ground_offset_z = float(offset)
        return float(offset)

    def retarget_qpos_from_pico_body_poses(
        self,
        body_poses: np.ndarray,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        if self.direct_pico_targets:
            positions = None
            quats_wxyz = None
        elif self.fast_pico_convert and not self.cpu_float32:
            positions, quats_wxyz = self._pico_body_poses_to_xrobot_arrays_fast(body_poses)
        elif self.cpu_float32:
            positions, quats_wxyz = pico_body_poses_to_xrobot_arrays32(body_poses)
        else:
            positions, quats_wxyz = pico_body_poses_to_xrobot_arrays(body_poses)
        t1 = time.perf_counter()
        if self.direct_pico_targets:
            qpos = np.asarray(
                self.newton_solver.retarget_from_pico_body_poses(body_poses),
                dtype=np.float32,
            )
        else:
            qpos = np.asarray(
                self.newton_solver.retarget_from_pico_arrays(positions, quats_wxyz),
                dtype=np.float32,
            )
        self.last_ground_offset_z = float(getattr(self.newton_solver, "last_ground_offset_z", 0.0))
        root_q_xyzw = _normalize_quat_wxyz(qpos[3:7]).astype(np.float32)
        root_rot_wxyz = np.array(
            [root_q_xyzw[3], root_q_xyzw[0], root_q_xyzw[1], root_q_xyzw[2]],
            dtype=np.float32,
        )
        root_rot_wxyz = self._make_root_quat_continuous(root_rot_wxyz)
        t2 = time.perf_counter()
        if qpos.shape[0] < QPOS_DIM:
            raise RuntimeError(
                f"qpos must have at least {QPOS_DIM} values, got {qpos.shape}"
            )

        root_pos = qpos[:3].astype(np.float32, copy=False)
        dof_pos = qpos[7:QPOS_DIM].astype(np.float32, copy=False)
        reference_qpos = np.empty(QPOS_DIM, dtype=np.float32)
        reference_qpos[:3] = root_pos
        reference_qpos[3:7] = root_rot_wxyz
        reference_qpos[7:] = dof_pos
        t3 = time.perf_counter()

        self.last_timing = {
            "holoretarget.pico_to_xrobot": t1 - t0,
            "holoretarget.solve": t2 - t1,
            "holoretarget.output": t3 - t2,
            "holoretarget.total": t3 - t0,
        }
        self.last_timing.update(getattr(self.newton_solver, "last_timing", {}))
        self.last_timing["holoretarget.total"] = t3 - t0
        return reference_qpos

    def retarget_qpos_from_target_arrays(
        self,
        positions: np.ndarray,
        quats_wxyz: np.ndarray,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        qpos = np.asarray(
            self.newton_solver.retarget_from_pico_arrays(positions, quats_wxyz),
            dtype=np.float32,
        )
        self.last_ground_offset_z = float(getattr(self.newton_solver, "last_ground_offset_z", 0.0))
        root_q_xyzw = _normalize_quat_wxyz(qpos[3:7]).astype(np.float32)
        root_rot_wxyz = np.array(
            [root_q_xyzw[3], root_q_xyzw[0], root_q_xyzw[1], root_q_xyzw[2]],
            dtype=np.float32,
        )
        root_rot_wxyz = self._make_root_quat_continuous(root_rot_wxyz)
        t1 = time.perf_counter()
        if qpos.shape[0] < QPOS_DIM:
            raise RuntimeError(
                f"qpos must have at least {QPOS_DIM} values, got {qpos.shape}"
            )

        root_pos = qpos[:3].astype(np.float32, copy=False)
        dof_pos = qpos[7:QPOS_DIM].astype(np.float32, copy=False)
        reference_qpos = np.empty(QPOS_DIM, dtype=np.float32)
        reference_qpos[:3] = root_pos
        reference_qpos[3:7] = root_rot_wxyz
        reference_qpos[7:] = dof_pos
        t2 = time.perf_counter()

        self.last_timing = {
            "holoretarget.target_arrays": 0.0,
            "holoretarget.solve": t1 - t0,
            "holoretarget.output": t2 - t1,
            "holoretarget.total": t2 - t0,
        }
        self.last_timing.update(getattr(self.newton_solver, "last_timing", {}))
        self.last_timing["holoretarget.total"] = t2 - t0
        return reference_qpos
