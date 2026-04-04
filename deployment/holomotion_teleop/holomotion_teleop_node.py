#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-process teleoperation pipeline.

This node reads raw Pico body tracking data from XRoboToolkit, converts it to
SMPL, applies GMR retargeting, and publishes a 65D observation vector to the
robot over ZMQ.

Data flow:
    xrobotoolkit_sdk (body_poses 24x7)
        -> body_poses_to_smpl_pose_trans
        -> SMPL_Parser / humanoid_fk
        -> GMR
        -> latest_obs(65)
        -> ZMQ PUB

Message format:
    [topic_bytes][1280-byte JSON header][binary payload]

Default payload fields:
    - latest_obs: (65,) float32
    - frame_index: (1,) int64
    - timestamp_realtime: (1,) float64
    - timestamp_monotonic: (1,) float64
    - timestamp_ns: (1,) int64
    - pico_dt: (1,) float32
    - pico_fps: (1,) float32
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from scipy.spatial.transform import Rotation as R


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
HOLOMOTION_ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
SMPL_ASSET_DIR = os.path.join(HOLOMOTION_ROOT_DIR, "assets", "smpl")
for extra_path in (
    FILE_DIR,
    os.path.join(FILE_DIR, "GMR"),
    os.path.join(FILE_DIR, "SMPLSim"),
):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)


try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

from third_party.GMR.general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
from smpl_sim.smpllib.smpl_parser import SMPL_Parser


MIRROR_POSE = False
MIRROR_AXIS = "x"
HEADER_SIZE = 1280
OUT_TOPIC = b"obs65"

GMR_LR_SWAP_PAIRS = [
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_foot", "right_foot"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
]

SMPL_PARENTS_24 = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in wxyz format.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle vectors to rotation matrices.
    Input shape: (..., 3)
    Output shape: (..., 3, 3)
    """
    orig_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    axis = aa / torch.clamp(theta, min=1e-8)

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    zeros = torch.zeros_like(x)

    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).unsqueeze(0).expand(aa.shape[0], -1, -1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    axis_outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)

    small = (theta.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    rot = cos_theta * eye + (1.0 - cos_theta) * axis_outer + sin_theta * K
    rot = torch.where(small, eye, rot)
    return rot.reshape(orig_shape + (3, 3))


class Humanoid_Batch_V2:
    """
    Minimal per-frame SMPL kinematics helper used by this script only.
    Keeping it local avoids importing the much larger training/visualization module.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.smpl_24_parents = [
            -1, 0, 0, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 9,
            9, 12, 13, 14, 16, 17,
            18, 19, 20, 21,
        ]

    @staticmethod
    def _relative_link_position(joints_world: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        return joints_world - root_pos.unsqueeze(0)

    def _relative_link_pose(self, full_pose_aa: torch.Tensor) -> torch.Tensor:
        joint_count = full_pose_aa.shape[0]
        assert joint_count == len(self.smpl_24_parents), (
            f"Joint count mismatch: {joint_count} vs {len(self.smpl_24_parents)}"
        )

        rotation_local = axis_angle_to_matrix(full_pose_aa)
        rotation_global = torch.empty_like(rotation_local)
        for joint_idx in range(joint_count):
            parent = self.smpl_24_parents[joint_idx]
            if parent == -1:
                rotation_global[joint_idx] = rotation_local[joint_idx]
            else:
                rotation_global[joint_idx] = rotation_global[parent] @ rotation_local[joint_idx]
        return rotation_global

    def step_per_frame(
        self,
        full_pose_aa: torch.Tensor,
        root_pos: torch.Tensor,
        joints: torch.Tensor,
    ) -> SimpleNamespace:
        global_joints_position = joints
        global_joints2root_pos = self._relative_link_position(joints[1:, :], root_pos)
        global_joints_rotation_mat = self._relative_link_pose(full_pose_aa)

        return SimpleNamespace(
            global_joints2root_pos=global_joints2root_pos,
            global_joints_rotation_mat=global_joints_rotation_mat,
            global_joints_position=global_joints_position,
        )


humanoid_fk = Humanoid_Batch_V2()


@dataclass
class PicoToSmplConfig:
    quat_scalar_first: bool = False
    apply_global_y_180: bool = True
    apply_root_rx90: bool = True
    root_align_degrees: float = 90.0
    root_align_axis: str = "x"


def body_poses_to_smpl_pose_trans(
    body_poses: np.ndarray,
    parents: np.ndarray = SMPL_PARENTS_24,
    cfg: Optional[PicoToSmplConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = PicoToSmplConfig()

    body_poses = np.asarray(body_poses, dtype=np.float32)
    if body_poses.shape != (24, 7):
        raise ValueError(f"body_poses shape must be (24,7), got {body_poses.shape}")

    positions = body_poses[:, 0:3].astype(np.float32)
    qx, qy, qz, qw = body_poses[:, 3], body_poses[:, 4], body_poses[:, 5], body_poses[:, 6]
    global_quats_sfirst = np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)
    global_rots = R.from_quat(global_quats_sfirst, scalar_first=True)

    if cfg.apply_global_y_180:
        global_rots = global_rots * R.from_euler("y", 180.0, degrees=True)

    local_rots = []
    for i in range(24):
        parent = int(parents[i])
        if parent == -1:
            local_rots.append(global_rots[i])
        else:
            local_rots.append(global_rots[parent].inv() * global_rots[i])

    pose_aa_24x3 = np.stack([rot.as_rotvec() for rot in local_rots], axis=0).astype(np.float32)
    trans = positions[0].astype(np.float32)

    if cfg.apply_root_rx90:
        rot_align = R.from_euler(cfg.root_align_axis, cfg.root_align_degrees, degrees=True).as_matrix().astype(
            np.float32
        )
        root_matrix = R.from_rotvec(pose_aa_24x3[0]).as_matrix().astype(np.float32)
        pose_aa_24x3[0] = R.from_matrix(rot_align @ root_matrix).as_rotvec().astype(np.float32)
        trans = (rot_align @ trans.reshape(3, 1)).reshape(3).astype(np.float32)

    return pose_aa_24x3, trans


def _mirror_matrix(mirror_axis: str) -> np.ndarray:
    if mirror_axis == "x":
        return np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    if mirror_axis == "y":
        return np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    if mirror_axis == "z":
        return np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    raise ValueError(f"mirror_axis must be one of x/y/z, got {mirror_axis}")


def safe_normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def mirror_pos_and_quat_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray, mirror_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32).reshape(3,)
    q = safe_normalize_quat_wxyz(quat_wxyz)
    M = _mirror_matrix(mirror_axis)

    pos_m = (M @ pos).astype(np.float32)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    rot_m = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
    rot_m = (M @ rot_m @ M).astype(np.float32)
    q_m_xyzw = R.from_matrix(rot_m).as_quat().astype(np.float32)
    quat_m_wxyz = np.array([q_m_xyzw[3], q_m_xyzw[0], q_m_xyzw[1], q_m_xyzw[2]], dtype=np.float32)
    return pos_m, safe_normalize_quat_wxyz(quat_m_wxyz)


def mirror_and_swap_gmr_input(gmr_input: Dict[str, Any], mirror_axis: str = "x") -> Dict[str, Any]:
    mirrored: Dict[str, Any] = {}
    for key, (pos, quat) in gmr_input.items():
        mirrored[key] = mirror_pos_and_quat_wxyz(pos, quat, mirror_axis)

    out = dict(mirrored)
    for a, b in GMR_LR_SWAP_PAIRS:
        if a in out and b in out:
            out[a], out[b] = out[b], out[a]
    return out


def pack_numpy_message(payload: dict, topic: bytes = OUT_TOPIC, version: int = 1) -> bytes:
    fields = []
    binary_data = []

    for key, value in payload.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == np.uint8:
            dtype_str = "u8"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))

        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        binary_data.append(value.tobytes())

    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    header_bytes = header_bytes.ljust(HEADER_SIZE, b"\x00")
    return topic + header_bytes + b"".join(binary_data)


class PicoReader:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_stamp_ns = None
        self._fps_ema = 0.0
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue

            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue

            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst_fps = 1.0 / device_dt
                self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst_fps)
            self._last_stamp_ns = stamp_ns

            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()
                body_poses_np = np.asarray(body_poses, dtype=np.float32)
                if body_poses_np.shape != (24, 7):
                    print(f"[PicoReader] WARNING: unexpected body_poses shape: {body_poses_np.shape}")

                sample = {
                    "body_poses_np": body_poses_np,
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": int(stamp_ns),
                    "dt": float(device_dt),
                    "fps": float(self._fps_ema),
                }
                with self._lock:
                    self._latest = sample

                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] shape={body_poses_np.shape}, "
                        f"dt_ts={device_dt * 1000.0:.2f} ms, fps={self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as exc:
                print(f"[PicoReader] read error: {exc}")


class ZmqObsSender:
    def __init__(self, uri: str, logger, topic: bytes = OUT_TOPIC, mode: str = "bind", conflate: bool = True):
        self.logger = logger
        self.topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        if conflate and hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1)

        if mode == "bind":
            self._socket.bind(uri)
        elif mode == "connect":
            self._socket.connect(uri)
        else:
            raise ValueError("mode must be 'bind' or 'connect'")

        self._last_send_time = None
        self._send_freq_log = []
        self._frame_count = 0
        self.logger.info(f"[ZMQOut] sender ready: mode={mode}, uri={uri}, topic={topic.decode('utf-8')}")

    def send(self, latest_obs: np.ndarray, frame_index: int, sample_meta: dict):
        payload = {
            "latest_obs": np.asarray(latest_obs, dtype=np.float32),
            "frame_index": np.array([frame_index], dtype=np.int64),
            "timestamp_realtime": np.array([sample_meta["timestamp_realtime"]], dtype=np.float64),
            "timestamp_monotonic": np.array([sample_meta["timestamp_monotonic"]], dtype=np.float64),
            "timestamp_ns": np.array([sample_meta["timestamp_ns"]], dtype=np.int64),
            "pico_dt": np.array([sample_meta["dt"]], dtype=np.float32),
            "pico_fps": np.array([sample_meta["fps"]], dtype=np.float32),
        }
        packet = pack_numpy_message(payload, topic=self.topic)
        self._socket.send(packet)

        now = time.time()
        if self._last_send_time is not None:
            dt = now - self._last_send_time
            if dt > 0:
                self._send_freq_log.append(1.0 / dt)
                self._frame_count += 1
                if self._frame_count >= 50:
                    avg_freq = sum(self._send_freq_log) / len(self._send_freq_log)
                    self.logger.info(f"Average ZMQ send rate: {avg_freq:.2f} Hz")
                    self._send_freq_log.clear()
                    self._frame_count = 0
        self._last_send_time = now

    def stop(self):
        self._socket.close(0)
        self._context.term()
        self.logger.info("🛑 ZMQ obs sender stopped")


class VRNodeXRTPicoGMRZmqOut:
    def __init__(
        self,
        robot_zmq_uri: str,
        robot_zmq_mode: str = "bind",
        loop_hz: float = 55.0,
        timing_log_every: int = 100,
        save_obs_path: str = "",
    ):
        self.device = "cpu"
        logging.getLogger("websockets").setLevel(logging.WARNING)
        self.info(f"✅ VRNodeXRTPicoGMRZmqOut running on device={self.device}")
        self.info("starting xrt pico -> gmr -> robot zmq node")

        self.gmr = GMR(src_human="smplx", tgt_robot="unitree_g1")
        self.smpl_parser = SMPL_Parser(model_path=SMPL_ASSET_DIR, gender="neutral")
        if hasattr(self.smpl_parser, "to"):
            self.smpl_parser = self.smpl_parser.to(self.device)

        self.betas = torch.zeros(1, 10, device=self.device)
        self.gmr_input_data: Dict[str, Any] = {}
        self.prev_dof_pos = None
        self.lasttime = None
        self.timing_log_every = max(1, timing_log_every)
        self.save_obs_path = save_obs_path
        self.mirror_pose = MIRROR_POSE
        self.mirror_axis = MIRROR_AXIS
        self.tick_count = 0
        self.frame_index = 0
        self.timing_sums_ms = defaultdict(float)
        self.saved_obs = []
        self.latest_sample = None

        self.reader = PicoReader()
        self.reader.start()
        self.sender = ZmqObsSender(uri=robot_zmq_uri, logger=self, mode=robot_zmq_mode)
        self.start_loop(loop_hz)

    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def debug(self, msg): print(f"[DEBUG] {msg}")

    def _accumulate_timing(self, name: str, start_time: float) -> float:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.timing_sums_ms[name] += elapsed_ms
        return elapsed_ms

    def _maybe_log_timing(self):
        if self.tick_count <= 0 or self.tick_count % self.timing_log_every != 0:
            return
        avg_parts = []
        for key in ("body_poses_to_smpl", "smpl_to_joints", "gmr_retarget", "postprocess_send", "tick_total"):
            if key in self.timing_sums_ms:
                avg_ms = self.timing_sums_ms[key] / self.timing_log_every
                avg_parts.append(f"{key}={avg_ms:.2f}ms")
        if avg_parts:
            self.info("[Timing] " + ", ".join(avg_parts))
        self.timing_sums_ms.clear()

    def process_smpl_pose_trans_to_gmr_input(self, smpl_pose_aa, smpl_trans) -> Dict[str, Any]:
        stage_start = time.perf_counter()
        if not isinstance(smpl_pose_aa, torch.Tensor):
            smpl_pose_aa = torch.tensor(smpl_pose_aa, dtype=torch.float32)
        if not isinstance(smpl_trans, torch.Tensor):
            smpl_trans = torch.tensor(smpl_trans, dtype=torch.float32)

        pose = smpl_pose_aa.to(self.device, dtype=torch.float32)
        trans = smpl_trans.to(self.device, dtype=torch.float32)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
        if trans.ndim == 1:
            trans = trans.unsqueeze(0)

        verts, joints = self.smpl_parser.get_joints_verts(pose, self.betas, trans)
        # joints[..., 2] -= verts[0, :, 2].min().item()

        pose = pose.squeeze(0)
        trans = trans.squeeze(0)
        joints = joints.squeeze(0)
        motion_state = humanoid_fk.step_per_frame(pose, trans, joints)

        global_joints_position = motion_state.global_joints_position
        global_joints_rotation_mat = motion_state.global_joints_rotation_mat
        global_joints_qua_wxyz = matrix_to_quaternion(global_joints_rotation_mat)

        smpl_to_gmr = {
            "pelvis": 0,
            "spine3": 9,
            "left_hip": 1,
            "right_hip": 2,
            "left_knee": 4,
            "right_knee": 5,
            "left_foot": 10,
            "right_foot": 11,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "left_elbow": 18,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
        }

        gmr_input_data: Dict[str, Any] = {}
        for name, idx in smpl_to_gmr.items():
            pos = global_joints_position[idx].detach().cpu().numpy()
            quat = global_joints_qua_wxyz[idx].detach().cpu().numpy()
            gmr_input_data[name] = (pos, quat)

        if self.mirror_pose:
            gmr_input_data = mirror_and_swap_gmr_input(gmr_input_data, mirror_axis=self.mirror_axis)

        self._accumulate_timing("smpl_to_joints", stage_start)
        return gmr_input_data

    def process_xrt_frame_to_gmr_input(self, sample: dict):
        body_poses = np.asarray(sample["body_poses_np"], dtype=np.float32)
        if body_poses.shape != (24, 7):
            raise ValueError(f"[XRT] body_poses_np must have shape (24,7), got {body_poses.shape}")

        stage_start = time.perf_counter()
        pose_aa, trans = body_poses_to_smpl_pose_trans(
            body_poses,
            cfg=PicoToSmplConfig(
                apply_global_y_180=True,
                apply_root_rx90=True,
                root_align_axis="x",
                root_align_degrees=90.0,
            ),
        )
        self._accumulate_timing("body_poses_to_smpl", stage_start)
        self.gmr_input_data = self.process_smpl_pose_trans_to_gmr_input(pose_aa, trans)

    def process_gmr_output(self):
        stage_start = time.perf_counter()
        qpos = self.gmr.retarget(self.gmr_input_data)
        self._accumulate_timing("gmr_retarget", stage_start)

        stage_start = time.perf_counter()
        root_pos = qpos[:3]
        root_rot = qpos[3:7]
        dof_pos = qpos[7:]

        now = time.time()
        delta_time = 1 / 50 if self.lasttime is None else (now - self.lasttime)
        self.lasttime = now

        if self.prev_dof_pos is None:
            dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
        else:
            dof_vel = (dof_pos - self.prev_dof_pos) / max(delta_time, 1e-6)
        self.prev_dof_pos = dof_pos.copy()

        latest_obs = np.concatenate([dof_pos, dof_vel, root_pos, root_rot], axis=0).astype(np.float32)
        self.publish_data(latest_obs)
        self.sender.send(latest_obs, self.frame_index, self.latest_sample)
        self.saved_obs.append(latest_obs.copy())
        self.frame_index += 1
        self._accumulate_timing("postprocess_send", stage_start)
        return latest_obs

    def publish_data(self, motion_state: np.ndarray):
        if motion_state.size != 65:
            self.error(f"Output dim {motion_state.size} != expected 65")
            return
        if np.isnan(motion_state).any():
            self.error("NaN detected")
            return

    def save_observations(self):
        if not self.save_obs_path:
            return
        if len(self.saved_obs) == 0:
            self.warning(f"[SaveObs] no observations to save: {self.save_obs_path}")
            return

        obs_array = np.stack(self.saved_obs, axis=0).astype(np.float32)
        save_dir = os.path.dirname(self.save_obs_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if self.save_obs_path.endswith(".npy"):
            np.save(self.save_obs_path, obs_array)
        else:
            np.savez_compressed(
                self.save_obs_path,
                latest_obs=obs_array,
                columns=np.array(["dof_pos(29)", "dof_vel(29)", "root_pos(3)", "root_rot_wxyz(4)"], dtype=object),
            )
        self.info(f"[SaveObs] saved {obs_array.shape[0]} frames to {self.save_obs_path}")

    def start_loop(self, hz=50):
        self.info(f"Starting main loop at {hz} Hz")
        interval = 1.0 / hz

        def loop():
            next_time = time.time()
            while True:
                self._tick()
                next_time += interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.time()

        threading.Thread(target=loop, daemon=True).start()

    def _tick(self):
        tick_start = time.perf_counter()
        sample = self.reader.get_latest()
        if sample is not None:
            try:
                self.latest_sample = sample
                self.process_xrt_frame_to_gmr_input(sample)
                self.process_gmr_output()
            except Exception as exc:
                self.error(f"[tick_error] {exc}")
                self.error(traceback.format_exc())
                return
        elif self.prev_dof_pos is not None:
            try:
                self.process_gmr_output()
            except Exception as exc:
                self.error(f"[tick_error] {exc}")
                self.error(traceback.format_exc())
                return

        self.tick_count += 1
        self._accumulate_timing("tick_total", tick_start)
        self._maybe_log_timing()

    def stop(self):
        self.reader.stop()
        self.sender.stop()
        self.save_observations()
        try:
            if xrt is not None and hasattr(xrt, "close"):
                xrt.close()
        except Exception:
            pass


def init_xrt(start_service: bool = True):
    if xrt is None:
        raise ImportError("XRoboToolkit SDK not available. Install xrobotoolkit_sdk first.")
    if start_service:
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="XRT Pico -> GMR -> robot ZMQ(65D)")
    parser.add_argument("--robot-zmq-uri", default="tcp://*:6001", help="Robot-side ZMQ uri for 65D obs output")
    parser.add_argument("--robot-zmq-mode", default="bind", choices=["bind", "connect"])
    parser.add_argument("--hz", type=float, default=55.0, help="Main loop frequency / publish cap")
    parser.add_argument("--timing-log-every", type=int, default=200, help="Print average stage timing every N ticks")
    parser.add_argument("--save-obs-path", type=str, default="", help="Optional path to save emitted 65D observations")
    parser.add_argument("--skip-start-service", action="store_true", help="Do not auto-run /opt/apps/roboticsservice/runService.sh")
    args = parser.parse_args()

    init_xrt(start_service=not args.skip_start_service)
    node = VRNodeXRTPicoGMRZmqOut(
        robot_zmq_uri=args.robot_zmq_uri,
        robot_zmq_mode=args.robot_zmq_mode,
        loop_hz=args.hz,
        timing_log_every=args.timing_log_every,
        save_obs_path=args.save_obs_path,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()
        print("🛑 Program terminated by user.")


if __name__ == "__main__":
    main()

