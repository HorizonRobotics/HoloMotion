#!/usr/bin/env python3
"""
Subscribe to obs65 packets and visualize the robot pose in MuJoCo.

Expected packet layout:
- topic: b"obs65"
- latest_obs: float32[65] = dof_pos[29] + dof_vel[29] + root_pos[3] + root_rot_wxyz[4]
- frame_index: int64[1]
- timestamp_realtime: float64[1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import zmq


HEADER_SIZE = 1280
OBS_DIM = 65
DOF_DIM = 29

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GMR_ROOT = PROJECT_ROOT / "thirdparties" / "GMR"
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import RobotMotionViewer  # noqa: E402


def dtype_from_str(dtype_str: str) -> np.dtype:
    mapping = {
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "u8": np.dtype("u1"),
        "bool": np.dtype("?"),
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype in message: {dtype_str}")
    return mapping[dtype_str]


def decode_numpy_message(packet: bytes, topic: bytes) -> Dict[str, np.ndarray]:
    if not packet.startswith(topic):
        raise ValueError("Unexpected ZMQ topic prefix")

    header_start = len(topic)
    header_end = header_start + HEADER_SIZE
    if len(packet) < header_end:
        raise ValueError("Packet too short for fixed-size header")

    header_bytes = packet[header_start:header_end]
    payload = memoryview(packet[header_end:])
    header_json = header_bytes.rstrip(b"\x00").decode("utf-8")
    header = json.loads(header_json)

    result: Dict[str, np.ndarray] = {}
    offset = 0
    for field in header.get("fields", []):
        name = field["name"]
        shape = tuple(field["shape"])
        dtype = dtype_from_str(field["dtype"])
        count = int(np.prod(shape))
        size_bytes = count * dtype.itemsize
        array = np.frombuffer(
            payload[offset : offset + size_bytes], dtype=dtype
        ).reshape(shape).copy()
        result[name] = array
        offset += size_bytes
    return result


class Obs65Receiver:
    def __init__(self, uri: str, topic: str):
        self.uri = uri
        self.topic = topic.encode("utf-8")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        if hasattr(zmq, "CONFLATE"):
            self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(uri)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def recv_latest(self, timeout_ms: int) -> Dict[str, np.ndarray] | None:
        events = dict(self.poller.poll(timeout=timeout_ms))
        if self.socket not in events:
            return None

        packet = self.socket.recv()
        while True:
            events = dict(self.poller.poll(timeout=0))
            if self.socket not in events:
                break
            packet = self.socket.recv()

        return decode_numpy_message(packet, topic=self.topic)

    def close(self):
        self.socket.close(0)
        self.context.term()


def obs_to_state(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if obs.size != OBS_DIM:
        raise ValueError(f"latest_obs must have {OBS_DIM} values, got {obs.size}")
    dof_pos = np.asarray(obs[:DOF_DIM], dtype=np.float32)
    root_pos = np.asarray(obs[58:61], dtype=np.float32)
    root_rot = np.asarray(obs[61:65], dtype=np.float32)
    return root_pos, root_rot, dof_pos


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize obs65 ZMQ stream in MuJoCo")
    parser.add_argument("--uri", default="tcp://127.0.0.1:6001", help="Publisher endpoint to connect to")
    parser.add_argument("--topic", default="obs65", help="ZMQ topic to subscribe")
    parser.add_argument("--robot", default="unitree_g1", help="Robot type supported by GMR RobotMotionViewer")
    parser.add_argument("--viewer-fps", type=float, default=55.0, help="Viewer refresh rate")
    parser.add_argument("--recv-timeout-ms", type=int, default=10, help="Polling timeout for new packets")
    parser.add_argument("--print-every", type=int, default=100, help="Print stream stats every N received frames")
    parser.add_argument("--no-follow-camera", action="store_true", help="Disable camera follow behavior")
    parser.add_argument("--dry-run", action="store_true", help="Receive and print packets without opening MuJoCo")
    return parser.parse_args()


def main():
    args = parse_args()

    receiver = Obs65Receiver(uri=args.uri, topic=args.topic)
    print(f"[INFO] obs65 viewer connected: uri={args.uri}, topic={args.topic}, robot={args.robot}")

    viewer = None
    latest_root_pos = np.array([0.0, 0.0, 0.793], dtype=np.float32)
    latest_root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    latest_dof_pos = np.zeros(DOF_DIM, dtype=np.float32)

    if not args.dry_run:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.viewer_fps,
            camera_follow=not args.no_follow_camera,
        )
        latest_root_pos = viewer.data.qpos[:3].copy().astype(np.float32)
        latest_root_rot = viewer.data.qpos[3:7].copy().astype(np.float32)
        latest_dof_pos = viewer.data.qpos[7 : 7 + DOF_DIM].copy().astype(np.float32)

    recv_count = 0
    last_recv_time = None
    recv_freq_log: list[float] = []

    try:
        while True:
            decoded = receiver.recv_latest(timeout_ms=args.recv_timeout_ms)
            if decoded is not None:
                latest_obs = decoded.get("latest_obs")
                if latest_obs is None:
                    print("[WARN] packet does not contain latest_obs")
                else:
                    latest_obs = np.asarray(latest_obs).reshape(-1)
                    latest_root_pos, latest_root_rot, latest_dof_pos = obs_to_state(latest_obs)

                    recv_count += 1
                    now = time.time()
                    if last_recv_time is not None:
                        dt = now - last_recv_time
                        if dt > 0:
                            recv_freq_log.append(1.0 / dt)
                    last_recv_time = now

                    if recv_count % max(1, args.print_every) == 0:
                        frame_index = int(decoded.get("frame_index", np.array([-1], dtype=np.int64))[0])
                        avg_recv_hz = sum(recv_freq_log) / len(recv_freq_log) if recv_freq_log else 0.0
                        print(
                            f"[obs65] frame={frame_index}, recv_hz={avg_recv_hz:.2f}, "
                            f"root_pos={np.array2string(latest_root_pos, precision=4, suppress_small=True)}, "
                            f"root_rot_wxyz={np.array2string(latest_root_rot, precision=4, suppress_small=True)}"
                        )
                        recv_freq_log.clear()

            if viewer is not None:
                viewer.step(
                    latest_root_pos,
                    latest_root_rot,
                    latest_dof_pos,
                    rate_limit=True,
                    follow_camera=not args.no_follow_camera,
                )
            elif decoded is None:
                time.sleep(max(args.recv_timeout_ms, 1) / 1000.0)
    except KeyboardInterrupt:
        print("\n[INFO] viewer stopped by user")
    finally:
        receiver.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
