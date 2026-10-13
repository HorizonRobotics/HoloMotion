"""Pico SDK and GPU Retarget run outside the policy interpreter/GIL."""

import argparse, fcntl, mmap, os, struct, time
from pathlib import Path
import numpy as np
from humanoid_policy.local_retarget import LocalPicoRetargetSource
from humanoid_policy.local_retarget_process import HEADER, SNAPSHOT_BYTES


class Logger:
    def info(self, msg):
        print("[Pico worker] " + msg, flush=True)

    warn = info
    error = info


def body_pose():
    body = np.zeros((24, 7), dtype=np.float32)
    body[:, 1] = np.linspace(0.2, 1.6, 24)
    body[:, 0] = 0.15 * np.sin(np.arange(24))
    body[:, 6] = 1
    return body


class TestSdk:
    def __init__(self, path):
        self.path = Path(path)
        self.connected = False

    def init(self):
        pass

    def close(self):
        pass

    def is_body_data_available(self):
        if not self.path.exists():
            return False
        if not self.connected:
            self.connected = True
            # Simulate a cold SDK connection holding this process's GIL for 2 seconds.
            end = time.monotonic() + 2
            while time.monotonic() < end:
                pass
        return True

    def get_time_stamp_ns(self):
        return time.time_ns() // 20_000_000 * 20_000_000

    def get_body_joints_pose(self):
        return body_pose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--asset-root", default="")
    parser.add_argument("--max-source-age", type=float, default=0.6)
    parser.add_argument("--test-connect")
    args = parser.parse_args()
    if args.test_connect and {
        p.name for p in Path("/sys/class/net").iterdir()
    } != {"lo"}:
        raise RuntimeError(
            "Fake Pico input is restricted to an isolated test network"
        )
    try:
        os.nice(5)
    except OSError:
        pass
    sdk = TestSdk(args.test_connect) if args.test_connect else None
    source = LocalPicoRetargetSource(
        logger=Logger(),
        asset_root=args.asset_root,
        max_source_age=args.max_source_age,
        sdk=sdk,
    )
    # Exercise full Retarget + device-to-host path before announcing initialization.
    for _ in range(2):
        source.retargeter.retarget_qpos_device_from_body_poses(
            body_pose()
        ).numpy()
    source.start()
    with open(args.snapshot, "r+b", buffering=0) as f:
        shared = mmap.mmap(f.fileno(), SNAPSHOT_BYTES)
        shared[: HEADER.size] = HEADER.pack(0, 1, -1, 0, 0, 0, 0, 0)
        version = 0
        next_tick = time.monotonic()
        while True:
            result = source.step()
            if result is not None:
                qpos, frame, meta = result
                values = np.concatenate(
                    (qpos.numpy(), meta["body_poses"].reshape(-1))
                ).astype("<f4", copy=False)
                if np.isfinite(values).all():
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        version += 2
                        struct.pack_into("<Q", shared, 0, version - 1)
                        shared[HEADER.size :] = values.tobytes()
                        shared[8 : HEADER.size] = HEADER.pack(
                            version,
                            1,
                            int(frame),
                            int(meta["timestamp_ns"]),
                            float(meta["timestamp_realtime"]),
                            float(meta["timestamp_monotonic"]),
                            float(meta["dt"]),
                            float(meta["fps"]),
                        )[8:]
                        struct.pack_into("<Q", shared, 0, version)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            next_tick += 0.02
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()


if __name__ == "__main__":
    main()
