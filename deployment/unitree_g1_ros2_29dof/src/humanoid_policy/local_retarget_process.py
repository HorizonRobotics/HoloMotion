"""Bounded snapshot reads from a separate Pico/Retarget process."""

from __future__ import annotations
import ctypes, fcntl, mmap, os, struct, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HEADER = struct.Struct(
    "<Qqqqdddd"
)  # version, ready, frame, SDK stamp, realtime, monotonic, dt, fps
SNAPSHOT_BYTES = HEADER.size + (36 + 24 * 7) * 4


class LocalPicoRetargetProcessSource:
    def __init__(
        self, *, logger, asset_root="", max_source_age=0.6, device="cuda:0"
    ):
        import warp as wp

        self.logger = logger
        self.asset_root = asset_root
        self.max_source_age = max_source_age
        self.frame_index = 0
        self.last_sample = None
        self._proc = None
        self._warned = False
        self._last_ready_sample = None
        fd, self._path = tempfile.mkstemp(
            prefix="holomotion_pico_", suffix=".snapshot"
        )
        os.ftruncate(fd, SNAPSHOT_BYTES)
        self._map = mmap.mmap(fd, SNAPSHOT_BYTES)
        self._fd = fd
        self._map[: HEADER.size] = HEADER.pack(0, 0, -1, 0, 0, 0, 0, 0)
        self.wp = wp
        self._host = wp.empty(36, dtype=wp.float32, device="cpu", pinned=True)
        self._numpy = np.ctypeslib.as_array(
            (ctypes.c_float * 36).from_address(self._host.ptr)
        )
        self._numpy.fill(0)
        self._numpy[3] = 1
        self._device = wp.empty(36, dtype=wp.float32, device=device)
        wp.copy(self._device, self._host)
        wp.synchronize_device(device)

    def start(self):
        script = Path(__file__).with_name("local_retarget_worker.py")
        args = [
            sys.executable,
            "-u",
            str(script),
            "--snapshot",
            self._path,
            "--asset-root",
            self.asset_root,
            "--max-source-age",
            str(self.max_source_age),
        ]
        test_file = os.environ.get("HOLOMOTION_TEST_PICO_CONNECT_FILE")
        if test_file:
            args += ["--test-connect", test_file]
        self._proc = subprocess.Popen(args, start_new_session=True)
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self.logger.error(
                    "[Pico] input process failed; walking remains available, teleop unavailable"
                )
                return
            if HEADER.unpack_from(self._map)[1] == 1:
                self.logger.info(
                    f"[Pico] input process initialized (pid={self._proc.pid}); waiting for a real Pico reference"
                )
                return
            time.sleep(0.05)
        self.logger.error(
            "[Pico] initialization timed out; walking remains available, teleop unavailable"
        )
        self.stop()

    @property
    def ready(self):
        if (
            self._proc is None
            or self._proc.poll() is not None
            or self._map.closed
        ):
            return False
        try:
            fcntl.flock(self._fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            # A writer holding the lock is not evidence that the stream stopped.
            # Reuse only readiness metadata from an earlier coherent read;
            # its original source timestamp still enforces the age limit.
            cached = self._last_ready_sample
            return (
                cached is not None
                and 0 <= time.monotonic() - cached <= self.max_source_age
            )
        try:
            _, initialized, frame, _, _, sample_monotonic, _, _ = (
                HEADER.unpack_from(self._map)
            )
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        valid = (
            initialized == 1 and frame >= 0 and np.isfinite(sample_monotonic)
        )
        self._last_ready_sample = sample_monotonic if valid else None
        return (
            valid
            and 0 <= time.monotonic() - sample_monotonic <= self.max_source_age
        )

    def step(self):
        if not self.ready:
            return None
        before = struct.unpack_from("<Q", self._map)[0]
        if before & 1:
            return None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            data = self._map[:]
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        if struct.unpack_from("<Q", self._map)[0] != before:
            return None
        _, ready, frame, stamp, realtime, monotonic, dt, fps = (
            HEADER.unpack_from(data)
        )
        if (
            not ready
            or frame < 0
            or time.monotonic() - monotonic > self.max_source_age
        ):
            return None
        values = np.frombuffer(data, dtype="<f4", offset=HEADER.size)
        if not np.isfinite(values).all():
            return None
        self._numpy[:] = values[:36]
        self.wp.copy(self._device, self._host)
        meta = {
            "timestamp_ns": stamp,
            "timestamp_realtime": realtime,
            "timestamp_monotonic": monotonic,
            "dt": dt,
            "fps": fps,
            "body_poses": values[36:].reshape(24, 7).copy(),
        }
        self.frame_index = int(frame)
        self.last_sample = meta
        return self._device, int(frame), meta

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        if not self._map.closed:
            self._map.close()
            os.close(self._fd)
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
