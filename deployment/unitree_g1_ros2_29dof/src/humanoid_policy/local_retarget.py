"""Local Pico input and HoloRetarget adapter for the Orin policy clock."""

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np


class AsyncDeviceQposSnapshotter:
    """Copy occasional device qpos snapshots without waiting in control code."""

    def __init__(
        self,
        *,
        device: str,
        max_hz: float,
        slot_count: int = 2,
    ) -> None:
        import warp as wp

        self.wp = wp
        self.device = wp.get_device(device)
        self.max_hz = float(max_hz)
        if not self.device.is_cuda:
            raise RuntimeError(f"Warp CUDA device is unavailable: {device}")
        if self.max_hz <= 0.0:
            raise ValueError("snapshot max_hz must be > 0")
        self._interval = 1.0 / self.max_hz
        self._next_offer_time = 0.0
        self._stream = wp.Stream(self.device, priority=0)
        self._slots = []
        for _ in range(max(int(slot_count), 1)):
            host = wp.empty(36, dtype=wp.float32, device="cpu", pinned=True)
            host_buffer = (ctypes.c_float * 36).from_address(host.ptr)
            self._slots.append(
                {
                    "device": wp.empty(
                        36, dtype=wp.float32, device=self.device
                    ),
                    "host": host,
                    "numpy": np.ctypeslib.as_array(host_buffer),
                    "complete": wp.Event(self.device),
                    "busy": False,
                    "frame_index": -1,
                    "sample_meta": None,
                }
            )

    def poll_completed(self):
        snapshots = []
        for slot in self._slots:
            if not slot["busy"] or not slot["complete"].is_complete:
                continue
            snapshots.append(
                (
                    slot["numpy"].copy(),
                    int(slot["frame_index"]),
                    slot["sample_meta"],
                )
            )
            slot["busy"] = False
            slot["sample_meta"] = None
        return snapshots

    def offer(
        self,
        qpos,
        *,
        frame_index: int,
        sample_meta: dict,
        now: float | None = None,
    ) -> bool:
        now = time.monotonic() if now is None else float(now)
        if now < self._next_offer_time:
            return False
        slot = next((item for item in self._slots if not item["busy"]), None)
        if slot is None:
            return False
        if tuple(qpos.shape) != (36,):
            return False

        control_stream = self.wp.get_stream(self.device)
        self.wp.copy(slot["device"], qpos, stream=control_stream)
        source_ready = control_stream.record_event()
        self._stream.wait_event(source_ready)
        self.wp.copy(slot["host"], slot["device"], stream=self._stream)
        self._stream.record_event(slot["complete"])
        slot["busy"] = True
        slot["frame_index"] = int(frame_index)
        slot["sample_meta"] = dict(sample_meta)
        self._next_offer_time = now + self._interval
        return True


class PicoBodyPoseReader:
    """Keep only the newest XRoboToolkit body frame in a background thread."""

    def __init__(self, *, logger, sdk=None) -> None:
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk
            except ImportError as exc:
                raise RuntimeError(
                    "reference_source=pico_local requires xrobotoolkit_sdk "
                    "in the Orin deployment environment"
                ) from exc
        self.logger = logger
        self.sdk = sdk
        self._stop_event = threading.Event()
        self._thread = None
        self._latest = None
        self._last_stamp_ns = None
        self._fps_ema = 0.0
        self._read_errors = 0
        self._sdk_initialized = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.sdk.is_body_data_available():
                    self._stop_event.wait(0.001)
                    continue
                stamp_ns = int(self.sdk.get_time_stamp_ns())
                previous = self._last_stamp_ns
                if previous is not None and stamp_ns == previous:
                    self._stop_event.wait(0.0001)
                    continue
                device_dt = (
                    (stamp_ns - previous) * 1.0e-9 if previous is not None else 0.0
                )
                if device_dt > 0.0:
                    fps = 1.0 / device_dt
                    self._fps_ema = (
                        fps
                        if self._fps_ema == 0.0
                        else 0.9 * self._fps_ema + 0.1 * fps
                    )
                body_poses = np.asarray(
                    self.sdk.get_body_joints_pose(), dtype=np.float32
                )
                if body_poses.shape != (24, 7):
                    raise ValueError(
                        f"XRoboToolkit body_poses must be (24, 7), got {body_poses.shape}"
                    )
                self._last_stamp_ns = stamp_ns
                self._latest = {
                    "body_poses": np.ascontiguousarray(body_poses),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": float(device_dt),
                    "fps": float(self._fps_ema),
                }
            except Exception as exc:
                self._read_errors += 1
                if self._read_errors == 1 or self._read_errors % 100 == 0:
                    self.logger.error(f"[Pico] body reader error: {exc}")
                self._stop_event.wait(0.01)

    def latest(self):
        return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        init = getattr(self.sdk, "init", None)
        if callable(init):
            init()
            self._sdk_initialized = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pico-body-reader",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("[Pico] local XRoboToolkit body reader started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sdk_initialized:
            close = getattr(self.sdk, "close", None)
            if callable(close):
                close()
            self._sdk_initialized = False
            self._thread = None


class LocalPicoRetargetSource:
    """Run one HoloRetarget step from the newest Pico pose per policy tick."""

    def __init__(
        self,
        *,
        logger,
        asset_root: str = "",
        max_source_age: float = 0.6,
        sdk=None,
        retargeter=None,
    ) -> None:
        if retargeter is None:
            from holoretarget import HoloRetargetConfig, HoloRetargeter

            retargeter = HoloRetargeter(
                HoloRetargetConfig(asset_root=str(asset_root or ""))
            )
        self.logger = logger
        self.retargeter = retargeter
        self.reader = PicoBodyPoseReader(logger=logger, sdk=sdk)
        self.max_source_age = float(max_source_age)
        self.frame_index = 0
        self.last_sample = None
        self._stale_count = 0
        self._retarget_errors = 0

    def start(self) -> None:
        self.reader.start()
        self.logger.info("[Retarget] local Pico -> HoloRetarget source ready")

    def step(self):
        sample = self.reader.latest()
        if sample is None:
            return None
        age = time.monotonic() - float(sample["timestamp_monotonic"])
        if age > self.max_source_age:
            self._stale_count += 1
            if self._stale_count == 1 or self._stale_count % 50 == 0:
                self.logger.warn(
                    f"[Pico] source stale: age={age * 1000.0:.1f}ms > "
                    f"{self.max_source_age * 1000.0:.1f}ms"
                )
            return None
        self._stale_count = 0
        try:
            qpos = self.retargeter.retarget_qpos_device_from_body_poses(
                sample["body_poses"]
            )
        except Exception as exc:
            self._retarget_errors += 1
            if self._retarget_errors == 1 or self._retarget_errors % 50 == 0:
                self.logger.error(f"[Retarget] local step failed: {exc}")
            return None
        self._retarget_errors = 0
        result = (qpos, self.frame_index, sample)
        self.frame_index += 1
        self.last_sample = sample
        return result

    def stop(self) -> None:
        self.reader.stop()


__all__ = [
    "AsyncDeviceQposSnapshotter",
    "LocalPicoRetargetSource",
    "PicoBodyPoseReader",
]
