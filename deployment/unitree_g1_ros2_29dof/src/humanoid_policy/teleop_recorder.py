"""Best-effort CPU snapshots; no disk I/O or waiting in the policy thread."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class AsyncPolicySnapshotter:
    """Freeze CUDA inputs privately, then copy to pinned CPU without waiting.

    D2D copies are ordered on the producer stream before next-tick writes.
    A private D2H stream waits on that event. Slots are not reused until their
    completion event is observed and independent CPU copies have been made.
    """

    def __init__(self, *, device, shapes, slot_count=2):
        import warp as wp
        self.wp = wp
        self.device = wp.get_device(device)
        self._stream = wp.Stream(self.device, priority=0)
        self._slots = []
        for _ in range(slot_count):
            arrays = {}
            for name, shape in shapes.items():
                host = wp.empty(shape, dtype=wp.float32, device="cpu", pinned=True)
                view = np.ctypeslib.as_array(
                    (ctypes.c_float * int(np.prod(shape))).from_address(host.ptr)
                ).reshape(shape)
                arrays[name] = {
                    "device": wp.empty(shape, dtype=wp.float32, device=self.device),
                    "host": host, "numpy": view,
                }
            self._slots.append({
                "arrays": arrays, "complete": wp.Event(self.device),
                "busy": False, "row": None,
            })

    def offer(self, sources, row):
        slot = next((item for item in self._slots if not item["busy"]), None)
        if slot is None:
            return False
        for name, buffers in slot["arrays"].items():
            if tuple(sources[name].shape) != tuple(buffers["device"].shape):
                raise ValueError(f"Recording snapshot shape mismatch: {name}")
        control_stream = self.wp.get_stream(self.device)
        for name, buffers in slot["arrays"].items():
            self.wp.copy(buffers["device"], sources[name], stream=control_stream)
        ready = control_stream.record_event()
        self._stream.wait_event(ready)
        for buffers in slot["arrays"].values():
            self.wp.copy(buffers["host"], buffers["device"], stream=self._stream)
        self._stream.record_event(slot["complete"])
        slot["busy"], slot["row"] = True, row
        return True

    def poll_completed(self):
        rows = []
        for slot in self._slots:
            if not slot["busy"] or not slot["complete"].is_complete:
                continue
            row = slot["row"]
            row.update({name: buffers["numpy"].copy()
                        for name, buffers in slot["arrays"].items()})
            rows.append(row)
            slot["busy"], slot["row"] = False, None
        return sorted(rows, key=lambda row: row["sample_sequence"])

    @property
    def pending(self):
        return any(slot["busy"] for slot in self._slots)


class TeleopRecorder:
    """Bounded, uncompressed NPZ chunks written by one background worker.

    ``submit`` takes ownership of fresh CPU arrays. Completed chunks survive
    interruption; shutdown is bounded and may discard an unfinished tail.
    """

    def __init__(self, directory, *, metadata, logger, queue_size=128, chunk_size=64):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = Path(directory).expanduser() / f"teleop_{stamp}_{os.getpid()}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.logger = logger
        self._queue = queue.Queue(maxsize=queue_size)
        self._chunk_size = chunk_size
        self._stop = threading.Event()
        self.failed = False
        self.accepted = 0
        self.written = 0
        self.dropped = 0
        self.budget_skipped = 0
        self.attempted = 0
        self.snapshot_dropped = 0
        self.snapshot_tail_dropped = 0
        self.first_written_sequence = None
        self.last_written_sequence = None
        self.sequence_gaps = 0
        self._metadata = dict(metadata)
        self._thread = threading.Thread(target=self._write, name="teleop-recorder", daemon=True)
        self._thread.start()

    def submit(self, reference_qpos, frame_index, sample_meta):
        payload = sample_meta.get("teleop_record")
        if payload is None or self.failed or self._stop.is_set():
            return False
        row = dict(payload)
        row["retarget_qpos"] = reference_qpos
        row["reference_frame_index"] = int(frame_index)
        return self.submit_row(row)

    def submit_row(self, row):
        if self.failed or self._stop.is_set():
            return False
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1
            return False
        self.accepted += 1
        return True

    def _save_chunk(self, rows, index):
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        temporary = self.directory / f"chunk_{index:06d}.npz.partial"
        destination = self.directory / f"chunk_{index:06d}.npz"
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(destination)
        self.written += len(rows)
        for row in rows:
            sequence = row.get("sample_sequence")
            if sequence is None:
                continue
            if self.first_written_sequence is None:
                self.first_written_sequence = int(sequence)
            if self.last_written_sequence is not None:
                self.sequence_gaps += max(int(sequence) - self.last_written_sequence - 1, 0)
            self.last_written_sequence = int(sequence)

    def _write(self):
        rows = []
        index = 0
        try:
            (self.directory / "metadata.json").write_text(
                json.dumps(self._metadata, indent=2), encoding="utf-8"
            )
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    rows.append(self._queue.get(timeout=0.1))
                except queue.Empty:
                    continue
                if len(rows) >= self._chunk_size:
                    self._save_chunk(rows, index)
                    index += 1
                    rows = []
            if rows:
                self._save_chunk(rows, index)
        except Exception as exc:
            self.failed = True
            self.logger.error(f"[Recording] disabled after writer failure: {exc}")
        finally:
            try:
                (self.directory / "status.json").write_text(json.dumps({
                    "accepted": self.accepted, "written": self.written,
                    "queue_dropped": self.dropped, "budget_skipped": self.budget_skipped,
                    "attempted": self.attempted,
                    "snapshot_dropped": self.snapshot_dropped,
                    "snapshot_tail_dropped": self.snapshot_tail_dropped,
                    "first_written_sequence": self.first_written_sequence,
                    "last_written_sequence": self.last_written_sequence,
                    "sequence_gaps_between_written_rows": self.sequence_gaps,
                    "failed": self.failed, "closed_realtime": time.time(),
                }, indent=2), encoding="utf-8")
            except OSError:
                pass

    def close(self, timeout=1.0):
        self._stop.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
