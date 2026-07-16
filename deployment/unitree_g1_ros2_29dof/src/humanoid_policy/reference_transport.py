"""ZMQ transport and buffering helpers for reference qpos packets."""

from __future__ import annotations

from collections import deque
import json
import threading
import time

import numpy as np

from humanoid_policy.cpu_affinity import set_thread_cpu_affinity


HEADER_SIZE = 1280
DEFAULT_ZMQ_TOPIC = b"reference_qpos"
_DTYPE_BY_NAME = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "bool": np.bool_,
}


def pack_numpy_message(
    payload: dict,
    topic: bytes = DEFAULT_ZMQ_TOPIC,
    version: int = 1,
) -> bytes:
    """Pack arrays using the existing reference_qpos wire protocol."""
    fields = []
    binary_data = []
    dtype_names = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(np.uint8): "u8",
        np.dtype(np.bool_): "bool",
    }
    for key, raw_value in payload.items():
        if not isinstance(raw_value, np.ndarray):
            continue
        value = raw_value
        dtype_name = dtype_names.get(value.dtype)
        if dtype_name is None:
            value = value.astype(np.float32)
            dtype_name = "f32"
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        fields.append(
            {"name": key, "dtype": dtype_name, "shape": list(value.shape)}
        )
        binary_data.append(value.tobytes())

    header = {"v": int(version), "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    return topic + header_bytes.ljust(HEADER_SIZE, b"\x00") + b"".join(binary_data)


def decode_zmq_topic(topic_value) -> bytes:
    if isinstance(topic_value, bytes):
        return topic_value
    return str(topic_value).encode("utf-8")


def unpack_numpy_message(
    packet: bytes, expected_topic: bytes | None = None
) -> dict:
    if expected_topic is not None:
        if not packet.startswith(expected_topic):
            raise ValueError("ZMQ packet topic prefix mismatch")
        packet = packet[len(expected_topic) :]

    if len(packet) < HEADER_SIZE:
        raise ValueError(
            f"ZMQ packet too short: {len(packet)} < {HEADER_SIZE}"
        )

    header_bytes = packet[:HEADER_SIZE].rstrip(b"\x00")
    if not header_bytes:
        raise ValueError("ZMQ packet has empty header")
    header = json.loads(header_bytes.decode("utf-8"))

    payload = memoryview(packet)[HEADER_SIZE:]
    result = {}
    offset = 0
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        shape = tuple(int(x) for x in field.get("shape", []))
        if dtype_name not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype in ZMQ packet: {dtype_name}")

        dtype = np.dtype(_DTYPE_BY_NAME[dtype_name]).newbyteorder("<")
        count = int(np.prod(shape, dtype=np.int64)) if len(shape) > 0 else 1
        nbytes = count * dtype.itemsize
        end = offset + nbytes
        if end > len(payload):
            raise ValueError(
                f"ZMQ packet field '{name}' exceeds payload size: "
                f"end={end}, payload={len(payload)}"
            )
        arr = np.frombuffer(payload[offset:end], dtype=dtype, count=count)
        if len(shape) > 0:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())
        result[name] = np.array(arr, copy=True)
        offset = end
    return result


class ReferenceBuffer:
    """Thread-safe buffer for delayed reference qpos access."""

    def __init__(self, max_queue_size: int = 20):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp = None
        self._sender_timestamp = None
        self._frame_index = None
        self._sequence = 0
        self._data_queue = deque(maxlen=max_queue_size)
        self._timestamp_queue = deque(maxlen=max_queue_size)
        self._sender_timestamp_queue = deque(maxlen=max_queue_size)
        self._frame_index_queue = deque(maxlen=max_queue_size)
        self._sequence_queue = deque(maxlen=max_queue_size)

    def set(
        self,
        arr: np.ndarray,
        sender_timestamp: float | None = None,
        frame_index: int | None = None,
    ):
        with self._lock:
            current_time = time.time()
            arr_copy = np.asarray(arr, dtype=np.float32).copy()
            self._data = arr_copy
            self._timestamp = current_time
            self._sender_timestamp = sender_timestamp
            self._frame_index = frame_index
            self._sequence += 1
            self._data_queue.append(arr_copy)
            self._timestamp_queue.append(current_time)
            self._sender_timestamp_queue.append(sender_timestamp)
            self._frame_index_queue.append(frame_index)
            self._sequence_queue.append(self._sequence)

    def get_with_age_and_delay(
        self, max_age: float = 0.1, delay_steps: int = 0
    ):
        """Return a delayed frame and report whether it is stale."""
        with self._lock:
            if len(self._data_queue) == 0:
                if self._data is None or self._timestamp is None:
                    return None, None, True, None, None, None
                current_time = time.time()
                age = current_time - self._timestamp
                return (
                    self._data,
                    self._timestamp,
                    age > max_age,
                    self._frame_index,
                    self._sender_timestamp,
                    self._sequence,
                )

            if delay_steps < 0:
                delay_steps = 0
            idx = len(self._data_queue) - 1 - delay_steps
            if idx < 0:
                idx = 0

            data = self._data_queue[idx]
            timestamp = self._timestamp_queue[idx]
            frame_index = self._frame_index_queue[idx]
            sender_timestamp = self._sender_timestamp_queue[idx]
            sequence = self._sequence_queue[idx]

        current_time = time.time()
        age = current_time - timestamp
        return (
            data,
            timestamp,
            age > max_age,
            frame_index,
            sender_timestamp,
            sequence,
        )

    def get_queue_stats(self):
        with self._lock:
            if len(self._data_queue) < 2:
                return {
                    "queue_size": len(self._data_queue),
                    "avg_interval": None,
                }
            intervals = []
            for index in range(1, len(self._timestamp_queue)):
                interval = (
                    self._timestamp_queue[index]
                    - self._timestamp_queue[index - 1]
                )
                intervals.append(interval)
            avg_interval = float(np.mean(intervals)) if intervals else None
            arrival_freq = (
                1.0 / avg_interval
                if avg_interval and avg_interval > 0
                else None
            )
            return {
                "queue_size": len(self._data_queue),
                "avg_interval": avg_interval,
                "arrival_freq": arrival_freq,
                "expected_freq": arrival_freq,
            }


class ZmqReferenceSubscriber:
    """Background ZMQ SUB receiver for reference qpos packets."""

    def __init__(
        self,
        uri: str,
        topic: bytes,
        buffer: ReferenceBuffer,
        logger,
        mode: str = "connect",
        cpu_affinity=None,
        conflate: bool = True,
    ):
        self.uri = uri
        self.topic = topic
        self.buffer = buffer
        self.logger = logger
        self.mode = str(mode).strip().lower()
        self.cpu_affinity = cpu_affinity or []
        self.conflate = bool(conflate)

        self._thread = None
        self._stop_event = threading.Event()
        self._context = None
        self._socket = None
        self._poller = None
        self._recv_count = 0

    def _process_packet(self, packet: bytes):
        payload = unpack_numpy_message(packet, expected_topic=self.topic)
        reference_qpos = payload.get("reference_qpos", None)
        if reference_qpos is None:
            raise ValueError("ZMQ packet missing reference_qpos field")

        frame_index = payload.get("frame_index", None)
        if frame_index is not None:
            frame_index = int(np.asarray(frame_index).reshape(-1)[0])

        sender_timestamp = payload.get("timestamp_realtime", None)
        if sender_timestamp is not None:
            sender_timestamp = float(
                np.asarray(sender_timestamp).reshape(-1)[0]
            )

        self.buffer.set(
            np.asarray(reference_qpos, dtype=np.float32),
            sender_timestamp=sender_timestamp,
            frame_index=frame_index,
        )
        self._recv_count += 1
        if self._recv_count == 1:
            self.logger.info(
                f"[ZMQ] first reference qpos packet received from {self.uri}, "
                f"topic={self.topic.decode('utf-8', errors='ignore')}"
            )

    def _run(self):
        import zmq

        if self.cpu_affinity and set_thread_cpu_affinity(self.cpu_affinity):
            self.logger.info(
                f"[ZMQ] subscriber thread pinned to CPUs {self.cpu_affinity}"
            )

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        if self.conflate and hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1)

        if self.mode == "bind":
            self._socket.bind(self.uri)
        elif self.mode == "connect":
            self._socket.connect(self.uri)
        else:
            raise ValueError("reference_zmq_mode must be 'bind' or 'connect'")

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self.logger.info(
            f"[ZMQ] reference subscriber ready: mode={self.mode}, uri={self.uri}, "
            f"topic={self.topic.decode('utf-8', errors='ignore')}, "
            f"conflate={self.conflate}"
        )

        try:
            while not self._stop_event.is_set():
                events = dict(self._poller.poll(50))
                if self._socket not in events:
                    continue
                try:
                    packet = self._socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                self._process_packet(packet)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.logger.error(f"[ZMQ] subscriber loop failed: {exc}")
        finally:
            try:
                if self._poller is not None and self._socket is not None:
                    self._poller.unregister(self._socket)
            except Exception:
                pass
            try:
                if self._socket is not None:
                    self._socket.close(0)
            except Exception:
                pass
            try:
                if self._context is not None:
                    self._context.term()
            except Exception:
                pass
            self._socket = None
            self._context = None
            self._poller = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.logger.info("[ZMQ] subscriber thread started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.logger.info("[ZMQ] subscriber thread stopped")


class BestEffortReferencePublisher:
    """Rate-limited telemetry publisher that never blocks the control thread."""

    def __init__(
        self,
        *,
        uri: str,
        topic: bytes,
        logger,
        publish_hz: float = 50.0,
        mode: str = "bind",
        cpu_affinity=None,
        deadline_pause_sec: float = 5.0,
    ) -> None:
        self.uri = str(uri)
        self.topic = bytes(topic)
        self.logger = logger
        self.publish_hz = float(publish_hz)
        self.mode = str(mode).strip().lower()
        self.cpu_affinity = cpu_affinity or []
        self.deadline_pause_sec = float(deadline_pause_sec)
        if self.publish_hz <= 0.0:
            raise ValueError("telemetry publish_hz must be > 0")
        if self.mode not in {"bind", "connect"}:
            raise ValueError("telemetry mode must be 'bind' or 'connect'")

        self._thread = None
        self._stop_event = threading.Event()
        self._latest = None
        self._sequence = 0
        self._pause_until = 0.0
        self._sent = 0
        self._dropped = 0
        self._failed = False

    def submit(
        self,
        reference_qpos: np.ndarray,
        *,
        frame_index: int,
        sample_meta: dict | None = None,
    ) -> None:
        """Replace the pending snapshot; older unsent snapshots are dropped."""
        qpos = np.asarray(reference_qpos, dtype=np.float32)
        if qpos.shape != (36,) or not np.isfinite(qpos).all():
            self._dropped += 1
            return
        meta = sample_meta or {}
        now_realtime = time.time()
        now_monotonic = time.monotonic()
        snapshot = {
            "reference_qpos": qpos.copy(),
            "frame_index": np.asarray([frame_index], dtype=np.int64),
            "timestamp_realtime": np.asarray([now_realtime], dtype=np.float64),
            "timestamp_monotonic": np.asarray([now_monotonic], dtype=np.float64),
            "timestamp_ns": np.asarray([time.time_ns()], dtype=np.int64),
            "source_timestamp_realtime": np.asarray(
                [meta.get("timestamp_realtime", now_realtime)], dtype=np.float64
            ),
            "source_timestamp_monotonic": np.asarray(
                [meta.get("timestamp_monotonic", now_monotonic)], dtype=np.float64
            ),
            "source_timestamp_ns": np.asarray(
                [meta.get("timestamp_ns", 0)], dtype=np.int64
            ),
            "pico_dt": np.asarray([meta.get("dt", 0.0)], dtype=np.float32),
            "pico_fps": np.asarray([meta.get("fps", 0.0)], dtype=np.float32),
        }
        if self._latest is not None:
            self._dropped += 1
        self._sequence += 1
        self._latest = (self._sequence, snapshot)

    def note_deadline_pressure(self) -> None:
        """Pause telemetry after a slow control tick without blocking that tick."""
        self._pause_until = max(
            self._pause_until,
            time.monotonic() + self.deadline_pause_sec,
        )

    def _run(self) -> None:
        import zmq

        if self.cpu_affinity and set_thread_cpu_affinity(self.cpu_affinity):
            self.logger.info(
                f"[Telemetry] publisher thread pinned to CPUs {self.cpu_affinity}"
            )
        context = None
        socket = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.LINGER, 0)
            if hasattr(zmq, "CONFLATE"):
                socket.setsockopt(zmq.CONFLATE, 1)
            if self.mode == "bind":
                socket.bind(self.uri)
            else:
                socket.connect(self.uri)
            self.logger.info(
                f"[Telemetry] best-effort reference_qpos ready: "
                f"mode={self.mode}, uri={self.uri}, max_hz={self.publish_hz:.1f}"
            )

            interval = 1.0 / self.publish_hz
            next_send = time.monotonic()
            sent_sequence = 0
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < self._pause_until or now < next_send:
                    self._stop_event.wait(min(0.02, max(0.0, next_send - now)))
                    continue
                latest = self._latest
                if latest is None or latest[0] == sent_sequence:
                    self._stop_event.wait(min(0.02, interval))
                    continue
                sequence, payload = latest
                try:
                    packet = pack_numpy_message(payload, topic=self.topic)
                    socket.send(packet, flags=zmq.NOBLOCK)
                    self._sent += 1
                    sent_sequence = sequence
                except zmq.Again:
                    self._dropped += 1
                    sent_sequence = sequence
                if self._latest is latest:
                    self._latest = None
                next_send = max(next_send + interval, time.monotonic())
        except Exception as exc:
            self._failed = True
            if not self._stop_event.is_set():
                self.logger.error(
                    f"[Telemetry] disabled after publisher failure: {exc}"
                )
        finally:
            if socket is not None:
                try:
                    socket.close(0)
                except Exception:
                    pass
            if context is not None:
                try:
                    context.term()
                except Exception:
                    pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="reference-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
