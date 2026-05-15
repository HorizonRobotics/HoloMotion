"""ZMQ transport and buffering helpers for latest_obs packets."""

from __future__ import annotations

from collections import deque
import json
import threading
import time

import numpy as np

from humanoid_policy.cpu_affinity import set_thread_cpu_affinity


HEADER_SIZE = 1280
DEFAULT_ZMQ_TOPIC = b"obs65"
_DTYPE_BY_NAME = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "bool": np.bool_,
}


def decode_zmq_topic(topic_value) -> bytes:
    if isinstance(topic_value, bytes):
        return topic_value
    return str(topic_value).encode("utf-8")


def unpack_numpy_message(packet: bytes, expected_topic: bytes | None = None) -> dict:
    if expected_topic is not None:
        if not packet.startswith(expected_topic):
            raise ValueError("ZMQ packet topic prefix mismatch")
        packet = packet[len(expected_topic) :]

    if len(packet) < HEADER_SIZE:
        raise ValueError(f"ZMQ packet too short: {len(packet)} < {HEADER_SIZE}")

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


class LatestObsBuffer:
    """Thread-safe buffer for delayed latest_obs access."""

    def __init__(self, max_queue_size: int = 20):
        self._lock = threading.Lock()
        self._data = None
        self._timestamp = None
        self._sender_timestamp = None
        self._frame_index = None
        self._data_queue = deque(maxlen=max_queue_size)
        self._timestamp_queue = deque(maxlen=max_queue_size)
        self._sender_timestamp_queue = deque(maxlen=max_queue_size)
        self._frame_index_queue = deque(maxlen=max_queue_size)

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
            self._data_queue.append(arr_copy)
            self._timestamp_queue.append(current_time)
            self._sender_timestamp_queue.append(sender_timestamp)
            self._frame_index_queue.append(frame_index)

    def get_with_age_and_delay(self, max_age: float = 0.1, delay_steps: int = 0):
        """Return a delayed frame and report whether it is stale."""
        with self._lock:
            if len(self._data_queue) == 0:
                if self._data is None or self._timestamp is None:
                    return None, None, True, None, None
                current_time = time.time()
                age = current_time - self._timestamp
                return (
                    self._data,
                    self._timestamp,
                    age > max_age,
                    self._frame_index,
                    self._sender_timestamp,
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

        current_time = time.time()
        age = current_time - timestamp
        return data, timestamp, age > max_age, frame_index, sender_timestamp

    def get_queue_stats(self):
        with self._lock:
            if len(self._data_queue) < 2:
                return {"queue_size": len(self._data_queue), "avg_interval": None}
            intervals = []
            for index in range(1, len(self._timestamp_queue)):
                interval = self._timestamp_queue[index] - self._timestamp_queue[index - 1]
                intervals.append(interval)
            avg_interval = float(np.mean(intervals)) if intervals else None
            return {
                "queue_size": len(self._data_queue),
                "avg_interval": avg_interval,
                "expected_freq": (
                    1.0 / avg_interval if avg_interval and avg_interval > 0 else None
                ),
            }


class ZmqLatestObsSubscriber:
    """Background ZMQ SUB receiver for latest_obs packets."""

    def __init__(
        self,
        uri: str,
        topic: bytes,
        buffer: LatestObsBuffer,
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
        latest_obs = payload.get("latest_obs", None)
        if latest_obs is None:
            raise ValueError("ZMQ packet missing latest_obs field")

        frame_index = payload.get("frame_index", None)
        if frame_index is not None:
            frame_index = int(np.asarray(frame_index).reshape(-1)[0])

        sender_timestamp = payload.get("timestamp_realtime", None)
        if sender_timestamp is not None:
            sender_timestamp = float(np.asarray(sender_timestamp).reshape(-1)[0])

        self.buffer.set(
            np.asarray(latest_obs, dtype=np.float32),
            sender_timestamp=sender_timestamp,
            frame_index=frame_index,
        )
        self._recv_count += 1
        if self._recv_count == 1:
            self.logger.info(
                f"[ZMQ] first latest_obs packet received from {self.uri}, "
                f"topic={self.topic.decode('utf-8', errors='ignore')}"
            )

    def _run(self):
        import zmq

        if self.cpu_affinity and set_thread_cpu_affinity(self.cpu_affinity):
            self.logger.info(f"[ZMQ] subscriber thread pinned to CPUs {self.cpu_affinity}")

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
            raise ValueError("latest_obs_zmq_mode must be 'bind' or 'connect'")

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self.logger.info(
            f"[ZMQ] latest_obs subscriber ready: mode={self.mode}, uri={self.uri}, "
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
