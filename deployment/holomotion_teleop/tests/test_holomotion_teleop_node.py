import time
import unittest
from collections import defaultdict

import numpy as np

from deployment.holomotion_teleop.holomotion_teleop_node import (
    HoloRetargetTeleopNode,
    QPOS_DIM,
)


class StaticReader:
    def __init__(self, sample):
        self.sample = sample

    def get_latest(self):
        return self.sample


def make_sample(*, age_sec=0.0):
    now_realtime = time.time()
    now_monotonic = time.monotonic()
    return {
        "body_poses_np": np.zeros((24, 7), dtype=np.float32),
        "timestamp_realtime": now_realtime - age_sec,
        "timestamp_monotonic": now_monotonic - age_sec,
        "timestamp_ns": int((now_realtime - age_sec) * 1_000_000_000),
        "dt": 0.02,
        "fps": 50.0,
    }


def make_node(sample):
    node = object.__new__(HoloRetargetTeleopNode)
    node.hz = 50.0
    node.timing_log_every = 1000
    node.reader = StaticReader(sample)
    node.latest_sample = None
    node.latest_body_poses_np = None
    node.last_processed_timestamp_ns = None
    node.tick_count = 0
    node.timing_sums_ms = defaultdict(float)
    node.detail_timing_sums_ms = defaultdict(float)
    node._source_timeout_sec = 0.1
    node._retarget_rate_window_start = time.monotonic()
    node._retarget_completed_in_window = 0
    node._retarget_rate_log_interval_sec = 1000.0
    node.push_count = 0
    node.retarget_count = 0
    node.published = []
    node.info = lambda message: None
    node.error = lambda message: None

    def push(body_poses):
        node.push_count += 1
        node.latest_body_poses_np = np.asarray(body_poses)

    def retarget():
        node.retarget_count += 1
        return np.zeros(QPOS_DIM, dtype=np.float32)

    def publish(reference_qpos, *, sample_meta, body_poses):
        node.published.append((reference_qpos, sample_meta, body_poses))

    node.push_body_poses = push
    node.retarget_latest = retarget
    node._publish = publish
    return node


class HoloRetargetTeleopNodeTest(unittest.TestCase):
    def test_retargets_every_tick_while_holding_latest_source_frame(self):
        sample = make_sample()
        node = make_node(sample)

        node._tick()
        node._tick()

        self.assertEqual(node.push_count, 1)
        self.assertEqual(node.retarget_count, 2)
        self.assertEqual(len(node.published), 2)
        first_meta = node.published[0][1]
        self.assertEqual(first_meta["source_timestamp_ns"], sample["timestamp_ns"])
        self.assertNotEqual(first_meta["timestamp_ns"], sample["timestamp_ns"])

    def test_stale_source_frame_is_not_retargeted_or_published(self):
        node = make_node(make_sample(age_sec=1.0))

        node._tick()

        self.assertEqual(node.push_count, 1)
        self.assertEqual(node.retarget_count, 0)
        self.assertEqual(node.published, [])


if __name__ == "__main__":
    unittest.main()
