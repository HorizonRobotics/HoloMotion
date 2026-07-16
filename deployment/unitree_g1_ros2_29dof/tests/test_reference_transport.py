import unittest

import numpy as np

from humanoid_policy.reference_transport import BestEffortReferencePublisher
from humanoid_policy.reference_transport import DEFAULT_ZMQ_TOPIC
from humanoid_policy.reference_transport import ReferenceBuffer
from humanoid_policy.reference_transport import pack_numpy_message
from humanoid_policy.reference_transport import unpack_numpy_message


class ReferenceBufferTest(unittest.TestCase):
    def test_telemetry_packet_round_trip_uses_existing_protocol(self):
        expected = {
            "reference_qpos": np.arange(36, dtype=np.float32),
            "frame_index": np.asarray([42], dtype=np.int64),
        }
        packet = pack_numpy_message(expected)
        actual = unpack_numpy_message(
            packet,
            expected_topic=DEFAULT_ZMQ_TOPIC,
        )
        np.testing.assert_array_equal(
            actual["reference_qpos"], expected["reference_qpos"]
        )
        np.testing.assert_array_equal(
            actual["frame_index"], expected["frame_index"]
        )

    def test_telemetry_submit_replaces_pending_snapshot(self):
        class Logger:
            def info(self, message):
                del message

            def error(self, message):
                raise AssertionError(message)

        publisher = BestEffortReferencePublisher(
            uri="tcp://*:6002",
            topic=DEFAULT_ZMQ_TOPIC,
            logger=Logger(),
        )
        publisher.submit(np.zeros(36, dtype=np.float32), frame_index=1)
        publisher.submit(np.ones(36, dtype=np.float32), frame_index=2)

        self.assertEqual(publisher._dropped, 1)
        self.assertEqual(int(publisher._latest[1]["frame_index"][0]), 2)

    def test_sequence_changes_only_when_a_new_packet_arrives(self):
        buffer = ReferenceBuffer()
        buffer.set(np.asarray([1.0], dtype=np.float32), frame_index=10)

        first = buffer.get_with_age_and_delay(max_age=1.0)
        repeated = buffer.get_with_age_and_delay(max_age=1.0)

        self.assertEqual(first[-1], repeated[-1])
        buffer.set(np.asarray([2.0], dtype=np.float32), frame_index=11)
        updated = buffer.get_with_age_and_delay(max_age=1.0)
        self.assertGreater(updated[-1], first[-1])
        self.assertGreater(buffer.get_queue_stats()["arrival_freq"], 0.0)

    def test_delayed_packet_keeps_its_transport_sequence(self):
        buffer = ReferenceBuffer()
        for frame in range(3):
            buffer.set(
                np.asarray([float(frame)], dtype=np.float32),
                frame_index=frame,
            )

        delayed = buffer.get_with_age_and_delay(max_age=1.0, delay_steps=1)

        self.assertEqual(delayed[3], 1)
        self.assertEqual(delayed[-1], 2)


if __name__ == "__main__":
    unittest.main()
