"""Exercise real flock contention without a SDK, GPU or robot."""

import fcntl
import mmap
import tempfile
import unittest
from unittest.mock import patch

from humanoid_policy import local_retarget_process as module


class Worker:
    alive = True

    def poll(self):
        return None if self.alive else 1


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()
        self.file.truncate(module.SNAPSHOT_BYTES)
        self.writer = open(self.file.name, "r+b")
        self.source = module.LocalPicoRetargetProcessSource.__new__(
            module.LocalPicoRetargetProcessSource
        )
        self.source._fd = self.file.fileno()
        self.source._map = mmap.mmap(self.source._fd, module.SNAPSHOT_BYTES)
        self.source._proc = Worker()
        self.source._last_ready_sample = None
        self.source.max_source_age = 0.6
        self.clock = patch.object(module.time, "monotonic", return_value=100.0)
        self.now = self.clock.start()
        self.header(frame=1, stamp=99.9)

    def tearDown(self):
        self.clock.stop()
        self.source._map.close()
        self.writer.close()
        self.file.close()

    def header(self, frame, stamp):
        self.source._map[: module.HEADER.size] = module.HEADER.pack(
            2, 1, frame, 0, 0, stamp, 0, 0
        )

    def lock(self):
        fcntl.flock(self.writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_fresh_cached_reference_survives_real_writer_lock(self):
        self.assertTrue(self.source.ready)
        self.lock()
        for _ in range(1000):
            self.assertTrue(self.source.ready)
        self.assertIsNone(self.source.step())

    def test_locked_cache_expires_without_refresh(self):
        self.assertTrue(self.source.ready)
        self.lock()
        self.now.return_value = 100.501
        self.assertFalse(self.source.ready)

    def test_no_previous_snapshot_does_not_qualify(self):
        self.lock()
        self.assertFalse(self.source.ready)

    def test_unlocked_stale_reference_fails(self):
        self.header(1, 99.0)
        self.assertFalse(self.source.ready)

    def test_dead_worker_fails_even_with_fresh_cache(self):
        self.assertTrue(self.source.ready)
        self.source._proc.alive = False
        self.lock()
        self.assertFalse(self.source.ready)

    def test_invalid_snapshot_clears_cached_readiness(self):
        self.assertTrue(self.source.ready)
        self.header(-1, 99.9)
        self.assertFalse(self.source.ready)
        self.lock()
        self.assertFalse(self.source.ready)

    def test_new_snapshot_resumes_readiness(self):
        self.header(1, 99.0)
        self.assertFalse(self.source.ready)
        self.header(2, 99.95)
        self.assertTrue(self.source.ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
