import unittest

from humanoid_policy.local_retarget import PicoBodyPoseReader


class _Logger:
    def info(self, message):
        del message

    def error(self, message):
        raise AssertionError(message)


class _Sdk:
    def __init__(self):
        self.initialized = False
        self.closed = False

    def init(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def is_body_data_available(self):
        return False


class PicoBodyPoseReaderTest(unittest.TestCase):
    def test_reader_owns_sdk_lifecycle(self):
        sdk = _Sdk()
        reader = PicoBodyPoseReader(logger=_Logger(), sdk=sdk)

        reader.start()
        self.assertTrue(sdk.initialized)
        self.assertFalse(sdk.closed)

        reader.stop()
        self.assertTrue(sdk.closed)
        self.assertIsNone(reader._thread)


if __name__ == "__main__":
    unittest.main()
