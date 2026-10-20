"""Report absent CUDA access before inference libraries can abort the process."""

import unittest
from unittest.mock import Mock, patch

from holoboxing.evaluation.runtime.onnx_device import require_cuda_driver


class CudaDriverTest(unittest.TestCase):
    def test_missing_driver_is_a_python_error(self):
        with patch("ctypes.CDLL", side_effect=OSError("driver missing")):
            with self.assertRaisesRegex(RuntimeError, "driver is required"):
                require_cuda_driver(0)

    def test_driver_initialization_failure_does_not_query_devices(self):
        driver = Mock()
        driver.cuInit.return_value = 100
        with patch("ctypes.CDLL", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                require_cuda_driver(0)
        driver.cuDeviceGetCount.assert_not_called()

    def test_no_visible_device_is_a_python_error(self):
        driver = Mock()
        driver.cuInit.return_value = 0
        driver.cuDeviceGetCount.return_value = 0
        with patch("ctypes.CDLL", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                require_cuda_driver(0)

    def test_requested_device_must_be_visible(self):
        driver = Mock()
        driver.cuInit.return_value = 0

        def count(pointer):
            pointer._obj.value = 1
            return 0

        driver.cuDeviceGetCount.side_effect = count
        with patch("ctypes.CDLL", return_value=driver):
            require_cuda_driver(0)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                require_cuda_driver(1)
