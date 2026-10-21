"""Bound export rounding in physical units without hiding action errors."""

import numpy as np
import pytest

from holoboxing.training.export import check_export_parity


def pose(angle=0.0):
    points = np.zeros((1, 5, 5, 9), dtype=np.float32)
    points[..., 3] = np.cos(angle)
    points[..., 4] = np.sin(angle)
    points[..., 8] = 1
    return points.reshape(1, 225)


def test_rotation_rounding_near_zero_uses_angle_instead_of_relative_error():
    result = check_export_parity(pose(3e-5), pose())
    assert result["max_rotation_error_rad"] == pytest.approx(3e-5)
    assert result["max_position_error_m"] == 0
    with pytest.raises(ValueError, match="physical tolerances"):
        check_export_parity(pose(2e-4), pose())


def test_translation_and_scaled_rotation_cannot_evade_parity():
    translated = pose()
    translated[0, 0] = 2e-5
    with pytest.raises(ValueError, match="physical tolerances"):
        check_export_parity(translated, pose())
    scaled = pose()
    scaled[0, 3] = 1.001
    with pytest.raises(ValueError, match="physical tolerances"):
        check_export_parity(scaled, pose())


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_outputs_fail_parity(invalid):
    broken = pose()
    broken[0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        check_export_parity(broken, pose())
