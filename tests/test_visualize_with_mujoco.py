import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.motion_retargeting.utils.visualize_with_mujoco import (
    _resolve_visualization_arrays,
)


def test_resolve_visualization_arrays_uses_robot_for_pose_and_ref_for_overlay():
    arrays = {
        "robot_dof_pos": np.array([[1.0, 2.0]], dtype=np.float32),
        "robot_global_translation": np.array(
            [[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]], dtype=np.float32
        ),
        "robot_global_rotation_quat": np.array(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        "ref_global_translation": np.array(
            [[[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]], dtype=np.float32
        ),
    }

    resolved = _resolve_visualization_arrays(
        arrays=arrays,
        key_prefix_order=["robot_"],
        draw_ref_body_spheres=True,
        ref_key_prefix_order=["ref_"],
    )

    np.testing.assert_allclose(resolved["dof_pos"], arrays["robot_dof_pos"])
    np.testing.assert_allclose(
        resolved["global_translation"], arrays["robot_global_translation"]
    )
    np.testing.assert_allclose(
        resolved["global_rotation_quat"],
        arrays["robot_global_rotation_quat"],
    )
    np.testing.assert_allclose(
        resolved["ref_body_positions"],
        arrays["ref_global_translation"],
    )
