"""Versioned development state layout; these are not official G1 joint indices."""

from __future__ import annotations

JOINT_COUNT = 29
ROBOT_WIDTH = 97
STATE_WIDTH = 194


def _robot(offset: int) -> dict:
    return {
        "offset": offset,
        "fields": {
            "root_position": {"offset": offset, "size": 3, "units": "m"},
            "root_quaternion": {
                "offset": offset + 3,
                "size": 4,
                "units": "unit quaternion",
            },
            "root_velocity": {"offset": offset + 7, "size": 3, "units": "m/s"},
            "joint_position": {
                "offset": offset + 10,
                "size": 29,
                "units": "rad",
            },
            "joint_velocity": {
                "offset": offset + 39,
                "size": 29,
                "units": "rad/s",
            },
            "control": {
                "offset": offset + 68,
                "size": 29,
                "units": "rad target",
            },
        },
    }


STATE_SCHEMA = {
    "id": "mock-boxing-v1",
    "version": 1,
    "width": STATE_WIDTH,
    "coordinate_system": "right-handed-z-up",
    "quaternion_order": "wxyz",
    "joint_names": [f"dev_joint_{i:02d}" for i in range(JOINT_COUNT)],
    "robots": {"red": _robot(0), "blue": _robot(ROBOT_WIDTH)},
    "description": "Development visualization state. Not G1 anatomy, MuJoCo integration state, or the official planner protocol.",
}
