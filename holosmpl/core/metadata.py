from __future__ import annotations


CANONICAL_COORDINATE_METADATA = {
    "coordinate_frame": "canonical_z_up_xyz_m",
    "up_axis": "z",
    "unit": "meter",
    "world_gravity": [0.0, 0.0, -1.0],
}


def with_canonical_coordinate_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {**metadata, **CANONICAL_COORDINATE_METADATA}
