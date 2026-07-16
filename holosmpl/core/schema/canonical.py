from __future__ import annotations

from dataclasses import dataclass


CANONICAL_COORDINATE_FRAME = "canonical_z_up_xyz_m"
CANONICAL_UP_AXIS = "z"
CANONICAL_UNIT = "meter"
WORLD_GRAVITY = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: str
    description: str


CANONICAL_FIELDS = (
    FieldSpec("root_orient", "[T,3]", "Root/global orientation in axis-angle."),
    FieldSpec("pose_body", "[T,D]", "Body pose in axis-angle; D is 63 or 69."),
    FieldSpec("trans", "[T,3]", "Root translation in canonical z-up meters."),
    FieldSpec("betas", "[B]", "Shape beta vector."),
    FieldSpec("gender", "string scalar", "Subject gender if available."),
    FieldSpec("source_fps", "float scalar", "Source frame rate before resampling."),
    FieldSpec("target_fps", "float scalar", "Canonical frame rate after resampling."),
    FieldSpec("metadata", "json scalar", "Clip metadata and provenance."),
)


ALLOWED_BODY_POSE_LAYOUTS = {
    "smplx_21_body": 63,
    "smpl_23_body": 69,
}


def canonical_schema_summary() -> dict[str, object]:
    return {
        "coordinate_frame": CANONICAL_COORDINATE_FRAME,
        "up_axis": CANONICAL_UP_AXIS,
        "unit": CANONICAL_UNIT,
        "world_gravity": list(WORLD_GRAVITY),
        "fields": [field.__dict__ for field in CANONICAL_FIELDS],
        "allowed_body_pose_layouts": dict(ALLOWED_BODY_POSE_LAYOUTS),
    }
