from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: str
    description: str
    derived: bool = False


FORMAL_FIELDS = (
    FieldSpec("human_pose_aa", "[T,72]", "Root + SMPL 23-body-joint axis-angle pose."),
    FieldSpec("human_shape_beta", "[B]", "Clip-level human shape beta."),
    FieldSpec("human_root_trans", "[T,3]", "Root translation in canonical z-up meters."),
    FieldSpec(
        "human_root_height",
        "[T,1]",
        "Derived root height: human_root_trans[:, 2:3].",
        derived=True,
    ),
    FieldSpec(
        "human_gravity_projection",
        "[T,3]",
        "Derived world gravity expressed in the root frame.",
        derived=True,
    ),
    FieldSpec("metadata", "json scalar", "Clip metadata and provenance."),
)


def formal_schema_summary() -> dict[str, object]:
    return {
        "fields": [field.__dict__ for field in FORMAL_FIELDS],
        "source_of_truth_fields": [
            "human_pose_aa",
            "human_shape_beta",
            "human_root_trans",
            "metadata",
        ],
        "derived_fields": [
            "human_root_height",
            "human_gravity_projection",
        ],
    }
