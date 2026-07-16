"""Shared motion-tracking policy utilities."""

from .actor_observation import (
    MOTION_ACTOR_CURRENT_TERMS,
    MOTION_ACTOR_FUTURE_TERMS,
    MOTION_ACTOR_TERMS,
    MotionActorObservationInput,
    build_motion_actor_observation_torch,
    derive_motion_actor_terms_torch,
    motion_actor_observation_dim,
)
from .reference_observation import (
    REFERENCE_DOF_DIM,
    REFERENCE_QPOS_DIM,
    ReferenceKinematics,
    derive_reference_kinematics_numpy,
    derive_reference_kinematics_torch,
    pack_reference_qpos,
)

__all__ = [
    "MOTION_ACTOR_CURRENT_TERMS",
    "MOTION_ACTOR_FUTURE_TERMS",
    "MOTION_ACTOR_TERMS",
    "MotionActorObservationInput",
    "REFERENCE_DOF_DIM",
    "REFERENCE_QPOS_DIM",
    "ReferenceKinematics",
    "build_motion_actor_observation_torch",
    "derive_motion_actor_terms_torch",
    "derive_reference_kinematics_numpy",
    "derive_reference_kinematics_torch",
    "motion_actor_observation_dim",
    "pack_reference_qpos",
]
