"""Shared policy utilities, loaded only when their public API is requested.

Link-only inference must not import the retargeting or legacy actor runtime.
The existing package-level exports remain available through lazy attributes.
"""

from importlib import import_module

_ACTOR_EXPORTS = {
    "MOTION_ACTOR_CURRENT_TERMS",
    "MOTION_ACTOR_FUTURE_TERMS",
    "MOTION_ACTOR_TERMS",
    "MotionActorObservationInput",
    "build_motion_actor_observation_torch",
    "derive_motion_actor_terms_torch",
    "motion_actor_observation_dim",
}
_REFERENCE_EXPORTS = {
    "REFERENCE_DOF_DIM",
    "REFERENCE_QPOS_DIM",
    "ReferenceKinematics",
    "derive_reference_kinematics_numpy",
    "derive_reference_kinematics_torch",
    "pack_reference_qpos",
}


def __getattr__(name):
    if name in _ACTOR_EXPORTS:
        module = import_module(".actor_observation", __name__)
    elif name in _REFERENCE_EXPORTS:
        module = import_module(".reference_observation", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


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
