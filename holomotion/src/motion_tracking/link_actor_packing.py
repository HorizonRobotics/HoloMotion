"""Canonical term-major masked-link actor packing.

This is the exported schema, distinct from LinkObservation.flat_command's
frame-major layout. With F timestamps: current=108+F, future=253F/127F,
with/without error terms. Proprioception is unchanged.
"""

import torch

from holomotion.src.motion_tracking.link_observation import LinkObservation

LINK_CURRENT_TERMS = (
    "actor_projected_gravity",
    "actor_rel_robot_root_ang_vel",
    "actor_dof_pos",
    "actor_dof_vel",
    "actor_last_action",
    "actor_link_reference_height",
    "actor_link_link_mask",
    "actor_link_time_valid",
)
LINK_FUTURE_TERMS = (
    "actor_link_target_pos",
    "actor_link_position_error",
    "actor_link_target_rot6d",
    "actor_link_rotation_error6d",
    "actor_link_time_offsets",
)
LINK_ERROR_TERMS = (
    "actor_link_position_error",
    "actor_link_rotation_error6d",
)


def link_future_terms(
    include_error_terms=True, include_future_link_mask=False
):
    """Select future terms; defaults preserve existing checkpoints."""
    if not isinstance(include_error_terms, bool):
        raise ValueError("include_error_terms must be a boolean")
    if not isinstance(include_future_link_mask, bool):
        raise ValueError("include_future_link_mask must be a boolean")
    terms = [
        name
        for name in LINK_FUTURE_TERMS
        if include_error_terms or name not in LINK_ERROR_TERMS
    ]
    if include_future_link_mask:
        terms.insert(-1, "actor_link_effective_mask")
    return tuple(terms)


def link_actor_layout(link_config, obs_schema=None):
    """Derive atomic shapes and validate the saved term order."""
    include = link_config.get("include_error_terms", True)
    include_future_link_mask = link_config.get(
        "include_future_link_mask", False
    )
    future = link_future_terms(include, include_future_link_mask)
    links = len(link_config["link_names"])
    frames = len(link_config["frame_offsets"])
    offsets = list(link_config["frame_offsets"])
    if (
        not offsets
        or any(type(value) is not int or value < 1 for value in offsets)
        or len(set(offsets)) != frames
    ):
        raise ValueError("frame_offsets must be distinct positive integers")
    shapes = dict(
        zip(
            LINK_CURRENT_TERMS,
            [(3,), (3,), (29,), (29,), (29,), (1,), (links,), (frames,)],
            strict=True,
        )
    )
    widths = dict(
        zip(
            LINK_FUTURE_TERMS,
            [links * 3, links * 3, links * 6, links * 6, 1],
            strict=True,
        )
    )
    widths["actor_link_effective_mask"] = links
    shapes.update({name: (frames, widths[name]) for name in future})
    names = LINK_CURRENT_TERMS + future
    if obs_schema is not None:
        groups = list(obs_schema.values())
        saved = tuple(
            name.split("/")[-1] for group in groups for name in group["terms"]
        )
        if saved != names or len(groups) != 2:
            raise ValueError(
                "Saved actor schema order disagrees with include_error_terms"
            )
        if [int(group["seq_len"]) for group in groups] != [1, frames]:
            raise ValueError(
                "Saved actor schema sequence lengths disagree with "
                "frame_offsets"
            )
    import math

    return {
        "reference_horizon": max(offsets),
        "include_error_terms": include,
        "include_future_link_mask": include_future_link_mask,
        "atomic_terms": names,
        "future_terms": future,
        "term_shapes": shapes,
        "current_dim": sum(math.prod(shapes[n]) for n in LINK_CURRENT_TERMS),
        "future_dim": sum(math.prod(shapes[n]) for n in future),
        "actor_dim": sum(math.prod(shapes[n]) for n in names),
    }


def link_actor_atomic_terms(
    command: LinkObservation,
    include_error_terms=True,
    include_future_link_mask=False,
) -> dict[str, torch.Tensor]:
    """Return atomic tensors with their batch axis retained."""
    terms = {
        "actor_link_reference_height": command.reference_height,
        "actor_link_link_mask": command.link_mask,
        "actor_link_time_valid": command.time_valid,
        "actor_link_target_pos": command.target_pos.flatten(start_dim=2),
        "actor_link_target_rot6d": command.target_rot6d.flatten(start_dim=2),
        "actor_link_time_offsets": command.time_offsets,
    }
    if include_future_link_mask:
        terms["actor_link_effective_mask"] = command.effective_mask
    if include_error_terms:
        if command.position_error is None or command.rotation_error6d is None:
            raise ValueError(
                "Error-enabled packing requires computed error tensors"
            )
        terms["actor_link_position_error"] = command.position_error.flatten(
            start_dim=2
        )
        terms["actor_link_rotation_error6d"] = (
            command.rotation_error6d.flatten(start_dim=2)
        )
    return terms


def pack_link_actor_current(
    proprioception: torch.Tensor, command: LinkObservation
) -> torch.Tensor:
    """Concatenate unchanged 93D proprio/action with height and validity."""
    if proprioception.ndim != 2 or proprioception.shape[-1] != 93:
        raise ValueError("Expected proprioception [B,93] for the 29-DoF actor")
    return torch.cat(
        (
            proprioception,
            command.reference_height,
            command.link_mask,
            command.time_valid,
        ),
        dim=-1,
    )


def pack_link_actor_future(
    command: LinkObservation,
    include_error_terms=True,
    include_future_link_mask=False,
) -> torch.Tensor:
    """Flatten each future term across time BEFORE concatenating terms."""
    terms = link_actor_atomic_terms(
        command, include_error_terms, include_future_link_mask
    )
    return torch.cat(
        [
            terms[name].flatten(start_dim=1)
            for name in link_future_terms(
                include_error_terms, include_future_link_mask
            )
        ],
        dim=-1,
    )


def pack_link_actor_observation(
    proprioception: torch.Tensor,
    command: LinkObservation,
    include_error_terms=True,
    include_future_link_mask=False,
) -> torch.Tensor:
    """Return current followed by future, matching the saved actor schema."""
    return torch.cat(
        (
            pack_link_actor_current(proprioception, command),
            pack_link_actor_future(
                command, include_error_terms, include_future_link_mask
            ),
        ),
        dim=-1,
    )
