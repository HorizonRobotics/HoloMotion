from __future__ import annotations

import torch
from torch.nn import functional as F

from holoboxing.evaluation.runtime.representation import (
    heading as heading,
    matrix as matrix,
    rot_xz as rot_xz,
    rotate as rotate,
)
from holoboxing.training.tasks.combat.config.sandbag import POINTS
from holomotion.src.motion_tracking.link_observation import (
    quat_apply_wxyz as quat_apply_wxyz,
    quat_conjugate_wxyz as quat_conjugate_wxyz,
)

FIVE_POINT_LINK_NAMES = POINTS


def project_xz(v):
    """X/Z convention, including degenerate and extreme finite columns."""
    raw_x = v[..., :3]
    raw_z = v[..., 3:]
    # Scaling a column preserves its direction and prevents norm/dot overflow.
    raw_x = raw_x / raw_x.abs().amax(-1, keepdim=True).clamp_min(1.0)
    raw_z = raw_z / raw_z.abs().amax(-1, keepdim=True).clamp_min(1.0)
    # A local fallback chart retains tangent gradients at a zero target mean.
    # Constant fallbacks would permanently freeze a zero-initialized C decoder.
    x = torch.where(
        raw_x.norm(dim=-1, keepdim=True) > 1e-8,
        raw_x,
        raw_x + raw_x.new_tensor([1.0, 0.0, 0.0]),
    )
    x = F.normalize(x, dim=-1, eps=1e-8)
    z = raw_z - (raw_z * x).sum(-1, keepdim=True) * x
    # Prefer the canonical Z axis. Picking Y when X is zero creates an
    # unnecessary quarter-turn and can place symmetric targets at a pi saddle.
    canonical_z = x.new_tensor([0.0, 0.0, 1.0]).expand_as(x)
    fallback = torch.where(
        x[..., 2:3].abs() < 0.9,
        canonical_z,
        F.one_hot(x.abs().argmin(-1), 3).to(x.dtype),
    )
    fallback = fallback - (fallback * x).sum(-1, keepdim=True) * x
    z = torch.where(z.norm(dim=-1, keepdim=True) > 1e-6, z, fallback + z)
    z = F.normalize(z, dim=-1, eps=1e-8)
    # Near-parallel columns amplify cancellation error in the subtraction.
    # Rebuild the axes by cross products before exposing a rotation to either
    # the frozen controller or the competition ONNX boundary.
    y = F.normalize(torch.cross(z, x, dim=-1), dim=-1, eps=1e-8)
    z = F.normalize(torch.cross(x, y, dim=-1), dim=-1, eps=1e-8)
    return torch.stack((x, y, z), -1)


def encode_c(position, quaternion, target_position, target_quaternion):
    inverse = matrix(quaternion[:, 0]).transpose(-1, -2)[:, None, None]
    p = rotate(inverse, target_position - position[:, None, :1])
    r = inverse @ matrix(target_quaternion)
    return torch.cat((p, rot_xz(r)), -1).flatten(1)


def decode_c(position, quaternion, values):
    values = values.reshape(-1, 5, 5, 9)
    r = matrix(quaternion[:, 0])[:, None, None]
    return position[:, None, :1] + rotate(r, values[..., :3]), r @ project_xz(
        values[..., 3:]
    )


def encode_output(
    position, quaternion, target_position, target_quaternion, action_dim=225
):
    if action_dim != 225:
        raise ValueError("Only C225 is supported")
    return encode_c(position, quaternion, target_position, target_quaternion)


def decode_output(position, quaternion, values, action_dim=225):
    if action_dim != 225:
        raise ValueError("Only C225 is supported")
    if values.shape != (position.shape[0], action_dim):
        raise ValueError("physical reference shape differs from action_dim")
    return decode_c(position, quaternion, values)


def raw_self(
    position,
    quaternion,
    link_velocity_w,
    omega_b,
    previous_position,
    previous_quaternion,
):
    current = torch.cat((position, rot_xz(matrix(quaternion))), -1).flatten(-2)
    previous = torch.cat(
        (previous_position, rot_xz(matrix(previous_quaternion))), -1
    ).flatten(-2)
    return torch.cat((omega_b, current, link_velocity_w.flatten(-2), previous), -1)


def reframe_c(raw_history, position, quaternion, opponent_position):
    """All ten raw states, including prev45, in the current root heading frame."""
    origin = position[:, 0]
    inverse = heading(quaternion[:, 0]).transpose(-1, -2)[:, None, None]

    def pose(values):
        values = values.reshape(-1, 10, 5, 9)
        p = rotate(inverse, values[..., :3] - origin[:, None, None])
        return torch.cat(
            (
                p,
                rotate(inverse, values[..., 3:6]),
                rotate(inverse, values[..., 6:9]),
            ),
            -1,
        ).flatten(-2)

    history = torch.cat(
        (
            raw_history[..., :3],
            pose(raw_history[..., 3:48]),
            rotate(inverse, raw_history[..., 48:63].reshape(-1, 10, 5, 3)).flatten(-2),
            pose(raw_history[..., 63:]),
        ),
        -1,
    )
    current = history[:, -1]
    full_inverse = matrix(quaternion[:, 0]).transpose(-1, -2)[:, None]
    own = rotate(full_inverse, position - origin[:, None]).flatten(1)
    other = rotate(full_inverse, opponent_position - origin[:, None]).flatten(1)
    return {
        "self": current,
        "task": torch.cat((current, own, other), -1),
        "history": history.flatten(1),
    }


def normalize_quaternion_wxyz(value: torch.Tensor) -> torch.Tensor:
    """Normalize a WXYZ quaternion tensor without hiding non-finite input."""

    if value.shape[-1] != 4:
        raise ValueError("quaternion tensor must end in four values")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quaternion tensor must be finite")
    norm = value.norm(dim=-1, keepdim=True)
    if bool((norm < 1.0e-8).any()):
        raise ValueError("quaternion tensor must have non-zero norm")
    return value / norm
