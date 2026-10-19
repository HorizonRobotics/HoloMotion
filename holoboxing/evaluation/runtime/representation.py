"""A and C representations of the SAME world-space physical record (wxyz)."""

import torch
from torch.nn import functional as F


def qmul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(
        -1, keepdim=True
    )
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    )
    return torch.cat((w, v), -1)


def qinv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotate(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


def heading(q):
    r = matrix(q)
    angle = torch.atan2(r[..., 1, 0], r[..., 0, 0])
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(
        *angle.shape, 3, 3
    )


def aa_to_q(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat(
        ((angle / 2).cos(), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1
    )


def q_to_aa(q):
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[..., :1])
    return q[..., 1:] * torch.where(
        sine > 1e-7, angle / sine.clamp_min(1e-7), 2.0
    )


def rot_xz(r):
    return torch.cat((r[..., :, 0], r[..., :, 2]), -1)


def project_xz(v):
    """C X/Z convention, with legal rotations even for tiny nonzero X columns."""
    raw_x = v[..., :3]
    # A local fallback chart retains tangent gradients at a zero target mean.
    # Constant fallbacks would permanently freeze a zero-initialized C decoder.
    x = torch.where(
        raw_x.norm(dim=-1, keepdim=True) > 1e-8,
        raw_x,
        raw_x + raw_x.new_tensor([1.0, 0.0, 0.0]),
    )
    x = F.normalize(x, dim=-1, eps=1e-8)
    z = v[..., 3:] - (v[..., 3:] * x).sum(-1, keepdim=True) * x
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
    return torch.stack((x, torch.cross(z, x, dim=-1), z), -1)


def encode_a(position, quaternion, target_position, target_quaternion):
    local = rotate(
        heading(quaternion[:, 0]).transpose(-1, -2)[:, None, None],
        target_position - position[:, None],
    )
    rel = qmul(
        qinv(quaternion)[:, None].expand_as(target_quaternion),
        target_quaternion,
    )
    return torch.cat((local, q_to_aa(rel)), -1).flatten(1)


def decode_a(position, quaternion, values):
    values = values.reshape(-1, 5, 5, 6)
    p = position[:, None] + rotate(
        heading(quaternion[:, 0])[:, None, None], values[..., :3]
    )
    q = qmul(
        quaternion[:, None].expand(*values.shape[:-1], 4),
        aa_to_q(values[..., 3:]),
    )
    return p, matrix(q)


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


# The public output uses the root-relative X/Z rotation representation.
encode_output = encode_c
decode_output = decode_c


def raw_self(
    position,
    quaternion,
    com_velocity_w,
    omega_b,
    previous_position,
    previous_quaternion,
):
    current = torch.cat((position, rot_xz(matrix(quaternion))), -1).flatten(-2)
    previous = torch.cat(
        (previous_position, rot_xz(matrix(previous_quaternion))), -1
    ).flatten(-2)
    return torch.cat(
        (omega_b, current, com_velocity_w.flatten(-2), previous), -1
    )


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
            rotate(
                inverse, raw_history[..., 48:63].reshape(-1, 10, 5, 3)
            ).flatten(-2),
            pose(raw_history[..., 63:]),
        ),
        -1,
    )
    current = history[:, -1]
    full_inverse = matrix(quaternion[:, 0]).transpose(-1, -2)[:, None]
    own = rotate(full_inverse, position - origin[:, None]).flatten(1)
    other = rotate(full_inverse, opponent_position - origin[:, None]).flatten(
        1
    )
    return {
        "self": current,
        "task": torch.cat((current, own, other), -1),
        "history": history.flatten(1),
    }


def physical_errors(p, r, target_p, target_q):
    """Per-row losses plus physical errors; never zero-fill non-finite values."""
    target_r = matrix(target_q)
    pos_loss = F.smooth_l1_loss(
        p / 0.02, target_p / 0.02, reduction="none"
    ).mean((1, 2, 3))
    rot_loss = F.smooth_l1_loss(
        rot_xz(r), rot_xz(target_r), reduction="none"
    ).mean((1, 2, 3))
    relative = r.transpose(-1, -2) @ target_r
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
        -1, 1
    )
    result = {
        "loss": pos_loss + 0.1 * rot_loss,
        "position_loss": pos_loss,
        "rotation_loss": rot_loss,
        "position_cm": (p - target_p).norm(dim=-1).mean((1, 2)) * 100,
        "rotation_deg": torch.rad2deg(torch.acos(cosine.detach())).mean(
            (1, 2)
        ),
        "point_position_cm": (p - target_p).norm(dim=-1).mean(1) * 100,
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite physical prediction or metric")
    return result
