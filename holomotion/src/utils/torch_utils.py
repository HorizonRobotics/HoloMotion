"""Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np
import torch


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(
        x, dtype=dtype, device=device, requires_grad=requires_grad
    )


@torch.compile
def quat_mul(
    a: torch.Tensor, b: torch.Tensor, w_last: bool = True
) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        a (torch.Tensor): (..., 4) quaternion.
        b (torch.Tensor): (..., 4) quaternion.
        w_last (bool): Whether the scalar part w is the last element.
                      If True, format is [x, y, z, w]; if False, format is [w, x, y, z].

    Returns:
        torch.Tensor: (..., 4) quaternion result of a * b.
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        # Format: [x, y, z, w]
        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        # Format: [w, x, y, z]
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(
            q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
        ).squeeze(-1)
        * 2.0
    )
    return a + b + c


# @torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(
            q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
        ).squeeze(-1)
        * 2.0
    )
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float64, n_dims=3):
    """Construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, (
        "the axis dim should be within the vector dimensions"
    )
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q: torch.Tensor) -> tuple:
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        copysign(torch.tensor(np.pi / 2.0, device=sinp.device), sinp),
        torch.asin(sinp),
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# @torch.jit.script
@torch.compile
def torch_random_dir_2(shape, device):
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def quat_to_angle_axis(q):
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, _, _, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def slerp(q0, q1, t):
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratio_a * q0 + ratio_b * q1

    new_q = torch.where(
        torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q
    )
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(
            q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
        ).squeeze(-1)
        * 2.0
    )
    return a + b + c


@torch.jit.script
def calc_heading(q):
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    # this is the x axis heading
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat(q):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


@torch.compile
def axis_angle_from_quat(
    quat: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Compute axis-angle (log map) vector from a quaternion.

    Args:
        quat (torch.Tensor): (..., 4) quaternion. If `w_last` is True, format is [x, y, z, w]; otherwise [w, x, y, z].
        w_last (bool): Whether the scalar part w is the last element.

    Returns:
        torch.Tensor: (..., 3) axis-angle vector (axis * angle), with angle in radians in [0, pi].

    Notes:
        - The quaternion is sign-adjusted to ensure w >= 0 and normalized to unit length for numerical stability.
        - Uses a stable small-angle handling to avoid NaNs and gradient issues.
    """
    # Handle different quaternion formats
    if w_last:
        # Quaternion is [q_x, q_y, q_z, q_w]
        quat_w_orig = quat[..., -1:]
    else:
        # Quaternion is [q_w, q_x, q_y, q_z]
        quat_w_orig = quat[..., 0:1]

    # Normalize quaternion to have w > 0
    quat = quat * (1.0 - 2.0 * (quat_w_orig < 0.0))

    # Ensure unit quaternion for stability
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(
        1.0e-9
    )

    # Recompute quat_xyz and quat_w after potential sign flip
    if w_last:
        quat_w = quat[..., -1:]
        quat_xyz = quat[..., :3]
    else:
        quat_w = quat[..., 0:1]
        quat_xyz = quat[..., 1:4]

    mag = torch.linalg.norm(quat_xyz, dim=-1)
    half_angle = torch.atan2(mag, quat_w.squeeze(-1))
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    use_taylor = angle.abs() <= 1.0e-6
    # To prevent NaN gradients with torch.where, we compute both branches and blend
    # based on the condition.
    # See: https://pytorch.org/docs/1.9.0/generated/torch.where.html#torch-where
    # "However, if you need the gradients to flow through the branches, please use torch.lerp"
    # Although we are not using lerp, the principle of avoiding sharp branches is the same.
    sin_half_angles_over_angles_approx = 0.5 - angle * angle / 48
    # Clamp angle to avoid division by zero in the non-taylor branch when angle is exactly 0.
    angle_safe = torch.where(use_taylor, torch.ones_like(angle), angle)
    sin_half_angles_over_angles_exact = torch.sin(half_angle) / angle_safe

    sin_half_angles_over_angles = torch.where(
        use_taylor,
        sin_half_angles_over_angles_approx,
        sin_half_angles_over_angles_exact,
    )
    return quat_xyz / sin_half_angles_over_angles[..., None]


@torch.compile
def quat_box_minus(
    q1: torch.Tensor,
    q2: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Right-invariant quaternion difference mapped to so(3) via log map.

    Computes log(q1 * q2^{-1}) using the shortest rotation convention.

    Args:
        q1 (torch.Tensor): (..., 4) quaternion. If `w_last` is True, format is [x, y, z, w]; otherwise [w, x, y, z].
        q2 (torch.Tensor): (..., 4) quaternion with the same format as `q1`.
        w_last (bool): Whether the scalar part w is the last element.

    Returns:
        torch.Tensor: (..., 3) axis-angle error vector.
    """
    if w_last:
        q1_xyzw = q1
        q2_xyzw = q2
    else:
        # Convert from (w, x, y, z) to (x, y, z, w)
        q1_xyzw = torch.cat([q1[..., 1:4], q1[..., 0:1]], dim=-1)
        q2_xyzw = torch.cat([q2[..., 1:4], q2[..., 0:1]], dim=-1)

    quat_diff = quat_mul(
        q1_xyzw,
        quat_conjugate(q2_xyzw),
        w_last=True,
    )  # q1 * q2^-1
    return axis_angle_from_quat(quat_diff, w_last=True)  # log(qd)


@torch.compile
def quat_error_magnitude(
    q1: torch.Tensor,
    q2: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Geodesic angle between two orientations given as quaternions.

    Args:
        q1 (torch.Tensor): (..., 4) quaternion. If `w_last` is True, format is [x, y, z, w]; otherwise [w, x, y, z].
        q2 (torch.Tensor): (..., 4) quaternion with the same format as `q1`.
        w_last (bool): Whether the scalar part w is the last element.

    Returns:
        torch.Tensor: (...,) rotation angle in radians in [0, pi].
    """
    axis_angle_error = quat_box_minus(q1, q2, w_last=w_last)
    return torch.norm(axis_angle_error, dim=-1)
