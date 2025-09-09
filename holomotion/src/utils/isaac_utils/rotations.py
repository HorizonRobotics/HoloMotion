# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# This file was originally copied from the [ASAP] repository:
# https://github.com/LeCAR-Lab/ASAP
# Modifications have been made to fit the needs of this project.

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from holomotion.src.utils.isaac_utils.maths import (
    copysign,
    normalize,
)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_yaw(quat: Tensor, vec: Tensor, w_last: bool) -> Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec, w_last)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
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
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
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
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch.atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


@torch.jit.script
def heading_to_quat(h_theta, w_last: bool):
    axis = torch.zeros(
        h_theta.shape
        + [
            3,
        ],
        device=h_theta.device,
    )
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(h_theta, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    return quat_rotate(q, v, w_last)


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    # ZL: could have issues.
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

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
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(
        torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q
    )
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


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
    # type: (Tensor) -> Tensor
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
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def calc_heading_quat(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q, w_last):
    # type: (Tensor, bool) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_inverse(x, w_last):
    # type: (Tensor, bool) -> Tensor
    """The inverse of the rotation"""
    return quat_conjugate(x, w_last=w_last)


@torch.jit.script
def get_euler_xyz(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
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
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp)
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


# @torch.jit.script
def get_euler_xyz_in_tensor(q):
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
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp)
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

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def quat_pos(x):
    """Make all the real part of the quaternion positive"""
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


@torch.jit.script
def is_valid_quat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return (w * w + x * x + y * y + z * z).allclose(torch.ones_like(w))


@torch.jit.script
def quat_normalize(q):
    """Construct 3D rotation from quaternion (the quaternion needs not to be normalized)."""
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
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
def quat_mul_norm(x, y, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    r"""Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y, w_last))


@torch.jit.script
def quat_mul_norm(x, y, w_last):
    # type: (Tensor, Tensor, bool) -> Tensor
    r"""Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_unit(quat_mul(x, y, w_last))


@torch.jit.script
def quat_identity(shape: List[int]):
    """Construct 3D identity rotation given shape"""
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)


@torch.jit.script
def quat_identity_like(x):
    """Construct identity 3D rotation with the same shape"""
    return quat_identity(x.shape[:-1])


@torch.jit.script
def transform_from_rotation_translation(
    r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None
):
    """Construct a transform from a quaternion and 3D translation. Only one of them can be None."""
    assert r is not None or t is not None, (
        "rotation and translation can't be all None"
    )
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)


@torch.jit.script
def transform_rotation(x):
    """Get rotation from transform"""
    return x[..., :4]


@torch.jit.script
def transform_translation(x):
    """Get translation from transform"""
    return x[..., 4:]


@torch.jit.script
def transform_mul(x, y):
    """Combine two transformation together"""
    z = transform_from_rotation_translation(
        r=quat_mul_norm(
            transform_rotation(x), transform_rotation(y), w_last=True
        ),
        t=quat_rotate(
            transform_rotation(x), transform_translation(y), w_last=True
        )
        + transform_translation(x),
    )
    return z


@torch.compile
def quaternion_to_matrix(
    quaternions: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions as tensor of shape (..., 4).
            If w_last=True (default): real part last (x, y, z, w)
            If w_last=False: real part first (w, x, y, z)
        w_last: If True, quaternion format is (x, y, z, w).
                If False, quaternion format is (w, x, y, z). Default: True.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    """
    if w_last:
        i, j, k, r = torch.unbind(quaternions, -1)
    else:
        r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


# @torch.jit.script
def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]


# @torch.jit.script
def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """W x y z
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quat_w_first(rot):
    rot = torch.cat([rot[..., [-1]], rot[..., :-1]], -1)
    return rot


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


@torch.compile
def remove_yaw_component(
    quat_raw: Tensor,
    quat_init: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Remove yaw component from quaternion while keeping roll and pitch.

    This function extracts the yaw component from the initial quaternion and uses
    it to normalize the raw quaternion, effectively removing the initial heading
    offset while preserving roll and pitch components.

    Args:
        quat_raw: Current quaternion from IMU, shape (..., 4)
        quat_init: Initial quaternion (contains the yaw to be removed), shape (..., 4)
        w_last: If True, quaternion format is (x, y, z, w).
                If False, quaternion format is (w, x, y, z). Default: True.

    Returns:
        Quaternion with initial yaw component removed, same shape as input.
        The resulting quaternion represents roll and pitch relative to the
        heading-aligned coordinate frame.

    Example:
        >>> # Initial robot orientation (roll=0°, pitch=0°, yaw=45°)
        >>> quat_init = quat_from_euler_xyz(
        ...     torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.7854)
        ... )
        >>> # Current IMU reading (roll=10°, pitch=20°, yaw=60°)
        >>> quat_raw = quat_from_euler_xyz(
        ...     torch.tensor(0.1745),
        ...     torch.tensor(0.3491),
        ...     torch.tensor(1.0472),
        ... )
        >>> quat_norm = remove_yaw_component(quat_raw, quat_init)
        >>> # quat_norm contains roll=10°, pitch=20°, with initial yaw offset removed
    """
    # Extract quaternion components based on format
    if w_last:
        q_w = quat_init[..., -1]
        q_vec = quat_init[..., :3]
    else:
        q_w = quat_init[..., 0]
        q_vec = quat_init[..., 1:]

    # Calculate heading by rotating x-axis with quaternion
    # ref_dir = [1, 0, 0] (x-axis)
    ref_dir = torch.zeros_like(q_vec)
    ref_dir[..., 0] = 1.0

    # Quaternion rotation: v' = v + 2 * w * (q_vec × v) + 2 * q_vec × (q_vec × v)
    cross1 = torch.cross(q_vec, ref_dir, dim=-1)
    cross2 = torch.cross(q_vec, cross1, dim=-1)
    rot_dir = ref_dir + 2.0 * q_w.unsqueeze(-1) * cross1 + 2.0 * cross2

    # Extract heading angle from rotated x-axis
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])

    # Create inverse heading quaternion (rotation about negative z-axis)
    half_heading = (-heading) * 0.5
    heading_q_inv = torch.zeros_like(quat_init)

    if w_last:
        heading_q_inv[..., 0] = 0.0  # x
        heading_q_inv[..., 1] = 0.0  # y
        heading_q_inv[..., 2] = torch.sin(half_heading)  # z
        heading_q_inv[..., 3] = torch.cos(half_heading)  # w
    else:
        heading_q_inv[..., 0] = torch.cos(half_heading)  # w
        heading_q_inv[..., 1] = 0.0  # x
        heading_q_inv[..., 2] = 0.0  # y
        heading_q_inv[..., 3] = torch.sin(half_heading)  # z

    # Quaternion multiplication: heading_q_inv * quat_raw
    shape = quat_raw.shape
    a = heading_q_inv.reshape(-1, 4)
    b = quat_raw.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    # Quaternion multiplication formula
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
        quat_result = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat_result = torch.stack([w, x, y, z], dim=-1).view(shape)

    # Normalize the result quaternion
    norm = torch.norm(quat_result, p=2, dim=-1, keepdim=True)
    quat_norm = quat_result / norm.clamp(min=1e-8)

    return quat_norm
