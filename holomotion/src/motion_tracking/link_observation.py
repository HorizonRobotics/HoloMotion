"""Shared masked link commands for training and deployment.

All quaternions are WXYZ; positions are metres. This module only depends on
PyTorch, and never accepts an actual robot world position. When error features
are enabled, actual link poses must come from FK of the same measured (possibly
noisy/delayed) absolute joint angles used by the actor, in the actual pelvis
frame.

A link's rotation uses the rotated x axis followed by the rotated z axis:
identity is [1, 0, 0, 0, 0, 1]. Masked rotations are six zeros; explicit
validity bits distinguish them from active targets. The reference pelvis used
to define the frame is independent of whether pelvis is a controlled link.
"""

from dataclasses import dataclass

import torch

DEFAULT_COMMAND_LINK_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
DEFAULT_FUTURE_FRAME_OFFSETS = (1, 2)
LINK_FEATURE_DIM = 18
QUATERNION_EPS = 1.0e-8


@dataclass(frozen=True)
class LinkObservationInput:
    """Tensor contract for reference and measured FK inputs.

    B=batch, F=future timestamps, L=ordered command links. Reference positions
    and quaternions share one world frame. Both future frames use the SAME
    current reference_root_pos [B,3] and current robot_root_quat_wxyz [B,4].

    frame_offsets are positive frame indices, represented as float32 in the
    output, NOT seconds. The controller period is external schema metadata.
    link_mask is [L] or [B,L], time_valid is [F] or [B,F]; missing masks mean
    all valid. All masks must be bool or contain 0/1 values.
    reference_root_height [B,1] is height above the reference ground, and
    defaults to reference_root_pos[...,2:3] for a zero-height ground frame.

    The joint order and link order must already have been mapped by name.
    This module does not perform FK or restore relative q to absolute q.
    Actual local link poses may be None when include_error_terms=False.
    """

    reference_link_pos: torch.Tensor  # [B,F,L,3]
    reference_link_quat_wxyz: torch.Tensor  # [B,F,L,4]
    reference_root_pos: torch.Tensor  # [B,3], CURRENT reference pelvis
    robot_root_quat_wxyz: torch.Tensor  # [B,4], CURRENT measured orientation
    robot_link_pos_local: torch.Tensor | None = None  # [B,L,3], FK
    robot_link_quat_local_wxyz: torch.Tensor | None = None  # [B,L,4], FK
    frame_offsets: torch.Tensor | None = None  # [F] or [B,F]
    link_mask: torch.Tensor | None = None  # [L] or [B,L]
    time_valid: torch.Tensor | None = None  # [F] or [B,F]
    reference_root_height: torch.Tensor | None = None  # [B,1]


@dataclass(frozen=True)
class LinkObservation:
    """Atomic float32 outputs and canonical packing helpers.

    link_mask [B,L] and time_valid [B,F] are EFFECTIVE validity after checking
    supplied masks and finite inputs. A corrupt link at any valid future
    timestamp is disabled at all timestamps, retaining a compact [B,L] mode.
    Corrupt data at an already-invalid time cannot disable an otherwise good
    link. effective_mask [B,F,L] combines the two independent masks.

    A common frame failure disables every timestamp and link. Callers should
    record such failures or terminate the episode; sanitation is not a sensor
    health policy.
    """

    target_pos: torch.Tensor
    position_error: torch.Tensor | None
    target_rot6d: torch.Tensor
    rotation_error6d: torch.Tensor | None
    time_offsets: torch.Tensor
    link_mask: torch.Tensor
    time_valid: torch.Tensor
    reference_height: torch.Tensor
    effective_mask: torch.Tensor

    @property
    def task(self) -> torch.Tensor:
        """Return [B,F,18*L+1], or [B,F,9*L+1] when errors are disabled."""
        if self.position_error is None and self.rotation_error6d is None:
            return self.reference_task
        if self.position_error is None or self.rotation_error6d is None:
            raise ValueError(
                "Position and rotation errors must be enabled together"
            )
        return torch.cat(
            (
                self.target_pos.flatten(start_dim=2),
                self.position_error.flatten(start_dim=2),
                self.target_rot6d.flatten(start_dim=2),
                self.rotation_error6d.flatten(start_dim=2),
                self.time_offsets,
            ),
            dim=-1,
        )

    @property
    def reference_task(self) -> torch.Tensor:
        """Return router targets [B,F,9*L+1] without tracking errors."""
        return torch.cat(
            (
                self.target_pos.flatten(start_dim=2),
                self.target_rot6d.flatten(start_dim=2),
                self.time_offsets,
            ),
            dim=-1,
        )

    @property
    def flat_command(self) -> torch.Tensor:
        """Flatten task, reference height, link mask, and time validity."""
        return torch.cat(
            (
                self.task.flatten(start_dim=1),
                self.reference_height,
                self.link_mask,
                self.time_valid,
            ),
            dim=-1,
        )

    @property
    def flat_reference_command(self) -> torch.Tensor:
        """Return router targets, height, masks: [B,271] for F=2,L=14."""
        return torch.cat(
            (
                self.reference_task.flatten(start_dim=1),
                self.reference_height,
                self.link_mask,
                self.time_valid,
            ),
            dim=-1,
        )


def _safe_vector(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    finite = torch.isfinite(value).all(dim=-1)
    return torch.where(finite.unsqueeze(-1), value, 0.0), finite


def _safe_quaternion(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value, finite = _safe_vector(value)
    # Scaling first also avoids squaring a very large finite quaternion.
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / scale.clamp_min(QUATERNION_EPS)
    norm = (
        scaled.square()
        .sum(dim=-1, keepdim=True)
        .clamp_min(QUATERNION_EPS**2)
        .sqrt()
    )
    valid = finite & (scale.squeeze(-1) > QUATERNION_EPS)
    normalized = scaled / norm.clamp_min(QUATERNION_EPS)
    identity = torch.cat(
        (
            torch.ones_like(normalized[..., :1]),
            torch.zeros_like(normalized[..., 1:]),
        ),
        dim=-1,
    )
    return torch.where(valid.unsqueeze(-1), normalized, identity), valid


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate WXYZ quaternions; inverse for a normalized quaternion."""
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_multiply_wxyz(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Hamilton product of broadcast-compatible WXYZ quaternions."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized broadcast-compatible WXYZ quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    vx, vy, vz = vector.unbind(dim=-1)
    return torch.stack(
        (
            (1 - 2 * (y * y + z * z)) * vx
            + 2 * (x * y - w * z) * vy
            + 2 * (x * z + w * y) * vz,
            2 * (x * y + w * z) * vx
            + (1 - 2 * (x * x + z * z)) * vy
            + 2 * (y * z - w * x) * vz,
            2 * (x * z - w * y) * vx
            + 2 * (y * z + w * x) * vy
            + (1 - 2 * (x * x + y * y)) * vz,
        ),
        dim=-1,
    )


def quat_to_rot6d_xz_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize WXYZ rotations and concatenate rotated x/z basis vectors.

    Invalid quaternions map to identity here. The observation builder
    separately propagates invalidity to its explicit masks.
    """
    quat, _ = _safe_quaternion(quat)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )


def _batch_field(
    value: torch.Tensor | None,
    batch: int,
    width: int,
    prototype: torch.Tensor,
    default: float = 1.0,
) -> torch.Tensor:
    if value is None:
        return prototype.new_full((batch, width), default)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != width:
        raise ValueError(
            f"Expected [{width}] or [B,{width}], got {value.shape}"
        )
    if value.shape[0] not in (1, batch):
        raise ValueError(
            f"Invalid batch size {value.shape[0]}, expected {batch}"
        )
    return value.to(device=prototype.device, dtype=torch.float32).expand(
        batch, width
    )


def build_link_observation(
    inputs: LinkObservationInput,
    *,
    root_link_index: int | None = 0,
    include_error_terms: bool = True,
) -> LinkObservation:
    """Build masked local link commands without actual global translation.

    Position error = rotate(actual_root_inverse, reference_link_future -
    reference_pelvis_current) - actual_link_from_FK_local.
    Orientation error = target_local_quaternion * inverse(actual_FK_local).
    Both are expressed in the current actual pelvis axes, including tilt.
    reference_height is hidden if root_link_index is masked; the independent
    reference_root_pos still defines every link target coordinate. Set
    root_link_index=None only if height is an independent mandatory command.
    With include_error_terms=False, actual FK inputs are not read and may be
    None; errors are not computed and their output fields are None.
    """
    if not isinstance(include_error_terms, bool):
        raise TypeError("include_error_terms must be a bool")
    shape = inputs.reference_link_pos.shape
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError("reference_link_pos must have shape [B,F,L,3]")
    batch, frames, links, _ = shape
    expected_shapes = {
        "reference_link_quat_wxyz": (batch, frames, links, 4),
        "reference_root_pos": (batch, 3),
        "robot_root_quat_wxyz": (batch, 4),
    }
    if include_error_terms:
        expected_shapes.update(
            {
                "robot_link_pos_local": (batch, links, 3),
                "robot_link_quat_local_wxyz": (batch, links, 4),
            }
        )
    for name, expected in expected_shapes.items():
        value = getattr(inputs, name)
        if value is None or value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if root_link_index is not None and not 0 <= root_link_index < links:
        raise ValueError("root_link_index must identify a command link")
    if frames < 1 or links < 1:
        raise ValueError("At least one frame and one link are required")

    ref_pos, ref_pos_valid = _safe_vector(inputs.reference_link_pos)
    ref_quat, ref_quat_valid = _safe_quaternion(
        inputs.reference_link_quat_wxyz
    )
    root_pos, root_pos_valid = _safe_vector(inputs.reference_root_pos)
    robot_root, robot_root_valid = _safe_quaternion(
        inputs.robot_root_quat_wxyz
    )
    if include_error_terms:
        actual_pos, actual_pos_valid = _safe_vector(
            inputs.robot_link_pos_local
        )
        actual_quat, actual_quat_valid = _safe_quaternion(
            inputs.robot_link_quat_local_wxyz
        )
    height = inputs.reference_root_height
    if height is None:
        height = inputs.reference_root_pos[..., 2:3]
    if height.shape != (batch, 1):
        raise ValueError("reference_root_height must have shape [B,1]")
    height, height_valid = _safe_vector(height)
    common_valid = root_pos_valid & robot_root_valid
    if root_link_index is None:
        common_valid = common_valid & height_valid

    link_requested = _batch_field(inputs.link_mask, batch, links, ref_pos)
    link_requested = torch.isfinite(link_requested) & (link_requested == 1)
    time_requested = _batch_field(inputs.time_valid, batch, frames, ref_pos)
    time_requested = torch.isfinite(time_requested) & (time_requested == 1)
    offsets = inputs.frame_offsets
    if offsets is None:
        offsets = torch.arange(
            1, frames + 1, device=ref_pos.device, dtype=torch.float32
        )
    offsets = _batch_field(offsets, batch, frames, ref_pos)
    time_valid = (
        time_requested
        & torch.isfinite(offsets)
        & (offsets > 0)
        & common_valid.unsqueeze(-1)
    )

    inv_root = quat_conjugate_wxyz(robot_root)[:, None, None, :]
    target_pos = quat_apply_wxyz(
        inv_root, ref_pos - root_pos[:, None, None, :]
    )
    target_quat = quat_multiply_wxyz(inv_root, ref_quat)
    target_rot6d = quat_to_rot6d_xz_wxyz(target_quat)
    position_error = None
    rotation_error6d = None
    target_valid = (
        ref_pos_valid & ref_quat_valid & torch.isfinite(target_pos).all(dim=-1)
    )
    if include_error_terms:
        position_error = target_pos - actual_pos[:, None, :, :]
        error_quat = quat_multiply_wxyz(
            target_quat,
            quat_conjugate_wxyz(actual_quat)[:, None, :, :],
        )
        rotation_error6d = quat_to_rot6d_xz_wxyz(error_quat)
        target_valid = target_valid & torch.isfinite(position_error).all(
            dim=-1
        )
    link_valid = (
        link_requested
        & (target_valid | ~time_valid.unsqueeze(-1)).all(dim=1)
        & common_valid.unsqueeze(-1)
    )
    if include_error_terms:
        link_valid = link_valid & actual_pos_valid & actual_quat_valid
    if root_link_index is not None:
        index = torch.arange(links, device=ref_pos.device)
        link_valid = link_valid & (
            (index != root_link_index).unsqueeze(0)
            | height_valid.unsqueeze(-1)
        )
    height_visible = common_valid & height_valid
    if root_link_index is not None:
        height_visible = height_visible & link_valid[:, root_link_index]
    effective = time_valid.unsqueeze(-1) & link_valid.unsqueeze(1)
    feature_valid = effective.unsqueeze(-1)

    # Use where, never multiply by zero: inactive inputs may contain NaN/Inf.
    return LinkObservation(
        target_pos=torch.where(feature_valid, target_pos, 0.0),
        position_error=(
            torch.where(feature_valid, position_error, 0.0)
            if include_error_terms
            else None
        ),
        target_rot6d=torch.where(feature_valid, target_rot6d, 0.0),
        rotation_error6d=(
            torch.where(feature_valid, rotation_error6d, 0.0)
            if include_error_terms
            else None
        ),
        time_offsets=torch.where(time_valid, offsets, 0.0).unsqueeze(-1),
        link_mask=link_valid.float(),
        time_valid=time_valid.float(),
        reference_height=torch.where(
            height_visible.unsqueeze(-1), height, 0.0
        ),
        effective_mask=effective.float(),
    )
