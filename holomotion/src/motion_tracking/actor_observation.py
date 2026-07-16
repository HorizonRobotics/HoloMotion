"""Shared Warp actor observation construction for motion tracking."""

from dataclasses import dataclass

import torch
import warp as wp

from .reference_observation import (
    REFERENCE_DOF_DIM,
    REFERENCE_QPOS_DIM,
    derive_reference_kinematics_torch,
)


MOTION_ACTOR_CURRENT_TERMS = (
    "actor_ref_gravity_projection_cur",
    "actor_ref_base_linvel_cur",
    "actor_ref_base_angvel_cur",
    "actor_ref_dof_pos_cur",
    "actor_ref_root_height_cur",
    "actor_ref_robot_yaw_error_sin_cos",
    "actor_projected_gravity",
    "actor_rel_robot_root_ang_vel",
    "actor_dof_pos",
    "actor_dof_vel",
    "actor_last_action",
)

MOTION_ACTOR_FUTURE_TERMS = (
    "actor_ref_dof_pos_fut",
    "actor_ref_root_height_fut",
    "actor_ref_gravity_projection_fut",
    "actor_ref_base_linvel_fut",
    "actor_ref_base_angvel_fut",
    "actor_ref_future_yaw_delta_sin_cos",
    "actor_ref_future_root_ori_robot_frame_6d",
)

MOTION_ACTOR_TERMS = MOTION_ACTOR_CURRENT_TERMS + MOTION_ACTOR_FUTURE_TERMS
MOTION_ACTOR_CURRENT_DIM = 134
MOTION_ACTOR_FUTURE_DIM = 47


@dataclass(frozen=True)
class MotionActorObservationInput:
    """Raw tensors needed by both training and deployed motion policies."""

    reference_qpos: torch.Tensor
    robot_root_quat_wxyz: torch.Tensor
    robot_root_angvel_local: torch.Tensor
    robot_dof_pos: torch.Tensor
    robot_dof_vel: torch.Tensor
    last_action: torch.Tensor
    default_dof_pos: torch.Tensor
    reference_dof_indices: torch.Tensor | None = None
    reference_sample_time: torch.Tensor | None = None
    reference_yaw_alignment_wxyz: torch.Tensor | None = None


@wp.func
def _normalize_quat_wxyz(q: wp.vec4) -> wp.vec4:
    norm = wp.sqrt(wp.dot(q, q))
    q = q / wp.max(norm, 1.0e-9)
    if q[0] < 0.0:
        q = -q
    return q


@wp.func
def _quat_mul_wxyz(q0: wp.vec4, q1: wp.vec4) -> wp.vec4:
    w0 = q0[0]
    xyz0 = wp.vec3(q0[1], q0[2], q0[3])
    w1 = q1[0]
    xyz1 = wp.vec3(q1[1], q1[2], q1[3])
    xyz = w0 * xyz1 + w1 * xyz0 + wp.cross(xyz0, xyz1)
    return wp.vec4(w0 * w1 - wp.dot(xyz0, xyz1), xyz[0], xyz[1], xyz[2])


@wp.func
def _quat_inv_wxyz(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def _yaw_from_quat_wxyz(q: wp.vec4) -> float:
    return wp.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


@wp.func
def _write_projected_gravity(
    q: wp.vec4,
    output: wp.array2d(dtype=float),
    batch: int,
    offset: int,
):
    output[batch, offset + 0] = 2.0 * (-q[3] * q[1] + q[0] * q[2])
    output[batch, offset + 1] = -2.0 * (q[3] * q[2] + q[0] * q[1])
    output[batch, offset + 2] = 1.0 - 2.0 * (q[0] * q[0] + q[3] * q[3])


@wp.func
def _write_rot6d_wxyz(
    q: wp.vec4,
    output: wp.array2d(dtype=float),
    batch: int,
    offset: int,
):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    output[batch, offset + 0] = 1.0 - 2.0 * (y * y + z * z)
    output[batch, offset + 1] = 2.0 * (x * y - z * w)
    output[batch, offset + 2] = 2.0 * (x * y + z * w)
    output[batch, offset + 3] = 1.0 - 2.0 * (x * x + z * z)
    output[batch, offset + 4] = 2.0 * (x * z - y * w)
    output[batch, offset + 5] = 2.0 * (y * z + x * w)


@wp.func
def _projected_gravity_component(q: wp.vec4, component: int) -> float:
    value = 2.0 * (-q[3] * q[1] + q[0] * q[2])
    if component == 1:
        value = -2.0 * (q[3] * q[2] + q[0] * q[1])
    elif component == 2:
        value = 1.0 - 2.0 * (q[0] * q[0] + q[3] * q[3])
    return value


@wp.func
def _rot6d_component_wxyz(q: wp.vec4, component: int) -> float:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    value = 1.0 - 2.0 * (y * y + z * z)
    if component == 1:
        value = 2.0 * (x * y - z * w)
    elif component == 2:
        value = 2.0 * (x * y + z * w)
    elif component == 3:
        value = 1.0 - 2.0 * (x * x + z * z)
    elif component == 4:
        value = 2.0 * (x * z - y * w)
    elif component == 5:
        value = 2.0 * (y * z + x * w)
    return value


@wp.kernel
def _motion_actor_observation_kernel(
    qpos: wp.array3d(dtype=float),
    dof_vel: wp.array3d(dtype=float),
    root_linvel_local: wp.array3d(dtype=float),
    root_angvel_local: wp.array3d(dtype=float),
    projected_gravity: wp.array3d(dtype=float),
    robot_root_quat_wxyz: wp.array2d(dtype=float),
    robot_root_angvel_local: wp.array2d(dtype=float),
    robot_dof_pos: wp.array2d(dtype=float),
    robot_dof_vel: wp.array2d(dtype=float),
    last_action: wp.array2d(dtype=float),
    default_dof_pos: wp.array2d(dtype=float),
    reference_dof_indices: wp.array1d(dtype=int),
    reference_yaw_alignment_wxyz: wp.array2d(dtype=float),
    use_yaw_alignment: int,
    current_index: int,
    num_future_frames: int,
    output: wp.array2d(dtype=float),
):
    index = wp.tid()
    observation_dim = (
        MOTION_ACTOR_CURRENT_DIM + num_future_frames * MOTION_ACTOR_FUTURE_DIM
    )
    batch = index // observation_dim
    element = index - batch * observation_dim
    current = current_index

    if element < 3:
        output[batch, element] = projected_gravity[batch, current, element]
    elif element < 6:
        output[batch, element] = root_linvel_local[batch, current, element - 3]
    elif element < 9:
        output[batch, element] = root_angvel_local[batch, current, element - 6]
    elif element < 38:
        dof = element - 9
        output[batch, element] = qpos[
            batch, current, 7 + reference_dof_indices[dof]
        ]
    elif element == 38:
        output[batch, element] = qpos[batch, current, 2]
    elif element < 41:
        robot_quat = _normalize_quat_wxyz(
            wp.vec4(
                robot_root_quat_wxyz[batch, 0],
                robot_root_quat_wxyz[batch, 1],
                robot_root_quat_wxyz[batch, 2],
                robot_root_quat_wxyz[batch, 3],
            )
        )
        ref_quat = _normalize_quat_wxyz(
            wp.vec4(
                qpos[batch, current, 3],
                qpos[batch, current, 4],
                qpos[batch, current, 5],
                qpos[batch, current, 6],
            )
        )
        if use_yaw_alignment != 0:
            alignment = _normalize_quat_wxyz(
                wp.vec4(
                    reference_yaw_alignment_wxyz[batch, 0],
                    reference_yaw_alignment_wxyz[batch, 1],
                    reference_yaw_alignment_wxyz[batch, 2],
                    reference_yaw_alignment_wxyz[batch, 3],
                )
            )
            ref_quat = _normalize_quat_wxyz(
                _quat_mul_wxyz(alignment, ref_quat)
            )
        yaw_error = _yaw_from_quat_wxyz(ref_quat) - _yaw_from_quat_wxyz(
            robot_quat
        )
        if element == 39:
            output[batch, element] = wp.sin(yaw_error)
        else:
            output[batch, element] = wp.cos(yaw_error)
    elif element < 44:
        robot_quat = _normalize_quat_wxyz(
            wp.vec4(
                robot_root_quat_wxyz[batch, 0],
                robot_root_quat_wxyz[batch, 1],
                robot_root_quat_wxyz[batch, 2],
                robot_root_quat_wxyz[batch, 3],
            )
        )
        output[batch, element] = _projected_gravity_component(
            robot_quat, element - 41
        )
    elif element < 47:
        output[batch, element] = robot_root_angvel_local[batch, element - 44]
    elif element < 76:
        dof = element - 47
        output[batch, element] = (
            robot_dof_pos[batch, dof] - default_dof_pos[batch, dof]
        )
    elif element < 105:
        output[batch, element] = robot_dof_vel[batch, element - 76]
    elif element < MOTION_ACTOR_CURRENT_DIM:
        output[batch, element] = last_action[batch, element - 105]
    else:
        future_element = element - MOTION_ACTOR_CURRENT_DIM
        dof_count = num_future_frames * REFERENCE_DOF_DIM
        height_count = num_future_frames
        vec3_count = num_future_frames * 3
        yaw_count = num_future_frames * 2
        if future_element < dof_count:
            future_index = future_element // REFERENCE_DOF_DIM
            dof = future_element - future_index * REFERENCE_DOF_DIM
            frame = current + 1 + future_index
            output[batch, element] = qpos[
                batch, frame, 7 + reference_dof_indices[dof]
            ]
        elif future_element < dof_count + height_count:
            future_index = future_element - dof_count
            output[batch, element] = qpos[batch, current + 1 + future_index, 2]
        elif future_element < dof_count + height_count + vec3_count:
            local = future_element - dof_count - height_count
            future_index = local // 3
            axis = local - future_index * 3
            output[batch, element] = projected_gravity[
                batch, current + 1 + future_index, axis
            ]
        elif future_element < dof_count + height_count + 2 * vec3_count:
            local = future_element - dof_count - height_count - vec3_count
            future_index = local // 3
            axis = local - future_index * 3
            output[batch, element] = root_linvel_local[
                batch, current + 1 + future_index, axis
            ]
        elif future_element < dof_count + height_count + 3 * vec3_count:
            local = future_element - dof_count - height_count - 2 * vec3_count
            future_index = local // 3
            axis = local - future_index * 3
            output[batch, element] = root_angvel_local[
                batch, current + 1 + future_index, axis
            ]
        elif (
            future_element
            < dof_count + height_count + 3 * vec3_count + yaw_count
        ):
            local = future_element - dof_count - height_count - 3 * vec3_count
            future_index = local // 2
            component = local - future_index * 2
            ref_quat_current = _normalize_quat_wxyz(
                wp.vec4(
                    qpos[batch, current, 3],
                    qpos[batch, current, 4],
                    qpos[batch, current, 5],
                    qpos[batch, current, 6],
                )
            )
            frame = current + 1 + future_index
            ref_quat_future = _normalize_quat_wxyz(
                wp.vec4(
                    qpos[batch, frame, 3],
                    qpos[batch, frame, 4],
                    qpos[batch, frame, 5],
                    qpos[batch, frame, 6],
                )
            )
            yaw_delta = _yaw_from_quat_wxyz(
                ref_quat_future
            ) - _yaw_from_quat_wxyz(ref_quat_current)
            if component == 0:
                output[batch, element] = wp.sin(yaw_delta)
            else:
                output[batch, element] = wp.cos(yaw_delta)
        else:
            local = (
                future_element
                - dof_count
                - height_count
                - 3 * vec3_count
                - yaw_count
            )
            future_index = local // 6
            component = local - future_index * 6
            frame = current + 1 + future_index
            robot_quat = _normalize_quat_wxyz(
                wp.vec4(
                    robot_root_quat_wxyz[batch, 0],
                    robot_root_quat_wxyz[batch, 1],
                    robot_root_quat_wxyz[batch, 2],
                    robot_root_quat_wxyz[batch, 3],
                )
            )
            ref_quat = _normalize_quat_wxyz(
                wp.vec4(
                    qpos[batch, frame, 3],
                    qpos[batch, frame, 4],
                    qpos[batch, frame, 5],
                    qpos[batch, frame, 6],
                )
            )
            if use_yaw_alignment != 0:
                alignment = _normalize_quat_wxyz(
                    wp.vec4(
                        reference_yaw_alignment_wxyz[batch, 0],
                        reference_yaw_alignment_wxyz[batch, 1],
                        reference_yaw_alignment_wxyz[batch, 2],
                        reference_yaw_alignment_wxyz[batch, 3],
                    )
                )
                ref_quat = _normalize_quat_wxyz(
                    _quat_mul_wxyz(alignment, ref_quat)
                )
            relative_quat = _normalize_quat_wxyz(
                _quat_mul_wxyz(_quat_inv_wxyz(robot_quat), ref_quat)
            )
            output[batch, element] = _rot6d_component_wxyz(
                relative_quat, component
            )


def motion_actor_observation_dim(num_future_frames: int) -> int:
    return (
        MOTION_ACTOR_CURRENT_DIM
        + int(num_future_frames) * MOTION_ACTOR_FUTURE_DIM
    )


def launch_motion_actor_observation_warp(
    *,
    qpos_wp,
    kinematics: tuple,
    robot_root_quat_wp,
    robot_root_angvel_wp,
    robot_dof_pos_wp,
    robot_dof_vel_wp,
    last_action_wp,
    default_dof_pos_wp,
    reference_dof_indices_wp,
    reference_yaw_alignment_wp,
    use_yaw_alignment: bool,
    current_index: int,
    num_future_frames: int,
    output_wp,
    device,
) -> None:
    """Launch the canonical actor Observation kernel on existing Warp arrays."""

    dof_vel_wp, root_linvel_local_wp, root_angvel_local_wp, gravity_wp = (
        kinematics
    )
    wp.launch(
        _motion_actor_observation_kernel,
        dim=qpos_wp.shape[0] * motion_actor_observation_dim(num_future_frames),
        inputs=[
            qpos_wp,
            dof_vel_wp,
            root_linvel_local_wp,
            root_angvel_local_wp,
            gravity_wp,
            robot_root_quat_wp,
            robot_root_angvel_wp,
            robot_dof_pos_wp,
            robot_dof_vel_wp,
            last_action_wp,
            default_dof_pos_wp,
            reference_dof_indices_wp,
            reference_yaw_alignment_wp,
            int(bool(use_yaw_alignment)),
            int(current_index),
            int(num_future_frames),
            output_wp,
        ],
        device=device,
    )


def _as_batch(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != width:
        raise ValueError(
            f"{name} must be [B,{width}], got {tuple(value.shape)}"
        )
    return value


def build_motion_actor_observation_torch(
    inputs: MotionActorObservationInput,
    *,
    fps: float = 50.0,
    current_index: int = 1,
    num_future_frames: int = 10,
) -> torch.Tensor:
    """Torch zero-copy adapter around the canonical Warp kernel."""

    qpos = inputs.reference_qpos
    if qpos.ndim == 2:
        qpos = qpos.unsqueeze(0)
    qpos = qpos.to(dtype=torch.float32).contiguous()
    if qpos.ndim != 3 or qpos.shape[-1] != REFERENCE_QPOS_DIM:
        raise ValueError(
            f"reference_qpos must be [B,T,{REFERENCE_QPOS_DIM}], got {tuple(qpos.shape)}"
        )
    batch, frames, _ = qpos.shape
    future_end = int(current_index) + 1 + int(num_future_frames)
    if int(current_index) < 0 or future_end > frames:
        raise ValueError(
            f"reference_qpos has {frames} frames but requires {future_end}"
        )

    def prepare(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        return (
            _as_batch(value, width, name)
            .to(device=qpos.device, dtype=torch.float32)
            .contiguous()
        )

    robot_quat = prepare(
        inputs.robot_root_quat_wxyz, 4, "robot_root_quat_wxyz"
    )
    robot_angvel = prepare(
        inputs.robot_root_angvel_local, 3, "robot_root_angvel_local"
    )
    robot_dof_pos = prepare(
        inputs.robot_dof_pos, REFERENCE_DOF_DIM, "robot_dof_pos"
    )
    robot_dof_vel = prepare(
        inputs.robot_dof_vel, REFERENCE_DOF_DIM, "robot_dof_vel"
    )
    last_action = prepare(inputs.last_action, REFERENCE_DOF_DIM, "last_action")
    default_dof_pos = prepare(
        inputs.default_dof_pos, REFERENCE_DOF_DIM, "default_dof_pos"
    )
    if default_dof_pos.shape[0] == 1 and batch != 1:
        default_dof_pos = default_dof_pos.expand(batch, -1).contiguous()

    indices = inputs.reference_dof_indices
    if indices is None:
        indices = torch.arange(
            REFERENCE_DOF_DIM, device=qpos.device, dtype=torch.int32
        )
    else:
        indices = indices.to(
            device=qpos.device, dtype=torch.int32
        ).contiguous()
    if indices.ndim != 1 or indices.numel() != REFERENCE_DOF_DIM:
        raise ValueError("reference_dof_indices must contain 29 indices")

    alignment = inputs.reference_yaw_alignment_wxyz
    use_alignment = alignment is not None
    if alignment is None:
        alignment = torch.zeros(
            (batch, 4), device=qpos.device, dtype=torch.float32
        )
        alignment[:, 0] = 1.0
    else:
        alignment = prepare(alignment, 4, "reference_yaw_alignment_wxyz")
        if alignment.shape[0] == 1 and batch != 1:
            alignment = alignment.expand(batch, -1).contiguous()

    kinematics = derive_reference_kinematics_torch(
        qpos,
        sample_time=inputs.reference_sample_time,
        fps=fps,
    )
    output = torch.empty(
        (batch, motion_actor_observation_dim(num_future_frames)),
        device=qpos.device,
        dtype=torch.float32,
    )
    device = wp.device_from_torch(qpos.device)
    launch_motion_actor_observation_warp(
        qpos_wp=wp.from_torch(qpos, dtype=wp.float32),
        kinematics=(
            wp.from_torch(kinematics.dof_vel, dtype=wp.float32),
            wp.from_torch(kinematics.root_linvel_local, dtype=wp.float32),
            wp.from_torch(kinematics.root_angvel_local, dtype=wp.float32),
            wp.from_torch(kinematics.projected_gravity, dtype=wp.float32),
        ),
        robot_root_quat_wp=wp.from_torch(robot_quat, dtype=wp.float32),
        robot_root_angvel_wp=wp.from_torch(robot_angvel, dtype=wp.float32),
        robot_dof_pos_wp=wp.from_torch(robot_dof_pos, dtype=wp.float32),
        robot_dof_vel_wp=wp.from_torch(robot_dof_vel, dtype=wp.float32),
        last_action_wp=wp.from_torch(last_action, dtype=wp.float32),
        default_dof_pos_wp=wp.from_torch(default_dof_pos, dtype=wp.float32),
        reference_dof_indices_wp=wp.from_torch(indices, dtype=wp.int32),
        reference_yaw_alignment_wp=wp.from_torch(alignment, dtype=wp.float32),
        use_yaw_alignment=use_alignment,
        current_index=current_index,
        num_future_frames=num_future_frames,
        output_wp=wp.from_torch(output, dtype=wp.float32),
        device=device,
    )
    return output


def derive_motion_actor_terms_torch(
    inputs: MotionActorObservationInput,
    *,
    fps: float = 50.0,
    current_index: int = 1,
    num_future_frames: int = 10,
) -> dict[str, torch.Tensor]:
    """Return named views into the canonical flattened actor Observation."""

    observation = build_motion_actor_observation_torch(
        inputs,
        fps=fps,
        current_index=current_index,
        num_future_frames=num_future_frames,
    )
    widths = (3, 3, 3, 29, 1, 2, 3, 3, 29, 29, 29)
    terms: dict[str, torch.Tensor] = {}
    offset = 0
    for name, width in zip(MOTION_ACTOR_CURRENT_TERMS, widths, strict=True):
        terms[name] = observation[:, offset : offset + width]
        offset += width
    future_widths = (29, 1, 3, 3, 3, 2, 6)
    for name, width in zip(
        MOTION_ACTOR_FUTURE_TERMS, future_widths, strict=True
    ):
        size = int(num_future_frames) * width
        terms[name] = observation[:, offset : offset + size].reshape(
            observation.shape[0], int(num_future_frames), width
        )
        offset += size
    return terms


__all__ = [
    "MOTION_ACTOR_CURRENT_TERMS",
    "MOTION_ACTOR_FUTURE_TERMS",
    "MOTION_ACTOR_TERMS",
    "MotionActorObservationInput",
    "build_motion_actor_observation_torch",
    "derive_motion_actor_terms_torch",
    "launch_motion_actor_observation_warp",
    "motion_actor_observation_dim",
]
