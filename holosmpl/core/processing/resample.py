from __future__ import annotations

import os


class ResamplingNotImplemented(NotImplementedError):
    """Raised until resampling is implemented."""


def target_frame_count(source_frame_count: int, source_fps: float, target_fps: float) -> int:
    if source_frame_count <= 0:
        raise ValueError(f"source_frame_count must be positive, got {source_frame_count}")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    return max(1, int(round(source_frame_count * target_fps / source_fps)))


def resample_motion_to_fps(
    *,
    root_orient: object,
    pose_body: object,
    trans: object,
    source_fps: float,
    target_fps: float,
) -> tuple[object, object, object, dict[str, object]]:
    """Resample axis-angle rotations and root translation to target FPS.

    Imports numpy/scipy lazily so non-conversion CLI commands can run in
    lightweight environments.
    """

    import numpy as np

    root_orient_arr = np.asarray(root_orient, dtype=np.float32)
    pose_body_arr = np.asarray(pose_body, dtype=np.float32)
    trans_arr = np.asarray(trans, dtype=np.float32)
    if root_orient_arr.ndim != 2 or root_orient_arr.shape[1] != 3:
        raise ValueError(f"root_orient must be [T,3], got {root_orient_arr.shape}")
    if pose_body_arr.ndim != 2 or pose_body_arr.shape[1] % 3 != 0:
        raise ValueError(f"pose_body must be [T,D] where D % 3 == 0, got {pose_body_arr.shape}")
    if trans_arr.ndim != 2 or trans_arr.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans_arr.shape}")
    if not (len(root_orient_arr) == len(pose_body_arr) == len(trans_arr)):
        raise ValueError(
            "root_orient, pose_body, and trans must have the same frame count: "
            f"{len(root_orient_arr)}, {len(pose_body_arr)}, {len(trans_arr)}"
        )

    source_fps = float(source_fps)
    target_fps = float(target_fps)
    source_frames = int(len(root_orient_arr))
    out_frames = target_frame_count(source_frames, source_fps, target_fps)

    if source_frames == 1 or abs(source_fps - target_fps) < 1e-6:
        policy = "identity" if source_frames != 1 else "single_frame_identity"
        return (
            root_orient_arr.astype(np.float32, copy=True),
            pose_body_arr.astype(np.float32, copy=True),
            trans_arr.astype(np.float32, copy=True),
            {
                "resample_policy": policy,
                "source_frame_count": source_frames,
                "canonical_frame_count": source_frames,
            },
        )

    torch_result = _try_resample_motion_torch(
        root_orient_arr=root_orient_arr,
        pose_body_arr=pose_body_arr,
        trans_arr=trans_arr,
        source_fps=source_fps,
        target_fps=target_fps,
        out_frames=out_frames,
    )
    if torch_result is not None:
        return torch_result

    source_times = np.arange(source_frames, dtype=np.float64) / source_fps
    target_times = np.arange(out_frames, dtype=np.float64) / target_fps
    target_times = np.clip(target_times, source_times[0], source_times[-1])

    out_trans = np.empty((out_frames, 3), dtype=np.float32)
    for axis in range(3):
        out_trans[:, axis] = np.interp(target_times, source_times, trans_arr[:, axis]).astype(
            np.float32
        )

    out_root = _slerp_rotvec_sequence(root_orient_arr[:, None, :], source_times, target_times)[
        :, 0, :
    ]
    joints = pose_body_arr.shape[1] // 3
    pose_joint = pose_body_arr.reshape(source_frames, joints, 3)
    out_pose = _slerp_rotvec_sequence(pose_joint, source_times, target_times).reshape(
        out_frames, joints * 3
    )

    return (
        out_root.astype(np.float32),
        out_pose.astype(np.float32),
        out_trans.astype(np.float32),
        {
            "resample_policy": "slerp_rotation_linear_translation",
            "source_frame_count": source_frames,
            "canonical_frame_count": out_frames,
        },
    )


def _try_resample_motion_torch(
    *,
    root_orient_arr: object,
    pose_body_arr: object,
    trans_arr: object,
    source_fps: float,
    target_fps: float,
    out_frames: int,
) -> tuple[object, object, object, dict[str, object]] | None:
    backend = os.environ.get("HOLOSMPL_RESAMPLE_BACKEND", "auto").strip().lower()
    if backend in {"", "scipy", "numpy", "cpu_scipy", "off", "disable", "disabled"}:
        return None
    try:
        import numpy as np
        import torch
    except Exception:
        return None
    if backend in {"torch_cuda", "cuda", "gpu", "auto"}:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    elif backend in {"torch", "torch_cpu"}:
        device = torch.device("cpu")
    else:
        return None

    try:
        root_np = np.asarray(root_orient_arr, dtype=np.float32)
        pose_np = np.asarray(pose_body_arr, dtype=np.float32)
        trans_np = np.asarray(trans_arr, dtype=np.float32)
        source_frames = int(root_np.shape[0])
        joints = int(pose_np.shape[1] // 3)
        rot_np = np.concatenate(
            [root_np[:, None, :], pose_np.reshape(source_frames, joints, 3)],
            axis=1,
        )
        rot_dtype_text = os.environ.get("HOLOSMPL_TORCH_ROT_DTYPE", "float32").strip().lower()
        rot_dtype = torch.float64 if rot_dtype_text in {"float64", "double", "fp64"} else torch.float32
        with torch.no_grad():
            rot = torch.as_tensor(rot_np, dtype=rot_dtype, device=device)
            trans = torch.as_tensor(trans_np, dtype=torch.float32, device=device)
            # Keep the sampling clock in float64; long clips otherwise drift
            # beyond the accepted tolerance when the rotation math is fp32.
            target_index = (
                torch.arange(out_frames, dtype=torch.float64, device=device)
                * float(source_fps)
                / float(target_fps)
            )
            target_index = torch.clamp(target_index, 0.0, float(source_frames - 1))
            i0 = torch.floor(target_index).to(torch.long)
            i1 = torch.clamp(i0 + 1, max=source_frames - 1)
            alpha_index = target_index - i0.to(torch.float64)
            alpha = alpha_index.to(rot_dtype).view(out_frames, 1, 1)

            q0 = _torch_rotvec_to_quat_xyzw(rot.index_select(0, i0))
            q1 = _torch_rotvec_to_quat_xyzw(rot.index_select(0, i1))
            q = _torch_quat_slerp_xyzw(q0, q1, alpha)
            out_rot = _torch_quat_xyzw_to_rotvec(q)
            trans0 = trans.index_select(0, i0)
            trans1 = trans.index_select(0, i1)
            alpha_trans = alpha_index.to(torch.float32).view(out_frames, 1)
            out_trans = trans0 * (1.0 - alpha_trans) + trans1 * alpha_trans

            out_rot_cpu = out_rot.detach().cpu().numpy().astype(np.float32)
            out_trans_cpu = out_trans.detach().cpu().numpy().astype(np.float32)
        return (
            out_rot_cpu[:, 0, :],
            out_rot_cpu[:, 1:, :].reshape(out_frames, joints * 3),
            out_trans_cpu,
            {
                "resample_policy": (
                    f"torch_{device.type}_{str(rot_dtype).rsplit('.', maxsplit=1)[-1]}"
                    "_slerp_rotation_linear_translation"
                ),
                "source_frame_count": source_frames,
                "canonical_frame_count": out_frames,
            },
        )
    except Exception:
        if backend == "auto":
            return None
        raise


def _torch_rotvec_to_quat_xyzw(rotvec):
    import torch

    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    angle_sq = angle * angle
    small = angle < 1e-8
    scale = torch.where(
        small,
        0.5 - angle_sq / 48.0,
        torch.sin(half) / torch.clamp(angle, min=1e-12),
    )
    xyz = rotvec * scale
    w = torch.cos(half)
    quat = torch.cat([xyz, w], dim=-1)
    return torch.nn.functional.normalize(quat, dim=-1)


def _torch_quat_xyzw_to_rotvec(quat):
    import torch

    quat = torch.nn.functional.normalize(quat, dim=-1)
    quat = torch.where(quat[..., 3:4] < 0.0, -quat, quat)
    xyz = quat[..., :3]
    w = torch.clamp(quat[..., 3:4], -1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half < 1e-8, 2.0 + angle * angle / 12.0, angle / sin_half)
    return xyz * scale


def _torch_quat_slerp_xyzw(q0, q1, alpha):
    import torch

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)
    linear = dot > 0.9995
    q_lerp = torch.nn.functional.normalize(q0 + alpha * (q1 - q0), dim=-1)

    theta0 = torch.acos(dot)
    sin_theta0 = torch.sin(theta0)
    s0 = torch.sin((1.0 - alpha) * theta0) / torch.clamp(sin_theta0, min=1e-8)
    s1 = torch.sin(alpha * theta0) / torch.clamp(sin_theta0, min=1e-8)
    q_slerp = s0 * q0 + s1 * q1
    return torch.where(linear, q_lerp, torch.nn.functional.normalize(q_slerp, dim=-1))


def _slerp_rotvec_sequence(rotvec: object, source_times: object, target_times: object) -> object:
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    rotvec_arr = np.asarray(rotvec, dtype=np.float32)
    source_times_arr = np.asarray(source_times, dtype=np.float64)
    target_times_arr = np.asarray(target_times, dtype=np.float64)
    out = np.empty((len(target_times_arr), rotvec_arr.shape[1], 3), dtype=np.float32)
    for joint_idx in range(rotvec_arr.shape[1]):
        rotations = Rotation.from_rotvec(rotvec_arr[:, joint_idx, :])
        slerp = Slerp(source_times_arr, rotations)
        out[:, joint_idx, :] = slerp(target_times_arr).as_rotvec().astype(np.float32)
    return out
