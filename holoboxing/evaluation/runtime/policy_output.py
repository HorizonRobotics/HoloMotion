"""Repair only nonfinite competitor target_pose components at the output boundary.

NaN/Inf positions become zero. A corrupted X/Z orientation slot uses identity
axes, since all-zero axes cannot represent a rotation. Finite invalid rotations
remain errors. Physical state, frozen features and caches never enter this API.
"""

import numpy as np


SANITIZATION_VERSION = "target_pose_nonfinite_zero_identity_axes_v1"


class CompetitorOutputError(ValueError):
    pass


def sanitize_target_pose(value):
    is_torch = type(value).__module__.startswith("torch")
    if is_torch:
        import torch

        if value.dtype != torch.float32 or tuple(value.shape) != (1, 225):
            raise CompetitorOutputError("Expected float32 target_pose[1,225]")
        invalid = ~torch.isfinite(value)
        cleaned = value.clone()
        cleaned[invalid] = 0
        slots = invalid.reshape(5, 5, 9)[..., 3:].any(dim=-1)
        points = cleaned.reshape(5, 5, 9)
        identity = value.new_tensor([1, 0, 0, 0, 0, 1])
        points[..., 3:] = torch.where(
            slots[..., None], identity, points[..., 3:]
        )
        x, z = points[..., 3:6], points[..., 6:9]
        valid_axes = (
            ((x.norm(dim=-1) - 1).abs() <= 1e-3).all()
            & ((z.norm(dim=-1) - 1).abs() <= 1e-3).all()
            & ((x * z).sum(-1).abs() <= 1e-3).all()
        )
        if not bool(valid_axes):
            raise CompetitorOutputError(
                "Finite target_pose X/Z must be orthogonal unit vectors"
            )
        count, orientations = (
            int(invalid.sum().item()),
            int(slots.sum().item()),
        )
    else:
        value = np.asarray(value)
        if value.dtype != np.float32 or value.shape != (1, 225):
            raise CompetitorOutputError("Expected float32 target_pose[1,225]")
        invalid = ~np.isfinite(value)
        cleaned = value.copy()
        cleaned[invalid] = 0
        slots = invalid.reshape(5, 5, 9)[..., 3:].any(axis=-1)
        points = cleaned.reshape(5, 5, 9)
        points[..., 3:][slots] = [1, 0, 0, 0, 0, 1]
        x, z = points[..., 3:6], points[..., 6:9]
        if not (
            np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-3, rtol=0)
            and np.allclose(np.linalg.norm(z, axis=-1), 1, atol=1e-3, rtol=0)
            and np.allclose((x * z).sum(-1), 0, atol=1e-3, rtol=0)
        ):
            raise CompetitorOutputError(
                "Finite target_pose X/Z must be orthogonal unit vectors"
            )
        count, orientations = int(invalid.sum()), int(slots.sum())
    return cleaned, dict(
        nonfinite_components=count, orientation_fallback_slots=orientations
    )


def run_competitor(session, observations):
    """Keep finite input and CUDA checks without rejecting output before repair."""
    from .onnx_device import verify_cuda_execution

    observations = np.asarray(observations)
    if (
        observations.dtype != np.float32
        or observations.shape != (1, 1097)
        or not np.isfinite(observations).all()
    ):
        raise ValueError("Expected finite measured float32 obs[1,1097]")
    outputs = session.run(["target_pose"], {"obs": observations})
    verify_cuda_execution(session)
    return outputs[0]
