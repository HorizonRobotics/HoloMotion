from __future__ import annotations

from typing import Any

import numpy as np

from holosmpl.visualization.body_model import BodyModelCache, BodyModelOutput
from holosmpl.visualization.canonical_loader import (
    CanonicalClip,
    evenly_spaced_indices,
)


def compute_geometry_summary(
    clip: CanonicalClip,
    body_cache: BodyModelCache,
    *,
    max_frames: int = 120,
    batch_size: int = 32,
) -> dict[str, Any]:
    frame_indices = evenly_spaced_indices(clip.frame_count, min(max_frames, clip.frame_count))
    output = body_cache.forward_clip_frames(clip, frame_indices, batch_size=batch_size)
    vertices = output.vertices
    joints = output.joints

    frame_bbox_min = vertices.min(axis=1)
    frame_bbox_max = vertices.max(axis=1)
    frame_extent = frame_bbox_max - frame_bbox_min
    all_bbox_min = vertices.reshape(-1, 3).min(axis=0)
    all_bbox_max = vertices.reshape(-1, 3).max(axis=0)
    lowest_vertex_z = vertices[:, :, 2].min(axis=1)
    root_height = clip.trans[:, 2]

    trans_speed = np.zeros((0,), dtype=np.float32)
    if clip.frame_count > 1:
        trans_speed = np.linalg.norm(np.diff(clip.trans, axis=0), axis=1) * float(clip.target_fps)

    metrics = {
        "model_type": output.model_type,
        "gender": output.gender,
        "visualized_frame_indices": frame_indices,
        "geometry_frame_policy": "evenly_sampled",
        "geometry_frame_count": len(frame_indices),
        "mesh_vertex_count": int(vertices.shape[1]),
        "mesh_face_count": int(output.faces.shape[0]),
        "joint_count": int(joints.shape[1]),
        "bbox_min": _round_list(all_bbox_min),
        "bbox_max": _round_list(all_bbox_max),
        "bbox_extent": _round_list(all_bbox_max - all_bbox_min),
        "body_height_min": _round_float(frame_extent[:, 2].min()),
        "body_height_max": _round_float(frame_extent[:, 2].max()),
        "body_height_median": _round_float(np.median(frame_extent[:, 2])),
        "root_height_min": _round_float(root_height.min()),
        "root_height_max": _round_float(root_height.max()),
        "root_height_median": _round_float(np.median(root_height)),
        "lowest_vertex_z_min": _round_float(lowest_vertex_z.min()),
        "lowest_vertex_z_max": _round_float(lowest_vertex_z.max()),
        "root_speed_max": _round_float(trans_speed.max()) if len(trans_speed) else 0.0,
        "root_speed_median": _round_float(np.median(trans_speed)) if len(trans_speed) else 0.0,
        "has_nan_or_inf": bool(
            (not np.isfinite(vertices).all())
            or (not np.isfinite(joints).all())
            or (not np.isfinite(clip.root_orient).all())
            or (not np.isfinite(clip.pose_body).all())
            or (not np.isfinite(clip.trans).all())
        ),
    }
    metrics["warnings"] = geometry_warnings(metrics)
    return metrics


def geometry_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if metrics["has_nan_or_inf"]:
        warnings.append("contains_nan_or_inf")
    if metrics["body_height_median"] < 1.0:
        warnings.append("body_height_median_too_small")
    if metrics["body_height_median"] > 2.4:
        warnings.append("body_height_median_too_large")
    if max(metrics["bbox_extent"]) > 30.0:
        warnings.append("bbox_extent_too_large")
    if metrics["root_speed_max"] > 10.0:
        warnings.append("root_speed_max_too_large")
    if metrics["lowest_vertex_z_min"] < -2.0:
        warnings.append("lowest_vertex_far_below_ground")
    return warnings


def _round_float(value: float | np.floating[Any]) -> float:
    return round(float(value), 6)


def _round_list(value: np.ndarray) -> list[float]:
    return [_round_float(x) for x in np.asarray(value).reshape(-1)]
