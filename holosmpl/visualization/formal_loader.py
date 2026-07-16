from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holosmpl.visualization.canonical_loader import CanonicalClip


def load_formal_clip_as_render_clip(path: str | Path) -> CanonicalClip:
    """Load formal NPZ and expose it through the render clip interface."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        human_pose_aa = np.asarray(data["human_pose_aa"], dtype=np.float32)
        human_shape_beta = np.asarray(data["human_shape_beta"], dtype=np.float32)
        human_root_trans = np.asarray(data["human_root_trans"], dtype=np.float32)
        metadata = json.loads(str(data["metadata"].item()))

    if human_pose_aa.ndim != 2 or human_pose_aa.shape[1] != 72:
        raise ValueError(f"human_pose_aa must be [T,72], got {human_pose_aa.shape}")
    if human_shape_beta.ndim != 1:
        raise ValueError(
            "human_shape_beta must be clip-level [B], "
            f"got {human_shape_beta.shape} and {human_pose_aa.shape}"
        )
    if human_root_trans.ndim != 2 or human_root_trans.shape != (human_pose_aa.shape[0], 3):
        raise ValueError(
            "human_root_trans must be [T,3] with the same T as human_pose_aa, "
            f"got {human_root_trans.shape} and {human_pose_aa.shape}"
        )

    layout = str(metadata.get("canonical_pose_body_layout"))
    if layout == "smplx_21_body":
        pose_body = human_pose_aa[:, 3:66]
    elif layout == "smpl_23_body":
        pose_body = human_pose_aa[:, 3:72]
    else:
        raise ValueError(f"unsupported canonical_pose_body_layout for rendering: {layout}")

    render_metadata = dict(metadata)
    render_metadata["pose_body_layout"] = layout
    render_metadata["render_source"] = "formal_npz"
    return CanonicalClip(
        path=path,
        clip_id=str(metadata.get("clip_id") or path.stem),
        root_orient=human_pose_aa[:, :3].astype(np.float32, copy=True),
        pose_body=pose_body.astype(np.float32, copy=True),
        trans=human_root_trans.astype(np.float32, copy=True),
        betas=human_shape_beta.astype(np.float32, copy=True),
        gender=_load_linked_canonical_gender(metadata),
        source_fps=float(metadata.get("source_fps") or metadata.get("target_fps") or 50.0),
        target_fps=float(metadata.get("target_fps") or 50.0),
        metadata=render_metadata,
    )


def _load_linked_canonical_gender(metadata: dict[str, object]) -> str:
    canonical_path = metadata.get("canonical_path")
    if not canonical_path:
        return "neutral"
    path = Path(str(canonical_path))
    if not path.is_file():
        return "neutral"
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["gender"])
    except Exception:
        return "neutral"
