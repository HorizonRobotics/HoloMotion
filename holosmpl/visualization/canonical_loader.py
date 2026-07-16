from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CanonicalClip:
    path: Path
    clip_id: str
    root_orient: np.ndarray
    pose_body: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str
    source_fps: float
    target_fps: float
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.root_orient.shape[0])

    @property
    def pose_body_layout(self) -> str:
        return str(self.metadata["pose_body_layout"])


def load_canonical_clip(path: str | Path) -> CanonicalClip:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        clip_id = str(metadata.get("clip_id") or path.stem)
        return CanonicalClip(
            path=path,
            clip_id=clip_id,
            root_orient=np.asarray(data["root_orient"], dtype=np.float32),
            pose_body=np.asarray(data["pose_body"], dtype=np.float32),
            trans=np.asarray(data["trans"], dtype=np.float32),
            betas=np.asarray(data["betas"], dtype=np.float32),
            gender=str(data["gender"]),
            source_fps=float(data["source_fps"]),
            target_fps=float(data["target_fps"]),
            metadata=metadata,
        )


def evenly_spaced_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if frame_count == 1:
        return [0]
    count = min(frame_count, count)
    return [int(round(x)) for x in np.linspace(0, frame_count - 1, count)]
