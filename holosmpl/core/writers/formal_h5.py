from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FORMAL_H5_ARRAY_FIELDS = (
    "human_pose_aa",
    "human_root_trans",
    "human_root_height",
    "human_gravity_projection",
)
FORMAL_H5_CLIP_FIELDS = (
    "clips/human_shape_beta",
)


class FormalH5ShardWriter:
    """Streaming writer for one frame-major formal H5 shard."""

    def __init__(
        self,
        path: str | Path,
        *,
        beta_dim: int,
        chunks_t: int = 1024,
        compression: str | None = "gzip",
    ) -> None:
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.beta_dim = int(beta_dim)
        self.chunks_t = max(1, int(chunks_t))
        self.compression = _normalize_compression(compression)
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["schema_version"] = "formal_h5_v1"
        self.handle.attrs["layout"] = "frame_major_flat"

        self.datasets = {
            "human_pose_aa": self._create_dataset("human_pose_aa", (72,)),
            "human_root_trans": self._create_dataset("human_root_trans", (3,)),
            "human_root_height": self._create_dataset("human_root_height", (1,)),
            "human_gravity_projection": self._create_dataset(
                "human_gravity_projection", (3,)
            ),
        }
        self.starts: list[int] = []
        self.lengths: list[int] = []
        self.motion_key_ids: list[str] = []
        self.metadata_json: list[str] = []
        self.formal_relative_paths: list[str] = []
        self.human_shape_betas: list[np.ndarray] = []
        self.frame_cursor = 0
        self.closed = False

    @property
    def clip_count(self) -> int:
        return len(self.lengths)

    @property
    def frame_count(self) -> int:
        return int(self.frame_cursor)

    def _create_dataset(self, name: str, tail: tuple[int, ...]):
        chunks = (self.chunks_t, *tail)
        return self.handle.create_dataset(
            name,
            shape=(0, *tail),
            maxshape=(None, *tail),
            chunks=chunks,
            compression=self.compression,
            shuffle=True if self.compression else False,
            dtype=np.float32,
        )

    def append_clip(self, clip: dict[str, Any]) -> tuple[int, int]:
        if self.closed:
            raise RuntimeError(f"cannot append to closed shard: {self.path}")
        _validate_clip_arrays([clip])
        t = int(clip["human_pose_aa"].shape[0])
        start = self.frame_cursor
        end = start + t
        for field, dataset in self.datasets.items():
            array = np.asarray(clip[field], dtype=np.float32)
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end, ...] = array
        self.starts.append(start)
        self.lengths.append(t)
        self.motion_key_ids.append(str(clip["clip_id"]))
        self.metadata_json.append(str(clip["metadata_json"]))
        self.formal_relative_paths.append(str(clip["formal_relative_path"]))
        self.human_shape_betas.append(np.asarray(clip["human_shape_beta"], dtype=np.float32))
        self.frame_cursor = end
        return start, t

    def flush(self) -> None:
        self.handle.flush()

    def finalize(self) -> dict[str, Any]:
        import h5py

        if self.closed:
            raise RuntimeError(f"shard already closed: {self.path}")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        clips_group = self.handle.create_group("clips")
        clips_group.create_dataset("start", data=np.asarray(self.starts, dtype=np.int64))
        clips_group.create_dataset("length", data=np.asarray(self.lengths, dtype=np.int64))
        clips_group.create_dataset(
            "motion_key_id",
            data=np.asarray(self.motion_key_ids, dtype=object),
            dtype=string_dtype,
        )
        clips_group.create_dataset(
            "metadata_json",
            data=np.asarray(self.metadata_json, dtype=object),
            dtype=string_dtype,
        )
        clips_group.create_dataset(
            "formal_relative_path",
            data=np.asarray(self.formal_relative_paths, dtype=object),
            dtype=string_dtype,
        )
        clips_group.create_dataset(
            "human_shape_beta",
            data=np.asarray(self.human_shape_betas, dtype=np.float32),
        )
        self.handle.attrs["frame_count"] = self.frame_count
        self.handle.attrs["clip_count"] = self.clip_count
        self.handle.flush()
        self.handle.close()
        self.closed = True
        return {
            "path": str(self.path),
            "clip_count": self.clip_count,
            "frame_count": self.frame_count,
            "starts": list(self.starts),
            "lengths": list(self.lengths),
        }


def write_formal_h5_shard(
    path: str | Path,
    clips: list[dict[str, Any]],
    *,
    compression: str | None = "gzip",
) -> dict[str, Any]:
    """Write one formal H5 shard with frame-major flat arrays and clip indices."""

    if not clips:
        raise ValueError("clips must not be empty")

    path = Path(path)
    _validate_clip_arrays(clips)
    beta_dim = int(clips[0]["human_shape_beta"].shape[0])
    writer = FormalH5ShardWriter(
        path,
        beta_dim=beta_dim,
        chunks_t=max(int(clip["human_pose_aa"].shape[0]) for clip in clips),
        compression=compression,
    )
    for clip in clips:
        writer.append_clip(clip)
    return writer.finalize()


def _validate_clip_arrays(clips: list[dict[str, Any]]) -> None:
    beta_dim = None
    for clip in clips:
        frame_count = int(clip["human_pose_aa"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"clip {clip.get('clip_id')} has zero frames")
        expected_shapes = {
            "human_pose_aa": (frame_count, 72),
            "human_root_trans": (frame_count, 3),
            "human_root_height": (frame_count, 1),
            "human_gravity_projection": (frame_count, 3),
        }
        for field, expected_shape in expected_shapes.items():
            array = clip[field]
            if array.shape != expected_shape:
                raise ValueError(
                    f"{clip.get('clip_id')} {field} shape {array.shape} != {expected_shape}"
                )
            if array.dtype != np.float32:
                raise ValueError(f"{clip.get('clip_id')} {field} dtype must be float32")
            if not np.isfinite(array).all():
                raise ValueError(f"{clip.get('clip_id')} {field} contains NaN or Inf")
        shape_beta = clip["human_shape_beta"]
        if shape_beta.ndim != 1:
            raise ValueError(
                f"{clip.get('clip_id')} human_shape_beta must be [B], got {shape_beta.shape}"
            )
        if shape_beta.dtype != np.float32:
            raise ValueError(f"{clip.get('clip_id')} human_shape_beta dtype must be float32")
        if not np.isfinite(shape_beta).all():
            raise ValueError(f"{clip.get('clip_id')} human_shape_beta contains NaN or Inf")
        beta_dim = int(shape_beta.shape[0]) if beta_dim is None else beta_dim
        if int(shape_beta.shape[0]) != beta_dim:
            raise ValueError(
                f"beta dim mismatch: expected {beta_dim}, got {shape_beta.shape[0]}"
            )


def _normalize_compression(compression: str | None) -> str | None:
    if compression is None:
        return None
    value = str(compression).strip().lower()
    if value in {"", "none", "null"}:
        return None
    return value
