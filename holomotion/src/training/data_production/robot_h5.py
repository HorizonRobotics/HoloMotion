"""Build training HDF5 references from HoloSMPL clips with HoloRetarget."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from holosmpl.converters.smpl_to_body_poses import SmplToBodyPosesConverter
from holoretarget.config import DEFAULT_CONFIG, HoloRetargetConfig
from holoretarget.online import HoloRetargeter
from holoretarget.schema import DOF_POS_DIM, QPOS_DIM


ROBOT_H5_ARRAY_NAMES = (
    "ref_root_pos",
    "ref_root_rot",
    "ref_dof_pos",
)
DEFAULT_MOTION_FPS = 50.0


@dataclass(frozen=True)
class HoloSmplClip:
    shard_path: Path
    clip_index: int
    motion_key: str
    metadata: dict[str, Any]
    pose_aa: np.ndarray
    root_trans: np.ndarray
    shape_beta: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.pose_aa.shape[0])


class RobotH5ShardWriter:
    """Small writer matching the existing HDF5 v2 training layout."""

    def __init__(
        self,
        path: str | Path,
        *,
        chunks_t: int = 1024,
        compression: str | None = "lzf",
    ) -> None:
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_t = max(1, int(chunks_t))
        self.compression = _normalize_compression(compression)
        self.handle = h5py.File(self.path, "w")
        self.datasets: dict[str, Any] = {}
        self.starts: list[int] = []
        self.lengths: list[int] = []
        self.motion_key_ids: list[int] = []
        self.metadata_json: list[str] = []
        self.frame_cursor = 0
        self.closed = False

    @property
    def clip_count(self) -> int:
        return len(self.lengths)

    @property
    def frame_count(self) -> int:
        return int(self.frame_cursor)

    def _create_dataset(self, name: str, tail: tuple[int, ...]):
        return self.handle.create_dataset(
            name,
            shape=(0, *tail),
            maxshape=(None, *tail),
            chunks=(self.chunks_t, *tail),
            compression=self.compression,
            shuffle=self.compression is not None,
            dtype=np.float32,
        )

    def append_motion(
        self,
        *,
        motion_id: int,
        arrays: dict[str, np.ndarray],
        metadata_json: str,
    ) -> tuple[int, int]:
        if self.closed:
            raise RuntimeError(f"cannot append to closed shard: {self.path}")
        _validate_robot_arrays(arrays)
        self._ensure_datasets(arrays)
        length = int(arrays["ref_dof_pos"].shape[0])
        start = self.frame_cursor
        end = start + length
        for name, dataset in self.datasets.items():
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end, ...] = arrays[name]
        self.starts.append(start)
        self.lengths.append(length)
        self.motion_key_ids.append(int(motion_id))
        self.metadata_json.append(str(metadata_json))
        self.frame_cursor = end
        return start, length

    def _ensure_datasets(self, arrays: dict[str, np.ndarray]) -> None:
        if not self.datasets:
            self.datasets = {
                name: self._create_dataset(name, tuple(arrays[name].shape[1:]))
                for name in ROBOT_H5_ARRAY_NAMES
            }
            return
        for name, dataset in self.datasets.items():
            if tuple(dataset.shape[1:]) != tuple(arrays[name].shape[1:]):
                raise ValueError(
                    f"{name} shape changed within shard: "
                    f"expected (*,{dataset.shape[1:]}), got {arrays[name].shape}"
                )

    def flush(self) -> None:
        self.handle.flush()

    def finalize(self) -> dict[str, Any]:
        import h5py

        if self.closed:
            raise RuntimeError(f"shard already closed: {self.path}")
        clips_group = self.handle.create_group("clips")
        clips_group.create_dataset(
            "start", data=np.asarray(self.starts, dtype=np.int64)
        )
        clips_group.create_dataset(
            "length", data=np.asarray(self.lengths, dtype=np.int64)
        )
        clips_group.create_dataset(
            "motion_key_id",
            data=np.asarray(self.motion_key_ids, dtype=np.int64),
        )
        clips_group.create_dataset(
            "metadata_json",
            data=np.asarray(self.metadata_json, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.handle.flush()
        self.handle.close()
        self.closed = True
        return {
            "file": str(self.path),
            "num_clips": self.clip_count,
            "num_frames": self.frame_count,
        }


def retarget_holosmpl_h5_to_robot_h5(
    *,
    holosmpl_h5_root: str | Path,
    output_root: str | Path,
    config: HoloRetargetConfig | None = None,
    overwrite: bool = False,
    compression: str | None = "lzf",
    chunks_t: int = 1024,
    shard_target_frames: int = 250_000,
    progress_interval: int = 10,
) -> dict[str, Any]:
    """Convert HoloSMPL H5 shards into the robot training H5 v2 format."""

    holosmpl_h5_root = Path(holosmpl_h5_root)
    output_root = Path(output_root)
    config = config or DEFAULT_CONFIG
    progress_interval = max(1, int(progress_interval))
    shard_target_frames = max(1, int(shard_target_frames))

    manifest = _read_json(holosmpl_h5_root / "manifest.json")
    clips = list(iter_holosmpl_h5_clips(holosmpl_h5_root))
    if not clips:
        raise ValueError(f"no HoloSMPL clips found under {holosmpl_h5_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} exists; pass overwrite=True to replace it"
            )
        shutil.rmtree(output_root)
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    retargeter = HoloRetargeter(config)
    body_pose_converter = SmplToBodyPosesConverter(retargeter)
    hdf5_shards: list[dict[str, Any]] = []
    clips_manifest: dict[str, dict[str, Any]] = {}
    motion_keys: list[str] = []
    motion_key_to_id: dict[str, int] = {}
    writer: RobotH5ShardWriter | None = None
    shard_index = 0
    total_frames = 0
    start_time = time.monotonic()

    for index, clip in enumerate(clips, start=1):
        if writer is None or (
            writer.clip_count > 0
            and writer.frame_count + clip.frame_count > shard_target_frames
        ):
            if writer is not None:
                hdf5_shards.append(_finalize_robot_shard(writer, output_root))
                shard_index += 1
            writer = RobotH5ShardWriter(
                shard_root / f"holoretarget_{shard_index:06d}.h5",
                chunks_t=chunks_t,
                compression=compression,
            )

        motion_id = motion_key_to_id.get(clip.motion_key)
        if motion_id is None:
            motion_id = len(motion_keys)
            motion_key_to_id[clip.motion_key] = motion_id
            motion_keys.append(clip.motion_key)

        arrays = retarget_holosmpl_clip_to_robot_arrays(
            retargeter=retargeter,
            body_pose_converter=body_pose_converter,
            clip=clip,
        )
        metadata = _robot_metadata(clip)
        start, length = writer.append_motion(
            motion_id=motion_id,
            arrays=arrays,
            metadata_json=json.dumps(
                metadata, ensure_ascii=False, sort_keys=True
            ),
        )
        clips_manifest[clip.motion_key] = {
            "motion_key": clip.motion_key,
            "shard": shard_index,
            "clip_idx": writer.clip_count - 1,
            "start": int(start),
            "length": int(length),
            "available_arrays": list(ROBOT_H5_ARRAY_NAMES),
            "metadata": metadata,
        }
        total_frames += length
        if index == len(clips) or index % progress_interval == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(
                f"[holoretarget-h5] {index}/{len(clips)} clips | frames={total_frames} "
                f"| {index / elapsed:.2f} clips/s | elapsed={elapsed:.1f}s",
                flush=True,
            )
    if writer is not None:
        hdf5_shards.append(_finalize_robot_shard(writer, output_root))

    output_manifest = {
        "version": 2,
        "root": str(output_root),
        "hdf5_shards": hdf5_shards,
        "clips": clips_manifest,
        "motion_keys": motion_keys,
        "dof_names": list(retargeter.dof_names),
        "array_names": list(ROBOT_H5_ARRAY_NAMES),
        "chunks_t": int(chunks_t),
        "compression": _normalize_compression_name(compression),
        "source_schema": str(manifest.get("schema_version")),
        "source_holosmpl_h5_root": str(holosmpl_h5_root),
        "retarget_backend": "HoloRetarget",
        "retarget_config": _config_to_manifest(config),
    }
    _write_json(output_root / "manifest.json", output_manifest)
    _write_json(output_root / "nan_npz_paths.json", [])
    return output_manifest


def iter_holosmpl_h5_clips(holosmpl_h5_root: str | Path):
    """Yield HoloSMPL clips from packed H5 shards."""

    import h5py

    root = Path(holosmpl_h5_root)
    manifest = _read_json(root / "manifest.json")
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError("HoloSMPL manifest field 'shards' must be a list")
    for shard in shards:
        rel = shard.get("path") or shard.get("file")
        if not rel:
            raise ValueError(
                f"HoloSMPL shard entry missing path/file: {shard}"
            )
        shard_path = root / str(rel)
        with h5py.File(shard_path, "r") as handle:
            pose_all = np.asarray(handle["human_pose_aa"], dtype=np.float32)
            trans_all = np.asarray(
                handle["human_root_trans"], dtype=np.float32
            )
            clips_group = handle["clips"]
            starts = np.asarray(clips_group["start"][:], dtype=np.int64)
            lengths = np.asarray(clips_group["length"][:], dtype=np.int64)
            motion_keys = [
                _decode_string(x) for x in clips_group["motion_key_id"][:]
            ]
            metadata_json = [
                _decode_string(x) for x in clips_group["metadata_json"][:]
            ]
            for clip_index, (start, length) in enumerate(zip(starts, lengths)):
                end = int(start) + int(length)
                beta = _read_clip_shape_beta(
                    handle,
                    clip_index=int(clip_index),
                    start=int(start),
                    end=end,
                    clip_count=len(starts),
                    total_frames=int(pose_all.shape[0]),
                )
                yield HoloSmplClip(
                    shard_path=shard_path,
                    clip_index=clip_index,
                    motion_key=str(motion_keys[clip_index]),
                    metadata=json.loads(metadata_json[clip_index]),
                    pose_aa=pose_all[int(start) : end].copy(),
                    root_trans=trans_all[int(start) : end].copy(),
                    shape_beta=_normalize_shape_beta(
                        beta, frame_count=int(length)
                    ),
                )


def retarget_holosmpl_clip_to_robot_arrays(
    *,
    retargeter: HoloRetargeter,
    body_pose_converter: SmplToBodyPosesConverter,
    clip: HoloSmplClip,
) -> dict[str, np.ndarray]:
    """Run one HoloSMPL clip through the same path used by online deployment."""

    _validate_holosmpl_clip(clip)
    frame_count = clip.frame_count
    reference_qpos = np.empty((frame_count, QPOS_DIM), dtype=np.float32)
    retargeter.reset_sequence()
    for frame in range(frame_count):
        body_poses = body_pose_converter(
            transl=clip.root_trans[frame],
            global_orient_aa=clip.pose_aa[frame, :3],
            body_pose_aa=clip.pose_aa[frame, 3:],
        )
        reference_qpos[frame] = retargeter.retarget_qpos_from_body_poses(
            body_poses
        )

    ref_root_pos = reference_qpos[:, :3]
    ref_root_rot = reference_qpos[:, [4, 5, 6, 3]]
    ref_dof_pos = reference_qpos[:, 7:]
    return {
        "ref_root_pos": ref_root_pos,
        "ref_root_rot": ref_root_rot,
        "ref_dof_pos": ref_dof_pos,
    }


def _robot_metadata(
    clip: HoloSmplClip,
) -> dict[str, Any]:
    metadata = dict(clip.metadata)
    fps = float(metadata.get("target_fps") or DEFAULT_MOTION_FPS)
    metadata.update(
        {
            "schema_version": "holoretarget_robot_h5_clip_v2",
            "num_frames": clip.frame_count,
            "motion_fps": fps,
            "retarget_backend": "HoloRetarget",
            "retarget_source": "HoloSMPL",
            "source_holosmpl_shard": str(clip.shard_path),
            "source_holosmpl_clip_index": int(clip.clip_index),
        }
    )
    return metadata


def _validate_holosmpl_clip(clip: HoloSmplClip) -> None:
    if clip.pose_aa.ndim != 2 or clip.pose_aa.shape[1] != 72:
        raise ValueError(
            f"human_pose_aa must be [T,72], got {clip.pose_aa.shape}"
        )
    if clip.root_trans.shape != (clip.pose_aa.shape[0], 3):
        raise ValueError(
            f"human_root_trans must be [T,3], got {clip.root_trans.shape}"
        )
    if clip.shape_beta.ndim != 1:
        raise ValueError(
            f"human_shape_beta must be [B], got {clip.shape_beta.shape}"
        )
    for name, array in {
        "human_pose_aa": clip.pose_aa,
        "human_root_trans": clip.root_trans,
        "human_shape_beta": clip.shape_beta,
    }.items():
        if array.dtype != np.float32:
            raise ValueError(
                f"{name} dtype must be float32, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")


def _normalize_shape_beta(
    beta: np.ndarray, *, frame_count: int | None = None
) -> np.ndarray:
    """Return clip-level beta, accepting legacy per-frame beta arrays."""

    beta = np.asarray(beta, dtype=np.float32)
    if beta.ndim == 1:
        return beta.copy()
    if beta.ndim == 2:
        if beta.shape[0] <= 0:
            raise ValueError(f"human_shape_beta has no frames: {beta.shape}")
        if frame_count is not None and beta.shape[0] not in {
            1,
            int(frame_count),
        }:
            raise ValueError(
                "human_shape_beta [T,B] frame count mismatch: "
                f"expected {frame_count}, got {beta.shape}"
            )
        return beta[0].astype(np.float32, copy=True)
    raise ValueError(
        f"human_shape_beta must be [B] or [T,B], got {beta.shape}"
    )


def _read_clip_shape_beta(
    handle: Any,
    *,
    clip_index: int,
    start: int,
    end: int,
    clip_count: int,
    total_frames: int,
) -> np.ndarray:
    """Read shape beta from current HoloSMPL or legacy frame-major layouts."""

    clips_group = handle["clips"]
    if "human_shape_beta" in clips_group:
        return np.asarray(
            clips_group["human_shape_beta"][clip_index], dtype=np.float32
        )
    if "human_shape_beta" not in handle:
        raise KeyError("missing human_shape_beta in HoloSMPL H5 shard")

    dataset = handle["human_shape_beta"]
    if dataset.ndim == 1:
        return np.asarray(dataset[:], dtype=np.float32)
    if dataset.ndim == 2:
        if int(dataset.shape[0]) == int(total_frames):
            return np.asarray(dataset[start:end], dtype=np.float32)
        if int(dataset.shape[0]) == int(clip_count):
            return np.asarray(dataset[clip_index], dtype=np.float32)
        if int(dataset.shape[0]) == 1:
            return np.asarray(dataset[0], dtype=np.float32)
    raise ValueError(
        "unsupported human_shape_beta layout: "
        f"shape={dataset.shape}, clip_count={clip_count}, total_frames={total_frames}"
    )


def _validate_robot_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected_shapes = {
        "ref_root_pos": (3,),
        "ref_root_rot": (4,),
        "ref_dof_pos": (DOF_POS_DIM,),
    }
    frame_count = None
    for name, tail_shape in expected_shapes.items():
        if name not in arrays:
            raise KeyError(f"missing robot array: {name}")
        array = np.asarray(arrays[name])
        if array.ndim != len(tail_shape) + 1:
            raise ValueError(f"{name} has invalid shape: {array.shape}")
        for axis, expected_size in enumerate(tail_shape, start=1):
            if (
                expected_size is not None
                and array.shape[axis] != expected_size
            ):
                raise ValueError(f"{name} has invalid shape: {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(
                f"{name} dtype must be float32, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        frame_count = array.shape[0] if frame_count is None else frame_count
        if int(array.shape[0]) != int(frame_count):
            raise ValueError(f"{name} frame count mismatch")


def _finalize_robot_shard(
    writer: RobotH5ShardWriter,
    output_root: Path,
) -> dict[str, Any]:
    summary = writer.finalize()
    return {
        "file": Path(summary["file"]).relative_to(output_root).as_posix(),
        "num_clips": int(summary["num_clips"]),
        "num_frames": int(summary["num_frames"]),
    }


def _config_to_manifest(config: HoloRetargetConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }


def _normalize_compression(compression: str | None) -> str | None:
    value = _normalize_compression_name(compression)
    return None if value == "none" else value


def _normalize_compression_name(compression: str | None) -> str:
    if compression is None:
        return "none"
    value = str(compression).strip().lower()
    return "none" if value in {"", "none", "null"} else value


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
