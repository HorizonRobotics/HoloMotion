# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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


import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
from loguru import logger
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class ArraySpec:
    name: str
    shape_tail: Tuple[int, ...]  # shape excluding time dim
    dtype: np.dtype


@dataclass
class ClipEntry:
    clip_id: int
    name: str
    path: str


class Hdf5ShardWriter:
    def __init__(
        self,
        h5_path: str,
        array_specs: List[ArraySpec],
        chunks_t: int,
        compression: str,
    ) -> None:
        self.h5_path = h5_path
        self.array_specs = array_specs
        self.chunks_t = int(chunks_t)
        self.compression = compression

        _ensure_dir(os.path.dirname(self.h5_path))
        self.h5 = h5py.File(self.h5_path, "w")

        self.datasets: Dict[str, h5py.Dataset] = {}
        for spec in self.array_specs:
            chunks = (self.chunks_t, *spec.shape_tail)
            maxshape = (None, *spec.shape_tail)
            ds = self.h5.create_dataset(
                spec.name,
                shape=(0, *spec.shape_tail),
                maxshape=maxshape,
                chunks=chunks,
                compression=(
                    self.compression if self.compression != "none" else None
                ),
                dtype=spec.dtype,
                shuffle=True if self.compression != "none" else False,
            )
            self.datasets[spec.name] = ds

        self._clip_starts: List[int] = []
        self._clip_lengths: List[int] = []
        self._clip_motion_ids: List[int] = []
        self._clip_metadata: List[str] = []

        self.t_cursor = 0

    def append_motion(
        self,
        motion_id: int,
        np_arrays: Dict[str, np.ndarray],
        metadata_json: str,
    ) -> Tuple[int, int]:
        if "ref_dof_pos" not in np_arrays:
            raise KeyError("ref_dof_pos missing for HDF5 v2 packing")
        t_len = int(np_arrays["ref_dof_pos"].shape[0])

        start = self.t_cursor
        end = start + t_len
        for spec in self.array_specs:
            if spec.name not in np_arrays:
                raise KeyError(
                    f"Missing array '{spec.name}' for HDF5 v2 packing"
                )
            ds = self.datasets[spec.name]
            ds.resize((end, *spec.shape_tail))
            ds[start:end, ...] = np_arrays[spec.name]

        self._clip_starts.append(start)
        self._clip_lengths.append(t_len)
        self._clip_motion_ids.append(motion_id)
        self._clip_metadata.append(metadata_json)
        self.t_cursor = end
        return start, t_len

    def finalize(self) -> Dict[str, Any]:
        g = self.h5.create_group("clips")
        g.create_dataset(
            "start", data=np.asarray(self._clip_starts, dtype=np.int64)
        )
        g.create_dataset(
            "length", data=np.asarray(self._clip_lengths, dtype=np.int64)
        )
        g.create_dataset(
            "motion_key_id",
            data=np.asarray(self._clip_motion_ids, dtype=np.int64),
        )
        vlen_str = h5py.string_dtype(encoding="utf-8")
        g.create_dataset(
            "metadata_json",
            data=np.asarray(self._clip_metadata, dtype=vlen_str),
        )

        summary = {
            "file": self.h5_path,
            "num_clips": len(self._clip_starts),
            "num_frames": int(self.t_cursor),
        }
        self.h5.flush()
        self.h5.close()
        return summary


def _normalize_root_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [str(value)]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in list(value)]
    return [str(value)]


def _discover_motion_entries(roots: List[str]) -> List[ClipEntry]:
    motion_key_to_path: Dict[str, str] = {}
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        parent_dir_name = root_path.name
        clips_dir = root_path / "clips"
        base_dir = clips_dir if clips_dir.is_dir() else root_path
        if not base_dir.is_dir():
            raise FileNotFoundError(f"NPZ directory not found: {base_dir}")
        for dirpath, _, filenames in os.walk(str(base_dir)):
            for fname in filenames:
                if not fname.endswith(".npz"):
                    continue
                base_key = os.path.splitext(fname)[0]
                motion_key = f"{parent_dir_name}_{base_key}"
                npz_path = os.path.join(dirpath, fname)
                if motion_key in motion_key_to_path:
                    raise ValueError(f"Duplicate motion key: {motion_key}")
                motion_key_to_path[motion_key] = npz_path

    entries = [
        ClipEntry(clip_id=i, name=key, path=motion_key_to_path[key])
        for i, key in enumerate(sorted(motion_key_to_path.keys()))
    ]
    if len(entries) == 0:
        raise ValueError("No NPZ files found in input directories.")
    return entries


def _load_metadata_json(npz_path: Path) -> Tuple[str, Dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "metadata" not in data:
            raise KeyError(f"'metadata' missing in NPZ: {npz_path}")
        metadata_json = str(data["metadata"])
        num_frames_from_dof = data["ref_dof_pos"].shape[0]
    metadata = json.loads(metadata_json)
    num_frames_from_metadata = metadata["num_frames"]
    assert num_frames_from_dof == num_frames_from_metadata, (
        f"num_frames_from_dof {num_frames_from_dof} != num_frames_from_metadata {num_frames_from_metadata} in {npz_path}"
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a JSON object in {npz_path}")
    return metadata_json, metadata


def _cast_array(array: np.ndarray, name: str, npz_path: Path) -> np.ndarray:
    if array.dtype == np.float32:
        return array
    if array.dtype.kind == "O":
        raise ValueError(f"Array '{name}' in {npz_path} has object dtype.")
    if np.issubdtype(array.dtype, np.integer):
        logger.warning(
            "Casting array '{}' in {} from {} to float32.",
            name,
            npz_path,
            array.dtype,
        )
        return array.astype(np.float32, copy=False)
    raise ValueError(
        f"Array '{name}' in {npz_path} has dtype {array.dtype}, "
        "expected float32 or integer."
    )


def _discover_array_specs(first_npz: Path) -> List[ArraySpec]:
    with np.load(first_npz, allow_pickle=False) as data:
        if "ref_dof_pos" not in data:
            raise KeyError(f"'ref_dof_pos' missing in NPZ: {first_npz}")
        if "ref_global_translation" not in data:
            raise KeyError(
                f"'ref_global_translation' missing in NPZ: {first_npz}"
            )
        if "ref_global_rotation_quat" not in data:
            raise KeyError(
                f"'ref_global_rotation_quat' missing in NPZ: {first_npz}"
            )
        dof_pos = data["ref_dof_pos"]
        global_trans = data["ref_global_translation"]
        global_rot = data["ref_global_rotation_quat"]
        if dof_pos.ndim < 2:
            raise ValueError(f"'ref_dof_pos' must be (T, ndof) in {first_npz}")
        if global_trans.ndim < 2 or global_trans.shape[-1] != 3:
            raise ValueError(
                f"'ref_global_translation' must end with 3 in {first_npz}"
            )
        if global_rot.ndim < 2 or global_rot.shape[-1] != 4:
            raise ValueError(
                f"'ref_global_rotation_quat' must end with 4 in {first_npz}"
            )
        dof_tail = tuple(dof_pos.shape[1:])
    return [
        ArraySpec(name="ref_dof_pos", shape_tail=dof_tail, dtype=np.float32),
        ArraySpec(name="ref_root_pos", shape_tail=(3,), dtype=np.float32),
        ArraySpec(name="ref_root_rot", shape_tail=(4,), dtype=np.float32),
    ]


def _load_npz_arrays(
    npz_path: Path,
    num_frames: int,
    dof_tail: Tuple[int, ...],
) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        dof_pos = _cast_array(data["ref_dof_pos"], "ref_dof_pos", npz_path)
        global_trans = _cast_array(
            data["ref_global_translation"], "ref_global_translation", npz_path
        )
        global_rot = _cast_array(
            data["ref_global_rotation_quat"],
            "ref_global_rotation_quat",
            npz_path,
        )

    if global_trans.ndim == 2:
        root_pos = global_trans
    elif global_trans.ndim >= 3:
        root_pos = global_trans[:, 0, :]
    else:
        raise ValueError(
            f"ref_global_translation must be (T,3) or (T,B,3) in {npz_path}"
        )
    if global_rot.ndim == 2:
        root_rot = global_rot
    elif global_rot.ndim >= 3:
        root_rot = global_rot[:, 0, :]
    else:
        raise ValueError(
            f"ref_global_rotation_quat must be (T,4) or (T,B,4) in {npz_path}"
        )

    expected_dof_shape = (num_frames, *dof_tail)
    if dof_pos.shape != expected_dof_shape:
        raise ValueError(
            f"ref_dof_pos shape {dof_pos.shape} does not match {expected_dof_shape} "
            f"in {npz_path}"
        )
    if root_pos.shape != (num_frames, 3):
        raise ValueError(
            f"ref_root_pos shape {root_pos.shape} does not match {(num_frames, 3)} "
            f"in {npz_path}"
        )
    if root_rot.shape != (num_frames, 4):
        raise ValueError(
            f"ref_root_rot shape {root_rot.shape} does not match {(num_frames, 4)} "
            f"in {npz_path}"
        )

    return {
        "ref_dof_pos": dof_pos,
        "ref_root_pos": root_pos,
        "ref_root_rot": root_rot,
    }


def _relative_npz_path(npz_path: Path, roots: List[str]) -> str:
    npz_path = npz_path.expanduser().resolve()
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            rel = npz_path.relative_to(root_path)
        except ValueError:
            continue
        return str(Path(root_path.name) / rel)
    return str(npz_path)


def _nan_array_names(arrays: Dict[str, np.ndarray]) -> List[str]:
    nan_names: List[str] = []
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.floating):
            continue
        if np.isnan(array).any():
            nan_names.append(name)
            return nan_names
    return []


def _estimate_bytes_for_motion(npz_path: Path, mode: str) -> int:
    """Estimate per-clip byte contribution for shard sizing.

    Note:
    - ``uncompressed_nbytes`` matches the in-memory float32 payload size and does
      *not* correspond to on-disk shard size when compression is enabled.
    - ``npz_filesize`` uses the compressed input file size as a cheap proxy for
      on-disk shard size.
    - ``h5_filesize`` does not use this estimator (it measures actual shard size
      after writes).
    """
    mode_norm = str(mode).lower().strip()
    if mode_norm in ("uncompressed_nbytes", "nbytes", "uncompressed"):
        with np.load(npz_path, allow_pickle=False) as data:
            total = 0
            for key in (
                "ref_dof_pos",
                "ref_global_translation",
                "ref_global_rotation_quat",
            ):
                if key in data:
                    total += int(data[key].nbytes)
        return total
    if mode_norm in ("npz_filesize", "npz_size", "npz_bytes"):
        return int(npz_path.stat().st_size)
    raise ValueError(
        f"Unsupported shard_target_mode '{mode}'. Expected one of: "
        "uncompressed_nbytes | npz_filesize | h5_filesize"
    )


@hydra.main(
    config_path="../../config",
    config_name="motion_retargeting/pack_hdf5_v2",
    version_base=None,
)
def main(cfg: OmegaConf) -> None:
    roots = _normalize_root_list(cfg.get("holomotion_npz_root", None))
    if len(roots) == 0:
        roots = _normalize_root_list(
            cfg.get("holomotion_retargeted_dirs", None)
        )
    if len(roots) == 0:
        legacy_root = cfg.get("precomputed_npz_root", None)
        roots = _normalize_root_list(legacy_root)
    if len(roots) == 0:
        raise ValueError("holomotion_npz_root must be provided.")

    hdf5_root = cfg.get(
        "hdf5_root", os.path.join(os.getcwd(), "holomotion_hdf5_v2")
    )
    chunks_t = int(cfg.get("chunks_t", 1024))
    compression = str(cfg.get("compression", "lzf")).lower()
    shard_target_gb = float(cfg.get("shard_target_gb", 2.0))
    shard_target_bytes = int(
        cfg.get("shard_target_bytes", shard_target_gb * (1 << 30))
    )
    shard_target_mode = str(
        cfg.get("shard_target_mode", "uncompressed_nbytes")
    )

    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"NPZ clips directory not found: {root}")

    entries = _discover_motion_entries(roots)
    motion_keys: List[str] = []
    motion_key2id: Dict[str, int] = {}
    nan_npz_paths: List[str] = []

    first_npz = Path(entries[0].path)
    array_specs = _discover_array_specs(first_npz)
    array_names_created = [s.name for s in array_specs]
    dof_tail = next(
        spec.shape_tail for spec in array_specs if spec.name == "ref_dof_pos"
    )
    logger.info(
        "HDF5 v2 datasets: {} (dof_tail={})",
        array_names_created,
        dof_tail,
    )

    dof_names: List[str] = []
    body_names: List[str] = []
    extended_body_names: List[str] = []
    robot_cfg = cfg.get("robot", None)
    if robot_cfg is not None and "motion" in robot_cfg:
        motion_cfg = robot_cfg["motion"]
        dof_names = list(motion_cfg.get("dof_names", []))
        body_names = list(motion_cfg.get("body_names", []))
        extended_body_names = list(
            list(motion_cfg.get("body_names", []))
            + [
                i.get("joint_name")
                for i in motion_cfg.get("extend_config", [])
            ]
        )

    shard_dir = os.path.join(str(hdf5_root), "shards")
    _ensure_dir(shard_dir)

    hdf5_shards: List[Dict[str, Any]] = []
    clips_manifest: Dict[str, Dict[str, Any]] = {}

    curr_shard_idx = 0
    curr_shard_bytes = 0
    writer: Optional[Hdf5ShardWriter] = None
    pbar = tqdm(total=len(entries), desc="Packing HDF5 v2 shards")

    for entry in entries:
        npz_path = Path(entry.path)
        metadata_json, metadata = _load_metadata_json(npz_path)
        if "num_frames" not in metadata:
            raise KeyError(f"'num_frames' missing in metadata: {npz_path}")
        num_frames = int(metadata["num_frames"])
        if num_frames <= 0:
            raise ValueError(f"Invalid num_frames {num_frames} in {npz_path}")

        arrays_np = _load_npz_arrays(
            npz_path=npz_path, num_frames=num_frames, dof_tail=dof_tail
        )
        nan_arrays = _nan_array_names(arrays_np)
        if len(nan_arrays) > 0:
            rel_npz_path = _relative_npz_path(npz_path, roots)
            nan_npz_paths.append(rel_npz_path)
            logger.warning(
                "NaN detected in NPZ (arrays: {}), skipping: {}",
                nan_arrays,
                npz_path,
            )
            pbar.update(1)
            continue

        shard_mode_norm = shard_target_mode.lower().strip()
        if shard_mode_norm in (
            "h5_filesize",
            "h5_size",
            "output_filesize",
            "disk",
        ):
            if writer is None:
                shard_name = f"holomotion_{curr_shard_idx:03d}.h5"
                shard_path = os.path.join(shard_dir, shard_name)
                writer = Hdf5ShardWriter(
                    shard_path,
                    array_specs,
                    chunks_t=chunks_t,
                    compression=compression,
                )
        else:
            motion_bytes = _estimate_bytes_for_motion(
                npz_path, shard_target_mode
            )
            if (
                writer is None
                or (curr_shard_bytes + motion_bytes) > shard_target_bytes
            ):
                if writer is not None:
                    shard_summary = writer.finalize()
                    hdf5_shards.append(
                        {
                            "file": os.path.relpath(
                                shard_summary["file"], str(hdf5_root)
                            ),
                            "num_clips": shard_summary["num_clips"],
                            "num_frames": shard_summary["num_frames"],
                        }
                    )
                    curr_shard_idx += 1
                    curr_shard_bytes = 0

                shard_name = f"holomotion_{curr_shard_idx:03d}.h5"
                shard_path = os.path.join(shard_dir, shard_name)
                writer = Hdf5ShardWriter(
                    shard_path,
                    array_specs,
                    chunks_t=chunks_t,
                    compression=compression,
                )

        motion_id = motion_key2id.get(entry.name)
        if motion_id is None:
            motion_id = len(motion_keys)
            motion_key2id[entry.name] = motion_id
            motion_keys.append(entry.name)
        start, length = writer.append_motion(
            motion_id=motion_id,
            np_arrays=arrays_np,
            metadata_json=metadata_json,
        )

        clips_manifest[entry.name] = {
            "motion_key": entry.name,
            "shard": curr_shard_idx,
            "clip_idx": len(writer._clip_starts) - 1,
            "start": int(start),
            "length": int(length),
            "available_arrays": list(array_names_created),
            "metadata": metadata,
        }
        if shard_mode_norm in (
            "h5_filesize",
            "h5_size",
            "output_filesize",
            "disk",
        ):
            writer.h5.flush()
            curr_shard_bytes = int(os.path.getsize(writer.h5_path))
        else:
            curr_shard_bytes += motion_bytes
        pbar.update(1)

        if (
            shard_mode_norm
            in ("h5_filesize", "h5_size", "output_filesize", "disk")
            and curr_shard_bytes >= shard_target_bytes
            and writer is not None
        ):
            shard_summary = writer.finalize()
            hdf5_shards.append(
                {
                    "file": os.path.relpath(
                        shard_summary["file"], str(hdf5_root)
                    ),
                    "num_clips": shard_summary["num_clips"],
                    "num_frames": shard_summary["num_frames"],
                }
            )
            curr_shard_idx += 1
            curr_shard_bytes = 0
            writer = None

    pbar.close()

    if writer is not None:
        shard_summary = writer.finalize()
        hdf5_shards.append(
            {
                "file": os.path.relpath(shard_summary["file"], str(hdf5_root)),
                "num_clips": shard_summary["num_clips"],
                "num_frames": shard_summary["num_frames"],
            }
        )

    manifest = {
        "version": 1,
        "root": str(hdf5_root),
        "hdf5_shards": hdf5_shards,
        "clips": clips_manifest,
        "motion_keys": motion_keys,
        "dof_names": dof_names,
        "body_names": body_names,
        "extended_body_names": extended_body_names,
        "array_names": array_names_created,
        "chunks_t": int(chunks_t),
        "compression": compression,
        "shard_target_mode": str(shard_target_mode),
        "shard_target_bytes": int(shard_target_bytes),
    }
    _ensure_dir(str(hdf5_root))
    nan_paths_path = os.path.join(str(hdf5_root), "nan_npz_paths.json")
    with open(nan_paths_path, "w") as f:
        json.dump(nan_npz_paths, f, indent=2)
    if len(nan_npz_paths) > 0:
        logger.warning(
            "Skipped {} NPZ files due to NaNs. List: {}",
            len(nan_npz_paths),
            nan_paths_path,
        )
    else:
        logger.info("No NaN detected in NPZ inputs.")
    with open(os.path.join(str(hdf5_root), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "HDF5 v2 packing complete. Shards: {}. Root: {}",
        len(hdf5_shards),
        hdf5_root,
    )


if __name__ == "__main__":
    main()
