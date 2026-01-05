# Project HoloMotion
#
# HDF5 packer: shard per-motion NPZs into few large HDF5 files optimized for JuiceFS.
# - Time-major, resizable datasets with chunks along time only
# - LZF compression for fast decode

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import hydra
import numpy as np
from loguru import logger
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class ArraySpec:
    name: str
    shape_tail: Tuple[int, ...]  # shape excluding time dim
    dtype: np.dtype


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

        # Datasets, resizable on time dimension
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

        # Index datasets (filled on finalize)
        self._clip_starts: List[int] = []
        self._clip_lengths: List[int] = []
        self._clip_motion_ids: List[int] = []
        self._clip_metadata: List[str] = []

        self.t_cursor = 0  # total time frames written

    def append_motion(
        self,
        motion_id: int,
        motion_key: str,
        npz_data: Dict[str, np.ndarray],
        arrays_present: List[str],
        metadata_json: str,
        fill_missing: bool = True,
    ) -> Tuple[int, int]:
        # Determine frames
        t_list: List[int] = []
        for n in arrays_present:
            if n in npz_data:
                t_list.append(int(npz_data[n].shape[0]))
        assert len(t_list) > 0, (
            f"No arrays found to determine length for {motion_key}"
        )
        t_len = int(t_list[0])

        # Resize and write
        start = self.t_cursor
        end = start + t_len
        for spec in self.array_specs:
            ds = self.datasets[spec.name]
            ds.resize((end, *spec.shape_tail))
            if spec.name in npz_data:
                arr = npz_data[spec.name]
                expected_shape = (t_len, *spec.shape_tail)
                try:
                    ds[start:end, ...] = arr
                except (ValueError, TypeError):
                    # Handle shape mismatch
                    if arr.size == np.prod(expected_shape):
                        arr = arr.reshape(expected_shape)
                    elif len(arr.shape) == 1 and len(expected_shape) == 2:
                        # Expand 1D to 2D: (t,) -> (t, 1) then broadcast
                        arr = arr[:, None]
                        arr = np.broadcast_to(arr, expected_shape)
                    else:
                        raise ValueError(
                            f"Cannot reshape {arr.shape} to {expected_shape} "
                            f"for array {spec.name}"
                        )
                    ds[start:end, ...] = arr
            else:
                if fill_missing:
                    ds[start:end, ...] = 0

        # Indices
        self._clip_starts.append(start)
        self._clip_lengths.append(t_len)
        self._clip_motion_ids.append(motion_id)
        self._clip_metadata.append(metadata_json)
        self.t_cursor = end
        return start, t_len

    def finalize(self) -> Dict:
        # Write index datasets under a group
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


def _discover_all_array_keys(npz_path: str) -> List[str]:
    """Discover all array keys in an NPZ file (excluding metadata)."""
    data = np.load(npz_path, allow_pickle=False)
    array_keys: List[str] = []
    for key in data.files:
        if key != "metadata":
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                array_keys.append(key)
    return array_keys


def _discover_specs_from_npz(
    first_npz_path: str, all_keys: List[str]
) -> List[ArraySpec]:
    """Discover array specs from NPZ file for all provided keys."""
    data = np.load(first_npz_path, allow_pickle=False)
    specs: List[ArraySpec] = []

    for key in all_keys:
        if key in data:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                shape_tail = tuple(arr.shape[1:])
                specs.append(
                    ArraySpec(name=key, shape_tail=shape_tail, dtype=arr.dtype)
                )
    return specs


def _estimate_bytes_for_motion(npz_path: str, array_names: List[str]) -> int:
    data = np.load(npz_path, allow_pickle=False)
    total = 0
    for key in array_names:
        if key in data:
            total += int(data[key].nbytes)
    return total


@hydra.main(
    config_path="../../config",
    config_name="motion_retargeting/pack_hdf5_database",
    version_base=None,
)
def main(cfg: OmegaConf) -> None:
    # Input directories (single string or list of strings)
    dirs_cfg = cfg.get("holomotion_retargeted_dirs", None)
    if dirs_cfg is None:
        # Backwards-compat: fall back to legacy key if provided
        legacy_dir = cfg.get(
            "precomputed_npz_root",
            os.path.join(os.getcwd(), "motion_kinematics_npz"),
        )
        holomotion_retargeted_dirs: List[str] = [str(legacy_dir)]
    else:
        if isinstance(dirs_cfg, (list, tuple, ListConfig)):
            holomotion_retargeted_dirs = [str(p) for p in list(dirs_cfg)]
        else:
            holomotion_retargeted_dirs = [str(dirs_cfg)]

    hdf5_root = cfg.get(
        "hdf5_root", os.path.join(os.getcwd(), "holomotion_hdf5")
    )
    chunks_t = int(cfg.get("chunks_t", 1024))
    compression = str(cfg.get("compression", "lzf")).lower()
    shard_target_gb = float(cfg.get("shard_target_gb", 2.0))
    shard_target_bytes = int(
        cfg.get("shard_target_bytes", shard_target_gb * (1 << 30))
    )

    # Validate input directories
    for d in holomotion_retargeted_dirs:
        assert os.path.isdir(d), f"NPZ clips directory not found: {d}"

    # Discover motions
    # Each motion entry: (motion_key, base_key, npz_path) where motion_key is prefixed with parent dir name
    motion_key_to_path: Dict[str, str] = {}
    for d in holomotion_retargeted_dirs:
        # Extract parent directory name (last component of path)
        parent_dir_name = os.path.basename(os.path.normpath(d))
        base_dir = (
            os.path.join(d, "clips")
            if os.path.isdir(os.path.join(d, "clips"))
            else d
        )
        for dirpath, _, filenames in os.walk(base_dir):
            for fname in filenames:
                if fname.endswith(".npz"):
                    base_key = os.path.splitext(fname)[0]
                    motion_key = f"{parent_dir_name}_{base_key}"
                    npz_path = os.path.join(dirpath, fname)
                    assert motion_key not in motion_key_to_path, (
                        f"Duplicate motion key across dirs: {motion_key}"
                    )
                    motion_key_to_path[motion_key] = npz_path
    motion_entries: List[Tuple[str, str, str]] = [
        (k, os.path.splitext(os.path.basename(v))[0], v)
        for k, v in motion_key_to_path.items()
    ]
    motion_entries.sort(key=lambda x: x[0])
    assert len(motion_entries) > 0, "No motion NPZ files found"
    motion_keys: List[str] = [e[0] for e in motion_entries]

    # Discover all array keys from first NPZ file
    first_npz = motion_entries[0][2]
    all_array_keys = _discover_all_array_keys(first_npz)
    logger.info(
        f"Discovered {len(all_array_keys)} array keys in first NPZ: {all_array_keys}"
    )

    # Specs from first file (all arrays present are created)
    specs = _discover_specs_from_npz(first_npz, all_array_keys)
    array_names_created = [s.name for s in specs]
    logger.info(f"Creating datasets for arrays: {array_names_created}")

    # Names from robot config if available
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

    # Shard loop
    shard_dir = os.path.join(hdf5_root, "shards")
    _ensure_dir(shard_dir)

    hdf5_shards: List[Dict] = []
    clips_manifest: Dict[str, Dict] = {}
    motion_key2id = {k: i for i, k in enumerate(motion_keys)}

    curr_shard_idx = 0
    curr_shard_bytes = 0
    writer: Optional[Hdf5ShardWriter] = None
    pbar = tqdm(total=len(motion_keys), desc="Packing HDF5 shards")

    for key, _base_key, npz_path in motion_entries:
        motion_bytes = _estimate_bytes_for_motion(
            npz_path, array_names_created
        )
        if (
            writer is None
            or (curr_shard_bytes + motion_bytes) > shard_target_bytes
        ):
            # finalize previous shard
            if writer is not None:
                shard_summary = writer.finalize()
                hdf5_shards.append(
                    {
                        "file": os.path.relpath(
                            shard_summary["file"], hdf5_root
                        ),
                        "num_clips": shard_summary["num_clips"],
                        "num_frames": shard_summary["num_frames"],
                    }
                )
                curr_shard_idx += 1
                curr_shard_bytes = 0

            # open new shard
            shard_name = f"holomotion_{curr_shard_idx:03d}.h5"
            shard_path = os.path.join(shard_dir, shard_name)
            writer = Hdf5ShardWriter(
                shard_path, specs, chunks_t=chunks_t, compression=compression
            )

        # append motion
        data = np.load(npz_path, allow_pickle=False)
        assert "metadata" in data, f"metadata missing in {npz_path}"
        metadata_json = str(data["metadata"])
        # Use keys directly from NPZ
        arrays_present = []
        resolved_arrays: Dict[str, np.ndarray] = {}
        for array_key in array_names_created:
            if array_key in data:
                arrays_present.append(array_key)
                resolved_arrays[array_key] = data[array_key]
        start, length = writer.append_motion(
            motion_id=motion_key2id[key],
            motion_key=key,
            npz_data=resolved_arrays,
            arrays_present=arrays_present,
            metadata_json=metadata_json,
        )

        clips_manifest[key] = {
            "motion_key": key,
            "shard": curr_shard_idx,
            "clip_idx": len(writer._clip_starts) - 1,
            "start": int(start),
            "length": int(length),
            "available_arrays": arrays_present,
            "metadata": json.loads(metadata_json),
        }
        curr_shard_bytes += motion_bytes
        pbar.update(1)

    pbar.close()

    # finalize last shard
    if writer is not None:
        shard_summary = writer.finalize()
        hdf5_shards.append(
            {
                "file": os.path.relpath(shard_summary["file"], hdf5_root),
                "num_clips": shard_summary["num_clips"],
                "num_frames": shard_summary["num_frames"],
            }
        )

    manifest = {
        "version": 1,
        "root": hdf5_root,
        "hdf5_shards": hdf5_shards,
        "clips": clips_manifest,
        "motion_keys": motion_keys,
        "dof_names": dof_names,
        "body_names": body_names,
        "extended_body_names": extended_body_names,
        "array_names": array_names_created,
        "chunks_t": chunks_t,
        "compression": compression,
    }
    _ensure_dir(hdf5_root)
    with open(os.path.join(hdf5_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"HDF5 packing complete. Shards: {len(hdf5_shards)}. Root: {hdf5_root}"
    )


if __name__ == "__main__":
    main()
