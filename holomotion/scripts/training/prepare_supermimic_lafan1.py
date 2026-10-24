"""Convert the reference G1 LAFAN1 CSVs into the existing HoloMotion HDF5 schema.

Source columns are root XYZ in meters, XYZW quaternion, and 29 SDK-order joints.
No retargeting, clip subdivision, filtering, or coordinate normalization occurs.
The normal HoloMotion reader owns its usual world-frame normalization afterward.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from holomotion.src.env.reference_tracking.robot import G1_JOINT_NAMES


def resample_csv(raw, source_fps=30.0, target_fps=50.0):
    raw = np.asarray(raw, dtype=np.float64)
    if (
        raw.ndim != 2
        or raw.shape[1] != 36
        or len(raw) < 2
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            "Expected at least two finite XYZ/XYZW/29-joint CSV rows"
        )
    if not 0 < source_fps < float("inf") or not 0 < target_fps < float("inf"):
        raise ValueError(
            "Source and target frame rates must be finite and positive"
        )
    if (np.linalg.norm(raw[:, 3:7], axis=-1) < 1e-6).any():
        raise ValueError("Source quaternion has zero norm")
    source_time = np.arange(len(raw), dtype=np.float64) / source_fps
    count = int(np.floor((len(raw) - 1) / source_fps * target_fps)) + 1
    target_time = np.arange(count, dtype=np.float64) / target_fps
    result = np.empty((count, 36), dtype=np.float64)
    for col in (*range(3), *range(7, 36)):
        result[:, col] = np.interp(target_time, source_time, raw[:, col])
    result[:, 3:7] = Slerp(source_time, Rotation.from_quat(raw[:, 3:7]))(
        target_time
    ).as_quat()
    return result.astype(np.float32)


def prepare(source, output, *, source_fps=30.0, expected_clips=40):
    files = sorted(Path(source).glob("*.csv"))
    if len(files) != expected_clips:
        raise ValueError(
            f"Expected {expected_clips} complete CSV clips, found {len(files)}"
        )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    clips, records, arrays = {}, [], []
    start = 0
    for path in files:
        raw = np.loadtxt(path, delimiter=",")
        converted = resample_csv(raw, source_fps)
        arrays.append(converted)
        clips[path.stem] = dict(
            shard=0,
            start=start,
            length=len(converted),
            motion_fps=50.0,
            metadata=dict(motion_fps=50.0, source_name=path.name),
        )
        records.append(
            dict(
                name=path.name,
                source_frames=len(raw),
                target_frames=len(converted),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        start += len(converted)
    values = np.concatenate(arrays)
    shard = output / "motions.h5"
    with h5py.File(shard, "w") as stream:
        for name, data in (
            ("ref_root_pos", values[:, :3]),
            ("ref_root_rot", values[:, 3:7]),
            ("ref_dof_pos", values[:, 7:]),
        ):
            stream.create_dataset(
                name, data=data, compression="gzip", shuffle=True
            )
    manifest = dict(
        version=2,
        schema_version="root_dof_full_candidate_v1",
        dataset_id="supermimic_lafan1_reference_40",
        motion_fps=50.0,
        root_quaternion_order="xyzw",
        dof_names=G1_JOINT_NAMES,
        hdf5_shards=[dict(file=shard.name)],
        clips=clips,
        provenance=dict(
            scope="Training-set reproduction benchmark; not unseen-motion evaluation",
            source_fps=source_fps,
            resampling="Linear XYZ/joints and shortest-arc quaternion SLERP",
            files=records,
        ),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    result = dict(
        clips=len(clips),
        frames=len(values),
        duration_hours=sum((len(x) - 1) / 50 for x in arrays) / 3600,
        shard_sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    (output / "conversion_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--expected-clips", type=int, default=40)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                source_fps=args.source_fps,
                expected_clips=args.expected_clips,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
