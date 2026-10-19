"""Produce compact HoloMotion V2 H5 and current boxing training records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import inside, new_directory, sha256, write_json


def source_records(root):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(
            "Each source must be a raw-motion or retargeted-motion directory"
        )
    paired = root / "motions.json"
    receipt = root / "download.json"
    if paired.is_file():
        metadata = json.loads(paired.read_text())
        if metadata.get("format") != "holoboxing_paired_motions_v1":
            raise ValueError("Unsupported paired-motion manifest")
        records = metadata["records"]
        dataset = metadata["dataset"]
    elif receipt.is_file():
        metadata = json.loads(receipt.read_text())
        if metadata.get("dataset") != "robostriker":
            raise ValueError(
                "Raw InterHuman must be retargeted first"
            )
        records = metadata["files"]
        dataset = "robostriker"
    else:
        paths = sorted(root.rglob("*.npz"))
        records = [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            for path in paths
        ]
        dataset = "robostriker"
    result = []
    for record in records:
        if not record["path"].endswith(".npz"):
            continue
        path = inside(root, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Source changed: {record['path']}")
        result.append(
            {
                **record,
                "file": path,
                "dataset": dataset,
                "group": path.stem.removesuffix("_mirror"),
            }
        )
    if not result:
        raise ValueError(f"No source motion NPZ files in {root.name}")
    return result


def assign_splits(records, *, seed=42, validation_fraction=0.1):
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Validation fraction must lie strictly between zero and one"
        )
    known, unlabelled = {}, set()
    for record in records:
        group = record["group"]
        split = record.get("source_split", "unknown")
        if split in ("train", "validation", "test"):
            if group in known and known[group] != split:
                raise ValueError(f"Conflicting official splits for {group}")
            known[group] = split
        else:
            unlabelled.add(group)
    unlabelled -= known.keys()
    ordered = sorted(
        unlabelled,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group}".encode()
        ).hexdigest(),
    )
    if ordered:
        count = max(1, math.ceil(len(ordered) * validation_fraction))
        if len(ordered) > 1:
            count = min(count, len(ordered) - 1)
        elif "train" not in known.values():
            count = 0
        known.update(
            {
                group: "validation" if i < count else "train"
                for i, group in enumerate(ordered)
            }
        )
    used = [record for record in records if known[record["group"]] != "test"]
    if not used:
        raise ValueError("No non-test source motions supplied")
    return known, used


def export_robot_h5(records, output, *, assets, splits, _partition=None):
    from holomotion.src.training.data_production.robot_h5 import (
        RobotH5ShardWriter,
    )
    from .records import (
        load_scene_model,
        load_trajectory,
        scene_layout,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if _partition is None:
        shards, all_clips, all_keys = [], {}, []
        for partition in ("train", "validation"):
            subset = [
                record
                for record in records
                if splits[record["group"]] == partition
            ]
            if not subset:
                continue
            manifest = export_robot_h5(
                subset,
                output / partition,
                assets=assets,
                splits=splits,
                _partition=partition,
            )
            shard_offset = len(shards)
            shards.extend(
                {**shard, "file": f"{partition}/{shard['file']}"}
                for shard in manifest["hdf5_shards"]
            )
            all_clips.update(
                {
                    key: {**clip, "shard": clip["shard"] + shard_offset}
                    for key, clip in manifest["clips"].items()
                }
            )
            all_keys.extend(manifest["motion_keys"])
        aggregate = {
            **manifest,
            "root": str(output),
            "hdf5_shards": shards,
            "clips": all_clips,
            "motion_keys": all_keys,
            "split": "aggregate_for_verification",
            **{
                ("training" if partition == "train" else partition) + "_manifest":
                f"{partition}/manifest.json"
                for partition in ("train", "validation")
                if (output / partition / "manifest.json").is_file()
            },
        }
        write_json(output / "manifest.json", aggregate)
        write_json(output / "nan_npz_paths.json", [])
        return aggregate
    model = load_scene_model(assets)
    names, _, _, _ = scene_layout(model)
    writer = RobotH5ShardWriter(output / "shards" / "boxing_000000.h5")
    clips, motion_keys = {}, []
    try:
        for record in records:
            trajectory = load_trajectory(record["file"], names)
            frames = len(trajectory["joint_position"])
            for role in range(trajectory["roles"]):
                key = f"{record['dataset']}/{record['file'].stem}/person{role + 1}"
                metadata = {
                    "motion_key": key,
                    "raw_motion_key": record["path"],
                    "source_id": record["dataset"],
                    "source_group": record["group"],
                    "source_sha256": record["sha256"],
                    "split": splits[record["group"]],
                    "motion_fps": 50.0,
                    "num_frames": frames,
                    "num_dofs": 29,
                    "root_quaternion_order": "xyzw",
                    "world_frame": "right-handed z-up meters",
                    "conversion": "named root/DOF from robot source; derivatives derived by HoloMotion",
                    "roles_in_source": trajectory["roles"],
                }
                arrays = {
                    "ref_root_pos": trajectory["root_position"][
                        :, role
                    ].astype(np.float32),
                    "ref_root_rot": trajectory["root_quaternion"][
                        :, role, [1, 2, 3, 0]
                    ].astype(np.float32),
                    "ref_dof_pos": trajectory["joint_position"][
                        :, role
                    ].astype(np.float32),
                }
                start, length = writer.append_motion(
                    motion_id=len(motion_keys),
                    arrays=arrays,
                    metadata_json=json.dumps(metadata, sort_keys=True),
                )
                clips[key] = {
                    "motion_key": key,
                    "shard": 0,
                    "clip_idx": writer.clip_count - 1,
                    "start": start,
                    "length": length,
                    "metadata": metadata,
                }
                motion_keys.append(key)
        summary = writer.finalize()
    except BaseException:
        writer.handle.close()
        raise
    shard = {
        "file": "shards/boxing_000000.h5",
        "num_clips": summary["num_clips"],
        "num_frames": summary["num_frames"],
        "sha256": sha256(writer.path),
    }
    manifest = {
        "version": 2,
        "root": str(output),
        "hdf5_shards": [shard],
        "clips": clips,
        "motion_keys": motion_keys,
        "dof_names": list(names),
        "array_names": ["ref_root_pos", "ref_root_rot", "ref_dof_pos"],
        "chunks_t": 1024,
        "compression": "lzf",
        "control_hz": 50,
        "asset_identity": assets.identity(),
        "producer": "holoboxing.data",
        "split": _partition,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "nan_npz_paths.json", [])
    return manifest


def verify_dataset(output, *, assets):
    import h5py
    import torch
    from holomotion.src.training.h5_dataloader import Hdf5RootDofDataset
    from .contract import CompetitionAssets
    from .dataset import PreparedDataset
    from .records import SOURCE_SCHEMA

    output = Path(output).expanduser().resolve(strict=True)
    tree = CompetitionAssets.load(assets)
    dataset = PreparedDataset(
        output / "prepared" / "manifest.json", asset_identity=tree.identity()
    )
    maximum_rotation_error = 0.0
    supervised_samples = 0
    amp_joint_names = json.loads(SOURCE_SCHEMA.read_text())["joint_names"]
    for record in dataset.records:
        with np.load(
            dataset.root / record["path"], allow_pickle=False
        ) as packet:
            if (
                packet["amp_joint_names"].tolist() != amp_joint_names
                or packet["amp_fps"].shape != ()
                or packet["amp_fps"].item() != 50
                or str(packet["amp_angular_velocity_frame"]) != "world"
            ):
                raise ValueError("AMP requires training joint order and 50 Hz")
            reference = packet["amp_reference"]
            if (
                reference.shape
                != ((record["frames"] - 1) * record["roles"], 122)
                or reference.dtype != np.float32
                or not np.isfinite(reference).all()
            ):
                raise ValueError("Invalid complete AMP reference transitions")
            quaternion = packet["amp_root_quaternion_wxyz"]
            if (
                quaternion.shape != (len(reference), 2, 4)
                or quaternion.dtype != np.float32
                or not np.isfinite(quaternion).all()
                or not np.allclose(
                    np.linalg.norm(quaternion, axis=-1), 1, atol=1e-3, rtol=0
                )
            ):
                raise ValueError("Invalid AMP root quaternions")
            target = packet["target_pose"].reshape(-1, 5, 5, 9)
            x, z = target[..., 3:6], target[..., 6:9]
            error = max(
                float(np.abs(np.linalg.norm(x, axis=-1) - 1).max()),
                float(np.abs(np.linalg.norm(z, axis=-1) - 1).max()),
                float(np.abs((x * z).sum(-1)).max()),
            )
            maximum_rotation_error = max(maximum_rotation_error, error)
            supervised_samples += len(target)
    if maximum_rotation_error > 2e-4:
        raise ValueError(
            "Prepared future-pose X/Z columns are not orthonormal"
        )
    robot_root = output / "robot_h5"
    robot = json.loads((robot_root / "manifest.json").read_text())
    if (
        robot.get("version") != 2
        or robot.get("asset_identity") != tree.identity()
    ):
        raise ValueError("Compact robot H5 contract or assets differ")
    for clip in robot["clips"].values():
        if clip.get("metadata", {}).get("motion_fps") != 50:
            raise ValueError(
                "V2 manifest must expose each clip's 50 Hz motion_fps to the training reader"
            )
    reader_roots = [robot_root]
    split_hashes = {}
    if robot.get("split") == "aggregate_for_verification":
        reader_roots = []
        for partition in ("train", "validation"):
            expected = {
                key: clip
                for key, clip in robot["clips"].items()
                if clip["metadata"]["split"] == partition
            }
            if not expected:
                continue
            split_root = robot_root / partition
            split_manifest = json.loads(
                (split_root / "manifest.json").read_text()
            )
            if (
                split_manifest.get("split") != partition
                or split_manifest.get("asset_identity") != tree.identity()
                or split_manifest.get("dof_names") != robot["dof_names"]
                or set(split_manifest["clips"]) != set(expected)
            ):
                raise ValueError(
                    f"The {partition} H5 manifest differs from the aggregate inventory"
                )
            for key, clip in split_manifest["clips"].items():
                aggregate_clip = expected[key]
                local_shard = split_manifest["hdf5_shards"][clip["shard"]]
                aggregate_shard = robot["hdf5_shards"][aggregate_clip["shard"]]
                if (
                    inside(split_root, local_shard["file"])
                    != inside(robot_root, aggregate_shard["file"])
                    or local_shard["sha256"] != aggregate_shard["sha256"]
                    or {
                        field: value
                        for field, value in clip.items()
                        if field != "shard"
                    }
                    != {
                        field: value
                        for field, value in aggregate_clip.items()
                        if field != "shard"
                    }
                ):
                    raise ValueError(
                        f"The {partition} H5 clip inventory changed: {key}"
                    )
            split_hashes[partition + "_h5_manifest_sha256"] = sha256(
                split_root / "manifest.json"
            )
            reader_roots.append(split_root)
    frames, clips = 0, 0
    for shard in robot["hdf5_shards"]:
        path = inside(robot_root, shard["file"])
        if sha256(path) != shard["sha256"]:
            raise ValueError("Compact robot H5 bytes changed")
        with h5py.File(path, "r") as file:
            count = len(file["ref_dof_pos"])
            for name, width in (
                ("ref_root_pos", 3),
                ("ref_root_rot", 4),
                ("ref_dof_pos", 29),
            ):
                values = file[name][:]
                if (
                    values.shape != (count, width)
                    or values.dtype != np.float32
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"Invalid robot H5 array: {name}")
            if not np.allclose(
                np.linalg.norm(file["ref_root_rot"][:], axis=-1),
                1,
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError(
                    "Robot H5 requires unit XYZW root quaternions"
                )
            metadata = [
                json.loads(value)
                for value in file["clips/metadata_json"].asstr()[:]
            ]
            if any(
                not isinstance(value.get("source_id"), str)
                or not value["source_id"].strip()
                for value in metadata
            ):
                raise ValueError("Each robot clip needs its dataset source_id")
            lengths, starts = file["clips/length"][:], file["clips/start"][:]
            if (
                len(metadata) != len(lengths)
                or not np.array_equal(
                    starts, np.concatenate(([0], np.cumsum(lengths)[:-1]))
                )
                or lengths.sum() != count
            ):
                raise ValueError(
                    "Robot H5 clip boundaries do not partition the frames"
                )
            if any(value["split"] == "test" for value in metadata):
                raise ValueError(
                    "Official test clips must not enter training output"
                )
            clips += len(lengths)
            frames += count
    from .robot import robot_file

    with robot_file(tree.resources.scene) as xml:
        for reader_root in reader_roots:
            reader = Hdf5RootDofDataset(
                str(reader_root / "manifest.json"),
                max_frame_length=64,
                fk_robot_file_path=str(xml),
                expected_dof_names=robot["dof_names"],
            )
            try:
                for index in {0, len(reader) - 1}:
                    sample = reader[index]
                    if not all(
                        torch.isfinite(value).all()
                        for value in sample.tensors.values()
                    ):
                        raise ValueError(
                            "HoloMotion V2 reader produced nonfinite tensors"
                        )
            finally:
                reader.close()
    return {
        "status": "PASS",
        "supervised_samples": supervised_samples,
        "robot_h5_clips": clips,
        "robot_h5_frames": frames,
        "maximum_rotation_basis_error": maximum_rotation_error,
        "prepared_manifest_sha256": sha256(dataset.manifest_path),
        "robot_manifest_sha256": sha256(robot_root / "manifest.json"),
        **split_hashes,
        "validation_scope": "file integrity, current boxing reader, actual HoloMotion V2 reader/FK and geometric basis; no tracking rollout",
    }


def prepare_dataset(
    sources, output, *, assets, workers=1, seed=42, validation_fraction=0.1
):
    from .contract import CompetitionAssets
    from .records import main as prepare_main

    if workers < 1:
        raise ValueError("Worker count must be positive")
    records = [record for root in sources for record in source_records(root)]
    if len({record["file"].name for record in records}) != len(records):
        raise ValueError(
            "Source clip names must be unique across supplied roots"
        )
    splits, used = assign_splits(
        records, seed=seed, validation_fraction=validation_fraction
    )
    tree = CompetitionAssets.load(assets)
    for root in sources:
        paired = Path(root).expanduser().resolve() / "motions.json"
        if (
            paired.is_file()
            and json.loads(paired.read_text()).get("asset_identity")
            != tree.identity()
        ):
            raise ValueError(
                "Retargeted source and requested competition assets differ"
            )
    output_path = Path(output).expanduser().resolve()
    for root in sources:
        root = Path(root).expanduser().resolve(strict=True)
        if output_path.is_relative_to(root) or root.is_relative_to(
            output_path
        ):
            raise ValueError(
                "Keep processed output outside every source directory"
            )
    output = new_directory(output_path)
    try:
        provenance = {
            "format": "holoboxing_data_sources_v1",
            "split_seed": seed,
            "validation_fraction": validation_fraction,
            "split_policy": "official split when available; exclude test; otherwise deterministic original-plus-mirror groups",
            "excluded_official_test": [
                record["path"]
                for record in records
                if splits[record["group"]] == "test"
            ],
            "records": [
                {key: value for key, value in record.items() if key != "file"}
                | {"split": splits[record["group"]]}
                for record in used
            ],
        }
        write_json(output / "sources.json", provenance)
        export_robot_h5(used, output / "robot_h5", assets=tree, splits=splits)
        arguments = [
            "--assets",
            str(assets),
            "--output",
            str(output / "prepared"),
            "--workers",
            str(workers),
        ]
        for record in used:
            arguments += ["--motion", str(record["file"])]
        prepare_main(arguments)
        report = verify_dataset(output, assets=assets)
        report.update(
            {
                "source_clips": len(used),
                "sources_sha256": sha256(output / "sources.json"),
                "conversion": "competition XML kinematics and current boxing prepare implementation",
            }
        )
        write_json(output / "COMPLETE.json", report)
        return report
    except BaseException as error:
        write_json(
            output / "FAILED.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
