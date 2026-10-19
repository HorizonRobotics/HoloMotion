"""Retarget paired InterHuman SMPL while preserving a shared world frame."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .common import (
    clip_timing,
    inside,
    new_directory,
    read_interhuman,
    sha256,
    write_json,
)


DEFAULT_RETARGET_ITERATIONS = 4


def retarget_config(iterations=DEFAULT_RETARGET_ITERATIONS):
    from holoretarget.config import DEFAULT_CONFIG

    if type(iterations) is not int or iterations < 1:
        raise ValueError("HoloRetarget iterations must be a positive integer")
    return replace(DEFAULT_CONFIG, newton_iterations=iterations)


def shared_pair_positions(pelvis_positions, human_heights, robot_height):
    """Scale ALL horizontal displacement once, about one fixed pair anchor."""
    pelvis_positions = np.asarray(pelvis_positions, dtype=np.float64)
    heights = np.asarray(human_heights, dtype=np.float64)
    if (
        pelvis_positions.ndim != 3
        or pelvis_positions.shape[1:] != (2, 3)
        or not len(pelvis_positions)
        or heights.shape != (2,)
        or not np.isfinite(pelvis_positions).all()
        or not np.isfinite(heights).all()
        or (heights <= 0).any()
        or not np.isfinite(robot_height)
        or robot_height <= 0
    ):
        raise ValueError(
            "Need finite paired pelvis positions and positive neutral-pose heights"
        )
    scale = float(robot_height / heights.mean())
    anchor = pelvis_positions[0, :, :2].mean(axis=0)
    result = scale * (pelvis_positions[..., :2] - anchor)
    return result, {
        "scale": scale,
        "anchor_xy_m": anchor.tolist(),
        "human_heights_m": heights.tolist(),
        "robot_height_m": float(robot_height),
    }


class SmplShapeModel:
    """Use licensed SMPL shape parameters for skeleton offsets AND body height."""

    def __init__(self, model_path):
        from holoretarget._engine_impl import _enable_chumpy_compat

        _enable_chumpy_compat()
        self.path = Path(model_path).expanduser().resolve(strict=True)
        # Unlike downloaded motion PKLs, this is the user's trusted licensed
        # SMPL model. Its Chumpy/SciPy types are required by the official file.
        with self.path.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
        regressor = data["J_regressor"]
        self.regressor = np.asarray(
            regressor.toarray() if hasattr(regressor, "toarray") else regressor
        )
        self.template = np.asarray(data["v_template"], dtype=np.float64)
        self.shape_directions = np.asarray(data["shapedirs"], dtype=np.float64)
        self.parents = np.asarray(
            data["kintree_table"][0, :24], dtype=np.int32
        )
        self.parents[0] = -1

    def skeleton(self, betas):
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)
        if (
            not len(betas)
            or len(betas) > self.shape_directions.shape[-1]
            or not np.isfinite(betas).all()
        ):
            raise ValueError(
                "SMPL shape coefficients do not match the supplied body model"
            )
        vertices = self.template + np.einsum(
            "vci,i->vc", self.shape_directions[..., : len(betas)], betas
        )
        joints = (self.regressor @ vertices)[:24]
        offsets = np.zeros_like(joints, dtype=np.float32)
        offsets[1:] = joints[1:] - joints[self.parents[1:]]
        # SMPL's neutral mesh is Y-up; this measurement is independent of
        # the current crouch or root orientation in the recorded sequence.
        height = float(np.ptp(vertices[:, 1]))
        return (
            SimpleNamespace(
                smpl_parents=self.parents, smpl_rest_parent_offsets=offsets
            ),
            joints[0],
            height,
        )


def read_pair(path):
    packet = read_interhuman(path)
    frames, fps = clip_timing(packet)
    people = []
    for role in ("person1", "person2"):
        person = packet[role]
        values = {}
        for key, width in (
            ("trans", 3),
            ("root_orient", 3),
            ("pose_body", 63),
        ):
            array = np.asarray(person[key], dtype=np.float64)
            if array.shape != (frames, width) or not np.isfinite(array).all():
                raise ValueError(f"{Path(path).name}: invalid {role}.{key}")
            values[key] = array
        betas = np.asarray(person["betas"], dtype=np.float64)
        if betas.ndim == 2:
            if betas.shape[0] != frames or not np.allclose(
                betas, betas[0], atol=1e-6, rtol=0
            ):
                raise ValueError(
                    "A clip must have constant SMPL shape for fixed pair scaling"
                )
            betas = betas[0]
        if betas.ndim != 1 or not np.isfinite(betas).all():
            raise ValueError("Invalid SMPL shape coefficients")
        gender = str(person.get("gender", "neutral")).lower()
        if gender not in ("neutral", "none"):
            raise ValueError(
                "This InterHuman adapter requires neutral SMPL, matching the supplied neutral model"
            )
        people.append({**values, "betas": betas})
    return people, frames, fps


def resample_person(person, frames, fps, *, frame_limit=None):
    source_time = np.arange(frames, dtype=np.float64) / fps
    times = np.arange(int(np.floor(source_time[-1] * 50 + 1e-8)) + 1) / 50
    if frame_limit is not None:
        if frame_limit < 6:
            raise ValueError(
                "A bounded conversion still needs at least six frames"
            )
        times = times[:frame_limit]
    if len(times) < 6:
        raise ValueError(
            "A source clip needs six 50 Hz frames without future padding"
        )
    sample_times = np.clip(times, source_time[0], source_time[-1])
    translation = np.stack(
        [
            np.interp(sample_times, source_time, person["trans"][:, axis])
            for axis in range(3)
        ],
        axis=-1,
    )
    local_pose = np.concatenate(
        (person["root_orient"], person["pose_body"]), axis=-1
    ).reshape(frames, 22, 3)
    rotations = np.stack(
        [
            Slerp(source_time, Rotation.from_rotvec(local_pose[:, joint]))(
                sample_times
            ).as_rotvec()
            for joint in range(22)
        ],
        axis=1,
    )
    return {
        "trans": translation,
        "pose": rotations.reshape(len(times), 66),
        "betas": person["betas"],
    }, times


def robot_neutral_height(assets):
    """Measure the selected competition robot's neutral visual-mesh extent."""
    import mujoco
    from .robot import load_scene_model

    model = load_scene_model(assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = []
    for geom in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[geom])).name or ""
        if not name.startswith("p1_") or model.geom_group[geom] != 1:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        start, count = (
            int(model.mesh_vertadr[mesh]),
            int(model.mesh_vertnum[mesh]),
        )
        vertices = model.mesh_vert[start : start + count]
        z = (vertices @ data.geom_xmat[geom].reshape(3, 3).T)[
            :, 2
        ] + data.geom_xpos[geom, 2]
        bounds.extend((float(z.min()), float(z.max())))
    if not bounds or max(bounds) <= min(bounds):
        raise ValueError(
            "Cannot determine robot height from its neutral visual meshes"
        )
    return max(bounds) - min(bounds)


def retarget_pair(
    people,
    frames,
    fps,
    shape_model,
    retargeters,
    robot_height,
    *,
    frame_limit=None,
):
    from holosmpl.converters.smpl_to_body_poses import SmplToBodyPosesConverter

    resampled, skeletons, pelvis, heights = [], [], [], []
    for person in people:
        sequence, times = resample_person(
            person, frames, fps, frame_limit=frame_limit
        )
        skeleton, root_joint, height = shape_model.skeleton(person["betas"])
        sequence["trans"] = sequence["trans"] + root_joint
        resampled.append(sequence)
        skeletons.append(skeleton)
        pelvis.append(sequence["trans"])
        heights.append(height)
    human_pelvis = np.stack(pelvis, axis=1)
    pair_xy, scaling = shared_pair_positions(
        human_pelvis, heights, robot_height
    )
    trajectories = []
    for role, retargeter in enumerate(retargeters):
        retargeter.reset_sequence()
        # The shared converter accepts a shape-specific adapter; keep this
        # assignment local to these data-production retargeter instances.
        retargeter._smpl_adapter = skeletons[role]
        converter = SmplToBodyPosesConverter(retargeter)
        result = np.empty((len(times), 36), dtype=np.float32)
        for frame, (translation, pose) in enumerate(
            zip(resampled[role]["trans"], resampled[role]["pose"])
        ):
            local_translation = translation.copy()
            local_translation[:2] = 0
            body = converter(
                transl=local_translation,
                global_orient_aa=pose[:3],
                body_pose_aa=pose[3:],
            )
            qpos = retargeter.retarget_qpos_from_body_poses(body)
            # HoloRetarget already scales its own root targets. Reconstruct
            # world XY from unscaled human pelvis data exactly once, rather
            # than multiplying the scaled robot displacement a second time.
            qpos[:2] = pair_xy[frame, role]
            result[frame] = qpos
        trajectories.append(result)
    qpos = np.stack(trajectories, axis=1)
    if not np.isfinite(qpos).all() or not np.allclose(
        np.linalg.norm(qpos[..., 3:7], axis=-1), 1, atol=1e-4, rtol=0
    ):
        raise ValueError(
            "HoloRetarget returned nonfinite or invalid robot states"
        )
    expected_delta = scaling["scale"] * (
        human_pelvis[:, 1, :2] - human_pelvis[:, 0, :2]
    )
    error = float(
        np.abs(qpos[:, 1, :2] - qpos[:, 0, :2] - expected_delta).max()
    )
    scaling["maximum_pair_xy_error_m"] = error
    scaling["human_height_definition"] = (
        "shape-specific neutral SMPL mesh Y extent"
    )
    scaling["robot_height_definition"] = (
        "neutral competition robot visual-mesh Z extent"
    )
    scaling["scope"] = (
        "shared fixed XY scale and first-frame pair anchor; retargeted Z retained"
    )
    return qpos, times, scaling


def downloaded_records(source):
    """Read the exact sample download receipt and preserve official splits."""
    source = Path(source)
    receipt = json.loads((source / "download.json").read_text())
    if (
        receipt.get("format") != "holoboxing_download_v1"
        or receipt.get("dataset") != "interhuman"
        or receipt.get("selection_mode") != "clip_ids"
    ):
        raise ValueError("Download the InterHuman sample clip IDs first")
    split_names = {
        "split/train.txt": "train",
        "split/val.txt": "validation",
        "split/test.txt": "test",
    }
    assignments = {}
    for record in receipt["files"]:
        split = split_names.get(record["path"])
        if split is None:
            continue
        path = inside(source, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Downloaded split changed: {record['path']}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            key = Path(line.strip()).stem
            if key in assignments and assignments[key] != split:
                raise ValueError(f"Official split lists overlap: {key}")
            assignments[key] = split
    records = []
    for record in receipt["files"]:
        path = Path(record["path"])
        if path.parent.as_posix() != "motions" or path.suffix != ".pkl":
            continue
        records.append(
            {
                **record,
                "source_group": "interhuman/" + path.stem,
                "source_split": assignments.get(path.stem, "unknown"),
            }
        )
    return records


def retarget_dataset(
    source,
    output,
    *,
    assets,
    smpl_model,
    iterations=DEFAULT_RETARGET_ITERATIONS,
    device="cuda:0",
    robot_height=None,
    limit_clips=None,
    frame_limit=None,
):
    import torch
    import warp as wp
    from holoretarget.online import HoloRetargeter
    from .contract import CompetitionAssets

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("HoloRetarget requires an available CUDA device")
    source = Path(source).expanduser().resolve(strict=True)
    records = downloaded_records(source)
    if limit_clips is not None:
        if limit_clips < 1:
            raise ValueError("Clip limit must be positive")
        records = records[:limit_clips]
    if not records:
        raise ValueError("The InterHuman download contains no sample clips")
    asset_tree = CompetitionAssets.load(assets)
    height = (
        robot_neutral_height(asset_tree)
        if robot_height is None
        else robot_height
    )
    shape_model = SmplShapeModel(smpl_model)
    config = retarget_config(iterations)
    output = new_directory(output, source=source)
    converted = []
    try:
        from .robot import retarget_assets

        config = replace(
            config,
            asset_root=str(retarget_assets(asset_tree, config, output / ".robot")),
        )
        with wp.ScopedDevice(device):
            retargeters = [HoloRetargeter(config), HoloRetargeter(config)]
            for record in records:
                path = inside(source, record["path"])
                if sha256(path) != record["sha256"]:
                    raise ValueError(
                        f"Downloaded source changed: {record['path']}"
                    )
                people, frames, fps = read_pair(path)
                qpos, times, scaling = retarget_pair(
                    people,
                    frames,
                    fps,
                    shape_model,
                    retargeters,
                    height,
                    frame_limit=frame_limit,
                )
                filename = "interhuman-" + path.stem + ".npz"
                np.savez_compressed(
                    output / filename,
                    schema_version=np.asarray("boxing_g1_pair_v1"),
                    dof_names=np.asarray(retargeters[0].dof_names),
                    q=qpos[..., 7:],
                    root_pos=qpos[..., :3],
                    root_quat_wxyz=qpos[..., 3:7],
                    timestamps=times,
                )
                converted.append(
                    {
                        "path": filename,
                        "sha256": sha256(output / filename),
                        "bytes": (output / filename).stat().st_size,
                        "source": path.name,
                        "source_sha256": record["sha256"],
                        "source_group": record["source_group"],
                        "source_split": record["source_split"],
                        "source_frames": frames,
                        "source_fps": fps,
                        "frames": len(times),
                        "roles": 2,
                        "pair_scaling": scaling,
                    }
                )
                print(
                    f"InterHuman retargeted: {len(converted)}/{len(records)}",
                    flush=True,
                )
        report = {
            "format": "holoboxing_paired_motions_v1",
            "dataset": "interhuman",
            "source_download_sha256": sha256(source / "download.json"),
            "smpl_model_sha256": sha256(shape_model.path),
            "asset_identity": asset_tree.identity(),
            "retarget_iterations": iterations,
            "device": device,
            "control_hz": 50,
            "quaternion_order": "wxyz",
            "world_frame": "right-handed z-up meters",
            "bounded_conversion": limit_clips is not None
            or frame_limit is not None,
            "records": converted,
        }
        write_json(output / "motions.json", report)
        return report
    except BaseException as error:
        write_json(
            output / "FAILED.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
