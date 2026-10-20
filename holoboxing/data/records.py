"""Prepare actual-state supervision and full reference motions for AMP."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from holoboxing.evaluation.runtime.model_contract import (
    INPUT_LINK_NAMES,
    sha256,
)
from .contract import CompetitionAssets
from .robot import load_scene_model

DATA_FORMAT = "holoboxing_competition_records_v2"
SOURCE_SCHEMA = (
    Path(__file__).resolve().parents[1] / "training" / "assets" / "robot_schema.json"
)


def backward_velocity(values, dt=0.02):
    result = np.zeros_like(values)
    result[1:] = (values[1:] - values[:-1]) / dt
    return result


def load_trajectory(path, scene_joint_names):
    """Read named paired trajectories or the explicit legacy G1 motion schema."""
    with np.load(path, allow_pickle=False) as packet:
        if (
            "schema_version" in packet
            and str(packet["schema_version"]) == "boxing_g1_pair_v1"
        ):
            names = tuple(str(v) for v in packet["dof_names"])
            joints = np.asarray(packet["q"], np.float64)
            position = np.asarray(packet["root_pos"], np.float64)
            quaternion = np.asarray(packet["root_quat_wxyz"], np.float64)
            times = np.asarray(packet["timestamps"], np.float64)
            roles = 2
            joint_velocity = angular_world = None
            schema = "boxing_g1_pair_v1"
        else:
            schema_record = json.loads(SOURCE_SCHEMA.read_text())
            names = (
                tuple(str(v) for v in packet["joint_names"])
                if "joint_names" in packet
                else tuple(schema_record["joint_names"])
            )
            joints = np.asarray(packet["joint_pos"], np.float64)[:, None]
            position = np.asarray(packet["body_pos_w"], np.float64)[:, None, 0]
            quaternion = np.asarray(packet["body_quat_w"], np.float64)[
                :, None, 0
            ]
            fps = np.asarray(packet["fps"]).reshape(-1)
            if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
                raise ValueError("Motion requires a positive scalar fps")
            times = np.arange(len(joints)) / float(fps[0])
            joint_velocity = (
                np.asarray(packet["joint_vel"], np.float64)[:, None]
                if "joint_vel" in packet
                else None
            )
            angular_world = (
                np.asarray(packet["body_ang_vel_w"], np.float64)[:, None, 0]
                if "body_ang_vel_w" in packet
                else None
            )
            roles = 1
            schema = "g1_breadth_first_motion_npz_v1"
    if (
        len(names) != 29
        or len(set(names)) != 29
        or set(names) != set(scene_joint_names)
    ):
        raise ValueError(
            "Motion joint names must match the training robot exactly"
        )
    n = len(joints)
    for name, value, shape in (
        ("joint positions", joints, (n, roles, 29)),
        ("root positions", position, (n, roles, 3)),
        ("root quaternions", quaternion, (n, roles, 4)),
        ("timestamps", times, (n,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}; expected {shape}")
    if n < 2 or not np.all(np.diff(times) > 0) or abs(times[0]) > 1e-8:
        raise ValueError("Need chronological frames starting at zero")
    if not np.allclose(
        np.linalg.norm(quaternion, axis=-1), 1, atol=1e-3, rtol=0
    ):
        raise ValueError("Motion root rotations must be unit WXYZ quaternions")
    source_quaternion = quaternion.copy()
    quaternion = quaternion / np.linalg.norm(
        quaternion, axis=-1, keepdims=True
    )
    order = [names.index(name) for name in scene_joint_names]
    joints = joints[..., order]
    query = np.arange(int(np.floor(times[-1] * 50 + 1e-8)) + 1) / 50
    if len(query) < 6:
        raise ValueError("A source clip needs six 50 Hz frames without future padding")
    # The endpoint can exceed the source by roundoff, not by a real frame.
    sample_times = np.clip(query, times[0], times[-1])

    def interpolate(values):
        flat = values.reshape(n, -1)
        return np.stack(
            [
                np.interp(sample_times, times, flat[:, i])
                for i in range(flat.shape[1])
            ],
            -1,
        ).reshape((len(query),) + values.shape[1:])

    position, joints = interpolate(position), interpolate(joints)
    quaternion = np.stack(
        [
            Slerp(
                times, Rotation.from_quat(quaternion[:, role, [1, 2, 3, 0]])
            )(sample_times).as_quat()[:, [3, 0, 1, 2]]
            for role in range(roles)
        ],
        1,
    )
    if joint_velocity is not None:
        if (
            joint_velocity.shape != (n, roles, 29)
            or not np.isfinite(joint_velocity).all()
        ):
            raise ValueError("Invalid recorded joint velocities")
        joint_velocity = interpolate(joint_velocity[..., order])
    else:
        joint_velocity = backward_velocity(joints)
    if angular_world is not None:
        if (
            angular_world.shape != (n, roles, 3)
            or not np.isfinite(angular_world).all()
        ):
            raise ValueError("Invalid recorded root angular velocity")
        angular_world = interpolate(angular_world)
    else:
        angular_world = np.zeros((len(query), roles, 3))
        for role in range(roles):
            rotations = Rotation.from_quat(quaternion[:, role, [1, 2, 3, 0]])
            angular_world[1:, role] = (
                rotations[1:] * rotations[:-1].inv()
            ).as_rotvec() * 50
    return {
        # Keep original samples when already at 50 Hz. AMP performs its
        # float32 world-to-body rotation in the training runtime.
        "amp_root_quaternion": source_quaternion
        if np.array_equal(query, times)
        else quaternion,
        "root_position": position,
        "root_quaternion": quaternion,
        "root_origin_velocity": backward_velocity(position),
        "root_angular_velocity_world": angular_world,
        "joint_position": joints,
        "joint_velocity": joint_velocity,
        "roles": roles,
        "source_schema": schema,
        "velocity_source": "recorded_joint_and_angular_when_available; causal_backward_root_origin_derivative; current_robot_FK",
    }


def scene_layout(model):
    joints, roots, bodies = [], [], []
    names = None
    for prefix in ("p1_", "p2_"):
        root_joint = int(model.body(prefix + "pelvis").jntadr[0])
        roots.append(
            (
                int(model.jnt_qposadr[root_joint]),
                int(model.jnt_dofadr[root_joint]),
            )
        )
        ids = [
            i
            for i in range(model.njnt)
            if model.joint(i).name.startswith(prefix)
            and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        role_names = tuple(model.joint(i).name[len(prefix) :] for i in ids)
        if names is not None and names != role_names:
            raise ValueError(
                "Training robots have different joint layouts"
            )
        names = role_names
        joints.append(
            (
                np.asarray([model.jnt_qposadr[i] for i in ids]),
                np.asarray([model.jnt_dofadr[i] for i in ids]),
            )
        )
        bodies.append(
            np.asarray(
                [model.body(prefix + name).id for name in INPUT_LINK_NAMES]
            )
        )
    if len(names) != 29:
        raise ValueError("Expected two 29-joint training robots")
    return names, roots, joints, np.asarray(bodies)


def initial_qpos(assets, model):
    names, roots, joints, _ = scene_layout(model)
    reset = json.loads(assets.resources.initial_state.read_text())
    reset_names = reset["dof_names"]
    if set(reset_names) != set(names):
        raise ValueError("Public reset joint names differ from training robots")
    order = [reset_names.index(name) for name in names]
    state = np.asarray(reset["qpos"], np.float64)
    result = model.qpos0.copy()
    for role, ((root_qi, _), (joint_qi, _)) in enumerate(zip(roots, joints)):
        result[root_qi : root_qi + 7] = state[role, :7]
        result[joint_qi] = state[role, 7:][order]
    return result


def motion_kinematics(assets, path):
    """Use competition FK; body origins define poses and linear velocities."""
    model = load_scene_model(assets)
    names, roots, joints, point_ids = scene_layout(model)
    trajectory = load_trajectory(path, names)
    trajectory["joint_names"] = names
    reset = initial_qpos(assets, model)
    data = mujoco.MjData(model)
    frames = len(trajectory["joint_position"])
    output = {
        name: []
        for name in (
            "position",
            "quaternion",
            "link_origin_velocity",
            "pelvis_angular_velocity_b",
            "joint_position",
            "joint_velocity",
        )
    }
    # A single source has an explicitly synthetic standing opponent. A paired
    # source retains both recorded people. Neither uses future opponent poses.
    standing = reset.copy()
    if trajectory["roles"] == 1:
        red_qi, blue_qi = roots[0][0], roots[1][0]
        current = trajectory["root_quaternion"][0, 0]
        old = reset[red_qi + 3 : red_qi + 7]
        yaw_current = Rotation.from_quat(current[[1, 2, 3, 0]]).as_euler(
            "xyz"
        )[2]
        yaw_old = Rotation.from_quat(old[[1, 2, 3, 0]]).as_euler("xyz")[2]
        turn = Rotation.from_euler("z", yaw_current - yaw_old)
        standing[blue_qi : blue_qi + 3] = trajectory["root_position"][
            0, 0
        ] + turn.apply(
            reset[blue_qi : blue_qi + 3] - reset[red_qi : red_qi + 3]
        )
        standing[blue_qi + 3 : blue_qi + 7] = (
            turn
            * Rotation.from_quat(
                reset[blue_qi + 3 : blue_qi + 7][[1, 2, 3, 0]]
            )
        ).as_quat()[[3, 0, 1, 2]]
        trajectory["standing_qpos"] = np.concatenate(
            (standing[blue_qi : blue_qi + 7], standing[joints[1][0]])
        )
    for index in range(frames):
        data.qpos[:] = standing
        data.qvel[:] = 0
        for role in range(trajectory["roles"]):
            root_qi, root_vi = roots[role]
            joint_qi, joint_vi = joints[role]
            q = trajectory["root_quaternion"][index, role]
            data.qpos[root_qi : root_qi + 3] = trajectory["root_position"][
                index, role
            ]
            data.qpos[root_qi + 3 : root_qi + 7] = q
            data.qpos[joint_qi] = trajectory["joint_position"][index, role]
            data.qvel[root_vi : root_vi + 3] = trajectory[
                "root_origin_velocity"
            ][index, role]
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            data.qvel[root_vi + 3 : root_vi + 6] = (
                rotation.T
                @ trajectory["root_angular_velocity_world"][index, role]
            )
            data.qvel[joint_vi] = trajectory["joint_velocity"][index, role]
        mujoco.mj_forward(model, data)
        angular = data.cvel[point_ids, :3]
        linear = data.cvel[point_ids, 3:]
        center = data.subtree_com[model.body_rootid[point_ids]]
        velocity = linear + np.cross(angular, data.xpos[point_ids] - center)
        output["position"].append(data.xpos[point_ids].copy())
        output["quaternion"].append(data.xquat[point_ids].copy())
        output["link_origin_velocity"].append(velocity)
        output["pelvis_angular_velocity_b"].append(
            np.stack(
                [
                    data.xmat[point_ids[role, 0]].reshape(3, 3).T
                    @ angular[role, 0]
                    for role in range(2)
                ]
            )
        )
        output["joint_position"].append(
            np.stack([data.qpos[qi] for qi, _ in joints])
        )
        output["joint_velocity"].append(
            np.stack([data.qvel[vi] for _, vi in joints])
        )
    return {
        name: torch.tensor(np.asarray(value), dtype=torch.float32)
        for name, value in output.items()
    }, trajectory


def prepare_clip(assets, source, output, *, collector):
    arrays, trajectory = motion_kinematics(assets, source)
    n = len(arrays["position"])
    roles = trajectory["roles"]
    amp_joint_names = json.loads(SOURCE_SCHEMA.read_text())["joint_names"]
    amp_order = [
        trajectory["joint_names"].index(name) for name in amp_joint_names
    ]
    amp_reference = []
    amp_root_quaternion = []
    world_angular = torch.as_tensor(
        trajectory["root_angular_velocity_world"], dtype=torch.float32
    )
    root_quaternion = torch.as_tensor(
        trajectory["amp_root_quaternion"], dtype=torch.float32
    )
    for role in range(roles):
        # Keep the world angular velocity and absolute joint states. Training
        # applies its existing rotation and nominal offsets exactly once.
        features = torch.cat(
            (
                world_angular[:, role],
                arrays["joint_position"][:, role, amp_order],
                arrays["joint_velocity"][:, role, amp_order],
            ),
            -1,
        )
        # AMP needs consecutive measurements, not five future pose labels.
        # Preserve every real transition, including the last four per clip.
        amp_reference.append(torch.cat((features[:-1], features[1:]), -1))
        amp_root_quaternion.append(
            torch.stack((root_quaternion[:-1, role], root_quaternion[1:, role]), 1)
        )
    packet = dict(
        format=np.asarray(DATA_FORMAT),
        amp_reference=torch.cat(amp_reference).numpy(),
        amp_root_quaternion_wxyz=torch.cat(amp_root_quaternion).numpy(),
        amp_angular_velocity_frame=np.asarray("world"),
        amp_joint_names=np.asarray(amp_joint_names),
        amp_fps=np.asarray(50),
    )
    report = {
        "source": Path(source).name,
        "source_sha256": sha256(source),
        "source_schema": trajectory["source_schema"],
        "source_joint_schema_sha256": sha256(SOURCE_SCHEMA),
        "velocity_source": trajectory["velocity_source"],
        "frames": n,
        "roles": roles,
        "amp_reference_samples": (n - 1) * roles,
        "opponent": "recorded paired person"
        if roles == 2
        else "synthetic standing opponent from evaluation reset",
        "future_padding": False,
    }
    measured, measured_report = collector.collect(arrays, trajectory)
    packet.update(measured)
    report.update(measured_report)
    np.savez_compressed(output, **packet)
    return report


def _prepare_group(arguments):
    from .collection import MeasuredCollector

    directory, lower, device, tasks = arguments
    torch.set_num_threads(1)
    assets = CompetitionAssets.load(directory)
    collector = MeasuredCollector(assets, lower, device)
    try:
        return [
            (index, prepare_clip(assets, source, output, collector=collector))
            for index, source, output in tasks
        ]
    finally:
        collector.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", default=[])
    parser.add_argument(
        "--motion-directory", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    for directory in args.motion_directory:
        args.motion.extend(sorted(directory.glob("*.npz")))
    if not args.motion or len(
        set(path.resolve() for path in args.motion)
    ) != len(args.motion):
        parser.error(
            "Supply unique motion files or nonempty motion directories"
        )
    if args.workers < 1:
        parser.error("Worker count must be positive")
    from holoboxing.evaluation.models import verify_bundle

    lower = verify_bundle(os.environ.get("HOLOBOXING_LOWER", "models/holomotion-v2"))
    devices = torch.cuda.device_count()
    if not devices:
        raise ValueError("Data preparation requires a CUDA GPU")
    assets = CompetitionAssets.load(args.assets)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty prepared-data directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    tasks = [
        (index, path, args.output / f"clip-{index:04d}.npz")
        for index, path in enumerate(args.motion)
    ]
    workers = min(args.workers, len(tasks))
    groups = [
        (args.assets, lower, f"cuda:{index % devices}", tasks[index::workers])
        for index in range(workers)
    ]
    if workers == 1:
        prepared = _prepare_group(groups[0])
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            prepared = [
                item for group in executor.map(_prepare_group, groups)
                for item in group
            ]
    for index, report in sorted(prepared):
        name = f"clip-{index:04d}.npz"
        report.update(
            path=name,
            sha256=sha256(args.output / name),
            bytes=(args.output / name).stat().st_size,
            source_group=args.motion[index].stem.removesuffix("_mirror"),
        )
        records.append(report)
    manifest = {
        "format": DATA_FORMAT,
        "policy_contract": assets.identity()["policy_contract"],
        "asset_identity": assets.identity(),
        "control_hz": 50,
        "records": records,
        "state": "complete",
        "collection_recipe": records[0]["collection_recipe"],
    }
    from .common import write_json

    write_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.output / "manifest.json"),
                "manifest_sha256": sha256(args.output / "manifest.json"),
                "clips": len(records),
                "samples": sum(row["samples"] for row in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
