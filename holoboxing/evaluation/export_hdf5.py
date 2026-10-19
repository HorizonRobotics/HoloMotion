"""Convert recorded match states to the existing HoloBoxing HDF5 contract."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from .replay_schema import ReplayData, read_replay, write_replay


def export(source, destination, context=None, asset_manifest=None):
    source, destination = Path(source), Path(destination)
    result = json.loads(source.read_text())
    frames = result["frames"]
    metadata = result["metadata"]
    if not metadata.get("match_completed") or len(frames) < 2:
        raise ValueError("A completed, nonempty match is required")
    context = context or {}
    fps = float(metadata["control_hz"])
    count = len(frames)
    positions = np.asarray([f["dof_pos"] for f in frames], dtype=np.float64)
    round_ids = []
    round_id = 1
    for frame in frames:
        round_ids.append(round_id)
        if any(e["type"] == "ROUND_END" for e in frame["events"]):
            round_id += 1
    native = all("joint_velocity" in f for f in frames)
    if native:
        velocities = np.asarray(
            [f["joint_velocity"] for f in frames], dtype=np.float64
        )
        velocity_valid = np.ones(count, dtype=np.bool_)
    else:
        # Old JSON never recorded velocity. Label finite differences explicitly;
        # neither interpolation across teleports nor fabricated native velocity.
        velocities = np.zeros_like(positions)
        velocity_valid = np.zeros(count, dtype=np.bool_)
        for i in range(1, count):
            if round_ids[i] == round_ids[i - 1]:
                velocities[i] = (positions[i] - positions[i - 1]) * fps
                velocity_valid[i] = True
    state = np.zeros((count, 200), dtype=np.float64)
    robots = {}
    actions = np.asarray([f["action_raw"] for f in frames], dtype=np.float64)
    for fighter, side in enumerate(("red", "blue")):
        offset = fighter * 100
        state[:, offset : offset + 3] = np.asarray(
            [f["root_pos"][fighter] for f in frames]
        )
        state[:, offset + 3 : offset + 7] = np.asarray(
            [f["root_quat"][fighter] for f in frames]
        )
        state[:, offset + 13 : offset + 42] = positions[:, fighter]
        state[:, offset + 42 : offset + 71] = velocities[:, fighter]
        state[:, offset + 71 : offset + 100] = actions[:, fighter]
        fields = {
            name: {"offset": offset + start, "size": size}
            for name, start, size in [
                ("root_position", 0, 3),
                ("root_quaternion", 3, 4),
                ("joint_position", 13, 29),
                ("joint_velocity", 42, 29),
                ("action_raw", 71, 29),
            ]
        }
        if native:
            state[:, offset + 7 : offset + 10] = [
                f["root_linear_velocity"][fighter] for f in frames
            ]
            state[:, offset + 10 : offset + 13] = [
                f["root_angular_velocity_world"][fighter] for f in frames
            ]
            fields.update(
                root_linear_velocity={"offset": offset + 7, "size": 3},
                root_angular_velocity={"offset": offset + 10, "size": 3},
            )
        robots[side] = {"joint_names": metadata["dof_names"], "fields": fields}
        if len(set(metadata["dof_names"])) != 29:
            raise ValueError("Expected 29 unique recorded joint names")
        np.testing.assert_allclose(
            np.linalg.norm(state[:, offset + 3 : offset + 7], axis=1),
            1,
            atol=1e-4,
            rtol=0,
        )
    events = []
    raw_events = []
    for index, frame in enumerate(frames):
        for event in frame["events"]:
            side = lambda value: (
                ("red", "blue")[value]
                if isinstance(value, int) and value in (0, 1)
                else str(value)
            )
            events.append(
                {
                    "frame": index,
                    "type": event["type"],
                    "actor": side(event.get("actor", "system")),
                    "target": side(event.get("target", "both")),
                    "value": float(sum(event.get("score_delta", [0, 0]))),
                }
            )
            raw_events.append(dict(event, recorded_frame_index=index))
    summary = result["summary"]
    score = summary["final_score"]
    winner = summary["winner"]
    asset_manifest = Path(asset_manifest) if asset_manifest else None
    asset_sha = (
        hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
        if asset_manifest
        else metadata["retargeted_sha256"]
    )
    render_theme = (
        json.loads(asset_manifest.read_text()).get("render_theme")
        if asset_manifest
        else None
    )
    scoring_version = metadata.get(
        "scoring_version", "local_boxing_original_rules_v1"
    )
    arena_version = metadata.get("arena_version", "holoboxing-g1-boxing")
    meta = {
        "schema_version": "1.1",
        "match_id": context.get("match_id", metadata["match_id"]),
        "seed": None,
        "frame_count": count,
        "duration_seconds": (count - 1) / fps,
        "control_fps": fps,
        "timebase": {
            "mode": "explicit_timestamps",
            "nominal_step_seconds": 1 / fps,
            "physics_dt": 1 / float(metadata["physics_hz"]),
            "decimation": int(metadata["physics_hz"] / fps),
        },
        "state_schema": {
            "id": "g1_kinematic_v1",
            "version": 1,
            "dataset": "frames/state",
            "width": 200,
            "coordinate_system": "right-handed-z-up",
            "quaternion_order": "wxyz",
            "robots": robots,
        },
        "engine": "g1_boxing_holomotion_v2",
        "scoring_version": scoring_version,
        "render_model": {"manifest": "/assets/g1-boxing/manifest.json"},
        "environment": {
            "environment_version": "local-boxing-gpu-hdf5-v2"
            if arena_version.startswith("smplolympics-boxing-ring-v")
            else "local-boxing-gpu-hdf5-v1",
            "simulator_version": metadata["physics_backend"],
            "asset_version": arena_version,
            "asset_sha256": asset_sha,
            "controller_version": metadata["holomotion_version"],
            "scoring_version": scoring_version,
        },
        "development_only": True,
        "official": False,
        "result": {
            "winner": "draw" if winner is None else ("red", "blue")[winner],
            "score_red": score[0],
            "score_blue": score[1],
        },
        "source": {
            "json_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "match_metadata": metadata,
            "summary": summary,
            "first_post_step_match_time_seconds": 1 / fps,
            "replay_time_origin": "first recorded post-step; no synthetic initial state",
            "joint_velocity_source": "recorded"
            if native
            else "finite_difference_of_recorded_joint_position; reset rows invalid",
        },
        "requested_duration_seconds": metadata["num_rounds"]
        * metadata["round_duration_s"],
    }
    if render_theme and arena_version.startswith("smplolympics-boxing-ring-v"):
        meta["render_model"]["arena_theme"] = render_theme["version"]
    if scoring_version in (
        "hands_only_normal_impulse100ms_5Ns_v2",
        "hands_only_normal_impulse100ms_5Ns_ko1000_v3",
    ):
        meta["render_model"]["arena_theme"] = "holomotion-hsv-zero-v1"
        meta["scoring_config"] = dict(
            window_ms=100,
            normal_impulse_threshold_ns=5,
            rearm_no_contact_ms=100,
            force_threshold_enabled=False,
            speed_threshold_enabled=False,
            normal_force_source="sum of measured contact-frame normal components at 200Hz",
        )
    if scoring_version == "hands_only_normal_impulse100ms_5Ns_ko1000_v3":
        meta["scoring_config"].update(round_duration_s=60, ko_bonus=1000)
    if (
        scoring_version in (
            "hands_only_normal_impulse100ms_5Ns_cooldown100ms_fall030_ko1000_v4",
            "hands_only_normal_impulse100ms_10Ns_cooldown100ms_fall030_ko1000_v5",
        )
    ):
        if not isinstance(metadata.get("scoring_config"), dict):
            raise ValueError("v4/v5 requires recorded scoring_config")
        meta["scoring_config"] = dict(metadata["scoring_config"])
        meta["render_model"]["arena_theme"] = "holomotion-hsv-zero-v1"
        meta["environment"]["environment_version"] = (
            "boxing-track2-20261015-v5" if scoring_version.endswith("_v5")
            else "boxing-track2-20261014-v4"
        )
    for fighter, side in enumerate(("red", "blue")):
        item = context.get(side, {})
        hashes = (
            metadata.get("upper_policy_sha256") or metadata["policy_sha256"]
        )
        identities = summary["model_id"]
        meta[side] = {
            "name": item.get("name", item.get("team_name", side)),
            "team_id": str(item.get("team_id", side)),
            "model_id": str(item.get("submission_id", identities[fighter])),
            "model_sha256": item.get("model_sha256", hashes[fighter]),
            "controller_kind": "onnx_holomotion_vr5",
        }
    replay = ReplayData(
        meta=meta,
        time=np.arange(count, dtype=np.float64) / fps,
        state=state,
        score=np.asarray([f["score"] for f in frames], dtype=np.float64),
        rounds=np.asarray(round_ids, dtype=np.int32),
        events=events,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hdf5-", dir=destination.parent
    ) as staging:
        staged = Path(staging) / "replay.hdf5"
        write_replay(staged, replay)
        # Preserve original event details and every additional recorded frame field.
        import h5py

        with h5py.File(staged, "a") as file:
            strings = h5py.string_dtype("utf-8")
            file.create_dataset(
                "events/json",
                data=[json.dumps(e, ensure_ascii=False) for e in raw_events],
                dtype=strings,
            )
            if any("contacts" in frame for frame in frames):
                contacts = [
                    dict(c, recorded_frame_index=i)
                    for i, f in enumerate(frames)
                    for c in f.get("contacts", [])
                ]
                file.create_dataset(
                    "contacts/json",
                    data=[
                        json.dumps(c, ensure_ascii=False, allow_nan=False)
                        for c in contacts
                    ],
                    dtype=strings,
                    compression="gzip" if contacts else None,
                )
            file.create_dataset(
                "frames/joint_velocity_valid", data=velocity_valid
            )
            for key in (
                "action_raw",
                "policy_start_ns",
                "policy_end_ns",
                "upper_policy_start_ns",
                "upper_policy_end_ns",
                "control_step_start_ns",
                "control_step_end_ns",
                "competitor_sanitization_ns",
            ):
                if all(key in f for f in frames):
                    file.create_dataset(
                        "policy/" + key,
                        data=np.asarray([f[key] for f in frames]),
                        compression="gzip",
                    )
            if all("competitor_sanitization" in frame for frame in frames):
                file.create_dataset(
                    "policy/competitor_sanitization_json",
                    data=[
                        json.dumps(
                            frame["competitor_sanitization"], allow_nan=False
                        )
                        for frame in frames
                    ],
                    dtype=strings,
                    compression="gzip",
                )
        checked = read_replay(staged)
        np.testing.assert_array_equal(checked.score, replay.score)
        for fighter in range(2):
            np.testing.assert_array_equal(
                checked.state[:, fighter * 100 + 13 : fighter * 100 + 42],
                positions[:, fighter],
            )
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(staged, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return {
        "match_id": meta["match_id"],
        "frames": count,
        "joint_velocity_source": meta["source"]["joint_velocity_source"],
        "bytes": destination.stat().st_size,
        "score": score,
        "requested_duration_seconds": meta["requested_duration_seconds"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--context")
    p.add_argument("--asset-manifest")
    args = p.parse_args()
    context = (
        json.loads(Path(args.context).read_text()) if args.context else None
    )
    print(
        json.dumps(
            export(args.source, args.destination, context, args.asset_manifest)
        )
    )
