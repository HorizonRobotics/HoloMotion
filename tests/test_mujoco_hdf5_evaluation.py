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

"""HDF5 loading, CPU evaluation, and fail-closed orchestration regressions."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import onnx
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals
from omegaconf import OmegaConf
from test_checkpoint_onnx_parity import fixture_actor

from holomotion.src.evaluation import mujoco_hdf5_evaluation as runtime
from holomotion.src.evaluation.checkpoint_onnx_parity import (
    actor_example_from_flat,
    construct_checkpoint_actor,
)
from holomotion.src.evaluation.mujoco_hdf5_evaluation import (
    build_hdf5_evaluation_dataset,
)
from holomotion.src.utils.config import setup_hydra_resolvers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hdf5_config(tmp_path):
    lengths = [8, 12]
    frames = sum(lengths)
    root_pos = np.zeros((frames, 3), dtype=np.float32)
    root_pos[:, 2] = 0.8
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1
    with h5py.File(tmp_path / "motions.h5", "w") as handle:
        handle["ref_root_pos"] = root_pos
        handle["ref_root_rot"] = root_rot
        handle["ref_dof_pos"] = np.zeros((frames, 29), dtype=np.float32)
    clips = {
        name: {
            "shard": 0,
            "start": start,
            "length": length,
            "metadata": {"motion_fps": 50.0},
        }
        for name, start, length in zip(
            ["short", "long"], [0, 8], lengths, strict=True
        )
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"hdf5_shards": [{"file": "motions.h5"}], "clips": clips})
    )
    urdf = (
        Path(__file__).resolve().parents[1]
        / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
    )
    return {
        "evaluation_data": {
            "manifest_path": str(manifest),
            "clip_names": ["short", "long"],
            "fk_robot_file_path": str(urdf),
            "reference_horizon": 5,
        }
    }


def test_hdf5_full_clips_load_into_existing_evaluator_without_npz(
    hdf5_config, tmp_path
):
    from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator

    dataset, index = build_hdf5_evaluation_dataset(hdf5_config)
    try:
        assert set(index) == {"short", "long"}
        assert len(dataset) == 2
        for name, frames in [("short", 8), ("long", 12)]:
            sample = dataset[index[name]]
            assert sample.length == frames
            assert sample.tensors["ref_rg_pos"].shape[0] == frames
            evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
            evaluator.config = SimpleNamespace(
                robot=SimpleNamespace(
                    body_names=dataset._fk_transform._fk.body_names,
                )
            )
            evaluator.num_actions = 29
            evaluator.n_fut_frames = 5
            arrays = {
                key: value.numpy()
                for key, value in sample.tensors.items()
                if key.startswith("ref_")
            }
            evaluator.load_motion_arrays(
                arrays, motion_key=name, reference_horizon=5
            )
            assert evaluator.n_motion_frames == frames
            assert evaluator.eval_motion_key == name
            np.testing.assert_array_equal(
                evaluator.ref_global_translation, arrays["ref_rg_pos"]
            )
            np.testing.assert_array_equal(
                evaluator.ref_global_rotation_quat_xyzw, arrays["ref_rb_rot"]
            )
        assert not list(tmp_path.rglob("*.npz"))
    finally:
        dataset.close()


@pytest.mark.parametrize("names", [["missing"], ["short", "short"], []])
def test_hdf5_rejects_wrong_or_duplicate_selection(hdf5_config, names):
    hdf5_config["evaluation_data"]["clip_names"] = names
    with pytest.raises(ValueError):
        build_hdf5_evaluation_dataset(hdf5_config)


def test_hdf5_selection_json_and_horizon(hdf5_config, tmp_path):
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({"val_handpicked_motion_names": ["long"]}))
    options = hdf5_config["evaluation_data"]
    options["selection_path"] = str(selection)
    options["clip_names"] = None
    dataset, index = build_hdf5_evaluation_dataset(hdf5_config)
    try:
        assert list(index) == ["long"]
        assert dataset[index["long"]].length == 12
    finally:
        dataset.close()
    options["reference_horizon"] = 12
    with pytest.raises(ValueError, match="future horizon"):
        build_hdf5_evaluation_dataset(hdf5_config)


def test_gate_requires_matching_identity_and_eight_fp32_steps(tmp_path):
    gate = tmp_path / "gate.json"
    passed = {
        "status": "passed",
        "checked_steps": 8,
        "identity_sha256": "same",
        "torch_allow_tf32": False,
    }
    for field, value in (
        ("status", "failed"),
        ("checked_steps", 7),
        ("identity_sha256", "different"),
        ("torch_allow_tf32", True),
    ):
        gate.write_text(json.dumps({**passed, field: value}))
        with pytest.raises(ValueError):
            runtime.verify_attestation(gate, {"sha256": "same"})
    gate.write_text(json.dumps(passed))
    runtime.verify_attestation(gate, {"sha256": "same"})


def test_failed_gate_never_launches_workers_or_aggregation(
    tmp_path, monkeypatch
):
    config = OmegaConf.create(
        {"evaluation_run": {"output_dir": str(tmp_path / "result")}}
    )
    monkeypatch.setattr(
        runtime, "resolve_hdf5_config", lambda cfg: (cfg, ["clip"])
    )
    monkeypatch.setattr(
        runtime, "runtime_identity", lambda cfg: {"sha256": "identity"}
    )
    stages = []

    def fail_gate(path, *args):
        stages.append(args)
        raise RuntimeError("parity failed")

    monkeypatch.setattr(runtime, "launch_stage", fail_gate)
    with pytest.raises(RuntimeError, match="parity failed"):
        runtime.run_hdf5_evaluation(config)
    assert stages == [("--gate",)]
    assert not (tmp_path / "result/done.json").exists()
    assert (tmp_path / "result/failed.json").is_file()


def test_full_cpu_hdf5_gate_rollout_mask_and_aggregation(tmp_path):
    """Test real PD rollout, not model quality, using a small random actor."""
    setup_globals()
    setup_hydra_resolvers()
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        train = compose(
            config_name="training/motion_tracking/motion_tracking_v2_0_0"
        )
    OmegaConf.set_struct(train, False)
    train.experiment_name = "fixture"
    train.num_envs = 1
    train.num_processes = 1
    train.main_process = True
    train.process_id = 0
    train.headless = True
    train.motion_h5_path = str(tmp_path / "data")
    train.experiment_save_dir = str(tmp_path / "model")
    cfg_actor, actor = fixture_actor(include_error_terms=True, frames=5)
    train.algo = OmegaConf.merge(train.algo, cfg_actor.algo)
    train.algo.config.module_dict.actor = (
        cfg_actor.algo.config.module_dict.actor
    )
    actor = construct_checkpoint_actor(
        train, actor_example_from_flat(torch.zeros(1, 1378), train)
    )
    model_root = tmp_path / "model"
    actor_dir = model_root / "model_0/actor"
    actor_dir.mkdir(parents=True)
    torch.save(actor.state_dict(), actor_dir / "pytorch_model.bin")
    onnx_path = model_root / "exported/model_0.onnx"
    onnx_path.parent.mkdir()
    actor.export_onnx(onnx_path)
    graph = onnx.load(onnx_path)
    metadata = {
        "joint_names": ",".join(train.robot.dof_names),
        "action_scale": ",".join(["0.1"] * 29),
        "joint_stiffness": ",".join(["20"] * 29),
        "joint_damping": ",".join(["2"] * 29),
        "default_joint_pos": ",".join(["0"] * 29),
    }
    onnx.helper.set_model_props(graph, metadata)
    onnx.save(graph, onnx_path)
    # Keep G1 inertias, joints and collisions; omit visual meshes.
    xml = ET.parse(ROOT / "assets/robots/unitree/G1/29dof/g1_29dof.xml")
    for parent in xml.iter():
        for child in list(parent):
            if child.tag == "mesh" or (
                child.tag == "geom" and "mesh" in child.attrib
            ):
                parent.remove(child)
    xml_path = tmp_path / "g1_primitives.xml"
    xml.write(xml_path)
    data = tmp_path / "data"
    data.mkdir()
    lengths = [13, 15]
    frames = sum(lengths)
    pos = np.zeros((frames, 3), np.float32)
    pos[:, 2] = 0.8
    rot = np.zeros((frames, 4), np.float32)
    rot[:, 3] = 1
    with h5py.File(data / "motions.h5", "w") as h5:
        h5["ref_root_pos"] = pos
        h5["ref_root_rot"] = rot
        h5["ref_dof_pos"] = np.zeros((frames, 29), np.float32)
    manifest = {
        "hdf5_shards": [{"file": "motions.h5"}],
        "clips": {
            name: {
                "shard": 0,
                "start": start,
                "length": length,
                "metadata": {"motion_fps": 50.0},
            }
            for name, start, length in zip(
                ("a", "b"), (0, 13), lengths, strict=True
            )
        },
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    OmegaConf.save(train, model_root / "config.yaml")
    evaluation = OmegaConf.load(
        ROOT / "holomotion/config/evaluation/eval_mujoco_sim2sim.yaml"
    )
    del evaluation["defaults"]
    cfg = OmegaConf.merge(
        train,
        evaluation,
        {
            "ckpt_onnx_path": str(onnx_path),
            "use_gpu": False,
            "headless": True,
            "robot_xml_path": str(xml_path),
            "link_eval": {
                "mask_mode": "custom",
                "active_link_names": ["pelvis"],
            },
            "evaluation_data": {
                "backend": "hdf5_v2",
                "manifest_path": str(data / "manifest.json"),
                "clip_names": ["a", "b"],
                "reference_horizon": 5,
            },
            "evaluation_run": {
                "output_dir": str(tmp_path / "result"),
                "workers": 2,
                "gpus": 1,
                "phase": "pelvis",
            },
        },
    )
    summary = runtime.run_hdf5_evaluation(cfg)
    assert summary["pelvis"]["status"] == "completed"
    assert summary["pelvis"]["coverage"]["active_link_names"] == ["pelvis"]
    assert summary["pelvis"]["coverage"]["completed_clips"] == 2
    gate = json.loads((tmp_path / "result/checkpoint_parity.json").read_text())
    assert gate["status"] == "passed" and gate["checked_steps"] == 8
    for worker in range(2):
        row = json.loads(
            (
                tmp_path
                / f"result/pelvis/results/worker{worker}/per_clip.json"
            ).read_text()
        )[0]
        assert row["executed_policy_steps"] == row["reference_frames"] - 5
    assert not list(tmp_path.rglob("*.npz"))
    with pytest.raises(FileExistsError):
        runtime.run_hdf5_evaluation(cfg)


@pytest.mark.parametrize(
    "names", [["same", "same"], ["../escape"], ["/absolute"], []]
)
def test_group_names_reject_duplicates_and_unsafe_paths(names):
    cfg = OmegaConf.create(
        {
            "evaluation_run": {
                "groups": [
                    {"name": name, "link_eval": {"mask_mode": "all14"}}
                    for name in names
                ]
            }
        }
    )
    with pytest.raises(ValueError):
        runtime.evaluation_groups(cfg)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"evaluation_data": {"backend": "hdf5"}}, "backend=hdf5_v2"),
        (
            {"ckpt_onnx_root_dir": "/unused/models"},
            "one explicit ckpt_onnx_path",
        ),
        (
            {"ckpt_onnx_names": ["a.onnx", "b.onnx"]},
            "one explicit ckpt_onnx_path",
        ),
        ({"ckpt_onnx_names": []}, "one explicit ckpt_onnx_path"),
        ({"metric_calculation": "per_frame"}, "metric_calculation=per_clip"),
        ({"generate_report": True}, "generate_report is unsupported"),
        ({"dump_onnx_io_npy": True}, "dump_onnx_io_npy is unsupported"),
        ({"max_policy_steps": 20}, "max_policy_steps must be 0"),
    ],
)
def test_hdf5_rejects_unsupported_options_before_opening_inputs(
    settings, message
):
    config = OmegaConf.merge(
        {
            "evaluation_data": {"backend": "hdf5_v2"},
            "evaluation_run": {"output_dir": None},
        },
        settings,
    )
    with pytest.raises(ValueError, match=message):
        runtime.resolve_hdf5_config(config)


def test_npz_only_options_are_reported_as_ignored():
    config = OmegaConf.create(
        {
            "evaluation_data": {"backend": "hdf5_v2"},
            "evaluation_run": {"output_dir": None},
            "link_observation": {"enabled": True},
            "dof_mode": "23",
            "ray_actors_per_gpu": 16,
            "dump_npzs": True,
        }
    )
    with pytest.warns(
        UserWarning, match="ignores NPZ-only options"
    ) as warnings:
        with pytest.raises(ValueError, match="fresh directory"):
            runtime.resolve_hdf5_config(config)
    assert len(warnings) == 1
    assert "link_eval" in str(warnings[0].message)
    assert "evaluation_run.workers" in str(warnings[0].message)


@pytest.mark.parametrize("backend", ["hdf5", "hdf5_v22", "invalid"])
def test_existing_entry_rejects_unknown_backend(monkeypatch, backend):
    from holomotion.src.evaluation import eval_mujoco_sim2sim as entry

    monkeypatch.setattr(
        entry.hydra.utils, "get_original_cwd", lambda: str(ROOT)
    )
    monkeypatch.setattr(entry, "process_config", lambda config: config)
    config = OmegaConf.create({"evaluation_data": {"backend": backend}})
    with pytest.raises(ValueError, match="npz or hdf5_v2"):
        entry.run_mujoco_sim2sim_eval(config)


def test_existing_entry_dispatches_hdf5_v2(monkeypatch):
    from holomotion.src.evaluation import eval_mujoco_sim2sim as entry

    monkeypatch.setattr(
        entry.hydra.utils, "get_original_cwd", lambda: str(ROOT)
    )
    monkeypatch.setattr(entry, "process_config", lambda config: config)
    config = OmegaConf.create({"evaluation_data": {"backend": "hdf5_v2"}})
    seen = []

    def run(cfg):
        seen.append(cfg)
        return "delegated"

    monkeypatch.setattr(runtime, "run_hdf5_evaluation", run)
    assert entry.run_mujoco_sim2sim_eval(config) == "delegated"
    assert seen == [config]
