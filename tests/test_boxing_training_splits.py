"""Keep held-out motion groups out of training and fitted statistics."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from holoboxing.data.collection import RECIPE, sample_indices
from holoboxing.data.common import sha256, write_json
from holoboxing.data.contract import ROBOT_KINEMATICS
from holoboxing.data.records import DATA_FORMAT
from holoboxing.evaluation.models import frozen_manifest
from holoboxing.evaluation.runtime.model_contract import CONTRACT
from holoboxing.training.rsl_rl.modules.amp import motion_files
from holoboxing.training.supervised.data import Records
from holoboxing.training.supervised.statistics import fit_statistics
from holoboxing.training.supervised.training import FORMAT, load_supervised
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def population(root):
    prepared = root / "prepared"
    prepared.mkdir()
    files = frozen_manifest()["files"]
    recipe = {
        **RECIPE,
        "lower_identity": {
            name: files[name]["sha256"]
            for name in ("exported/model_10500.onnx", "config.yaml")
        },
    }
    sources, records = [], []
    for name, dataset, group, split, value in (
        ("human", "interhuman", "human", "train", 1),
        ("human_mirror", "interhuman", "human", "train", 1),
        ("heldout", "interhuman", "heldout", "validation", 100),
        ("robot", "robostriker", "robot", "train", 5),
        ("robot-heldout", "robostriker", "robot-heldout", "validation", 200),
    ):
        path = prepared / (name + ".npz")
        indices = sample_indices(8, 2)
        target = np.tile(
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32), (6, 25)
        )
        np.savez(
            path,
            format=np.asarray(DATA_FORMAT),
            obs=np.full((6, 1097), value, np.float32),
            target_pose=target,
            sample_index=indices,
            window_index=np.zeros((6, 2), np.int64),
        )
        digest = hashlib.sha256(name.encode()).hexdigest()
        sources.append(
            {
                "path": path.name,
                "sha256": digest,
                "dataset": dataset,
                "group": group,
                "split": split,
            }
        )
        records.append(
            {
                "path": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "source_sha256": digest,
                "source_group": group,
                "samples": 6,
                "collection_recipe": recipe,
                "observation_source": RECIPE["observation_source"],
                "frames": 8,
                "roles": 2,
                "planned_samples": 6,
                "rejected_samples": 0,
                "early_failures": [],
            }
        )
    manifest = prepared / "manifest.json"
    write_json(
        manifest,
        {
            "format": DATA_FORMAT,
            "state": "complete",
            "collection_recipe": recipe,
            "policy_contract": CONTRACT,
            "control_hz": 50,
            "asset_identity": {"robot_kinematics": ROBOT_KINEMATICS},
            "records": records,
        },
    )
    write_json(
        root / "sources.json",
        {
            "format": "holoboxing_data_sources_v1",
            "expected_source_clips": len(sources),
            "records": sources,
        },
    )
    receipt(root)
    return manifest


def receipt(root):
    write_json(
        root / "COMPLETE.json",
        {
            "sources_sha256": sha256(root / "sources.json"),
            "prepared_manifest_sha256": sha256(
                root / "prepared/manifest.json"
            ),
        },
    )


def test_training_statistics_and_amp_exclude_heldout_groups(tmp_path):
    manifest = population(tmp_path)
    train = Records(manifest)
    heldout = Records(manifest, split="validation")
    assert train.sample_count() == 12
    assert heldout.sample_count() == 6
    assert {r["source_group"] for r in train.records}.isdisjoint(
        r["source_group"] for r in heldout.records
    )
    statistics, normalizer = fit_statistics(train, batch_size=4)
    assert statistics["count"] == int(normalizer.count) == 12
    assert statistics["source_selection"]["split"] == "train"
    assert normalizer.self_mean.eq(1).all()
    assert {p.name for p in motion_files(manifest.parent)} == {
        "human.npz",
        "human_mirror.npz",
        "robot.npz",
    }
    with pytest.raises(ValueError, match="train split only"):
        fit_statistics(heldout)


def test_original_and_mirror_cannot_cross_splits(tmp_path):
    manifest = population(tmp_path)
    path = tmp_path / "sources.json"
    sources = json.loads(path.read_text())
    sources["records"][1]["split"] = "validation"
    write_json(path, sources)
    receipt(tmp_path)
    with pytest.raises(ValueError, match="source group crosses"):
        Records(manifest)


def test_tampered_split_receipt_is_rejected(tmp_path):
    manifest = population(tmp_path)
    path = tmp_path / "sources.json"
    sources = json.loads(path.read_text())
    sources["records"][2]["split"] = "train"
    write_json(path, sources)
    with pytest.raises(ValueError, match="completion receipt"):
        Records(manifest)


def test_legacy_checkpoint_without_train_split_provenance_cannot_initialize_rl(
    tmp_path,
):
    config = TrainConfig()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format": FORMAT,
            "config": asdict(config),
            "loss_statistics": {"count": 1},
        },
        path,
    )
    with pytest.raises(ValueError, match="verified training split provenance"):
        load_supervised(path, config, None)
