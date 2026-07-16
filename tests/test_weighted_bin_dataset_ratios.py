import json

import pytest

from holomotion.src.training.h5_dataloader import (
    _collect_manifest_keys,
    _configure_weighted_bins,
    normalize_hdf5_root_entries,
    weighted_bin_cfg_from_motion_cfg,
)


def _write_manifest(root, clip_key):
    root.mkdir(parents=True)
    path = root / "manifest.json"
    path.write_text(
        json.dumps({"clips": {clip_key: {"length": 10}}}),
        encoding="utf-8",
    )
    return str(path)


def test_dataset_ratios_classify_by_manifest_directory(tmp_path):
    amass = _write_manifest(
        tmp_path / "train_amass" / "final" / "holoretarget_h5",
        "shared_clip_key",
    )
    motionmillion = _write_manifest(
        tmp_path / "train_motionmillion" / "final" / "holoretarget_h5",
        "shared_clip_key",
    )
    custom = _write_manifest(
        tmp_path / "train_custom_smplx" / "final" / "holoretarget_h5",
        "custom_clip",
    )

    keys, _, _ = _collect_manifest_keys([amass, motionmillion, custom])
    bins, ratios, specs = _configure_weighted_bins(
        keys,
        {
            "dataset_ratios": {
                "train_amass": 0.1,
                "train_motionmillion": 0.02,
            }
        },
        batch_size_for_log=100,
    )

    assert keys == [
        "trainamass::shared_clip_key",
        "trainmotionmillion::shared_clip_key",
        "traincustomsmplx::custom_clip",
    ]
    assert [len(indices) for indices in bins] == [1, 1]
    assert ratios == pytest.approx([0.1 / 0.12, 0.02 / 0.12])
    assert [spec["name"] for spec in specs] == [
        "train_amass",
        "train_motionmillion",
    ]


def test_legacy_regex_configuration_remains_supported():
    bins, ratios, _ = _configure_weighted_bins(
        ["AMASS_clip", "other_clip"],
        {"bin_regex_patterns": [{"regex": ".*AMASS.*", "ratio": 0.1}]},
        batch_size_for_log=10,
    )

    assert [len(indices) for indices in bins] == [1, 1]
    assert ratios == [0.1, 0.9]


def test_ratios_follow_train_root_entries_and_override_inherited_regex():
    roots = [
        {"root": "/run/train_amass/final/holoretarget_h5", "ratio": 0.1},
        {
            "root": "/run/train_motionmillion/final/holoretarget_h5",
            "ratio": 0.02,
        },
        {"root": "/run/train_custom_smplx/final/holoretarget_h5"},
    ]
    motion_cfg = {
        "train_hdf5_roots": roots,
        "sampling_strategy": "weighted_bin",
        "weighted_bin": {
            "bin_regex_patterns": [{"regex": "legacy", "ratio": 1.0}],
        },
    }

    paths, ratios = normalize_hdf5_root_entries(roots)
    weighted_cfg = weighted_bin_cfg_from_motion_cfg(motion_cfg)

    assert paths == [entry["root"] for entry in roots]
    assert ratios == {"trainamass": 0.1, "trainmotionmillion": 0.02}
    assert weighted_cfg["dataset_ratios"] == ratios
    assert "bin_regex_patterns" not in weighted_cfg
