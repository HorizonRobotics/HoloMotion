"""Protect sample downloads, provenance and shared pair geometry."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from holoboxing.data.common import read_interhuman, sha256
from holoboxing.data.interhuman import shared_pair_positions
from holoboxing.data.prepare import assign_splits


def test_interhuman_rejects_empty_clip_ids_before_creating_output(tmp_path):
    from holoboxing.data.download import download_interhuman

    output = tmp_path / "sample"
    with pytest.raises(ValueError, match="clip IDs must not be empty"):
        download_interhuman(output, clip_ids=[])
    assert not output.exists()


@pytest.mark.parametrize("limit_clips", [None, 1])
@pytest.mark.parametrize("use_default", [False, True])
def test_interhuman_download_uses_ids_not_annotation_words(
    tmp_path, monkeypatch, limit_clips, use_default
):
    from holoboxing.data import download
    from holoboxing.data.interhuman import downloaded_records

    contents = {
        "motions/1.pkl": b"first motion",
        "motions/2.pkl": b"second motion",
        "motions/3.pkl": b"unrequested motion",
        "annots/1.txt": b"standing still",
        "annots/2.txt": b"walking",
        "annots/3.txt": b"boxing",
        "split/train.txt": b"1\n3\n",
        "split/val.txt": b"2\n",
        "split/test.txt": b"",
        "LICENSE.md": b"source license",
    }
    listing = [
        {"path": name, "google_drive_id": "id-" + str(index)}
        for index, name in enumerate(contents)
    ]

    def discover(output):
        (output / "source-listing.json").write_text(json.dumps(listing))
        return listing

    def fetch(record, partial, retries):
        partial.write_bytes(contents[record["path"]])

    monkeypatch.setattr(download, "discover_interhuman", discover)
    monkeypatch.setattr(download, "_google_download", fetch)
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps({"files": ["motions/2.pkl", "motions/1.pkl"]})
    )
    monkeypatch.setattr(download, "INTERHUMAN_SAMPLE_LIST", sample)
    report = download.download_interhuman(
        tmp_path,
        clip_ids=None if use_default else ["2", "1"],
        limit_clips=limit_clips,
        quiet=True,
    )
    expected = ["1", "2"] if limit_clips is None else ["2"]
    assert report["requested_clip_ids"] == ["2", "1"]
    assert report["sample_clip_list_sha256"] == (
        sha256(sample) if use_default else None
    )
    assert report["selected_motion_files"] == len(expected)
    records = downloaded_records(tmp_path)
    assert [record["path"] for record in records] == [
        f"motions/{clip}.pkl" for clip in expected
    ]
    assert [record["source_split"] for record in records] == [
        {"1": "train", "2": "validation"}[clip] for clip in expected
    ]
    assert not (tmp_path / "motions/3.pkl").exists()
    assert not (tmp_path / "annots/3.txt").exists()
    (tmp_path / "split/train.txt").write_text("2\n")
    with pytest.raises(ValueError, match="Downloaded split changed"):
        downloaded_records(tmp_path)


@pytest.mark.parametrize(
    "packet",
    [
        {"files": []},
        {"files": ["wrong/1.pkl"]},
        {"files": ["motions/../1.pkl"]},
        {"files": ["motions/1.npz"]},
        {"files": ["motions/1.pkl", "motions/1.pkl"]},
    ],
)
def test_reject_invalid_bundled_sample_list(tmp_path, monkeypatch, packet):
    from holoboxing.data import download

    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(packet))
    monkeypatch.setattr(download, "INTERHUMAN_SAMPLE_LIST", sample)
    with pytest.raises(ValueError):
        download.sample_clip_ids()


def test_interhuman_unknown_ids_fail_before_downloading(tmp_path, monkeypatch):
    from holoboxing.data import download

    monkeypatch.setattr(
        download,
        "discover_interhuman",
        lambda _: [{"path": "motions/1.pkl", "google_drive_id": "id-1"}],
    )
    with pytest.raises(ValueError, match="Unknown InterHuman clip IDs"):
        download.download_interhuman(tmp_path, clip_ids=["missing"])


def test_reject_executable_pickle(tmp_path):
    path = tmp_path / "untrusted.pkl"
    path.write_bytes(b"cos\nsystem\n(S'false'\ntR.")
    with pytest.raises(
        pickle.UnpicklingError, match="Unsupported source type"
    ):
        read_interhuman(path)


def test_legacy_annotation_keeps_original_bytes(tmp_path):
    from holoboxing.data.common import read_annotation

    path = tmp_path / "annotation.txt"
    content = "one person’s PUNCH misses".encode("gb18030")
    path.write_bytes(content)
    text, encoding = read_annotation(path)
    assert encoding == "gb18030"
    assert text == "one person’s PUNCH misses"
    assert path.read_bytes() == content


def test_same_name_drive_files_keep_independent_paths():
    from holoboxing.data.download import normalize_drive_files

    files = [
        {"path": "motions/1.pkl", "google_drive_id": "a"},
        {"path": "motions/1.pkl", "google_drive_id": "b"},
        {"path": "motions/1(1).pkl", "google_drive_id": "c"},
    ]
    records = normalize_drive_files(files)
    assert [r["path"] for r in records] == [
        "motions/1.pkl",
        "motions/1(2).pkl",
        "motions/1(1).pkl",
    ]
    assert [r["google_drive_id"] for r in records] == ["a", "b", "c"]
    assert normalize_drive_files(records) == records


def test_download_detects_corruption_and_repairs_only_on_request(tmp_path):
    import hashlib

    from holoboxing.data.download import _fetch

    content = b"verified public motion bytes"
    record = {
        "path": "motions/1.pkl",
        "google_drive_id": "public-id",
        "sha256": hashlib.sha256(content).hexdigest(),
        "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
        "bytes": len(content),
    }

    def fetch(_record, path):
        path.write_bytes(content)

    result = _fetch(record, tmp_path, fetch=fetch)
    assert result["sha256"] == record["sha256"]
    path = tmp_path / record["path"]
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Cached source changed"):
        _fetch(record, tmp_path, fetch=fetch)
    _fetch(record, tmp_path, fetch=fetch, repair=True)
    assert path.read_bytes() == content


def test_one_scale_fixed_pair_origin():
    pelvis = np.array(
        [
            [[3.0, 4.0, 1.0], [5.0, 4.0, 1.0]],
            [[4.0, 5.0, 1.0], [7.0, 6.0, 1.0]],
        ]
    )
    xy, report = shared_pair_positions(pelvis, [1.6, 1.8], 1.3)
    scale = 1.3 / 1.7
    np.testing.assert_allclose(xy[0].mean(0), 0)
    np.testing.assert_allclose(xy[1].mean(0), scale * np.array([1.5, 1.5]))
    np.testing.assert_allclose(
        xy[:, 1] - xy[:, 0], scale * (pelvis[:, 1, :2] - pelvis[:, 0, :2])
    )
    assert report["scale"] == pytest.approx(scale)


def test_original_and_mirror_do_not_leak_and_official_test_is_excluded():
    records = [
        {"group": "a"},
        {"group": "a"},
        {"group": "b"},
        {"group": "human1", "source_split": "train"},
        {"group": "human2", "source_split": "validation"},
        {"group": "human3", "source_split": "test"},
    ]
    splits, used = assign_splits(records)
    assert splits["human1"] == "train"
    assert splits["human2"] == "validation"
    assert records[-1] not in used
    assert splits == assign_splits(records)[0]


def test_data_iterations_do_not_change_global_default():
    from holoboxing.data.interhuman import retarget_config
    from holoretarget.config import DEFAULT_CONFIG

    before = DEFAULT_CONFIG.newton_iterations
    assert retarget_config().newton_iterations == 4
    assert DEFAULT_CONFIG.newton_iterations == before


def test_robot_export_runs_both_real_training_readers(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")
    from holoboxing.data.contract import CompetitionAssets
    from holoboxing.data.prepare import prepare_dataset
    from holoboxing.data.records import (
        initial_qpos,
        load_scene_model,
        scene_layout,
    )

    monkeypatch.setenv("LOGNAME", "holomotion-test")
    assets = Path(
        os.environ.get(
            "HOLOBOXING_TEST_ASSETS",
            Path(__file__).resolve().parents[1] / "holoboxing" / "assets",
        )
    )
    if not (assets / "resources.json").is_file():
        pytest.skip("Set HOLOBOXING_TEST_ASSETS to the competition assets")
    tree = CompetitionAssets.load(assets)
    model = load_scene_model(tree)
    names, roots, joints, _ = scene_layout(model)
    reset = initial_qpos(tree, model)
    source = tmp_path / "paired"
    source.mkdir()
    records = []
    for stem, split in (
        ("interhuman-a", "train"),
        ("interhuman-b", "validation"),
    ):
        path = source / (stem + ".npz")
        np.savez_compressed(
            path,
            schema_version=np.asarray("boxing_g1_pair_v1"),
            dof_names=np.asarray(names),
            root_pos=np.tile(
                np.stack([reset[root[0] : root[0] + 3] for root in roots]),
                (8, 1, 1),
            ),
            root_quat_wxyz=np.tile(
                np.stack([reset[root[0] + 3 : root[0] + 7] for root in roots]),
                (8, 1, 1),
            ),
            q=np.tile(
                np.stack([reset[joint[0]] for joint in joints]), (8, 1, 1)
            ),
            timestamps=np.arange(8) / 50,
        )
        records.append(
            {"path": path.name, "sha256": sha256(path), "source_split": split}
        )
    (source / "motions.json").write_text(
        json.dumps(
            {
                "format": "holoboxing_paired_motions_v1",
                "dataset": "interhuman",
                "records": records,
                "asset_identity": tree.identity(),
            }
        )
    )
    report = prepare_dataset(
        [source], tmp_path / "processed", assets=assets, workers=1
    )
    assert report["status"] == "PASS"
    assert report["robot_h5_clips"] == 4
    assert report["robot_h5_frames"] == 32
    assert report["supervised_samples"] == 12
    from holoboxing.training.supervised.data import Records

    supervision = Records(
        tmp_path / "processed" / "prepared" / "manifest.json"
    )
    assert supervision.sample_count() == 6
    assert sum(len(batch["self"]) for batch in supervision.batches(4)) == 6
    validation = Records(
        tmp_path / "processed" / "prepared" / "manifest.json",
        split="validation",
    )
    assert validation.sample_count() == 6
    manifest = json.loads(
        (tmp_path / "processed" / "robot_h5" / "manifest.json").read_text()
    )
    assert all(
        clip["metadata"]["motion_fps"] == 50
        for clip in manifest["clips"].values()
    )
    for partition in ("train", "validation"):
        split_manifest = json.loads(
            (
                tmp_path
                / "processed"
                / "robot_h5"
                / partition
                / "manifest.json"
            ).read_text()
        )
        assert {
            clip["metadata"]["split"]
            for clip in split_manifest["clips"].values()
        } == {partition}
    # The split-specific root, rather than only the aggregate, must be safe
    # to hand to a trainer that enumerates every clip in its manifest.
    from holoboxing.data.prepare import verify_dataset

    robot_root = tmp_path / "processed" / "robot_h5"
    train_manifest = json.loads(
        (robot_root / "train" / "manifest.json").read_text()
    )
    validation_path = robot_root / "validation" / "manifest.json"
    validation_manifest = json.loads(validation_path.read_text())
    validation_manifest["clips"].update(train_manifest["clips"])
    validation_path.write_text(json.dumps(validation_manifest))
    with pytest.raises(ValueError, match="validation H5 manifest differs"):
        verify_dataset(tmp_path / "processed", assets=assets)
