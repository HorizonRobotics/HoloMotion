"""Selection never sees test poses, and validation stopping survives resume."""

import json

import numpy as np
import pytest
import torch
from test_boxing_motion_prior import reference_population

from holoboxing.data.common import sha256
from holoboxing.data.prepare import assign_splits
from holoboxing.training.evaluate_prior import evaluate
from holoboxing.training.prior_data import PriorRecords
from holoboxing.training.select_prior import select
from holoboxing.training.train_prior import train
from holoboxing.training.validation_stop import ValidationStop


def three_way_population(root):
    manifest = reference_population(root)
    data = json.loads(manifest.read_text())
    for index, dataset in enumerate(("robostriker", "interhuman"), 4):
        source = root / f"{index - 4}.npz"
        target = root / f"{index}.npz"
        with np.load(source) as packet:
            values = packet["target_pose"].copy()
        values.reshape(-1, 25, 9)[..., :3] += 20
        np.savez_compressed(target, target_pose=values)
        data["records"].append(
            dict(
                data["records"][index - 4],
                path=target.name,
                bytes=target.stat().st_size,
                sha256=sha256(target),
                group=str(index),
                split="test",
                dataset=dataset,
            )
        )
    manifest.write_text(json.dumps(data))
    return manifest


def test_robostriker_three_way_group_split_keeps_mirrors_together():
    records = [
        dict(group=f"motion-{i}", path=f"motion-{i}{suffix}.npz")
        for i in range(43)
        for suffix in ("", "_mirror")
    ]
    records.extend(
        dict(group=f"human-{split}", source_split=split)
        for split in ("train", "validation", "test")
    )
    splits, used = assign_splits(records, test_fraction=0.1)
    assert {
        split: sum(splits[f"motion-{i}"] == split for i in range(43))
        for split in ("train", "validation", "test")
    } == dict(train=33, validation=5, test=5)
    assert len([r for r in used if r["group"].startswith("motion")]) == 76
    assert all(splits[r["group"]] != "test" for r in used)
    assert all(
        splits[f"human-{split}"] == split
        for split in ("train", "validation", "test")
    )
    assert (
        assign_splits(list(reversed(records)), test_fraction=0.1)[0] == splits
    )


def test_early_stopping_accumulates_small_gains_and_restores_patience():
    stop = ValidationStop(patience=2, min_relative_improvement=0.01)
    assert not stop.observe(1.0)
    assert not stop.observe(0.995)
    assert not stop.observe(0.989)
    assert stop.bad_checks == 0  # The cumulative improvement exceeds 1%.
    assert not stop.observe(0.988)
    resumed = ValidationStop(patience=2, min_relative_improvement=0.01)
    resumed.load_state_dict(stop.state_dict())
    assert resumed.observe(0.987)
    assert resumed.stopped
    with pytest.raises(ValueError, match="recipe differs"):
        ValidationStop(patience=3).load_state_dict(stop.state_dict())
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="finite"):
            resumed.observe(value)


def test_resume_of_early_stopped_prior_does_not_update_again(tmp_path):
    torch.set_num_threads(1)
    manifest = three_way_population(tmp_path / "data")
    train_data = PriorRecords(manifest)
    validation = PriorRecords(manifest, split="validation")
    recipe = dict(
        epochs=20,
        batch_size=8,
        hidden_dims=(8,),
        patience=1,
        min_relative_improvement=0.99,
    )
    stopped = train(train_data, validation, tmp_path / "run", **recipe)
    before = torch.load(stopped, weights_only=False)
    assert before["cursor"]["epoch"] == 2
    resumed = train(
        train_data, validation, tmp_path / "resume", resume=stopped, **recipe
    )
    after = torch.load(resumed, weights_only=False)
    assert before["steps"] == after["steps"]
    assert all(
        torch.equal(value, after["model"][name])
        for name, value in before["model"].items()
    )
    receipt = json.loads((resumed.parent / "training-result.json").read_text())
    assert receipt["progress"]["stop_reason"] == "validation_plateau"


def test_validation_selection_precedes_separate_test_evaluation(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    manifest = three_way_population(tmp_path / "data")
    records = PriorRecords(manifest)
    validation = PriorRecords(manifest, split="validation")
    for i, widths in enumerate(((8,), (16,))):
        train(
            records,
            validation,
            tmp_path / str(i),
            epochs=2,
            batch_size=8,
            hidden_dims=widths,
        )
    candidates = [tmp_path / str(i) / "best.pt" for i in range(2)]
    selection_path = tmp_path / "selection.json"
    # Candidate choice must not open any NPZ, including the test poses.
    with monkeypatch.context() as patch:
        patch.setattr(
            np, "load", lambda *a, **kw: pytest.fail("Selection read data")
        )
        selection = select(candidates, selection_path)
    expected = min(
        selection["candidates"],
        key=lambda row: (
            row["validation"]["reconstruction"],
            row["checkpoint_sha256"],
        ),
    )
    assert (
        selection["selected_checkpoint_sha256"]
        == expected["checkpoint_sha256"]
    )
    assert selection["test_evaluated"] is False
    report = evaluate(
        expected["checkpoint"],
        manifest,
        selection_path,
        tmp_path / "test.json",
        batch_size=3,
    )
    assert report["population"]["split"] == "test"
    assert report["population"]["samples"] == 16
    assert report["by_dataset"]["interhuman"]["samples"] == 8
    assert report["by_dataset"]["robostriker"]["samples"] == 8
    assert report["closed_loop_verified"] is False
    unselected = next(
        p for p in candidates if sha256(p) != expected["checkpoint_sha256"]
    )
    with pytest.raises(ValueError, match="fixed validation selection"):
        evaluate(unselected, manifest, selection_path, tmp_path / "wrong.json")
    with pytest.raises(FileExistsError):
        evaluate(
            expected["checkpoint"],
            manifest,
            selection_path,
            tmp_path / "test.json",
        )


def test_selection_rejects_last_weights_with_an_older_best_score(tmp_path):
    torch.set_num_threads(1)
    manifest = three_way_population(tmp_path / "data")
    path = train(
        PriorRecords(manifest),
        PriorRecords(manifest, split="validation"),
        tmp_path / "run",
        epochs=1,
        batch_size=8,
        hidden_dims=(8,),
    )
    saved = torch.load(path, weights_only=False)
    saved["cursor"]["epoch"] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="best completed validation weights"):
        select([path], tmp_path / "selection.json")
