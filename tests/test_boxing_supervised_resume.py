"""Interrupted minibatches resume without replaying already optimized rows."""

from dataclasses import replace

import pytest
import torch
from test_boxing_training_splits import population

from holoboxing.training.supervised.data import Records
from holoboxing.training.supervised.statistics import fit_statistics
from holoboxing.training.supervised.training import train
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig
from holoboxing.training.time_budget import TrainingBudget


class TwoUpdates(TrainingBudget):
    def __init__(self):
        self.now = 0.0
        super().__init__(
            clock=lambda: self.now,
            max_runtime_seconds=100,
            max_training_seconds=2.5,
            save_reserve_seconds=1,
        )

    def update_finished(self, seconds):
        self.now += 1
        super().update_finished(1)


def test_mid_epoch_resume_matches_uninterrupted_optimizer_and_weights(
    tmp_path,
):
    torch.set_num_threads(1)
    manifest = population(tmp_path)
    records = Records(manifest)
    statistics, normalizer = fit_statistics(records)
    config = replace(TrainConfig(), hidden_dims=(16,))
    neutral = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1.0]).repeat(25)
    kwargs = dict(
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        seed=42,
    )
    complete = train(
        records,
        config,
        neutral,
        statistics,
        normalizer,
        tmp_path / "whole",
        **kwargs,
    )
    stopped = train(
        records,
        config,
        neutral,
        statistics,
        normalizer,
        tmp_path / "partial",
        time_budget=TwoUpdates(),
        **kwargs,
    )
    saved = torch.load(stopped, weights_only=False)
    assert saved["step"] == saved["resume"]["cursor"]["batch"] == 2
    assert saved["epoch"] == 0
    assert not saved["coverage"]["whole_epoch_complete"]
    resumed = train(
        records,
        config,
        neutral,
        statistics,
        normalizer,
        tmp_path / "resumed",
        resume=stopped,
        **kwargs,
    )
    expected, actual = (
        torch.load(p, weights_only=False) for p in (complete, resumed)
    )
    assert expected["step"] == actual["step"] == 12
    assert expected["resume"]["cursor"] == actual["resume"]["cursor"]
    assert all(
        torch.equal(v, actual["actor"][k])
        for k, v in expected["actor"].items()
    )
    for key, state in expected["resume"]["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(
                value, actual["resume"]["optimizer"]["state"][key][name]
            )
    assert torch.equal(
        expected["resume"]["rng"]["torch"], actual["resume"]["rng"]["torch"]
    )
    assert (tmp_path / "whole/metrics.jsonl").read_bytes() == (
        tmp_path / "resumed/metrics.jsonl"
    ).read_bytes()
    with pytest.raises(
        ValueError, match="same dataset, device and optimizer recipe"
    ):
        train(
            records,
            config,
            neutral,
            statistics,
            normalizer,
            tmp_path / "wrong",
            resume=stopped,
            **{**kwargs, "batch_size": 3},
        )


def test_resume_skips_finished_shards_without_loading_their_arrays(
    tmp_path, monkeypatch
):
    records = Records(population(tmp_path))
    expected = list(records.batches(2, seed=73))
    records.cached_index, records.cached = None, None
    original = records._read
    loaded = []

    def read(row):
        loaded.append(row["path"])
        return original(row)

    monkeypatch.setattr(records, "_read", read)
    actual = list(records.batches(2, seed=73, start_batch=3))
    assert len(loaded) == 1
    assert len(actual) == 3
    for a, b in zip(actual, expected[3:], strict=True):
        assert all(torch.equal(value, b[key]) for key, value in a.items())
