import json
from types import SimpleNamespace

import pytest
import torch

from holomotion.scripts.training.train_supermimic import build_parser
from holomotion.src.algo.supermimic.config import SuperMimicConfig
from holomotion.src.algo.supermimic.policy_progress import (
    PolicyProgressSnapshots,
)


class Clock:
    now = 7.0

    def __call__(self):
        return self.now


def recorder(tmp_path, interval=3, budget=1800):
    clock = Clock()
    policy = torch.nn.Linear(2, 1)
    schedule = SimpleNamespace(
        state_dict=lambda: dict(
            vector_steps=1, world_updates=0, actor_updates=0
        )
    )
    return (
        PolicyProgressSnapshots(
            tmp_path, policy, {}, schedule, clock, interval, budget
        ),
        clock,
        policy,
    )


def test_eleven_real_snapshots_with_immutable_policy_and_no_rng_draw(tmp_path):
    r, c, p = recorder(tmp_path)
    rng = torch.random.get_rng_state().clone()
    r.initial()
    c.now = 179.999
    r.due()
    assert len(r.records) == 1
    for target in range(180, 1800, 180):
        c.now = float(target) + 0.25
        with torch.no_grad():
            p.weight.add_(1)
        r.due()
    c.now = 1800.2
    r.due()
    r.final()
    assert [x["target_seconds"] for x in r.records] == list(
        range(0, 1801, 180)
    )
    assert len(r.records) == 11
    assert torch.equal(rng, torch.random.get_rng_state())
    a = torch.load(tmp_path / r.records[0]["path"], weights_only=False)
    b = torch.load(tmp_path / r.records[-1]["path"], weights_only=False)
    assert not torch.equal(a["pi"]["weight"], b["pi"]["weight"])
    assert a["purpose"] == "policy_progress_only" and not a["resume_supported"]
    assert not ({"replay", "opt_pi", "opt_wm", "wm", "critic"} & a.keys())
    assert json.loads((tmp_path / "policy_progress/index.json").read_text())[
        "finished"
    ]


def test_late_boundary_keeps_actual_capture_time_not_target_time(tmp_path):
    r, c, _ = recorder(tmp_path)
    r.initial()
    c.now = 400
    r.due()
    assert [x["target_seconds"] for x in r.records] == [0, 180, 360]
    assert [x["training_wall_s"] for x in r.records] == [7, 400, 400]
    c.now = 500
    r.final()
    assert r.records[-1]["target_seconds"] == 1800
    assert len(r.records) == 4  # final never fills unavailable future stages


def test_non_divisible_budget_and_duplicate_final(tmp_path):
    r, c, _ = recorder(tmp_path, interval=3, budget=500)
    r.initial()
    c.now = 500
    r.due()
    r.final()
    assert [x["target_seconds"] for x in r.records] == [0, 180, 360, 500]
    with pytest.raises(RuntimeError):
        r.final()


def test_config_defaultoff_validation_and_cli(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("robot")
    base = dict(manifest=manifest, robot_urdf=urdf, output=tmp_path / "output")
    assert SuperMimicConfig.from_mapping(base).policy_snapshot_minutes == 0
    assert (
        SuperMimicConfig.from_mapping(
            dict(base, policy_snapshot_minutes=3)
        ).policy_snapshot_minutes
        == 3
    )
    with pytest.raises(ValueError):
        SuperMimicConfig.from_mapping(dict(base, policy_snapshot_minutes=-1))
    with pytest.raises(ValueError):
        SuperMimicConfig.from_mapping(
            dict(base, policy_snapshot_minutes=3, resume="resume.pt")
        )
    p = build_parser().parse_args(
        [
            "--manifest",
            "a",
            "--robot-urdf",
            "b",
            "--output",
            "c",
            "--policy-snapshot-minutes",
            "3",
        ]
    )
    assert p.policy_snapshot_minutes == 3


def test_save_cost_uses_training_clock(tmp_path, monkeypatch):
    r, c, _ = recorder(tmp_path)
    original = torch.save

    def save(bundle, path):
        original(bundle, path)
        c.now += 0.25

    monkeypatch.setattr(torch, "save", save)
    r.initial()
    assert r.records[0]["training_wall_s"] == 7
    assert r.records[0]["save_seconds"] == 0.25
    assert c.now == 7.25
