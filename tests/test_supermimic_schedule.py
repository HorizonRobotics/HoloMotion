"""Update budgets must survive uneven clocks, warmup and changed hardware."""

from dataclasses import replace
from fractions import Fraction

import pytest

from holomotion.src.algo.supermimic.schedule import (
    ScheduleConfig,
    UpdateSchedule,
)


def config(**changes):
    return replace(
        ScheduleConfig(
            physical_envs=4096,
            actor_batch_size=4096,
            imagination_horizon=32,
            world_batch_size=2048,
            world_window_length=8,
            world_batches_per_update=10,
            warmup_vector_steps=0,
        ),
        **changes,
    )


def finish(clock):
    world, actor = clock.advance()
    for _ in range(world):
        clock.complete_world()
    for _ in range(actor):
        clock.complete_actor()
    return world, actor


def test_fractional_clocks_resume_without_drift_or_lost_pending_events():
    clock = UpdateSchedule(
        config(
            actor_updates_per_vector_step="1/3",
            world_updates_per_vector_step=".4",
        )
    )
    for _ in range(1003):
        finish(clock)
    assert clock.world_updates == 401 and clock.world_credit == Fraction(1, 5)
    assert clock.actor_updates == 334 and clock.actor_credit == Fraction(1, 3)
    saved = clock.state_dict()
    restored = UpdateSchedule(clock.config)
    restored.load_state_dict(saved)
    for _ in range(37):
        assert finish(clock) == finish(restored)
    assert clock.state_dict() == restored.state_dict()
    pending = UpdateSchedule(
        config(
            actor_updates_per_vector_step="2",
            world_updates_per_vector_step="3/2",
        )
    )
    assert pending.advance() == (1, 2)
    pending.complete_world()
    pending.complete_actor()
    with pytest.raises(RuntimeError, match="pending"):
        pending.advance()
    with pytest.raises(RuntimeError, match="pending"):
        pending.state_dict()
    pending.complete_actor()
    assert pending.state_dict()["actor_updates"] == 2
    with pytest.raises(ValueError, match="credits"):
        restored.load_state_dict({**saved, "world_updates": 402})


def test_reuse_ratios_adjust_rates_when_envs_or_logical_batches_change():
    baseline = UpdateSchedule(config())
    assert Fraction(
        baseline.ratios()["steady_imagined_steps_per_real_transition"]
    ) == Fraction("12.8")
    assert (
        Fraction(baseline.ratios()["steady_fitted_steps_per_real_transition"])
        == 16
    )
    smaller = UpdateSchedule(
        ScheduleConfig.from_reuse_ratios(
            imagined_steps_per_real_transition="12.8",
            fitted_steps_per_real_transition="16",
            physical_envs=2048,
            actor_batch_size=2048,
            imagination_horizon=32,
            world_batch_size=2048,
            world_window_length=8,
            world_batches_per_update=10,
        )
    )
    assert smaller.actor_rate == Fraction("0.4")
    assert smaller.world_rate == Fraction("0.2")
    for name in (
        "steady_imagined_steps_per_real_transition",
        "steady_fitted_steps_per_real_transition",
    ):
        assert baseline.ratios()[name] == smaller.ratios()[name]
    changed = UpdateSchedule(config(physical_envs=2048))
    assert Fraction(
        changed.ratios()["steady_imagined_steps_per_real_transition"]
    ) == Fraction("25.6")
    with pytest.raises(ValueError, match="differs"):
        smaller.load_state_dict(baseline.state_dict())


def test_world_only_warmup_does_not_earn_later_actor_updates():
    clock = UpdateSchedule(
        config(
            warmup_vector_steps=3,
            world_only_updates=3,
            world_updates_per_vector_step="1.5",
            actor_updates_per_vector_step=".4",
        )
    )
    events = []
    for step in range(1, 12):
        world, actor = finish(clock)
        events.append((world, actor))
        if step == 5:
            resumed = UpdateSchedule(clock.config)
            resumed.load_state_dict(clock.state_dict())
        elif step > 5:
            assert finish(resumed) == (world, actor)
    assert events[:3] == [(0, 0)] * 3
    assert next(i for i, (_, actor) in enumerate(events, 1) if actor) == 8
    assert (clock.world_updates, clock.actor_updates) == (12, 2)
    assert resumed.state_dict() == clock.state_dict()
