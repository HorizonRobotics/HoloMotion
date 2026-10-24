"""Learning-state recovery preserves labels and cuts physical discontinuities."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from holomotion.src.algo.supermimic.buffer import ReplayBuffer
from holomotion.src.algo.supermimic.checkpoint import (
    FORMAT,
    capture_learning_state,
    restore_learning_state,
    validate_checkpoint,
)
from holomotion.src.algo.supermimic.schedule import (
    ScheduleConfig,
    UpdateSchedule,
)


def fill(buffer, count):
    for i in range(count):
        transition = {
            key: torch.full_like(value[0], i)
            for key, value in buffer.data.items()
        }
        transition["done"].zero_()
        buffer.add(transition)


@pytest.mark.parametrize("count", [3, 6, 9])
def test_replay_resume_retains_storage_cursor_and_labels(count):
    original = ReplayBuffer(6, 2, "cpu")
    fill(original, count)
    saved = original.state_dict()
    restored = ReplayBuffer(6, 2, "cpu")
    restored.load_state_dict(saved)
    assert (restored.size, restored.ptr) == (original.size, original.ptr)
    for key in original.data:
        assert torch.equal(
            restored.data[key][: original.size],
            original.data[key][: original.size],
        )
    restored.mark_episode_boundary()
    assert not restored.data["done"].any()  # Physical labels remain original.
    assert restored.break_after[(original.ptr - 1) % 6].all()
    # CPU snapshots own their data even when replay itself lives on CPU.
    restored.data["s"][0].fill_(-1)
    assert not (saved["data"]["s"][0] == -1).any()


def test_world_windows_cannot_cross_restart_and_new_history_stays_local():
    replay = ReplayBuffer(8, 1, "cpu")
    fill(replay, 4)
    replay.mark_episode_boundary()
    before = replay.data["done"].clone()
    fill(replay, 3)
    # The newest episode starts with ep_step0; this is the collector contract.
    replay.data["ep_step"][4:7, 0] = torch.arange(3)
    replay.data["f"][4:7, 0] = 100 + torch.arange(3)
    windows = replay.sample_windows(
        512, 4, generator=torch.Generator().manual_seed(4)
    )
    for starts, mask in zip(
        windows["s"][:, 0, 0].long(), windows["mask"], strict=False
    ):
        expected = torch.arange(4) <= 3 - int(starts)
        assert torch.equal(mask.bool(), expected)
    assert torch.equal(replay.data["done"], before)
    for _ in range(20):
        start, history = replay.sample_starts(16, 5)
        selected = start["f"] >= 100
        if selected.any():
            assert (history["f"][selected] >= 100).all()
            assert (history["f"][selected] <= start["f"][selected, None]).all()


def test_ring_wrap_overwrites_only_its_boundary():
    replay = ReplayBuffer(4, 1, "cpu")
    fill(replay, 4)
    replay.mark_episode_boundary()
    fill(replay, 3)
    assert replay.break_after[3].all()
    fill(replay, 1)
    assert not replay.break_after.any()


def spec():
    return dict(
        arguments=dict(
            output="one",
            device="cuda:0",
            max_vector_steps=12,
            hist=10,
        ),
        source_sha256={"code": "a"},
        input_sha256={"data": "b"},
        generated_model_sha256="c",
        schedule={"rate": ".4"},
        parameters={"actor": 2},
        actor_observation_dim=93,
        goal_dim=730,
        physics={"dt": 0.005},
        runtime=dict(
            torch="2.9",
            mujoco="3.16",
            warp="1.18",
            mujoco_warp="3.16",
            torch_cuda="12.8",
        ),
    )


def test_resume_contract_allows_budgets_and_rejects_recipe_source_data_changes():
    original = spec()
    bundle = dict(format=FORMAT, specification=original)
    current = deepcopy(original)
    current["arguments"].update(
        output="two", device="cuda:1", max_vector_steps=30
    )
    validate_checkpoint(bundle, current)
    for field in (
        "source_sha256",
        "input_sha256",
        "schedule",
        "generated_model_sha256",
    ):
        wrong = deepcopy(current)
        wrong[field] = "different"
        with pytest.raises(ValueError, match="contract"):
            validate_checkpoint(bundle, wrong)
    wrong = deepcopy(current)
    wrong["runtime"]["mujoco"] = "different"
    with pytest.raises(ValueError, match="runtime"):
        validate_checkpoint(bundle, wrong)
    current["arguments"]["hist"] = 32
    with pytest.raises(ValueError, match="recipe"):
        validate_checkpoint(bundle, current)
    with pytest.raises(ValueError, match="complete"):
        validate_checkpoint(dict(specification=original), original)


def make_learning_state():
    learner = SimpleNamespace(dev="cpu")
    for model, optimizer in [
        ("wm", "opt_wm"),
        ("pi", "opt_pi"),
        ("critic", "opt_v"),
    ]:
        module = torch.nn.Linear(2, 2)
        opt = torch.optim.Adam(module.parameters(), lr=0.02)
        module(torch.ones(1, 2)).sum().backward()
        opt.step()
        setattr(learner, model, module)
        setattr(learner, optimizer, opt)
    schedule = UpdateSchedule(
        ScheduleConfig(2, 2, 2, 2, 2, 1, warmup_vector_steps=0)
    )
    for _ in range(5):
        world, actor = schedule.advance()
        for _ in range(world):
            schedule.complete_world()
        for _ in range(actor):
            schedule.complete_actor()
    replay = ReplayBuffer(8, 2, "cpu")
    fill(replay, 5)
    env = SimpleNamespace(
        N=2, bin_G=torch.arange(3).float(), bin_w_avg=torch.ones(3) / 3
    )
    env.reset = lambda ids: setattr(env, "reset_ids", ids)
    return learner, replay, schedule, env


def test_full_state_restore_keeps_adam_clock_replay_and_rng(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: torch.tensor([7], dtype=torch.uint8),
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: None
    )
    first = make_learning_state()
    bundle = deepcopy(capture_learning_state(*first))
    expected_random = torch.rand(4)
    second = make_learning_state()
    restore_learning_state(bundle, *second)
    learner, replay, schedule, env = second
    assert torch.equal(torch.rand(4), expected_random)
    assert schedule.state_dict() == first[2].state_dict()
    assert torch.equal(env.reset_ids, torch.arange(2))
    assert replay.break_after[4].all()
    for model, optimizer in [
        ("wm", "opt_wm"),
        ("pi", "opt_pi"),
        ("critic", "opt_v"),
    ]:
        for key, value in getattr(first[0], model).state_dict().items():
            assert torch.equal(
                getattr(learner, model).state_dict()[key], value
            )
        original = getattr(first[0], optimizer).state_dict()
        restored = getattr(learner, optimizer).state_dict()
        assert restored["param_groups"] == original["param_groups"]
        for index, state in original["state"].items():
            for name, value in state.items():
                assert torch.equal(restored["state"][index][name], value)
