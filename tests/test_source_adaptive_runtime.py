"""CPU contracts for S1 PPO-barrier accumulation and one-gather validation."""

from types import SimpleNamespace

import pytest
import torch

from holomotion.src.training.source_adaptive_runtime import (
    WindowErrorAccumulator,
    pack_sync_payload,
    record_command_mpkpe,
    reduce_sync_payload,
    synchronize_command_sampler,
)

_IDENTITY = "ab" * 32


class _Controller:
    def __init__(self):
        self.identity = _IDENTITY
        self.round = 2
        self.version = 1
        self.observations = []

    def observe(self, indices, sums, counts):
        self.observations.append(
            (indices.clone(), sums.clone(), counts.clone())
        )
        self.round += 1
        self.version += 1


class _Accelerator:
    num_processes = 2

    def __init__(self, other_packet):
        self.other_packet = other_packet
        self.calls = 0

    def gather(self, packet):
        self.calls += 1
        return torch.cat((packet, self.other_packet))


def _command(*, pending=True):
    batch = SimpleNamespace(window_indices=torch.tensor([10, 20]))
    cache = SimpleNamespace(
        adaptive_sampling_enabled=True,
        adaptive_controller=_Controller(),
        current_batch=batch,
        clip_count=2,
        _batch_size=4,
        mode="train",
        swap_index=3,
    )
    command = SimpleNamespace(
        _motion_cache=cache,
        _swap_pending=pending,
        _is_evaluating=False,
        device=torch.device("cpu"),
        num_envs=3,
        _clip_indices=torch.tensor([0, 1, 0]),
        _env=SimpleNamespace(episode_length_buf=torch.tensor([1, 0, 2])),
        _filter_env_ids_for_motion_task=lambda ids: ids,
    )
    return command


def _other_packet(
    *, indices=(20, 30), sums=(2.0, 3.0), counts=(1.0, 1.0), pending=True
):
    return pack_sync_payload(
        identity=_IDENTITY,
        enabled=True,
        swap_pending=pending,
        training=True,
        clip_count=2,
        capacity=4,
        round=2,
        version=1,
        swap_index=3,
        window_indices=torch.tensor(indices),
        sums=torch.tensor(sums),
        counts=torch.tensor(counts),
        device="cpu",
    )


def test_window_error_accumulator_skips_invalid_samples_and_rows():
    accumulator = WindowErrorAccumulator(torch.tensor([10, 20]), "cpu")
    accumulator.record(
        torch.tensor([0, 1, 1, -1, 5]),
        torch.tensor([1.0, 0.5, float("nan"), 2.0, 3.0]),
    )
    assert accumulator.sums.tolist() == pytest.approx([1.0, 0.5])
    assert accumulator.counts.tolist() == pytest.approx([1.0, 1.0])
    assert accumulator.invalid.item() == 3.0
    with pytest.raises(ValueError, match="integer cache-row"):
        accumulator.record(torch.tensor([0.1]), torch.tensor([1.0]))


def test_command_recording_excludes_terminal_reset_and_restarts_on_new_batch():
    command = _command()
    record_command_mpkpe(command, torch.tensor([1.0, 2.0, 3.0]))
    stats = command._adaptive_window_metrics
    assert stats.sums.tolist() == pytest.approx([4.0, 0.0])
    assert stats.counts.tolist() == pytest.approx([2.0, 0.0])
    old_marker = command._adaptive_metrics_batch

    command._motion_cache.current_batch = SimpleNamespace(
        window_indices=torch.tensor([30, 40])
    )
    command._motion_cache.swap_index += 1
    record_command_mpkpe(command, torch.tensor([0.5, 1.0, 0.5]))
    assert command._adaptive_metrics_batch != old_marker
    assert command._adaptive_window_metrics.sums.tolist() == pytest.approx(
        [1.0, 0.0]
    )

    command._is_evaluating = True
    prior = command._adaptive_window_metrics.sums.clone()
    record_command_mpkpe(command, torch.ones(3))
    assert torch.equal(command._adaptive_window_metrics.sums, prior)


def test_packet_bitcasts_float64_and_uses_fixed_capacity():
    packet = _other_packet(indices=(2, 3), sums=(0.1, 1.1), counts=(1.0, 2.0))
    assert packet.dtype == torch.int64
    assert packet.numel() == 11 + 32 + 3 * 4
    reduced = reduce_sync_payload(
        torch.cat((packet, packet)), world=2, capacity=4
    )
    assert reduced is not None
    indices, sums, counts = reduced
    assert indices.tolist() == [2, 3, 2, 3]
    assert sums.tolist() == pytest.approx([0.1, 1.1, 0.1, 1.1])
    assert counts.tolist() == pytest.approx([1.0, 2.0, 1.0, 2.0])


def test_one_collective_gathers_rank_major_cache_rows_before_swap():
    command = _command()
    record_command_mpkpe(command, torch.tensor([1.0, 9.0, 3.0]))
    accelerator = _Accelerator(_other_packet())
    synchronize_command_sampler(command, accelerator)
    assert accelerator.calls == 1
    assert len(command._motion_cache.adaptive_controller.observations) == 1
    indices, sums, counts = (
        command._motion_cache.adaptive_controller.observations[0]
    )
    assert indices.tolist() == [10, 20, 20, 30]
    assert sums.tolist() == pytest.approx([4.0, 0.0, 2.0, 3.0])
    assert counts.tolist() == pytest.approx([2.0, 0.0, 1.0, 1.0])
    assert command._adaptive_window_metrics is None
    assert command._adaptive_metrics_batch is None


def test_no_pending_swap_still_participates_but_keeps_accumulation():
    command = _command(pending=False)
    record_command_mpkpe(command, torch.tensor([1.0, 9.0, 3.0]))
    accelerator = _Accelerator(_other_packet(pending=False))
    synchronize_command_sampler(command, accelerator)
    assert accelerator.calls == 1
    assert command._motion_cache.adaptive_controller.observations == []
    assert command._adaptive_window_metrics.counts.sum().item() == 2.0


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (2, "swap_pending"),
        (4, "clip_count"),
        (6, "round"),
        (11, "identity"),
    ],
)
def test_rank_disagreement_fails_before_controller_mutation(offset, expected):
    command = _command()
    other = _other_packet()
    other[offset] += 1
    accelerator = _Accelerator(other)
    with pytest.raises(RuntimeError, match=expected):
        synchronize_command_sampler(command, accelerator)
    assert accelerator.calls == 1
    assert command._motion_cache.adaptive_controller.observations == []


def test_invalid_metric_on_one_rank_fails_all_ranks_after_one_gather():
    command = _command()
    other = _other_packet()
    other[9] = 1
    accelerator = _Accelerator(other)
    with pytest.raises(FloatingPointError, match="Invalid MPKPE"):
        synchronize_command_sampler(command, accelerator)
    assert accelerator.calls == 1
    assert command._motion_cache.adaptive_controller.observations == []


def test_malformed_rank_payload_fails_before_update():
    command = _command()
    other = _other_packet()
    other[10] = 1
    with pytest.raises(RuntimeError, match="Malformed"):
        synchronize_command_sampler(command, _Accelerator(other))
    assert command._motion_cache.adaptive_controller.observations == []


def test_zero_exposure_rank_still_contributes_cache_indices():
    command = _command()
    accelerator = _Accelerator(
        _other_packet(sums=(0.0, 0.0), counts=(0.0, 0.0))
    )
    synchronize_command_sampler(command, accelerator)
    assert accelerator.calls == 1
    indices, sums, counts = (
        command._motion_cache.adaptive_controller.observations[0]
    )
    assert indices.tolist() == [10, 20, 20, 30]
    assert sums.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert counts.tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, ["sync", "swap"]), (False, ["swap"])],
)
def test_ppo_hook_syncs_before_swap_only_when_s1_enabled(
    monkeypatch, enabled, expected
):
    from holomotion.src.algo import ppo as ppo_module

    events = []
    command = SimpleNamespace(
        apply_cache_swap_if_pending_barrier=lambda **kwargs: events.append(
            "swap"
        )
    )
    algo = ppo_module.PPO.__new__(ppo_module.PPO)
    algo.command_name = "ref_motion"
    algo.source_adaptive_enabled = enabled
    algo.accelerator = object()
    algo.env = SimpleNamespace(
        _env=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command)
        )
    )
    monkeypatch.setattr(
        ppo_module,
        "synchronize_command_sampler",
        lambda cmd, accelerator: events.append("sync"),
    )
    algo._post_iteration_hook(0)
    assert events == expected


def test_ppo_logs_adaptive_metrics_without_replay():
    from holomotion.src.algo.ppo import PPO

    cache = SimpleNamespace(
        swap_index=3,
        source_sampling_metrics=lambda: None,
        adaptive_sampling_metrics=lambda: {"weight_version": 2.0},
        adaptive_sampling_enabled=True,
        cache_curriculum_pool_statistics=lambda: None,
    )
    command = SimpleNamespace(_motion_cache=cache)
    algo = PPO.__new__(PPO)
    algo.command_name = "ref_motion"
    algo.env = SimpleNamespace(
        _env=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command)
        )
    )
    algo._get_mean_policy_std = lambda: torch.tensor(0.2)
    metrics = algo._get_additional_log_metrics()
    assert metrics["1-Perf/Cache/Adaptive/weight_version"] == 2.0
    assert metrics["1-Perf/Cache/Adaptive/source_adaptive_enabled"] == 1.0
    assert not any(key.startswith("1-Perf/FailureReplay/") for key in metrics)


def test_ppo_logs_failure_replay_metrics_without_adaptation():
    from holomotion.src.algo.ppo import PPO

    cache = SimpleNamespace(
        swap_index=3,
        source_sampling_metrics=lambda: None,
        cache_curriculum_pool_statistics=lambda: None,
    )
    command = SimpleNamespace(
        _motion_cache=cache,
        failure_replay_metrics=lambda: {"active_fraction": 0.4},
    )
    algo = PPO.__new__(PPO)
    algo.command_name = "ref_motion"
    algo.env = SimpleNamespace(
        _env=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command)
        )
    )
    algo._get_mean_policy_std = lambda: torch.tensor(0.2)
    metrics = algo._get_additional_log_metrics()
    assert metrics["1-Perf/FailureReplay/active_fraction"] == 0.4


def test_ppo_swap_refreshes_observation_before_next_action():
    from holomotion.src.algo.ppo import PPO

    events = []
    expected_obs = {"policy": torch.tensor([1.0])}

    def reset(env_ids):
        events.append(("reset", env_ids.tolist()))

    def compute(*, update_history):
        events.append(("compute", update_history))
        return expected_obs

    command = SimpleNamespace(
        _link_observation_adapter=object(),
        _filter_env_ids_for_motion_task=lambda ids: ids[::2],
        apply_cache_swap_if_pending_barrier=lambda **kwargs: (
            events.append("swap") or True
        ),
    )
    isaac_env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene=SimpleNamespace(
            write_data_to_sim=lambda: events.append("write")
        ),
        sim=SimpleNamespace(forward=lambda: events.append("forward")),
        observation_manager=SimpleNamespace(reset=reset, compute=compute),
    )
    algo = PPO.__new__(PPO)
    algo.command_name = "ref_motion"
    algo.source_adaptive_enabled = False
    algo.accelerator = object()
    algo.device = torch.device("cpu")
    algo.env = SimpleNamespace(_env=isaac_env, num_envs=3)
    algo._wrap_obs_dict = lambda obs: events.append("wrap") or obs

    fresh_obs = algo._post_iteration_hook(0)
    assert fresh_obs is expected_obs
    assert events == [
        "swap",
        "write",
        "forward",
        ("reset", [0, 2]),
        ("compute", True),
        "wrap",
    ]
