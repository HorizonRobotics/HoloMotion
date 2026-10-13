"""CPU checks for installing named modes around a link adapter."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from holomotion.src.motion_tracking.link_mode_mask import ModeMaskLinkAdapter

WRAPPER_FILE = (
    Path(__file__).resolve().parents[1]
    / "holomotion/src/env/mode_masked_motion_tracking.py"
)
LINKS = ("pelvis", "left_wrist", "right_wrist")
MODE_CONFIG = {
    "strategy": "modes",
    "mode_candidates": {
        "Pelvis-1": ["pelvis"],
        "LeftWrist-1": ["left_wrist"],
        "WholeBody-3": list(LINKS),
    },
    "mode_weights": {"Pelvis-1": 0, "LeftWrist-1": 1, "WholeBody-3": 0},
}


class FakeMotionTrackingEnv:
    def _init_isaaclab_env(self):
        self.base_init_calls += 1
        return self._env


class FakeLinkAdapter:
    def __init__(self, command):
        self.command = command
        self.link_names = LINKS
        self.actor_frame_offsets = (1, 2)
        self.critic_frame_offsets = (1, 2, 3)
        self.link_mask = torch.ones(command.num_envs, len(LINKS))
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1


@pytest.fixture
def env_class(monkeypatch):
    # Import the real wrapper against a tiny base class, without importing
    # motion_tracking.py or its Isaac Lab dependencies.
    fake_module = ModuleType("holomotion.src.env.motion_tracking")
    fake_module.MotionTrackingEnv = FakeMotionTrackingEnv
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    spec = importlib.util.spec_from_file_location(
        "_mode_masked_motion_tracking_runtime_test", WRAPPER_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModeMaskedMotionTrackingEnv


def build_env(
    env_class, log_dir, *, rank=0, accelerator=False, raw_omega=False
):
    env = env_class.__new__(env_class)
    env.config = {"process_id": rank}
    env.accelerator = (
        SimpleNamespace(process_index=rank) if accelerator else None
    )
    env.log_dir = log_dir
    env.base_init_calls = 0
    raw = {"mask": MODE_CONFIG}
    if raw_omega:
        raw = OmegaConf.create(raw)
    command = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(link_observation=raw),
    )
    base = FakeLinkAdapter(command)
    command._link_observation_adapter = base
    requested_terms = []

    def get_term(name):
        requested_terms.append(name)
        return command

    env._env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=get_term)
    )
    return env, command, base, requested_terms


@pytest.mark.parametrize(
    "rank,accelerator,raw_omega", [(0, False, False), (7, True, True)]
)
def test_wrapper_installs_sampler_and_writes_rank_receipt(
    env_class, tmp_path, capsys, rank, accelerator, raw_omega
):
    env, command, base, requested_terms = build_env(
        env_class,
        tmp_path,
        rank=rank,
        accelerator=accelerator,
        raw_omega=raw_omega,
    )
    if accelerator:
        env.config["process_id"] = 99  # Accelerator rank has priority.

    result = env._init_isaaclab_env()

    assert result is env._env
    assert env.base_init_calls == 1
    assert requested_terms == ["ref_motion"]
    adapter = command._link_observation_adapter
    assert isinstance(adapter, ModeMaskLinkAdapter)
    assert adapter.base is base
    assert base.invalidations == 1
    assert adapter.sampled_mode_ids.tolist() == [1, 1, 1]
    torch.testing.assert_close(
        base.link_mask,
        torch.tensor([[0, 1, 0]] * command.num_envs, dtype=torch.float32),
    )

    rank_path = tmp_path / f"link_command_modes_rank{rank}.json"
    receipt = json.loads(rank_path.read_text(encoding="utf-8"))
    assert receipt["process_id"] == rank
    assert receipt["mode_names"] == list(MODE_CONFIG["mode_candidates"])
    assert receipt["mode_counts"] == [0, command.num_envs, 0]
    assert receipt["resample_calls"] == 1
    assert receipt["actor_features_masked"] is True
    assert receipt["critic_targets_and_rewards"] == "complete_clean_reference"
    assert receipt["future_frames_share_mode"] is True
    assert receipt["frame_offsets"] == [1, 2]
    assert receipt["actor_frame_offsets"] == [1, 2]
    assert receipt["critic_frame_offsets"] == [1, 2, 3]
    assert receipt["mode_table"] == [[1, 0, 0], [0, 1, 0], [1, 1, 1]]
    assert receipt["probabilities"] == [0, 1, 0]
    assert "LINK_COMMAND_MODES " in capsys.readouterr().out
    global_path = tmp_path / "link_command_modes.json"
    assert global_path.exists() is (rank == 0)
    if rank == 0:
        assert json.loads(global_path.read_text(encoding="utf-8")) == receipt


def test_non_boolean_tracking_extensions_fail_before_base_init(
    env_class, tmp_path
):
    env, command, base, requested_terms = build_env(env_class, tmp_path)
    env.config["tracking_extensions"] = {"enabled": "yes"}

    with pytest.raises(
        TypeError, match="tracking_extensions.enabled must be boolean"
    ):
        env._init_isaaclab_env()

    assert env.base_init_calls == 0
    assert requested_terms == []
    assert command._link_observation_adapter is base
    assert list(tmp_path.iterdir()) == []


def test_missing_link_adapter_fails_before_writing_receipt(
    env_class, tmp_path
):
    env, command, _, requested_terms = build_env(env_class, tmp_path)
    command._link_observation_adapter = None

    with pytest.raises(
        ValueError, match="Mode masks require enabled link observations"
    ):
        env._init_isaaclab_env()

    assert env.base_init_calls == 1
    assert requested_terms == ["ref_motion"]
    assert list(tmp_path.iterdir()) == []
