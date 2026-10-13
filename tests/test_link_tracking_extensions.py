"""CPU contracts for opt-in link tracking extensions around AllModes."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from holomotion.src.env.link_augmented_motion_tracking import (
    install_link_tracking_extensions,
)
from holomotion.src.motion_tracking.link_feature_noise import (
    FeatureNoiseLinkAdapter,
)

MODE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion/src/env/mode_masked_motion_tracking.py"
)


def _mode_fixture(monkeypatch, tmp_path):
    calls = []
    base_adapter = SimpleNamespace(
        actor_frame_offsets=(1, 2), critic_frame_offsets=(1, 2)
    )
    command = SimpleNamespace(
        _link_observation_adapter=base_adapter,
        cfg=SimpleNamespace(link_observation={"mask": {}}),
    )

    class BaseEnv:
        def _init_isaaclab_env(self):
            calls.append("base")
            self._env = SimpleNamespace(
                command_manager=SimpleNamespace(get_term=lambda _: command)
            )
            return "base-result"

    class MaskAdapter:
        def __init__(self, base, mask):
            assert base is base_adapter
            assert mask == {}
            self.sampler = SimpleNamespace(description=lambda: {})

        def statistics(self):
            return {}

    motion_module = ModuleType("holomotion.src.env.motion_tracking")
    motion_module.MotionTrackingEnv = BaseEnv
    mask_module = ModuleType("holomotion.src.motion_tracking.link_mode_mask")
    mask_module.ModeMaskLinkAdapter = MaskAdapter
    extension_module = ModuleType(
        "holomotion.src.env.link_augmented_motion_tracking"
    )

    def install(env):
        assert isinstance(command._link_observation_adapter, MaskAdapter)
        assert env._env.command_manager.get_term("ref_motion") is command
        calls.append("install")

    extension_module.install_link_tracking_extensions = install
    monkeypatch.setitem(sys.modules, motion_module.__name__, motion_module)
    monkeypatch.setitem(sys.modules, mask_module.__name__, mask_module)
    monkeypatch.setitem(
        sys.modules, extension_module.__name__, extension_module
    )
    spec = importlib.util.spec_from_file_location(
        "_link_extension_mode_contract", MODE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    instance = module.ModeMaskedMotionTrackingEnv()
    instance.accelerator = None
    instance.log_dir = str(tmp_path)
    return instance, calls


@pytest.mark.parametrize(
    "extensions",
    [None, {}, {"enabled": False, "rewards": {"unused": {}}}],
)
def test_disabled_extension_preserves_mode_path_and_rng(
    monkeypatch, tmp_path, extensions
):
    env, calls = _mode_fixture(monkeypatch, tmp_path)
    env.config = (
        {} if extensions is None else {"tracking_extensions": extensions}
    )
    torch.manual_seed(43)
    rng_before = torch.random.get_rng_state().clone()
    assert env._init_isaaclab_env() == "base-result"
    assert calls == ["base"]
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert (tmp_path / "link_command_modes_rank0.json").exists()
    assert not (tmp_path / "link_training_extensions_rank0.json").exists()


def test_enabled_extension_installs_after_mode_mask(monkeypatch, tmp_path):
    env, calls = _mode_fixture(monkeypatch, tmp_path)
    env.config = {"tracking_extensions": {"enabled": True}}
    assert env._init_isaaclab_env() == "base-result"
    assert calls == ["base", "install"]


def test_non_boolean_extension_gate_fails_before_environment_init(
    monkeypatch, tmp_path
):
    env, calls = _mode_fixture(monkeypatch, tmp_path)
    env.config = {"tracking_extensions": {"enabled": "true"}}
    with pytest.raises(TypeError, match="must be boolean"):
        env._init_isaaclab_env()
    assert calls == []


@pytest.mark.parametrize(
    ("noise_enabled", "manager_fails"),
    [(False, False), (True, False), (True, True)],
)
def test_installer_preserves_base_rewards_and_activates_two_extra_terms(
    monkeypatch, tmp_path, noise_enabled, manager_fails
):
    class Term:
        def __init__(self, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    class Manager:
        def __init__(self, cfg, _):
            if manager_fails:
                raise RuntimeError("simulated manager rejection")
            self.active_terms = [
                name
                for name, value in vars(cfg).items()
                if isinstance(value, Term) and value.weight != 0.0
            ]

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    managers = ModuleType("isaaclab.managers")
    managers.RewardManager = Manager
    managers.RewardTermCfg = Term
    utils = ModuleType("isaaclab.utils")
    utils.__path__ = []
    io = ModuleType("isaaclab.utils.io")
    dumped = []
    io.dump_yaml = lambda path, cfg: dumped.append((path, cfg))
    for module in (isaaclab, managers, utils, io):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    old_term = Term(func=lambda _: None, weight=1.0, params={})
    old_cfg = SimpleNamespace(existing=old_term)
    command = SimpleNamespace(
        robot=SimpleNamespace(body_names=("pelvis",)),
        urdf_body_names=("pelvis",),
    )
    base_adapter = SimpleNamespace(command=command)
    command._link_observation_adapter = base_adapter
    original_manager = object()
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(rewards=old_cfg),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_manager=original_manager,
        num_envs=2,
    )
    env = SimpleNamespace(
        _env=runtime,
        config={
            "tracking_extensions": {
                "enabled": True,
                "command_name": "ref_motion",
                "command_noise": {"enabled": noise_enabled},
                "rewards": {
                    "anchor_height_tracking_exp": {
                        "weight": 0.5,
                        "params": {"std": 0.3, "anchor_body_name": "pelvis"},
                    },
                    "global_body_rotation_tracking_exp": {
                        "weight": 1.0,
                        "params": {"std": 0.4, "body_names": ["pelvis"]},
                    },
                },
            }
        },
        accelerator=None,
        log_dir=tmp_path,
    )
    torch.manual_seed(47)
    rng_before = torch.random.get_rng_state().clone()
    if manager_fails:
        with pytest.raises(RuntimeError, match="simulated manager rejection"):
            install_link_tracking_extensions(env)
        assert runtime.cfg.rewards is old_cfg
        assert runtime.reward_manager is original_manager
        assert command._link_observation_adapter is base_adapter
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        return
    install_link_tracking_extensions(env)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    assert old_cfg.existing is old_term
    assert runtime.cfg.rewards is not old_cfg
    assert runtime.cfg.rewards.existing.weight == old_term.weight
    assert runtime.cfg.rewards.existing.func is old_term.func
    if noise_enabled:
        assert isinstance(
            command._link_observation_adapter, FeatureNoiseLinkAdapter
        )
        assert command._link_observation_adapter.base is base_adapter
    else:
        assert command._link_observation_adapter is base_adapter
    assert set(runtime.reward_manager.active_terms) == {
        "existing",
        "anchor_height_tracking_exp",
        "global_body_rotation_tracking_exp",
    }
    assert len(dumped) == 2
    receipt = json.loads(
        (tmp_path / "link_training_extensions_rank0.json").read_text()
    )
    assert receipt["actor_command_noise_only"] is noise_enabled
    assert receipt["critic_and_reward_reference_clean"] is True


@pytest.mark.parametrize(
    ("rewards", "message"),
    [
        ({"unknown": {"weight": 1.0, "params": {"std": 0.3}}}, "Unknown"),
        (
            {
                "anchor_height_tracking_exp": {
                    "weight": float("nan"),
                    "params": {"std": 0.3},
                }
            },
            "weight must be finite",
        ),
        (
            {
                "anchor_height_tracking_exp": {
                    "weight": 0.5,
                    "params": {"std": 0.0},
                }
            },
            "std must be finite and positive",
        ),
        (
            {
                "global_body_rotation_tracking_exp": {
                    "weight": 1.0,
                    "params": {"std": 0.4, "body_names": ["unknown_body"]},
                }
            },
            "Body names missing",
        ),
    ],
)
def test_invalid_reward_spec_fails_before_mutating_environment(
    monkeypatch, tmp_path, rewards, message
):
    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    managers = ModuleType("isaaclab.managers")

    def should_not_build_manager(*_args):
        raise AssertionError("Invalid reward spec reached RewardManager")

    managers.RewardManager = should_not_build_manager
    managers.RewardTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    utils = ModuleType("isaaclab.utils")
    utils.__path__ = []
    io = ModuleType("isaaclab.utils.io")
    io.dump_yaml = lambda *_args: None
    for module in (isaaclab, managers, utils, io):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    command = SimpleNamespace(
        robot=SimpleNamespace(body_names=("pelvis",)),
        urdf_body_names=("pelvis",),
    )
    base_adapter = SimpleNamespace(command=command)
    command._link_observation_adapter = base_adapter
    original_rewards = SimpleNamespace()
    original_manager = SimpleNamespace(active_terms=[])
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(rewards=original_rewards),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_manager=original_manager,
    )
    env = SimpleNamespace(
        _env=runtime,
        config={
            "tracking_extensions": {
                "enabled": True,
                "command_noise": {"enabled": True},
                "rewards": rewards,
            }
        },
        log_dir=tmp_path,
    )
    with pytest.raises(ValueError, match=message):
        install_link_tracking_extensions(env)
    assert command._link_observation_adapter is base_adapter
    assert runtime.cfg.rewards is original_rewards
    assert runtime.reward_manager is original_manager
    assert not (tmp_path / "link_training_extensions_rank0.json").exists()
