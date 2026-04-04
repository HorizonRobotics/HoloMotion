from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from holomotion.src.algo.ppo import PPO, _checkpoint_state_to_cpu
from holomotion.src.modules.agent_modules import PPOActor


class _DummyActor(nn.Module):
    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events
        self.noise_std_type = "log"
        self.log_std = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def override_sigma(self, sigma_override):
        self.events.append("override_sigma")
        PPOActor.override_sigma(self, sigma_override)


def test_ppo_load_reapplies_sigma_override_after_checkpoint_restore():
    events: list[str] = []
    actor = _DummyActor(events)
    algo = PPO.__new__(PPO)
    algo.is_main_process = False
    algo.device = torch.device("cpu")
    algo.actor = actor
    algo.critic = nn.Linear(1, 1)
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo.config = {"override_sigma": True, "sigma_override": 0.1}
    algo._load_extra_checkpoint_state = mock.Mock()
    algo._resolve_model_file_path = (
        lambda ckpt_path, model_name: f"{ckpt_path}:{model_name}"
    )

    algo.actor_optimizer = mock.Mock()
    algo.actor_optimizer.load_state_dict.side_effect = (
        lambda state_dict: events.append("actor_optimizer")
    )
    algo.critic_optimizer = mock.Mock()
    algo.critic_optimizer.load_state_dict.side_effect = (
        lambda state_dict: events.append("critic_optimizer")
    )

    loaded_sigma = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32)

    def _fake_load_accelerate_model(model, model_path, *, strict):
        events.append(f"load:{model_path}")
        if model is actor:
            with torch.no_grad():
                model.log_std.copy_(loaded_sigma.log())

    algo._load_accelerate_model = mock.Mock(
        side_effect=_fake_load_accelerate_model
    )

    loaded_dict = {
        "actor_optimizer_state_dict": {"state": {}},
        "critic_optimizer_state_dict": {"state": {}},
        "iter": 123,
        "infos": {"source": "unit-test"},
    }

    with mock.patch(
        "holomotion.src.algo.ppo.torch.load", return_value=loaded_dict
    ):
        infos = algo.load("checkpoint.pt")

    assert infos == {"source": "unit-test"}
    assert torch.allclose(
        actor.log_std.exp(),
        torch.full((3,), 0.1, dtype=torch.float32),
    )
    assert events.index("override_sigma") > events.index("actor_optimizer")
    assert events.index("override_sigma") > events.index("critic_optimizer")
    algo._load_extra_checkpoint_state.assert_called_once_with(loaded_dict)


def test_ppo_load_skips_optimizer_restore_during_offline_eval():
    algo = PPO.__new__(PPO)
    algo.is_main_process = False
    algo.is_offline_eval = True
    algo.device = torch.device("cpu")
    algo.actor = nn.Linear(1, 1)
    algo.critic = nn.Linear(1, 1)
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo.config = {}
    algo._load_extra_checkpoint_state = mock.Mock()
    algo._resolve_model_file_path = (
        lambda ckpt_path, model_name: f"{ckpt_path}:{model_name}"
    )
    algo._load_accelerate_model = mock.Mock()
    algo._maybe_override_loaded_actor_sigma = mock.Mock()

    algo.actor_optimizer = mock.Mock()
    algo.critic_optimizer = mock.Mock()

    loaded_dict = {
        "actor_optimizer_state_dict": {"state": {"stale": {}}},
        "critic_optimizer_state_dict": {"state": {"stale": {}}},
        "iter": 321,
        "infos": {"source": "offline-eval"},
    }

    with mock.patch(
        "holomotion.src.algo.ppo.torch.load", return_value=loaded_dict
    ):
        infos = algo.load("checkpoint.pt")

    assert infos == {"source": "offline-eval"}
    assert algo.current_learning_iteration == 321
    algo.actor_optimizer.load_state_dict.assert_not_called()
    algo.critic_optimizer.load_state_dict.assert_not_called()
    algo._maybe_override_loaded_actor_sigma.assert_called_once_with()
    algo._load_extra_checkpoint_state.assert_called_once_with(loaded_dict)


def test_ppo_load_skips_incompatible_optimizer_state_restore():
    algo = PPO.__new__(PPO)
    algo.is_main_process = False
    algo.is_offline_eval = False
    algo.device = torch.device("cpu")
    algo.actor = nn.Linear(1, 1)
    algo.critic = nn.Linear(1, 1)
    algo.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    algo.config = {}
    algo._load_extra_checkpoint_state = mock.Mock()
    algo._resolve_model_file_path = (
        lambda ckpt_path, model_name: f"{ckpt_path}:{model_name}"
    )
    algo._load_accelerate_model = mock.Mock()
    algo._maybe_override_loaded_actor_sigma = mock.Mock()

    algo.actor_optimizer = mock.Mock()
    algo.actor_optimizer.state_dict.return_value = {
        "state": {},
        "param_groups": [{"params": [0]}],
    }
    algo.actor_optimizer.load_state_dict.side_effect = AssertionError(
        "incompatible actor optimizer state should be skipped"
    )
    algo.critic_optimizer = mock.Mock()
    algo.critic_optimizer.state_dict.return_value = {
        "state": {},
        "param_groups": [{"params": [0]}],
    }

    loaded_dict = {
        "actor_optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(1)}},
            "param_groups": [{"params": [0, 1]}],
        },
        "critic_optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(2)}},
            "param_groups": [{"params": [0]}],
        },
        "iter": 77,
        "infos": {"source": "resume-training"},
    }

    with mock.patch(
        "holomotion.src.algo.ppo.torch.load", return_value=loaded_dict
    ):
        infos = algo.load("checkpoint.pt")

    assert infos == {"source": "resume-training"}
    assert algo.current_learning_iteration == 77
    algo.actor_optimizer.load_state_dict.assert_not_called()
    algo.critic_optimizer.load_state_dict.assert_called_once_with(
        loaded_dict["critic_optimizer_state_dict"]
    )
    algo._maybe_override_loaded_actor_sigma.assert_called_once_with()
    algo._load_extra_checkpoint_state.assert_called_once_with(loaded_dict)


def test_checkpoint_state_to_cpu_moves_nested_tensors():
    source = {
        "state": {
            0: {
                "exp_avg": torch.tensor([1.0, 2.0], requires_grad=True),
                "exp_avg_sq": torch.tensor([3.0, 4.0]),
            }
        },
        "param_groups": [{"lr": 1.0e-3}],
        "step_tensor": torch.tensor([5]),
    }

    converted = _checkpoint_state_to_cpu(source)

    assert converted is not source
    assert converted["state"] is not source["state"]
    assert converted["state"][0]["exp_avg"].device.type == "cpu"
    assert converted["state"][0]["exp_avg_sq"].device.type == "cpu"
    assert converted["step_tensor"].device.type == "cpu"
    assert converted["state"][0]["exp_avg"].requires_grad is False
    torch.testing.assert_close(
        converted["state"][0]["exp_avg"],
        source["state"][0]["exp_avg"].detach(),
    )
