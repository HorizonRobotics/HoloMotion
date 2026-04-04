import torch
import torch.nn as nn
from holomotion.src.modules.agent_modules import PPOTFActor
from holomotion.src.modules.network_modules import EmpiricalNormalization


def _make_actor_with_obs_norm(obs_dim: int = 16) -> PPOTFActor:
    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.obs_norm_enabled = True
    actor.obs_norm_clip = 10.0
    actor.obs_normalizer = EmpiricalNormalization(shape=(obs_dim,))
    return actor


def test_obs_norm_update_is_not_captured_by_dynamo():
    actor = _make_actor_with_obs_norm()
    obs = torch.randn(8, 16)

    def normalize_with_update(x: torch.Tensor) -> torch.Tensor:
        return actor._normalize_actor_obs(x, True)

    explanation = torch._dynamo.explain(normalize_with_update)(obs)
    graph_code = "\n".join(graph.code for graph in explanation.graphs)

    assert "torch.var" not in graph_code
    assert "torch.mean" not in graph_code

    count_before_compile = actor.obs_normalizer.count.item()
    compiled = torch.compile(normalize_with_update, backend="eager")
    normalized = compiled(obs)

    assert normalized.shape == obs.shape
    assert (
        actor.obs_normalizer.count.item() - count_before_compile
        == obs.shape[0]
    )
