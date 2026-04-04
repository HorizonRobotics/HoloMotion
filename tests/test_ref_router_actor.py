from __future__ import annotations

import pytest
import torch
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from tensordict import TensorDict


def _make_ref_router_obs_schema() -> dict:
    return {
        "flattened_obs": {
            "seq_len": 1,
            "terms": [
                "unified/actor_ref_dof_pos_cur",
                "unified/actor_dof_pos",
                "unified/actor_ref_root_height_cur",
                "unified/actor_last_action",
            ],
        },
        "flattened_obs_fut": {
            "seq_len": 2,
            "terms": [
                "unified/actor_ref_dof_pos_fut",
                "unified/actor_ref_root_height_fut",
            ],
        },
    }


def _make_ref_router_obs(batch_size: list[int]) -> TensorDict:
    shape = list(batch_size)
    fut_shape = shape + [2]
    unified = TensorDict(
        {
            "actor_ref_dof_pos_cur": torch.randn(*shape, 2),
            "actor_dof_pos": torch.randn(*shape, 3),
            "actor_ref_root_height_cur": torch.randn(*shape, 1),
            "actor_last_action": torch.randn(*shape, 2),
            "actor_ref_dof_pos_fut": torch.randn(*fut_shape, 2),
            "actor_ref_root_height_fut": torch.randn(*fut_shape, 1),
        },
        batch_size=shape,
    )
    return TensorDict({"unified": unified}, batch_size=shape)


def _make_ref_router_actor(
    *,
    num_actions: int = 4,
    freeze_router: bool = False,
    aux_router_future_recon: dict | None = None,
) -> PPOTFRefRouterActor:
    obs_schema = _make_ref_router_obs_schema()
    obs_example = _make_ref_router_obs([2])
    module_config = {
        "type": "ReferenceRoutedGroupedMoETransformerPolicy",
        "num_fine_experts": 3,
        "num_shared_experts": 1,
        "top_k": 1,
        "routing_score_fn": "softmax",
        "routing_scale": 1.0,
        "use_dynamic_bias": False,
        "bias_update_rate": 0.001,
        "expert_bias_clip": 0.0,
        "obs_embed_mlp_hidden": 16,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 1,
        "use_gated_attn": False,
        "use_qk_norm": True,
        "ff_mult": 1.0,
        "ff_mult_dense": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "max_ctx_len": 4,
        "freeze_router": freeze_router,
        "obs_norm": {"enabled": False},
        "output_dim": num_actions,
        "aux_router_future_recon": aux_router_future_recon
        or {"enabled": False},
    }
    return PPOTFRefRouterActor(
        obs_schema=obs_schema,
        module_config_dict=module_config,
        num_actions=num_actions,
        init_noise_std=0.2,
        obs_example=obs_example,
    )


def test_ref_router_actor_infers_only_actor_ref_feature_indices():
    obs_schema = _make_ref_router_obs_schema()
    obs_example = _make_ref_router_obs([2])

    indices = PPOTFRefRouterActor.infer_router_feature_indices(
        obs_schema, obs_example
    )

    assert indices == [0, 1, 5, 8, 9, 10, 11, 12, 13]


def test_ref_router_actor_single_step_and_sequence_logp_match_contract():
    actor = _make_ref_router_actor()
    obs_td = _make_ref_router_obs([2])

    inference_out = actor(
        obs_td,
        mode="inference",
        update_obs_norm=False,
    )
    assert inference_out["actions"].shape == (2, 4)
    assert inference_out["mu"].shape == (2, 4)
    assert inference_out["sigma"].shape == (2, 4)

    cache_shape = actor.onnx_past_key_values_shape(batch_size=2)
    past_key_values = torch.zeros(*cache_shape, dtype=torch.float32)
    step_idx = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        actions, present = actor(
            obs_td,
            past_key_values=past_key_values,
            current_pos=step_idx,
        )
    assert actions.shape == (2, 4)
    assert present.shape == cache_shape

    obs_seq = _make_ref_router_obs([2, 3])
    actions_seq = torch.randn(2, 3, 4)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )
    seq_out = actor(
        obs_seq,
        actions=actions_seq,
        mode="sequence_logp",
        attn_mask=attn_mask,
        update_obs_norm=False,
    )

    assert seq_out["mu"].shape == (2, 3, 4)
    assert seq_out["sigma"].shape == (2, 3, 4)
    assert seq_out["actions_log_prob"].shape == (2, 3, 1)
    assert seq_out["entropy"].shape == (2, 3, 1)


def test_ref_router_actor_rejects_aux_router_future_recon():
    with pytest.raises(
        ValueError,
        match="does not support aux_router_future_recon",
    ):
        _make_ref_router_actor(
            aux_router_future_recon={"enabled": True, "weight": 1.0}
        )


def test_ppotf_select_actor_wrapper_rejects_ref_router_cross_attn():
    with pytest.raises(
        ValueError,
        match="ReferenceRoutedGroupedMoETransformerPolicy",
    ):
        PPOTF._select_actor_wrapper_cls(
            {
                "type": "ReferenceRoutedGroupedMoETransformerPolicy",
                "use_future_cross_attn": True,
            }
        )


def test_ref_router_actor_freeze_router_freezes_router_obs_embed():
    actor = _make_ref_router_actor(freeze_router=True)
    module = actor.actor_module

    assert module.freeze_router is True
    assert module.router_obs_embed[0].weight.requires_grad is False
    assert module.router_obs_embed[0].bias.requires_grad is False
    assert module.router_obs_embed[2].weight.requires_grad is False
    assert module.router_obs_embed[2].bias.requires_grad is False


def test_ref_router_actor_freeze_router_reapplies_after_load_state_dict():
    actor = _make_ref_router_actor(freeze_router=True)
    module = actor.actor_module
    state_dict = module.state_dict()

    module.router_obs_embed.requires_grad_(True)
    for layer in module.layers:
        if hasattr(layer, "router"):
            layer.router.requires_grad_(True)

    result = module.load_state_dict(state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.router_obs_embed[0].weight.requires_grad is False
    assert module.router_obs_embed[0].bias.requires_grad is False
    assert module.router_obs_embed[2].weight.requires_grad is False
    assert module.router_obs_embed[2].bias.requires_grad is False
    for layer in module.layers:
        if hasattr(layer, "router"):
            assert layer.router.weight.requires_grad is False
