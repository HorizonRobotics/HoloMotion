from __future__ import annotations

import pytest
import torch
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import (
    PPOTFRefRouterSeqActor,
    PPOTFRefRouterV3Actor,
)
from tensordict import TensorDict


REF_CUR_TERM_DIMS = {
    "actor_ref_gravity_projection_cur": 3,
    "actor_ref_base_linvel_cur": 3,
    "actor_ref_base_angvel_cur": 3,
    "actor_ref_dof_pos_cur": 2,
    "actor_ref_root_height_cur": 1,
}

REF_FUT_TERM_DIMS = {
    "actor_ref_gravity_projection_fut": 3,
    "actor_ref_base_linvel_fut": 3,
    "actor_ref_base_angvel_fut": 3,
    "actor_ref_dof_pos_fut": 2,
    "actor_ref_root_height_fut": 1,
}


def _make_ref_router_v2_obs_schema(
    *,
    include_ref_cur: bool = True,
    include_ref_fut: bool = True,
) -> dict:
    flat_terms = []
    if include_ref_cur:
        flat_terms.extend(
            [
                "unified/actor_ref_gravity_projection_cur",
                "unified/actor_ref_base_linvel_cur",
                "unified/actor_ref_base_angvel_cur",
                "unified/actor_ref_dof_pos_cur",
                "unified/actor_ref_root_height_cur",
            ]
        )
    flat_terms.extend(
        [
            "unified/actor_projected_gravity",
            "unified/actor_rel_robot_root_ang_vel",
            "unified/actor_dof_pos",
            "unified/actor_dof_vel",
            "unified/actor_last_action",
        ]
    )
    schema = {
        "flattened_obs": {"seq_len": 1, "terms": flat_terms},
    }
    if include_ref_fut:
        schema["flattened_obs_fut"] = {
            "seq_len": 5,
            "terms": [
                "unified/actor_ref_gravity_projection_fut",
                "unified/actor_ref_base_linvel_fut",
                "unified/actor_ref_base_angvel_fut",
                "unified/actor_ref_dof_pos_fut",
                "unified/actor_ref_root_height_fut",
            ],
        }
    return schema


def _make_ref_router_v2_obs(batch_size: list[int]) -> TensorDict:
    shape = list(batch_size)
    fut_shape = shape + [5]
    unified = TensorDict(
        {
            "actor_ref_gravity_projection_cur": torch.randn(*shape, 3),
            "actor_ref_base_linvel_cur": torch.randn(*shape, 3),
            "actor_ref_base_angvel_cur": torch.randn(*shape, 3),
            "actor_ref_dof_pos_cur": torch.randn(*shape, 2),
            "actor_ref_root_height_cur": torch.randn(*shape, 1),
            "actor_projected_gravity": torch.randn(*shape, 3),
            "actor_rel_robot_root_ang_vel": torch.randn(*shape, 3),
            "actor_dof_pos": torch.randn(*shape, 4),
            "actor_dof_vel": torch.randn(*shape, 4),
            "actor_last_action": torch.randn(*shape, 2),
            "actor_ref_gravity_projection_fut": torch.randn(*fut_shape, 3),
            "actor_ref_base_linvel_fut": torch.randn(*fut_shape, 3),
            "actor_ref_base_angvel_fut": torch.randn(*fut_shape, 3),
            "actor_ref_dof_pos_fut": torch.randn(*fut_shape, 2),
            "actor_ref_root_height_fut": torch.randn(*fut_shape, 1),
        },
        batch_size=shape,
    )
    return TensorDict({"unified": unified}, batch_size=shape)


def _make_ref_router_v2_actor(
    *,
    obs_schema: dict | None = None,
    num_actions: int = 4,
    aux_state_pred: dict | None = None,
    aux_router_command_recon: dict | None = None,
    freeze_router: bool = False,
) -> PPOTFRefRouterSeqActor:
    obs_schema = (
        _make_ref_router_v2_obs_schema() if obs_schema is None else obs_schema
    )
    obs_example = _make_ref_router_v2_obs([2])
    module_config = {
        "type": "ReferenceRoutedGroupedMoETransformerPolicyV2",
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
        "ref_hist_n_layers": 1,
        "ref_future_conv_channels": 8,
        "ref_future_conv_layers": 2,
        "ref_future_conv_kernel_size": 3,
        "ref_future_conv_stride": 2,
        "obs_norm": {"enabled": False},
        "output_dim": num_actions,
        "aux_state_pred": aux_state_pred or {"enabled": False},
        "aux_router_command_recon": aux_router_command_recon
        or {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
    }
    return PPOTFRefRouterSeqActor(
        obs_schema=obs_schema,
        module_config_dict=module_config,
        num_actions=num_actions,
        init_noise_std=0.2,
        obs_example=obs_example,
    )


def _make_ref_router_v3_actor(
    *,
    obs_schema: dict | None = None,
    num_actions: int = 4,
    freeze_router: bool = False,
    aux_router_future_recon: dict | None = None,
) -> PPOTFRefRouterV3Actor:
    obs_schema = (
        _make_ref_router_v2_obs_schema() if obs_schema is None else obs_schema
    )
    obs_example = _make_ref_router_v2_obs([2])
    module_config = {
        "type": "ReferenceRoutedGroupedMoETransformerPolicyV3",
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
        "ref_hist_n_layers": 1,
        "router_future_hidden_dim": 12,
        "router_layer_proj_hidden_dim": 10,
        "obs_norm": {"enabled": False},
        "output_dim": num_actions,
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_future_recon": aux_router_future_recon
        or {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
    }
    return PPOTFRefRouterV3Actor(
        obs_schema=obs_schema,
        module_config_dict=module_config,
        num_actions=num_actions,
        init_noise_std=0.2,
        obs_example=obs_example,
    )


def test_ppotf_select_actor_wrapper_uses_ref_router_seq_actor():
    actor_cls = PPOTF._select_actor_wrapper_cls(
        {"type": "ReferenceRoutedGroupedMoETransformerPolicyV2"}
    )

    assert actor_cls is PPOTFRefRouterSeqActor


def test_ppotf_select_actor_wrapper_uses_ref_router_v3_actor():
    actor_cls = PPOTF._select_actor_wrapper_cls(
        {"type": "ReferenceRoutedGroupedMoETransformerPolicyV3"}
    )

    assert actor_cls is PPOTFRefRouterV3Actor


def test_ref_router_seq_actor_infers_shared_ref_partitions_without_router_schemas():
    actor = _make_ref_router_v2_actor()

    assert actor.state_obs_input_dim > 0
    assert actor.ref_cur_token_dim == sum(REF_CUR_TERM_DIMS.values())
    assert actor.ref_fut_token_dim == sum(REF_FUT_TERM_DIMS.values())
    assert actor.ref_fut_seq_len == 5

    cache_shape = actor.onnx_past_key_values_shape(batch_size=2)
    assert len(cache_shape) == 6
    assert cache_shape[0] == actor.actor_module.onnx_kv_layers
    assert cache_shape[1] == 2
    assert cache_shape[2] == 2
    assert cache_shape[-1] == 4


def test_ref_router_v3_actor_keeps_full_obs_backbone_and_layer_router_adapters():
    actor = _make_ref_router_v3_actor()
    module = actor.actor_module

    assert actor.full_obs_input_dim > actor.state_obs_input_dim
    assert module.full_obs_input_dim == actor.full_obs_input_dim
    assert module.obs_embed[0].in_features == actor.full_obs_input_dim
    assert len(module.router_layer_projections) == sum(
        isinstance(layer, type(module.layers[1])) for layer in module.layers
    )


def test_ref_router_v3_history_backbone_consumes_flat_ref_motion():
    actor = _make_ref_router_v3_actor()
    module = actor.actor_module
    x = torch.randn(2, 3, module.full_obs_input_dim)

    _, ref_cur_x, ref_fut_x = module._split_actor_ref_inputs(x)

    assert hasattr(module, "_build_router_ref_motion")
    ref_motion_x = module._build_router_ref_motion(ref_cur_x, ref_fut_x)
    assert ref_motion_x.shape == (
        2,
        3,
        actor.ref_cur_token_dim
        + actor.ref_fut_seq_len * actor.ref_fut_token_dim,
    )
    assert module.ref_frame_embed[0].in_features == ref_motion_x.shape[-1]
    assert not hasattr(module, "router_future_obs_embed")
    assert not hasattr(module, "router_future_pool")
    assert not hasattr(module, "router_summary_fusion")
    assert not hasattr(module, "router_summary_norm")


def test_ref_router_v3_actor_sequence_logp_emits_aux_router_future_recon():
    actor = _make_ref_router_v3_actor(
        aux_router_future_recon={
            "enabled": True,
            "hidden_dim": 9,
            "weight": 1.0,
        }
    )
    obs_td = _make_ref_router_v2_obs([2, 3])
    actions_seq = torch.randn(2, 3, 4)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    seq_out = actor(
        obs_td,
        actions=actions_seq,
        mode="sequence_logp",
        attn_mask=attn_mask,
        update_obs_norm=False,
    )

    assert seq_out["aux_router_future_recon"].shape == (
        2,
        3,
        actor.ref_fut_seq_len * actor.ref_fut_token_dim,
    )


def test_ref_router_v3_actor_updates_future_recon_empirical_normalizer():
    actor = _make_ref_router_v3_actor(
        aux_router_future_recon={
            "enabled": True,
            "hidden_dim": 9,
            "weight": 1.0,
        }
    )
    obs_td = _make_ref_router_v2_obs([2, 3])
    actions_seq = torch.randn(2, 3, 4)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    assert actor.aux_router_future_recon_assembler is not None
    assert (
        int(actor.actor_module.aux_router_future_recon_normalizer.count) == 0
    )

    actor(
        obs_td,
        actions=actions_seq,
        mode="sequence_logp",
        attn_mask=attn_mask,
        update_obs_norm=True,
    )

    assert (
        int(actor.actor_module.aux_router_future_recon_normalizer.count) == 6
    )


def test_ref_router_v2_freeze_router_freezes_reference_router_path():
    actor = _make_ref_router_v2_actor(freeze_router=True)
    module = actor.actor_module
    x = torch.randn(2, 3, module.full_obs_input_dim, requires_grad=True)

    mu, router_h, router_temporal_features = module.sequence_mu(
        x,
        return_ref_aux_hidden=True,
        return_router_temporal_features=True,
    )
    mu.sum().backward()

    first_moe = next(
        layer for layer in module.layers if hasattr(layer, "router")
    )
    assert module.freeze_router is True
    assert module.ref_frame_embed[0].weight.requires_grad is False
    assert module.ref_hist_attn.q_proj.weight.requires_grad is False
    assert module.ref_future_conv[0].weight.requires_grad is False
    assert module.router_ref_pool.q_proj.weight.requires_grad is False
    assert module.router_query.requires_grad is False
    assert first_moe.router.weight.requires_grad is False
    assert router_h.requires_grad is False
    assert router_temporal_features.requires_grad is False
    assert module.ref_frame_embed[0].weight.grad is None
    assert module.ref_hist_attn.q_proj.weight.grad is None
    assert module.ref_future_conv[0].weight.grad is None
    assert module.router_ref_pool.q_proj.weight.grad is None
    assert module.router_query.grad is None
    assert first_moe.router.weight.grad is None


def test_ref_router_v3_freeze_router_reapplies_after_load_state_dict():
    actor = _make_ref_router_v3_actor(freeze_router=True)
    module = actor.actor_module
    state_dict = module.state_dict()

    module.ref_frame_embed.requires_grad_(True)
    module.ref_hist_attn.requires_grad_(True)
    module.router_layer_projections.requires_grad_(True)
    for layer in module.layers:
        if hasattr(layer, "router"):
            layer.router.requires_grad_(True)

    result = module.load_state_dict(state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.ref_frame_embed[0].weight.requires_grad is False
    assert module.ref_hist_attn.q_proj.weight.requires_grad is False
    assert module.router_layer_projections[0][1].weight.requires_grad is False
    assert not hasattr(module, "router_future_obs_embed")
    assert not hasattr(module, "router_future_pool")
    assert not hasattr(module, "router_summary_fusion")
    assert not hasattr(module, "router_summary_norm")
    for layer in module.layers:
        if hasattr(layer, "router"):
            assert layer.router.weight.requires_grad is False


def test_ref_router_v2_film_head_starts_near_identity():
    actor = _make_ref_router_v2_actor()
    module = actor.actor_module

    assert hasattr(module, "_actor_film_gain")
    gains = module._actor_film_gain().detach()
    assert gains.shape == (module.d_model,)
    assert torch.allclose(gains, torch.full_like(gains, 0.05), atol=1.0e-5)
    last_linear = module.actor_ref_film[-1]
    assert torch.count_nonzero(last_linear.weight.detach()) == 0
    assert torch.count_nonzero(last_linear.bias.detach()) == 0

    hidden = torch.randn(2, 1, module.d_model)
    actor_ref_ctx = torch.randn(2, 1, module.d_model)
    conditioned = module._apply_actor_ref_film(hidden, actor_ref_ctx)
    assert torch.allclose(conditioned, hidden)


def test_ref_router_v2_pre_moe_hidden_precedes_film_modulation():
    actor = _make_ref_router_v2_actor()
    module = actor.actor_module

    with torch.no_grad():
        module.actor_film_gain_raw.fill_(100.0)
        module.actor_ref_film[-1].weight.zero_()
        module.actor_ref_film[-1].bias.fill_(2.0)

    x = torch.randn(2, module.full_obs_input_dim)
    x_seq = x[:, None, :]
    state_x, ref_cur_x, ref_fut_x = module._split_actor_ref_inputs(x_seq)
    state_h = module.obs_embed(state_x)
    ref_cur_h = module.ref_frame_embed(ref_cur_x)
    ref_hist_attn = module.ref_hist_attn(
        module.ref_hist_norm(ref_cur_h),
        *module.get_cos_sin(ref_cur_h, torch.zeros(2, 1, dtype=torch.long)),
        mask=None,
    )
    ref_hist_h = module.ref_hist_out_norm(ref_cur_h + ref_hist_attn)
    ref_fut_tokens = module._encode_future_tokens(ref_fut_x)
    shared_ref_tokens = torch.cat(
        [ref_hist_h.unsqueeze(2), ref_fut_tokens], dim=2
    )
    router_h = module._pool_router_context(shared_ref_tokens)
    cos, sin = module.get_cos_sin(state_h, torch.zeros(2, 1, dtype=torch.long))
    block0_hidden = module._forward_layers_range(
        state_h,
        cos=cos,
        sin=sin,
        mask=None,
        router_h=router_h,
        start_layer=0,
        end_layer=1,
    )

    _, pre_moe_hidden = module.sequence_mu(
        x_seq,
        return_pre_moe_hidden=True,
    )

    assert torch.allclose(pre_moe_hidden, block0_hidden)


def test_ref_router_v2_film_gain_is_bounded_per_channel():
    actor = _make_ref_router_v2_actor()
    module = actor.actor_module

    assert hasattr(module, "actor_film_gain_raw")
    with torch.no_grad():
        module.actor_film_gain_raw.copy_(
            torch.linspace(-100.0, 100.0, module.d_model)
        )

    gains = module._actor_film_gain()

    assert gains.shape == (module.d_model,)
    assert torch.all(gains >= 0.0)
    assert torch.all(gains <= module.actor_film_gain_max + 1.0e-6)
    assert torch.unique(gains).numel() > 1


def test_ref_router_v2_film_perturbation_rms_stays_bounded():
    actor = _make_ref_router_v2_actor()
    module = actor.actor_module

    assert hasattr(module, "actor_film_gain_raw")
    with torch.no_grad():
        module.actor_ref_film[-1].weight.zero_()
        module.actor_ref_film[-1].bias.fill_(100.0)
        module.actor_film_gain_raw.fill_(100.0)

    hidden = torch.randn(4, 3, module.d_model)
    actor_ref_ctx = torch.randn(4, 3, module.d_model)
    conditioned = module._apply_actor_ref_film(hidden, actor_ref_ctx)
    delta = conditioned - hidden
    delta_rms = delta.pow(2).mean(dim=-1).sqrt()

    assert torch.all(delta_rms <= module.actor_film_gain_max + 1.0e-5)


def test_ref_router_v2_aux_prediction_stays_bound_to_returned_pre_moe_hidden():
    actor = _make_ref_router_v2_actor(
        aux_state_pred={
            "enabled": True,
            "w_base_lin_vel": 1.0,
            "w_keybody_contact": 1.0,
            "w_ref_keybody_rel_pos": 1.0,
            "w_robot_keybody_rel_pos": 1.0,
            "keybody_contact_names": ["knee"],
            "keybody_rel_pos_names": ["knee"],
        }
    )
    module = actor.actor_module
    module.eval()

    x_a = torch.randn(1, 1, module.full_obs_input_dim)
    x_b = x_a + 0.5

    with torch.no_grad():
        _, pre_a, ref_aux_a = module.sequence_mu(
            x_a,
            return_pre_moe_hidden=True,
            return_ref_aux_hidden=True,
        )
        aux_a = module.predict_aux_from_pre_moe(
            pre_a, ref_aux_hidden=ref_aux_a
        )
        _, pre_b, ref_aux_b = module.sequence_mu(
            x_b,
            return_pre_moe_hidden=True,
            return_ref_aux_hidden=True,
        )
        aux_a_late = module.predict_aux_from_pre_moe(
            pre_a, ref_aux_hidden=ref_aux_a
        )
        aux_b = module.predict_aux_from_pre_moe(
            pre_b, ref_aux_hidden=ref_aux_b
        )

    assert torch.allclose(
        aux_a_late["ref_keybody_rel_pos"], aux_a["ref_keybody_rel_pos"]
    )
    assert torch.allclose(
        aux_a_late["base_lin_vel_loc"], aux_a["base_lin_vel_loc"]
    )
    assert not torch.allclose(
        aux_a["ref_keybody_rel_pos"], aux_b["ref_keybody_rel_pos"]
    )
    assert not hasattr(pre_a, "_ref_aux_hidden")


def test_ref_router_v2_sequence_single_step_and_cached_onnx_agree():
    actor = _make_ref_router_v2_actor()
    module = actor.actor_module
    module.eval()

    x_seq = torch.randn(1, 2, module.full_obs_input_dim)
    attn_mask = torch.tril(torch.ones(2, 2, dtype=torch.bool)).unsqueeze(0)

    with torch.no_grad():
        mu_seq = module.sequence_mu(x_seq, attn_mask=attn_mask)

        module.reset_kv_cache(num_envs=1, device=x_seq.device)
        mu_step_0 = module.single_step_mu(x_seq[:, 0, :])
        mu_step_1 = module.single_step_mu(x_seq[:, 1, :])
        mu_single_step = torch.stack([mu_step_0, mu_step_1], dim=1)

        cache_shape = actor.onnx_past_key_values_shape(batch_size=1)
        past_key_values = torch.zeros(*cache_shape, dtype=x_seq.dtype)
        step_0 = torch.zeros(1, dtype=torch.long)
        step_1 = torch.ones(1, dtype=torch.long)
        mu_onnx_0, present_0 = module.forward(
            x_seq[:, 0, :],
            past_key_values=past_key_values,
            current_pos=step_0,
        )
        mu_onnx_1, present_1 = module.forward(
            x_seq[:, 1, :],
            past_key_values=present_0,
            current_pos=step_1,
        )
        mu_onnx = torch.stack([mu_onnx_0, mu_onnx_1], dim=1)

    assert torch.allclose(mu_single_step, mu_seq, atol=1.0e-5, rtol=1.0e-4)
    assert torch.allclose(mu_onnx, mu_seq, atol=1.0e-5, rtol=1.0e-4)
    assert present_0.shape == cache_shape
    assert present_1.shape == cache_shape


def test_ref_router_v3_sequence_single_step_and_cached_onnx_agree():
    actor = _make_ref_router_v3_actor()
    module = actor.actor_module
    module.eval()

    x_seq = torch.randn(1, 2, module.full_obs_input_dim)
    attn_mask = torch.tril(torch.ones(2, 2, dtype=torch.bool)).unsqueeze(0)

    with torch.no_grad():
        mu_seq = module.sequence_mu(x_seq, attn_mask=attn_mask)

        module.reset_kv_cache(num_envs=1, device=x_seq.device)
        mu_step_0 = module.single_step_mu(x_seq[:, 0, :])
        mu_step_1 = module.single_step_mu(x_seq[:, 1, :])
        mu_single_step = torch.stack([mu_step_0, mu_step_1], dim=1)

        cache_shape = actor.onnx_past_key_values_shape(batch_size=1)
        past_key_values = torch.zeros(*cache_shape, dtype=x_seq.dtype)
        step_0 = torch.zeros(1, dtype=torch.long)
        step_1 = torch.ones(1, dtype=torch.long)
        mu_onnx_0, present_0 = module.forward(
            x_seq[:, 0, :],
            past_key_values=past_key_values,
            current_pos=step_0,
        )
        mu_onnx_1, present_1 = module.forward(
            x_seq[:, 1, :],
            past_key_values=present_0,
            current_pos=step_1,
        )
        mu_onnx = torch.stack([mu_onnx_0, mu_onnx_1], dim=1)

    assert torch.allclose(mu_single_step, mu_seq, atol=1.0e-5, rtol=1.0e-4)
    assert torch.allclose(mu_onnx, mu_seq, atol=1.0e-5, rtol=1.0e-4)
    assert present_0.shape == cache_shape
    assert present_1.shape == cache_shape


def test_ref_router_seq_actor_single_step_and_sequence_logp_match_contract():
    actor = _make_ref_router_v2_actor()
    obs_td = _make_ref_router_v2_obs([2])

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

    obs_seq = _make_ref_router_v2_obs([2, 3])
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


def test_ref_router_v3_actor_single_step_and_sequence_logp_match_contract():
    actor = _make_ref_router_v3_actor()
    obs_td = _make_ref_router_v2_obs([2])

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

    obs_seq = _make_ref_router_v2_obs([2, 3])
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


def test_ref_router_seq_actor_sequence_logp_emits_aux_preds_without_metadata():
    actor = _make_ref_router_v2_actor(
        aux_state_pred={
            "enabled": True,
            "w_base_lin_vel": 1.0,
            "w_keybody_contact": 1.0,
            "w_ref_keybody_rel_pos": 1.0,
            "w_robot_keybody_rel_pos": 1.0,
            "keybody_contact_names": ["knee"],
            "keybody_rel_pos_names": ["knee"],
        }
    )

    obs_seq = _make_ref_router_v2_obs([2, 3])
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

    assert "aux_ref_keybody_rel_pos" in seq_out.keys()
    assert "aux_robot_keybody_rel_pos" in seq_out.keys()
    assert "aux_base_lin_vel_loc" in seq_out.keys()
    assert seq_out["aux_ref_keybody_rel_pos"].shape == (2, 3, 1, 3)
    assert seq_out["aux_robot_keybody_rel_pos"].shape == (2, 3, 1, 3)


def test_ref_router_seq_actor_requires_all_shared_ref_terms():
    obs_schema = _make_ref_router_v2_obs_schema(include_ref_cur=False)

    with pytest.raises(ValueError, match="missing required current ref term"):
        _make_ref_router_v2_actor(obs_schema=obs_schema)


def test_ref_router_seq_actor_rejects_aux_router_command_recon():
    with pytest.raises(ValueError, match="aux_router_command_recon"):
        _make_ref_router_v2_actor(
            aux_router_command_recon={"enabled": True, "hidden_dim": 8}
        )


def test_ref_router_seq_actor_rejects_unsupported_aux_state_pred_weights():
    with pytest.raises(ValueError, match="root_height"):
        _make_ref_router_v2_actor(
            aux_state_pred={
                "enabled": True,
                "w_base_lin_vel": 0.0,
                "w_keybody_contact": 0.0,
                "w_ref_keybody_rel_pos": 0.0,
                "w_robot_keybody_rel_pos": 0.0,
                "w_root_height": 1.0,
                "keybody_contact_names": [],
                "keybody_rel_pos_names": [],
            }
        )
