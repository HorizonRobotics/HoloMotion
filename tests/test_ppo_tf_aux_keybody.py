import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from holomotion.src.algo.algo_utils import PpoAuxTransition, RolloutStorage
from holomotion.src.algo.ppo import PPO
from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import (
    PPOTFActor,
    TensorDictAssembler,
)
from holomotion.src.modules.network_modules import GroupedMoETransformerPolicy
from holomotion.src.modules.network_modules import GroupedMoEBlock
from holomotion.src.modules.network_modules import ModernTransformerBlock
from tensordict import TensorDict


def _make_aux_policy() -> GroupedMoETransformerPolicy:
    module_config = {
        "num_fine_experts": 2,
        "num_shared_experts": 1,
        "top_k": 1,
        "routing_score_fn": "softmax",
        "routing_scale": 1.0,
        "use_dynamic_bias": False,
        "bias_update_rate": 0.001,
        "expert_bias_clip": 0.0,
        "obs_embed_mlp_hidden": 16,
        "d_model": 8,
        "n_heads": 2,
        "n_kv_heads": 1,
        "use_gated_attn": False,
        "n_layers": 1,
        "ff_mult": 1.0,
        "ff_mult_dense": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "max_ctx_len": 4,
        "aux_state_pred": {
            "enabled": True,
            "keybody_contact_names": [
                "left_knee_link",
                "right_knee_link",
            ],
            "keybody_rel_pos_names": [
                "left_knee_link",
                "right_knee_link",
            ],
        },
    }
    return GroupedMoETransformerPolicy(
        input_dim=6,
        output_dim=4,
        module_config_dict=module_config,
    )


def _make_aux_actor() -> PPOTFActor:
    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.actor_module = _make_aux_policy()
    actor.aux_state_pred_enabled = True
    actor.aux_router_command_recon_enabled = False
    actor.aux_router_switch_penalty_enabled = False
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.actor_obs_transforms = []
    actor.assembler = TensorDictAssembler(
        {"flat_obs": {"seq_len": 1, "terms": ["flat_obs"]}},
        output_mode="flat",
    )
    actor.min_sigma = 0.01
    actor.max_sigma = 1.0
    actor.log_std = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    return actor


def _make_aux_command_policy(
    *,
    n_layers: int = 3,
    dense_layer_at_last: bool = False,
    enable_aux_router_command_recon: bool = True,
    freeze_router: bool = False,
) -> GroupedMoETransformerPolicy:
    module_config = {
        "num_fine_experts": 2,
        "num_shared_experts": 1,
        "top_k": 1,
        "routing_score_fn": "softmax",
        "routing_scale": 1.0,
        "use_dynamic_bias": False,
        "bias_update_rate": 0.001,
        "expert_bias_clip": 0.0,
        "obs_embed_mlp_hidden": 16,
        "d_model": 8,
        "n_heads": 2,
        "n_kv_heads": 1,
        "use_gated_attn": False,
        "n_layers": n_layers,
        "ff_mult": 1.0,
        "ff_mult_dense": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "max_ctx_len": 4,
        "dense_layer_at_last": dense_layer_at_last,
        "freeze_router": freeze_router,
        "aux_router_command_recon": {
            "enabled": enable_aux_router_command_recon,
            "output_dim": 5,
            "hidden_dim": 7,
        },
    }
    return GroupedMoETransformerPolicy(
        input_dim=6,
        output_dim=4,
        module_config_dict=module_config,
    )


def _make_temporal_aux_actor() -> PPOTFActor:
    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.actor_module = _make_aux_command_policy()
    actor.aux_state_pred_enabled = False
    actor.aux_router_command_recon_enabled = True
    actor.aux_router_switch_penalty_enabled = True
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.actor_obs_transforms = []
    actor.assembler = TensorDictAssembler(
        {"flat_obs": {"seq_len": 1, "terms": ["flat_obs"]}},
        output_mode="flat",
    )
    actor.min_sigma = 0.01
    actor.max_sigma = 1.0
    actor.log_std = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    return actor


def _make_temporal_only_aux_actor() -> PPOTFActor:
    actor = _make_temporal_aux_actor()
    actor.aux_router_command_recon_enabled = False
    return actor


def test_rollout_storage_allocates_ref_and_robot_keybody_targets():
    original_tokens = dict(PpoAuxTransition.SHAPE_TOKENS)
    PpoAuxTransition.SHAPE_TOKENS["C"] = 2
    PpoAuxTransition.SHAPE_TOKENS["K"] = 8
    try:
        obs_template = TensorDict(
            {"flat_obs": torch.zeros(2, 5)},
            batch_size=[2],
        )
        storage = RolloutStorage(
            num_envs=2,
            num_transitions_per_env=3,
            obs_template=obs_template,
            actions_shape=(4,),
            transition_cls=PpoAuxTransition,
        )
    finally:
        PpoAuxTransition.SHAPE_TOKENS = original_tokens

    assert storage.data["gt_ref_keybody_rel_pos"].shape == (3, 2, 8, 3)
    assert storage.data["gt_robot_keybody_rel_pos"].shape == (3, 2, 8, 3)


def test_grouped_moe_policy_returns_keybody_position_predictions():
    policy = _make_aux_policy()
    pre_moe_hidden = torch.randn(2, 3, policy.d_model)

    outputs = policy.predict_aux_from_pre_moe(pre_moe_hidden)

    assert outputs["base_lin_vel_loc"].shape == (2, 3, 3)
    assert outputs["base_lin_vel_log_std"].shape == (2, 3, 3)
    assert outputs["root_height_loc"].shape == (2, 3, 1)
    assert outputs["root_height_log_std"].shape == (2, 3, 1)
    assert outputs["keybody_contact_logits"].shape == (2, 3, 2)
    assert outputs["ref_keybody_rel_pos"].shape == (2, 3, 2, 3)
    assert outputs["robot_keybody_rel_pos"].shape == (2, 3, 2, 3)


def test_grouped_moe_policy_default_layout_keeps_dense_first_and_moe_tail():
    policy = _make_aux_command_policy(
        enable_aux_router_command_recon=False,
    )

    assert len(policy.layers) == 3
    assert isinstance(policy.layers[0], ModernTransformerBlock)
    assert all(
        isinstance(layer, GroupedMoEBlock) for layer in policy.layers[1:]
    )
    assert policy._num_moe_layers == 2
    assert policy._last_moe_layer_idx == 2


def test_grouped_moe_policy_dense_layer_at_last_keeps_only_middle_layers_moe():
    policy = _make_aux_command_policy(
        n_layers=4,
        dense_layer_at_last=True,
    )
    obs_seq = torch.randn(2, 3, 6, dtype=torch.float32)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    _, router_features = policy.sequence_mu(
        obs_seq,
        attn_mask=attn_mask,
        return_router_features=True,
    )

    assert isinstance(policy.layers[0], ModernTransformerBlock)
    assert isinstance(policy.layers[1], GroupedMoEBlock)
    assert isinstance(policy.layers[2], GroupedMoEBlock)
    assert isinstance(policy.layers[3], ModernTransformerBlock)
    assert policy._num_moe_layers == 2
    assert policy._last_moe_layer_idx == 2
    assert router_features.shape == (2, 3, 4)


def test_grouped_moe_policy_dense_layer_at_last_allows_shallow_fully_dense():
    policy = _make_aux_command_policy(
        n_layers=2,
        dense_layer_at_last=True,
        enable_aux_router_command_recon=False,
    )

    assert len(policy.layers) == 2
    assert all(
        isinstance(layer, ModernTransformerBlock) for layer in policy.layers
    )
    assert policy._num_moe_layers == 0
    assert policy._last_moe_layer_idx is None


def test_grouped_moe_policy_command_recon_uses_live_router_features():
    policy = _make_aux_command_policy()
    obs_seq = torch.randn(2, 3, 6, dtype=torch.float32, requires_grad=True)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    _, router_features = policy.sequence_mu(
        obs_seq,
        attn_mask=attn_mask,
        return_router_features=True,
    )
    pred = policy.predict_aux_router_command_from_router_features(
        router_features
    )

    assert policy._num_moe_layers == 2
    assert router_features.shape == (2, 3, 4)
    assert pred.shape == (2, 3, 5)
    assert router_features.requires_grad

    pred.sum().backward()

    first_moe = next(
        layer for layer in policy.layers if isinstance(layer, GroupedMoEBlock)
    )
    assert first_moe.last_router_distribution is not None
    assert first_moe.last_router_distribution.requires_grad
    assert first_moe.router.weight.grad is not None


def test_grouped_moe_policy_freeze_router_detaches_router_features_and_params():
    policy = _make_aux_command_policy(freeze_router=True)
    obs_seq = torch.randn(2, 3, 6, dtype=torch.float32, requires_grad=True)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    _, router_features, router_temporal_features = policy.sequence_mu(
        obs_seq,
        attn_mask=attn_mask,
        return_router_features=True,
        return_router_temporal_features=True,
    )
    pred = policy.predict_aux_router_command_from_router_features(
        router_features
    )

    first_moe = next(
        layer for layer in policy.layers if isinstance(layer, GroupedMoEBlock)
    )
    assert first_moe.freeze_router is True
    assert first_moe.router.weight.requires_grad is False
    assert router_features.requires_grad is False
    assert router_temporal_features.requires_grad is False

    pred.sum().backward()

    assert first_moe.last_router_distribution is not None
    assert first_moe.last_router_distribution.requires_grad is False
    assert first_moe.last_router_logits is not None
    assert first_moe.last_router_logits.requires_grad is False
    assert first_moe.router.weight.grad is None


def test_grouped_moe_policy_loads_legacy_aux_command_recon_head_keys():
    policy = _make_aux_command_policy(enable_aux_router_command_recon=True)
    state_dict = copy.deepcopy(policy.state_dict())

    expected_tensors = {}
    for key in list(state_dict.keys()):
        if "aux_router_command_recon_head." not in key:
            continue
        legacy_key = key.replace(
            "aux_router_command_recon_head.",
            "aux_command_recon_head.",
        )
        legacy_value = torch.randn_like(state_dict[key])
        expected_tensors[key] = legacy_value
        state_dict[legacy_key] = legacy_value
        del state_dict[key]

    result = policy.load_state_dict(state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, expected in expected_tensors.items():
        actual = policy.state_dict()[key]
        assert torch.allclose(actual, expected)


def test_grouped_moe_policy_ignores_legacy_aux_command_recon_head_keys_when_disabled():
    policy = _make_aux_command_policy(enable_aux_router_command_recon=False)
    legacy_policy = _make_aux_command_policy(
        enable_aux_router_command_recon=True
    )
    state_dict = copy.deepcopy(policy.state_dict())

    for key, value in legacy_policy.state_dict().items():
        if "aux_router_command_recon_head." not in key:
            continue
        legacy_key = key.replace(
            "aux_router_command_recon_head.",
            "aux_command_recon_head.",
        )
        state_dict[legacy_key] = value.clone()

    result = policy.load_state_dict(state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert policy.aux_router_command_recon_head is None


def test_grouped_moe_policy_clears_router_cache_before_deepcopy():
    policy = _make_aux_command_policy()
    obs_seq = torch.randn(2, 3, 6, dtype=torch.float32, requires_grad=True)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    _, router_features = policy.sequence_mu(
        obs_seq,
        attn_mask=attn_mask,
        return_router_features=True,
    )
    pred = policy.predict_aux_router_command_from_router_features(
        router_features
    )
    pred.sum().backward()

    first_moe = next(
        layer for layer in policy.layers if isinstance(layer, GroupedMoEBlock)
    )
    assert first_moe.last_router_distribution is not None

    policy.clear_router_distribution_cache()
    copied = copy.deepcopy(policy)

    copied_first_moe = next(
        layer for layer in copied.layers if isinstance(layer, GroupedMoEBlock)
    )
    assert copied_first_moe.last_router_distribution is None


def test_grouped_moe_block_tracks_least_utilized_expert_stats():
    block = GroupedMoEBlock(
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        num_fine_experts=4,
        num_shared_experts=1,
        top_k=1,
        ff_mult=1.0,
        use_qk_norm=True,
        use_gated_attn=False,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        use_dynamic_bias=False,
        routing_score_fn="softmax",
    )

    block._apply_bias_update_from_counts(torch.tensor([5, 3, 0, 2]))
    assert block.last_active_expert_ratio.item() == pytest.approx(0.75)
    assert block.last_max_expert_frac.item() == pytest.approx(0.5)
    assert block.last_min_expert_frac.item() == pytest.approx(0.0)
    assert block.last_dead_expert_ratio.item() == pytest.approx(0.25)

    block._apply_bias_update_from_counts(torch.tensor([5, 3, 1, 1]))
    assert block.last_min_expert_frac.item() == pytest.approx(0.1)
    assert block.last_dead_expert_ratio.item() == pytest.approx(0.0)


def test_grouped_moe_block_tracks_inactive_expert_margin_to_topk_loss():
    block = GroupedMoEBlock(
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        num_fine_experts=3,
        num_shared_experts=1,
        top_k=1,
        ff_mult=1.0,
        use_qk_norm=True,
        use_gated_attn=False,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        use_dynamic_bias=False,
        routing_score_fn="softmax",
        inactive_expert_margin_to_topk_enabled=True,
        inactive_expert_margin_to_topk_ratio_floor=0.5,
    )

    topk_idx = torch.tensor([[[0], [0]]], dtype=torch.long)
    dense_distribution = torch.tensor(
        [[[0.8, 0.15, 0.05], [0.7, 0.2, 0.1]]], dtype=torch.float32
    )
    choice_scores = torch.log(dense_distribution)

    loss = block._update_routed_expert_stats_and_floor_loss(
        topk_idx=topk_idx,
        dense_distribution=dense_distribution,
        choice_scores=choice_scores,
    )

    expected = torch.relu(
        choice_scores.gather(-1, topk_idx)[..., -1:] - choice_scores
    )
    expected = expected[..., 1:].sum() / 4.0

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        block.last_inactive_expert_margin_to_topk_loss, expected
    )
    torch.testing.assert_close(
        block.last_inactive_expert_margin_to_topk_loss_value,
        expected.detach(),
    )
    torch.testing.assert_close(
        block.last_inactive_expert_margin_to_topk_target,
        choice_scores.gather(-1, topk_idx)[..., -1:].mean(),
    )
    torch.testing.assert_close(
        block.last_dense_expert_usage,
        dense_distribution.mean(dim=(0, 1)),
    )


def test_grouped_moe_block_tracks_selected_expert_margin_to_unselected():
    block = GroupedMoEBlock(
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        num_fine_experts=4,
        num_shared_experts=1,
        top_k=2,
        ff_mult=1.0,
        use_qk_norm=True,
        use_gated_attn=False,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        use_dynamic_bias=False,
        routing_score_fn="softmax",
        selected_expert_margin_to_unselected_enabled=True,
        selected_expert_margin_to_unselected_target=0.4,
    )

    topk_idx = torch.tensor([[[0, 2], [1, 0]]], dtype=torch.long)
    dense_distribution = torch.tensor(
        [
            [
                [0.42, 0.21, 0.28, 0.09],
                [0.27, 0.36, 0.22, 0.15],
            ]
        ],
        dtype=torch.float32,
    )
    choice_scores = torch.tensor(
        [[[1.0, 0.3, 0.8, 0.1], [0.9, 1.2, 0.7, 0.4]]],
        dtype=torch.float32,
    )

    block._update_routed_expert_stats_and_floor_loss(
        topk_idx=topk_idx,
        dense_distribution=dense_distribution,
        choice_scores=choice_scores,
    )

    expected_margin = torch.tensor((0.5 + 0.2) / 2.0)
    expected_loss = torch.tensor((0.0 + 0.2) / 2.0)

    torch.testing.assert_close(
        block.last_selected_expert_margin_to_unselected,
        expected_margin,
    )
    torch.testing.assert_close(
        block.last_selected_expert_margin_to_unselected_loss,
        expected_loss,
    )
    torch.testing.assert_close(
        block.last_selected_expert_margin_to_unselected_loss_value,
        expected_loss,
    )


def test_ppotf_summarize_moe_layer_stats_uses_ema_metrics():
    moe_layers = [
        SimpleNamespace(
            last_ema_dead_expert_ratio=torch.tensor(0.25),
            last_ema_max_expert_frac=torch.tensor(0.50),
            last_selected_expert_margin_to_unselected=torch.tensor(0.30),
        ),
        SimpleNamespace(
            last_ema_dead_expert_ratio=torch.tensor(0.50),
            last_ema_max_expert_frac=torch.tensor(0.30),
            last_selected_expert_margin_to_unselected=torch.tensor(0.10),
        ),
    ]

    metrics = PPOTF._summarize_moe_layer_stats(moe_layers)

    assert metrics["moe_ema_dead_expert_ratio"] == pytest.approx(0.375)
    assert metrics["moe_ema_max_expert_frac"] == pytest.approx(0.40)
    assert metrics[
        "moe_selected_expert_margin_to_unselected"
    ] == pytest.approx(0.20)


def test_compute_routed_expert_orthogonal_loss_uses_active_experts_only():
    algo = PPOTF.__new__(PPOTF)
    algo.router_expert_orthogonal_min_active_usage = 0.1
    algo.router_expert_orthogonal_eps = 1.0e-8

    moe_layer = SimpleNamespace(
        last_routed_expert_usage=torch.tensor(
            [0.2, 0.12, 0.05], dtype=torch.float32
        ),
        down_proj=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 1.0]],
                [[0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
    )

    loss, active_count, mean_offdiag = (
        algo._compute_routed_expert_orthogonal_loss(
            moe_layer,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    active_vecs = F.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        p=2.0,
        dim=-1,
        eps=1.0e-8,
    )
    gram = active_vecs @ active_vecs.transpose(0, 1)
    offdiag = gram.masked_select(~torch.eye(2, dtype=torch.bool))

    torch.testing.assert_close(active_count, torch.tensor(2.0))
    torch.testing.assert_close(loss, offdiag.square().sum())
    torch.testing.assert_close(mean_offdiag, offdiag.abs().mean())


def test_compute_routed_expert_orthogonal_loss_returns_zero_below_two_active():
    algo = PPOTF.__new__(PPOTF)
    algo.router_expert_orthogonal_min_active_usage = 0.1
    algo.router_expert_orthogonal_eps = 1.0e-8

    moe_layer = SimpleNamespace(
        last_routed_expert_usage=torch.tensor(
            [0.2, 0.05, 0.01], dtype=torch.float32
        ),
        down_proj=torch.randn(3, 1, 2, dtype=torch.float32),
    )

    loss, active_count, mean_offdiag = (
        algo._compute_routed_expert_orthogonal_loss(
            moe_layer,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(active_count, torch.tensor(1.0))
    torch.testing.assert_close(mean_offdiag, torch.tensor(0.0))


def test_ppotf_actor_sequence_logp_emits_router_features_for_aux_router_losses():
    actor = _make_temporal_aux_actor()
    obs_td = TensorDict(
        {"flat_obs": torch.randn(2, 3, 6, dtype=torch.float32)},
        batch_size=[2, 3],
    )
    actions = torch.randn(2, 3, 4, dtype=torch.float32)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    outputs = actor(
        obs_td,
        actions=actions,
        mode="sequence_logp",
        attn_mask=attn_mask,
        update_obs_norm=False,
    )

    assert outputs["router_features"].shape == (2, 3, 4)
    assert outputs["router_temporal_features"].shape == (2, 3, 4)
    assert outputs["aux_router_command_recon"].shape == (2, 3, 5)


def test_ppotf_actor_sequence_logp_emits_only_router_features_for_temporal_only_aux():
    actor = _make_temporal_only_aux_actor()
    obs_td = TensorDict(
        {"flat_obs": torch.randn(2, 3, 6, dtype=torch.float32)},
        batch_size=[2, 3],
    )
    actions = torch.randn(2, 3, 4, dtype=torch.float32)
    attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).expand(
        2, -1, -1
    )

    outputs = actor(
        obs_td,
        actions=actions,
        mode="sequence_logp",
        attn_mask=attn_mask,
        update_obs_norm=False,
    )

    assert outputs["router_features"].shape == (2, 3, 4)
    assert outputs["router_temporal_features"].shape == (2, 3, 4)
    assert "aux_router_command_recon" not in outputs.keys()


def test_masked_adjacent_router_js_averages_only_valid_adjacent_tokens():
    router_features = torch.tensor(
        [
            [
                [0.8, 0.2, 0.6, 0.4],
                [0.6, 0.4, 0.5, 0.5],
                [0.1, 0.9, 0.4, 0.6],
            ]
        ],
        dtype=torch.float32,
    )
    valid_tok = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    loss = PPOTF._masked_adjacent_router_js(
        router_features=router_features,
        valid_tok=valid_tok,
        num_moe_layers=2,
        num_fine_experts=2,
    )

    layer0_prev = torch.tensor([0.8, 0.2], dtype=torch.float32)
    layer0_curr = torch.tensor([0.6, 0.4], dtype=torch.float32)
    mix0 = 0.5 * (layer0_prev + layer0_curr)
    js0 = 0.5 * (
        (layer0_prev * (torch.log(layer0_prev) - torch.log(mix0))).sum()
        + (layer0_curr * (torch.log(layer0_curr) - torch.log(mix0))).sum()
    )
    layer1_prev = torch.tensor([0.6, 0.4], dtype=torch.float32)
    layer1_curr = torch.tensor([0.5, 0.5], dtype=torch.float32)
    mix1 = 0.5 * (layer1_prev + layer1_curr)
    js1 = 0.5 * (
        (layer1_prev * (torch.log(layer1_prev) - torch.log(mix1))).sum()
        + (layer1_curr * (torch.log(layer1_curr) - torch.log(mix1))).sum()
    )
    expected = 0.5 * (js0 + js1)
    assert torch.isclose(loss, expected)


def test_masked_adjacent_router_normed_smooth_l1_averages_only_valid_adjacent_tokens():
    router_temporal_features = torch.tensor(
        [
            [
                [3.0, 1.0, 0.0],
                [2.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    valid_tok = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    loss = PPOTF._masked_adjacent_router_normed_smooth_l1(
        router_temporal_features=router_temporal_features,
        valid_tok=valid_tok,
        num_moe_layers=1,
        num_fine_experts=3,
    )

    prev_logits = router_temporal_features[:, :1].reshape(1, 1, 1, 3)
    curr_logits = router_temporal_features[:, 1:2].reshape(1, 1, 1, 3)
    prev_norm = F.normalize(
        prev_logits - prev_logits.mean(dim=-1, keepdim=True),
        p=2.0,
        dim=-1,
        eps=1.0e-5,
    )
    curr_norm = F.normalize(
        curr_logits - curr_logits.mean(dim=-1, keepdim=True),
        p=2.0,
        dim=-1,
        eps=1.0e-5,
    )
    expected = F.smooth_l1_loss(
        curr_norm,
        prev_norm,
        reduction="none",
        beta=1.0,
    ).mean()
    assert torch.isclose(loss, expected)


def test_masked_aux_keybody_mse_averages_only_valid_tokens():
    pred = torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
    target = torch.zeros_like(pred)
    valid_tok = torch.tensor([[1.0, 0.0]])

    loss = PPOTF._masked_aux_keybody_mse(pred, target, valid_tok)

    expected = torch.tensor((1.0 + 4.0 + 9.0) / 3.0)
    assert torch.isclose(loss, expected)


def test_masked_aux_huber_averages_only_valid_tokens():
    pred = torch.zeros(1, 2, 1, 3)
    target = torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
    valid_tok = torch.tensor([[1.0, 0.0]])

    loss = PPOTF._masked_aux_huber(
        pred=pred,
        target=target,
        valid_tok=valid_tok,
        beta=1.0,
    )

    expected = torch.tensor((0.5 + 1.5 + 2.5) / 3.0)
    assert torch.isclose(loss, expected)


def test_setup_configs_rejects_router_aux_terms_outside_motion_tracking():
    algo = PPOTF.__new__(PPOTF)
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {
            "enabled": False,
        },
        "aux_router_switch_penalty": {"enabled": True, "weight": 1.0},
    }
    algo.command_name = "velocity"

    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="aux_router_switch_penalty"):
            algo._setup_configs()


def test_setup_configs_rejects_unknown_router_switch_penalty_metric():
    algo = PPOTF.__new__(PPOTF)
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {
            "enabled": False,
        },
        "aux_router_switch_penalty": {
            "enabled": True,
            "weight": 1.0,
            "metric": "not_a_metric",
        },
    }
    algo.command_name = "ref_motion"

    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(
            ValueError, match="aux_router_switch_penalty.metric"
        ):
            algo._setup_configs()


def test_setup_configs_reads_inactive_expert_margin_to_topk_only():
    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"

    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
        "inactive_expert_margin_to_topk": {
            "enabled": True,
            "weight": 0.7,
            "ratio_floor": 0.2,
        },
    }
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()
    assert algo.use_inactive_expert_margin_to_topk is True
    assert algo.inactive_expert_margin_to_topk_weight == pytest.approx(0.7)
    assert algo.inactive_expert_margin_to_topk_ratio_floor == pytest.approx(
        0.2
    )

    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
    }
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()
    assert algo.use_inactive_expert_margin_to_topk is False
    assert algo.inactive_expert_margin_to_topk_weight == pytest.approx(0.0)
    assert algo.inactive_expert_margin_to_topk_ratio_floor == pytest.approx(
        0.0
    )


def test_setup_configs_reads_selected_expert_margin_to_unselected():
    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"

    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
        "selected_expert_margin_to_unselected": {
            "enabled": True,
            "weight": 0.9,
            "target": 0.3,
        },
    }
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()
    assert algo.use_selected_expert_margin_to_unselected is True
    assert algo.selected_expert_margin_to_unselected_weight == pytest.approx(
        0.9
    )
    assert algo.selected_expert_margin_to_unselected_target == pytest.approx(
        0.3
    )

    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
    }
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()
    assert algo.use_selected_expert_margin_to_unselected is False
    assert algo.selected_expert_margin_to_unselected_weight == pytest.approx(
        0.0
    )
    assert algo.selected_expert_margin_to_unselected_target == pytest.approx(
        0.0
    )


def test_setup_configs_reads_aux_router_future_recon():
    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
        "aux_router_future_recon": {
            "enabled": True,
            "weight": 0.7,
            "hidden_dim": 13,
            "huber_beta": 0.3,
        },
        "module_dict": {
            "actor": {
                "type": "ReferenceRoutedGroupedMoETransformerPolicyV3",
            }
        },
    }

    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()

    assert algo.use_aux_router_future_recon is True
    assert algo.aux_router_future_recon_weight == pytest.approx(0.7)
    assert algo.aux_router_future_recon_hidden_dim == 13
    assert algo.aux_router_future_recon_huber_beta == pytest.approx(0.3)


def test_setup_configs_reads_router_expert_orthogonal():
    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"

    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
        "inactive_expert_margin_to_topk": {"enabled": True, "weight": 0.7},
        "router_expert_orthogonal": {
            "enabled": True,
            "weight": 0.9,
            "min_active_usage": 0.2,
            "eps": 1.0e-6,
        },
    }
    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        algo._setup_configs()
    assert algo.use_router_expert_orthogonal is True
    assert algo.router_expert_orthogonal_weight == pytest.approx(0.9)
    assert algo.router_expert_orthogonal_min_active_usage == pytest.approx(0.2)
    assert algo.router_expert_orthogonal_eps == pytest.approx(1.0e-6)


def test_setup_configs_rejects_router_expert_orthogonal_without_inactive_margin():
    algo = PPOTF.__new__(PPOTF)
    algo.command_name = "ref_motion"
    algo.config = {
        "aux_state_pred": {"enabled": False},
        "aux_router_command_recon": {"enabled": False},
        "aux_router_switch_penalty": {"enabled": False},
        "router_expert_orthogonal": {
            "enabled": True,
            "weight": 0.9,
        },
    }

    with mock.patch.object(PPO, "_setup_configs", return_value=None):
        with pytest.raises(ValueError, match="requires.*inactive_expert"):
            algo._setup_configs()


def test_compute_aux_router_future_recon_loss_uses_normalized_future_targets():
    algo = PPOTF.__new__(PPOTF)
    algo.aux_router_future_recon_huber_beta = 0.5

    obs_schema = {
        "flattened_obs_fut": {
            "seq_len": 2,
            "terms": [
                "unified/actor_ref_base_linvel_fut",
                "unified/actor_ref_dof_pos_fut",
            ],
        }
    }
    obs_b = TensorDict(
        {
            "unified": TensorDict(
                {
                    "actor_ref_base_linvel_fut": torch.tensor(
                        [
                            [
                                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                    "actor_ref_dof_pos_fut": torch.tensor(
                        [
                            [
                                [[0.1, 0.2], [0.3, 0.4]],
                                [[0.5, 0.6], [0.7, 0.8]],
                                [[0.9, 1.0], [1.1, 1.2]],
                            ]
                        ],
                        dtype=torch.float32,
                    ),
                },
                batch_size=[1, 3],
            )
        },
        batch_size=[1, 3],
    )
    assembler = TensorDictAssembler(obs_schema, output_mode="flat")

    class _DummyPolicy(nn.Module):
        def normalize_aux_router_future_recon_target(
            self, future_target: torch.Tensor
        ) -> torch.Tensor:
            return future_target * 0.25

    actor_wrapper = SimpleNamespace(
        aux_router_future_recon_assembler=assembler,
        actor_module=_DummyPolicy(),
    )
    raw_target = assembler(obs_b.flatten(0, 1)).reshape(1, 3, -1)
    normalized_target = raw_target * 0.25
    pred = normalized_target + torch.tensor(
        [
            [
                [0.0] * raw_target.shape[-1],
                [0.5] * raw_target.shape[-1],
                [1.0] * raw_target.shape[-1],
            ]
        ],
        dtype=torch.float32,
    )
    actor_out = TensorDict(
        {"aux_router_future_recon": pred},
        batch_size=[1, 3],
    )
    valid_tok = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    loss = algo._compute_aux_router_future_recon_loss(
        actor_wrapper=actor_wrapper,
        actor_out=actor_out,
        obs_b=obs_b,
        valid_tok=valid_tok,
    )

    expected = PPOTF._masked_aux_huber(
        pred=pred,
        target=normalized_target,
        valid_tok=valid_tok,
        beta=0.5,
    )
    assert torch.isclose(loss, expected)


def test_root_relative_body_pos_uses_consistent_environment_frame():
    body_pos_w = torch.tensor(
        [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=torch.float32
    )
    root_pos_env = torch.zeros(1, 3, dtype=torch.float32)
    root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    env_origins = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)

    rel = PPOTF._root_relative_body_pos_from_mixed_position_frames(
        body_pos_w=body_pos_w,
        root_pos_env=root_pos_env,
        root_quat_w=root_quat_w,
        env_origins=env_origins,
    )

    expected = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32
    )
    assert torch.allclose(rel, expected)
