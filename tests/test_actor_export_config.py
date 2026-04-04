import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.modules.agent_modules import (
    PPOTFActor,
    PPOTFRefRouterActor,
    PPOTFRefRouterSeqActor,
    PPOTFRefRouterV3Actor,
    _clone_module_for_cpu_export,
)
from holomotion.src.modules.network_modules import (
    GroupedMoEBlock,
    GroupedMoETransformerPolicy,
    ReferenceRoutedGroupedMoETransformerPolicy,
    ReferenceRoutedGroupedMoETransformerPolicyV2,
    ReferenceRoutedGroupedMoETransformerPolicyV3,
    export_safe_scaled_dot_product_attention,
)
from holomotion.src.utils.onnx_export import export_policy_to_onnx
from tensordict import TensorDict

try:
    onnx = importlib.import_module("onnx")
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        f"Optional ONNX test dependency missing: {exc.name}"
    ) from exc


class _DummyTFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 1
        self.max_ctx_len = 4
        self.n_kv_heads = 1
        self.head_dim = 2

    def forward(
        self,
        obs: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return obs[:, :2], past_key_values


class _DummyAttentionTFModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 1
        self.max_ctx_len = 4
        self.n_kv_heads = 1
        self.head_dim = 2

    def forward(
        self,
        obs: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        max_len = past_key_values.shape[3]
        valid_len = (current_pos + 1).clamp(max=max_len)
        pos_idx = torch.arange(max_len, device=obs.device, dtype=torch.int64)
        mask = (pos_idx[None, :] < valid_len[:, None])[:, None, None, :]

        q = obs[:, :2].reshape(batch_size, 1, 1, 2)
        k = torch.zeros(batch_size, 1, max_len, 2, device=obs.device)
        v = torch.ones(batch_size, 1, max_len, 2, device=obs.device)
        attn_out = export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        actions = attn_out.reshape(batch_size, 2)
        return actions, past_key_values


class _RecordingDeviceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_device = "cuda:0"
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(str(device))
        self.current_device = str(device)
        return self


def _make_minimal_real_transformer_actor(
    *,
    n_layers: int = 1,
    routing_score_fn: str = "softmax",
    num_fine_experts: int = 1,
    top_k: int = 1,
    use_dynamic_bias: bool = False,
    dense_layer_at_last: bool = False,
    selected_expert_margin_to_unselected_enabled: bool = False,
    selected_expert_margin_to_unselected_target: float = 0.0,
) -> PPOTFActor:
    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.actor_module = GroupedMoETransformerPolicy(
        input_dim=6,
        output_dim=2,
        module_config_dict={
            "type": "GroupedMoETransformerPolicy",
            "num_fine_experts": num_fine_experts,
            "num_shared_experts": 0,
            "top_k": top_k,
            "obs_embed_mlp_hidden": 8,
            "d_model": 8,
            "n_layers": n_layers,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ff_mult": 1.0,
            "ff_mult_dense": 1,
            "attn_dropout": 0.0,
            "mlp_dropout": 0.0,
            "max_ctx_len": 4,
            "dense_layer_at_last": dense_layer_at_last,
            "use_gated_attn": False,
            "use_qk_norm": True,
            "routing_score_fn": routing_score_fn,
            "use_dynamic_bias": use_dynamic_bias,
            "selected_expert_margin_to_unselected": {
                "enabled": selected_expert_margin_to_unselected_enabled,
                "target": selected_expert_margin_to_unselected_target,
            },
        },
    )
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.assembler = SimpleNamespace(output_dim=6)
    return actor


def _capture_moe_router_outputs(
    monkeypatch,
    *,
    export_mode: bool,
    top_k: int,
    use_dynamic_bias: bool,
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_x: torch.Tensor | None = None,
    expert_bias: torch.Tensor | None = None,
):
    block = GroupedMoEBlock(
        d_model=x.shape[-1],
        n_heads=2,
        n_kv_heads=1,
        num_fine_experts=router_weight.shape[0],
        num_shared_experts=1,
        top_k=top_k,
        ff_mult=1.0,
        use_qk_norm=True,
        use_gated_attn=False,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        use_dynamic_bias=use_dynamic_bias,
        routing_score_fn="softmax",
    )
    block.eval()

    with torch.no_grad():
        block.router.weight.copy_(router_weight)
        if expert_bias is not None:
            block.expert_bias.copy_(expert_bias)

    captured = {}

    def _fake_sparse_experts(
        x_input: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> torch.Tensor:
        captured["topk_idx"] = topk_idx.detach().clone()
        captured["topk_scores"] = topk_scores.detach().clone()
        return torch.zeros_like(x_input)

    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: export_mode)
    monkeypatch.setattr(block, "_compute_sparse_experts", _fake_sparse_experts)

    captured["output"] = block.compute_moe_ffn(x, router_x=router_x)
    return captured


def _make_minimal_ref_router_actor() -> PPOTFRefRouterActor:
    actor = PPOTFRefRouterActor.__new__(PPOTFRefRouterActor)
    nn.Module.__init__(actor)
    actor.actor_module = ReferenceRoutedGroupedMoETransformerPolicy(
        input_dim=8,
        output_dim=2,
        module_config_dict={
            "type": "ReferenceRoutedGroupedMoETransformerPolicy",
            "num_fine_experts": 4,
            "num_shared_experts": 0,
            "top_k": 2,
            "obs_embed_mlp_hidden": 8,
            "router_embed_mlp_hidden": 8,
            "router_input_dim": 4,
            "router_feature_indices": [0, 1, 4, 5],
            "d_model": 8,
            "n_layers": 2,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ff_mult": 1.0,
            "ff_mult_dense": 1,
            "attn_dropout": 0.0,
            "mlp_dropout": 0.0,
            "max_ctx_len": 4,
            "use_gated_attn": False,
            "use_qk_norm": True,
            "routing_score_fn": "softmax",
            "use_dynamic_bias": False,
        },
    )
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.assembler = SimpleNamespace(output_dim=8)
    return actor


def _make_ref_router_v2_obs_schema() -> dict:
    return {
        "flattened_obs": {
            "seq_len": 1,
            "terms": [
                "unified/actor_ref_gravity_projection_cur",
                "unified/actor_ref_base_linvel_cur",
                "unified/actor_ref_base_angvel_cur",
                "unified/actor_ref_dof_pos_cur",
                "unified/actor_projected_gravity",
                "unified/actor_rel_robot_root_ang_vel",
                "unified/actor_dof_vel",
                "unified/actor_dof_pos",
                "unified/actor_ref_root_height_cur",
                "unified/actor_last_action",
            ],
        },
        "flattened_obs_fut": {
            "seq_len": 5,
            "terms": [
                "unified/actor_ref_gravity_projection_fut",
                "unified/actor_ref_base_linvel_fut",
                "unified/actor_ref_base_angvel_fut",
                "unified/actor_ref_dof_pos_fut",
                "unified/actor_ref_root_height_fut",
            ],
        },
    }


def _make_ref_router_v2_obs(batch_size: list[int]) -> TensorDict:
    shape = list(batch_size)
    actor_fut_shape = shape + [5]
    unified = TensorDict(
        {
            "actor_ref_gravity_projection_cur": torch.randn(*shape, 3),
            "actor_ref_base_linvel_cur": torch.randn(*shape, 3),
            "actor_ref_base_angvel_cur": torch.randn(*shape, 3),
            "actor_ref_dof_pos_cur": torch.randn(*shape, 2),
            "actor_projected_gravity": torch.randn(*shape, 3),
            "actor_rel_robot_root_ang_vel": torch.randn(*shape, 3),
            "actor_dof_vel": torch.randn(*shape, 3),
            "actor_dof_pos": torch.randn(*shape, 3),
            "actor_ref_root_height_cur": torch.randn(*shape, 1),
            "actor_last_action": torch.randn(*shape, 2),
            "actor_ref_gravity_projection_fut": torch.randn(
                *actor_fut_shape, 3
            ),
            "actor_ref_base_linvel_fut": torch.randn(*actor_fut_shape, 3),
            "actor_ref_base_angvel_fut": torch.randn(*actor_fut_shape, 3),
            "actor_ref_dof_pos_fut": torch.randn(*actor_fut_shape, 2),
            "actor_ref_root_height_fut": torch.randn(*actor_fut_shape, 1),
        },
        batch_size=shape,
    )
    return TensorDict({"unified": unified}, batch_size=shape)


def _make_minimal_ref_router_v2_actor() -> PPOTFRefRouterSeqActor:
    obs_schema = _make_ref_router_v2_obs_schema()
    obs_example = _make_ref_router_v2_obs([2])
    return PPOTFRefRouterSeqActor(
        obs_schema=obs_schema,
        module_config_dict={
            "type": "ReferenceRoutedGroupedMoETransformerPolicyV2",
            "num_fine_experts": 4,
            "num_shared_experts": 0,
            "top_k": 2,
            "obs_embed_mlp_hidden": 8,
            "d_model": 8,
            "n_layers": 2,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ff_mult": 1.0,
            "ff_mult_dense": 1,
            "attn_dropout": 0.0,
            "mlp_dropout": 0.0,
            "max_ctx_len": 4,
            "use_gated_attn": False,
            "use_qk_norm": True,
            "routing_score_fn": "softmax",
            "use_dynamic_bias": False,
            "ref_hist_n_layers": 1,
            "ref_future_conv_channels": 8,
            "ref_future_conv_layers": 2,
            "ref_future_conv_kernel_size": 3,
            "ref_future_conv_stride": 2,
            "obs_norm": {"enabled": False},
            "output_dim": 2,
        },
        num_actions=2,
        init_noise_std=0.2,
        obs_example=obs_example,
    )


def _make_minimal_ref_router_v3_actor() -> PPOTFRefRouterV3Actor:
    obs_schema = _make_ref_router_v2_obs_schema()
    obs_example = _make_ref_router_v2_obs([2])
    return PPOTFRefRouterV3Actor(
        obs_schema=obs_schema,
        module_config_dict={
            "type": "ReferenceRoutedGroupedMoETransformerPolicyV3",
            "num_fine_experts": 4,
            "num_shared_experts": 0,
            "top_k": 2,
            "obs_embed_mlp_hidden": 8,
            "d_model": 8,
            "n_layers": 2,
            "n_heads": 2,
            "n_kv_heads": 1,
            "ff_mult": 1.0,
            "ff_mult_dense": 1,
            "attn_dropout": 0.0,
            "mlp_dropout": 0.0,
            "max_ctx_len": 4,
            "use_gated_attn": False,
            "use_qk_norm": True,
            "routing_score_fn": "softmax",
            "use_dynamic_bias": False,
            "ref_hist_n_layers": 1,
            "router_future_hidden_dim": 12,
            "router_layer_proj_hidden_dim": 10,
            "obs_norm": {"enabled": False},
            "output_dim": 2,
        },
        num_actions=2,
        init_noise_std=0.2,
        obs_example=obs_example,
    )


def test_export_policy_to_onnx_uses_opset_17(monkeypatch, tmp_path):
    captured = {}

    class _FakeActor:
        def eval(self):
            return self

        def export_onnx(
            self,
            *,
            onnx_path,
            opset_version,
            use_kv_cache=True,
        ):
            captured["onnx_path"] = onnx_path
            captured["opset_version"] = opset_version
            captured["use_kv_cache"] = use_kv_cache
            return str(onnx_path)

    actor = _FakeActor()
    algo = SimpleNamespace(
        actor=actor,
        critic=SimpleNamespace(eval=lambda: None),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
        env=SimpleNamespace(_env=object()),
    )

    monkeypatch.setattr(
        "holomotion.src.utils.onnx_export.attach_onnx_metadata_holomotion",
        lambda env, onnx_path: None,
    )

    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"")
    export_policy_to_onnx(algo, str(checkpoint_path), use_kv_cache=False)

    assert captured["opset_version"] == 17
    assert captured["use_kv_cache"] is False


def test_export_policy_to_onnx_restores_training_mode(monkeypatch, tmp_path):
    class _FakeActor:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def export_onnx(
            self,
            *,
            onnx_path,
            opset_version,
            use_kv_cache=True,
        ):
            return str(onnx_path)

    class _FakeCritic:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

    actor = _FakeActor()
    critic = _FakeCritic()
    algo = SimpleNamespace(
        actor=actor,
        critic=critic,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
        env=SimpleNamespace(_env=object()),
    )

    monkeypatch.setattr(
        "holomotion.src.utils.onnx_export.attach_onnx_metadata_holomotion",
        lambda env, onnx_path: None,
    )

    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"")
    export_policy_to_onnx(algo, str(checkpoint_path), use_kv_cache=False)

    assert actor.training is True
    assert critic.training is True


def test_clone_module_for_cpu_export_does_not_move_live_module(monkeypatch):
    module = _RecordingDeviceModule()

    monkeypatch.setattr(
        "holomotion.src.modules.agent_modules._module_device",
        lambda _: torch.device("cuda:0"),
    )

    cloned = _clone_module_for_cpu_export(module)

    assert module.to_calls == []
    assert module.current_device == "cuda:0"
    assert isinstance(cloned, _RecordingDeviceModule)
    assert cloned is not module


def test_ppotf_actor_export_uses_legacy_torchscript(monkeypatch, tmp_path):
    export_calls = []

    def _fake_export(*args, **kwargs):
        export_calls.append(kwargs)

    monkeypatch.setattr(torch.onnx, "export", _fake_export)

    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.actor_module = _DummyTFModule()
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.assembler = SimpleNamespace(output_dim=6)

    out_path = tmp_path / "policy.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    assert len(export_calls) == 1
    assert export_calls[0]["opset_version"] == 17
    assert export_calls[0]["dynamo"] is False


def test_ppotf_actor_export_onnx_avoids_isnan(tmp_path):
    actor = PPOTFActor.__new__(PPOTFActor)
    nn.Module.__init__(actor)
    actor.actor_module = _DummyAttentionTFModule()
    actor.obs_norm_enabled = False
    actor.obs_normalizer = nn.Identity()
    actor.obs_norm_clip = 0.0
    actor.assembler = SimpleNamespace(output_dim=2)

    out_path = tmp_path / "policy.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    op_types = [node.op_type for node in model.graph.node]

    assert "IsNaN" not in op_types


def test_ppotf_real_transformer_export_onnx_avoids_isnan(tmp_path):
    actor = _make_minimal_real_transformer_actor()

    out_path = tmp_path / "policy_real_tf.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    op_types = [node.op_type for node in model.graph.node]

    assert "IsNaN" not in op_types


def test_ppotf_real_moe_transformer_export_reaches_router_ops(tmp_path):
    actor = _make_minimal_real_transformer_actor(
        n_layers=2,
        num_fine_experts=4,
        top_k=2,
        routing_score_fn="softmax",
    )

    out_path = tmp_path / "policy_real_moe_tf.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    op_types = [node.op_type for node in model.graph.node]

    assert "TopK" in op_types


def test_ppotf_real_moe_transformer_export_exposes_routing_outputs(tmp_path):
    actor = _make_minimal_real_transformer_actor(
        n_layers=3,
        num_fine_experts=4,
        top_k=2,
        routing_score_fn="softmax",
    )

    out_path = tmp_path / "policy_real_moe_tf_outputs.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    output_names = [value.name for value in model.graph.output]

    assert output_names == [
        "actions",
        "present_key_values",
        "moe_layer_1_expert_indices",
        "moe_layer_1_expert_logits",
        "moe_layer_2_expert_indices",
        "moe_layer_2_expert_logits",
    ]


def test_ppotf_real_moe_transformer_export_dense_last_uses_actual_moe_indices(
    tmp_path,
):
    actor = _make_minimal_real_transformer_actor(
        n_layers=4,
        num_fine_experts=4,
        top_k=2,
        routing_score_fn="softmax",
        dense_layer_at_last=True,
    )

    out_path = tmp_path / "policy_real_moe_tf_dense_last_outputs.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    output_names = [value.name for value in model.graph.output]

    assert output_names == [
        "actions",
        "present_key_values",
        "moe_layer_1_expert_indices",
        "moe_layer_1_expert_logits",
        "moe_layer_2_expert_indices",
        "moe_layer_2_expert_logits",
    ]


def test_ppotf_real_moe_transformer_export_avoids_reduce_log_sum_exp(
    tmp_path,
):
    actor = _make_minimal_real_transformer_actor(
        n_layers=2,
        num_fine_experts=4,
        top_k=2,
        routing_score_fn="softmax",
    )

    out_path = tmp_path / "policy_real_moe_tf_no_rlse.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    op_types = [node.op_type for node in model.graph.node]

    assert "ReduceLogSumExp" not in op_types


def test_export_safe_moe_router_matches_training_scores_for_topk1(monkeypatch):
    x = torch.tensor([[[1.0, -0.5, 0.25, 2.0]]], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [0.1, 0.3, -0.2, 0.5],
            [0.2, -0.4, 0.1, 0.7],
            [-0.3, 0.6, 0.2, -0.1],
            [0.4, 0.1, -0.5, 0.2],
        ],
        dtype=torch.float32,
    )

    eager = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=1,
        use_dynamic_bias=False,
        x=x,
        router_weight=router_weight,
    )
    export = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=True,
        top_k=1,
        use_dynamic_bias=False,
        x=x,
        router_weight=router_weight,
    )

    assert torch.equal(export["topk_idx"], eager["topk_idx"])
    torch.testing.assert_close(
        export["topk_scores"],
        eager["topk_scores"],
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_export_safe_moe_router_matches_training_scores_with_dynamic_bias(
    monkeypatch,
):
    x = torch.tensor(
        [
            [[0.2, -1.0, 0.5, 1.1], [0.4, 0.3, -0.7, 0.9]],
            [[-0.6, 0.8, 1.0, -0.2], [0.1, -0.4, 0.6, 0.7]],
        ],
        dtype=torch.float32,
    )
    router_weight = torch.tensor(
        [
            [0.2, -0.1, 0.5, 0.3],
            [-0.4, 0.7, 0.2, 0.1],
            [0.6, 0.2, -0.3, 0.4],
            [0.1, 0.5, 0.4, -0.6],
        ],
        dtype=torch.float32,
    )
    expert_bias = torch.tensor([0.0, 0.4, -0.3, 0.2], dtype=torch.float32)

    eager = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=2,
        use_dynamic_bias=True,
        x=x,
        router_weight=router_weight,
        expert_bias=expert_bias,
    )
    export = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=True,
        top_k=2,
        use_dynamic_bias=True,
        x=x,
        router_weight=router_weight,
        expert_bias=expert_bias,
    )

    assert torch.equal(export["topk_idx"], eager["topk_idx"])
    torch.testing.assert_close(
        export["topk_scores"],
        eager["topk_scores"],
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_grouped_moe_router_x_keeps_topk_when_main_input_changes(monkeypatch):
    router_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    router_x = torch.tensor([[[4.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    x_a = torch.tensor([[[0.0, 0.5, 1.0, 1.5]]], dtype=torch.float32)
    x_b = torch.tensor([[[3.0, -2.0, -1.0, 6.0]]], dtype=torch.float32)

    out_a = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=1,
        use_dynamic_bias=False,
        x=x_a,
        router_x=router_x,
        router_weight=router_weight,
    )
    out_b = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=1,
        use_dynamic_bias=False,
        x=x_b,
        router_x=router_x,
        router_weight=router_weight,
    )

    assert torch.equal(out_a["topk_idx"], out_b["topk_idx"])
    assert not torch.allclose(out_a["output"], out_b["output"])


def test_grouped_moe_router_x_changes_topk_when_router_input_changes(
    monkeypatch,
):
    router_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)
    router_x_a = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    router_x_b = torch.tensor([[[0.0, 5.0, 0.0, 0.0]]], dtype=torch.float32)

    out_a = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=1,
        use_dynamic_bias=False,
        x=x,
        router_x=router_x_a,
        router_weight=router_weight,
    )
    out_b = _capture_moe_router_outputs(
        monkeypatch,
        export_mode=False,
        top_k=1,
        use_dynamic_bias=False,
        x=x,
        router_x=router_x_b,
        router_weight=router_weight,
    )

    assert not torch.equal(out_a["topk_idx"], out_b["topk_idx"])


def test_ref_router_actor_export_keeps_single_obs_input_and_reaches_moe(
    tmp_path,
):
    actor = _make_minimal_ref_router_actor()

    out_path = tmp_path / "policy_ref_router.onnx"
    PPOTFActor.export_onnx(
        actor,
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    input_names = [value.name for value in model.graph.input]
    op_types = [node.op_type for node in model.graph.node]

    assert input_names == ["obs", "past_key_values", "step_idx"]
    assert "TopK" in op_types


def test_ref_router_v2_actor_export_keeps_single_obs_input_and_reaches_moe(
    tmp_path,
):
    actor = _make_minimal_ref_router_v2_actor()

    assert actor.onnx_past_key_values_shape(batch_size=1) == (
        3,
        2,
        1,
        4,
        1,
        4,
    )

    out_path = tmp_path / "policy_ref_router_v2.onnx"
    actor.export_onnx(
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    input_names = [value.name for value in model.graph.input]
    op_types = [node.op_type for node in model.graph.node]

    assert input_names == ["obs", "past_key_values", "step_idx"]
    assert "TopK" in op_types


def test_ref_router_v3_actor_export_keeps_single_obs_input_and_reaches_moe(
    tmp_path,
):
    actor = _make_minimal_ref_router_v3_actor()

    assert actor.onnx_past_key_values_shape(batch_size=1) == (
        3,
        2,
        1,
        4,
        1,
        4,
    )

    out_path = tmp_path / "policy_ref_router_v3.onnx"
    actor.export_onnx(
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    input_names = [value.name for value in model.graph.input]
    op_types = [node.op_type for node in model.graph.node]

    assert input_names == ["obs", "past_key_values", "step_idx"]
    assert "TopK" in op_types


def test_real_transformer_actor_export_supports_selected_expert_margin(
    tmp_path,
):
    actor = _make_minimal_real_transformer_actor(
        n_layers=2,
        num_fine_experts=4,
        top_k=2,
        selected_expert_margin_to_unselected_enabled=True,
        selected_expert_margin_to_unselected_target=0.4,
    )

    out_path = tmp_path / "policy_selected_expert_margin.onnx"
    actor.export_onnx(
        out_path,
        opset_version=17,
        use_kv_cache=True,
    )

    model = onnx.load(str(out_path))
    input_names = [value.name for value in model.graph.input]
    op_types = [node.op_type for node in model.graph.node]

    assert input_names == ["obs", "past_key_values", "step_idx"]
    assert "TopK" in op_types
