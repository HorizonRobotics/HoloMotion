# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Small real network strict-load and recurrent ONNX parity."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from omegaconf import OmegaConf

from holomotion.src.evaluation.checkpoint_onnx_parity import (
    CheckpointOnnxParity,
    actor_example_from_flat,
    compare_arrays,
    construct_checkpoint_actor,
    load_actor_strict,
)
from holomotion.src.motion_tracking.link_actor_packing import (
    LINK_CURRENT_TERMS,
    link_actor_layout,
    link_future_terms,
)


def fixture_actor(include_error_terms=True, frames=2):
    cfg = {
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
        "router_embed_mlp_hidden": 16,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "use_gated_attn": False,
        "ff_mult": 1.0,
        "ff_mult_dense": 2,
        "attn_dropout": 0.0,
        "mlp_dropout": 0.0,
        "max_ctx_len": 16,
        "rope_max_seq_len": 64,
        "obs_norm": {
            "enabled": True,
            "epsilon": 1e-8,
            "update_method": "ema",
            "ema_momentum": 0.0001,
            "enable_clipping": True,
            "clip_range": 10.0,
        },
        "output_dim": 29,
        "router_obs_terms": [
            "unified/actor_link_reference_height",
            "unified/actor_link_link_mask",
            "unified/actor_link_time_valid",
            "unified/actor_link_target_pos",
            "unified/actor_link_target_rot6d",
            "unified/actor_link_time_offsets",
        ],
        "obs_schema": {
            "flattened_obs": {
                "seq_len": 1,
                "terms": ["unified/" + key for key in LINK_CURRENT_TERMS],
            },
            "flattened_obs_fut": {
                "seq_len": frames,
                "terms": [
                    "unified/" + key
                    for key in link_future_terms(include_error_terms)
                ],
            },
        },
    }
    config = OmegaConf.create(
        {
            "link_observation": {
                "link_names": list(range(14)),
                "frame_offsets": list(range(1, frames + 1)),
                "include_error_terms": include_error_terms,
            },
            "algo": {
                "config": {
                    "module_dict": {"actor": cfg},
                    "init_noise_std": 0.3,
                    "noise_std_type": "log",
                    "min_sigma": 0.01,
                    "max_sigma": 1.2,
                }
            },
        }
    )
    dim = link_actor_layout(
        config.link_observation,
        config.algo.config.module_dict.actor.obs_schema,
    )["actor_dim"]
    example = actor_example_from_flat(torch.zeros(1, dim), config)
    actor = construct_checkpoint_actor(config, example)
    with torch.no_grad():
        actor.obs_normalizer._mean.fill_(0.2)
        actor.obs_normalizer._std.fill_(1.3)
        actor.obs_normalizer.count.fill_(11)
    return config, actor


@pytest.mark.parametrize("include_errors", [True, False])
@pytest.mark.parametrize("frames", [2, 5])
def test_actor_example_preserves_flat_term_order(include_errors, frames):
    config, actor = fixture_actor(include_errors, frames)
    flat = torch.arange(actor.flat_obs_dim, dtype=torch.float32)[None]
    torch.testing.assert_close(
        actor.assembler(actor_example_from_flat(flat, config)), flat
    )


def test_numeric_mismatch_and_nonfinite_never_pass():
    assert not compare_arrays(np.array([1.1]), np.array([1.0]), 1e-4, 1e-4)[
        "passed"
    ]
    assert not compare_arrays(np.array([np.nan]), np.array([1.0]), 1e-4, 1e-4)[
        "passed"
    ]
    assert not compare_arrays(np.zeros((1, 2)), np.zeros((2, 1)), 1e-4, 1e-4)[
        "passed"
    ]


def test_strict_checkpoint_load_preserves_normalizer_and_rejects_missing_key(
    tmp_path,
):
    config, actor = fixture_actor()
    directory = tmp_path / "model_8" / "actor"
    directory.mkdir(parents=True)
    torch.save(actor.state_dict(), directory / "pytorch_model.bin")
    loaded = construct_checkpoint_actor(
        config, actor_example_from_flat(torch.zeros(1, 616))
    )
    metadata = load_actor_strict(loaded, directory)
    assert metadata["strict_actor_state"] and metadata["normalizer_restored"]
    torch.testing.assert_close(
        loaded.obs_normalizer._mean, actor.obs_normalizer._mean
    )
    state = actor.state_dict()
    del state["obs_normalizer._mean"]
    torch.save(state, directory / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_actor_strict(loaded, directory)


@pytest.mark.parametrize("include_errors", [True, False])
@pytest.mark.parametrize("frames", [2, 5])
def test_real_small_actor_continuous_cache_and_named_onnx_outputs(
    tmp_path, include_errors, frames
):
    torch.manual_seed(101)
    config, actor = fixture_actor(include_errors, frames)
    expected = 108 + frames + frames * (253 if include_errors else 127)
    assert actor.flat_obs_dim == expected
    directory = tmp_path / "checkpoint" / "actor"
    directory.mkdir(parents=True)
    torch.save(actor.state_dict(), directory / "pytorch_model.bin")
    loaded = construct_checkpoint_actor(
        config, actor_example_from_flat(torch.zeros(1, expected), config)
    )
    load_actor_strict(loaded, directory)
    path = tmp_path / "small.onnx"
    actor.export_onnx(path, use_kv_cache=True)
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    names = {
        "obs": "obs",
        "past": "past_key_values",
        "step": "step_idx",
        "actions": "actions",
        "present": "present_key_values",
    }
    loaded.actor_module.register_buffer("routing_diagnostic", torch.zeros(1))
    checker = CheckpointOnnxParity(loaded, session, names)
    cache = np.zeros(
        actor.onnx_past_key_values_shape(batch_size=1), np.float32
    )
    generator = np.random.default_rng(19)
    for step in range(8):
        feed = {
            "obs": generator.normal(0, 0.1, (1, expected)).astype(np.float32),
            "past_key_values": cache,
            "step_idx": np.array([step], np.int64),
        }
        output_names = [
            "present_key_values",
            "actions",
        ]  # deliberately opposite order
        values = session.run(output_names, feed)
        mapped = dict(zip(output_names, values, strict=True))
        loaded.actor_module.routing_diagnostic.add_(1)
        checker.check_step(feed, mapped)
        cache = mapped["present_key_values"]
    report = checker.report()
    assert report["status"] == "passed", report
    assert report["checked_steps"] == 8 and report["normalizer_unchanged"]
    # Diagnostic updates are allowed, but changing normalization must fail.
    loaded.obs_normalizer._mean.add_(1)
    report = checker.report()
    assert report["status"] == "failed"
    assert not report["normalizer_unchanged"]


def test_unintegrated_conditional_actor_fails_explicitly():
    config, _ = fixture_actor(include_error_terms=True, frames=5)
    config.algo.config.module_dict.actor.use_future_cross_attn = True
    with pytest.raises(ValueError, match="ActorRef export is not supported"):
        construct_checkpoint_actor(
            config, actor_example_from_flat(torch.zeros(1, 1378), config)
        )
