"""Static four-mode contract checks against the checked-in training recipe."""

from __future__ import annotations
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import yaml
from humanoid_policy.link_policy_contract import (
    _validate_v2_outputs,
    _validate_v2_runtime_session,
    compile_control_modes,
    validate_v2_reference_frequency,
)

from holomotion.src.motion_tracking.link_mode_mask import ModeMaskSampler

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "holomotion/config/training/motion_tracking"


def _frequency_config():
    return {
        "obs": {"target_fps": 50},
        "env": {
            "config": {
                "commands": {"ref_motion": {"params": {"target_fps": 50}}}
            }
        },
    }


def test_v2_frequency_contract_uses_saved_env_config_command_path():
    checked = validate_v2_reference_frequency(_frequency_config(), 50)
    assert checked == {
        "obs.target_fps": 50.0,
        "env.config.commands.ref_motion.params.target_fps": 50.0,
        "robot.policy_freq": 50.0,
    }


@pytest.mark.parametrize("field", ["observation", "command", "policy"])
def test_v2_frequency_contract_rejects_mismatch(field):
    config = _frequency_config()
    policy_freq = 50
    if field == "observation":
        config["obs"]["target_fps"] = 60
    elif field == "command":
        config["env"]["config"]["commands"]["ref_motion"]["params"][
            "target_fps"
        ] = 60
    else:
        policy_freq = 60
    with pytest.raises(ValueError, match="50 Hz"):
        validate_v2_reference_frequency(config, policy_freq)


def test_v2_frequency_contract_rejects_missing_command_path():
    config = _frequency_config()
    del config["env"]["config"]
    with pytest.raises(ValueError, match="env.config.commands"):
        validate_v2_reference_frequency(config, 50)


def test_v2_frequency_contract_accepts_optional_saved_bundle():
    path = os.environ.get("HMV2_SAVED_CONFIG_PATH")
    if not path:
        pytest.skip("Set HMV2_SAVED_CONFIG_PATH for private exported config")
    checked = validate_v2_reference_frequency(Path(path), 50)
    assert set(checked.values()) == {50.0}


def _training_contract():
    base = yaml.safe_load((CONFIG / "motion_tracking_v2_0_0.yaml").read_text())
    link = deepcopy(base["link_observation"])
    sampler = ModeMaskSampler(link["link_names"], link["mask"], device="cpu")
    record = {
        "link_names": list(sampler.link_names),
        "mode_names": list(sampler.mode_names),
        "mode_table": sampler.table.tolist(),
        "actor_frame_offsets": list(link["frame_offsets"]),
    }
    return link, record


def test_four_deploy_modes_use_one_fixed_14_link_schema():
    link, record = _training_contract()
    compiled = compile_control_modes(link, record)
    assert {name: sum(mask) for name, mask in compiled.items()} == {
        "Pelvis-1": 1,
        "Keypoint-3": 3,
        "Keypoint-5": 5,
        "WholeBody-14": 14,
    }
    assert all(len(mask) == 14 for mask in compiled.values())


def test_reject_mode_table_that_differs_from_saved_config():
    link, record = _training_contract()
    row = record["mode_names"].index("Keypoint-5")
    record["mode_table"][row][0] = 0
    with pytest.raises(ValueError, match="differs from config"):
        compile_control_modes(link, record)


def test_saved_mode_labels_resolve_to_canonical_control_point_names():
    link, record = _training_contract()
    expected = compile_control_modes(link, record)
    labels = {
        "Keypoint-3": "saved_three_points",
        "Keypoint-5": "saved_five_points",
    }
    link["mask"]["mode_candidates"] = {
        labels.get(name, name): selected
        for name, selected in link["mask"]["mode_candidates"].items()
    }
    record["mode_names"] = [
        labels.get(name, name) for name in record["mode_names"]
    ]

    assert compile_control_modes(link, record) == expected


def test_renamed_mode_still_rejects_a_mismatched_saved_table():
    link, record = _training_contract()
    record["mode_names"][record["mode_names"].index("Keypoint-5")] = (
        "different_name"
    )
    with pytest.raises(ValueError, match="names/order differ"):
        compile_control_modes(link, record)


def test_reject_changed_training_link_membership():
    link, record = _training_contract()
    link["mask"]["mode_candidates"]["Keypoint-3"] = [
        "pelvis",
        "left_wrist_yaw_link",
        "left_ankle_roll_link",
    ]
    sampler = ModeMaskSampler(link["link_names"], link["mask"], device="cpu")
    record["mode_table"] = sampler.table.tolist()
    with pytest.raises(ValueError, match="selects unexpected links"):
        compile_control_modes(link, record)


def test_reject_noncontiguous_future_offsets():
    link, record = _training_contract()
    link["frame_offsets"][1] = 9
    record["actor_frame_offsets"] = list(link["frame_offsets"])
    with pytest.raises(ValueError, match="contiguous"):
        compile_control_modes(link, record)


def test_runtime_session_rejects_bad_kv_step_and_nonfinite_gains():
    def node(name, dtype, shape):
        return SimpleNamespace(name=name, type=dtype, shape=shape)

    kv = [1, 2, 1, 32, 4, 64]
    inputs = [
        node("obs", "tensor(float)", [1, 1378]),
        node("past_key_values", "tensor(float)", kv),
        node("step_idx", "tensor(int64)", [1]),
    ]
    outputs = [
        node("actions", "tensor(float)", [1, 29]),
        node("present_key_values", "tensor(float)", kv),
    ]
    metadata = {
        name: ",".join(["0"] * 29)
        for name in (
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        )
    }
    metadata["rope_max_seq_len"] = "8192"
    session = SimpleNamespace(
        get_inputs=lambda: inputs,
        get_outputs=lambda: outputs,
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata),
    )
    actor = {
        "max_ctx_len": 32,
        "rope_max_seq_len": 8192,
        "n_layers": 1,
        "dense_layer_at_first": True,
        "top_k": 2,
        "num_fine_experts": 1024,
    }
    _validate_v2_runtime_session(session, actor, 1378)
    inputs[-1] = node("step_idx", "tensor(float)", [1])
    with pytest.raises(ValueError, match="type or shape"):
        _validate_v2_runtime_session(session, actor, 1378)
    inputs[-1] = node("step_idx", "tensor(int64)", [1])
    metadata["joint_stiffness"] = ",".join(["nan"] * 29)
    with pytest.raises(ValueError, match="finite"):
        _validate_v2_runtime_session(session, actor, 1378)


def _moe_output_nodes(specs, *, runtime):
    if runtime:
        dtype_names = {
            onnx.TensorProto.FLOAT: "tensor(float)",
            onnx.TensorProto.INT64: "tensor(int64)",
        }
        return [
            SimpleNamespace(name=name, type=dtype_names[dtype], shape=shape)
            for name, (dtype, shape) in specs.items()
        ]
    return [
        onnx.helper.make_tensor_value_info(name, dtype, shape)
        for name, (dtype, shape) in specs.items()
    ]


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_core",
        "missing_diagnostic",
        "unknown",
        "bad_shape",
        "bad_type",
    ],
)
def test_saved_moe_output_contract_matches_graph_and_runtime(
    runtime, mutation
):
    actor = {
        "n_layers": 1,
        "dense_layer_at_first": False,
        "top_k": 2,
        "num_fine_experts": 1024,
    }
    kv_shape = [1, 2, 1, 32, 4, 64]
    specs = {
        "actions": (onnx.TensorProto.FLOAT, [1, 29]),
        "present_key_values": (onnx.TensorProto.FLOAT, kv_shape),
        "moe_layer_0_expert_indices": (onnx.TensorProto.INT64, [1, 1, 2]),
        "moe_layer_0_expert_logits": (onnx.TensorProto.FLOAT, [1, 1, 1024]),
    }
    if mutation == "missing_core":
        del specs["actions"]
    elif mutation == "missing_diagnostic":
        del specs["moe_layer_0_expert_logits"]
    elif mutation == "unknown":
        specs["unexpected_output"] = (onnx.TensorProto.FLOAT, [1, 1])
    elif mutation == "bad_shape":
        specs["moe_layer_0_expert_logits"] = (
            onnx.TensorProto.FLOAT,
            [1, 1, 512],
        )
    elif mutation == "bad_type":
        specs["moe_layer_0_expert_indices"] = (
            onnx.TensorProto.FLOAT,
            [1, 1, 2],
        )
    nodes = _moe_output_nodes(specs, runtime=runtime)
    if mutation is None:
        _validate_v2_outputs(nodes, actor, kv_shape, runtime=runtime)
    else:
        message = (
            "names differ"
            if mutation
            in {
                "missing_core",
                "missing_diagnostic",
                "unknown",
            }
            else "type/shape"
        )
        with pytest.raises(ValueError, match=message):
            _validate_v2_outputs(nodes, actor, kv_shape, runtime=runtime)
