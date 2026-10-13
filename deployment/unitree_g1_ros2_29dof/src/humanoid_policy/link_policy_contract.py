"""Read-only contract check for a G1 v2 link policy bundle."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import onnx
from omegaconf import OmegaConf

from holomotion.src.motion_tracking.link_actor_packing import link_actor_layout
from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from holomotion.src.motion_tracking.link_mode_mask import ModeMaskSampler
from holomotion.src.motion_tracking.link_observation import (
    DEFAULT_COMMAND_LINK_NAMES,
)

DEPLOY_MODES = {
    "Pelvis-1": ("pelvis",),
    "Keypoint-3": ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    "Keypoint-5": (
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
    "WholeBody-14": DEFAULT_COMMAND_LINK_NAMES,
}

LINK_TARGET_FPS_HZ = 50.0


def validate_v2_reference_frequency(
    source, policy_freq=None
) -> dict[str, float]:
    """Validate the saved command and actor timing before loading ONNX sessions."""
    if isinstance(source, (str, Path)):
        config = OmegaConf.load(source)
    elif OmegaConf.is_config(source):
        config = source
    else:
        config = OmegaConf.create(source)
    values = {
        "obs.target_fps": OmegaConf.select(config, "obs.target_fps"),
        "env.config.commands.ref_motion.params.target_fps": OmegaConf.select(
            config, "env.config.commands.ref_motion.params.target_fps"
        ),
    }
    if policy_freq is not None:
        values["robot.policy_freq"] = policy_freq
    checked = {}
    for name, raw in values.items():
        if raw is None or isinstance(raw, bool):
            raise ValueError(
                f"V2 frequency is missing or invalid: {name}={raw!r}"
            )
        try:
            frequency = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"V2 frequency is invalid: {name}={raw!r}"
            ) from exc
        if not math.isfinite(frequency) or frequency != LINK_TARGET_FPS_HZ:
            raise ValueError(
                f"V2 frequency must be {LINK_TARGET_FPS_HZ:g} Hz: "
                f"{name}={raw!r}"
            )
        checked[name] = frequency
    return checked


def resolve_v2_link_config(source) -> dict:
    """Resolve only deployment fields, without invoking training-only resolvers."""
    if isinstance(source, (str, Path)):
        config = OmegaConf.load(source)
    else:
        config = OmegaConf.create(
            OmegaConf.to_container(source, resolve=False)
        )
    raw = OmegaConf.to_container(config, resolve=False)
    for field, offsets in (
        ("n_fut_frames", "frame_offsets"),
        ("critic_n_fut_frames", "critic_frame_offsets"),
    ):
        expected = "${eval:'len(${link_observation." + offsets + "})'}"
        if raw["obs"].get(field) == expected:
            OmegaConf.update(
                config,
                "obs." + field,
                len(config.link_observation[offsets]),
                merge=False,
            )
    return {
        "modules": {
            "actor": OmegaConf.to_container(
                OmegaConf.select(config, "modules.actor"), resolve=True
            )
        },
        **{
            key: OmegaConf.to_container(
                OmegaConf.select(config, key), resolve=True
            )
            for key in ("link_observation", "robot", "obs")
        },
    }


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _names(values, label, count):
    result = tuple(values)
    if len(result) != count or len(set(result)) != count:
        raise ValueError(f"{label} must contain {count} distinct names")
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain nonempty strings")
    return result


def compile_control_modes(link_config: dict, mode_record: dict) -> dict:
    """Validate saved masks and expose the canonical deployment mode names."""
    link_config = _mapping(link_config, "link_observation")
    mode_record = _mapping(mode_record, "link_command_modes")
    if link_config.get("schema_version") != "g1_link_xz6d_v1":
        raise ValueError("Unsupported link observation schema")
    links = _names(link_config["link_names"], "link_names", 14)
    if links != DEFAULT_COMMAND_LINK_NAMES:
        raise ValueError("Link order differs from the G1 v2 deployment schema")
    offsets = tuple(link_config["frame_offsets"])
    if not offsets or offsets != tuple(range(1, len(offsets) + 1)):
        raise ValueError(
            "V2 deployment requires contiguous future frame offsets"
        )
    mask_config = _mapping(link_config["mask"], "link_observation.mask")
    if mask_config.get("strategy") != "modes":
        raise ValueError("The deployed model must have named control modes")
    _mapping(mask_config["mode_candidates"], "mode_candidates")
    sampler = ModeMaskSampler(links, mask_config, device="cpu")
    mode_names = sampler.mode_names
    if tuple(mode_record.get("link_names", ())) != links:
        raise ValueError("Saved mode table link order differs from config")
    if tuple(mode_record.get("mode_names", ())) != mode_names:
        raise ValueError("Saved mode names/order differ from config")
    rows = mode_record.get("mode_table")
    if not isinstance(rows, list) or len(rows) != len(mode_names):
        raise ValueError("Saved mode table has the wrong number of rows")
    compiled = {}
    deploy_modes_by_links = {
        frozenset(selected): name for name, selected in DEPLOY_MODES.items()
    }
    for name, row, expected in zip(
        mode_names, rows, sampler.table.tolist(), strict=True
    ):
        if (
            not isinstance(row, list)
            or len(row) != len(links)
            or any(type(v) not in (int, float) or v not in (0, 1) for v in row)
        ):
            raise ValueError(f"Saved mode {name} is not a 14-bit binary mask")
        if tuple(row) != tuple(int(value) for value in expected):
            raise ValueError(f"Saved mode {name} differs from config")
        selected = frozenset(
            link for link, value in zip(links, expected, strict=True) if value
        )
        # Saved bundles may use different labels for the same control points.
        # Validate their original table before resolving the public mode name.
        deploy_name = (
            name
            if name in DEPLOY_MODES
            else deploy_modes_by_links.get(selected)
        )
        if deploy_name is not None:
            if selected != frozenset(DEPLOY_MODES[deploy_name]):
                raise ValueError(
                    f"Deploy mode {name} selects unexpected links"
                )
            compiled[deploy_name] = tuple(int(value) for value in expected)
    if set(compiled) != set(DEPLOY_MODES):
        raise ValueError("Model is missing one or more required deploy modes")
    if tuple(mode_record.get("actor_frame_offsets", ())) != tuple(
        link_config["frame_offsets"]
    ):
        raise ValueError("Saved actor frame offsets differ from config")
    return compiled


def _tensor_info(value):
    tensor = value.type.tensor_type
    shape = tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in tensor.shape.dim
    )
    return tensor.elem_type, shape


def _require_tensor(nodes, name, dtype, shape):
    if name not in nodes:
        raise ValueError(f"ONNX is missing {name}")
    actual = _tensor_info(nodes[name])
    if actual != (dtype, tuple(shape)):
        raise ValueError(
            f"ONNX {name} is {actual}, expected {(dtype, tuple(shape))}"
        )


def _metadata_floats(metadata, name, count):
    try:
        values = tuple(float(value) for value in metadata[name].split(","))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid ONNX metadata {name}") from exc
    if len(values) != count or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError(
            f"ONNX metadata {name} must contain {count} finite values"
        )
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_robot_fk_asset(robot: dict) -> None:
    asset = _mapping(robot["asset"], "robot.asset")
    motion = _mapping(robot["motion"], "robot.motion")
    if motion["fk_robot_file_path"] != asset["urdf_file"]:
        raise ValueError("Motion FK asset differs from robot URDF asset")


def _expected_v2_output_schema(actor: dict, kv_shape) -> dict:
    """Mirror the saved actor's ONNX routing-debug export contract."""
    n_layers = int(actor["n_layers"])
    top_k = int(actor["top_k"])
    num_experts = int(actor["num_fine_experts"])
    if n_layers < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("Invalid saved v2 MoE actor dimensions")
    outputs = {
        "actions": (onnx.TensorProto.FLOAT, (1, 29)),
        "present_key_values": (onnx.TensorProto.FLOAT, tuple(kv_shape)),
    }
    dense_first = bool(actor.get("dense_layer_at_first", True))
    dense_last = bool(actor.get("dense_layer_at_last", False))
    for layer_idx in range(n_layers):
        if (dense_first and layer_idx == 0) or (
            dense_last and layer_idx == n_layers - 1
        ):
            continue
        prefix = f"moe_layer_{layer_idx}"
        outputs[f"{prefix}_expert_indices"] = (
            onnx.TensorProto.INT64,
            (1, 1, top_k),
        )
        outputs[f"{prefix}_expert_logits"] = (
            onnx.TensorProto.FLOAT,
            (1, 1, num_experts),
        )
    return outputs


def _validate_v2_outputs(
    nodes, actor: dict, kv_shape, *, runtime: bool
) -> None:
    expected = _expected_v2_output_schema(actor, kv_shape)
    names = [node.name for node in nodes]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(
            "V2 ONNX output names differ from saved actor contract: "
            f"actual={sorted(names)}, expected={sorted(expected)}"
        )
    runtime_dtypes = {
        "tensor(float)": onnx.TensorProto.FLOAT,
        "tensor(int64)": onnx.TensorProto.INT64,
    }
    for node in nodes:
        actual = (
            (runtime_dtypes.get(node.type), tuple(node.shape))
            if runtime
            else _tensor_info(node)
        )
        if actual != expected[node.name]:
            raise ValueError(
                f"V2 ONNX output {node.name} has type/shape {actual}, "
                f"expected {expected[node.name]}"
            )


def _validate_v2_runtime_session(session, actor: dict, actor_dim: int) -> None:
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"obs", "past_key_values", "step_idx"}:
        raise ValueError(
            "V2 ONNX input names differ from saved actor contract"
        )

    def require(node, dtype, shape):
        if node.type != dtype or tuple(node.shape) != tuple(shape):
            raise ValueError(f"V2 ONNX {node.name} has wrong type or shape")

    require(inputs["obs"], "tensor(float)", (1, actor_dim))
    require(inputs["step_idx"], "tensor(int64)", (1,))
    kv = inputs["past_key_values"]
    if (
        kv.type != "tensor(float)"
        or len(kv.shape) != 6
        or kv.shape[3] != actor["max_ctx_len"]
    ):
        raise ValueError("V2 ONNX KV input disagrees with actor context")
    _validate_v2_outputs(session.get_outputs(), actor, kv.shape, runtime=True)
    metadata = session.get_modelmeta().custom_metadata_map
    for name in (
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "action_scale",
    ):
        _metadata_floats(metadata, name, 29)
    if int(metadata["rope_max_seq_len"]) != int(actor["rope_max_seq_len"]):
        raise ValueError(
            "V2 ONNX RoPE metadata differs from actor configuration"
        )


def build_runtime_link_contract(
    *,
    config: dict,
    mode_record: dict,
    asset_path: Path,
    session,
    onnx_joint_names,
    default_joint_pos_onnx,
    obs_shape,
    actions_shape,
) -> dict:
    """Compile the v2 runtime contract from an already-loaded ONNX session."""
    links = _mapping(config["link_observation"], "link_observation")
    modes = compile_control_modes(links, mode_record)
    actor = _mapping(config["modules"]["actor"], "modules.actor")
    if actor.get("type") != "ReferenceRoutedGroupedMoETransformerPolicy":
        raise ValueError("Unsupported v2 link actor type")
    layout = link_actor_layout(links, actor["obs_schema"])
    if not layout["include_error_terms"] or layout["include_future_link_mask"]:
        raise ValueError("Unsupported deployment link feature layout")
    if config["obs"]["context_length"] != 1:
        raise ValueError("Deployment requires one current actor frame")
    if config["obs"]["n_fut_frames"] != len(links["frame_offsets"]):
        raise ValueError("Reference horizon differs from frame offsets")
    robot = _mapping(config["robot"], "robot")
    _validate_robot_fk_asset(robot)
    if robot["actions_dim"] != 29:
        raise ValueError("G1 v2 actor must output 29 actions")
    config_joints = _names(robot["dof_names"], "robot.dof_names", 29)
    exported_joints = _names(onnx_joint_names, "ONNX joint_names", 29)
    if set(config_joints) != set(exported_joints):
        raise ValueError("ONNX action joints differ from config robot joints")
    defaults = tuple(float(value) for value in default_joint_pos_onnx)
    if len(defaults) != 29 or not all(math.isfinite(v) for v in defaults):
        raise ValueError("ONNX default_joint_pos must have 29 finite values")
    if tuple(obs_shape) != (1, layout["actor_dim"]):
        raise ValueError(
            "ONNX observation shape differs from link actor layout"
        )
    if tuple(actions_shape) != (1, 29):
        raise ValueError("ONNX action shape must be [1,29]")
    _validate_v2_runtime_session(session, actor, layout["actor_dim"])
    asset_path = Path(asset_path)
    configured_asset = Path(robot["asset"]["urdf_file"])
    if (
        configured_asset.is_absolute()
        or tuple(asset_path.parts[-len(configured_asset.parts) :])
        != configured_asset.parts
    ):
        raise ValueError("FK URDF path differs from saved robot asset")
    return {
        "schema": links["schema_version"],
        "actor_type": actor["type"],
        "actor_dim": layout["actor_dim"],
        "include_error_terms": layout["include_error_terms"],
        "include_future_link_mask": layout["include_future_link_mask"],
        "action_dim": 29,
        "link_names": list(links["link_names"]),
        "frame_offsets": list(links["frame_offsets"]),
        "modes": {name: list(mask) for name, mask in modes.items()},
        "onnx_joint_names": list(exported_joints),
        "config_joint_names": list(config_joints),
        "default_joint_pos_onnx": list(defaults),
        "onnx_index_for_config_joint": [
            exported_joints.index(name) for name in config_joints
        ],
        "asset_sha256": _sha256_file(asset_path),
    }


def inspect_v2_link_bundle(
    *,
    config_path: Path,
    onnx_path: Path,
    mode_table_path: Path,
    asset_path: Path,
    policy_freq=None,
) -> dict:
    """Validate files without ROS startup, inference, or action publication."""
    config_path = Path(config_path)
    onnx_path = Path(onnx_path)
    mode_table_path = Path(mode_table_path)
    asset_path = Path(asset_path)
    bundle_root = config_path.resolve().parent
    onnx_dir = onnx_path.resolve().parent
    if mode_table_path.resolve().parent != bundle_root or onnx_dir not in (
        bundle_root,
        bundle_root / "exported",
    ):
        raise ValueError("V2 config, mode table and ONNX are not co-bundled")
    frequency = validate_v2_reference_frequency(config_path, policy_freq)
    config = resolve_v2_link_config(config_path)
    mode_record = _mapping(
        json.loads(mode_table_path.read_text()), "mode table"
    )
    links = _mapping(config["link_observation"], "link_observation")
    modes = compile_control_modes(links, mode_record)
    actor = _mapping(config["modules"]["actor"], "modules.actor")
    if actor.get("type") != "ReferenceRoutedGroupedMoETransformerPolicy":
        raise ValueError(
            "This deployment preflight requires the v2 link actor"
        )
    layout = link_actor_layout(links, actor["obs_schema"])
    if not layout["include_error_terms"] or layout["include_future_link_mask"]:
        raise ValueError(
            "Deployment builder requires error terms and no future mask"
        )
    if config["obs"]["context_length"] != 1:
        raise ValueError("Link deployment requires one current actor frame")
    if config["obs"]["n_fut_frames"] != len(links["frame_offsets"]):
        raise ValueError("Actor future frame count differs from link offsets")
    robot = _mapping(config["robot"], "robot")
    _validate_robot_fk_asset(robot)
    if robot["actions_dim"] != 29:
        raise ValueError("G1 v2 actor must output 29 actions")
    asset_record = _mapping(robot["asset"], "robot.asset")
    configured_asset = Path(asset_record["urdf_file"])
    if configured_asset.is_absolute() or not configured_asset.parts:
        raise ValueError("robot.asset.urdf_file must be a relative asset path")
    if (
        tuple(asset_path.parts[-len(configured_asset.parts) :])
        != configured_asset.parts
    ):
        raise ValueError(
            "Selected URDF path differs from saved robot.asset.urdf_file"
        )
    config_joints = _names(robot["dof_names"], "robot.dof_names", 29)
    fk = SharedLinkFK(str(asset_path), device="cpu", joint_names=config_joints)
    if fk.num_dof != 29 or not set(links["link_names"]).issubset(
        fk.body_names
    ):
        raise ValueError("Robot asset lacks the configured G1 joints/links")

    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = {node.name: node for node in model.graph.input}
    if set(inputs) != {"obs", "past_key_values", "step_idx"}:
        raise ValueError(f"Unexpected ONNX inputs: {sorted(inputs)}")
    _require_tensor(
        inputs, "obs", onnx.TensorProto.FLOAT, (1, layout["actor_dim"])
    )
    _require_tensor(inputs, "step_idx", onnx.TensorProto.INT64, (1,))
    kv_dtype, kv_shape = _tensor_info(inputs["past_key_values"])
    if (
        kv_dtype != onnx.TensorProto.FLOAT
        or len(kv_shape) != 6
        or kv_shape[3] != actor["max_ctx_len"]
    ):
        raise ValueError("ONNX KV cache disagrees with actor configuration")
    _validate_v2_outputs(model.graph.output, actor, kv_shape, runtime=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    exported_joints = _names(
        metadata["joint_names"].split(","), "ONNX joint_names", 29
    )
    if set(exported_joints) != set(config_joints):
        raise ValueError("ONNX action joints differ from config robot joints")
    for name in (
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "action_scale",
    ):
        _metadata_floats(metadata, name, 29)
    if int(metadata["rope_max_seq_len"]) != int(actor["rope_max_seq_len"]):
        raise ValueError("ONNX RoPE length differs from actor configuration")
    return {
        "frequency_hz": frequency,
        "schema": links["schema_version"],
        "actor_type": actor["type"],
        "actor_dim": layout["actor_dim"],
        "include_error_terms": layout["include_error_terms"],
        "include_future_link_mask": layout["include_future_link_mask"],
        "action_dim": 29,
        "link_names": list(links["link_names"]),
        "frame_offsets": list(links["frame_offsets"]),
        "reference_horizon": layout["reference_horizon"],
        "modes": {name: list(mask) for name, mask in modes.items()},
        "onnx_joint_names": list(exported_joints),
        "config_joint_names": list(config_joints),
        "default_joint_pos_onnx": list(
            _metadata_floats(metadata, "default_joint_pos", 29)
        ),
        "onnx_index_for_config_joint": [
            exported_joints.index(name) for name in config_joints
        ],
        "asset_sha256": _sha256_file(asset_path),
        "config_sha256": _sha256_file(config_path),
        "mode_table_sha256": _sha256_file(mode_table_path),
        "onnx_sha256": _sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_opset": max(
            item.version for item in model.opset_import if item.domain == ""
        ),
        "checked_without_robot_action": True,
    }
