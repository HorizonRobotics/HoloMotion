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


import inspect
import re
from pathlib import Path

from loguru import logger


def _list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata_holomotion(
    env, onnx_path: str, actor_module=None
) -> None:
    import onnx

    robot_data = env.scene["robot"].data
    action_term = env.action_manager.get_term("dof_pos")
    simulator_joint_names = list(robot_data.joint_names)
    action_joint_names = list(action_term._joint_names)
    if len(action_joint_names) != len(simulator_joint_names) or set(
        action_joint_names
    ) != set(simulator_joint_names):
        raise ValueError(
            "The exported action term must cover every simulator joint "
            "exactly once. Got action joints "
            f"{action_joint_names} and simulator joints "
            f"{simulator_joint_names}."
        )

    simulator_index = {
        name: index for index, name in enumerate(simulator_joint_names)
    }
    action_from_simulator = [
        simulator_index[name] for name in action_joint_names
    ]

    def _robot_values_in_action_order(values) -> list:
        simulator_values = values[0].cpu().tolist()
        return [simulator_values[index] for index in action_from_simulator]

    action_scale = action_term._scale
    if isinstance(action_scale, (int, float)):
        action_scale_values = [float(action_scale)] * len(action_joint_names)
    else:
        action_scale_values = action_scale[0].cpu().tolist()
    if len(action_scale_values) != len(action_joint_names):
        raise ValueError(
            "Action scale size does not match exported action order: "
            f"{len(action_scale_values)} vs {len(action_joint_names)}."
        )

    metadata = {
        # The actor output, action scale/offset, and deployment metadata must
        # share one order. In v1.4 this is the Isaac articulation order.
        "joint_names": action_joint_names,
        "joint_stiffness": _robot_values_in_action_order(
            robot_data.default_joint_stiffness
        ),
        "joint_damping": _robot_values_in_action_order(
            robot_data.default_joint_damping
        ),
        "default_joint_pos": _robot_values_in_action_order(
            robot_data.default_joint_pos
        ),
        "action_scale": action_scale_values,
    }
    if actor_module is not None:
        rope_max_seq_len = getattr(actor_module, "rope_max_seq_len", None)
        if rope_max_seq_len is not None:
            metadata["rope_max_seq_len"] = int(rope_max_seq_len)

    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = (
            _list_to_csv_str(value) if isinstance(value, list) else str(value)
        )
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


def export_policy_to_onnx(
    algo,
    checkpoint_path: str,
    *,
    onnx_name_suffix: str | None = None,
    use_kv_cache: bool = True,
) -> str:
    checkpoint = Path(checkpoint_path)
    export_dir = checkpoint.parent / "exported"
    export_dir.mkdir(exist_ok=True)

    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    if onnx_name_suffix is not None:
        suffix = re.sub(r"[\s+]", "_", str(onnx_name_suffix))
        onnx_name = onnx_name.replace(".onnx", f"_{suffix}.onnx")
    onnx_path = export_dir / onnx_name

    logger.info("Starting ONNX minimal policy export (actions-only)...")
    actor_was_training = getattr(algo.actor, "training", None)
    critic_was_training = getattr(algo.critic, "training", None)
    algo.actor.eval()
    algo.critic.eval()

    try:
        actor_for_export = algo.accelerator.unwrap_model(algo.actor)
        orig_mod = getattr(actor_for_export, "_orig_mod", None)
        if orig_mod is not None:
            actor_for_export = orig_mod

        export_signature = inspect.signature(actor_for_export.export_onnx)
        export_kwargs = {"onnx_path": onnx_path, "opset_version": 17}
        if "use_kv_cache" in export_signature.parameters:
            export_kwargs["use_kv_cache"] = bool(use_kv_cache)

        onnx_path_str = actor_for_export.export_onnx(**export_kwargs)
        metadata_signature = inspect.signature(attach_onnx_metadata_holomotion)
        metadata_kwargs = {"onnx_path": onnx_path_str}
        if "actor_module" in metadata_signature.parameters:
            metadata_kwargs["actor_module"] = getattr(
                actor_for_export, "actor_module", actor_for_export
            )
        attach_onnx_metadata_holomotion(algo.env._env, **metadata_kwargs)
        logger.info(
            f"Successfully exported minimal policy to: {onnx_path_str}"
        )
        return onnx_path_str
    finally:
        if actor_was_training is not None:
            algo.actor.train(actor_was_training)
        if critic_was_training is not None:
            algo.critic.train(critic_was_training)
