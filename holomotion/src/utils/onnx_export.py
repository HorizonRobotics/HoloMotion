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


def attach_onnx_metadata_holomotion(env, onnx_path: str) -> None:
    import onnx

    metadata = {
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"]
        .data.default_joint_stiffness[0]
        .cpu()
        .tolist(),
        "joint_damping": env.scene["robot"]
        .data.default_joint_damping[0]
        .cpu()
        .tolist(),
        "default_joint_pos": env.scene["robot"]
        .data.default_joint_pos[0]
        .cpu()
        .tolist(),
        "action_scale": env.action_manager.get_term("dof_pos")
        ._scale[0]
        .cpu()
        .tolist(),
    }

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
        attach_onnx_metadata_holomotion(algo.env._env, onnx_path=onnx_path_str)
        logger.info(
            f"Successfully exported minimal policy to: {onnx_path_str}"
        )
        return onnx_path_str
    finally:
        if actor_was_training is not None:
            algo.actor.train(actor_was_training)
        if critic_was_training is not None:
            algo.critic.train(critic_was_training)
