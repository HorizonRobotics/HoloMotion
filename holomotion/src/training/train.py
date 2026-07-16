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

import os
import sys
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hydra.utils import get_class
from omegaconf import ListConfig, OmegaConf

from holomotion.src.utils.config import compile_config


def _resolve_mujoco_eval_onnx_names(
    exported_dir: Path, ckpt_onnx_names
) -> list[str]:
    if not exported_dir.is_dir():
        raise FileNotFoundError(
            f"Exported ONNX directory not found: {exported_dir}"
        )
    existing = sorted([p.name for p in exported_dir.glob("*.onnx")])
    if len(existing) == 0:
        raise FileNotFoundError(f"No .onnx files found under {exported_dir}")
    existing_set = set(existing)

    if ckpt_onnx_names is None:
        return existing
    if isinstance(ckpt_onnx_names, ListConfig):
        requested = list(ckpt_onnx_names)
    elif isinstance(ckpt_onnx_names, (list, tuple)):
        requested = list(ckpt_onnx_names)
    else:
        raise TypeError(
            "mujoco_eval.ckpt_onnx_names must be a list/tuple, "
            f"got {type(ckpt_onnx_names)}"
        )
    requested_norm = []
    for name in requested:
        name_str = str(name).strip()
        if name_str == "":
            continue
        requested_norm.append(Path(name_str).name)
    if len(requested_norm) == 0:
        return existing

    selected = [name for name in requested_norm if name in existing_set]
    if len(selected) == 0:
        raise ValueError(
            "No requested ONNX checkpoints exist under exported directory. "
            f"exported_dir={exported_dir}, requested={requested_norm}, "
            f"existing={existing}"
        )
    return selected


def _exec_mujoco_eval(eval_override_dict: dict) -> None:
    cli_args = []
    for key in sorted(eval_override_dict.keys()):
        value = eval_override_dict[key]
        if value is None:
            continue
        if isinstance(value, bool):
            cli_args.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            cli_args.append(f"{key}={value}")
        elif isinstance(value, str):
            cli_args.append(f"{key}={value}")
        elif isinstance(value, (list, tuple)):
            inner = ",".join([str(v) for v in value])
            cli_args.append(f"{key}=[{inner}]")
        else:
            cli_args.append(f"{key}={value}")

    argv = [
        sys.executable,
        "-m",
        "holomotion.src.evaluation.eval_mujoco_sim2sim",
    ] + cli_args
    os.execv(sys.executable, argv)


@hydra.main(
    config_path="../../config",
    config_name="training/train_base",
    version_base=None,
)
def main(config: OmegaConf):
    """Train the motion tracking model.

    Args:
        config: OmegaConf object containing the configuration.

    """

    # Accelerate initializes NCCL before it calls set_device(). Bind the local
    # CUDA device first so multi-node ranks cannot initialize on the wrong GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    config = compile_config(config, accelerator=None)
    dist = None

    # Hydra resolves ${now:...} independently in distributed processes, so
    # experiment_save_dir can differ by rank. Initialize the process group and
    # broadcast rank 0's path so every rank writes to the same directory.
    if getattr(config, "num_processes", 1) > 1:
        project_config = ProjectConfiguration(
            project_dir=config.experiment_save_dir,
            logging_dir=config.experiment_save_dir,
        )
        _accelerator = Accelerator(project_config=project_config)
        import torch.distributed as dist

        path_list = (
            [config.experiment_save_dir]
            if _accelerator.is_main_process
            else [None]
        )
        dist.broadcast_object_list(path_list, src=0)
        config.experiment_save_dir = path_list[0]

    log_dir = config.experiment_save_dir
    headless = config.headless
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
    )

    algo.load(config.checkpoint)
    algo.learn()

    if not bool(config.mujoco_eval.get("enabled", False)):
        return
    if not bool(config.algo.config.get("export_policy", False)):
        msg = (
            "mujoco_eval.enabled=true requires "
            "algo.config.export_policy=true to export ONNX "
            "before post-training evaluation."
        )
        raise ValueError(msg)

    if not bool(algo.is_main_process):
        os._exit(0)

    exported_dir = Path(log_dir) / "exported"
    selected_onnx_names = _resolve_mujoco_eval_onnx_names(
        exported_dir, config.mujoco_eval.get("ckpt_onnx_names", None)
    )
    eval_override_dict = OmegaConf.to_container(
        config.mujoco_eval, resolve=True
    )
    eval_override_dict.pop("enabled", None)
    eval_override_dict["ckpt_onnx_root_dir"] = str(exported_dir)
    eval_override_dict["ckpt_onnx_names"] = selected_onnx_names
    _exec_mujoco_eval(eval_override_dict)


if __name__ == "__main__":
    main()
