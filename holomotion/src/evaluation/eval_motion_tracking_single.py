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
import re
from pathlib import Path

import hydra
from hydra.utils import get_class
from loguru import logger
from omegaconf import ListConfig, OmegaConf

from holomotion.src.evaluation.metrics import run_evaluation
from holomotion.src.utils.config import compile_config
from holomotion.src.utils.onnx_export import export_policy_to_onnx


def load_training_config(
    checkpoint_path: str, eval_config: OmegaConf
) -> OmegaConf:
    """Load training config from checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint file.
        eval_config: Full evaluation config (including command line overrides).

    Returns:
        Merged config with training config as base.
    """
    checkpoint = Path(checkpoint_path)
    config_path = checkpoint.parent / "config.yaml"

    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            logger.warning(
                f"Training config not found at {config_path}, using evaluation config"
            )
            return eval_config

    logger.info(f"Loading training config from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Apply eval_overrides from training config if they exist
    if train_config.get("eval_overrides") is not None:
        train_config = OmegaConf.merge(
            train_config, train_config.eval_overrides
        )

    # Set checkpoint path
    train_config.checkpoint = checkpoint_path
    train_config.algo.config.checkpoint = checkpoint_path

    # For evaluation, merge eval_config into train_config
    config = OmegaConf.merge(train_config, eval_config)

    # force set the terminations and domain rand with eval_config's
    config.env.config.terminations = eval_config.env.config.terminations
    config.env.config.domain_rand = eval_config.env.config.domain_rand
    obs_groups = config.env.config.obs.obs_groups
    if "policy" in obs_groups:
        obs_groups.policy.enable_corruption = False
    if "critic" in obs_groups:
        obs_groups.critic.enable_corruption = False
    if "unified" in obs_groups:
        obs_groups.unified.enable_corruption = False

    return config


def _infer_dataset_suffix(output_dir: str, checkpoint_path: str) -> str:
    output_name = Path(output_dir).name
    model_name = Path(checkpoint_path).stem
    expected_prefix = f"isaaclab_eval_output_{model_name}_"
    if output_name.startswith(expected_prefix):
        return output_name[len(expected_prefix) :]
    return output_name


def _checkpoint_sort_key(checkpoint_path: Path):
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    if match is not None:
        return (0, int(match.group(1)), checkpoint_path.name)
    return (1, checkpoint_path.name)


def _normalize_ckpt_pt_names(ckpt_pt_names) -> list[str]:
    if ckpt_pt_names is None:
        return []

    if isinstance(ckpt_pt_names, ListConfig):
        raw_names = list(ckpt_pt_names)
    elif isinstance(ckpt_pt_names, (list, tuple)):
        raw_names = list(ckpt_pt_names)
    else:
        raise TypeError(
            f"ckpt_pt_names must be a list/tuple, got {type(ckpt_pt_names)}"
        )

    normalized_names = []
    for name in raw_names:
        name_str = str(name).strip()
        if name_str == "":
            continue
        if not name_str.endswith(".pt"):
            name_str = f"{name_str}.pt"
        normalized_names.append(name_str)
    return normalized_names


def _resolve_export_ckpt_paths(config: OmegaConf) -> list[Path]:
    log_dir_value = config.get("log_dir", None)
    checkpoint_value = config.get("checkpoint", None)

    if log_dir_value is None or str(log_dir_value).strip() == "":
        if checkpoint_value is None or str(checkpoint_value).strip() == "":
            raise ValueError(
                "When export_only=true, set log_dir or checkpoint."
            )
        log_dir = Path(str(checkpoint_value)).parent
    else:
        log_dir = Path(str(log_dir_value))

    if not log_dir.is_dir():
        raise NotADirectoryError(
            f"log_dir does not exist or is not a directory: {log_dir}"
        )

    ckpt_pt_names = _normalize_ckpt_pt_names(config.get("ckpt_pt_names", None))
    if len(ckpt_pt_names) > 0:
        selected_paths = []
        missing_names = []
        for name in ckpt_pt_names:
            ckpt_path = log_dir / name
            if ckpt_path.is_file():
                selected_paths.append(ckpt_path)
            else:
                missing_names.append(name)

        if len(missing_names) > 0:
            raise FileNotFoundError(
                f"Missing checkpoints in log_dir={log_dir}: {missing_names}"
            )
        return selected_paths

    discovered_paths = sorted(log_dir.glob("*.pt"), key=_checkpoint_sort_key)
    if len(discovered_paths) == 0:
        raise FileNotFoundError(
            f"No .pt checkpoints found in log_dir={log_dir}"
        )
    return discovered_paths


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the motion tracking model.

    Args:
        config: OmegaConf object containing the evaluation configuration.

    """
    export_only = bool(config.get("export_only", False))
    if export_only:
        checkpoint_paths = _resolve_export_ckpt_paths(config)
        config = load_training_config(str(checkpoint_paths[0]), config)
    else:
        if config.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation")
        checkpoint_paths = [Path(str(config.checkpoint))]
        config = load_training_config(config.checkpoint, config)

    # Compile config without accelerator (PPO will create it)
    config = compile_config(config, accelerator=None)

    # Use checkpoint directory as log_dir for offline evaluation/export.
    log_dir = str(checkpoint_paths[0].parent)
    headless = config.headless

    # PPO creates Accelerator, AppLauncher, and environment internally
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
        is_offline_eval=True,
    )

    if (
        algo.accelerator.is_main_process
        and os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1"
    ):
        logger.info(
            "Tip: If you encounter Triton/compilation errors during evaluation,"
        )
        logger.info(
            "     set environment variable: export TORCH_COMPILE_DISABLE=1"
        )

    if algo.accelerator.is_main_process:
        with open(os.path.join(log_dir, "eval_config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    if export_only:
        if algo.accelerator.is_main_process:
            logger.info(
                "Running export-only mode for "
                f"{len(checkpoint_paths)} checkpoints in {log_dir}"
            )
        onnx_name_suffix = config.get("onnx_name_suffix", None)
        use_kv_cache = config.get("use_kv_cache", True)
        for i, checkpoint_path in enumerate(checkpoint_paths, start=1):
            ckpt_path = str(checkpoint_path)
            if algo.accelerator.is_main_process:
                logger.info(
                    f"[{i}/{len(checkpoint_paths)}] Loading checkpoint: "
                    f"{ckpt_path}"
                )
            algo.load(ckpt_path)
            if algo.accelerator.is_main_process:
                onnx_path = export_policy_to_onnx(
                    algo,
                    ckpt_path,
                    onnx_name_suffix=onnx_name_suffix,
                    use_kv_cache=use_kv_cache,
                )
                logger.info(f"Successfully exported policy to: {onnx_path}")
            algo.accelerator.wait_for_everyone()
        if algo.accelerator.is_main_process:
            logger.info("Export-only mode completed successfully!")
        return

    if algo.accelerator.is_main_process:
        logger.info(f"Loading checkpoint for evaluation: {config.checkpoint}")
    algo.load(config.checkpoint)

    command_name = list(config.env.config.commands.keys())[0]
    if command_name == "ref_motion":
        motion_cmd = algo.env._env.command_manager.get_term("ref_motion")
        algo.env._env.reset()
        motion_cmd._update_ref_motion_state()

    # Export ONNX if requested
    if config.get("export_policy", True):
        if algo.accelerator.is_main_process:
            onnx_name_suffix = config.get("onnx_name_suffix", None)
            onnx_path = export_policy_to_onnx(
                algo,
                config.checkpoint,
                onnx_name_suffix=onnx_name_suffix,
                use_kv_cache=config.get("use_kv_cache", True),
            )
            logger.info(f"Successfully exported policy to: {onnx_path}")
        algo.accelerator.wait_for_everyone()

    calc_per_clip_metrics = bool(config.get("calc_per_clip_metrics", False))
    generate_report = bool(config.get("generate_report", False))
    dump_npzs = bool(config.get("dump_npzs", False)) or calc_per_clip_metrics
    dof_mode = config.get("dof_mode", "29")
    if (
        calc_per_clip_metrics
        and not bool(config.get("dump_npzs", False))
        and algo.accelerator.is_main_process
    ):
        logger.info(
            "calc_per_clip_metrics=true requires dumped NPZs; "
            "enabling dump_npzs automatically."
        )

    result = algo.offline_evaluate_policy(dump_npzs)
    algo.accelerator.wait_for_everyone()

    if algo.accelerator.is_main_process:
        logger.info("Evaluation completed successfully!")
        output_dir = (
            result.get("output_dir") if isinstance(result, dict) else None
        )
        if output_dir is not None:
            logger.info(f"NPZs saved to: {output_dir}")

        if calc_per_clip_metrics:
            if output_dir is None:
                logger.warning(
                    "Skipping per-clip metric calculation because "
                    "output_dir is unavailable."
                )
            else:
                dataset_suffix = _infer_dataset_suffix(
                    output_dir, config.checkpoint
                )
                run_evaluation(
                    npz_dir=output_dir,
                    dataset_suffix=dataset_suffix,
                    failure_pos_err_thresh_m=0.25,
                    dof_mode=dof_mode,
                )
                logger.info(
                    f"Finished per-clip metric calculation for: {output_dir}"
                )

        if generate_report:
            if output_dir is None:
                logger.warning(
                    "Skipping report generation because output_dir is unavailable."
                )
            else:
                from holomotion.scripts.evaluation import (
                    mean_process_5metrics,
                )

                report_path = mean_process_5metrics.generate_macro_mean_report_from_json_dir(
                    output_dir
                )
                logger.info(f"Generated metrics report at: {report_path}")


if __name__ == "__main__":
    main()
