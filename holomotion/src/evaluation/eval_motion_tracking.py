import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger


def find_checkpoints_to_evaluate(
    eval_h5_dataset_path: str,
    root_dir: Path,
    target_checkpoints: Optional[List[str]],
    config_name: str,
) -> List[Tuple[str, str]]:
    """Scan all model subdirectories and collect checkpoints that need evaluation.

    Behavior:
        - If `target_checkpoints` is provided and non-empty:
            only these checkpoint stems are considered (e.g. ['model_17500']).
        - If `target_checkpoints` is None or empty:
            all checkpoints matching 'model_*.pt' under each model directory will be considered.
    Returns:
        A list of (checkpoint_path, config_name) tuples to be evaluated.
    """
    checkpoints_to_evaluate: List[Tuple[str, str]] = []
    dataset_path = Path(eval_h5_dataset_path)
    dataset_suffix = (
        dataset_path.name if dataset_path.name else "dataset_unknown"
    )

    if root_dir.is_file():
        checkpoint_file = root_dir

        model_dir_path = checkpoint_file.parent
        checkpoint_stem = checkpoint_file.stem

        eval_out_dir = (
            model_dir_path
            / f"isaaclab_eval_output_{checkpoint_stem}_{dataset_suffix}"
        )

        cfg_name = f"evaluation/{config_name}"
        return [(str(checkpoint_file), cfg_name)]

    if not root_dir.is_dir():
        logger.error(
            f"Checkpoint root directory '{root_dir}' does not exist or is not a directory."
        )
        return []

    if target_checkpoints:
        logger.info(
            f"Searching for explicit target checkpoints: {target_checkpoints}"
        )

    # Iterate over each model directory directly under root_dir
    for model_dir_path in root_dir.iterdir():
        if not model_dir_path.is_dir():
            continue

        if target_checkpoints:
            # Use only the requested checkpoint stems
            candidate_files = [
                model_dir_path / f"{stem}.pt" for stem in target_checkpoints
            ]
        else:
            candidate_files = sorted(model_dir_path.glob("model_*.pt"))

        if not candidate_files:
            continue

        for checkpoint_file in candidate_files:
            if not checkpoint_file.is_file():
                logger.debug(f"Target checkpoint not found: {checkpoint_file}")
                continue

            checkpoint_stem = checkpoint_file.stem
            eval_out_dir = (
                model_dir_path
                / f"isaaclab_eval_output_{checkpoint_stem}_{dataset_suffix}"
            )
            if eval_out_dir.is_dir():
                logger.debug(
                    f"Skipping {checkpoint_file.name}, output exists."
                )
                continue

            # Construct Hydra config name from the folder name
            cfg_name = f"evaluation/{config_name}"
            checkpoints_to_evaluate.append((str(checkpoint_file), cfg_name))

    checkpoints_to_evaluate.sort(key=lambda x: x[0])
    return checkpoints_to_evaluate


def main(
    checkpoint_dir: str,
    target_checkpoints: Optional[List[str]],
    eval_h5_dataset_path: str,
    config_name: str,
    num_envs: str,
) -> None:
    """
    Entry point for batch evaluation.

    Args:
        checkpoint_root_dir: Root directory containing subdirectories for models.
        target_checkpoints: Optional list of checkpoint stems to evaluate
        single_eval_script: Path to the shell script to run a single evaluation.
    """
    root_path = Path(checkpoint_dir)

    checkpoints_to_evaluate = find_checkpoints_to_evaluate(
        eval_h5_dataset_path=eval_h5_dataset_path,
        root_dir=root_path,
        target_checkpoints=target_checkpoints,
        config_name=config_name,
    )

    if not checkpoints_to_evaluate:
        logger.warning(
            f"No pending evaluations found under '{checkpoint_dir}'."
        )
        return

    logger.info(
        f"Found {len(checkpoints_to_evaluate)} checkpoints to evaluate."
    )

    for i, (ckpt_path, cfg_name) in enumerate(
        checkpoints_to_evaluate, start=1
    ):
        logger.info(
            f"[{i}/{len(checkpoints_to_evaluate)}] Evaluating: {cfg_name}/{ckpt_path}"
        )

        command = [
            "bash",
            "holomotion/scripts/evaluation/eval_motion_tracking_single.sh",
            ckpt_path,
            cfg_name,
            eval_h5_dataset_path,
            num_envs,
        ]
        subprocess.run(
            command,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the batch evaluation script."""
    parser = argparse.ArgumentParser(description="motion-tracking evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--target_checkpoints", type=str, nargs="*", default=None
    )
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--eval_h5_dataset_path", type=str, required=True)
    parser.add_argument("--num_envs", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        checkpoint_dir=args.checkpoint_dir,
        target_checkpoints=args.target_checkpoints,
        eval_h5_dataset_path=args.eval_h5_dataset_path,
        config_name=args.config_name,
        num_envs=args.num_envs,
    )
