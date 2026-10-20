"""Train paired boxing policies from supervised weights or resume RL."""

import argparse
from dataclasses import replace
from pathlib import Path

from holoboxing.training import cli as _common
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_controller_arguments(parser)
    parser.add_argument(
        "--amp-motions",
        type=_common.existing_path,
        required=True,
        help="prepared directory from holoboxing.data prepare containing 50 Hz AMP references",
    )
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument(
        "--supervised-checkpoint",
        type=_common.existing_path,
        help="epoch_*.pt produced by holoboxing.training.train_sl",
    )
    start.add_argument("--resume", type=_common.existing_path)
    parser.add_argument("--allow-resume-code-change", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new empty training output directory",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=40000,
        help="Additional PPO iterations to run (default: 40000)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device after CUDA_VISIBLE_DEVICES remapping",
    )
    parser.add_argument("--num-envs", type=int, default=TrainConfig().num_envs)
    args = parser.parse_args(argv)
    if not (args.supervised_checkpoint or args.resume).is_file():
        parser.error("Initialization must be a checkpoint file")
    if args.allow_resume_code_change and not args.resume:
        parser.error("--allow-resume-code-change requires --resume")
    if args.updates <= 0:
        parser.error("--updates must be positive")
    config = replace(TrainConfig(), num_envs=args.num_envs)
    args.num_envs, args.seed, args.action_dim = (
        config.num_envs,
        config.seed,
        config.action_dim,
    )
    from holoboxing.training.scripts.train_mujoco import train

    train(args, config)


if __name__ == "__main__":
    main()
