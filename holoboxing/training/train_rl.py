"""Train paired boxing policies from a supervised attacker checkpoint."""

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
    parser.add_argument(
        "--supervised-checkpoint",
        type=_common.existing_path,
        required=True,
        help="epoch_*.pt produced by holoboxing.training.train_sl",
    )
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
        help="PPO iterations to run (default: 40000)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device after CUDA_VISIBLE_DEVICES remapping",
    )
    parser.add_argument("--num-envs", type=int, default=TrainConfig().num_envs)
    args = parser.parse_args(argv)
    if not args.supervised_checkpoint.is_file():
        parser.error("--supervised-checkpoint must be a model file")
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
