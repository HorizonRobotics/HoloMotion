"""Train or resume the boxing planner on competition physics."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from holoboxing.training import cli as _common
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig
from holoboxing.training.time_budget import (
    add_budget_arguments,
    budget_from_arguments,
)


def main(argv=None):
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_controller_arguments(parser)
    add_budget_arguments(parser)
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
    start.add_argument(
        "--policy-checkpoint",
        type=_common.existing_path,
        help="RL actor initialization for a new stage; resets learner state",
    )
    start.add_argument(
        "--prior-checkpoint",
        type=_common.existing_path,
        help="start latent RL without imitation",
    )
    parser.add_argument(
        "--config",
        type=_common.existing_path,
        help="experiment-specific TrainConfig JSON",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--training-mode",
        choices=("paired", "standing", "selfplay"),
        default="paired",
        help="paired learning, standing targets, or historical opponents",
    )
    parser.add_argument("--opponent-pool-capacity", type=int, default=8)
    parser.add_argument("--opponent-refresh-updates", type=int, default=256)
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
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args(argv)
    if not (
        args.supervised_checkpoint
        or args.resume
        or args.prior_checkpoint
        or args.policy_checkpoint
    ).is_file():
        parser.error("Initialization must be a checkpoint file")
    if args.allow_resume_code_change and not args.resume:
        parser.error("--allow-resume-code-change requires --resume")
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.opponent_pool_capacity < 2 or args.opponent_refresh_updates < 1:
        parser.error(
            "Opponent pool needs >=2 slots and a positive refresh interval"
        )
    try:
        budget = budget_from_arguments(args, started_at=started_at)
    except ValueError as error:
        parser.error(str(error))
    config = (
        TrainConfig(**json.loads(args.config.read_text()))
        if args.config
        else TrainConfig()
    )
    changes = {
        k: v
        for k, v in {"num_envs": args.num_envs, "seed": args.seed}.items()
        if v is not None
    }
    config = replace(config, **changes)
    args.num_envs, args.seed, args.action_dim = (
        config.num_envs,
        config.seed,
        config.action_dim,
    )
    from holoboxing.training.scripts.train_mujoco import train

    with budget.handle_signals():
        train(args, config, time_budget=budget)


if __name__ == "__main__":
    main()
