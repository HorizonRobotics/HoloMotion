"""Launch the SuperMimic reference recipe using framework training components."""

import argparse
from pathlib import Path

from holomotion.src.algo.supermimic.training import execute
from holomotion.src.utils.artifacts import (  # noqa: F401
    atomic_json,
    fingerprint,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile", choices=("paper", "vibed"), default="paper"
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--buffer-steps", type=int, default=1000)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-vector-steps", type=int, default=800064)
    parser.add_argument("--max-updates", type=int, default=320000)
    parser.add_argument("--utd", default="0.4")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--goal-frames", type=int, default=10)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm-window", type=int, default=8)
    parser.add_argument("--wm-batches", type=int, default=10)
    parser.add_argument("--wm-batch", type=int, default=2048)
    parser.add_argument("--pi-batch", type=int, default=4096)
    parser.add_argument("--pi-microbatch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--critic-minibatches", type=int, default=4)
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--world-only-updates", type=int, default=500)
    parser.add_argument("--lr-decay-updates", type=int, default=250000)
    parser.add_argument("--policy-snapshot-minutes", type=float, default=0.0)
    parser.add_argument("--checkpoint-minutes", type=float, default=15.0)
    parser.add_argument("--evaluate-minutes", type=float, default=30.0)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--capture-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--release-phase-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Release unused Torch cache before returning to Warp physics",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Execution diagnostics only; cannot establish reproduction quality",
    )
    parser.add_argument(
        "--physics-backend",
        choices=("mujoco_warp", "isaac_physx"),
        default="mujoco_warp",
    )
    from dataclasses import MISSING, fields

    from holomotion.src.algo.supermimic.config import SuperMimicConfig

    parser.set_defaults(
        **{
            field.name: field.default
            for field in fields(SuperMimicConfig)
            if field.default is not MISSING
        }
    )
    return parser


def arguments():
    from holomotion.src.algo.supermimic.config import SuperMimicConfig

    parser = build_parser()
    args = parser.parse_args()
    try:
        return SuperMimicConfig.from_mapping(vars(args))
    except (ValueError, TypeError) as error:
        parser.error(str(error))


def main():
    execute(arguments())


if __name__ == "__main__":
    main()
