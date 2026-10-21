"""Train the boxing planner on complete epochs of InterHuman demonstrations."""

import argparse
import json
import time
from pathlib import Path

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.time_budget import (
    add_budget_arguments,
    budget_from_arguments,
)


def main(argv=None):
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    add_budget_arguments(parser)
    parser.add_argument(
        "--data",
        type=existing_path,
        required=True,
        help="prepared directory; supervised training uses only InterHuman",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=existing_path,
        help="optional TrainConfig JSON; use the same actor dimensions for RL",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="total target epochs, including resumed progress",
    )
    parser.add_argument("--resume", type=existing_path)
    parser.add_argument(
        "--prior-checkpoint",
        type=existing_path,
        help="freeze this reference-only sphere32 decoder",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")
    try:
        budget = budget_from_arguments(args, started_at=started_at)
    except ValueError as error:
        parser.error(str(error))
    with budget.handle_signals():
        _run(args, budget)


def _run(args, budget):
    from holoboxing.data.contract import ROBOT_KINEMATICS
    from holoboxing.training.resources import (
        load_neutral_reference,
        resolve_resources,
    )
    from holoboxing.training.supervised.data import Records
    from holoboxing.training.supervised.statistics import fit_statistics
    from holoboxing.training.supervised.training import train
    from holoboxing.training.tasks.combat.config.sandbag import TrainConfig

    config = (
        TrainConfig(**json.loads(args.config.read_text()))
        if args.config
        else TrainConfig()
    )
    tree, paths, _ = resolve_resources(args.assets, args.lower)
    records = Records(args.data / "manifest.json")
    validation = Records(
        args.data / "manifest.json", split="validation", allow_empty=True
    )
    print(
        json.dumps(
            {
                "supervised_source_selection": records.source_selection,
                "validation_source_selection": validation.source_selection,
            }
        ),
        flush=True,
    )
    identity = records.manifest["asset_identity"]
    if identity.get("robot_kinematics") != ROBOT_KINEMATICS:
        raise ValueError(
            "Regenerate source motions with holoboxing.data prepare; "
            "training requires competition XML reference kinematics"
        )
    if (
        records.manifest["asset_identity"]["resource_manifest_sha256"]
        != tree.manifest_sha256
    ):
        raise ValueError("Prepared data and training assets differ")
    statistics, normalizer = fit_statistics(records)
    reference = load_neutral_reference(
        paths["lower_onnx"], paths["arena_scene"]
    )
    result = train(
        records,
        config,
        reference,
        statistics,
        normalizer,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        validation_records=validation,
        time_budget=budget,
        resume=args.resume,
        prior_checkpoint=args.prior_checkpoint,
    )
    print(result)


if __name__ == "__main__":
    main()
