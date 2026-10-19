"""Train the boxing planner on complete epochs of reference demonstrations."""

import argparse
from pathlib import Path

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.resources import (
    load_neutral_reference,
    resolve_resources,
)
from holoboxing.training.supervised.data import Records
from holoboxing.training.supervised.statistics import fit_statistics
from holoboxing.training.supervised.training import train
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument(
        "--data",
        type=existing_path,
        required=True,
        help="prepared directory produced by holoboxing.data prepare",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")
    tree, paths, _ = resolve_resources(args.assets, args.lower)
    records = Records(args.data / "manifest.json")
    identity = records.manifest["asset_identity"]
    if identity.get("robot_kinematics") != "competition_xml_v1":
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
    reference = load_neutral_reference(paths["lower_onnx"], paths["arena_scene"])
    result = train(
        records,
        TrainConfig(),
        reference,
        statistics,
        normalizer,
        args.output,
        epochs=args.epochs,
        batch_size=256,
        learning_rate=0.0003,
        seed=42,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
