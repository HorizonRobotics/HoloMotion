"""Evaluate a selected frozen prior on held-out test motions."""

import argparse
import json
from pathlib import Path

import torch

from holoboxing.data.common import sha256, write_json
from holoboxing.training.motion_prior import load_prior
from holoboxing.training.prior_data import PriorRecords
from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.select_prior import FORMAT as SELECTION_FORMAT
from holoboxing.training.supervised.statistics import physical_metrics


def evaluate(
    checkpoint, manifest, selection, output, *, batch_size=1024, device="cpu"
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    decision = json.loads(Path(selection).read_text())
    digest = sha256(checkpoint)
    if (
        decision.get("format") != SELECTION_FORMAT
        or decision.get("selected_checkpoint_sha256") != digest
        or decision.get("test_evaluated") is not False
    ):
        raise ValueError(
            "Test evaluation requires a fixed validation selection"
        )
    model, saved = load_prior(checkpoint, device=device)
    if (
        decision.get("source_selection") != saved.get("source_selection")
        or decision.get("validation_selection")
        != saved.get("validation_selection")
        or saved["source_selection"]["manifest_sha256"] != sha256(manifest)
    ):
        raise ValueError(
            "Test population differs from the selected experiment"
        )
    dataset = saved["source_selection"]["dataset"]
    records = PriorRecords(manifest, split="test", dataset=dataset).to(device)
    trained_sources = {row[0] for row in saved.get("training_groups", [])}
    if {row["dataset"] for row in records.records} != trained_sources:
        raise ValueError(
            "Test population must cover every trained data source"
        )
    groups = {(r["dataset"], r["group"]) for r in records.records}
    fitted = {
        tuple(r)
        for key in ("training_groups", "validation_groups")
        for r in saved.get(key, [])
    }
    if (
        not saved.get("training_groups")
        or not saved.get("validation_groups")
        or groups & fitted
    ):
        raise ValueError(
            "Test groups overlap fitting or model-selection groups"
        )
    total, by_dataset, offset = {}, {}, 0
    latent_sum, count = None, 0
    with torch.no_grad():
        for row in records.records:
            values = records.targets[offset : offset + row["samples"]]
            offset += row["samples"]
            source = by_dataset.setdefault(
                row["dataset"], {"samples": 0, "sums": {}}
            )
            for target in values.split(batch_size):
                latent = model.encode(target)
                prediction = model.decoder(latent)
                metrics = {
                    "reconstruction": (
                        (prediction - target) / model.decoder.reference_scale
                    )
                    .square()
                    .mean(),
                    **physical_metrics(prediction, {"target_pose": target}),
                }
                require_finite(metrics, "held-out prior evaluation")
                n = len(target)
                count += n
                source["samples"] += n
                latent_sum = (
                    latent.sum(0)
                    if latent_sum is None
                    else latent_sum + latent.sum(0)
                )
                for name, value in metrics.items():
                    value = float(value) * n
                    total[name] = total.get(name, 0.0) + value
                    source["sums"][name] = (
                        source["sums"].get(name, 0.0) + value
                    )
    report = dict(
        format="holoboxing_selected_prior_test_result_v1",
        status="PASS",
        scope="offline_reference_reconstruction",
        checkpoint_sha256=digest,
        selection_sha256=sha256(selection),
        population=records.selection,
        groups=sorted(groups),
        metrics={name: value / count for name, value in total.items()},
        mean_latent_length=float((latent_sum / count).norm()),
        by_dataset={
            name: {
                "samples": values["samples"],
                **{
                    k: v / values["samples"] for k, v in values["sums"].items()
                },
            }
            for name, values in by_dataset.items()
        },
        closed_loop_verified=False,
    )
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    evaluate(
        args.checkpoint,
        args.manifest,
        args.selection,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
