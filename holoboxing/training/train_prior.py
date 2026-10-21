"""Train a source-only spherical prior before policy learning."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from holoboxing.training.time_budget import (
    add_budget_arguments,
    budget_from_arguments,
)


def train(
    records,
    validation,
    output,
    *,
    epochs=1000,
    batch_size=1024,
    learning_rate=0.001,
    seed=42,
    hidden_dims=(512, 256),
    time_budget=None,
    resume=None,
    patience=20,
    min_relative_improvement=0.001,
):
    import torch

    from holoboxing.training.checkpoint_io import (
        random_state,
        restore_random_state,
        save_checkpoint,
        training_result,
    )
    from holoboxing.training.motion_prior import (
        MotionPrior,
        PriorConfig,
        load_prior,
        model_payload,
    )
    from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
    from holoboxing.training.supervised.statistics import physical_metrics
    from holoboxing.training.validation_stop import ValidationStop

    stopping = ValidationStop(patience, min_relative_improvement)

    if (
        records.selection["split"] != "train"
        or validation.selection["split"] != "validation"
    ):
        raise ValueError(
            "Separate train and held-out reference populations are required"
        )
    if records.manifest_sha256 != validation.manifest_sha256:
        raise ValueError("Prior train/validation manifests differ")
    if any(type(v) is not int or v < 1 for v in (epochs, batch_size)):
        raise ValueError("Positive epoch and batch budgets are required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid prior optimization recipe")
    recipe = dict(
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        uniformity_weight=0.0,
        patience=patience,
        min_relative_improvement=min_relative_improvement,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    values = records.targets
    floors = values.new_tensor([0.01] * 3 + [0.05] * 6).repeat(25)
    mean = values.double().mean(0).float()
    scale = values.double().std(0, unbiased=False).float().clamp_min(floors)
    model = MotionPrior(mean, scale, PriorConfig(hidden_dims=hidden_dims)).to(
        values.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cursor = {"epoch": 0, "batch": 0, "samples": 0, "sums": {}}
    steps = 0
    best = None
    if resume is not None:
        loaded, saved = load_prior(resume, device=values.device, frozen=False)
        if (
            saved.get("recipe") != recipe
            or saved.get("source_selection") != records.selection
            or saved.get("device_type") != values.device.type
            or loaded.config != model.config
            or not torch.equal(loaded.decoder.reference_mean, mean)
            or not torch.equal(loaded.decoder.reference_scale, scale)
        ):
            raise ValueError(
                "Prior resume recipe, statistics or population differ"
            )
        model.load_state_dict(loaded.state_dict())
        optimizer.load_state_dict(saved["optimizer"])
        cursor, steps, best = (
            saved["cursor"],
            saved["steps"],
            saved["best_validation"],
        )
        restore_random_state(saved["rng"])
        stopping.load_state_dict(saved["early_stopping"])
        if best is not None:
            previous_best = Path(resume).parent / "best.pt"
            candidate = torch.load(
                previous_best, map_location="cpu", weights_only=False
            )
            if (
                candidate.get("best_validation") != best
                or candidate.get("source_selection") != records.selection
            ):
                raise ValueError("Resume best-validation checkpoint differs")
            shutil.copyfile(previous_best, output / "best.pt")
        if cursor["epoch"] > epochs:
            raise ValueError("Prior resume target precedes completed epochs")

    def save(name):
        require_finite(
            (model.state_dict(), optimizer.state_dict()), "prior checkpoint"
        )
        path = output / name
        save_checkpoint(
            path,
            {
                **model_payload(model),
                "recipe": recipe,
                "source_selection": records.selection,
                "validation_selection": validation.selection,
                "training_groups": sorted(
                    {(r["dataset"], r["group"]) for r in records.records}
                ),
                "validation_groups": sorted(
                    {(r["dataset"], r["group"]) for r in validation.records}
                ),
                "cursor": cursor,
                "steps": steps,
                "optimizer": optimizer.state_dict(),
                "rng": random_state(),
                "best_validation": best,
                "device_type": values.device.type,
                "asset_identity": records.manifest["asset_identity"],
                "early_stopping": stopping.state_dict(),
            },
        )
        if time_budget is not None:
            time_budget.checkpoint_saved()
        return path

    plateau = stopping.stopped
    for epoch in range(
        cursor["epoch"], epochs if not plateau else cursor["epoch"]
    ):
        model.train()
        sums, seen = cursor["sums"], cursor["samples"]
        for target in records.batches(
            batch_size, seed=seed + epoch, start_batch=cursor["batch"]
        ):
            if time_budget is not None and time_budget.should_stop():
                break
            started = time.monotonic()
            latent = model.encode(target)
            reconstructed = model.decoder(latent)
            reconstruction = ((reconstructed - target) / scale).square().mean()
            loss = reconstruction
            require_finite((loss, reconstructed), "prior update")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            seen += len(target)
            metrics = {
                "loss": loss,
                "reconstruction": reconstruction,
                **physical_metrics(
                    reconstructed.detach(), {"target_pose": target}
                ),
            }
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(
                    target
                )
            cursor = {
                "epoch": epoch,
                "batch": cursor["batch"] + 1,
                "samples": seen,
                "sums": sums,
            }
            if time_budget is not None:
                time_budget.update_finished(time.monotonic() - started)
                if time_budget.checkpoint_due():
                    save("last.pt")
        if seen != len(values):
            break
        report = {
            "epoch": epoch + 1,
            "steps": steps,
            "train": {k: v / seen for k, v in sums.items()},
        }
        total, count, latent_sum, by_dataset = {}, 0, None, {}
        model.eval()
        with torch.no_grad():
            for dataset, target in validation.source_batches(batch_size):
                if time_budget is not None and time_budget.should_stop():
                    break
                latent = model.encode(target)
                prediction = model.decoder(latent)
                metrics = {
                    "reconstruction": ((prediction - target) / scale)
                    .square()
                    .mean(),
                    **physical_metrics(prediction, {"target_pose": target}),
                }
                require_finite(metrics, "prior validation")
                count += len(target)
                source = by_dataset.setdefault(
                    dataset, {"samples": 0, "sums": {}}
                )
                source["samples"] += len(target)
                latent_sum = (
                    latent.sum(0)
                    if latent_sum is None
                    else latent_sum + latent.sum(0)
                )
                for name, value in metrics.items():
                    total[name] = total.get(name, 0.0) + float(value) * len(
                        target
                    )
                    source["sums"][name] = source["sums"].get(
                        name, 0.0
                    ) + float(value) * len(target)
        complete = count == len(validation.targets) and count > 0
        report["validation"] = {
            "status": "PASS" if complete else "INCOMPLETE",
            "samples": count,
            "expected_samples": len(validation.targets),
        }
        if complete:
            report["validation"].update(
                {k: v / count for k, v in total.items()}
            )
            report["validation"]["mean_latent_length"] = float(
                (latent_sum / count).norm()
            )
            report["validation"]["by_dataset"] = {
                name: {
                    "samples": values["samples"],
                    **{
                        k: v / values["samples"]
                        for k, v in values["sums"].items()
                    },
                }
                for name, values in by_dataset.items()
            }
        cursor = {"epoch": epoch + 1, "batch": 0, "samples": 0, "sums": {}}
        if complete:
            plateau = stopping.observe(report["validation"]["reconstruction"])
        report["early_stopping"] = stopping.state_dict()
        if complete and (
            best is None
            or report["validation"]["reconstruction"] < best["reconstruction"]
        ):
            best = {"epoch": epoch + 1, **report["validation"]}
            save("best.pt")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)
        if plateau:
            if time_budget is not None:
                time_budget.request_stop(reason="validation_plateau")
            break
        if time_budget is not None and time_budget.should_stop():
            break
    path = save("last.pt")
    training_result(
        output,
        path,
        budget=time_budget,
        progress={
            "epochs_completed": cursor["epoch"],
            "next_batch": cursor["batch"],
            "steps": steps,
            "stop_reason": "validation_plateau"
            if plateau
            else (
                time_budget.report()["stop_reason"]
                if time_budget is not None
                else "requested_progress_complete"
            ),
        },
    )
    return path


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="source-only reference directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("all", "robostriker", "interhuman"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=(512, 256)
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="validation patience; 0 disables early stopping",
    )
    parser.add_argument(
        "--min-relative-improvement", type=float, default=0.001
    )
    add_budget_arguments(parser)
    args = parser.parse_args(argv)
    try:
        budget = budget_from_arguments(args, started_at=started)
    except ValueError as error:
        parser.error(str(error))
    with budget.handle_signals():
        from holoboxing.training.prior_data import PriorRecords

        records = PriorRecords(
            args.data / "manifest.json", dataset=args.dataset
        ).to(args.device)
        validation = PriorRecords(
            args.data / "manifest.json",
            dataset=args.dataset,
            split="validation",
        ).to(args.device)
        path = train(
            records,
            validation,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            hidden_dims=args.hidden_dims,
            time_budget=budget,
            resume=args.resume,
            patience=args.patience,
            min_relative_improvement=args.min_relative_improvement,
        )
        print(path)


if __name__ == "__main__":
    main()
