"""Whole-epoch unweighted C225 supervision."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from holoboxing.training.checkpoint_io import (
    random_state,
    restore_random_state,
    save_checkpoint as atomic_checkpoint,
    training_result,
)
from holoboxing.training.rsl_rl.modules.actor_critic import JointActor
from holoboxing.training.rsl_rl.modules.normalization import (
    ActorNormalizer,
    validate_statistics,
)
from holoboxing.training.spherical_actor import (
    FORMAT as SPHERICAL_FORMAT,
    actor_from_prior,
    actor_from_state,
)
from holoboxing.training.supervised.statistics import (
    normalized_loss,
    physical_metrics,
)

FORMAT = "holoboxing_supervised_link_velocity_v2"
ACTOR_CONFIG = (
    "actor_dim",
    "action_dim",
    "hidden_dims",
    "position_scale_m",
    "rotation_scale",
)


def _actor_config(config):
    result = asdict(config) if not isinstance(config, dict) else config
    return {
        key: tuple(result[key]) if key == "hidden_dims" else result[key]
        for key in ACTOR_CONFIG
    }


def _finite_state(state):
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite actor or normalizer checkpoint")


def load_supervised(path, config, neutral_reference, device="cpu"):
    """Load attack-only policy without advancing global random streams."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") not in (FORMAT, SPHERICAL_FORMAT) or _actor_config(
        saved["config"]
    ) != _actor_config(config):
        raise ValueError("Supervised action/network configuration differs")
    selection = saved.get("loss_statistics", {}).get("source_selection", {})
    if (
        selection.get("dataset") != "interhuman"
        or selection.get("split") != "train"
        or not selection.get("sources_sha256")
        or selection.get("samples") != saved["loss_statistics"]["count"]
        or saved.get("coverage", {}).get("source_selection") != selection
    ):
        raise ValueError(
            "Regenerate the supervised checkpoint: verified training split "
            "provenance is required"
        )
    _finite_state(saved["actor"])
    _finite_state(saved["normalizer"])
    with torch.random.fork_rng(devices=[]):
        actor = (
            actor_from_state(saved["actor"], saved["policy_spec"], config)
            if saved["format"] == SPHERICAL_FORMAT
            else JointActor(neutral_reference, config)
        )
        normalizer = ActorNormalizer(until=int(saved["normalizer"]["until"]))
    for key in (
        ()
        if saved["format"] == SPHERICAL_FORMAT
        else ("neutral_reference", "physical_scale")
    ):
        expected, actual = actor.state_dict()[key], saved["actor"][key]
        # Old checkpoints used CUDA FK; CPU differs by a few rounding units.
        matches = (
            torch.allclose(expected, actual, atol=1e-6, rtol=0)
            if key == "neutral_reference"
            else torch.equal(expected, actual)
        )
        if not matches:
            raise ValueError(
                f"Cannot change supervised execution mapping: {key}"
            )
    actor.load_state_dict(saved["actor"], strict=True)
    normalizer.load_state_dict(saved["normalizer"], strict=True)
    if int(normalizer.count) <= 0:
        raise ValueError("Supervised input normalizer is unfitted")
    if any(
        (saved["normalizer"][key] < 0).any()
        for key in ("self_variance", "task_variance")
    ):
        raise ValueError("Supervised input variance is negative")
    if int(normalizer.count) != saved["loss_statistics"]["count"]:
        raise ValueError("Supervised statistics population differs")
    return actor.to(device), normalizer.to(device), saved


def save_checkpoint(
    path,
    *,
    actor,
    normalizer,
    config,
    statistics,
    records,
    epoch,
    step,
    seed,
    learning_rate,
    batch_size,
    max_grad_norm,
    optimizer=None,
    cursor=None,
):
    path = Path(path)
    payload = {
        "format": SPHERICAL_FORMAT if hasattr(actor, "spec") else FORMAT,
        "actor": {
            k: v.detach().cpu().clone() for k, v in actor.state_dict().items()
        },
        "normalizer": {
            k: v.detach().cpu().clone()
            for k, v in normalizer.state_dict().items()
        },
        "config": asdict(config),
        "loss_statistics": statistics,
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "learning_rate": learning_rate,
        "training_recipe": {
            "batch_size": batch_size,
            "max_grad_norm": max_grad_norm,
            "optimizer": "Adam",
        },
        "device_type": actor.neutral_reference.device.type,
        "coverage": {
            "whole_epoch_complete": cursor is None or cursor["batch"] == 0,
            "rows_per_epoch": records.sample_count(),
            "source_selection": records.source_selection,
        },
    }
    if hasattr(actor, "spec"):
        actor.check_frozen()
        payload["policy_spec"] = actor.spec
    if optimizer is not None:
        payload["resume"] = {
            "format": "supervised_resume_v1",
            "optimizer": optimizer.state_dict(),
            "rng": random_state(),
            "cursor": cursor,
            "manifest_sha256": records.manifest_sha256,
        }
    _finite_state(payload["actor"])
    _finite_state(payload["normalizer"])
    atomic_checkpoint(path, payload)


@torch.enable_grad()
def train(
    records,
    config,
    neutral_reference,
    statistics,
    normalizer,
    output,
    *,
    epochs,
    batch_size,
    learning_rate,
    seed=42,
    device="cpu",
    save_every=1,
    max_grad_norm=1.0,
    validation_records=None,
    time_budget=None,
    resume=None,
    prior_checkpoint=None,
):
    if any(
        type(v) is not int or v < 1 for v in (epochs, batch_size, save_every)
    ):
        raise ValueError(
            "Epochs, batch size and save cadence must be positive integers"
        )
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("Invalid supervised optimizer configuration")
    if statistics["manifest_sha256"] != records.manifest_sha256:
        raise ValueError("Statistics do not belong to this dataset")
    if statistics.get("source_selection") != records.source_selection:
        raise ValueError(
            "Statistics do not belong to this supervised source selection"
        )
    if records.source_selection.get("split") != "train":
        raise ValueError("Supervised optimization requires the train split")
    if validation_records is not None and (
        validation_records.manifest_sha256 != records.manifest_sha256
        or validation_records.source_selection.get("split") != "validation"
        or validation_records.source_selection.get("sources_sha256")
        != records.source_selection.get("sources_sha256")
    ):
        raise ValueError(
            "Validation must use the held-out split of the same "
            "verified dataset"
        )
    expected_train_rows = records.sample_count()
    if (
        statistics["count"] != expected_train_rows
        or int(normalizer.count) != expected_train_rows
    ):
        raise ValueError(
            "Statistics must cover the complete training population"
        )
    _finite_state(normalizer.state_dict())
    validate_statistics(normalizer.state_dict())
    if normalizer.actor_dim != config.actor_dim:
        raise ValueError(
            "Supervised normalizer and actor input layouts differ"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    if resume is not None:
        actor, loaded_normalizer, saved = load_supervised(
            resume, config, neutral_reference, device
        )
        # The frozen decoder is embedded; restart needs no original prior file.
        if prior_checkpoint is not None:
            requested = actor_from_prior(
                prior_checkpoint,
                config,
                records,
                validation_records,
            )
            if saved.get("policy_spec") != requested.spec:
                raise ValueError(
                    "Resume prior or observation contract differs"
                )
    else:
        actor = (
            JointActor(neutral_reference, config)
            if prior_checkpoint is None
            else actor_from_prior(
                prior_checkpoint,
                config,
                records,
                validation_records,
            )
        ).to(device)
    normalizer = normalizer.to(device).eval().requires_grad_(False)
    step = 0
    optimizer = torch.optim.Adam(
        (p for p in actor.parameters() if p.requires_grad), lr=learning_rate
    )
    cursor = {"epoch": 0, "batch": 0, "seen": 0, "sums": {}}
    if resume is not None:
        state = saved.get("resume", {})
        if (
            state.get("format") != "supervised_resume_v1"
            or state.get("manifest_sha256") != records.manifest_sha256
            or saved["loss_statistics"]["source_selection"]
            != records.source_selection
            or saved["training_recipe"]
            != {
                "batch_size": batch_size,
                "max_grad_norm": max_grad_norm,
                "optimizer": "Adam",
            }
            or saved["seed"] != seed
            or saved["learning_rate"] != learning_rate
            or saved["device_type"] != torch.device(device).type
            or saved.get("policy_spec") != getattr(actor, "spec", None)
        ):
            raise ValueError(
                "Resume requires the same dataset, device and optimizer recipe"
            )
        for name, value in loaded_normalizer.state_dict().items():
            if not torch.equal(
                value.cpu(), normalizer.state_dict()[name].cpu()
            ):
                raise ValueError("Resume input statistics differ")
        optimizer.load_state_dict(state["optimizer"])
        step, cursor = saved["step"], state["cursor"]
        restore_random_state(state["rng"])
        if cursor["epoch"] > epochs:
            raise ValueError("Resume target precedes completed epochs")
    loss_statistics = {
        key: {**statistics[key], "std": statistics[key]["std"].to(device)}
        for key in ("position", "rotation225")
    }
    fixed_buffers = {k: v.detach().clone() for k, v in actor.named_buffers()}
    input_stats = {
        k: v.detach().clone() for k, v in normalizer.state_dict().items()
    }
    log = output / "metrics.jsonl"

    def publish(path):
        save_checkpoint(
            path,
            actor=actor,
            normalizer=normalizer,
            config=config,
            statistics=statistics,
            records=records,
            epoch=cursor["epoch"],
            step=step,
            seed=seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            cursor=cursor,
        )
        if time_budget is not None:
            time_budget.checkpoint_saved()
        return path

    last = None
    for epoch in range(cursor["epoch"], epochs):
        report = {"epoch": epoch + 1}
        actor.train()
        sums, seen = cursor["sums"], cursor["seen"]
        for batch in records.batches(
            batch_size,
            seed=seed + epoch,
            device=device,
            start_batch=cursor["batch"],
        ):
            if time_budget is not None and time_budget.should_stop():
                break
            started = time.monotonic()
            prediction = actor(normalizer(batch))
            target = batch["target_pose"]
            losses = normalized_loss(prediction, target, loss_statistics)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            metrics = {
                **losses,
                **physical_metrics(prediction.detach(), batch),
            }
            size = len(batch["self"])
            seen += size
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * size
            cursor = {
                "epoch": epoch,
                "batch": cursor["batch"] + 1,
                "seen": seen,
                "sums": sums,
            }
            if time_budget is not None:
                time_budget.update_finished(time.monotonic() - started)
                if time_budget.checkpoint_due():
                    last = publish(output / "last.pt")
        if seen != expected_train_rows:
            if time_budget is not None and time_budget.should_stop():
                break
            raise ValueError("Epoch did not cover the complete dataset")
        report["train"] = {
            **{key: value / seen for key, value in sums.items()},
            "samples": seen,
        }
        if (
            validation_records is not None
            and validation_records.sample_count()
        ):
            actor.eval()
            validation_sums, validation_seen = {}, 0
            with torch.no_grad():
                for batch in validation_records.batches(
                    batch_size, device=device
                ):
                    if time_budget is not None and time_budget.should_stop():
                        break
                    prediction = actor(normalizer(batch))
                    metrics = {
                        **normalized_loss(
                            prediction, batch["target_pose"], loss_statistics
                        ),
                        **physical_metrics(prediction, batch),
                    }
                    size = len(batch["self"])
                    validation_seen += size
                    for key, value in metrics.items():
                        validation_sums[key] = (
                            validation_sums.get(key, 0.0) + float(value) * size
                        )
            if validation_seen != validation_records.sample_count():
                report["validation"] = {
                    "status": "INCOMPLETE",
                    "samples": validation_seen,
                    "expected_samples": validation_records.sample_count(),
                    "reason": "Training deadline reached",
                }
            else:
                report["validation"] = {
                    **{
                        key: value / validation_seen
                        for key, value in validation_sums.items()
                    },
                    "samples": validation_seen,
                    "status": "PASS",
                }
        else:
            report["validation"] = {
                "status": "NOT_RUN",
                "samples": 0,
                "reason": "No held-out InterHuman clips in the selected data",
            }
        if any(
            not torch.equal(value, dict(actor.named_buffers())[key])
            for key, value in fixed_buffers.items()
        ):
            raise ValueError("Training changed the action execution mapping")
        if any(
            not torch.equal(value, normalizer.state_dict()[key])
            for key, value in input_stats.items()
        ):
            raise ValueError("SL updated frozen input statistics")
        report["step"] = step
        with log.open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)
        cursor = {"epoch": epoch + 1, "batch": 0, "seen": 0, "sums": {}}
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            last = publish(output / f"epoch_{epoch + 1:06d}.pt")
        if time_budget is not None and time_budget.should_stop():
            break
    if cursor["epoch"] < epochs or last is None:
        last = publish(output / "last.pt")
    training_result(
        output,
        last,
        budget=time_budget,
        progress={
            "epochs_completed": cursor["epoch"],
            "next_batch": cursor["batch"],
            "steps": step,
        },
    )
    return last
