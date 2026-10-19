"""Whole-epoch unweighted C225 supervision."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from holoboxing.training.rsl_rl.modules.actor_critic import JointActor
from holoboxing.training.rsl_rl.modules.normalization import (
    ActorNormalizer,
    validate_statistics,
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
    if saved.get("format") != FORMAT or _actor_config(saved["config"]) != _actor_config(
        config
    ):
        raise ValueError("Supervised action/network configuration differs")
    _finite_state(saved["actor"])
    _finite_state(saved["normalizer"])
    with torch.random.fork_rng(devices=[]):
        actor = JointActor(neutral_reference, config)
        normalizer = ActorNormalizer(until=int(saved["normalizer"]["until"]))
    for key in ("neutral_reference", "physical_scale"):
        expected, actual = actor.state_dict()[key], saved["actor"][key]
        # Old checkpoints used CUDA FK for this fixed pose; CPU differs by ulps.
        matches = (
            torch.allclose(expected, actual, atol=1e-6, rtol=0)
            if key == "neutral_reference"
            else torch.equal(expected, actual)
        )
        if not matches:
            raise ValueError(f"Cannot change supervised execution mapping: {key}")
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
):
    path = Path(path)
    payload = {
        "format": FORMAT,
        "actor": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        "normalizer": {
            k: v.detach().cpu().clone() for k, v in normalizer.state_dict().items()
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
            "whole_epoch_complete": True,
            "rows_per_epoch": records.sample_count(),
        },
    }
    _finite_state(payload["actor"])
    _finite_state(payload["normalizer"])
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
):
    if any(type(v) is not int or v < 1 for v in (epochs, batch_size, save_every)):
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
    expected_train_rows = records.sample_count()
    if (
        statistics["count"] != expected_train_rows
        or int(normalizer.count) != expected_train_rows
    ):
        raise ValueError("Statistics must cover the complete training population")
    _finite_state(normalizer.state_dict())
    validate_statistics(normalizer.state_dict())
    if normalizer.actor_dim != config.actor_dim:
        raise ValueError("Supervised normalizer and actor input layouts differ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    actor = JointActor(neutral_reference, config).to(device)
    normalizer = normalizer.to(device).eval().requires_grad_(False)
    step = 0
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    loss_statistics = {
        key: {**statistics[key], "std": statistics[key]["std"].to(device)}
        for key in ("position", "rotation225")
    }
    fixed_buffers = {k: v.detach().clone() for k, v in actor.named_buffers()}
    input_stats = {k: v.detach().clone() for k, v in normalizer.state_dict().items()}
    log = output / "metrics.jsonl"
    for epoch in range(epochs):
        report = {"epoch": epoch + 1}
        actor.train()
        sums, seen = {}, 0
        for batch in records.batches(batch_size, seed=seed + epoch, device=device):
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
            metrics = {**losses, **physical_metrics(prediction.detach(), batch)}
            size = len(batch["self"])
            seen += size
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * size
        if seen != expected_train_rows:
            raise ValueError("Epoch did not cover the complete dataset")
        report["train"] = {
            **{key: value / seen for key, value in sums.items()},
            "samples": seen,
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
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(
                output / f"epoch_{epoch + 1:06d}.pt",
                actor=actor,
                normalizer=normalizer,
                config=config,
                statistics=statistics,
                records=records,
                epoch=epoch + 1,
                step=step,
                seed=seed,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_grad_norm=max_grad_norm,
            )
    return output / f"epoch_{epochs:06d}.pt"
