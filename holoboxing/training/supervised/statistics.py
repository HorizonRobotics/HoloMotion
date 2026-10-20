"""Training-population statistics, losses and physical metrics."""

import math

import torch

from holoboxing.training.rsl_rl.modules.normalization import ActorNormalizer
from holoboxing.training.tasks.combat.mdp import geometry as g


class Moments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def update(self, value):
        value = value.double().cpu()
        n, mean = len(value), value.mean(0)
        m2 = ((value - mean) ** 2).sum(0)
        if not self.count:
            self.mean, self.m2, self.count = mean, m2, n
            return
        delta, total = mean - self.mean, self.count + n
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def result(self, floor):
        std = (self.m2 / self.count).clamp_min(0).sqrt()
        return {
            "std": std.clamp_min(floor).float(),
            "raw_std": std.float(),
            "mean": self.mean.float(),
            "floor": floor,
            "floored_indices": (std < floor).nonzero().tolist(),
            "count": self.count,
        }


@torch.no_grad()
def fit_statistics(
    records, *, position_floor=0.01, rotation225_floor=0.05, batch_size=4096
):
    floors = dict(position=position_floor, rotation225=rotation225_floor)
    if any(not math.isfinite(v) or v <= 0 for v in floors.values()):
        raise ValueError("Statistics floors must be positive finite values")
    if records.source_selection.get("split") != "train":
        raise ValueError(
            "Fit normalization and loss statistics on the train split only"
        )
    count = records.sample_count()
    normalizer = ActorNormalizer(until=max(100_000_000, count))
    moments = {key: Moments() for key in floors}
    for batch in records.batches(batch_size):
        normalizer.update(batch["self"], batch["task"])
        target = batch["target_pose"].reshape(-1, 5, 5, 9)
        moments["position"].update(target[..., :3])
        moments["rotation225"].update(target[..., 3:])
    if int(normalizer.count) != count:
        raise ValueError("Statistics did not cover the full train population")
    statistics = {
        key: value.result(floors[key]) for key, value in moments.items()
    }
    statistics.update(
        manifest_sha256=records.manifest_sha256,
        source_selection=records.source_selection,
        count=count,
    )
    return statistics, normalizer


def normalized_loss(prediction, target, statistics):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 225
    ):
        raise ValueError("Expected aligned 225D outputs")
    if (
        not torch.isfinite(prediction).all()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("Nonfinite supervised output or target")
    dim = prediction.shape[1]
    difference = (prediction - target).reshape(-1, 5, 5, dim // 25)
    pstd = statistics["position"]["std"].to(prediction)
    rstd = statistics[f"rotation{dim}"]["std"].to(prediction)
    if (
        pstd.shape != (5, 5, 3)
        or rstd.shape != (5, 5, dim // 25 - 3)
        or not (pstd > 0).all()
        or not (rstd > 0).all()
        or not torch.isfinite(pstd).all()
        or not torch.isfinite(rstd).all()
    ):
        raise ValueError("Invalid frozen loss statistics")
    position = (difference[..., :3] / pstd).square().mean()
    rotation = (difference[..., 3:] / rstd).square().mean()
    return {
        "loss": 0.5 * position + 0.5 * rotation,
        "position_loss": position,
        "rotation_loss": rotation,
    }


@torch.no_grad()
def physical_metrics(prediction, batch):
    predicted = prediction.reshape(-1, 5, 5, 9)
    target = batch["target_pose"].reshape(-1, 5, 5, 9)
    x, z = target[..., 3:6], target[..., 6:9]
    target_rotation = torch.stack((x, torch.cross(z, x, dim=-1), z), -1)
    relative = (
        g.project_xz(predicted[..., 3:]).transpose(-1, -2) @ target_rotation
    )
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
        -1, 1
    )
    return {
        "position_m": (predicted[..., :3] - target[..., :3])
        .norm(dim=-1)
        .mean(),
        "rotation_radians": torch.acos(cosine).mean(),
    }
