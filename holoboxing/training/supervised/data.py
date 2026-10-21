"""Minibatches from the shared reference-derived competition records."""

import numpy as np
import torch

from holoboxing.data.dataset import PreparedDataset
from holoboxing.training.rsl_rl.modules.normalization import (
    unpack_actor_observations,
)


class Records(PreparedDataset):
    def __init__(
        self,
        manifest,
        *,
        asset_identity=None,
        split="train",
        allow_empty=False,
    ):
        # Validate arrays when consumed, rather than decompressing every clip twice.
        super().__init__(
            manifest, asset_identity=asset_identity, validate_contents=False
        )
        self.select_sources(
            split=split, dataset="interhuman", allow_empty=allow_empty
        )
        self.cached_index, self.cached = None, None

    def _read(self, record):
        packet = super()._read(record)
        rotations = packet["target_pose"].reshape(-1, 5, 5, 9)[..., 3:]
        x, z = rotations[..., :3], rotations[..., 3:]
        if (
            not torch.allclose(
                x.norm(dim=-1), torch.ones_like(x[..., 0]), atol=2e-4
            )
            or not torch.allclose(
                z.norm(dim=-1), torch.ones_like(z[..., 0]), atol=2e-4
            )
            or not ((x * z).sum(-1).abs() <= 2e-4).all()
        ):
            raise ValueError("Targets must contain orthonormal rotation axes")
        return packet

    def batches(self, batch_size, *, seed=None, device="cpu", start_batch=0):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        count = sum(
            (row["samples"] + batch_size - 1) // batch_size
            for row in self.records
        )
        if type(start_batch) is not int or not 0 <= start_batch <= count:
            raise ValueError("Resume batch is outside the selected epoch")
        rng = np.random.default_rng(seed)
        shards = list(range(len(self.records)))
        if seed is not None:
            rng.shuffle(shards)
        batch_index = 0
        for index in shards:
            order = np.arange(self.records[index]["samples"])
            if seed is not None:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch_index += 1
                if batch_index <= start_batch:
                    continue
                if index != self.cached_index:
                    self.cached = self._read(self.records[index])
                    self.cached_index = index
                rows = self.cached
                ids = order[start : start + batch_size]
                current, task, history = unpack_actor_observations(
                    rows["obs"][ids].to(device)
                )
                yield {
                    "self": current,
                    "task": task,
                    "history": history,
                    "target_pose": rows["target_pose"][ids].to(device),
                }
