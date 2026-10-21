"""Verified reference windows; observations are never read by the prior."""

import json
from pathlib import Path

import numpy as np
import torch

from holoboxing.data.common import inside, sha256
from holoboxing.training.prepare_prior import FORMAT


class PriorRecords:
    def __init__(
        self, manifest, *, split="train", dataset="all", allow_empty=False
    ):
        self.manifest_path = Path(manifest).resolve(strict=True)
        self.manifest_sha256 = sha256(self.manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        if (
            self.manifest.get("format") != FORMAT
            or self.manifest.get("control_hz") != 50
            or self.manifest.get("future_offsets") != [1, 2, 3, 4, 5]
            or self.manifest.get("coordinate_frame")
            != "source_current_pelvis_full_3d"
            or self.manifest.get("input") != "reference_only_225"
            or self.manifest.get("synthetic_opponent_samples") != 0
        ):
            raise ValueError("Unsupported source-only reference contract")
        if split not in ("train", "validation", "test") or dataset not in (
            "all",
            "interhuman",
            "robostriker",
        ):
            raise ValueError("Unknown prior source selection")
        groups, paths = {}, set()
        self.records, arrays = [], []
        for row in self.manifest["records"]:
            key = (row["dataset"], row["group"])
            if row["dataset"] not in ("interhuman", "robostriker") or row[
                "split"
            ] not in ("train", "validation", "test"):
                raise ValueError("Invalid source or split")
            if key in groups and groups[key] != row["split"]:
                raise ValueError(
                    "A source group crosses train/validation/test splits"
                )
            groups[key] = row["split"]
            path = inside(self.root, row["path"])
            if path in paths:
                raise ValueError("Duplicate reference shard")
            paths.add(path)
            if (
                sha256(path) != row["sha256"]
                or path.stat().st_size != row["bytes"]
            ):
                raise ValueError("Reference shard differs from its manifest")
            if (
                type(row["frames"]) is not int
                or row["frames"] < 6
                or row["roles"] not in (1, 2)
                or row["samples"] != (row["frames"] - 5) * row["roles"]
            ):
                raise ValueError(
                    "Reference coverage differs from real source roles"
                )
            if row["split"] != split or dataset not in ("all", row["dataset"]):
                continue
            with np.load(path, allow_pickle=False) as packet:
                values = packet["target_pose"]
                if (
                    values.shape != (row["samples"], 225)
                    or values.dtype != np.float32
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid source-only reference array")
                points = values.reshape(-1, 25, 9)
                x, z = points[..., 3:6], points[..., 6:]
                if not (
                    np.allclose(np.linalg.norm(x, axis=-1), 1, atol=2e-4)
                    and np.allclose(np.linalg.norm(z, axis=-1), 1, atol=2e-4)
                    and (np.abs((x * z).sum(-1)) <= 2e-4).all()
                ):
                    raise ValueError(
                        "Reference rotation axes are not orthonormal"
                    )
                arrays.append(torch.from_numpy(values.copy()))
            self.records.append(row)
        if not arrays and not allow_empty:
            raise ValueError("No reference windows in the selected population")
        self.targets = torch.cat(arrays) if arrays else torch.empty(0, 225)
        self.selection = {
            "split": split,
            "dataset": dataset,
            "samples": len(self.targets),
            "clips": len(self.records),
            "manifest_sha256": self.manifest_sha256,
        }

    def to(self, device):
        self.targets = self.targets.to(device)
        return self

    def source_batches(self, batch_size):
        """Keep source labels for complete validation population reports."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        offset = 0
        for row in self.records:
            values = self.targets[offset : offset + row["samples"]]
            offset += row["samples"]
            for batch in values.split(batch_size):
                yield row["dataset"], batch

    def batches(self, batch_size, *, seed=0, start_batch=0):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        count = (len(self.targets) + batch_size - 1) // batch_size
        if type(start_batch) is not int or not 0 <= start_batch <= count:
            raise ValueError("Invalid reference resume batch")
        order = np.random.default_rng(seed).permutation(len(self.targets))
        for start in range(start_batch * batch_size, len(order), batch_size):
            indices = torch.as_tensor(
                order[start : start + batch_size], device=self.targets.device
            )
            yield self.targets[indices]
