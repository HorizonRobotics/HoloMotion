"""Validated reference records for boxing supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from holoboxing.evaluation.runtime.model_contract import CONTRACT, sha256
from .records import DATA_FORMAT


class PreparedDataset:
    def __init__(self, manifest, *, asset_identity=None, validate_contents=True):
        self.manifest_path = Path(manifest).resolve(strict=True)
        self.root = self.manifest_path.parent
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if (
            self.manifest.get("format") != DATA_FORMAT
            or self.manifest.get("policy_contract") != CONTRACT
        ):
            raise ValueError(
                "Rebuild old datasets under the native competition contract"
            )
        if self.manifest.get("control_hz") != 50:
            raise ValueError("Prepared data must use 50 Hz control")
        if (
            asset_identity is not None
            and self.manifest["asset_identity"] != asset_identity
        ):
            raise ValueError(
                "Prepared data and current evaluation assets/rules differ"
            )
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError("Empty prepared dataset")
        paths = set()
        for record in self.records:
            path = (self.root / record["path"]).resolve(strict=True)
            if not path.is_relative_to(self.root) or path in paths:
                raise ValueError(
                    "Dataset paths must be unique and stay inside the manifest directory"
                )
            paths.add(path)
            if (
                path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(f"Changed prepared clip: {record['path']}")
            if type(record["samples"]) is not int or record["samples"] <= 0:
                raise ValueError("Prepared clips must contain a positive sample count")
            if validate_contents:
                self._read(record)

    def _read(self, record):
        with np.load(self.root / record["path"], allow_pickle=False) as packet:
            if str(packet["format"]) != DATA_FORMAT:
                raise ValueError("Prepared clip has the wrong schema")
            result = {
                key: torch.from_numpy(packet[key])
                for key in ("obs", "target_pose")
            }
        n = len(result["obs"])
        for key, width in (
            ("obs", 1097),
            ("target_pose", 225),
        ):
            value = result[key]
            if (
                value.shape != (n, width)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid prepared {key}")
        if n != record["samples"]:
            raise ValueError("Prepared sample count differs from its manifest")
        return result

    def sample_count(self):
        return sum(row["samples"] for row in self.records)
