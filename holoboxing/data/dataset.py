"""Validated actual-state records for boxing supervision."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from holoboxing.evaluation.runtime.model_contract import CONTRACT, sha256

from .contract import same_data_assets
from .records import DATA_FORMAT


class PreparedDataset:
    def __init__(
        self,
        manifest,
        *,
        asset_identity=None,
        validate_contents=True,
        allow_incomplete_parent=False,
    ):
        self.manifest_path = Path(manifest).resolve(strict=True)
        self.root = self.manifest_path.parent
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.collection_recipe = self.manifest.get("collection_recipe")
        if self.collection_recipe is None:
            raise ValueError("Regenerate data with actual-state collection")
        if self.collection_recipe is not None:
            from holoboxing.evaluation.models import frozen_manifest

            from .collection import RECIPE

            files = frozen_manifest()["files"]
            lower = {
                name: files[name]["sha256"]
                for name in ("exported/model_10500.onnx", "config.yaml")
            }
            if (
                not isinstance(self.collection_recipe, dict)
                or self.manifest.get("state") != "complete"
                or any(
                    self.collection_recipe.get(k) != v
                    for k, v in RECIPE.items()
                )
                or self.collection_recipe.get("lower_identity") != lower
            ):
                raise ValueError(
                    "Incomplete or unsupported measured collection recipe"
                )
            parent = self.root.parent
            if (parent / "FAILED.json").exists():
                raise ValueError("Failed dataset preparation")
            if (
                not allow_incomplete_parent
                and (parent / "sources.json").exists()
                and not (parent / "COMPLETE.json").is_file()
            ):
                raise ValueError("Dataset preparation is incomplete")
            if (parent / "COMPLETE.json").is_file():
                complete = json.loads((parent / "COMPLETE.json").read_text())
                if (
                    complete.get("prepared_manifest_sha256")
                    != self.manifest_sha256
                ):
                    raise ValueError(
                        "Dataset differs from its completion receipt"
                    )
        if (
            self.manifest.get("format") != DATA_FORMAT
            or self.manifest.get("policy_contract") != CONTRACT
        ):
            raise ValueError(
                "Rebuild old datasets under the native competition contract"
            )
        if self.manifest.get("control_hz") != 50:
            raise ValueError("Prepared data must use 50 Hz control")
        if asset_identity is not None and not same_data_assets(
            self.manifest.get("asset_identity"), asset_identity
        ):
            raise ValueError("Prepared data and current data assets differ")
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError("Empty prepared dataset")
        paths = set()
        for record in self.records:
            if record.get("collection_recipe") != self.collection_recipe:
                raise ValueError("Mixed collection recipes")
            measured = (
                record.get("observation_source")
                == "measured_frozen_lower_rollout_current_state"
            )
            if measured != (self.collection_recipe is not None):
                raise ValueError(
                    "Measured observations require an explicit collection recipe"
                )
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
                raise ValueError(
                    "Prepared clips must contain a positive sample count"
                )
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
            if self.collection_recipe is not None:
                indices, windows = (
                    packet["sample_index"],
                    packet["window_index"],
                )
                n = len(result["obs"])
                if (
                    indices.shape != (n, 2)
                    or indices.dtype != np.int64
                    or windows.shape != (n, 2)
                    or windows.dtype != np.int64
                    or len(np.unique(indices, axis=0)) != n
                ):
                    raise ValueError(
                        "Invalid or duplicate measured sample IDs"
                    )
                frames, roles = indices.T
                start, core = windows.T
                if (
                    record["roles"] not in (1, 2)
                    or (roles < 0).any()
                    or (roles >= record["roles"]).any()
                    or (frames < 0).any()
                    or (frames + 5 >= record["frames"]).any()
                    or (frames // 3000 != (frames + 5) // 3000).any()
                    or (core != frames // 100 * 100).any()
                    or (
                        start != np.maximum(core // 3000 * 3000, core - 10)
                    ).any()
                ):
                    raise ValueError(
                        "Measured window or future labels cross a source boundary"
                    )
                if (
                    record["planned_samples"] != n + record["rejected_samples"]
                    or record["rejected_samples"] < 0
                ):
                    raise ValueError("Invalid measured coverage counts")
                from .collection import (
                    expected_retained_indices,
                    sample_indices,
                )

                if record["planned_samples"] != len(
                    sample_indices(record["frames"], record["roles"])
                ):
                    raise ValueError(
                        "Measured planned coverage has the wrong denominator"
                    )
                expected = expected_retained_indices(
                    record["frames"], record["early_failures"], record["roles"]
                )
                if not np.array_equal(indices, expected):
                    raise ValueError(
                        "Measured rows do not match the recorded window failures"
                    )
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

    def source_metadata(self, *, require_complete=True):
        """Resolve each prepared clip to its verified source and group split."""
        source_path = self.root.parent / "sources.json"
        complete_path = self.root.parent / "COMPLETE.json"
        if not source_path.is_file():
            raise ValueError(
                "Regenerate data: verified sources.json is required"
            )
        sources = json.loads(source_path.read_text())
        if sources.get("format") != "holoboxing_data_sources_v1":
            raise ValueError("Unsupported prepared source provenance")
        if require_complete:
            if not complete_path.is_file():
                raise ValueError("Dataset preparation is incomplete")
            complete = json.loads(complete_path.read_text())
            if complete.get("sources_sha256") != sha256(source_path):
                raise ValueError(
                    "Source provenance differs from its completion receipt"
                )
        rows = sources["records"]
        if sources.get("expected_source_clips") != len(
            self.manifest["records"]
        ) or len(rows) != len(self.manifest["records"]):
            raise ValueError("Source provenance does not cover prepared clips")
        groups = {}
        for source, record in zip(rows, self.manifest["records"], strict=True):
            if source["sha256"] != record["source_sha256"]:
                raise ValueError(
                    "Source provenance differs from prepared clip order"
                )
            dataset, group, split = (
                source.get(k) for k in ("dataset", "group", "split")
            )
            if (
                not isinstance(dataset, str)
                or not dataset
                or not isinstance(group, str)
                or not group
                or split not in ("train", "validation")
            ):
                raise ValueError(
                    "Prepared clips require a source dataset, group and train/validation split"
                )
            if record.get("source_group") != group:
                raise ValueError(
                    "Prepared source group differs from its provenance"
                )
            key = dataset, group
            if key in groups and groups[key] != split:
                raise ValueError(
                    "An original/mirror source group crosses data splits"
                )
            groups[key] = split
            for field in ("dataset", "split"):
                if field in record and record[field] != source[field]:
                    raise ValueError(
                        f"Prepared {field} differs from its source provenance"
                    )
        return rows

    def select_sources(
        self, *, split="train", dataset=None, allow_empty=False
    ):
        """Select a population before fitting statistics or consuming training rows."""
        if split not in ("train", "validation"):
            raise ValueError("Select the train or validation split explicitly")
        sources = self.source_metadata()
        self.records = [
            record
            for source, record in zip(
                sources, self.manifest["records"], strict=True
            )
            if source["split"] == split
            and (dataset is None or source["dataset"] == dataset)
        ]
        if not self.records and not allow_empty:
            detail = f"{dataset} " if dataset else ""
            raise ValueError(
                f"No {detail}{split} clips; prepare the required source data and preserve its split"
            )
        self.source_selection = {
            "dataset": dataset or "all",
            "split": split,
            "sources_sha256": sha256(self.root.parent / "sources.json"),
            "clips": len(self.records),
            "samples": self.sample_count(),
        }
        return self
