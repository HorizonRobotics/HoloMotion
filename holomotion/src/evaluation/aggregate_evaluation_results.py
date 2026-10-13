# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Aggregate full-clip evaluation results and validate selected-clip coverage."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from holomotion.src.evaluation.metrics import root_height_success
from holomotion.src.evaluation.mujoco_metric_tracker import (
    PROTOCOL_ID as COMMON_PROTOCOL_ID,
    ROOT_HEIGHT_SR_STANDARD_ID as METRIC_STANDARD_ID,
    ROOT_HEIGHT_SR_THRESHOLD_M as ROOT_HEIGHT_THRESHOLD_M,
)
from holomotion.src.motion_tracking.link_mode_mask import resolve_control_links

REQUIRED_METRICS = (
    "active_world_link_pos_error_m",
    "active_local_link_pos_error_m",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def metric_mean(rows, accessor):
    keys = sorted(set.intersection(*(set(accessor(row)) for row in rows)))
    return {
        key: float(np.mean([float(accessor(row)[key]) for row in rows]))
        for key in keys
        if all(accessor(row).get(key) is not None for row in rows)
    }


def root_height_result(row, threshold_m=ROOT_HEIGHT_THRESHOLD_M):
    """Validate stored SR using the shared v1.4.1 calculation."""
    value = row.get("common_metrics", {}).get("root_height_SR", {})
    if value.get("metric_standard_id") != METRIC_STANDARD_ID:
        raise ValueError("Missing or wrong Root-height SR standard")
    if value.get("threshold_m") != threshold_m:
        raise ValueError(
            "Root-height SR threshold differs from evaluation config"
        )
    error = value.get("max_root_height_error_m")
    success = value.get("height_tracking_success")
    if (
        type(error) not in (int, float)
        or not math.isfinite(float(error))
        or error < 0
    ):
        raise ValueError("Invalid max_root_height_error_m")
    if not isinstance(success, bool):
        raise ValueError("height_tracking_success must be a JSON boolean")
    if success != root_height_success(float(error), threshold_m):
        raise ValueError("Stored SR disagrees with configured threshold")
    return success, float(error)


def aggregate_shards(
    *,
    phase_dir: Path,
    workers: int,
    expected_clips: int,
    selection_path: Path,
    model: str,
    phase: str,
    output_dir: Path,
    failure_pos_err_thresh_m: float = ROOT_HEIGHT_THRESHOLD_M,
    link_eval: dict | None = None,
    reference_horizon: int = 5,
) -> dict:
    """Audit and aggregate one dataset-mode group into a fresh directory.

    Return the persisted summary. Incomplete coverage returns status="failed"
    and no official SR; callers must check status before publishing results.
    Invalid arguments raise ValueError without replacing existing outputs.
    """
    root_height_success(0.0, failure_pos_err_thresh_m)
    failure_pos_err_thresh_m = float(failure_pos_err_thresh_m)
    phase_dir = Path(phase_dir)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir)
    if reference_horizon < 1:
        raise ValueError("reference_horizon must be positive")
    if workers < 1 or expected_clips < 1:
        raise ValueError("workers and expected-clips must be positive")
    if output_dir.exists():
        raise ValueError(
            "output-dir must be fresh; never overwrite "
            "raw or published results"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    intended = selection["val_handpicked_motion_names"]
    if (
        not isinstance(intended, list)
        or len(intended) != expected_clips
        or not all(isinstance(name, str) and name for name in intended)
        or len(set(intended)) != len(intended)
    ):
        raise ValueError(
            "selection must contain exactly "
            "expected-clips unique nonempty names"
        )
    if set(intended).intersection(
        selection.get("train_handpicked_motion_names", [])
    ):
        raise ValueError("selection has train/val overlap")

    worker_rows = []
    worker_summaries = []
    missing = []
    for worker in range(workers):
        root = phase_dir / "results" / f"worker{worker}"
        per_clip = root / "per_clip.json"
        summary = root / "summary.json"
        if not per_clip.is_file() or not summary.is_file():
            missing.append(worker)
            continue
        rows = json.loads(per_clip.read_text(encoding="utf-8"))
        worker_rows.extend(rows)
        worker_summaries.append(
            json.loads(summary.read_text(encoding="utf-8"))
        )

    keys = [row.get("clip") for row in worker_rows]
    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError("result clip names must be nonempty strings")
    missing_clips = sorted(set(intended) - set(keys))
    unexpected_clips = sorted(set(keys) - set(intended))
    duplicates = sorted(
        key for key, count in Counter(keys).items() if count > 1
    )
    failed = [
        {"clip": row.get("clip"), "error": row.get("error")}
        for row in worker_rows
        if row.get("status") != "completed"
    ]
    complete = [row for row in worker_rows if row.get("status") == "completed"]
    incomplete_metrics = [
        row.get("clip")
        for row in complete
        if not row.get("common_metrics", {}).get("coverage_complete", False)
    ]
    invalid_rollout_contract = []
    for row in complete:
        expected = row.get("expected_policy_steps")
        common = row.get("common_metrics", {})
        if not (
            isinstance(expected, int)
            and expected > 0
            and row.get("executed_policy_steps") == expected
            and row.get("reference_frames") == expected + reference_horizon
            and common.get("expected_steps") == expected
            and common.get("observed_steps") == expected
        ):
            invalid_rollout_contract.append(row.get("clip"))
    invalid_common_metrics = []
    for row in complete:
        common = row.get("common_metrics", {})
        metrics = common.get("metrics", {})
        if common.get("protocol_id") != COMMON_PROTOCOL_ID or any(
            type(metrics.get(key)) not in (int, float)
            or not math.isfinite(metrics[key])
            or metrics[key] < 0
            for key in REQUIRED_METRICS
        ):
            invalid_common_metrics.append(row.get("clip"))
    invalid_root_height = []
    root_height_rows = []
    for row in complete:
        try:
            success, error = root_height_result(row, failure_pos_err_thresh_m)
            root_height_rows.append((row, success, error))
        except ValueError as exc:
            invalid_root_height.append(
                {"clip": row.get("clip"), "error": str(exc)}
            )

    invalid_control_clips = []
    expected_active = None
    for row in complete:
        common = row.get("common_metrics", {})
        actual = common.get("active_link_names")
        try:
            if link_eval is not None:
                selected, _ = resolve_control_links(
                    common.get("all14_link_names", []),
                    **link_eval,
                )
            else:
                selected = (
                    expected_active if expected_active is not None else actual
                )
            if selected is not None:
                if (
                    not actual
                    or len(actual) != len(set(actual))
                    or set(actual) != set(selected)
                ):
                    raise ValueError(
                        "Result active points differ from configuration"
                    )
                expected_active = list(selected)
        except ValueError:
            invalid_control_clips.append(row.get("clip"))

    coverage_passed = (
        not missing
        and not invalid_control_clips
        and not missing_clips
        and not unexpected_clips
        and not invalid_common_metrics
        and not duplicates
        and not failed
        and not incomplete_metrics
        and not invalid_rollout_contract
        and not invalid_root_height
        and len(complete) == expected_clips
    )

    common_metrics = (
        metric_mean(
            complete,
            lambda row: row.get("common_metrics", {}).get("metrics", {}),
        )
        if complete
        else {}
    )

    success_clip_count = sum(
        int(success) for _row, success, _error in root_height_rows
    )
    root_height_micro = (
        100.0 * success_clip_count / expected_clips if expected_clips else 0.0
    )
    coverage = {
        "passed": coverage_passed,
        "failure_pos_err_thresh_m": failure_pos_err_thresh_m,
        "active_link_names": expected_active,
        "invalid_control_clips": invalid_control_clips,
        "model": model,
        "phase": phase,
        "workers": workers,
        "reference_horizon": reference_horizon,
        "missing_workers": missing,
        "missing_clips": missing_clips,
        "unexpected_clips": unexpected_clips,
        "invalid_common_metric_clips": invalid_common_metrics,
        "selection_sha256": hashlib.sha256(
            selection_path.read_bytes()
        ).hexdigest(),
        "expected_clips": expected_clips,
        "observed_rows": len(worker_rows),
        "completed_clips": len(complete),
        "duplicate_clips": duplicates,
        "failed_clips": failed,
        "incomplete_metric_clips": incomplete_metrics,
        "invalid_rollout_contract_clips": invalid_rollout_contract,
        "invalid_root_height_clips": invalid_root_height,
    }
    root_height_sr = {
        "metric_standard_id": METRIC_STANDARD_ID,
        "criterion": (
            "max_t abs(robot_root_z-reference_root_z) <= "
            f"{failure_pos_err_thresh_m:g} m"
        ),
        "threshold_m": failure_pos_err_thresh_m,
        "success_clips": success_clip_count,
        "total_clips": expected_clips,
        "root_height_SR_percent": root_height_micro
        if coverage_passed
        else None,
        "aggregation": "equal_clip_micro",
    }
    summary = {
        "status": "completed" if coverage_passed else "failed",
        "model": model,
        "phase": phase,
        "metric_standard_id": METRIC_STANDARD_ID,
        "coverage": coverage,
        "root_height_SR": root_height_sr,
        "equal_clip_common_metrics_completed_only": common_metrics,
        "worker_summaries": len(worker_summaries),
    }
    write_json(output_dir / "per_clip.json", worker_rows)
    write_json(output_dir / "coverage.json", coverage)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "done.json",
        {"status": summary["status"], "coverage": coverage},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        dest="selection_path",
        type=Path,
        required=True,
        help="Phase selection with val_handpicked_motion_names",
    )
    parser.add_argument("--reference-horizon", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--failure-pos-err-thresh-m", type=float)
    parser.add_argument("--active-link-names", nargs="+")
    args = parser.parse_args()
    values = vars(args)
    config_path = values.pop("config")
    active_links = values.pop("active_link_names")
    try:
        config = {}
        if config_path:
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(
                OmegaConf.load(config_path), resolve=True
            )
        if values["failure_pos_err_thresh_m"] is None:
            values["failure_pos_err_thresh_m"] = config.get(
                "failure_pos_err_thresh_m", ROOT_HEIGHT_THRESHOLD_M
            )
        if values["reference_horizon"] is None:
            values["reference_horizon"] = config.get(
                "evaluation_data", {}
            ).get("reference_horizon", 5)
        values["link_eval"] = (
            {"mask_mode": "custom", "active_link_names": active_links}
            if active_links is not None
            else config.get("link_eval")
        )
        summary = aggregate_shards(**values)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary["coverage"], ensure_ascii=False), flush=True)
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
