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

"""CPU regressions for MuJoCo metrics and clip-level result aggregation."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from holomotion.src.evaluation import mujoco_metric_tracker as metrics
from holomotion.src.evaluation.aggregate_evaluation_results import (
    aggregate_shards,
)

ROOT = Path(__file__).resolve().parents[1]


def row(name, height=0.1, world=0.2, steps=10, threshold=0.25):
    return {
        "clip": name,
        "status": "completed",
        "expected_policy_steps": steps,
        "executed_policy_steps": steps,
        "reference_frames": steps + 5,
        "common_metrics": {
            "all14_link_names": ["pelvis"] + [f"link{i}" for i in range(13)],
            "active_link_names": ["pelvis"] + [f"link{i}" for i in range(4)],
            "protocol_id": metrics.PROTOCOL_ID,
            "coverage_complete": True,
            "expected_steps": steps,
            "observed_steps": steps,
            "metrics": {
                "active_world_link_pos_error_m": world,
                "active_local_link_pos_error_m": 0.05,
            },
            "root_height_SR": {
                "metric_standard_id": metrics.ROOT_HEIGHT_SR_STANDARD_ID,
                "criterion": (
                    "max_t abs(robot_root_z-reference_root_z) <= 0.25 m"
                ),
                "threshold_m": threshold,
                "max_root_height_error_m": height,
                "height_tracking_success": height <= threshold,
            },
        },
    }


def run_aggregate(tmp_path, rows, expected=None, workers=1, extra_args=()):
    raw = tmp_path / "phase/results/worker0"
    raw.mkdir(parents=True)
    (raw / "per_clip.json").write_text(json.dumps(rows))
    (raw / "summary.json").write_text("{}")
    expected = expected if expected is not None else [r["clip"] for r in rows]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"val_handpicked_motion_names": expected}))
    output = tmp_path / "aggregate"
    cmd = [
        sys.executable,
        "-m",
        "holomotion.src.evaluation.aggregate_evaluation_results",
        "--phase-dir",
        str(tmp_path / "phase"),
        "--workers",
        str(workers),
        "--expected-clips",
        str(len(expected)),
        "--model",
        "fixture",
        "--phase",
        "full29",
        "--selection",
        str(selection),
        "--output-dir",
        str(output),
    ]
    cmd.extend(extra_args)
    result = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    summary = (
        json.loads((output / "summary.json").read_text())
        if output.exists()
        else None
    )
    return result, summary, cmd


def test_sr_failure_still_contributes_to_mkpe_and_clips_have_equal_weight(
    tmp_path,
):
    result, summary, cmd = run_aggregate(
        tmp_path,
        [
            row("short", height=0.25, world=0.1, steps=5),
            row("long", height=0.26, world=0.9, steps=100),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert summary["root_height_SR"]["root_height_SR_percent"] == 50
    assert (
        summary["equal_clip_common_metrics_completed_only"][
            "active_world_link_pos_error_m"
        ]
        == 0.5
    )
    before = (tmp_path / "aggregate/summary.json").read_bytes()
    rerun = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert rerun.returncode != 0
    assert (tmp_path / "aggregate/summary.json").read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_clip",
        "duplicate",
        "missing_clip",
        "execution_failed",
        "incomplete",
        "wrong_steps",
        "wrong_protocol",
        "wrong_threshold",
        "wrong_sr",
        "missing_metric",
        "negative_metric",
        "missing_worker",
    ],
)
def test_invalid_coverage_never_produces_official_sr(tmp_path, mutation):
    rows = [row("a"), row("b")]
    workers = 1
    if mutation == "wrong_clip":
        rows[1]["clip"] = "substitute"
    elif mutation == "duplicate":
        rows[1]["clip"] = "a"
    elif mutation == "missing_clip":
        rows.pop()
    elif mutation == "execution_failed":
        rows[1]["status"] = "failed"
    elif mutation == "incomplete":
        rows[1]["common_metrics"]["coverage_complete"] = False
    elif mutation == "wrong_steps":
        rows[1]["executed_policy_steps"] = 9
    elif mutation == "wrong_protocol":
        rows[1]["common_metrics"]["protocol_id"] = "other"
    elif mutation == "wrong_threshold":
        rows[1]["common_metrics"]["root_height_SR"]["threshold_m"] = 0.5
    elif mutation == "wrong_sr":
        rows[1]["common_metrics"]["root_height_SR"][
            "height_tracking_success"
        ] = False
    elif mutation == "missing_metric":
        rows[1]["common_metrics"]["metrics"].pop(
            "active_world_link_pos_error_m"
        )
    elif mutation == "negative_metric":
        rows[1]["common_metrics"]["metrics"][
            "active_world_link_pos_error_m"
        ] = -1
    elif mutation == "missing_worker":
        workers = 2
    result, summary, _ = run_aggregate(tmp_path, rows, ["a", "b"], workers)
    assert result.returncode != 0
    assert summary["status"] == "failed"
    assert summary["root_height_SR"]["root_height_SR_percent"] is None


def simulator_state_bytes(data):
    return {
        name: np.asarray(getattr(data, name)).tobytes()
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xpos", "xquat")
    }


def test_passive_observer_global_local_and_full_trajectory_sr():
    mujoco = pytest.importorskip("mujoco")
    names = ["pelvis"] + [f"link{i}" for i in range(13)]
    children = "".join(
        f'<body name="link{i}" pos="{i / 10} 0 0">'
        '<geom size=".01" mass=".1"/></body>'
        for i in range(13)
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis" pos="0 0 1"><freejoint/>'
        '<geom size=".1" mass="1"/>'
        + children
        + "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names
    ]
    target = data.xpos[ids].copy()
    quaternions = data.xquat[ids].copy()
    target += np.array([0.1, 0.2, 0.3])
    before = simulator_state_bytes(data)
    tracker = metrics.MujocoMetricTracker(model, names, names[:5])
    tracker.observe(data, target, quaternions, 0)
    tracker.observe(data, data.xpos[ids], quaternions, 1)
    result = tracker.report(2)
    assert simulator_state_bytes(data) == before
    assert result["coverage_complete"]
    assert result["metrics"]["active_world_link_pos_error_m"] == pytest.approx(
        np.sqrt(0.14) / 2
    )
    assert result["metrics"]["active_local_link_pos_error_m"] == pytest.approx(
        0
    )
    assert not result["root_height_SR"]["height_tracking_success"]
    assert not tracker.report(3)["coverage_complete"]


@pytest.mark.parametrize("missing_worker", [False, True])
def test_python_api_matches_cli(tmp_path, missing_worker):
    rows = [row("first", world=0.1), row("second", height=0.3, world=0.5)]
    workers = 2 if missing_worker else 1
    _, cli_summary, _ = run_aggregate(tmp_path, rows, workers=workers)
    api_summary = aggregate_shards(
        phase_dir=tmp_path / "phase",
        workers=workers,
        expected_clips=2,
        selection_path=tmp_path / "selection.json",
        model="fixture",
        phase="full29",
        output_dir=tmp_path / "api_output",
    )
    assert api_summary == cli_summary
    assert (
        json.loads((tmp_path / "api_output/summary.json").read_text())
        == api_summary
    )


def test_python_api_refuses_existing_output(tmp_path):
    run_aggregate(tmp_path, [row("a")])
    before = (tmp_path / "aggregate/summary.json").read_bytes()
    with pytest.raises(ValueError, match="output-dir must be fresh"):
        aggregate_shards(
            phase_dir=tmp_path / "phase",
            workers=1,
            expected_clips=1,
            selection_path=tmp_path / "selection.json",
            model="fixture",
            phase="full29",
            output_dir=tmp_path / "aggregate",
        )
    assert (tmp_path / "aggregate/summary.json").read_bytes() == before


def test_shared_root_height_definition_matches_legacy():
    from holomotion.src.evaluation.metrics import (
        compute_root_height_error,
        root_height_success,
    )

    actual = np.array([0.9, 0.5, 1.0])
    reference = np.array([0.8, 0.75, 0.7])
    errors = compute_root_height_error(actual, reference)
    np.testing.assert_array_equal(errors, np.abs(actual - reference))
    maximum = float(np.max(errors))
    for threshold in (0.0, 0.25, 0.4):
        assert root_height_success(maximum, threshold) == bool(
            maximum <= threshold
        )
    assert root_height_success(0.25, 0.25)
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            root_height_success(0.0, invalid)


@pytest.mark.parametrize(
    "selected", [["pelvis"], ["link2", "pelvis"], ["link4"]]
)
def test_named_control_configuration_preserves_schema_order(selected):
    from holomotion.src.motion_tracking.link_mode_mask import (
        resolve_control_links,
    )

    names = ["pelvis"] + [f"link{i}" for i in range(13)]
    active, mask = resolve_control_links(
        names,
        mask_mode="custom",
        active_link_names=selected,
    )
    assert active == [name for name in names if name in selected]
    assert [
        name for name, enabled in zip(names, mask, strict=True) if enabled
    ] == active
    assert all(resolve_control_links(names)[1])


@pytest.mark.parametrize(
    "selected", [[], ["missing"], ["pelvis", "pelvis"], "pelvis"]
)
def test_bad_control_selection_is_rejected(selected):
    from holomotion.src.motion_tracking.link_mode_mask import (
        resolve_control_links,
    )

    with pytest.raises(ValueError):
        resolve_control_links(
            ["pelvis"],
            mask_mode="custom",
            active_link_names=selected,
        )


def test_yaml_configuration_controls_sr_and_checks_point_identity(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        "failure_pos_err_thresh_m: 0.4\nlink_eval:\n  mask_mode: custom\n"
        "  active_link_names: [pelvis]\n"
    )
    rows = [
        row("a", height=0.3, threshold=0.4),
        row("b", height=0.5, threshold=0.4),
    ]
    for item in rows:
        item["common_metrics"]["active_link_names"] = ["pelvis"]
    result, summary, _ = run_aggregate(
        tmp_path, rows, extra_args=["--config", str(config)]
    )
    assert result.returncode == 0, result.stderr
    assert summary["root_height_SR"]["threshold_m"] == 0.4
    assert summary["root_height_SR"]["root_height_SR_percent"] == 50
    assert summary["coverage"]["active_link_names"] == ["pelvis"]


@pytest.mark.parametrize(
    "overrides",
    [
        ["--failure-pos-err-thresh-m", "0.4"],
        ["--active-link-names", "pelvis"],
    ],
)
def test_cannot_mix_thresholds_or_control_sets(tmp_path, overrides):
    result, summary, _ = run_aggregate(
        tmp_path, [row("a")], extra_args=overrides
    )
    assert result.returncode != 0
    assert summary["root_height_SR"]["root_height_SR_percent"] is None


def test_configured_observer_uses_active_points_but_sr_always_uses_root():
    import mujoco

    names = ["pelvis"] + [f"link{i}" for i in range(13)]
    children = "".join(
        f'<body name="{name}"><geom size=".01" mass=".1"/></body>'
        for name in names[1:]
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis" pos="0 0 1"><freejoint/>'
        '<geom size=".1" mass="1"/>'
        + children
        + "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in names
    ]
    target = data.xpos[ids].copy()
    target[0, 2] += 0.3
    target[2, 0] += (
        10.0  # inactive link must not enter the active-point metric
    )
    for threshold in (0.25, 0.4):
        tracker = metrics.MujocoMetricTracker.from_config(
            model,
            names,
            {
                "failure_pos_err_thresh_m": threshold,
                "link_eval": {
                    "mask_mode": "custom",
                    "active_link_names": ["link0"],
                },
            },
        )
        tracker.observe(data, target, data.xquat[ids], 0)
        result = tracker.report(1)
        assert result["metrics"][
            "active_world_link_pos_error_m"
        ] == pytest.approx(0)
        assert result["root_height_SR"]["height_tracking_success"] == (
            threshold == 0.4
        )
        assert tracker.control_mask.sum() == 1


@pytest.mark.parametrize("root_index", [None, 0, 4])
def test_shared_position_error_preserves_framewise_formulas(root_index):
    from holomotion.src.evaluation.metrics import compute_position_error

    rng = np.random.default_rng(7)
    actual, reference = rng.normal(size=(2, 11, 14, 3))
    delta = actual - reference
    if root_index is not None:
        delta -= delta[:, [root_index], :]
    expected = np.linalg.norm(delta, axis=-1)
    np.testing.assert_array_equal(
        compute_position_error(actual, reference, root_index=root_index),
        expected,
    )
    for frame in range(11):
        np.testing.assert_array_equal(
            compute_position_error(
                actual[frame], reference[frame], root_index=root_index
            ),
            expected[frame],
        )


def test_translation_only_local_error_retains_heading_difference():
    from holomotion.src.evaluation.metrics import compute_position_error

    reference = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    actual = np.array([[2.0, 3.0, 1.0], [2.0, 4.0, 1.0]])
    np.testing.assert_allclose(
        compute_position_error(actual, reference, root_index=0),
        [0.0, np.sqrt(2)],
    )
