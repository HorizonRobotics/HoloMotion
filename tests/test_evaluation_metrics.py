import csv
import json
from pathlib import Path

import numpy as np

from holomotion.src.evaluation.metrics import (
    _compute_clip_stability_summary,
    _per_frame_metrics_from_npz,
    offline_evaluate_dumped_npzs,
)


def _make_eval_data(
    robot_dof_torque: np.ndarray,
    *,
    robot_dof_vel: np.ndarray | None = None,
    robot_dof_acc: np.ndarray | None = None,
    robot_action_rate: np.ndarray | None = None,
    robot_low_level_dof_torque: np.ndarray | None = None,
    robot_global_angular_velocity: np.ndarray | None = None,
    robot_low_level_foot_contact: np.ndarray | None = None,
    robot_low_level_foot_normal_force: np.ndarray | None = None,
    robot_low_level_foot_tangent_speed: np.ndarray | None = None,
    robot_moe_expert_logits: np.ndarray | None = None,
):
    num_frames = int(robot_dof_torque.shape[0])
    num_dofs = int(robot_dof_torque.shape[1])

    root = np.zeros((num_frames, 1, 3), dtype=np.float32)
    child = np.tile(
        np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32), (num_frames, 1, 1)
    )
    global_translation = np.concatenate([root, child], axis=1)
    global_rotation = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (num_frames, 2, 1),
    )

    zeros_dof = np.zeros((num_frames, num_dofs), dtype=np.float32)

    payload = {
        "ref_dof_pos": zeros_dof.copy(),
        "robot_dof_pos": zeros_dof.copy(),
        "ref_dof_vel": zeros_dof.copy(),
        "ref_global_translation": global_translation.copy(),
        "robot_global_translation": global_translation.copy(),
        "ref_global_rotation_quat": global_rotation.copy(),
        "robot_global_rotation_quat": global_rotation.copy(),
        "ref_global_velocity": np.zeros((num_frames, 2, 3), dtype=np.float32),
        "ref_global_angular_velocity": np.zeros(
            (num_frames, 2, 3), dtype=np.float32
        ),
        "robot_global_velocity": np.zeros(
            (num_frames, 2, 3), dtype=np.float32
        ),
        "robot_global_angular_velocity": (
            np.zeros((num_frames, 2, 3), dtype=np.float32)
            if robot_global_angular_velocity is None
            else robot_global_angular_velocity.astype(np.float32)
        ),
        "robot_dof_vel": (
            zeros_dof.copy()
            if robot_dof_vel is None
            else robot_dof_vel.astype(np.float32)
        ),
        "robot_dof_acc": (
            zeros_dof.copy()
            if robot_dof_acc is None
            else robot_dof_acc.astype(np.float32)
        ),
        "robot_dof_torque": robot_dof_torque.astype(np.float32),
        "robot_action_rate": (
            np.zeros((num_frames,), dtype=np.float32)
            if robot_action_rate is None
            else robot_action_rate.astype(np.float32)
        ),
    }
    if robot_low_level_dof_torque is not None:
        payload["robot_low_level_dof_torque"] = (
            robot_low_level_dof_torque.astype(np.float32)
        )
    if robot_low_level_foot_contact is not None:
        payload["robot_low_level_foot_contact"] = (
            robot_low_level_foot_contact.astype(np.float32)
        )
    if robot_low_level_foot_normal_force is not None:
        payload["robot_low_level_foot_normal_force"] = (
            robot_low_level_foot_normal_force.astype(np.float32)
        )
    if robot_low_level_foot_tangent_speed is not None:
        payload["robot_low_level_foot_tangent_speed"] = (
            robot_low_level_foot_tangent_speed.astype(np.float32)
        )
    if robot_moe_expert_logits is not None:
        payload["robot_moe_expert_logits"] = robot_moe_expert_logits.astype(
            np.float32
        )
    return payload


def test_per_frame_metrics_include_torque_jump_diagnostics():
    constant_torque = np.ones((4, 2), dtype=np.float32)
    constant_df = _per_frame_metrics_from_npz(
        motion_key="constant",
        data=_make_eval_data(constant_torque),
        robot_control_dt=0.5,
    )

    assert "mean_torque_jump_norm" in constant_df.columns
    assert "mean_torque_jump_ratio" in constant_df.columns
    assert np.isnan(constant_df["mean_torque_jump_norm"].iloc[0])
    assert np.isnan(constant_df["mean_torque_jump_ratio"].iloc[0])
    np.testing.assert_allclose(
        np.nan_to_num(constant_df["mean_torque_jump_norm"].to_numpy()),
        np.zeros(4, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.nan_to_num(constant_df["mean_torque_jump_ratio"].to_numpy()),
        np.zeros(4, dtype=np.float64),
    )

    jump_torque = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    jump_df = _per_frame_metrics_from_npz(
        motion_key="jump",
        data=_make_eval_data(jump_torque),
        robot_control_dt=0.5,
    )

    assert jump_df["mean_torque_jump_norm"].iloc[2] > 3.9
    assert jump_df["mean_torque_jump_ratio"].iloc[2] > 1.9


def test_offline_evaluate_dumped_npzs_exports_torque_jump_summary_metrics(
    tmp_path: Path,
):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    jump_torque_50hz = np.tile(
        np.array([[1.0, 0.0]], dtype=np.float32), (4, 1)
    )
    jump_torque_low_level = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    payload = _make_eval_data(
        jump_torque_50hz,
        robot_low_level_dof_torque=jump_torque_low_level,
    )
    payload["metadata"] = np.array(
        json.dumps({"clip_length": 4, "robot_low_level_torque_dt": 0.005}),
        dtype=np.str_,
    )

    np.savez_compressed(eval_dir / "demo_clip.npz", **payload)

    output_json_path = eval_dir / "summary.json"
    result = offline_evaluate_dumped_npzs(
        npz_dir=str(eval_dir),
        output_json_path=str(output_json_path),
    )

    per_clip = result["per_clip"][0]
    for key in (
        "mean_torque_jump_norm",
        "p95_torque_jump_norm",
        "mean_torque_jump_ratio",
        "p95_torque_jump_ratio",
    ):
        assert key in per_clip
        assert key in result["dataset"]["mean"]

    assert per_clip["mean_dof_torque"] == 1.0
    assert per_clip["p95_torque_jump_norm"] > 300.0
    assert per_clip["p95_torque_jump_ratio"] > 1.0

    with output_json_path.open("r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert "p95_torque_jump_ratio" in written["dataset"]["mean"]

    csv_path = eval_dir / "per_clip_metrics.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    assert "p95_torque_jump_ratio" in row
    assert "mean_torque_jump_norm" in row


def test_compute_clip_stability_summary_detects_chatter_and_support_events():
    num_frames = 50
    num_low_level = 200
    policy_dt = 0.02
    low_level_dt = 0.005

    t_policy = np.arange(num_frames, dtype=np.float32) * policy_dt
    t_low = np.arange(num_low_level, dtype=np.float32) * low_level_dt

    smooth_ang_vel = np.zeros((num_frames, 2, 3), dtype=np.float32)
    smooth_ang_vel[:, 0, 0] = 0.2 * np.sin(2.0 * np.pi * 1.0 * t_policy)

    unstable_ang_vel = smooth_ang_vel.copy()
    unstable_ang_vel[:, 0, 0] += 0.7 * np.sin(
        2.0 * np.pi * 8.0 * t_policy
    ).astype(np.float32)
    unstable_ang_vel[:, 0, 1] += 0.4 * np.sin(
        2.0 * np.pi * 6.0 * t_policy
    ).astype(np.float32)

    smooth_low_level_torque = np.zeros((num_low_level, 2), dtype=np.float32)
    smooth_low_level_torque[:, 0] = np.sin(2.0 * np.pi * 1.0 * t_low)

    unstable_low_level_torque = smooth_low_level_torque.copy()
    unstable_low_level_torque[:, 0] += 0.8 * np.sin(
        2.0 * np.pi * 15.0 * t_low
    ).astype(np.float32)
    unstable_low_level_torque[80:85, 0] += 2.5
    unstable_low_level_torque[120:123, 0] -= 2.5

    stable_contact = np.zeros((num_low_level, 2), dtype=np.float32)
    stable_contact[:100, 0] = 1.0
    stable_contact[100:, 1] = 1.0
    stable_normal_force = stable_contact * np.array(
        [[80.0, 75.0]], dtype=np.float32
    )
    stable_tangent_speed = stable_contact * 0.01

    unstable_contact = np.zeros((num_low_level, 2), dtype=np.float32)
    for start in range(0, num_low_level, 10):
        unstable_contact[start : start + 5, 0] = 1.0
        unstable_contact[start + 5 : start + 10, 1] = 1.0
    unstable_normal_force = unstable_contact * 60.0
    touchdown_mask = unstable_contact.copy()
    touchdown_mask[1:] = np.clip(
        unstable_contact[1:] - unstable_contact[:-1], a_min=0.0, a_max=None
    )
    unstable_normal_force += touchdown_mask * 120.0
    unstable_tangent_speed = unstable_contact * 0.25

    smooth_metrics = _compute_clip_stability_summary(
        data=_make_eval_data(
            np.zeros((num_frames, 2), dtype=np.float32),
            robot_low_level_dof_torque=smooth_low_level_torque,
            robot_global_angular_velocity=smooth_ang_vel,
            robot_low_level_foot_contact=stable_contact,
            robot_low_level_foot_normal_force=stable_normal_force,
            robot_low_level_foot_tangent_speed=stable_tangent_speed,
        ),
        robot_control_dt=policy_dt,
        low_level_contact_dt=low_level_dt,
    )
    unstable_metrics = _compute_clip_stability_summary(
        data=_make_eval_data(
            np.zeros((num_frames, 2), dtype=np.float32),
            robot_low_level_dof_torque=unstable_low_level_torque,
            robot_global_angular_velocity=unstable_ang_vel,
            robot_low_level_foot_contact=unstable_contact,
            robot_low_level_foot_normal_force=unstable_normal_force,
            robot_low_level_foot_tangent_speed=unstable_tangent_speed,
        ),
        robot_control_dt=policy_dt,
        low_level_contact_dt=low_level_dt,
    )

    assert (
        unstable_metrics["torque_chatter_hf_ratio"]
        > smooth_metrics["torque_chatter_hf_ratio"]
    )
    assert (
        unstable_metrics["torque_jump_burst_max"]
        > smooth_metrics["torque_jump_burst_max"]
    )
    assert (
        unstable_metrics["torso_rp_hf_ratio"]
        > smooth_metrics["torso_rp_hf_ratio"]
    )
    assert (
        unstable_metrics["torso_rp_angacc_p95"]
        > smooth_metrics["torso_rp_angacc_p95"]
    )
    assert (
        unstable_metrics["foot_contact_toggle_rate"]
        > smooth_metrics["foot_contact_toggle_rate"]
    )
    assert (
        unstable_metrics["foot_impact_force_p95"]
        > smooth_metrics["foot_impact_force_p95"]
    )
    assert (
        unstable_metrics["stance_slip_speed_p95"]
        > smooth_metrics["stance_slip_speed_p95"]
    )


def test_compute_clip_stability_summary_reports_expert_switching_js_div():
    num_frames = 8
    stable_logits = np.tile(
        np.array(
            [
                [8.0, -4.0, -4.0],
                [-4.0, 8.0, -4.0],
            ],
            dtype=np.float32,
        )[None, :, :],
        (num_frames, 1, 1),
    )
    switching_logits = stable_logits.copy()
    switching_logits[1::2, 0, :] = np.array(
        [-4.0, 8.0, -4.0], dtype=np.float32
    )
    switching_logits[1::2, 1, :] = np.array(
        [-4.0, -4.0, 8.0], dtype=np.float32
    )

    stable_metrics = _compute_clip_stability_summary(
        data=_make_eval_data(
            np.zeros((num_frames, 2), dtype=np.float32),
            robot_moe_expert_logits=stable_logits,
        ),
        robot_control_dt=0.02,
        low_level_contact_dt=0.02,
    )
    switching_metrics = _compute_clip_stability_summary(
        data=_make_eval_data(
            np.zeros((num_frames, 2), dtype=np.float32),
            robot_moe_expert_logits=switching_logits,
        ),
        robot_control_dt=0.02,
        low_level_contact_dt=0.02,
    )

    assert stable_metrics["expert_switching_js_div"] < 1e-6
    assert (
        switching_metrics["expert_switching_js_div"]
        > stable_metrics["expert_switching_js_div"]
    )


def test_offline_evaluate_dumped_npzs_reports_nan_contact_metrics_for_legacy_npz(
    tmp_path: Path,
):
    eval_dir = tmp_path / "legacy_eval"
    eval_dir.mkdir()

    payload = _make_eval_data(np.ones((8, 2), dtype=np.float32))
    payload["metadata"] = np.array(
        json.dumps({"clip_length": 8, "robot_low_level_torque_dt": 0.005}),
        dtype=np.str_,
    )
    np.savez_compressed(eval_dir / "legacy_clip.npz", **payload)

    output_json_path = eval_dir / "summary.json"
    result = offline_evaluate_dumped_npzs(
        npz_dir=str(eval_dir),
        output_json_path=str(output_json_path),
    )

    per_clip = result["per_clip"][0]
    for key in (
        "torque_chatter_hf_ratio",
        "torque_jump_burst_max",
        "torso_rp_hf_ratio",
        "torso_rp_angacc_p95",
        "foot_contact_toggle_rate",
        "foot_impact_force_p95",
        "stance_slip_speed_p95",
        "expert_switching_js_div",
    ):
        assert key in per_clip
        assert key in result["dataset"]["mean"]

    assert np.isnan(per_clip["foot_contact_toggle_rate"])
    assert np.isnan(per_clip["foot_impact_force_p95"])
    assert np.isnan(per_clip["stance_slip_speed_p95"])
    assert np.isnan(per_clip["expert_switching_js_div"])
