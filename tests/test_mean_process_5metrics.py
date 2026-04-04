import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.scripts.evaluation import mean_process_5metrics


LEGACY_METRICS = [
    "mpjpe_g",
    "mpjpe_l",
    "whole_body_joints_dist",
    "root_vel_error",
    "root_r_error",
    "root_p_error",
    "root_y_error",
    "root_height_error",
    "mean_dof_vel",
    "mean_dof_acc",
    "mean_dof_torque",
    "mean_action_rate",
    "success",
]

TORQUE_JUMP_METRICS = [
    "mean_torque_jump_norm",
    "p95_torque_jump_norm",
    "mean_torque_jump_ratio",
    "p95_torque_jump_ratio",
]


def test_macro_report_appends_torque_jump_columns_to_legacy_tables(
    tmp_path, monkeypatch
):
    json_path = tmp_path / "model_a.json"
    payload = {
        "per_clip": [
            {
                "motion_key": "clips_AMASS_demo",
                "mpjpe_g": 1.0,
                "mpjpe_l": 2.0,
                "whole_body_joints_dist": 3.0,
                "root_vel_error": 4.0,
                "root_r_error": 5.0,
                "root_p_error": 6.0,
                "root_y_error": 7.0,
                "root_height_error": 8.0,
                "mean_dof_vel": 9.0,
                "mean_dof_acc": 10.0,
                "mean_dof_torque": 11.0,
                "mean_action_rate": 12.0,
                "success": 1.0,
                "mean_torque_jump_norm": 13.0,
                "p95_torque_jump_norm": 14.0,
                "mean_torque_jump_ratio": 15.0,
                "p95_torque_jump_ratio": 16.0,
            }
        ]
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    mean_df, _ = mean_process_5metrics.process_data(str(tmp_path))

    assert mean_df.columns.tolist() == [
        "Method",
        "Dataset",
        *LEGACY_METRICS,
        *TORQUE_JUMP_METRICS,
    ]

    captured_headers = {}

    def _fake_tabulate(_rows, headers, **_kwargs):
        captured_headers["headers"] = headers
        return "fake-table"

    monkeypatch.setattr(mean_process_5metrics, "tabulate", _fake_tabulate)

    report_path = (
        mean_process_5metrics.generate_macro_mean_report_from_json_dir(
            str(tmp_path)
        )
    )

    tsv_path = tmp_path / "sub_dataset_macro_mean_metrics.tsv"
    header = tsv_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header == [
        "Dataset",
        "Global Bodylink Pos Err",
        "Local Bodylink Pos Err",
        "Dof Position Err",
        "Root Vel Err",
        "Root Roll Err",
        "Root Pitch Err",
        "Root Yaw Err",
        "Root Height Err",
        "Mean Dof Vel",
        "Mean Dof Acc",
        "Mean Dof Torque",
        "Mean Action Rate",
        "Success Rate",
        "Mean Torque Jump Norm",
        "P95 Torque Jump Norm",
        "Mean Torque Jump Ratio",
        "P95 Torque Jump Ratio",
    ]
    assert captured_headers["headers"] == header

    report_text = Path(report_path).read_text(encoding="utf-8")
    legacy_index = report_text.index("Success Rate")
    torque_index = report_text.index("Mean Torque Jump Norm")
    assert torque_index > legacy_index
