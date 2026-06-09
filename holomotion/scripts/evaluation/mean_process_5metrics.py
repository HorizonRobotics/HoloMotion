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


import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from tabulate import tabulate


# 需要统计的指标 Key
METRICS = [
    "mpjpe_g",
    "mpjpe_l",
    "whole_body_joints_dist",
    "root_vel_error",
    "root_ang_vel_error",
    "p95_root_ang_vel_error",
    "root_r_error",
    "root_p_error",
    "root_y_error",
    "p95_root_y_error",
    "root_quat_error",
    "p95_root_quat_error",
    "root_yaw_drift_error",
    "ref_root_yaw_range",
    "turning_motion_rate",
    "turning_root_y_error",
    "turning_p95_root_y_error",
    "turning_root_yaw_drift_error",
    "root_height_error",
    "mean_dof_vel",
    "mean_dof_acc",
    "mean_dof_torque",
    "mean_action_rate",
    "success",
    "mean_torque_jump_norm",
    "p95_torque_jump_norm",
    "mean_torque_jump_ratio",
    "p95_torque_jump_ratio",
]

# 表头映射 (Json Key -> 表格显示名称)
COLUMN_MAPPING = {
    "mpjpe_g": "Global Bodylink Pos Err",
    "mpjpe_l": "Local Bodylink Pos Err",
    "whole_body_joints_dist": "Dof Position Err",
    "root_vel_error": "Root Vel Err",
    "root_ang_vel_error": "Root Ang Vel Err",
    "p95_root_ang_vel_error": "P95 Root Ang Vel Err",
    "root_r_error": "Root Roll Err",
    "root_p_error": "Root Pitch Err",
    "root_y_error": "Root Yaw Err",
    "p95_root_y_error": "P95 Root Yaw Err",
    "root_quat_error": "Root Quat Err",
    "p95_root_quat_error": "P95 Root Quat Err",
    "root_yaw_drift_error": "Root Yaw Drift Err",
    "ref_root_yaw_range": "Ref Root Yaw Range",
    "turning_motion_rate": "Turning Clip Rate",
    "turning_root_y_error": "Turning Root Yaw Err",
    "turning_p95_root_y_error": "Turning P95 Root Yaw Err",
    "turning_root_yaw_drift_error": "Turning Root Yaw Drift Err",
    "root_height_error": "Root Height Err",
    "mean_dof_vel": "Mean Dof Vel",
    "mean_dof_acc": "Mean Dof Acc",
    "mean_dof_torque": "Mean Dof Torque",
    "mean_action_rate": "Mean Action Rate",
    "success": "Success Rate",
    "mean_torque_jump_norm": "Mean Torque Jump Norm",
    "p95_torque_jump_norm": "P95 Torque Jump Norm",
    "mean_torque_jump_ratio": "Mean Torque Jump Ratio",
    "p95_torque_jump_ratio": "P95 Torque Jump Ratio",
}


def get_dataset_name(motion_key):
    if not isinstance(motion_key, str):
        return "Unknown"

    match_old = re.search(r"clips_([a-zA-Z0-9]+)_", motion_key)
    if match_old:
        return match_old.group(1)

    match_new = re.search(r"v1.1_eval_([a-zA-Z0-9]+)_", motion_key)
    if match_new:
        return match_new.group(1)

    return motion_key.split("_")[0]


def process_data(folder_path):
    folder_path = os.path.expanduser(folder_path)
    search_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(search_pattern)
    json_files = [
        file for file in json_files if "batch_" not in os.path.basename(file)
    ]

    if not json_files:
        raise FileNotFoundError(
            f"No .json files found in directory: {folder_path}"
        )

    all_records = []

    for file_path in json_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 结构兼容性处理
        if isinstance(data, dict) and "per_clip" in data:
            clips_data = data["per_clip"]
        elif isinstance(data, list):
            clips_data = data
        elif isinstance(data, dict) and "motion_key" in data:
            clips_data = [data]
        else:
            continue

        for entry in clips_data:
            if "motion_key" not in entry:
                continue

            dataset_name = get_dataset_name(entry["motion_key"])

            record = {"Method": model_name, "Dataset": dataset_name}

            for metric in METRICS:
                val = entry.get(metric, None)
                if val is not None:
                    record[metric] = val

            is_turning_clip = (
                float(entry.get("turning_motion_rate", 0.0)) >= 0.5
            )
            if is_turning_clip:
                for source_metric, turning_metric in (
                    ("root_y_error", "turning_root_y_error"),
                    ("p95_root_y_error", "turning_p95_root_y_error"),
                    ("root_yaw_drift_error", "turning_root_yaw_drift_error"),
                ):
                    val = entry.get(source_metric, None)
                    if val is not None:
                        record[turning_metric] = val

            all_records.append(record)

    if not all_records:
        raise ValueError(
            f"No valid per-clip metric records extracted from: {folder_path}"
        )

    df = pd.DataFrame(all_records)
    df = df.reindex(columns=["Method", "Dataset", *METRICS])
    grouped_ds = df.groupby(["Method", "Dataset"])[METRICS]
    df_mean_ds = grouped_ds.mean().reset_index()
    df_median_ds = grouped_ds.median().reset_index()

    # Macro-Mean calculation
    df_mean_total = (
        df_mean_ds.groupby(["Method"])[METRICS].mean().reset_index()
    )

    # Macro-Median calculation
    df_median_total = (
        df_median_ds.groupby(["Method"])[METRICS].mean().reset_index()
    )

    df_mean_total["Dataset"] = "Total (Macro)"
    df_median_total["Dataset"] = "Total (Macro)"

    final_mean = pd.concat([df_mean_ds, df_mean_total], ignore_index=True)
    final_median = pd.concat(
        [df_median_ds, df_median_total], ignore_index=True
    )

    return final_mean, final_median


def highlight_best(val, best_val):
    """Return a highlighted HTML string when value is best."""
    if val is None or pd.isna(val):
        return str(val)

    val_float = float(val)
    best_val_float = float(best_val)
    formatted_val = f"{val_float:.4f}"
    if np.isclose(val_float, best_val_float, atol=1e-6):
        return f"<b><span style='color: green'>{formatted_val}</span></b>"
    return formatted_val


def generate_report(
    df,
    folder_path,
    file_name="result_table_mean.md",
    title="Evaluation Results (Mean)",
):
    out_md = os.path.join(folder_path, file_name)

    all_datasets = df["Dataset"].unique().tolist()

    # 排序：将 Total 放到最后
    total_key = "Total (Macro)"
    if total_key in all_datasets:
        all_datasets.remove(total_key)
        all_datasets.sort()
        all_datasets.append(total_key)
    else:
        all_datasets.sort()

    md_content_accumulator = f"# {title}\n\n"
    md_content_accumulator += (
        "> **Note:** 'Total (Macro)' represents the **Macro-Average**, "
        "calculated as the arithmetic mean of the scores across all datasets, "
        "treating each dataset equally regardless of sample size.\n\n"
    )

    for ds_name in all_datasets:
        sub_df = df[df["Dataset"] == ds_name].copy()

        for metric in METRICS:
            if metric in sub_df.columns:
                if metric == "success":
                    best_val = sub_df[metric].max()
                else:
                    best_val = sub_df[metric].min()
                sub_df[metric] = sub_df[metric].apply(
                    lambda x, best_val=best_val: highlight_best(x, best_val)
                )

        sub_df = sub_df.drop(columns=["Dataset"])
        sub_df.rename(columns=COLUMN_MAPPING, inplace=True)

        cols = list(sub_df.columns)
        if "Method" in cols:
            cols.insert(0, cols.pop(cols.index("Method")))
            sub_df = sub_df[cols]

        md_content_accumulator += f"### Dataset: {ds_name}\n"
        # 使用 to_markdown 生成表格
        table_str = sub_df.to_markdown(index=False)
        md_content_accumulator += table_str + "\n\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content_accumulator)

    return os.path.abspath(out_md)


def _format_metric_values_for_cli(sub_df: pd.DataFrame) -> pd.DataFrame:
    cli_df = sub_df.copy()
    for metric in METRICS:
        if metric in cli_df.columns:
            cli_df[metric] = cli_df[metric].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "nan"
            )
    return cli_df


def _print_cli_tables(df: pd.DataFrame, title: str, folder_path: str) -> None:
    total_key = "Total (Macro)"
    all_datasets = df["Dataset"].unique().tolist()
    dataset_order = sorted([d for d in all_datasets if d != total_key])
    if total_key in all_datasets:
        dataset_order.append(total_key)

    merged_df = df.copy()
    merged_df["Dataset"] = pd.Categorical(
        merged_df["Dataset"], categories=dataset_order, ordered=True
    )
    merged_df = merged_df.sort_values(
        by=["Dataset", "Method"], kind="stable"
    ).reset_index(drop=True)
    merged_df["Dataset"] = merged_df["Dataset"].astype(str)

    merged_df = _format_metric_values_for_cli(merged_df)
    merged_df.rename(columns=COLUMN_MAPPING, inplace=True)

    metric_display_cols = [
        COLUMN_MAPPING[m] for m in METRICS if COLUMN_MAPPING[m] in merged_df
    ]
    # table_cols = ["Dataset", "Method"] + metric_display_cols
    table_cols = ["Dataset"] + metric_display_cols
    merged_df = merged_df[table_cols]

    output_tsv_path = os.path.join(
        folder_path, "sub_dataset_macro_mean_metrics.tsv"
    )
    with open(output_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(merged_df.columns.tolist())
        writer.writerows(merged_df.values.tolist())

    table_str = tabulate(
        merged_df.values.tolist(),
        headers=merged_df.columns.tolist(),
        tablefmt="simple_outline",
        colalign=("left",) * len(merged_df.columns),
    )

    block = (
        "\n"
        + "=" * 80
        + f"\n{title}\n"
        + "=" * 80
        + f"\n\n{table_str}\n"
        + "=" * 80
        + "\n"
    )
    print(block)
    metric_log_path = os.path.join(folder_path, "metric.log")
    with open(metric_log_path, "a", encoding="utf-8") as file:
        file.write(block)


def generate_macro_mean_report_from_json_dir(folder_path: str) -> str:
    mean_df, _ = process_data(folder_path)
    report_path = generate_report(
        df=mean_df,
        folder_path=folder_path,
        file_name="result_table_macro_mean.md",
        title="Evaluation Results (Macro-Averaging Mean)",
    )
    _print_cli_tables(
        df=mean_df,
        title="DATASET-WISE METRICS (MACRO-AVERAGING MEAN)",
        folder_path=folder_path,
    )
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="json文件夹路径")
    args = parser.parse_args()

    out_md = generate_macro_mean_report_from_json_dir(args.dir)
    print(f"报告已生成: {out_md}")
