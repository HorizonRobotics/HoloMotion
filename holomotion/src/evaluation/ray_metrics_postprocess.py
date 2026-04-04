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


import shutil
import sys
from pathlib import Path
from typing import Any

import ray
from loguru import logger


@ray.remote
def run_metrics_postprocess_job(
    output_dir: str,
    dataset_name: str,
    calc_per_clip_metrics: bool,
    failure_pos_err_thresh_m: float,
    metric_calculation: str,
    dof_mode: str,
    metrics_threadpool_max_workers: int | None,
    generate_report: bool,
    job_log_dir: str | None,
    ckpt_stem: str,
) -> dict[str, Any]:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    if calc_per_clip_metrics:
        from holomotion.src.evaluation.metrics import run_evaluation

        run_evaluation(
            npz_dir=output_dir,
            dataset_suffix=dataset_name,
            failure_pos_err_thresh_m=failure_pos_err_thresh_m,
            metric_calculation=metric_calculation,
            dof_mode=dof_mode,
            threadpool_max_workers=metrics_threadpool_max_workers,
        )

    report_path = None
    if generate_report:
        from holomotion.scripts.evaluation import mean_process_5metrics

        report_path = (
            mean_process_5metrics.generate_macro_mean_report_from_json_dir(
                output_dir
            )
        )

    exported_summary_tsv = None
    if job_log_dir is not None:
        job_log_dir_path = Path(job_log_dir)
        sub_dataset_tsv = (
            Path(output_dir) / "sub_dataset_macro_mean_metrics.tsv"
        )
        if sub_dataset_tsv.is_file():
            export_name = f"{ckpt_stem}_sub_dataset_macro_mean_metrics.tsv"
            export_path = job_log_dir_path / export_name
            shutil.copy2(sub_dataset_tsv, export_path)
            exported_summary_tsv = export_path

    return {
        "ckpt_stem": ckpt_stem,
        "output_dir": output_dir,
        "report_path": str(report_path) if report_path is not None else "",
        "exported_summary_tsv": str(exported_summary_tsv)
        if exported_summary_tsv is not None
        else "",
    }
