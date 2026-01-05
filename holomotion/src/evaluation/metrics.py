from pathlib import Path
from typing import Dict, List, Optional

import argparse
import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
from tabulate import tabulate
from tqdm import tqdm


def quat_inv(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    xyz = q[:, None, :3]
    w = q[:, None, 3:4]

    t = 2.0 * np.cross(xyz, v, axis=-1)
    return v + w * t + np.cross(xyz, t, axis=-1)


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Procrustes-aligned MPJPE between predicted and ground truth.

    Reference:
        This function is inspired by and partially adapted from the SMPLSim:
        https://github.com/ZhengyiLuo/SMPLSim/blob/0d672790a7672f28361d59dadd98ae2fc1b9685e/smpl_sim/smpllib/smpl_eval.py.

    """
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    # Per-frame SVD with graceful handling for non-convergence: mark those frames as NaN
    batch_size = int(h.shape[0])
    jdim = int(h.shape[1])
    u = np.empty((batch_size, jdim, jdim), dtype=h.dtype)
    s = np.empty((batch_size, jdim), dtype=h.dtype)
    vt = np.empty((batch_size, jdim, jdim), dtype=h.dtype)
    for i in range(batch_size):
        try:
            ui, si, vti = np.linalg.svd(h[i])
            u[i] = ui
            s[i] = si
            vt[i] = vti
        except np.linalg.LinAlgError:
            u[i].fill(np.nan)
            s[i].fill(np.nan)
            vt[i].fill(np.nan)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Corrected rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    predicted_aligned = a * np.matmul(predicted, r) + t

    return np.linalg.norm(
        predicted_aligned - target, axis=len(target.shape) - 1
    )


def _parse_clip_len_from_name(filename: str) -> Optional[int]:
    """Extract clip length from filename suffix '__start_XXX_len_N'."""
    m = re.search(r"__start_\d+_len_(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else None


def _per_frame_metrics_from_npz(
    motion_key: str,
    data: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute per-frame metrics for a single motion clip from loaded npz arrays.

    Expects the following keys in `data` (URDF order):
    - dof_pos, robot_dof_pos
    - global_translation, robot_global_translation
    - global_rotation_quat, robot_global_rotation_quat (xyzw)
    """
    # Required arrays
    jpos_gt = np.asarray(data["ref_global_translation"])  # (T, J, 3)
    jpos_pred = np.asarray(data["robot_global_translation"])  # (T, J, 3)
    rot_gt = np.asarray(data["ref_global_rotation_quat"])  # (T, J, 4) xyzw
    rot_pred = np.asarray(data["robot_global_rotation_quat"])  # (T, J, 4)
    dof_gt = np.asarray(data["ref_dof_pos"])  # (T, D)
    dof_pred = np.asarray(data["robot_dof_pos"])  # (T, D)

    assert jpos_gt.shape == jpos_pred.shape
    assert rot_gt.shape == rot_pred.shape
    assert dof_gt.shape == dof_pred.shape

    num_frames = int(jpos_gt.shape[0])

    # Global MPJPE [mm]
    mpjpe_g = (
        np.mean(np.linalg.norm(jpos_gt - jpos_pred, axis=2), axis=1) * 1000.0
    )

    # Per-frame maximum body-link position error [m] (used for failure criterion)
    per_joint_err = np.linalg.norm(jpos_pred - jpos_gt, axis=2)
    frame_max_body_pos_err = np.max(per_joint_err, axis=1)

    # Localize by root (index 0)
    jpos_gt_local = jpos_gt - jpos_gt[:, [0]]
    jpos_pred_local = jpos_pred - jpos_pred[:, [0]]
    robot_body_pos_root_rel = quat_apply(
        quat_inv(rot_gt[:, 0, :]),
        jpos_gt - jpos_gt[:, [0]],
    )
    ref_body_pos_root_rel = quat_apply(
        quat_inv(rot_pred[:, 0, :]),
        jpos_pred - jpos_pred[:, [0]],
    )

    mpjpe_l = (
        np.mean(
            np.linalg.norm(
                robot_body_pos_root_rel - ref_body_pos_root_rel, axis=2
            ),
            axis=1,
        )
        * 1000.0
    )

    # Procrustes-aligned MPJPE [mm]
    pa_per_joint = p_mpjpe(jpos_pred_local, jpos_gt_local)
    mpjpe_pa = np.mean(pa_per_joint, axis=1) * 1000.0

    # Velocity/acceleration errors from positions (discrete frame diffs) [mm/frame],[mm/frame^2]
    vel_gt = jpos_gt[1:] - jpos_gt[:-1]
    vel_pred = jpos_pred[1:] - jpos_pred[:-1]
    vel_dist = (
        np.mean(np.linalg.norm(vel_pred - vel_gt, axis=2), axis=1) * 1000.0
    )

    acc_gt = jpos_gt[:-2] - 2 * jpos_gt[1:-1] + jpos_gt[2:]
    acc_pred = jpos_pred[:-2] - 2 * jpos_pred[1:-1] + jpos_pred[2:]
    accel_dist = (
        np.mean(np.linalg.norm(acc_pred - acc_gt, axis=2), axis=1) * 1000.0
    )

    # DOF angle errors [radians] — whole body average
    dof_err = np.abs(dof_pred - dof_gt)
    whole_body_joints_dist = np.mean(dof_err, axis=1)

    # Root orientation errors [radians] — handle zero-norm/invalid quaternions by NaN
    q_gt_root = rot_gt[:, 0, :]
    q_pred_root = rot_pred[:, 0, :]
    norms_gt = np.linalg.norm(q_gt_root, axis=1)
    norms_pred = np.linalg.norm(q_pred_root, axis=1)
    valid_mask = (
        (norms_gt > 0.0)
        & (norms_pred > 0.0)
        & np.isfinite(norms_gt)
        & np.isfinite(norms_pred)
    )

    root_r_error = np.full((num_frames,), np.nan, dtype=float)
    root_p_error = np.full((num_frames,), np.nan, dtype=float)
    root_y_error = np.full((num_frames,), np.nan, dtype=float)

    if np.any(valid_mask):
        q_gt_valid = q_gt_root[valid_mask] / norms_gt[valid_mask, None]
        q_pred_valid = q_pred_root[valid_mask] / norms_pred[valid_mask, None]
        rel_valid = sRot.from_quat(q_gt_valid).inv() * sRot.from_quat(
            q_pred_valid
        )
        euler_xyz = rel_valid.as_euler("xyz", degrees=False)
        root_r_error[valid_mask] = np.abs(euler_xyz[:, 0])
        root_p_error[valid_mask] = np.abs(euler_xyz[:, 1])
        root_y_error[valid_mask] = np.abs(euler_xyz[:, 2])

    # Root velocity error [m/frame]
    root_pos_gt = jpos_gt[:, 0, :]
    root_pos_pred = jpos_pred[:, 0, :]
    root_vel_err = np.linalg.norm(
        (root_pos_pred[1:] - root_pos_pred[:-1])
        - (root_pos_gt[1:] - root_pos_gt[:-1]),
        axis=1,
    )

    # Root height error [m]
    root_height_error = np.abs(root_pos_pred[:, 2] - root_pos_gt[:, 2])

    # Frame DataFrame (align lengths by padding NaN at the start where needed)
    def pad_front(x: np.ndarray, pad: int) -> np.ndarray:
        if pad <= 0:
            return x
        return np.concatenate(
            [np.full((pad,), np.nan, dtype=float), x], axis=0
        )

    df = pd.DataFrame(
        {
            "motion_key": [motion_key] * num_frames,
            "frame_idx": np.arange(num_frames, dtype=int),
            "mpjpe_g": mpjpe_g,
            "mpjpe_l": mpjpe_l,
            "mpjpe_pa": mpjpe_pa,
            "vel_dist": pad_front(vel_dist, 1),
            "accel_dist": pad_front(accel_dist, 2),
            "frame_max_body_pos_err": frame_max_body_pos_err,
            "whole_body_joints_dist": whole_body_joints_dist,
            "root_r_error": root_r_error,
            "root_p_error": root_p_error,
            "root_y_error": root_y_error,
            "root_vel_error": pad_front(root_vel_err, 1),
            "root_height_error": root_height_error,
        }
    )
    return df


def offline_evaluate_dumped_npzs(
    npz_dir: str,
    output_json_path: str,
    failure_pos_err_thresh_m: float = 0.25,
) -> Dict[str, dict]:
    """Evaluate dumped NPZs in `npz_dir` and write a JSON summary to `output_dir`.

    The function produces dataset-wide averages and per-clip averages across frames.
    """
    npz_dir_abs = Path(npz_dir).resolve()
    os.makedirs(npz_dir_abs, exist_ok=True)

    # Add file handler for logging to metric.log
    metric_log_path = npz_dir_abs / "metric.log"
    logger.add(
        str(metric_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )

    logger.info(f"Input NPZ directory (absolute): {npz_dir_abs}")

    # Gather NPZ files
    files = sorted(glob(os.path.join(npz_dir_abs, "*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found in: {npz_dir_abs}")

    # Accumulate per-frame metrics
    frame_tables: List[pd.DataFrame] = []
    clip_meta: Dict[str, dict] = {}

    skipped_files_count = 0

    for fpath in tqdm(files, desc="Compute metrics from NPZs"):
        try:
            with np.load(fpath, allow_pickle=True) as npz_data:
                # Extract arrays and metadata
                required_keys = [
                    "ref_dof_pos",
                    "ref_dof_vel",
                    "ref_global_translation",
                    "ref_global_rotation_quat",
                    "ref_global_velocity",
                    "ref_global_angular_velocity",
                    "robot_dof_pos",
                    "robot_dof_vel",
                    "robot_global_translation",
                    "robot_global_rotation_quat",
                    "robot_global_velocity",
                    "robot_global_angular_velocity",
                ]
                data = {k: npz_data[k] for k in required_keys}

                metadata = {}
                if "metadata" in npz_data:
                    obj = npz_data["metadata"].item()
                    if isinstance(obj, dict):
                        metadata = obj

            motion_key = os.path.splitext(os.path.basename(fpath))[0]
            clip_len_from_name = _parse_clip_len_from_name(fpath)

            df_frames = _per_frame_metrics_from_npz(
                motion_key=motion_key, data=data
            )
            frame_tables.append(df_frames)

            # Clip-level info and failure criterion (max body-link pos error > threshold)
            num_frames_clip = int(df_frames.shape[0])
            clip_length = int(
                metadata.get(
                    "clip_length", clip_len_from_name or num_frames_clip
                )
            )
            max_body_err = float(
                np.nanmax(df_frames["frame_max_body_pos_err"].to_numpy())
            )
            success = 1.0 if max_body_err <= failure_pos_err_thresh_m else 0.0
            clip_meta[motion_key] = {
                "motion_key": motion_key,
                "num_frames": num_frames_clip,
                "clip_length": clip_length,
                "success": success,
                "max_body_pos_err": max_body_err,
                "failure_threshold_m": float(failure_pos_err_thresh_m),
            }
        except ValueError as e:
            # If a ValueError occurs (likely due to array length mismatch), log it and skip the file.
            logger.warning(
                f"\nCaught a ValueError while processing file: {fpath}"
            )
            logger.warning(f"Error message: {e}")
            logger.warning("This file will be SKIPPED.")
            skipped_files_count += 1
            continue  # Move to the next file in the loop

    if skipped_files_count > 0:
        logger.info(
            f"\nFinished processing. Skipped a total of {skipped_files_count} files due to errors."
        )

    # If all files were skipped, there's nothing to process further.
    if not frame_tables:
        logger.error(
            "No valid NPZ files could be processed. Aborting evaluation."
        )
        return {}

    # Concatenate per-frame metrics
    all_frames = pd.concat(frame_tables, ignore_index=True)

    # Per-clip averages
    metric_cols = [
        "mpjpe_g",
        "mpjpe_l",
        "whole_body_joints_dist",
        "root_vel_error",
        "root_r_error",
        "root_p_error",
        "root_y_error",
        "root_height_error",
    ]
    # Metric display configuration: metric_key -> (display_name, unit)
    metric_display_map = {
        "mpjpe_g": ("Global Bodylink Mean Position Error", "mm"),
        "mpjpe_l": ("Local Bodylink Mean Position Error", "mm"),
        "whole_body_joints_dist": ("DOF Position Error", "rad"),
        "root_vel_error": ("Root Velocity Error", "m/s"),
        "root_r_error": ("Root Roll Error", "rad"),
        "root_p_error": ("Root Pitch Error", "rad"),
        "root_y_error": ("Root Yaw Error", "rad"),
        "root_height_error": ("Root Height Error", "mm"),
    }

    per_clip_mean = (
        all_frames.groupby("motion_key")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Merge with success flags
    per_clip_records = []
    for _, row in per_clip_mean.iterrows():
        mk = row["motion_key"]
        rec = {**row.to_dict(), **clip_meta.get(mk, {})}
        per_clip_records.append(rec)

    # Dataset-level averages
    dataset_means = {
        k: float(np.nanmean(all_frames[k].to_numpy())) for k in metric_cols
    }
    success_rate = float(
        np.mean([clip_meta[mk]["success"] for mk in clip_meta])
        if len(clip_meta) > 0
        else 0.0
    )
    dataset_means["success_rate"] = success_rate

    # Compose result and write
    result = {
        "dataset": dataset_means,
        "num_clips": int(len(clip_meta)),
        "per_clip": per_clip_records,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Conversion factors for unit conversion (assuming 50Hz)
    frame_rate_hz = 50.0
    unit_conversions = {
        "root_height_error": 1000.0,  # m to mm
        "root_vel_error": frame_rate_hz,  # m/frame to m/s
    }

    table_data = []
    # Iterate through metric_display_map to preserve order
    for key in metric_display_map.keys():
        if key not in dataset_means:
            continue

        value = dataset_means[key]
        display_name, unit = metric_display_map[key]

        # Apply unit conversion if needed
        if key in unit_conversions:
            value = value * unit_conversions[key]

        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        table_data.append([display_name, formatted_value, unit])

    table_str = tabulate(
        table_data,
        headers=["Metric", "Value", "Unit"],
        tablefmt="simple_outline",
        colalign=("left", "left", "left"),
    )
    logger.info(
        "\n"
        + "=" * 80
        + "\nDATASET-WISE METRICS\n"
        + "=" * 80
        + f"\n\n{table_str}\n"
        + "=" * 80
        + "\n"
    )

    return result


def parse_ckpt_and_dataset_from_eval_dirname(
    eval_dir_name: str, dataset_suffix: str
):
    prefix = "isaaclab_eval_output_"
    if not eval_dir_name.startswith(prefix):
        return None, None

    rest = eval_dir_name[len(prefix) :]
    if not rest.endswith(dataset_suffix):
        return None, None

    model_part = rest[: -len(dataset_suffix)]
    if model_part.endswith("_"):
        model_part = model_part[:-1]

    m = re.search(r"model_(\d+)$", model_part)
    if not m:
        return None, dataset_suffix

    return m.group(1), dataset_suffix


def run_evaluation(
    npz_dir: str,
    dataset_suffix: str,
    failure_pos_err_thresh_m: float = 0.25,
):
    """
    Main function to run evaluation. It scans a root directory, runs evaluation
    for each found subdirectory, and generates a final summary report.

    Args:
        npz_dir (str): Top-level directory containing all model evaluation results (e.g., 'logs/test').
        output_dir (str): Directory to store all generated JSON files and logs.
        failure_pos_err_thresh_m (float): The position error threshold in meters to determine a failure.
    """
    root_path = Path(npz_dir)

    logger.info(f"Starting batch evaluation. Root directory: '{root_path}'")
    logger.info(
        f"Searching for directories matching pattern: '{dataset_suffix}'"
    )

    def has_npz_files(path: Path) -> bool:
        return path.is_dir() and any(path.glob("*.npz"))

    is_single_eval_dir = (
        root_path.is_dir()
        and root_path.name.startswith("isaaclab_eval_output_")
        and has_npz_files(root_path)
    )

    if is_single_eval_dir:
        output_path = root_path
    else:
        output_path = root_path / f"metrics_output_{dataset_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    if is_single_eval_dir:
        logger.info(
            f"Detected '{root_path}' as a single evaluation directory. "
            "Running offline evaluation only for this directory."
        )
        model_name = root_path.parent.name

        ckpt_str, ds = parse_ckpt_and_dataset_from_eval_dirname(
            root_path.name, dataset_suffix
        )
        if ckpt_str is None:
            logger.warning(
                f"Could not parse checkpoint/dataset from directory name '{root_path.name}'. "
                "Using 'checkpoint_unknown' in output filename."
            )
            ckpt_str = "checkpoint_unknown"
            ds = dataset_suffix

        output_json_name = f"{model_name}_{ckpt_str}.json"
        output_json_path = output_path / output_json_name

        offline_evaluate_dumped_npzs(
            npz_dir=str(root_path),
            output_json_path=str(output_json_path),
            failure_pos_err_thresh_m=failure_pos_err_thresh_m,
        )
        logger.success(
            f"Finished single-directory evaluation: model='{model_name}', checkpoint={ckpt_str}"
        )
        return
    logger.info(
        f"Treating '{root_path}' as root directory for batch evaluation."
    )
    # Find all directories matching the evaluation output pattern.
    eval_dirs = sorted(
        p
        for p in root_path.glob(f"**/isaaclab_eval_output_*_{dataset_suffix}")
        if p.is_dir()
    )
    if not eval_dirs:
        logger.error(
            f"No directories matching the pattern '{dataset_suffix}' found under '{root_path}'. "
            "Please check the path and pattern."
        )
        return

    all_results = []

    # Process each found evaluation directory.
    for eval_dir in tqdm(eval_dirs, desc="Overall Progress"):
        # Extract model name from the parent directory.
        model_name = eval_dir.parent.name
        # Parse the checkpoint number from the directory name.
        ckpt_str, ds = parse_ckpt_and_dataset_from_eval_dirname(
            eval_dir.name, dataset_suffix
        )
        if ckpt_str is None:
            logger.warning(
                f"Could not parse ckpt/dataset from '{eval_dir.name}'. Skipping."
            )
            continue

        checkpoint = int(ckpt_str)

        logger.info(
            f"\n--- Processing: model='{model_name}', dataset='{ds}', checkpoint={checkpoint} ---"
        )

        # Construct a unique output JSON filename.
        output_json_name = f"{model_name}_{checkpoint}.json"
        output_json_path = output_path / output_json_name

        # Call the evaluation function for the current directory.
        result = offline_evaluate_dumped_npzs(
            npz_dir=str(eval_dir),
            output_json_path=str(output_json_path),
            failure_pos_err_thresh_m=failure_pos_err_thresh_m,
        )

        if result and "dataset" in result:
            # Collect dataset-level average metrics for the final summary.
            flat_result = {
                "model": model_name,
                "checkpoint": checkpoint,
                **result["dataset"],
            }
            all_results.append(flat_result)
            logger.success(
                f"--- Finished processing: model='{model_name}', checkpoint={checkpoint} ---"
            )
        else:
            logger.error(
                f"--- Failed to process: model='{model_name}', checkpoint={checkpoint} ---"
            )

    if not all_results:
        logger.error(
            "No evaluations succeeded. Cannot generate a summary report."
        )
        return

    logger.info("\n" + "=" * 80)
    logger.info("Batch evaluation finished successfully.")
    logger.info(f"Total successful evaluations: {len(all_results)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--npz_dir", type=str, required=True)
    argument_parser.add_argument(
        "--dataset_suffix",
        type=str,
        required=True,
    )
    argument_parser.add_argument(
        "--failure_pos_err_thresh_m", type=float, default=0.25
    )
    args = argument_parser.parse_args()

    run_evaluation(
        npz_dir=args.npz_dir,
        dataset_suffix=args.dataset_suffix,
        failure_pos_err_thresh_m=args.failure_pos_err_thresh_m,
    )
