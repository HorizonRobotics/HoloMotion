# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
#
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(
    "./holomotion/src/data_curation/omomo_release/human_body_prior/src"
)


def calc_max_xy_translation(motion_data: dict):
    """Calculate max xy translation."""
    trans = motion_data["trans"]
    root_trans_offset = trans
    max_xy_translation = np.max(
        np.linalg.norm(
            root_trans_offset[:, :2] - root_trans_offset[0:1, :2],
            axis=1,
        )
    )
    return max_xy_translation


def calc_max_z_translation(motion_data: dict):
    """Calculate max and min z translation."""
    trans = motion_data["trans"]
    root_trans_offset = trans
    max_z_translation = np.max(
        root_trans_offset[:, 2] - root_trans_offset[0:1, 2]
    )
    min_z_translation = np.min(
        root_trans_offset[:, 2] - root_trans_offset[0:1, 2]
    )
    return max_z_translation, min_z_translation


def calc_max_velocity_scale(motion_data: dict, fps: float = 30):
    """Calculate max velocity scale."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    root_vel_norm = np.linalg.norm(est_root_vel, axis=-1)
    max_velocity_scale = np.max(root_vel_norm)
    return max_velocity_scale


def calc_mean_velocity_scale(motion_data: dict, fps: float = 30):
    """Calculate mean velocity scale."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    root_vel_norm = np.linalg.norm(est_root_vel, axis=-1)
    mean_velocity_scale = np.mean(root_vel_norm)
    return mean_velocity_scale


def calc_std_velocity_scale(motion_data: dict, fps: float = 30):
    """Calculate std velocity scale."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    root_vel_norm = np.linalg.norm(est_root_vel, axis=-1)
    std_velocity_scale = np.std(root_vel_norm)
    return std_velocity_scale


def calc_max_vxy_scale(motion_data: dict, fps: float = 30):
    """Calculate smax vx, vy scale."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    root_vel_norm = np.linalg.norm(est_root_vel[:, :2], axis=-1)
    max_vxy_scale = np.max(root_vel_norm)
    mean_vxy_scale = np.mean(root_vel_norm)
    std_vxy_scale = np.std(root_vel_norm)
    return max_vxy_scale, mean_vxy_scale, std_vxy_scale


def calc_std_accel(motion_data: dict, fps: float = 30.0) -> float:
    """Calculate the standard deviation of root joint acceleration.

    This function computes the per-frame acceleration of the root joint in the
    XY plane from its translation data and returns the standard deviation
    of those values.

    Args:
        motion_data (dict): A dictionary that must contain a 'trans' key
        representing global translation of the root joint.
        Shape should be (T, 3), where T is the number of frames.
        fps (float): Frames per second of the motion sequence.

    Returns:
        float: Standard deviation of the acceleration magnitudes
        on the XY plane. Returns 0.0 if there are fewer than 3 frames.

    """
    trans = motion_data["trans"]  # shape: (T, 3)
    if trans.shape[0] < 3:
        return 0.0  # At least 3 frames are needed to compute two differences

    # Compute velocity (frame-to-frame displacement * fps)
    velocities = np.diff(trans, axis=0) * fps  # shape: (T-1, 3)

    # Compute acceleration (frame-to-frame velocity difference * fps)
    accelerations = np.diff(velocities, axis=0) * fps  # shape: (T-2, 3)

    # Compute acceleration magnitude in XY plane
    accel_xy_norm = np.linalg.norm(
        accelerations[:, :2], axis=1
    )  # shape: (T-2,)

    # Return standard deviation
    return np.std(accel_xy_norm)


def calc_max_vz_scale(motion_data: dict, fps: float = 30):
    """Calculate max vz scale."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    root_vel_norm = np.abs(est_root_vel[:, 2])
    max_vz_scale = np.max(root_vel_norm)
    mean_vz_scale = np.mean(root_vel_norm)
    std_vz_scale = np.std(root_vel_norm)
    return max_vz_scale, mean_vz_scale, std_vz_scale


def calc_vz_scale_with_direction(motion_data: dict, fps: float = 30):
    """Calculate vz scale with direction."""
    root_trans_offset = motion_data["trans"]
    est_root_vel = np.diff(root_trans_offset * fps, axis=0)
    vz = est_root_vel[:, 2]

    max_up_vz = np.max(vz[vz > 0]) if np.any(vz > 0) else 0.0
    max_down_vz = np.min(vz[vz < 0]) if np.any(vz < 0) else 0.0
    mean_vz = np.mean(vz)
    std_vz = np.std(vz)

    return max_up_vz, max_down_vz, mean_vz, std_vz


def beyond_upper_dof_limits(
    motion_data: dict,
    upper_dof_mapping: dict,
    upper_dof_max_limits: dict,
):
    """Check whether or not the motion data is beyond upper dof limits."""
    for dof_name, dof_idx in upper_dof_mapping.items():
        dof_data = motion_data["dof"][:, dof_idx]
        max_dof_scale = np.max(dof_data)
        min_dof_scale = np.min(dof_data)
        if (
            max_dof_scale < upper_dof_max_limits[dof_name][0]
            or max_dof_scale > upper_dof_max_limits[dof_name][1]
            or min_dof_scale < upper_dof_max_limits[dof_name][0]
            or min_dof_scale > upper_dof_max_limits[dof_name][1]
        ):
            return True
    return False


class HyperParams:
    max_xy_translation: float = 2.0
    max_z_translation: float = 0.3
    max_velocity_scale: float = 1.0
    max_vxy_scale: float = 1.2
    max_vz_scale: float = 0.3
    upper_dof_mapping: dict = {
        "left_shoulder_pitch_joint": 13,
        "left_shoulder_roll_joint": 14,
        "left_shoulder_yaw_joint": 15,
        "left_elbow_joint": 16,
        "right_shoulder_pitch_joint": 17,
        "right_shoulder_roll_joint": 18,
        "right_shoulder_yaw_joint": 19,
        "right_elbow_joint": 20,
    }
    upper_dof_max_limits: dict = {
        "left_shoulder_pitch_joint": [-1.0, 1.0],
        "left_shoulder_roll_joint": [0.0, 0.5],
        "left_shoulder_yaw_joint": [-0.5, 0.5],
        "left_elbow_joint": [0.5, 1.3],
        "right_shoulder_pitch_joint": [-1.0, 1.0],
        "right_shoulder_roll_joint": [-0.5, 0.0],
        "right_shoulder_yaw_joint": [-0.5, 0.3],
        "right_elbow_joint": [0.5, 1.5],
    }


def label_data_with_metrics(data_folder, jsonl_path: str, parent_folder: str):
    """Calculate the metics and load them into a jsonl file."""
    assert jsonl_path.endswith(".jsonl")
    with open(jsonl_path, "w") as f_out:
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".npz"):
                    npz_path = os.path.join(root, file)
                    content = {}
                    content["path"] = os.path.relpath(npz_path, parent_folder)
                    data = np.load(npz_path)
                    fps = data.get("mocap_frame_rate")
                    if fps is None:
                        fps = data.get("mocap_framerate")
                    if fps is None:
                        continue

                    try:
                        content["max_xy_translation"] = round(
                            calc_max_xy_translation(data), 2
                        )
                        max_z_translation, min_z_translation = (
                            calc_max_z_translation(data)
                        )
                        content["max_z_translation"] = round(
                            max_z_translation, 2
                        )
                        content["min_z_translation"] = round(
                            min_z_translation, 2
                        )
                        content["max_velocity"] = round(
                            calc_max_velocity_scale(data, fps), 2
                        )
                        content["mean_velocity"] = round(
                            calc_mean_velocity_scale(data, fps), 2
                        )
                        content["std_velocity"] = round(
                            calc_std_velocity_scale(data, fps), 2
                        )
                        content["std_accel"] = round(
                            calc_std_accel(data, fps), 2
                        )
                        max_xy_v, mean_xy_v, std_xy_v = calc_max_vxy_scale(
                            data, fps
                        )
                        content["max_xy_velocity"] = round(max_xy_v, 2)
                        content["mean_xy_velocity"] = round(mean_xy_v, 2)
                        content["std_xy_velocity"] = round(std_xy_v, 2)
                        max_up_z_v, max_down_z_v, mean_z_v, std_z_v = (
                            calc_vz_scale_with_direction(data, fps)
                        )
                        content["max_up_z_velocity"] = round(max_up_z_v, 2)
                        content["max_down_z_velocity"] = round(max_down_z_v, 2)
                        content["mean_z_velocity"] = round(mean_z_v, 2)
                        content["std_z_velocity"] = round(std_z_v, 2)
                    except Exception as e:
                        print(f"Error: {e}")

                    def convert_to_builtin_type(obj):
                        if isinstance(obj, dict):
                            return {
                                k: convert_to_builtin_type(v)
                                for k, v in obj.items()
                            }
                        elif isinstance(obj, list):
                            return [convert_to_builtin_type(i) for i in obj]
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.int32, np.int64)):
                            return int(obj)
                        else:
                            return obj

                    f_out.write(
                        json.dumps(convert_to_builtin_type(content)) + "\n"
                    )

    print(f"Annotated file saved to: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_list",
        nargs="+",
        default=["humanact12", "MotionX", "OMOMO", "ZJU_Mocap", "amass"],
        help="List of jsonl files to process.",
    )
    args = parser.parse_args()

    amass_folder = "./data/amass_compatible_datasets/amass"
    other_folder = "./data/amass_compatible_datasets"
    caption_folder = "./data/dataset_labels"
    os.makedirs(caption_folder, exist_ok=True)

    for name in args.jsonl_list:
        file = name + ".jsonl"
        if name != "amass":
            label_data_with_metrics(
                os.path.join(other_folder, name),
                os.path.join(caption_folder, file),
                other_folder,
            )
        else:
            label_data_with_metrics(
                amass_folder, os.path.join(caption_folder, file), amass_folder
            )


if __name__ == "__main__":
    main()
