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

import argparse
import json
import os

import numpy as np


def checksitpose(
    npz_path, ref_pose_path, threshold=0.75, frame_thresh=1
) -> bool:
    """Check if the given motion sequence is close to the pose.

    Args:
        npz_path (str): Path to the .npz file of the motion sequence.
        ref_pose_path (str): the reference sitting pose.
        threshold (float, optional): Euclidean distance threshold.
        frame_thresh (int, optional): Minimum number of frames.

    Returns:
        bool: True if the sequence contains sitting-like frames.

    """
    count = 0
    try:
        sitdata = np.load(ref_pose_path)
        sitpose = sitdata["poses"][535][:66]  # reference sitting pose
    except Exception:
        return False

    sitpose_down = sitpose[3:36]  # lower-body joints only

    bdata = np.load(npz_path)
    curposes = bdata["poses"]  # shape: (N, 165)

    for pose in curposes:
        pose_down = pose[3:36]
        dist = np.linalg.norm(pose_down - sitpose_down)
        if dist < threshold:
            count += 1
        if count >= frame_thresh:
            return True

    return False


def process_dataset(
    parent_folder,
    json_path,
    output_path,
    abnormal_path,
    sit_pose_reference,
    stair_keywords=None,
    sit_keywords=None,
    sit_threshold=0.75,
    frame_threshold=20,
    velocity_threshold=100.0,
):
    """Label the dataset under parent folder."""
    stair_keywords = stair_keywords or [
        "stairs",
        "staircase",
        "upstairs",
        "downstairs",
    ]
    sit_keywords = sit_keywords or ["sitting", "Sitting"]
    abnormal_dataset = ["aist"]

    stairs = sit = untrack = 0
    filtered_paths = set()

    with (
        open(json_path) as f_in,
        open(output_path, "w") as f_out_normal,
        open(abnormal_path, "w") as f_out_abnormal,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                content = json.loads(line)
                path = content.get("path", "")
                npz_path = os.path.join(parent_folder, path)

                # skip the path if it is abnormal
                if path in filtered_paths:
                    f_out_abnormal.write(line + "\n")
                    continue

                up_z = content.get("max_up_z_velocity", 0)
                down_z = content.get("max_down_z_velocity", 0)
                max_z = content.get("max_z_translation", 0)
                min_z = content.get("min_z_translation", 0)
                mean_v = content.get("mean_velocity", 0)

                # filter by keywords
                if any(kw in path for kw in stair_keywords):
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    stairs += 1
                    continue
                elif any(kw in path for kw in sit_keywords):
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    sit += 1
                    continue

                elif any(kw in path for kw in abnormal_dataset):
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    continue

                if mean_v > velocity_threshold:
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    untrack += 1
                    continue

                if up_z >= 0.6 and max_z > 0.7:
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    stairs += 1
                    continue

                elif down_z <= -0.7 and min_z < -0.7:
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.clear()
                    filtered_paths.add(path)
                    stairs += 1
                    continue

                if checksitpose(
                    npz_path,
                    sit_pose_reference,
                    sit_threshold,
                    frame_threshold,
                ):
                    f_out_abnormal.write(line + "\n")
                    filtered_paths.add(path)
                    sit += 1
                    continue

                # normal motion
                f_out_normal.write(line + "\n")

            except Exception as e:
                print(f"Error processing line: {line}\nException: {e}")

    print(
        f"total abnormal data:upstairs {stairs}, sitting {sit}, \
            velocity {untrack}"
    )


def jsonl_to_yaml(jsonl_path, yaml_output_path):
    """Convert jsonl file into yaml file."""
    output_set = set()

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                path = data.get("path", "")
                if path:
                    clean_path = os.path.splitext(path.strip().lstrip("/"))[0]
                    new_name = "0-" + clean_path.replace("/", "_").replace(
                        "\\", "_"
                    )
                    output_set.add(f"{new_name}")
            except json.JSONDecodeError:
                print(f"skip json line: {line.strip()}")
                continue

    with open(yaml_output_path, "w") as out:
        out.write("[\n")
        for item in sorted(output_set):
            out.write(f"  {item},\n")
        out.write("]\n")

    print(f"done, total {len(output_set)} items -> {yaml_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter AMASS dataset and save results."
    )

    parser.add_argument(
        "--parent_folder",
        type=str,
        default="./data/amass_compatible_datasets",
        help="Path to the parent folder of AMASS data",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/dataset_labels/OMOMO.jsonl",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/dataset_labels/temp.jsonl",
        help="Path to save the filtered output JSONL",
    )
    parser.add_argument(
        "--abnormal_path",
        type=str,
        default="./data/dataset_labels/temp2.jsonl",
        help="Path to save abnormal data JSONL",
    )
    parser.add_argument(
        "--sit_pose_reference",
        type=str,
        default="./data/amass_compatible_datasets/amass/BioMotionLab_NTroje/rub062/0016_sitting2_poses.npz",
        help="Path to the reference sitting pose npz",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./holomotion/config/data_curation/base.yaml",
        help="Path to the excluded yaml file",
    )

    args = parser.parse_args()

    process_dataset(
        parent_folder=args.parent_folder,
        json_path=args.json_path,
        output_path=args.output_path,
        abnormal_path=args.abnormal_path,
        sit_pose_reference=args.sit_pose_reference,
    )
    os.makedirs(os.path.dirname(args.yaml_path), exist_ok=True)
    jsonl_to_yaml(args.abnormal_path, args.yaml_path)
