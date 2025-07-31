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
import os

from smplify.smplify_humanact12 import humanact12_to_amass
from smplify.smplify_motionx import motionx_to_amass
from smplify.smplify_omomo import omomo_to_amass

from holomotion.holomotion.src.data_curation.smplify.smplify_zjumocap import (
    zju_to_amass,
)


def ensure_dir(path):
    """Make sure the dir exist.

    Args:
        path: The path of the dir.

    """
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    """Convert multiple motion capture datasets to AMASS format.

    This function parses command-line arguments to specify the root directory
    of raw datasets and an optional save directory. It iterates over the
    supported datasets (MotionX, ZJU_Mocap, HumanAct12, OMOMO), and if the
    corresponding data directory exists, converts it to AMASS format and saves
    it in a unified directory structure.

    Raises:
        SystemExit: If required command-line arguments are missing.

    Side Effects:
        Creates output directories and writes converted data files.
        Prints progress and warning messages to stdout.

    """
    parser = argparse.ArgumentParser(
        description="Convert all datasets to AMASS format"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the root directory of raw datasets",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="Path to save the unified data (default: data_root/smplx_data)",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    save_root = args.save_root or "./data/amass_compatible_datasets"

    print(f"Raw data root: {data_root}")
    print(f"Unified data will be saved to: {save_root}")
    ensure_dir(save_root)

    datasets = [
        ("MotionX", motionx_to_amass),
        ("ZJU_Mocap", zju_to_amass),
        ("humanact12", humanact12_to_amass),
        ("OMOMO", omomo_to_amass),
    ]

    for name, func in datasets:
        data_dir = os.path.join(data_root, name)
        save_dir = os.path.join(save_root, name)
        ensure_dir(save_dir)
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist. Skipping {name}.")
            continue

        print(f"Processing {name}...")
        func(data_dir, save_dir)
        print(f"{name} done. Saved to {save_dir}.\n")

    print("All datasets processed.")


if __name__ == "__main__":
    main()
