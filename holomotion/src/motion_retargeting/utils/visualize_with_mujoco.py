# Project RoboOrchard
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
# This file was originally copied from the [PHC] repository:
# https://github.com/ZhengyiLuo/PHC
# Modifications have been made to fit the needs of this project.
#

import glob
import os
from typing import Any, Dict

import cv2
import hydra
import joblib
import mujoco
import numpy as np
import ray
from omegaconf import DictConfig
from tqdm.auto import tqdm


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene.

    Reference:
        https://github.com/ZhengyiLuo/PHC/blob/master/scripts/vis/
        vis_motion_mj.py
    """
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


class OffscreenRenderer:
    def __init__(self, model, height, width, show_markers=True):
        self.model = model
        self.height = height
        self.width = width
        self.show_markers = show_markers  # Flag to control marker visibility

        # Create OpenGL context
        self.ctx = mujoco.GLContext(width, height)
        self.ctx.make_current()

        # Create scene with increased capacity for markers
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Initialize camera
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 4.0  # mini: 2.0; g1: 4.0; giant: 6.0
        self.cam.azimuth = 60.0
        self.cam.elevation = -20
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

        # Create context
        self.con = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_100
        )

        # Pre-allocate buffers
        self.rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.viewport = mujoco.MjrRect(0, 0, width, height)

        # Pre-allocate markers for all joints
        self.num_markers = 24  # Number of SMPL joints
        for i in range(self.num_markers):
            self.scene.ngeom += 1
            mujoco.mjv_initGeom(
                self.scene.geoms[i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.array([1, 0, 0, 1], dtype=np.float32),
            )
            # Set initial size
            self.scene.geoms[i].size[0] = 0.001
            # self.scene.geoms[i].size[:] = np.array([0.05, 0.05, 0.05])

    def render(self, data, joint_positions):
        """Render the scene.

        Reference:
            https://github.com/ZhengyiLuo/PHC/blob/master/scripts/vis/
            vis_motion_mj.py
        """
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        # Only add markers if show_markers is True
        if self.show_markers and joint_positions is not None:
            for i in range(min(self.num_markers, len(joint_positions))):
                add_visual_capsule(
                    self.scene,
                    joint_positions[i],
                    joint_positions[i]
                    + np.array([0.001, 0, 0]),  # Tiny offset
                    0.01,
                    np.array([1, 0, 0, 1], dtype=np.float32),
                )

        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.con)
        return np.flipud(self.rgb_buffer)

    def close(self):
        self.ctx.free()


def collect_all_motions(pkl_files: list) -> list:
    """Collect all individual motions from all pkl files.

    Returns:
        List of tuples: (motion_data, motion_key, source_file, motion_name)

    """
    all_motions = []

    print("Collecting all motions from pkl files...")
    for pkl_file in tqdm(pkl_files, desc="Loading pkl files"):
        try:
            motion_data = joblib.load(pkl_file)
            source_file = os.path.basename(pkl_file)

            for motion_key in motion_data.keys():
                # Create unique motion name: filename_motionkey
                base_name = os.path.splitext(source_file)[0]
                if len(motion_data.keys()) == 1:
                    # If only one motion in file, use just the filename
                    motion_name = base_name
                else:
                    # If multiple motions, append the motion key
                    motion_name = f"{base_name}_{motion_key}"

                all_motions.append(
                    (
                        motion_data[motion_key],  # Individual motion data
                        motion_key,  # Original motion key
                        pkl_file,  # Source file path
                        motion_name,  # Unique motion name for output
                    )
                )

        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")
            continue

    print(
        f"Collected {len(all_motions)} individual motions from "
        f"{len(pkl_files)} pkl files"
    )
    return all_motions


@ray.remote
def process_single_motion_remote(
    motion_data: Dict[str, Any], motion_name: str, cfg_dict: dict
) -> str:
    """Ray remote function to process a single motion."""
    try:
        # Convert dict back to DictConfig for compatibility
        cfg = DictConfig(cfg_dict)

        # Initialize MuJoCo resources (each worker needs its own)
        mj_model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
        mj_data = mujoco.MjData(mj_model)

        # Setup rendering
        width, height = 1280, 720
        show_markers = cfg.get("show_markers", True)
        renderer = OffscreenRenderer(
            mj_model, height, width, show_markers=show_markers
        )

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(cfg.video_dir, f"{motion_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Get fps from motion data
        if "fps" in motion_data:
            target_fps = motion_data["fps"]
        else:
            target_fps = 30

        skip_frames = cfg.skip_frames
        actual_fps = target_fps / skip_frames
        out = cv2.VideoWriter(out_path, fourcc, actual_fps, (width, height))

        try:
            total_frames = motion_data["dof"].shape[0]

            for time_step in range(0, total_frames, skip_frames):
                mj_data.qpos[:3] = motion_data["root_trans_offset"][time_step]
                mj_data.qpos[3:7] = motion_data["root_rot"][time_step][
                    [3, 0, 1, 2]
                ]
                mj_data.qpos[7:] = motion_data["dof"][time_step]

                mujoco.mj_forward(mj_model, mj_data)

                # Only pass joint positions if markers are enabled
                joint_positions = (
                    motion_data["smpl_joints"][time_step]
                    if show_markers
                    else None
                )

                rgb_buffer = renderer.render(mj_data, joint_positions)
                out.write(rgb_buffer)

        finally:
            out.release()
            renderer.close()

        return motion_name

    except Exception as e:
        # Return error information for debugging
        return f"ERROR_{motion_name}: {str(e)}"


class MotionRenderer:
    def __init__(self):
        # Will be initialized in process_motion to get the correct model path
        pass

    def process_single_motion(self, motion_data, motion_name, cfg):
        """Process a single motion (for sequential processing)."""
        # Initialize MuJoCo resources
        mj_model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
        mj_data = mujoco.MjData(mj_model)

        # Setup rendering
        width, height = 1280, 720
        show_markers = cfg.get("show_markers", True)
        renderer = OffscreenRenderer(
            mj_model, height, width, show_markers=show_markers
        )

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(cfg.video_dir, f"{motion_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Get fps from motion data
        if "fps" in motion_data:
            target_fps = motion_data["fps"]
        else:
            target_fps = 30

        skip_frames = cfg.skip_frames
        actual_fps = target_fps / skip_frames
        out = cv2.VideoWriter(out_path, fourcc, actual_fps, (width, height))

        try:
            total_frames = motion_data["dof"].shape[0]

            for time_step in tqdm(
                range(0, total_frames, skip_frames),
                desc=f"Rendering {motion_name}",
            ):
                mj_data.qpos[:3] = motion_data["root_trans_offset"][time_step]
                mj_data.qpos[3:7] = motion_data["root_rot"][time_step][
                    [3, 0, 1, 2]
                ]
                mj_data.qpos[7:] = motion_data["dof"][time_step]

                mujoco.mj_forward(mj_model, mj_data)

                # Only pass joint positions if markers are enabled
                joint_positions = (
                    motion_data["smpl_joints"][time_step]
                    if show_markers
                    else None
                )

                rgb_buffer = renderer.render(mj_data, joint_positions)
                out.write(rgb_buffer)

        finally:
            out.release()
            renderer.close()

        return motion_name

    def process_motion(self, motion_file, cfg):
        """Sequential processing method for single pkl file."""
        # Load motion data
        motion_data = joblib.load(motion_file)
        motion_data_keys = list(motion_data.keys())

        # Extract motion name from file path
        base_motion_name = os.path.splitext(os.path.basename(motion_file))[0]

        results = []
        for motion_key in motion_data_keys:
            if len(motion_data_keys) == 1:
                motion_name = base_motion_name
            else:
                motion_name = f"{base_motion_name}_{motion_key}"

            result = self.process_single_motion(
                motion_data[motion_key], motion_name, cfg
            )
            results.append(result)

        return results


@hydra.main(
    version_base=None,
    config_path="../../../config/motion_retargeting",
    config_name="mujoco_viz_config",
)
def main(cfg: DictConfig) -> None:
    try:
        # Set default for show_markers if not specified
        if "show_markers" not in cfg:
            cfg.show_markers = True

        if cfg.motion_name == "all":
            # Find all pkl files
            pkl_files = glob.glob(os.path.join(cfg.motion_pkl_root, "*.pkl"))
            print(f"Found {len(pkl_files)} pkl files")

            # First, collect all individual motions from all pkl files
            all_motions = collect_all_motions(pkl_files)

            if not all_motions:
                print("No motions found to process!")
                return

            print(
                f"Markers are {'enabled' if cfg.show_markers else 'disabled'}"
            )

            # Initialize Ray
            if not ray.is_initialized():
                # Get number of CPUs,
                # but limit to reasonable number for rendering
                num_cpus = min(os.cpu_count(), cfg.get("max_workers", 8))
                ray.init(num_cpus=num_cpus)
                print(f"Initialized Ray with {num_cpus} workers")

            # Convert DictConfig to regular dict for Ray serialization
            cfg_dict = dict(cfg)

            # Submit all individual motions to Ray
            print("Submitting parallel tasks for individual motions...")
            tasks = []
            for (
                motion_data,
                _motion_key,
                _source_file,
                motion_name,
            ) in all_motions:
                task = process_single_motion_remote.remote(
                    motion_data, motion_name, cfg_dict
                )
                tasks.append((task, motion_name))

            # Monitor progress and collect results
            print("Processing motions in parallel...")
            completed_tasks = []
            failed_tasks = []

            with tqdm(total=len(tasks), desc="Processing Motions") as pbar:
                remaining_tasks = [task for task, _ in tasks]
                task_to_name = {task: name for task, name in tasks}

                while remaining_tasks:
                    # Check for completed tasks
                    ready_tasks, remaining_tasks = ray.wait(
                        remaining_tasks, num_returns=1, timeout=1.0
                    )

                    for task in ready_tasks:
                        motion_name = task_to_name[task]
                        try:
                            result = ray.get(task)
                            if result.startswith("ERROR_"):
                                failed_tasks.append(result)
                                print(f"Failed: {result}")
                            else:
                                completed_tasks.append(result)
                                print(f"Completed: {result}")
                        except Exception as e:
                            failed_tasks.append(
                                f"Task error for {motion_name}: {str(e)}"
                            )
                            print(f"Task failed with exception: {e}")

                        pbar.update(1)

            print("\nProcessing complete!")
            print(f"Successfully processed: {len(completed_tasks)} motions")
            if failed_tasks:
                print(f"Failed: {len(failed_tasks)} motions")
                for failure in failed_tasks:
                    print(f"  - {failure}")

            # Shutdown Ray
            ray.shutdown()

        else:
            # Single file processing (no Ray needed)
            motion_file = os.path.join(
                cfg.motion_pkl_root, f"{cfg.motion_name}.pkl"
            )
            print(
                f"Markers are {'enabled' if cfg.show_markers else 'disabled'}"
            )
            renderer = MotionRenderer()
            results = renderer.process_motion(motion_file, cfg)
            print(f"Processed motions: {results}")

    except Exception as e:
        print(f"Error during processing: {e}")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
