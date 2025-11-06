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

import glob
import os
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import mujoco
import numpy as np
import ray
from omegaconf import DictConfig
from tqdm.auto import tqdm


class OffscreenRenderer:
    """Offscreen renderer (no SMPL markers or joint spheres)."""

    def __init__(self, model, height, width):
        self.model = model
        self.height = height
        self.width = width

        # Create OpenGL context
        self.ctx = mujoco.GLContext(width, height)
        self.ctx.make_current()

        # Scene and camera setup
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 4.0
        self.cam.azimuth = 60.0
        self.cam.elevation = -20
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

        # Rendering context
        self.con = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_100
        )

        # Buffers
        self.rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.viewport = mujoco.MjrRect(0, 0, width, height)

    def render(self, data):
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.con)
        return np.flipud(self.rgb_buffer)

    def close(self):
        self.ctx.free()


def _load_npz_as_motion(
    npz_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str]:
    """
    Load a single .npz file and return (arrays_dict, metadata_dict, motion_name)
    - metadata: parsed from JSON
    - motion_name: file name without extension
    """
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files if k != "metadata"}
        meta_raw = z.get("metadata", None)
        if meta_raw is None:
            metadata = {}
        else:
            metadata = {}
            try:
                metadata = dict(np.atleast_1d(meta_raw).tolist())
            except Exception:
                try:
                    metadata = {**(dict()), **(eval(str(meta_raw)))}
                except Exception:
                    pass
            # Parse metadata as JSON string
            try:
                import json

                metadata = json.loads(str(np.atleast_1d(meta_raw)[0]))
            except Exception:
                pass

    motion_name = os.path.splitext(os.path.basename(npz_path))[0]
    return arrays, metadata, motion_name


def _collect_all_npz(
    npz_root: str, motion_name: str
) -> List[Tuple[Dict[str, np.ndarray], Dict[str, Any], str]]:
    """Collect all NPZ files to process based on configuration."""
    print("Collecting NPZ files...", npz_root, motion_name)
    if motion_name == "all":
        npz_files = glob.glob(os.path.join(npz_root, "*.npz"))
    else:
        npz_files = [os.path.join(npz_root, f"{motion_name}.npz")]

    motions = []
    for f in tqdm(npz_files, desc="Loading npz files"):
        try:
            arrays, metadata, name = _load_npz_as_motion(f)
            motions.append((arrays, metadata, name))
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return motions


def _infer_fps_from_meta(
    metadata: Dict[str, Any], default_fps: float = 30.0
) -> float:
    """Infer FPS value from metadata."""
    try:
        return float(metadata.get("motion_fps", default_fps))
    except Exception:
        return float(default_fps)


def _time_length(*arrays) -> int:
    """Return the smallest time dimension length among given arrays, ignoring None."""
    T = None
    for a in arrays:
        if isinstance(a, np.ndarray) and a.ndim >= 1:
            t = a.shape[0]
            T = t if T is None else min(T, t)
    return T if T is not None else 0


@ray.remote
def process_single_motion_remote_npz(
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    motion_name: str,
    cfg_dict: dict,
) -> str:
    try:
        cfg = DictConfig(cfg_dict)

        # MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
        mj_data = mujoco.MjData(mj_model)

        # Renderer
        width, height = 1280, 720
        renderer = OffscreenRenderer(mj_model, height, width)

        # FPS
        src_fps = _infer_fps_from_meta(metadata, default_fps=30.0)
        skip_frames = getattr(cfg, "skip_frames", 1)
        actual_fps = src_fps / max(1, int(skip_frames))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(cfg.video_dir, f"{motion_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = cv2.VideoWriter(out_path, fourcc, actual_fps, (width, height))

        try:
            dof_pos = arrays.get("dof_pos", None)
            gpos = arrays.get("global_translation", None)  # (T, nb, 3)
            grot = arrays.get("global_rotation_quat", None)  # (T, nb, 4) xyzw

            if (
                not isinstance(dof_pos, np.ndarray)
                or not isinstance(gpos, np.ndarray)
                or not isinstance(grot, np.ndarray)
            ):
                raise ValueError(
                    "Missing required NPZ keys: dof_pos / global_translation / global_rotation_quat"
                )

            # Time dimension alignment
            T = _time_length(dof_pos, gpos, grot)
            if T == 0:
                raise ValueError("No valid frames found.")

            for t in range(0, T, max(1, int(skip_frames))):
                # Root position and quaternion: take from body 0
                root_pos = gpos[t, 0]
                root_quat_xyzw = grot[t, 0]
                root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]

                mj_data.qpos[:3] = root_pos
                mj_data.qpos[3:7] = root_quat_wxyz
                mj_data.qpos[7:] = dof_pos[t]

                mujoco.mj_forward(mj_model, mj_data)
                frame = renderer.render(mj_data)
                # Convert RGB (MuJoCo) -> BGR (OpenCV) before writing
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        finally:
            out.release()
            renderer.close()

        return motion_name

    except Exception as e:
        return f"ERROR_{motion_name}: {str(e)}"


class MotionRendererNPZ:
    def process_single_motion(
        self,
        arrays: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        motion_name: str,
        cfg: DictConfig,
    ):
        mj_model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
        mj_data = mujoco.MjData(mj_model)

        width, height = 1280, 720
        renderer = OffscreenRenderer(mj_model, height, width)

        src_fps = _infer_fps_from_meta(metadata, default_fps=30.0)
        skip_frames = getattr(cfg, "skip_frames", 1)
        actual_fps = src_fps / max(1, int(skip_frames))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(cfg.video_dir, f"{motion_name}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = cv2.VideoWriter(out_path, fourcc, actual_fps, (width, height))

        try:
            dof_pos = arrays.get("dof_pos", None)
            gpos = arrays.get("global_translation", None)
            grot = arrays.get("global_rotation_quat", None)

            if (
                not isinstance(dof_pos, np.ndarray)
                or not isinstance(gpos, np.ndarray)
                or not isinstance(grot, np.ndarray)
            ):
                raise ValueError(
                    "Missing required NPZ keys: dof_pos / global_translation / global_rotation_quat"
                )

            T = _time_length(dof_pos, gpos, grot)
            if T == 0:
                raise ValueError("No valid frames found.")

            for t in tqdm(
                range(0, T, max(1, int(skip_frames))),
                desc=f"Rendering {motion_name}",
            ):
                root_pos = gpos[t, 0]
                root_quat_xyzw = grot[t, 0]
                root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]

                mj_data.qpos[:3] = root_pos
                mj_data.qpos[3:7] = root_quat_wxyz
                mj_data.qpos[7:] = dof_pos[t]

                mujoco.mj_forward(mj_model, mj_data)
                frame = renderer.render(mj_data)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()
            renderer.close()

        return motion_name


@hydra.main(
    version_base=None,
    config_path="../../../config/motion_retargeting",
    config_name="unitree_G1_29dof_retargeting",
)
def main(cfg: DictConfig) -> None:
    """
    Required config fields:
    - cfg.robot.asset.assetFileName : Path to the MuJoCo XML file
    - cfg.video_dir : Output video directory
    - cfg.motion_npz_root : Directory containing NPZ files
    - cfg.motion_name : "all" or a specific clip name (without extension)
    - cfg.skip_frames : Frame step size (>=1)
    """
    try:
        # NPZ input
        motions = _collect_all_npz(cfg.motion_npz_root, cfg.motion_name)
        if not motions:
            print("No NPZ motions found.")
            return

        # Ray parallel or single-thread mode
        if cfg.motion_name == "all":
            if not ray.is_initialized():
                num_cpus = min(os.cpu_count(), cfg.get("max_workers", 8))
                ray.init(num_cpus=num_cpus)
                print(f"Initialized Ray with {num_cpus} workers")

            cfg_dict = dict(cfg)
            tasks = [
                process_single_motion_remote_npz.remote(
                    arr, meta, name, cfg_dict
                )
                for (arr, meta, name) in motions
            ]

            completed, failed = [], []
            with tqdm(total=len(tasks), desc="Processing Motions") as pbar:
                remaining = list(tasks)
                while remaining:
                    ready, remaining = ray.wait(
                        remaining, num_returns=1, timeout=1.0
                    )
                    for t in ready:
                        try:
                            res = ray.get(t)
                            if isinstance(res, str) and res.startswith(
                                "ERROR_"
                            ):
                                failed.append(res)
                                print(f"Failed: {res}")
                            else:
                                completed.append(res)
                                print(f"Completed: {res}")
                        except Exception as e:
                            failed.append(f"Task exception: {e}")
                        pbar.update(1)

            print("\nProcessing complete!")
            print(f"Success: {len(completed)}; Failed: {len(failed)}")
            if failed:
                for f in failed:
                    print("  -", f)
            ray.shutdown()
        else:
            renderer = MotionRendererNPZ()
            for arr, meta, name in motions:
                res = renderer.process_single_motion(arr, meta, name, cfg)
                print(f"Processed: {res}")

    except Exception as e:
        print(f"Error during processing: {e}")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
