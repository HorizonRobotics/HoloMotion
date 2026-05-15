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


"""Minimal Ray actor for batch eval. Lives in its own module so the class
can be pickled without pulling in torch/jit from eval_mujoco_sim2sim.
"""

import importlib
import os
import sys

import ray
from loguru import logger


class RayEvaluatorActor:
    """Persistent Ray actor: one evaluator (one ONNX session) per actor.

    Schedule with num_gpus=1/actors_per_gpu so that multiple actors share one GPU.
    Ray sets CUDA_VISIBLE_DEVICES so this actor sees a single GPU as device 0.
    """

    def __init__(self, config_dict, output_dir):
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        self.output_dir = output_dir
        self.config_dict = config_dict
        model_type = config_dict.get("model_type") or "holomotion"
        self.evaluator = _load_ray_evaluator(config_dict, model_type)
        self.evaluator.setup()
        if model_type == "gmt":
            self.evaluator.gmt_proprio_buf.clear()

    def ready(self):
        return "ready"

    def run_clip(self, file_path):
        from holomotion.src.evaluation.eval_mujoco_sim2sim import (
            _build_onnx_io_dump_dir,
            _build_onnx_io_dump_path,
        )

        fname = os.path.basename(file_path)
        save_name = fname.replace(".npz", "_eval.npz")
        save_path = os.path.join(self.output_dir, save_name)
        self.evaluator.load_specific_motion(file_path)
        self.evaluator.reset_state_teleport()
        for i in range(self.evaluator.n_motion_frames):
            self.evaluator.motion_frame_idx = i
            self.evaluator._update_policy()
            self.evaluator._apply_control(sleep=False)
            self.evaluator.counter += 1
        meta = {
            "source_file": fname,
            "model": str(self.config_dict.get("ckpt_onnx_path", "")),
            "source_npz": fname,
            "onnx_model": str(self.config_dict.get("ckpt_onnx_path", "")),
        }
        self.evaluator.save_batch_result(save_path, meta)
        model_type = self.config_dict.get("model_type") or "holomotion"
        if bool(self.config_dict.get("dump_onnx_io_npy", False)) and (
            model_type == "holomotion"
        ):
            onnx_io_dir = _build_onnx_io_dump_dir(self.output_dir)
            os.makedirs(onnx_io_dir, exist_ok=True)
            self.evaluator.save_onnx_io_dump(
                _build_onnx_io_dump_path(self.output_dir, fname), meta
            )
        return "success"


def _load_ray_evaluator(config_dict, model_type):
    module_name = config_dict.get(
        "ray_evaluator_module",
        "holomotion.src.evaluation.eval_mujoco_sim2sim",
    )
    factory_module = importlib.import_module(module_name)
    return factory_module._create_ray_evaluator(config_dict, model_type)
