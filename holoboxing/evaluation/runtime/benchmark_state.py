"""Frozen scene buffers for inference benchmarks, without a physics solver.

The benchmark already keeps its scene at the reset pose. Allocate the same
GPU state consumed by production observations without compiling collision or
constraint kernels. This fixture cannot advance a match.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

from .onnx_device import require_gpu


class BenchmarkState:
    def __init__(self, scene, device):
        self.scene = scene
        self.device = require_gpu(device)
        self.steps = self.gpu_frames = 0
        self.targets_gpu = wp.zeros(58, dtype=float, device=str(self.device))
        self.forward_from_host()

    def forward_from_host(self):
        scene = self.scene
        mujoco.mj_kinematics(scene.model, scene.data)
        mujoco.mj_comPos(scene.model, scene.data)
        mujoco.mj_comVel(scene.model, scene.data)
        arrays = {}
        for name, dtype in (
            ("qpos", float),
            ("qvel", float),
            ("xpos", wp.vec3),
            ("xquat", wp.quat),
            ("xipos", wp.vec3),
            ("cvel", wp.spatial_vector),
            ("subtree_com", wp.vec3),
        ):
            arrays[name] = wp.array(
                np.asarray(getattr(scene.data, name), dtype=np.float32)[None],
                dtype=dtype,
                device=str(self.device),
            )
        self.data = SimpleNamespace(**arrays)
        self.steps = self.gpu_frames = 0
        wp.synchronize_device(str(self.device))

    def frame(self):
        raise RuntimeError("An inference benchmark fixture cannot step physics")

    def close(self):
        wp.synchronize_device(str(self.device))
