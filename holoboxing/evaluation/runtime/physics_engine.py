"""MuJoCo Warp allocation and reset for one shared GPU arena."""

import warp as wp
import mujoco_warp as mjw

from .onnx_device import require_gpu
from .deterministic_physics import (
    OrderedArena,
    check_supported,
    ordered_physics,
)


class WarpPhysics:
    def __init__(self, scene, device):
        self.scene = scene
        self.device = require_gpu(device)
        self.warp, self.engine = wp, mjw
        self.steps = 0
        self.max_contacts = 4096
        self.max_constraints = 8192
        check_supported(scene.model)
        with wp.ScopedDevice(str(self.device)):
            self.model = mjw.put_model(scene.model)
            self.ordering = OrderedArena(self.model)
            self.forward_from_host()

    def forward_from_host(self):
        with wp.ScopedDevice(str(self.device)):
            self.data = mjw.put_data(
                self.scene.model,
                self.scene.data,
                nworld=1,
                nconmax=self.max_contacts,
                njmax=self.max_constraints,
            )
            self.ordering.overflow.zero_()
            with ordered_physics(self.ordering):
                mjw.forward(self.model, self.data)
            wp.synchronize_device(str(self.device))
            if self.ordering.overflow.numpy()[0]:
                raise RuntimeError(
                    "Ordered GPU collision capacity exhausted during reset"
                )
        self.steps = 0
        # Initial and explicit reset state, never a CPU simulation step.
        for name in (
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "xipos",
            "cvel",
            "subtree_com",
            "geom_xpos",
        ):
            getattr(self.scene.data, name)[:] = getattr(
                self.data, name
            ).numpy()[0]
        self.scene.data.time = float(self.data.time.numpy()[0])

    def close(self):
        wp.synchronize_device(str(self.device))
