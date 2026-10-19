"""Current-qpos body poses for policies, separate from physics/referee data.

This is the independent MuJoCo FK correction shared by the source and compiled
observation runtimes. Only body positions and rotations are refreshed here.
The transport uses Warp so the participant image does not require PyTorch.
"""

import mujoco
import warp as wp


class CurrentObservationFK:
    """Fixed-address body-pose buffers, refreshed once per control step."""

    def __init__(self, scene):
        self.scene = scene
        self.data = mujoco.MjData(scene.model)
        device = scene._gpu_physics.data.qpos.device
        self.qpos_host = wp.zeros(
            scene.model.nq, device="cpu", pinned=device.is_cuda
        )
        self.positions_host = wp.zeros(
            (scene.model.nbody, 3), device="cpu", pinned=device.is_cuda
        )
        self.rotations_host = wp.zeros(
            (scene.model.nbody, 4), device="cpu", pinned=device.is_cuda
        )
        self.positions = wp.zeros((scene.model.nbody, 3), device=device)
        self.rotations = wp.zeros((scene.model.nbody, 4), device=device)
        self.refreshes = 0
        self.refresh()

    def refresh(self):
        # Resolve data on every call: reset can replace the GPU joint buffers.
        physics = self.scene._gpu_physics.data
        wp.copy(self.qpos_host, physics.qpos, count=self.scene.model.nq)
        wp.synchronize_device(physics.qpos.device)
        self.data.qpos[:] = self.qpos_host.numpy()
        mujoco.mj_kinematics(self.scene.model, self.data)
        self.positions_host.numpy()[:] = self.data.xpos
        self.rotations_host.numpy()[:] = self.data.xquat
        wp.copy(self.positions, self.positions_host)
        wp.copy(self.rotations, self.rotations_host)
        wp.synchronize_device(physics.qpos.device)
        self.refreshes += 1
