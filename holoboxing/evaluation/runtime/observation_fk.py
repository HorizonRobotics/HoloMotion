"""Current-qpos/qvel policy state, separate from physics/referee data.

This is the independent MuJoCo FK correction shared by the source and compiled
observation runtimes. Refresh poses and spatial velocities from one state.
The transport uses Warp so the participant image does not require PyTorch.
"""

import mujoco
import warp as wp


class CurrentObservationFK:
    """Fixed-address current-state buffers, refreshed once per control step."""

    def __init__(self, scene):
        self.scene = scene
        self.data = mujoco.MjData(scene.model)
        device = scene._gpu_physics.data.qpos.device
        self.qpos_host = wp.zeros(
            scene.model.nq, device="cpu", pinned=device.is_cuda
        )
        self.qvel_host = wp.zeros(
            scene.model.nv, device="cpu", pinned=device.is_cuda
        )
        self.positions_host = wp.zeros(
            (scene.model.nbody, 3), device="cpu", pinned=device.is_cuda
        )
        self.rotations_host = wp.zeros(
            (scene.model.nbody, 4), device="cpu", pinned=device.is_cuda
        )
        self.cvel_host = wp.zeros(
            (scene.model.nbody, 6), device="cpu", pinned=device.is_cuda
        )
        self.subtree_com_host = wp.zeros(
            (scene.model.nbody, 3), device="cpu", pinned=device.is_cuda
        )
        self.qvel = wp.zeros(scene.model.nv, device=device)
        self.positions = wp.zeros((scene.model.nbody, 3), device=device)
        self.rotations = wp.zeros((scene.model.nbody, 4), device=device)
        self.cvel = wp.zeros((scene.model.nbody, 6), device=device)
        self.subtree_com = wp.zeros((scene.model.nbody, 3), device=device)
        self.refreshes = 0
        self.refresh()

    def refresh(self):
        # Resolve data on every call: reset can replace the GPU joint buffers.
        physics = self.scene._gpu_physics.data
        wp.copy(self.qpos_host, physics.qpos, count=self.scene.model.nq)
        wp.copy(self.qvel, physics.qvel, count=self.scene.model.nv)
        wp.copy(self.qvel_host, self.qvel)
        wp.synchronize_device(physics.qpos.device)
        self.data.qpos[:] = self.qpos_host.numpy()
        self.data.qvel[:] = self.qvel_host.numpy()
        mujoco.mj_kinematics(self.scene.model, self.data)
        # comPos supplies both the current tree centers and joint motion axes.
        # comVel then expresses current qvel at those same reference centers.
        # Never forward the real physics data: contacts/solver caches are separate.
        mujoco.mj_comPos(self.scene.model, self.data)
        mujoco.mj_comVel(self.scene.model, self.data)
        self.positions_host.numpy()[:] = self.data.xpos
        self.rotations_host.numpy()[:] = self.data.xquat
        self.cvel_host.numpy()[:] = self.data.cvel
        self.subtree_com_host.numpy()[:] = self.data.subtree_com
        wp.copy(self.positions, self.positions_host)
        wp.copy(self.rotations, self.rotations_host)
        wp.copy(self.cvel, self.cvel_host)
        wp.copy(self.subtree_com, self.subtree_com_host)
        wp.synchronize_device(physics.qpos.device)
        self.refreshes += 1
