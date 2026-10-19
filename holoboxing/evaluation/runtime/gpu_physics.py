"""GPU-resident 200 Hz physics/PD/contact reduction; one 50 Hz host snapshot.

Reuse the original model, solver, force threshold, strike/target mapping and
Python referee. No physical substep or scoring contact is skipped.
"""

import numpy as np
import warp as wp
from .physics_engine import WarpPhysics
from .deterministic_physics import ordered_physics


@wp.kernel
def pd(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qi: wp.array(dtype=int),
    vi: wp.array(dtype=int),
    ai: wp.array(dtype=int),
    targets: wp.array(dtype=float),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    limits: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    i = wp.tid()
    joint = i % 29
    torque = (
        kp[joint] * (targets[i] - qpos[0, qi[i]]) - kd[joint] * qvel[0, vi[i]]
    )
    ctrl[0, ai[i]] = wp.clamp(torque, -limits[joint], limits[joint])


@wp.func
def body_velocity(
    body: int,
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xipos: wp.array2d(dtype=wp.vec3),
    com: wp.array2d(dtype=wp.vec3),
    roots: wp.array(dtype=int),
):
    value = cvel[0, body]
    angular = wp.vec3(value[0], value[1], value[2])
    linear = wp.vec3(value[3], value[4], value[5])
    return linear + wp.cross(angular, xipos[0, body] - com[0, roots[body]])


@wp.kernel
def reduce_contacts(
    ncon: wp.array(dtype=int),
    nefc: wp.array(dtype=int),
    geoms: wp.array(dtype=wp.vec2i),
    forces: wp.array(dtype=wp.spatial_vector),
    slot_map: wp.array2d(dtype=int),
    geom_body: wp.array(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    com: wp.array2d(dtype=wp.vec3),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    roots: wp.array(dtype=int),
    threshold: float,
    max_contacts: int,
    max_constraints: int,
    substep_index: int,
    stats: wp.array2d(dtype=float),
    overflow: wp.array(dtype=int),
):
    slot = wp.tid()
    if ncon[0] > max_contacts or nefc[0] > max_constraints:
        wp.atomic_max(overflow, 0, 1)
        return
    peak = stats[slot, 0]
    closing = stats[slot, 1]
    position = wp.vec3(stats[slot, 2], stats[slot, 3], stats[slot, 4])
    first_order = stats[slot, 5]
    # One thread per semantic key preserves first-contact tie handling and
    # removes atomics/races when choosing the strongest contact's position.
    for contact in range(ncon[0]):
        pair = geoms[contact]
        if pair[0] < 0 or pair[1] < 0:
            continue
        value = forces[contact]
        force = wp.length(wp.vec3(value[0], value[1], value[2]))
        if force < threshold:
            continue
        for direction in range(2):
            strike = pair[direction]
            target = pair[1 - direction]
            if slot_map[strike, target] != slot:
                continue
            if peak == 0.0:
                first_order = float(
                    substep_index * max_contacts * 2 + contact * 2 + direction
                )
            strike_body = geom_body[strike]
            target_body = geom_body[target]
            start = xpos[0, strike_body]
            end = xpos[0, target_body]
            delta = end - start
            axis = delta / wp.max(wp.length(delta), 1.0e-8)
            speed = wp.max(
                0.0,
                wp.dot(
                    body_velocity(strike_body, cvel, xipos, com, roots)
                    - body_velocity(target_body, cvel, xipos, com, roots),
                    axis,
                ),
            )
            closing = wp.max(closing, speed)
            if force > peak:
                peak = force
                position = 0.5 * (start + end)
    stats[slot, 0] = peak
    stats[slot, 1] = closing
    stats[slot, 2] = position[0]
    stats[slot, 3] = position[1]
    stats[slot, 4] = position[2]
    stats[slot, 5] = first_order


class ResidentPhysics(WarpPhysics):
    def __init__(self, scene, device):
        self.frame_graph = None
        self.gpu_frames = 0
        super().__init__(scene, device)
        self.targets_gpu = wp.zeros(58, dtype=float, device=str(self.device))
        self.targets_host = wp.zeros(
            58, dtype=float, device="cpu", pinned=True
        )
        self.qi = wp.array(
            np.asarray(scene._joint_qpos).reshape(-1).astype(np.int32),
            device=str(self.device),
        )
        self.vi = wp.array(
            np.asarray(scene._joint_dof).reshape(-1).astype(np.int32),
            device=str(self.device),
        )
        self.ai = wp.array(
            np.asarray(scene._actuator).reshape(-1).astype(np.int32),
            device=str(self.device),
        )
        self.kp_gpu = wp.array(
            scene._kp.astype(np.float32), device=str(self.device)
        )
        self.kd_gpu = wp.array(
            scene._kd.astype(np.float32), device=str(self.device)
        )
        self.limits_gpu = wp.array(
            scene._torque_limit.astype(np.float32), device=str(self.device)
        )
        self.keys = []
        self.semantics = []
        self.slot_map = None

    def forward_from_host(self):
        self.frame_graph = None
        super().forward_from_host()
        if (
            int(self.data.nacon.numpy()[0]) > self.max_contacts
            or int(self.data.nefc.numpy()[0]) > self.max_constraints
        ):
            raise RuntimeError(
                "GPU contact/constraint capacity exhausted during reset"
            )

    def _readback(self):
        # The canonical policy/JSON interfaces consume only these snapshots.
        # Do not reconstruct contact Jacobians, inertia or factorization on CPU.
        self.warp.synchronize_device(str(self.device))
        for name in (
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "xipos",
            "cvel",
            "subtree_com",
        ):
            getattr(self.scene.data, name)[:] = getattr(
                self.data, name
            ).numpy()[0]
        self.scene.data.time = float(self.data.time.numpy()[0])
        if hasattr(self, "stats"):
            self._snapshot_stats = self.stats.numpy().copy()
            self._snapshot_overflow = bool(self.overflow.numpy()[0])
            self._snapshot_overflow |= bool(self.ordering.overflow.numpy()[0])

    def prepare_contacts(self):
        scene = self.scene
        mapping = np.full(
            (scene.model.ngeom, scene.model.ngeom), -1, dtype=np.int32
        )
        slots = {}
        for strike, (attacker, attacking_body) in scene._strike_geom.items():
            for target in range(scene.model.ngeom):
                resolved = scene._opponent_target_semantic(attacker, target)
                if resolved is None:
                    continue
                _, link, semantic = resolved
                key = (attacker, attacking_body, f"{link}:{semantic}")
                if key not in slots:
                    slots[key] = len(self.keys)
                    self.keys.append(key)
                    self.semantics.append(semantic)
                mapping[strike, target] = slots[key]
        device = str(self.device)
        self.slot_map = wp.array(mapping, device=device)
        self.geom_body = wp.array(
            scene.model.geom_bodyid.astype(np.int32), device=device
        )
        self.body_roots = wp.array(
            scene.model.body_rootid.astype(np.int32), device=device
        )
        self.ids = wp.array(
            np.arange(self.max_contacts, dtype=np.int32), device=device
        )
        self.forces = wp.zeros(
            self.max_contacts, dtype=wp.spatial_vector, device=device
        )
        self.stats = wp.zeros((len(self.keys), 6), dtype=float, device=device)
        self.overflow = wp.zeros(1, dtype=int, device=device)

    def substep(self, index=0):
        wp.launch(
            pd,
            dim=58,
            inputs=[
                self.data.qpos,
                self.data.qvel,
                self.qi,
                self.vi,
                self.ai,
                self.targets_gpu,
                self.kp_gpu,
                self.kd_gpu,
                self.limits_gpu,
                self.data.ctrl,
            ],
        )
        with ordered_physics(self.ordering):
            self.engine.step(self.model, self.data)
            self.engine.contact_force(
                self.model, self.data, self.ids, False, self.forces
            )
        wp.launch(
            reduce_contacts,
            dim=len(self.keys),
            inputs=[
                self.data.nacon,
                self.data.nefc,
                self.data.contact.geom,
                self.forces,
                self.slot_map,
                self.geom_body,
                self.data.xpos,
                self.data.xipos,
                self.data.subtree_com,
                self.data.cvel,
                self.body_roots,
                self.scene.minimum_contact_force,
                self.max_contacts,
                self.max_constraints,
                index,
                self.stats,
                self.overflow,
            ],
        )

    def frame(self):
        if self.slot_map is None:
            self.prepare_contacts()
        # Match setup replaces gains with metadata from the saved controller.
        # Refresh the same buffers without changing the captured addresses.
        if self.frame_graph is None:
            self.kp_gpu.assign(self.scene._kp.astype(np.float32))
            self.kd_gpu.assign(self.scene._kd.astype(np.float32))
        if not getattr(self.scene, "_targets_on_gpu", False):
            self.targets_host.numpy()[:] = self.scene._targets.reshape(-1)
        with wp.ScopedDevice(str(self.device)):
            if not getattr(self.scene, "_targets_on_gpu", False):
                wp.copy(self.targets_gpu, self.targets_host)
            self.stats.zero_()
            self.overflow.zero_()
            self.ordering.overflow.zero_()
            if self.frame_graph is None:
                with wp.ScopedCapture() as capture:
                    for index in range(self.scene.control_decimation):
                        self.substep(index)
                self.frame_graph = capture.graph
            wp.capture_launch(self.frame_graph)
        self.steps += self.scene.control_decimation
        self.gpu_frames += 1
        self._readback()
        if self._snapshot_overflow:
            raise RuntimeError(
                "GPU contact/constraint capacity exhausted; refusing truncated physics"
            )
        pending = {}
        if (
            getattr(self.scene, "_gpu_referee", None) is not None
            and __import__("os").environ.get("BOXING_COMPARE_OBSERVATIONS")
            != "1"
        ):
            return pending
        ordered = sorted(
            zip(self.keys, self.semantics, self._snapshot_stats, strict=True),
            key=lambda item: item[2][5],
        )
        for key, semantic, values in ordered:
            if values[0] < self.scene.minimum_contact_force:
                continue
            pending[key] = dict(
                target_body=semantic,
                closing_speed=float(values[1]),
                peak_force=float(values[0]),
                position=values[2:5].astype(np.float64),
            )
        return pending
