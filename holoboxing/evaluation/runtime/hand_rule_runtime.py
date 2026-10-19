"""Raw 200Hz contact adapter for the existing GPU solver and scene."""

import numpy as np
import warp as wp
from .hand_hit_rules import configured_thresholds, target_part


@wp.kernel
def sample_contacts(
    ncon: wp.array(dtype=int),
    geoms: wp.array(dtype=wp.vec2i),
    distances: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.mat33),
    forces: wp.array(dtype=wp.spatial_vector),
    slot_map: wp.array2d(dtype=int),
    geom_body: wp.array(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    com: wp.array2d(dtype=wp.vec3),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    roots: wp.array(dtype=int),
    thresholds: wp.array(dtype=float),
    index: int,
    capacity: int,
    output: wp.array3d(dtype=float),
):
    slot = wp.tid()
    touching = float(0.0)
    best_force = float(-1.0)
    best_speed = float(0.0)
    best_valid = float(0.0)
    best_position = wp.vec3(0.0)
    normal_force_sum = float(0.0)
    for contact in range(wp.min(ncon[0], capacity)):
        pair = geoms[contact]
        if pair[0] < 0 or pair[1] < 0 or distances[contact] > 0.0:
            continue
        for direction in range(2):
            strike, target = pair[direction], pair[1 - direction]
            if slot_map[strike, target] != slot:
                continue
            touching = 1.0
            value = forces[contact]
            force = wp.length(wp.vec3(value[0], value[1], value[2]))
            normal_force_sum += wp.max(value[0], 0.0)
            a, b = geom_body[strike], geom_body[target]
            va, vb = cvel[0, a], cvel[0, b]
            linear_a = wp.vec3(va[3], va[4], va[5]) + wp.cross(
                wp.vec3(va[0], va[1], va[2]),
                positions[contact] - com[0, roots[a]],
            )
            linear_b = wp.vec3(vb[3], vb[4], vb[5]) + wp.cross(
                wp.vec3(vb[0], vb[1], vb[2]),
                positions[contact] - com[0, roots[b]],
            )
            frame = frames[contact]
            axis = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
            if direction == 1:
                axis = -axis
            speed = wp.max(0.0, wp.dot(linear_a - linear_b, axis))
            valid = float(force >= thresholds[0] and speed >= thresholds[1])
            if valid > best_valid or (
                valid == best_valid and force > best_force
            ):
                best_force, best_speed, best_valid = force, speed, valid
                best_position = positions[contact]
    output[index, slot, 0] = touching
    output[index, slot, 1] = wp.max(0.0, best_force)
    output[index, slot, 2] = best_speed
    output[index, slot, 3] = best_position[0]
    output[index, slot, 4] = best_position[1]
    output[index, slot, 5] = best_position[2]
    output[index, slot, 6] = best_valid
    output[index, slot, 7] = normal_force_sum


def install():
    from .gpu_physics import ResidentPhysics

    if getattr(ResidentPhysics, "_hand_rules_installed", False):
        return
    ResidentPhysics._hand_rules_installed = True
    prepare, substep, frame = (
        ResidentPhysics.prepare_contacts,
        ResidentPhysics.substep,
        ResidentPhysics.frame,
    )

    def prepare_contacts(self):
        prepare(self)
        self.hand_thresholds = wp.array(
            np.asarray(configured_thresholds(), np.float32),
            device=str(self.device),
        )
        self.hand_samples = wp.zeros(
            (self.scene.control_decimation, len(self.keys), 8),
            dtype=float,
            device=str(self.device),
        )

    def step(self, index=0):
        substep(self, index)
        wp.launch(
            sample_contacts,
            dim=len(self.keys),
            inputs=[
                self.data.nacon,
                self.data.contact.geom,
                self.data.contact.dist,
                self.data.contact.pos,
                self.data.contact.frame,
                self.forces,
                self.slot_map,
                self.geom_body,
                self.data.xpos,
                self.data.xipos,
                self.data.subtree_com,
                self.data.cvel,
                self.body_roots,
                self.hand_thresholds,
                index,
                self.max_contacts,
                self.hand_samples,
            ],
        )

    def take_frame(self):
        pending = frame(self)
        samples = self.hand_samples.numpy()
        self._hand_sample_values = samples.copy()
        dt = float(self.scene.model.opt.timestep)
        end = self.steps * dt
        self.scene._hand_contact_samples = []
        for index, sub in enumerate(samples):
            contacts = []
            for slot, ((actor, attack, _), semantic, values) in enumerate(
                zip(self.keys, self.semantics, sub)
            ):
                if not values[0] or target_part(semantic) is None:
                    continue
                contacts.append(
                    dict(
                        attacker=actor,
                        defender=1 - actor,
                        attacking_body=attack,
                        target_body=semantic,
                        physical_contact=True,
                        peak_force=float(values[1]),
                        closing_speed=float(values[2]),
                        position=values[3:6].tolist(),
                        normal_force_sum_n=float(values[7]),
                        normal_impulse_step_ns=float(values[7]) * dt,
                        sample_dt_s=dt,
                        pair_id=slot,
                    )
                )
            self.scene._hand_contact_samples.append(
                (end - (len(samples) - 1 - index) * dt, contacts)
            )
        return pending

    ResidentPhysics.prepare_contacts = prepare_contacts
    ResidentPhysics.substep = step
    ResidentPhysics.frame = take_frame
