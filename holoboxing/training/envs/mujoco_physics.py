"""Batched competition XML physics with explicit named state and PD inputs."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import collision_gjk

from holoboxing.data.records import initial_qpos
from holoboxing.data.robot import load_scene_model
from holoboxing.evaluation.runtime.deterministic_physics import (
    OrderedArena,
    check_supported,
    ordered_physics,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)
from holoboxing.training.tasks.combat.mdp.geometry import matrix, rotate


@wp.func
def _project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):
    # Float32 EPA can rank a thin face incorrectly and put hand/ring contacts
    # hundreds of metres away. Keep the plane calculation in float64, not the
    # simulation state; do not discard contacts or alter solver parameters.
    a, b, c = wp.vec3d(v1), wp.vec3d(v2), wp.vec3d(v3)
    normal = wp.cross(b - a, c - a)
    squared_norm = wp.dot(normal, normal)
    if squared_norm == wp.float64(0.0):
        return wp.vec3(0.0), int(1)
    return wp.vec3(normal * (wp.dot(normal, a) / squared_norm)), int(0)


@wp.kernel
def pd(
    q: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    qi: wp.array2d(dtype=int),
    vi: wp.array2d(dtype=int),
    ai: wp.array2d(dtype=int),
    target: wp.array3d(dtype=float),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    limits: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    world, role, j = wp.tid()
    effort = kp[j] * (target[world, role, j] - q[world, qi[role, j]])
    effort -= kd[j] * v[world, vi[role, j]]
    ctrl[world, ai[role, j]] = wp.clamp(
        effort, -limits[role, j], limits[role, j]
    )


@wp.kernel
def indexed_contact_force(
    count: wp.array(dtype=int),
    worlds: wp.array(dtype=int),
    dimensions: wp.array(dtype=int),
    addresses: wp.array2d(dtype=int),
    friction: wp.array2d(dtype=float),
    frames: wp.array(dtype=wp.mat33),
    constraint_count: wp.array(dtype=int),
    constraint_force: wp.array2d(dtype=float),
    result: wp.array(dtype=wp.spatial_vector),
):
    c = wp.tid()
    if c >= count[0]:
        return
    world, dimension = worlds[c], dimensions[c]
    force = wp.spatial_vector()
    if dimension == 1:
        address = addresses[c, 0]
        if address >= 0 and address < constraint_count[world]:
            force[0] = constraint_force[world, address]
    else:
        # Native batched allocation can interleave edges from different contacts.
        # Each edge has its own address; base + offset is not a valid lookup.
        for i in range(dimension - 1):
            first, second = addresses[c, 2 * i], addresses[c, 2 * i + 1]
            positive, negative = float(0.0), float(0.0)
            if first >= 0 and first < constraint_count[world]:
                positive = constraint_force[world, first]
            if second >= 0 and second < constraint_count[world]:
                negative = constraint_force[world, second]
            force[0] += positive + negative
            force[i + 1] = (positive - negative) * friction[c, i]
    result[c] = wp.spatial_vector(
        wp.spatial_top(force) @ frames[c],
        wp.spatial_bottom(force) @ frames[c],
    )


@wp.kernel
def contact_forces(
    count: wp.array(dtype=int),
    worlds: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    distance: wp.array(dtype=float),
    force: wp.array(dtype=wp.spatial_vector),
    bodies: wp.array(dtype=int),
    slots: wp.array2d(dtype=int),
    net: wp.array3d(dtype=wp.vec3),
    paired: wp.array3d(dtype=wp.vec3),
):
    c = wp.tid()
    if c >= count[0]:
        return
    w, pair = worlds[c], pairs[c]
    if w < 0 or pair[0] < 0 or pair[1] < 0:
        return
    f = wp.vec3(force[c][0], force[c][1], force[c][2])
    for side in range(2):
        body = bodies[pair[side]]
        if body >= 0:
            sign = float(2 * side - 1)
            wp.atomic_add(net, w, body // 30, body % 30, sign * f)
        slot = slots[pair[side], pair[1 - side]]
        if slot >= 0:
            wp.atomic_add(
                paired, w, slot // 2, slot % 2, float(2 * side - 1) * f
            )


class CompetitionPhysics:
    """No task rewards, policy updates, material changes or failure masking."""

    def __init__(
        self,
        assets,
        worlds,
        device,
        *,
        contact_capacity=None,
        constraint_capacity=8192,
        ordered=None,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda" or worlds < 1:
            raise ValueError(
                "Competition physics requires CUDA and positive worlds"
            )
        self.num_envs = worlds
        # Batch contacts share a pool, unlike the evaluator's single-arena reserve.
        if contact_capacity is None:
            contact_capacity = 4096 if worlds == 1 else 256
        wp.set_device(str(self.device))
        wp.set_stream(
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
        )
        self.host = load_scene_model(assets)
        check_supported(self.host)
        # Version-gated above; install before Warp compiles collision kernels.
        collision_gjk._project_origin_plane = _project_origin_plane
        self.initial = mujoco.MjData(self.host)
        self.initial.qpos[:] = initial_qpos(assets, self.host)
        mujoco.mj_forward(self.host, self.initial)
        self.model = mjw.put_model(self.host)
        self.data = mjw.put_data(
            self.host,
            self.initial,
            nworld=worlds,
            nconmax=contact_capacity,
            njmax=constraint_capacity,
        )
        self.fk = mjw.put_data(
            self.host,
            self.initial,
            nworld=worlds,
            nconmax=max(1, self.initial.ncon),
            njmax=max(1, self.initial.nefc),
        )
        self.ordered = worlds == 1 if ordered is None else ordered
        if self.ordered and worlds != 1:
            raise ValueError(
                "Evaluation's ordered physics supports one world only"
            )
        self.ordering = OrderedArena(self.model) if self.ordered else None
        qi, vi, ai, bids, roots = [], [], [], [], []
        actuator_by_joint = {
            int(j[0]): i for i, j in enumerate(self.host.actuator_trnid)
        }
        for prefix in ("p1_", "p2_"):
            joints = [
                self.host.joint(prefix + name).id for name in JOINT_NAMES
            ]
            qi.append(self.host.jnt_qposadr[joints])
            vi.append(self.host.jnt_dofadr[joints])
            ai.append([actuator_by_joint[j] for j in joints])
            bids.append(
                [self.host.body(prefix + name).id for name in BODY_NAMES]
            )
            roots.append(
                int(
                    self.host.jnt_qposadr[
                        self.host.body(prefix + "pelvis").jntadr[0]
                    ]
                )
            )
        self.qi, self.vi, self.ai = [
            torch.tensor(np.asarray(x), device=device) for x in (qi, vi, ai)
        ]
        self.body_ids = torch.tensor(bids, device=device)
        self.root_qi = roots
        self.root_ids = torch.tensor(
            self.host.body_rootid[np.asarray(bids)], device=device
        )
        limits = self.host.actuator_ctrlrange[np.asarray(ai)]
        if not self.host.actuator_ctrllimited[
            np.asarray(ai)
        ].all() or not np.allclose(limits[..., 0], -limits[..., 1]):
            raise ValueError("Expected symmetric scene torque limits")
        self.limits = wp.array(limits[..., 1].astype(np.float32))
        self.indices = [
            wp.array(np.asarray(x, dtype=np.int32)) for x in (qi, vi, ai)
        ]
        self.targets = (
            torch.tensor(
                self.initial.qpos[np.asarray(qi)],
                device=device,
                dtype=torch.float32,
            )
            .expand(worlds, -1, -1)
            .clone()
        )
        self.forces = wp.zeros(self.data.naconmax, dtype=wp.spatial_vector)
        self.contact_ids = wp.array(
            np.arange(self.data.naconmax, dtype=np.int32)
        )
        mapping = np.full(self.host.nbody, -1, dtype=np.int32)
        mapping[np.asarray(bids).flatten()] = np.arange(60)
        self.geom_bodies = wp.array(mapping[self.host.geom_bodyid])
        slots = np.full((self.host.ngeom, self.host.ngeom), -1, dtype=np.int32)
        for role, prefix in enumerate(("p1_", "p2_")):
            other = "p2_" if role == 0 else "p1_"
            torso = self.host.body(prefix + "torso_link").id
            for hand, side in enumerate(("left", "right")):
                wrist = self.host.body(other + side + "_wrist_yaw_link").id
                slots[
                    np.ix_(
                        self.host.geom_bodyid == torso,
                        self.host.geom_bodyid == wrist,
                    )
                ] = role * 2 + hand
        self.slots = wp.array(slots)
        self.net = wp.zeros((worlds, 2, 30), dtype=wp.vec3)
        self.paired = wp.zeros((worlds, 2, 2), dtype=wp.vec3)
        self.net_history = torch.zeros((worlds, 6, 2, 30, 3), device=device)
        self.paired_history = torch.zeros((worlds, 6, 2, 2, 3), device=device)
        self.steps = 0
        self.joint_acceleration = torch.zeros((worlds, 2, 29), device=device)
        self.forward()
        # Per-world reset copies do not recompute or disturb any other world.
        self.reset_state = {
            name: wp.to_torch(getattr(self.data, name))[:1].clone()
            for name in (
                "qpos",
                "qvel",
                "qacc",
                "qacc_warmstart",
                "ctrl",
                "time",
                "xpos",
                "xquat",
                "xipos",
                "cvel",
                "subtree_com",
            )
        }

    def forward(self):
        if self.ordered:
            with ordered_physics(self.ordering):
                mjw.forward(self.model, self.data)
        else:
            mjw.forward(self.model, self.data)
        self.check_capacity()

    def check_capacity(self):
        if (
            int(wp.to_torch(self.data.nacon)[0]) > self.data.naconmax
            or int(wp.to_torch(self.data.ncollision)[0]) > self.data.naccdmax
            or bool((wp.to_torch(self.data.nefc) > self.data.njmax).any())
        ):
            raise RuntimeError(
                "Contact/constraint capacity exhausted; physics is invalid"
            )
        if self.ordering is not None and int(
            wp.to_torch(self.ordering.overflow)[0]
        ):
            raise RuntimeError("Evaluation ordered contact capacity exhausted")

    def set_gains(self, kp, kd):
        self.kp, self.kd = [
            wp.from_torch(x.to(self.device, torch.float32).contiguous())
            for x in (kp, kd)
        ]

    def push(self, ids, role, delta_velocity_w):
        """Add a world-frame velocity impulse at the root center of mass."""
        root = self.host.body(("p1_", "p2_")[role] + "pelvis")
        qi = self.root_qi[role]
        vi = int(self.host.jnt_dofadr[root.jntadr[0]])
        qpos, qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
        rotation = matrix(qpos[ids, qi + 3 : qi + 7])
        ipos = wp.to_torch(self.model.body_ipos)
        local_com = (
            ipos[0, root.id] if ipos.shape[0] == 1 else ipos[ids, root.id]
        )
        offset_w = rotate(rotation, local_com)
        # Free-joint translation is at the body origin; angular velocity is local.
        qvel[ids, vi : vi + 3] += delta_velocity_w[:, :3] - torch.cross(
            delta_velocity_w[:, 3:], offset_w, dim=-1
        )
        qvel[ids, vi + 3 : vi + 6] += rotate(
            rotation.transpose(-1, -2), delta_velocity_w[:, 3:]
        )

    def randomize_torso_com(self, offsets):
        # MuJoCo-Warp broadcasts model parameters unless explicitly expanded.
        # Derived inertial parameters must also be per-world before set_const.
        fields = (
            "body_ipos",
            "body_subtreemass",
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
            "cam_pos0",
            "cam_poscom0",
            "cam_mat0",
            "light_pos0",
            "light_poscom0",
            "light_dir0",
            "actuator_acc0",
        )
        for name in fields:
            array = getattr(self.model, name)
            value = wp.to_torch(array)
            if value.shape[0] == 1:
                expanded = value.expand(
                    self.num_envs, *value.shape[1:]
                ).clone()
                setattr(
                    self.model,
                    name,
                    wp.from_torch(expanded, dtype=array.dtype),
                )
        inertia = wp.to_torch(self.model.stat.meaninertia)
        self.model.stat.meaninertia = wp.from_torch(
            inertia.expand(self.num_envs).clone()
        )
        com = wp.to_torch(self.model.body_ipos)
        for role, prefix in enumerate(("p1_", "p2_")):
            com[:, self.host.body(prefix + "torso_link").id] += offsets[
                :, role
            ]
        mjw.set_const(self.model, self.data)
        self.forward()

    def substep(self):
        previous_velocity = wp.to_torch(self.data.qvel)[:, self.vi].clone()
        wp.launch(
            pd,
            dim=(self.num_envs, 2, 29),
            inputs=[
                self.data.qpos,
                self.data.qvel,
                *self.indices,
                wp.from_torch(self.targets),
                self.kp,
                self.kd,
                self.limits,
                self.data.ctrl,
            ],
        )
        if self.ordered:
            with ordered_physics(self.ordering):
                mjw.step(self.model, self.data)
        else:
            mjw.step(self.model, self.data)
        self.check_capacity()
        # Use per-physics-step finite differences, not control-period averaging.
        self.joint_acceleration.copy_(
            (wp.to_torch(self.data.qvel)[:, self.vi] - previous_velocity)
            / self.host.opt.timestep
        )
        self.read_contact_forces()
        self.net.zero_()
        self.paired.zero_()
        wp.launch(
            contact_forces,
            dim=self.data.naconmax,
            inputs=[
                self.data.nacon,
                self.data.contact.worldid,
                self.data.contact.geom,
                self.data.contact.dist,
                self.forces,
                self.geom_bodies,
                self.slots,
                self.net,
                self.paired,
            ],
        )
        self.net_history[:, 1:] = self.net_history[:, :-1].clone()
        self.paired_history[:, 1:] = self.paired_history[:, :-1].clone()
        self.net_history[:, 0] = wp.to_torch(self.net)
        self.paired_history[:, 0] = wp.to_torch(self.paired)
        self.steps += 1

    def read_contact_forces(self):
        # check_supported restricts this scene to pyramidal friction cones.
        wp.launch(
            indexed_contact_force,
            dim=self.data.naconmax,
            inputs=[
                self.data.nacon,
                self.data.contact.worldid,
                self.data.contact.dim,
                self.data.contact.efc_address,
                wp.from_torch(wp.to_torch(self.data.contact.friction)),
                self.data.contact.frame,
                self.data.nefc,
                self.data.efc.force,
                self.forces,
            ],
        )

    def reset(self, ids):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        for name, value in self.reset_state.items():
            wp.to_torch(getattr(self.data, name))[ids] = value
        self.targets[ids] = self.reset_state["qpos"][:, self.qi]
        self.net_history[ids] = 0
        self.paired_history[ids] = 0
        self.joint_acceleration[ids] = 0

    def state(self):
        wp.copy(self.fk.qpos, self.data.qpos)
        wp.copy(self.fk.qvel, self.data.qvel)
        mjw.kinematics(self.model, self.fk)
        mjw.com_pos(self.model, self.fk)
        mjw.com_vel(self.model, self.fk)
        pos = wp.to_torch(self.fk.xpos)[:, self.body_ids]
        quat = wp.to_torch(self.fk.xquat)[:, self.body_ids]
        cv = wp.to_torch(self.fk.cvel)[:, self.body_ids]
        com = wp.to_torch(self.fk.subtree_com)[:, self.root_ids]
        vel = cv[..., 3:] + torch.cross(cv[..., :3], pos - com, dim=-1)
        return dict(
            position=pos,
            quaternion=quat,
            velocity=vel,
            angular=cv[..., :3],
            joint_position=wp.to_torch(self.data.qpos)[:, self.qi],
            joint_velocity=wp.to_torch(self.data.qvel)[:, self.vi],
            joint_acceleration=self.joint_acceleration,
        )
