"""Sandbag task on competition physics; policy and reward algorithms are shared."""

import logging
import re
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from holoboxing.data.robot import robot_file
from holoboxing.evaluation.runtime.fall_rules import (
    FALL_BODY_NAMES,
    FALL_HEIGHT_M,
)
from holoboxing.evaluation.runtime.hand_hit_rules import scoring_config
from holoboxing.training.envs.control import (
    CommonLower,
    FrozenHoloMotionV2Controller,
)
from holoboxing.training.envs.mujoco_physics import CompetitionPhysics
from holoboxing.training.envs.mujoco_rewards import terms
from holoboxing.training.envs.nonfinite import (
    clear_invalid_rows,
    nonfinite_rows,
)
from holoboxing.training.envs.onnx_lower import OnnxLower
from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.rsl_rl.modules.amp import amp_frame
from holoboxing.training.tasks.combat.config.sandbag import (
    AGENTS,
    BODY_NAMES,
    JOINT_NAMES,
    POINTS,
    ROLES,
)
from holoboxing.training.tasks.combat.mdp.geometry import (
    matrix,
    raw_self,
    reframe_c,
    rotate,
)
from holoboxing.training.tasks.combat.mdp.observations import (
    ActorObservationNoise,
)
from holoboxing.training.tasks.combat.mdp.rewards import reset_punch_state
from holomotion.src.motion_tracking.link_observation import quat_multiply_wxyz


NOMINAL = [
    -0.312,
    -0.312,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.669,
    0.669,
    0.2,
    0.2,
    -0.363,
    -0.363,
    0.2,
    -0.2,
    0,
    0,
    0,
    0,
    0.6,
    0.6,
    0,
    0,
    0,
    0,
    0,
    0,
]
CRITIC_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

logger = logging.getLogger(__name__)


def action_scale(device):
    frequency = 10 * 2 * 3.1415926535
    result = []
    for n in JOINT_NAMES:
        inertia, effort = 0.003609725, 25.0
        if "hip_roll" in n or "knee" in n:
            inertia, effort = 0.025101925, 139.0
        elif "hip" in n or n == "waist_yaw_joint":
            inertia, effort = 0.010177520, 88.0
        elif "ankle" in n or n in ("waist_pitch_joint", "waist_roll_joint"):
            inertia, effort = 2 * 0.003609725, 50.0
        elif "wrist_pitch" in n or "wrist_yaw" in n:
            inertia, effort = 0.00425, 5.0
        result.append(0.25 * effort / (inertia * frequency**2))
    return torch.tensor(result, device=device)


class Scene(dict):
    pass


class Robot:
    body_names, joint_names = BODY_NAMES, JOINT_NAMES

    def __init__(self):
        self.data = SimpleNamespace()

    def find_joints(self, names, preserve_order=False):
        return self._find(names, JOINT_NAMES, preserve_order)

    def find_bodies(self, names, preserve_order=False):
        return self._find(names, BODY_NAMES, preserve_order)

    @staticmethod
    def _find(patterns, names, ordered):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [
            i
            for pattern in patterns
            for i, name in enumerate(names)
            if re.fullmatch(pattern, name)
        ]
        ids = list(dict.fromkeys(ids)) if ordered else sorted(set(ids))
        return ids, [names[i] for i in ids]


class MujocoSandbagEnv:
    actor_dim, action_dim = 1097, 225

    def __init__(self, assets, paths, worlds, device, directory, seed=42):
        self.device, self.num_envs = torch.device(device), worlds
        self.physics = CompetitionPhysics(assets, worlds, device)
        self.scene = Scene({a: Robot() for a in AGENTS})
        self.scene.env_origins = torch.zeros((worlds, 3), device=device)
        self.scene.sensors = {}
        self.nominal = torch.tensor(NOMINAL, device=device)
        self.step_dt = 0.02
        self.max_episode_length = int(
            np.ceil(scoring_config()["round_duration_s"] / self.step_dt)
        )
        # Use the evaluator's CPU frame timestamps even on CUDA. Reciprocal
        # multiplication can otherwise move a strict KO boundary by one frame.
        self.frame_times = torch.tensor(
            [frame / 50 for frame in range(self.max_episode_length + 1)],
            dtype=torch.float64,
            device=device,
        )
        # Keep the runner's initial episode-length randomization unchanged.
        self.env = self
        self._sim_step_counter = self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            worlds, dtype=torch.long, device=device
        )
        self.fall_started = torch.full(
            (worlds, 2), -1, dtype=torch.long, device=device
        )
        self.continuous_time_buffers = {
            a: torch.zeros(worlds, device=device) for a in AGENTS
        }
        self.actor_noise = ActorObservationNoise(device, seed)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        com_offsets = self.uniform((worlds, 2, 3), -1.0, 1.0) * torch.tensor(
            [0.025, 0.05, 0.05], device=device
        )
        self.physics.randomize_torso_com(com_offsets)
        self.previous_p = self.previous_q = self.history = None
        self.action_managers, self.controllers = {}, {}
        self.reward_terms, self.metrics = {}, {}
        session = OnnxLower(paths["lower_onnx"], worlds, device, directory)
        names = session.metadata["joint_names"].split(",")
        order = [names.index(n) for n in JOINT_NAMES]
        gains = [
            torch.tensor(
                [float(v) for v in session.metadata[k].split(",")],
                device=device,
            )[order]
            for k in ("joint_stiffness", "joint_damping")
        ]
        self.physics.set_gains(*gains)
        scale = action_scale(device)
        for role, agent in enumerate(AGENTS):
            with robot_file(
                assets.resources.scene, ("red", "blue")[role]
            ) as xml:
                self.controllers[agent] = FrozenHoloMotionV2Controller(
                    config_path=paths["lower_config"],
                    lower_session=session,
                    robot_file_path=xml,
                    batch_size=worlds,
                    scene_joint_names=JOINT_NAMES,
                    device=device,
                )
            term = SimpleNamespace(
                _offset=self.nominal.expand(worlds, -1).clone()
                + self.uniform((worlds, 29), -0.01, 0.01),
                _scale=scale,
            )
            manager = SimpleNamespace(
                action=torch.zeros((worlds, 29), device=device),
                prev_action=torch.zeros((worlds, 29), device=device),
            )
            manager.get_term = lambda name, term=term: term
            self.action_managers[agent] = manager
            limits = np.asarray(
                [
                    self.physics.host.joint(("p1_", "p2_")[role] + n).range
                    for n in JOINT_NAMES
                ]
            )
            midpoint, half = (
                limits.mean(-1),
                (limits[:, 1] - limits[:, 0]) * 0.5 * 0.9,
            )
            self.scene[agent].data.soft_joint_pos_limits = torch.tensor(
                np.stack((midpoint - half, midpoint + half), -1),
                device=device,
                dtype=torch.float32,
            ).expand(worlds, -1, -1)
            for suffix in ("", "_torso"):
                self.scene.sensors[f"contact_sensor_{agent}{suffix}"] = (
                    SimpleNamespace(
                        body_names=BODY_NAMES, data=SimpleNamespace()
                    )
                )
        self.lower = CommonLower(self, self.controllers, AGENTS)
        self.next_push = self.uniform((worlds, 2), 1.0, 3.0)
        self.next_contact = torch.full((worlds, 2), 0.5, device=device)

    def uniform(self, shape, lo, hi):
        return (
            torch.rand(shape, device=self.device, generator=self.generator)
            * (hi - lo)
            + lo
        )

    def get_opponent_id(self, agent):
        return AGENTS[1 - AGENTS.index(agent)]

    def get_opponent(self, agent):
        return self.scene[self.get_opponent_id(agent)]

    def _state(self):
        state = self.physics.state()
        for role, agent in enumerate(AGENTS):
            data = self.scene[agent].data
            p, q = state["position"][:, role], state["quaternion"][:, role]
            v, omega = state["velocity"][:, role], state["angular"][:, role]
            ipos = wp.to_torch(self.physics.model.body_ipos)
            if ipos.ndim == 2:
                ipos = ipos[None]
            local_com = ipos[:, self.physics.body_ids[role]]
            com_velocity = v + torch.cross(
                omega, rotate(matrix(q), local_com), dim=-1
            )
            data.body_pos_w, data.body_quat_w = p, q
            data.body_lin_vel_w, data.body_ang_vel_w = com_velocity, omega
            data.body_vel_w = torch.cat((com_velocity, omega), -1)
            data.root_pos_w, data.root_quat_w = p[:, 0], q[:, 0]
            data.root_lin_vel_w = com_velocity[:, 0]
            data.root_ang_vel_b = rotate(
                matrix(q[:, 0]).transpose(-1, -2), omega[:, 0]
            )
            data.joint_pos, data.joint_vel = (
                state["joint_position"][:, role],
                state["joint_velocity"][:, role],
            )
            data.joint_acc = state["joint_acceleration"][:, role]
            data.default_joint_pos_nominal = self.nominal
            data.default_joint_pos = (
                self.action_managers[agent].get_term("joint_pos")._offset
            )
            sensor = self.scene.sensors[f"contact_sensor_{agent}"].data
            sensor.net_forces_w_history = self.physics.net_history[:, :, role]
            sensor.net_forces_w = sensor.net_forces_w_history[:, 0]
            torso = self.scene.sensors[f"contact_sensor_{agent}_torso"].data
            torso.force_matrix_w_history = self.physics.paired_history[
                :, :, role, None
            ]
            torso.force_matrix_w = torso.force_matrix_w_history[:, 0]
        return state

    def _critic(self, agent, worlds=None):
        data = self.scene[agent].data
        count = self.num_envs if worlds is None else len(worlds)
        worlds = slice(None) if worlds is None else worlds
        body_pos, body_quat = data.body_pos_w[worlds], data.body_quat_w[worlds]
        ids = [BODY_NAMES.index(n) for n in CRITIC_BODIES]
        torso = BODY_NAMES.index("torso_link")
        inverse = matrix(body_quat[:, torso]).transpose(-1, -2)
        pos = rotate(
            inverse[:, None],
            body_pos[:, ids] - body_pos[:, torso, None],
        )
        orientation = inverse[:, None] @ matrix(body_quat[:, ids])
        root_inv = matrix(data.root_quat_w[worlds]).transpose(-1, -2)
        fists = [BODY_NAMES.index(n) for n in POINTS[1:3]]
        target = rotate(
            root_inv[:, None],
            body_pos[:, fists]
            - self.get_opponent(agent).data.body_pos_w[worlds, torso, None],
        ).flatten(1)
        return torch.cat(
            (
                pos.flatten(1),
                orientation[..., :2].flatten(1),
                rotate(root_inv, data.root_lin_vel_w[worlds]),
                data.root_ang_vel_b[worlds]
                + self.uniform((count, 3), -0.2, 0.2),
                data.joint_pos[worlds]
                - data.default_joint_pos[worlds]
                + self.uniform((count, 29), -0.01, 0.01),
                data.joint_vel[worlds] + self.uniform((count, 29), -0.5, 0.5),
                self.action_managers[agent].action[worlds],
                target + self.uniform((count, 6), -0.2, 0.2),
            ),
            -1,
        )

    def _read(self, reset, previous=None):
        """Read a frame, or refresh only reset worlds in an already-read frame."""
        measured_before = self.measured if previous is not None else None
        state = self._state()
        ids = [BODY_NAMES.index(n) for n in POINTS]
        p = state["position"][:, :, ids].flatten(0, 1)
        q = state["quaternion"][:, :, ids].flatten(0, 1)
        v = state["velocity"][:, :, ids].flatten(0, 1)
        omega = torch.stack(
            [self.scene[a].data.root_ang_vel_b for a in AGENTS], 1
        ).flatten(0, 1)
        mask = reset.repeat_interleave(2)
        pp = (
            p
            if self.previous_p is None
            else torch.where(mask[:, None, None], p, self.previous_p)
        )
        pq = (
            q
            if self.previous_q is None
            else torch.where(mask[:, None, None], q, self.previous_q)
        )
        raw = raw_self(p, q, v, omega, pp, pq)
        repeated = raw[:, None].expand(-1, 10, -1)
        self.history = (
            repeated.clone()
            if self.history is None
            else torch.where(
                mask[:, None, None],
                repeated,
                self.history if previous is not None
                else torch.cat((self.history[:, 1:], raw[:, None]), 1),
            )
        )
        self.measured = reframe_c(
            self.history,
            p,
            q,
            p.reshape(self.num_envs, 2, 5, 3).flip(1).flatten(0, 1),
        )
        actors = []
        torso, fists = BODY_NAMES.index("torso_link"), ids[1:3]
        for agent in AGENTS:
            data, other = self.scene[agent].data, self.get_opponent(agent).data
            inv = matrix(data.root_quat_w).transpose(-1, -2)
            actors.append(
                torch.cat(
                    (
                        data.root_ang_vel_b,
                        data.joint_pos - data.default_joint_pos,
                        data.joint_vel,
                        self.action_managers[agent].action,
                        rotate(
                            inv[:, None],
                            data.body_pos_w - data.root_pos_w[:, None],
                        ).flatten(1),
                        rotate(
                            inv[:, None],
                            data.body_pos_w[:, fists]
                            - other.body_pos_w[:, torso, None],
                        ).flatten(1),
                        rotate(
                            inv[:, None],
                            other.body_pos_w[:, fists]
                            - data.body_pos_w[:, torso, None],
                        ).flatten(1),
                    ),
                    -1,
                )
            )
        self.measured.update(
            actor=torch.stack(actors, 1).flatten(0, 1),
            position=p,
            quaternion=q,
            link_velocity_w=v,
            omega_b=omega,
        )
        if measured_before is not None:
            # A local reset must not replace healthy lower-controller inputs.
            for name, value in self.measured.items():
                measured_before[name][mask] = value[mask]
            self.measured = measured_before
            p, q = self.measured["position"], self.measured["quaternion"]
        noisy = self.actor_noise.observe(
            self.measured, self.common_step_counter, reset,
            refresh=previous is not None,
        )
        packed = torch.cat(
            (
                noisy["history"],
                noisy["task"][:, -15:],
                torch.zeros((2 * self.num_envs, 2), device=self.device),
            ),
            -1,
        )
        self.previous_p, self.previous_q = p.clone(), q.clone()
        if previous is not None:
            worlds = reset.nonzero().flatten()
            for i, agent in enumerate(AGENTS):
                previous[agent]["task_policy"][worlds] = packed[i::2][worlds]
                previous[agent]["critic"][worlds] = self._critic(agent, worlds)
            return previous
        return {
            a: {"task_policy": packed[i::2].clone(), "critic": self._critic(a)}
            for i, a in enumerate(AGENTS)
        }

    def _reset(self, ids):
        self.physics.reset(ids)
        qpos = wp.to_torch(self.physics.data.qpos)
        extent = torch.tensor(
            [0.05, 0.05, 0.01, 0.1, 0.1, 0.2], device=self.device
        )
        for role, agent in enumerate(AGENTS):
            qi = self.physics.root_qi[role]
            noise = self.uniform((len(ids), 6), -1.0, 1.0) * extent
            qpos[ids, qi : qi + 3] += noise[:, :3]
            r, p, y = (noise[:, 3:] * 0.5).unbind(-1)
            cr, cp, cy, sr, sp, sy = (
                r.cos(),
                p.cos(),
                y.cos(),
                r.sin(),
                p.sin(),
                y.sin(),
            )
            delta = torch.stack(
                (
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ),
                -1,
            )
            qpos[ids, qi + 3 : qi + 7] = quat_multiply_wxyz(
                delta, qpos[ids, qi + 3 : qi + 7]
            )
            joints = qpos[ids[:, None], self.physics.qi[role]] + self.uniform(
                (len(ids), 29), -0.1, 0.1
            )
            limits = self.scene[agent].data.soft_joint_pos_limits[ids]
            qpos[ids[:, None], self.physics.qi[role]] = joints.clamp(
                limits[..., 0], limits[..., 1]
            )
            self.action_managers[agent].action[ids] = 0
            self.action_managers[agent].prev_action[ids] = 0
            self.controllers[agent].reset(ids)
            self.continuous_time_buffers[agent][ids] = 0
        reset_punch_state(self, ids)
        self.episode_length_buf[ids] = 0
        self.fall_started[ids] = -1
        self.next_push[ids] = self.uniform((len(ids), 2), 1.0, 3.0)
        self.next_contact[ids] = 0.5

    def reset(self):
        self.common_step_counter = 0
        self.previous_p = self.previous_q = self.history = None
        self.actor_noise.reset_history()
        self._reset(torch.arange(self.num_envs, device=self.device))
        return self._read(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def _amp(self, agent):
        data = self.scene[agent].data
        return amp_frame(
            data.root_ang_vel_b,
            data.joint_pos,
            data.joint_vel,
            self.nominal,
            torch.zeros_like(self.nominal),
        ).clone()

    def _update_falls(self):
        # The official referee reads solver xpos, not the refreshed policy FK.
        body_ids = self.physics.body_ids[
            :, [BODY_NAMES.index(name) for name in FALL_BODY_NAMES]
        ]
        heights = wp.to_torch(self.physics.data.xpos)[:, body_ids, 2]
        low = (
            heights.double() < self.physics.floor_height + FALL_HEIGHT_M
        ).any(-1)
        step = self.episode_length_buf[:, None]
        self.fall_started = torch.where(
            low,
            torch.where(self.fall_started < 0, step, self.fall_started),
            -1,
        )
        fallen = low & (
            (
                self.frame_times[step]
                - self.frame_times[self.fall_started.clamp_min(0)]
            )
            >= scoring_config()["ko_confirmation_s"]
        )
        return low, fallen

    def _nonfinite_worlds(self, values):
        invalid = nonfinite_rows(values, self.num_envs, self.device)
        if bool(invalid.any()):
            try:
                require_finite(values, "local environment transition")
            except FloatingPointError as error:
                logger.warning(
                    "Resetting %d nonfinite worlds at step %d; first ids=%s; %s",
                    int(invalid.sum()),
                    self.common_step_counter,
                    invalid.nonzero().flatten()[:16].tolist(),
                    error,
                )
        return invalid

    def step(self, actions):
        physical = torch.stack([actions[a] for a in AGENTS], 1).flatten(0, 1)
        start = {a: self._amp(a) for a in AGENTS}
        joint_actions, outputs = self.lower.autonomous(self.measured, physical)
        require_finite((joint_actions, outputs), "lower controller output")
        for role, agent in enumerate(AGENTS):
            manager = self.action_managers[agent]
            manager.prev_action.copy_(manager.action)
            manager.action.copy_(joint_actions[agent])
            self.physics.targets[:, role] = outputs["joint_target"][agent]
        for _ in range(4):
            self.physics.substep()
        self._sim_step_counter += 4
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._state()
        low, fallen = self._update_falls()
        terminated = {}
        for role, agent in enumerate(AGENTS):
            buffer = self.continuous_time_buffers[agent]
            terminated[agent] = fallen[:, role] | (buffer >= 5.0)
        terminal = torch.stack(list(terminated.values())).any(0)
        timeout = self.episode_length_buf >= self.max_episode_length
        done = terminal | timeout
        rewards, infos = {}, {}
        for role, agent in zip(ROLES, AGENTS):
            self.reward_terms[agent] = {
                name: weight * fn(self, **params) * 0.02
                for name, weight, fn, params in terms(agent)
            }
            rewards[agent] = sum(self.reward_terms[agent].values())
            infos[agent] = {
                "time_outs": timeout & ~terminal,
                "terminated": terminated[agent].clone(),
            }
            infos[f"amp_obs_{role}"] = torch.cat(
                (start[agent], self._amp(agent)), -1
            )
            infos[f"terminal_critic_{role}"] = self._critic(agent)
            self.metrics[agent] = {"fallen": low[:, AGENTS.index(agent)]}
        invalid = self._nonfinite_worlds(
            {
                "physics": {
                    name: wp.to_torch(getattr(self.physics.data, name))
                    for name in (
                        "qpos", "qvel", "qacc", "qacc_warmstart", "ctrl"
                    )
                },
                "targets": self.physics.targets,
                "joint_acceleration": self.physics.joint_acceleration,
                "net_history": self.physics.net_history,
                "paired_history": self.physics.paired_history,
                "rewards": rewards,
                "reward_terms": self.reward_terms,
                "infos": infos,
            }
        )
        done = done | invalid
        ids = done.nonzero().flatten()
        if len(ids):
            self._reset(ids)
        for role, agent in enumerate(AGENTS):
            self.next_contact[:, role] -= 0.02
            due = self.next_contact[:, role] <= 1e-6
            force = self.physics.paired_history[:, :, 1 - role].norm(dim=-1)
            hit = (force.sum(-1).amax(1) > 1.0) & (force.amax((1, 2)) > 1.0)
            buffer = self.continuous_time_buffers[agent]
            buffer[due] = torch.where(hit[due], buffer[due] + 0.5, 0.0)
            self.next_contact[due, role] = 0.5
            self.next_push[:, role] -= 0.02
            pushed = (self.next_push[:, role] <= 0).nonzero().flatten()
            if len(pushed):
                extent = torch.tensor(
                    [0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device
                )
                delta_velocity_w = (
                    self.uniform((len(pushed), 6), -1.0, 1.0) * extent
                )
                self.physics.push(pushed, role, delta_velocity_w)
                self.next_push[pushed, role] = self.uniform(
                    (len(pushed),), 1.0, 3.0
                )
        observations = self._read(done)
        bad_observation = self._nonfinite_worlds(observations)
        if bool(bad_observation.any()):
            self._reset(bad_observation.nonzero().flatten())
            invalid |= bad_observation
            done |= bad_observation
            observations = self._read(bad_observation, previous=observations)
        # A failed reset is not recoverable; never pass bad data to normalizers.
        require_finite(observations, "post-reset environment observation")
        if bool(invalid.any()):
            clear_invalid_rows(
                (rewards, infos, self.reward_terms, self.metrics), invalid
            )
        infos["invalid_envs"] = invalid
        return observations, rewards, done, infos

    def close(self):
        wp.synchronize_device(str(self.device))

    def diagnostic_state(self):
        """Failure-only evidence; post-step/reset state is not a replay checkpoint."""
        physics = self.physics
        data = physics.data
        ncon = min(int(wp.to_torch(data.nacon)[0]), data.naconmax)
        nefc = wp.to_torch(data.nefc)
        width = min(int(nefc.max()), data.njmax)
        return {
            "phase": "exception_time_may_follow_episode_reset",
            "physics_steps": physics.steps,
            "state": {
                name: wp.to_torch(getattr(data, name)).detach().cpu().clone()
                for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "time")
            },
            "contacts": {
                name: wp.to_torch(getattr(data.contact, name))[:ncon].detach().cpu().clone()
                for name in ("worldid", "geom", "dim", "dist", "frame", "pos", "efc_address", "friction")
            },
            "constraint_count": nefc.cpu().clone(),
            "constraint_force": wp.to_torch(data.efc.force)[:, :width].cpu().clone(),
            "body_ipos": wp.to_torch(physics.model.body_ipos).cpu().clone(),
            "targets": physics.targets.cpu().clone(),
            "joint_acceleration": physics.joint_acceleration.cpu().clone(),
            "net_history": physics.net_history.cpu().clone(),
            "paired_history": physics.paired_history.cpu().clone(),
        }
