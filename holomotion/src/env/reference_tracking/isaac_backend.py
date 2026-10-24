"""Isaac PhysX execution of the fixed common motion-tracking task.

Each world has its own articulation, infinite floor, friction and torso COM.
The backend records physical next states before reset and executes direct PD
setpoints without the v1 controller's additional delay or randomization.
"""

from pathlib import Path

import torch

from holomotion.src.env.reference_tracking.mdp import (
    MAX_STEP_REWARD,
    flatten_state,
)
from holomotion.src.env.reference_tracking.robot import (
    G1_JOINT_NAMES,
    action_scale,
    default_joint_pos_list,
)
from holomotion.src.env.reference_tracking.rotations import (
    mat_to_quat,
    quat_to_mat,
    so3_exp,
)


class IsaacTrackingBackend:
    _active = None

    def __init__(
        self,
        model,
        motion,
        mdp,
        num_envs,
        *,
        device="cuda:0",
        gamma=0.99,
        evaluation=False,
        max_episode_steps=500,
        nconmax=128,
        njmax=768,
        capture_graph=True,
        robot_urdf=None,
        cache_dir=None,
    ):
        if num_envs < 1 or max_episode_steps < 1 or not 0 <= gamma < 1:
            raise ValueError("Invalid physical rollout dimensions")
        if robot_urdf is None or cache_dir is None:
            raise ValueError(
                "Isaac physics requires an explicit robot URDF and private cache directory"
            )
        self.robot_urdf = Path(robot_urdf).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.cpu_model, self.motion, self.mdp = model, motion, mdp
        self.N, self.device, self.gamma = num_envs, device, gamma
        self.eval_mode = evaluation
        self.max_episode_steps = 10**9 if evaluation else max_episode_steps
        self.reset_noise = not evaluation
        if self.__class__._active is not None:
            self.__class__._active.close()
        self.default_q = torch.tensor(default_joint_pos_list(), device=device)
        self.alpha = torch.tensor(
            [action_scale(name) for name in G1_JOINT_NAMES], device=device
        )
        self._torso_id = model.body("torso_link").id
        self._ground_id = model.geom("floor").id
        self.qpos = torch.zeros(num_envs, 36, device=device)
        self.qvel = torch.zeros(num_envs, 35, device=device)
        self.ctrl = torch.zeros(num_envs, 29, device=device)
        self._friction = torch.zeros(num_envs, model.ngeom, 3, device=device)
        self._com = torch.tensor(
            model.body_ipos.copy(), device=device, dtype=torch.float32
        )[None].repeat(num_envs, 1, 1)
        self._nominal_com = self._com[:, self._torso_id].clone()
        # Preserve actor and critic initialization streams during scene setup.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self._setup_scene(model, num_envs, device, evaluation)
        self.__class__._active = self
        self.xi = torch.zeros(num_envs, 33, device=device)
        if not evaluation:
            self.xi[:, 30:33] = torch.rand(
                num_envs, 3, device=device
            ) * torch.tensor([0.05, 0.1, 0.1], device=device) - torch.tensor(
                [0.025, 0.05, 0.05], device=device
            )
        self._com[:, self._torso_id] = self._nominal_com + self.xi[:, 30:33]
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 29, device=device)
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1 / motion.num_bins)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self._overflow_seen = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )
        self.capacity_mask = 0
        self._apply_com()
        self.reset(torch.arange(num_envs, device=device))

    def _advance(self):
        self.robot.set_joint_position_target(
            self.ctrl, joint_ids=self.joint_ids
        )
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(0.005)
        self._read_physical_state()
        if self.geometry_audit is None:
            self._audit_geometry()

    @staticmethod
    def _physical_state_valid(state):
        # These are corruption checks, far beyond this bounded-joint robot's
        # normal range. Reject evidence rather than clipping or resetting it.
        return (
            all(torch.isfinite(value).all() for value in state)
            and bool(state[4].abs().max() <= 100.0)
            and bool(state[5].abs().max() <= 10000.0)
        )

    def state(self):
        p, quat = self.qpos[:, :3].clone(), self.qpos[:, 3:7].clone()
        rotation = quat_to_mat(quat)
        # Isaac already exposes world-frame angular velocity. Preserve it
        # directly rather than introducing a matrix round trip.
        angular = self.qvel[:, 3:6].clone()
        return (
            p,
            rotation,
            self.qvel[:, :3].clone(),
            angular,
            self.qpos[:, 7:].clone(),
            self.qvel[:, 6:].clone(),
        )

    def sample_frames(self, n):
        motion = self.motion
        soft = torch.softmax(-25 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probabilities = (
            0.5 / motion.num_bins
            + 0.4 * soft
            + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        )
        bins = torch.multinomial(probabilities, n, replacement=True)
        start = motion.bin_start[bins]
        end = torch.minimum(start + 50, motion.clip_end[motion.bin_clip[bins]])
        frame = (
            start
            + (
                torch.rand(n, device=self.device) * (end - start).float()
            ).long()
        )
        return frame, bins

    def reset(self, ids, frames=None):
        if not len(ids):
            return
        if frames is None:
            frames, bins = self.sample_frames(len(ids))
        else:
            bins = self.motion.frame_bin[frames]
        p, rotation, v, w, q, qd = (
            value.clone() for value in self.motion.ref_state(frames)
        )
        if self.reset_noise:

            def uniform(half):
                return (
                    2 * torch.rand(len(ids), len(half), device=self.device) - 1
                ) * torch.tensor(half, device=self.device)

            p += uniform([0.05, 0.05, 0.01])
            rotation = so3_exp(uniform([0.1, 0.1, 0.2])) @ rotation
            v += uniform([0.5, 0.5, 0.2])
            w += uniform([0.52, 0.52, 0.78])
            q += uniform([0.1] * 29)
        q = q.clamp(self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.qpos[ids] = torch.cat((p, mat_to_quat(rotation), q), -1)
        self.qvel[ids] = torch.cat(
            (
                v,
                w,
                qd,
            ),
            -1,
        )
        self.ctrl[ids] = q
        self.f[ids], self.ep_step[ids], self.a_prev[ids] = frames, 0, 0.0
        self.start_bin[ids] = bins
        self.ep_disc_ret[ids], self.ep_ret[ids] = 0.0, 0.0
        if self.eval_mode:
            self.xi[ids, :30] = 0.0
            self.xi[ids, 0] = 1.0
        else:
            self.xi[ids, 0] = torch.empty(
                len(ids), device=self.device
            ).uniform_(0.3, 1.2)
            self.xi[ids, 1:30] = torch.empty(
                len(ids), 29, device=self.device
            ).uniform_(-0.01, 0.01)
        self._friction[ids, :, 0] = self.xi[ids, :1]
        self._friction[ids, self._ground_id, 0] = (
            1.0 if self.eval_mode else 0.3
        )
        self._write_physical_state(ids)
        self._apply_friction(ids)

    @torch.no_grad()
    def step(self, action):
        if self.eval_mode:
            # Worlds are independent. Completed trials contribute no further
            # metrics and stay at a neutral, finite preparation state.
            ids = self.done_eval.nonzero().flatten()
            self.qpos[ids, :3] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device
            )
            self.qpos[ids, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            )
            self.qpos[ids, 7:] = self.default_q
            self.qvel[ids] = 0.0
            self._write_physical_state(ids)
            action = torch.where(self.done_eval[:, None], 0.0, action)
        state, frame = self.state(), self.f.clone()
        previous, context, step = (
            self.a_prev.clone(),
            self.xi.clone(),
            self.ep_step.clone(),
        )
        self.ctrl.copy_(
            self.default_q + context[:, 1:30] + self.alpha * action
        )
        self._advance()
        next_state = self.state()
        if not self._physical_state_valid(next_state):
            cache = self.cache_dir
            torch.save(
                dict(
                    state=tuple(value.cpu() for value in next_state),
                    action=action.cpu(),
                    frames=frame.cpu(),
                    context=context.cpu(),
                    origins=self.origins.cpu(),
                ),
                cache.parent / "physics_state_failure.pt",
            )
            raise FloatingPointError(
                "Invalid physical state or gross joint divergence; "
                "rollout rejected"
            )
        # Finished evaluation worlds are kept at their last reference frame.
        next_frame = torch.minimum(
            frame + 1, self.motion.clip_end[self.motion.clip_id[frame]]
        )
        reward, terms = self.mdp.reward(
            next_state, next_frame, action, previous, terms=True
        )
        terminated = self.mdp.termination(next_state, next_frame)
        timeout = (
            next_frame >= self.motion.clip_end[self.motion.clip_id[frame]]
        ) | (step + 1 >= self.max_episode_steps)
        done = terminated | timeout
        out = dict(
            s=flatten_state(state),
            a=action.clone(),
            a_prev=previous,
            xi=context,
            f=frame,
            s1=flatten_state(next_state),
            r=reward,
            term=terminated,
            done=done,
            ep_step=step,
            terms=terms,
        )
        self.ep_disc_ret += self.gamma ** step.float() * reward
        self.ep_ret += reward
        self.f, self.ep_step, self.a_prev = (
            next_frame,
            step + 1,
            action.clone(),
        )
        if self.eval_mode:
            self.failed_eval |= terminated & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero().flatten()
        if len(ids):
            length = self.ep_step[ids].float()
            bootstrap = torch.where(
                terminated[ids],
                0.0,
                self.gamma**length
                * self.ep_ret[ids]
                / length
                / (1 - self.gamma),
            )
            returns = self.ep_disc_ret[ids] + bootstrap
            bins = self.start_bin[ids]
            # Parallel episodes can update the same bin. Aggregate first so
            # duplicate scatter writes cannot silently discard other returns.
            totals, counts = (
                torch.zeros_like(self.bin_G),
                torch.zeros_like(self.bin_G),
            )
            totals.scatter_add_(0, bins, returns)
            counts.scatter_add_(0, bins, torch.ones_like(returns))
            used = counts > 0
            self.bin_G[used] = (
                0.8 * self.bin_G[used] + 0.2 * totals[used] / counts[used]
            )
            out["ep_len"], out["ep_ret_norm"] = (
                length,
                self.ep_ret[ids]
                / (
                    length
                    * getattr(self.mdp, "max_step_reward", MAX_STEP_REWARD)
                ),
            )
            self.reset(ids)
        return out

    def status_summary(self, *, include_world_bits=True):
        result = dict(
            capacity_overflow_worlds_seen=int(
                ((self._overflow_seen & self.capacity_mask) != 0).sum()
            ),
            solver_limit_worlds_seen=int(
                ((self._overflow_seen & ~self.capacity_mask) != 0).sum()
            ),
        )
        if include_world_bits:
            result.update(
                physics_status_bits=self._overflow_seen.tolist(),
                capacity_overflow_bits=(
                    self._overflow_seen & self.capacity_mask
                ).tolist(),
            )
        result.update(
            schema_audit=self.schema_audit,
            geometry_audit=self.geometry_audit,
            capacity_overflow_worlds_seen=None,
            solver_limit_worlds_seen=None,
            physics_status_bits=None,
            capacity_overflow_bits=None,
            engine="isaac_physx",
            environment_spacing_m=2.5,
            maximum_origin_coordinate_m=float(self.origins.abs().max()),
            capacity_overflow_detection=(
                "PhysX errors and finite-state checks; "
                "no MuJoCo overflow bitset exists"
            ),
        )
        return result

    def _setup_scene(self, model, count, device, evaluation):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils import configclass
        from isaacsim.core.utils.stage import create_new_stage

        from holomotion.src.env.reference_tracking.robot import (
            _kd,
            _kp,
            motor_of,
        )

        create_new_stage()
        self.sim = SimulationContext(
            SimulationCfg(
                dt=0.005,
                device=device,
                gravity=(0.0, 0.0, -9.81),
                render_interval=4,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0 if evaluation else 0.3,
                    dynamic_friction=1.0 if evaluation else 0.3,
                    restitution=0.0,
                    friction_combine_mode="max",
                    restitution_combine_mode="max",
                ),
            )
        )
        urdf, directory = self.robot_urdf, self.cache_dir
        motor_configs = {}
        for name in G1_JOINT_NAMES:
            effort, _, armature = motor_of(name)
            motor_configs[name] = ImplicitActuatorCfg(
                joint_names_expr=[name],
                effort_limit_sim=effort,
                velocity_limit_sim=motor_of(name)[1],
                stiffness=_kp(armature),
                damping=_kd(armature),
                armature=armature,
                friction=0.0,
            )
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            usd_dir=str(directory),
            force_usd_conversion=False,
            fix_base=False,
            merge_fixed_joints=True,
            root_link_name="pelvis",
            replace_cylinders_with_capsules=False,
            self_collision=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                )
            ),
        )
        floor_material = self.sim.cfg.physics_material

        def spawn_ground(prim_path, cfg, **kwargs):
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import Gf, PhysicsSchemaTools

            stage = get_current_stage()
            PhysicsSchemaTools.addGroundPlane(
                stage, prim_path, "Z", 200000.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5)
            )
            material_path = prim_path + "/PhysicsMaterial"
            floor_material.func(material_path, floor_material)
            sim_utils.bind_physics_material(prim_path + "/geom", material_path)
            return stage.GetPrimAtPath(prim_path)

        @configclass
        class Scene(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/Ground",
                spawn=sim_utils.SpawnerCfg(func=spawn_ground),
            )
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=spawn,
                actuators=motor_configs,
                soft_joint_pos_limit_factor=0.9,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0)
                ),
            )

        self.scene = InteractiveScene(
            # Worlds have explicit collision isolation. Keep their numerical
            # origins compact: an 8192-world 5 m grid spans 450 m and
            # float32 link transforms exceed the 50 micrometre FK check.
            Scene(num_envs=count, env_spacing=2.5, replicate_physics=True)
        )
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.robot = self.scene["robot"]
        self.sim.reset()
        self.scene.update(0.005)
        self.joint_ids = [
            self.robot.joint_names.index(name) for name in G1_JOINT_NAMES
        ]
        self.torso_body_id = self.robot.body_names.index("torso_link")
        self.origins = self.scene.env_origins
        self.materials = (
            self.robot.root_physx_view.get_material_properties().clone()
        )
        self.nominal_coms = self.robot.root_physx_view.get_coms().clone()
        self._closed = False
        self.geometry_audit = None
        self._audit_schema(model)

    def _apply_com(self):
        value = self.nominal_coms.clone()
        value[:, self.torso_body_id, :3] += self.xi[:, 30:33].cpu()
        self.robot.root_physx_view.set_coms(value, torch.arange(self.N))

    def _apply_friction(self, ids):
        cpu_ids = ids.cpu()
        coefficient = self.xi[ids, 0].cpu()
        self.materials[cpu_ids, :, :2] = coefficient[:, None, None]
        self.materials[cpu_ids, :, 2] = 0.0
        self.robot.root_physx_view.set_material_properties(
            self.materials, cpu_ids
        )

    def _write_physical_state(self, ids):
        if not len(ids):
            return
        root = torch.cat(
            (
                self.qpos[ids, :7].clone(),
                self.qvel[ids, :3],
                self.qvel[ids, 3:6],
            ),
            -1,
        )
        root[:, :3] += self.origins[ids]
        self.robot.write_root_link_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(
            self.qpos[ids, 7:],
            self.qvel[ids, 6:],
            joint_ids=self.joint_ids,
            env_ids=ids,
        )
        self.robot.reset(ids)

    def _read_physical_state(self):
        data = self.robot.data
        self.qpos[:, :3] = data.root_link_pos_w - self.origins
        self.qpos[:, 3:7] = data.root_link_quat_w
        self.qpos[:, 7:] = data.joint_pos[:, self.joint_ids]
        self.qvel[:, :3] = data.root_link_lin_vel_w
        self.qvel[:, 3:6] = data.root_link_ang_vel_w
        self.qvel[:, 6:] = data.joint_vel[:, self.joint_ids]

    def close(self):
        if not getattr(self, "_closed", True):
            from isaaclab.sim import SimulationContext

            self.sim.stop()
            self.sim.clear_all_callbacks()
            SimulationContext.clear_instance()
            self._closed = True
            self.__class__._active = None

    def _audit_schema(self, model):
        from holomotion.src.env.reference_tracking.robot import (
            _kd,
            _kp,
            motor_of,
        )

        view = self.robot.root_physx_view
        result = {}
        for label, actual, expected in (
            (
                "stiffness",
                view.get_dof_stiffnesses()[0, self.joint_ids],
                [_kp(motor_of(n)[2]) for n in G1_JOINT_NAMES],
            ),
            (
                "damping",
                view.get_dof_dampings()[0, self.joint_ids],
                [_kd(motor_of(n)[2]) for n in G1_JOINT_NAMES],
            ),
            (
                "effort_limit",
                view.get_dof_max_forces()[0, self.joint_ids],
                [motor_of(n)[0] for n in G1_JOINT_NAMES],
            ),
            (
                "velocity_limit",
                view.get_dof_max_velocities()[0, self.joint_ids],
                [motor_of(n)[1] for n in G1_JOINT_NAMES],
            ),
            (
                "armature",
                view.get_dof_armatures()[0, self.joint_ids],
                [motor_of(n)[2] for n in G1_JOINT_NAMES],
            ),
        ):
            expected = torch.tensor(expected, device=actual.device)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            result[label + "_maximum_error"] = float(
                (actual - expected).abs().max()
            )
        limits = view.get_dof_limits()[0, self.joint_ids].cpu()
        expected_limits = torch.tensor(
            model.jnt_range[1:].copy(), dtype=limits.dtype
        )
        torch.testing.assert_close(
            limits, expected_limits, atol=3e-6, rtol=1e-5
        )
        mass = float(view.get_masses()[0].sum())
        if abs(mass - float(model.body_mass.sum())) > 1e-4:
            raise ValueError("Engine import changed robot mass")
        result.update(
            total_mass_kg=mass,
            joint_limit_maximum_error_rad=float(
                (limits - expected_limits).abs().max()
            ),
        )
        self.schema_audit = result

    def _audit_geometry(self):
        from holomotion.src.env.reference_tracking.rotations import quat_to_mat

        ids = [self.robot.body_names.index(n) for n in self.mdp.kin.body_names]
        selected_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            positions, rotations, velocities, angular = self.mdp.kin.forward(
                *self.state(), all_bodies=True
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = selected_tf32
        data = self.robot.data
        actual_pos = data.body_link_pos_w[:, ids] - self.origins[:, None]
        actual_rot = quat_to_mat(data.body_link_quat_w[:, ids])
        actual_vel = data.body_link_lin_vel_w[:, ids]
        actual_ang = data.body_link_ang_vel_w[:, ids]
        result = dict(
            position_m=float((positions - actual_pos).abs().max()),
            rotation_matrix=float((rotations - actual_rot).abs().max()),
            linear_velocity_m_s=float((velocities - actual_vel).abs().max()),
            angular_velocity_rad_s=float((angular - actual_ang).abs().max()),
        )
        if (
            result["position_m"] > 5e-5
            or result["rotation_matrix"] > 5e-5
            or result["linear_velocity_m_s"] > 5e-3
            or result["angular_velocity_rad_s"] > 5e-3
        ):
            cache = self.cache_dir
            torch.save(
                dict(
                    state=tuple(v.detach().cpu() for v in self.state()),
                    positions=positions.cpu(),
                    rotations=rotations.cpu(),
                    actual_positions=actual_pos.cpu(),
                    actual_rotations=actual_rot.cpu(),
                    velocities=velocities.cpu(),
                    actual_velocities=actual_vel.cpu(),
                    angular=angular.cpu(),
                    actual_angular=actual_ang.cpu(),
                    body_names=self.mdp.kin.body_names,
                    joint_names=self.robot.joint_names,
                    joint_ids=self.joint_ids,
                    origins=self.origins.cpu(),
                    context=self.xi.cpu(),
                    frames=self.f.cpu(),
                    errors=result,
                ),
                cache.parent / "physics_geometry_failure.pt",
            )
            raise ValueError(
                f"Physics and reference forward kinematics differ: {result}"
            )
        result.update(
            audit_matmul_precision="ieee_float32",
            tf32_enabled_for_learning=selected_tf32,
        )
        self.geometry_audit = result
