"""Batched MuJoCo Warp physics for the reference-method experiment.

Each world has its own articulation, infinite floor, friction and torso COM.
The backend records physical next states before reset and executes direct PD
setpoints without the v1 controller's additional delay or randomization.
"""

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .mdp import MAX_STEP_REWARD, flatten_state
from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from .rotations import mat_to_quat, quat_to_mat, so3_exp


class WarpTrackingBackend:
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
    ):
        if num_envs < 1 or max_episode_steps < 1 or not 0 <= gamma < 1:
            raise ValueError("Invalid physical rollout dimensions")
        self.cpu_model, self.motion, self.mdp = model, motion, mdp
        self.N, self.device, self.gamma = num_envs, device, gamma
        self.eval_mode = evaluation
        self.max_episode_steps = 10**9 if evaluation else max_episode_steps
        self.reset_noise = not evaluation
        wp.init()
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        self.default_q = torch.tensor(default_joint_pos_list(), device=device)
        self.alpha = torch.tensor(
            [action_scale(name) for name in G1_JOINT_NAMES], device=device
        )
        self._torso_id = model.body("torso_link").id
        self._ground_id = model.geom("floor").id
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(
                model,
                batch_sizes={"geom_friction": num_envs, "body_ipos": num_envs},
            )
            self.model.opt.warn_overflow = 0
            initial = mujoco.MjData(model)
            initial.qpos[:3] = [0, 0, 1.0]
            initial.qpos[7:] = self.default_q.cpu().numpy()
            mujoco.mj_forward(model, initial)
            self.data = mjw.put_data(
                model, initial, nworld=num_envs, nconmax=nconmax, njmax=njmax
            )
        self.qpos, self.qvel, self.ctrl = (
            wp.to_torch(value)
            for value in (self.data.qpos, self.data.qvel, self.data.ctrl)
        )
        self._friction = wp.to_torch(self.model.geom_friction)
        self._com = wp.to_torch(self.model.body_ipos)
        self._nominal_com = self._com[:, self._torso_id].clone()
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
        # MuJoCo Warp also stores ordinary bounded solver-iteration statuses in
        # its overflow bitset. These are distinct from truncated contact or
        # constraint buffers. Retain both; reject only capacity/invalid states.
        solver_limits = (
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        self.capacity_mask = int(mjw.OverflowType.ALL & ~solver_limits)
        self.graph = None
        self.reset(torch.arange(num_envs, device=device))
        # Eager compilation precedes graph capture. These preparation steps are
        # excluded from the physical training transition counter.
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self._advance()
            if capture_graph:
                # Warp cannot begin capture on Torch's legacy default stream.
                # Capture on a registered Warp stream, then replay on the
                # caller's stream. Keep Torch bookkeeping in the same capture.
                torch.cuda.synchronize(device)
                self.capture_stream = wp.Stream(device)
                with (
                    torch.cuda.stream(wp.stream_to_torch(self.capture_stream)),
                    wp.ScopedStream(self.capture_stream),
                ):
                    with wp.ScopedCapture(
                        stream=self.capture_stream
                    ) as capture:
                        self._advance()
                self.graph = capture.graph
        self.reset(torch.arange(num_envs, device=device))
        self._overflow_seen.zero_()

    def _advance(self):
        for _ in range(4):
            mjw.step(self.model, self.data)
            self._overflow_seen.bitwise_or_(wp.to_torch(self.data.overflow))

    def state(self):
        p, quat = self.qpos[:, :3].clone(), self.qpos[:, 3:7].clone()
        rotation = quat_to_mat(quat)
        # MuJoCo's free-joint angular generalized velocity is body-local.
        angular = (rotation @ self.qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)
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
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        mask[ids] = True
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        self.qpos[ids] = torch.cat((p, mat_to_quat(rotation), q), -1)
        self.qvel[ids] = torch.cat(
            (
                v,
                (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
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
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            if self.graph is None:
                self._advance()
            else:
                wp.capture_launch(self.graph, stream=self.stream)
        if (self._overflow_seen & self.capacity_mask).any():
            error = RuntimeError(
                f"MuJoCo Warp status bitmask {self._overflow_seen.tolist()}; rollout stopped for diagnosis"
            )
            error.physics_diagnostic = dict(
                status_bits=self._overflow_seen.clone(),
                qpos=self.qpos.clone(),
                qvel=self.qvel.clone(),
                action=action.clone(),
                context=context,
                active_constraints=wp.to_torch(self.data.nefc).clone(),
                contacts=wp.to_torch(self.data.nacon).clone(),
            )
            raise error
        next_state = self.state()
        if not all(torch.isfinite(value).all() for value in next_state):
            raise FloatingPointError(
                "Invalid physical state; rollout rejected"
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
                self.ep_ret[ids] / (length * MAX_STEP_REWARD),
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
        return result
