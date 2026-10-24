# Adapted from zita-ch/supermimic_vibed, commit a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Differentiable forward kinematics of the G1 in torch, built from the compiled MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import G1_JOINT_NAMES, TRACKED_BODIES
from .rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
    def __init__(
        self, device: str, model, tracked: list[str] = TRACKED_BODIES
    ):
        m = model
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(
            np.array(axes), dtype=torch.float32, device=device
        )
        self.off_p = torch.tensor(
            np.array(offs_p), dtype=torch.float32, device=device
        )
        self.off_R = quat_to_mat(
            torch.tensor(np.array(offs_R), dtype=torch.float32, device=device)
        )
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in G1_JOINT_NAMES order)
        lo, hi = np.zeros(29), np.zeros(29)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (pelvis) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,29) in G1_JOINT_NAMES order.
        """
        P, Rs, V, W = (
            [None] * self.num_bodies,
            [None] * self.num_bodies,
            [None] * self.num_bodies,
            [None] * self.num_bodies,
        )
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
