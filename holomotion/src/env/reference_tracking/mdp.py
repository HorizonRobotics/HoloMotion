# Adapted from zita-ch/supermimic_vibed, commit a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: pelvis position, rotation matrix, world linear/angular
velocity of the pelvis link frame, joint positions and velocities.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch

from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import (
    mat_to_6d,
    mat_to_quat,
    quat_to_mat,
    rot_angle_sq,
    yaw_mat,
)

STATE_DIM = 71
OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6
XI_DIM = 33
NUM_J = 29

# Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
        torch.full((6,), 0.05),
    )
)
# normalization of the obs before the policy (qd and angular velocity are large)
OBS_SCALE = torch.cat(
    (
        torch.ones(29),
        torch.full((29,), 0.1),
        torch.full((3,), 0.25),
        torch.ones(3),
        torch.ones(29),
        torch.ones(6),
    )
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return (
        x[..., 0:3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13:42],
        x[..., 42:71],
    )


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 29 + 29


class MDP:
    def __init__(
        self,
        kin: G1Kinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        cfg: RewardCfg = RewardCfg(),
        *,
        reference_orientation: bool = False,
    ):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.reference_orientation = reference_orientation
        self.obs_dim = OBS_DIM if reference_orientation else OBS_DIM - 6
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = (
            mid - half * cfg.soft_limit_factor,
            mid + half * cfg.soft_limit_factor,
        )
        dev = default_q.device
        self.obs_noise = OBS_NOISE[: self.obs_dim].to(dev)
        self.obs_scale = OBS_SCALE[: self.obs_dim].to(dev)

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1:30]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        parts = (
            q - self.default_q - dq0,
            qd,
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            g_b,
            a_prev,
        )
        if self.reference_orientation:
            parts = (*parts, mat_to_6d(Rt @ self.motion.root_R[f]))
        return torch.cat(parts, -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            (
                dR.unsqueeze(-3)
                @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)
            ).squeeze(-1)
        )
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (
            torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)
        ).sum(-1)
        r = (
            r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim
        ) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(
        self, S, f, tz: float = 0.02, tt: float = 0.04
    ) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (
            1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt)
        )
        return 1 - alive

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            (
                dR.unsqueeze(-3)
                @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)
            ).squeeze(-1)
        )
        e_body = (
            (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1))
            .squeeze(-1)
            .flatten(-2)
        )
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat(
            (invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1
        )


CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            (xi[..., :1] - 0.75) / 0.45,
            xi[..., 1:30] / 0.01,
            xi[..., 30:33] / 0.05,
        ),
        -1,
    )
