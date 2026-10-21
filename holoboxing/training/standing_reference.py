"""Hold the opponent's neutral world targets until episode reset."""

import torch
from torch.nn import functional as F

from holoboxing.training.tasks.combat.mdp.geometry import decode_output
from holomotion.src.utils.torch_utils import quat_from_matrix


class WorldStandingReference:
    def __init__(self, neutral, num_envs, device, *, role=1):
        self.neutral = torch.as_tensor(neutral, device=device).float()
        if (
            self.neutral.shape != (225,)
            or not torch.isfinite(self.neutral).all()
        ):
            raise ValueError(
                "Standing reference requires a finite neutral 225D pose"
            )
        if role not in (0, 1) or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Invalid standing role or environment count")
        self.role, self.num_envs = role, num_envs
        self.pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.position = torch.empty(num_envs, 5, 5, 3, device=device)
        self.quaternion = torch.empty(num_envs, 5, 5, 4, device=device)
        self.anchor = torch.empty(num_envs, 3, device=device)
        self.height = torch.empty(num_envs, 1, device=device)

    def reset(self, ids):
        self.pending[ids] = True

    @torch.no_grad()
    def apply(self, measured, position, quaternion, anchor, height):
        ids = self.pending.nonzero().flatten()
        rows = 2 * ids + self.role
        if len(ids):
            p, r = decode_output(
                measured["position"][rows],
                measured["quaternion"][rows],
                self.neutral.expand(len(ids), -1),
                action_dim=225,
            )
            self.position[ids] = p
            self.quaternion[ids] = F.normalize(quat_from_matrix(r), dim=-1)
            self.anchor[ids] = anchor[rows]
            self.height[ids] = height[rows]
            self.pending[ids] = False
        role_rows = slice(self.role, None, 2)
        # Never recenter these targets on a robot displaced by contact.
        for output, fixed in (
            (position, self.position),
            (quaternion, self.quaternion),
            (anchor, self.anchor),
            (height, self.height),
        ):
            output[role_rows] = fixed
        return position, quaternion, anchor, height
