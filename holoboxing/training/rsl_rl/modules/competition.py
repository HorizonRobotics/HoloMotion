"""Export boundary: measured public observations to physical orthogonal poses."""
import torch
from torch import nn

from holoboxing.training.tasks.combat.mdp.geometry import project_xz, rot_xz


class CompetitionPolicy(nn.Module):
    def __init__(self, actor, normalizer):
        super().__init__()
        if actor.actor_dim != normalizer.actor_dim:
            raise ValueError("Actor and normalizer layouts differ")
        self.actor, self.normalizer = actor, normalizer

    def forward(self, obs):
        if obs.ndim != 2 or obs.shape[1] != 1097:
            raise ValueError("Expected public observations with shape [batch, 1097]")
        history = obs[..., :1080]
        current = history[..., -108:]
        other = obs[..., 1080:1095]
        task = torch.cat((current, other), -1)
        prediction = self.actor(self.normalizer(dict(self=current, task=task, history=history)))
        points = prediction.reshape(-1, 5, 5, 9)
        return torch.cat((points[..., :3], rot_xz(project_xz(points[..., 3:]))), -1).flatten(1)
