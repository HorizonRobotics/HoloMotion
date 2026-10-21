"""PPO samples raw 32D vectors and executes frozen decoded directions."""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from holoboxing.training.rsl_rl.modules.actor_critic import ActorCritic, mlp
from holoboxing.training.spherical_actor import SphericalActor
from holoboxing.training.tasks.combat.config.sandbag import CRITIC_DIM, ROLES

FORMAT = "holoboxing_frozen_sphere32_rl_v1"


class LatentMeanActor(SphericalActor):
    def forward(self, observations):
        return self.latent(observations)


class SphericalActorCritic(ActorCritic):
    action_dim = 32
    checkpoint_format = FORMAT

    def __init__(self, actor, config):
        nn.Module.__init__(self)
        if config.horizon_rho:
            raise ValueError(
                "Horizon reference noise does not apply to a 32D latent"
            )
        self.config = SimpleNamespace(**{**config.to_dict(), "action_dim": 32})
        self.actor_dim = config.actor_dim
        self.actor_storage_dim = (
            2 * self.actor_dim + 33
            if config.exploration_rho
            else self.actor_dim
        )
        self.policy_spec = copy.deepcopy(actor.spec)
        # Keep the checkpoint's input contract through RL and export.
        for role in ROLES:
            mean = LatentMeanActor(actor.decoder, config, self.policy_spec)
            mean.latent_policy.load_state_dict(
                actor.latent_policy.state_dict()
            )
            setattr(self, f"taskpolicy_{role}", mean)
            setattr(
                self, f"critic_{role}", mlp(CRITIC_DIM, 1, config.hidden_dims)
            )
            setattr(
                self,
                f"std_{role}",
                nn.Parameter(torch.full((32,), config.init_noise_std).log()),
            )
        self.distributions, self.base_distributions = {}, {}
        self._actor_graphs = {}
        self._actor_graphs_enabled = False
        self._graph_batch_roles = None
        self._graph_capture_counts = dict.fromkeys(ROLES, 0)
        self._graph_capture_seconds = 0.0

    def role_parameters(self, role):
        return [p for p in super().role_parameters(role) if p.requires_grad]

    def execute(self, role, raw_action):
        # RolloutStorage retains raw_action and Normal.log_prob(raw_action).
        return getattr(self, f"taskpolicy_{role}").decoder(raw_action)

    def check_frozen(self):
        for role in ROLES:
            getattr(self, f"taskpolicy_{role}").check_frozen()

    def validate_action_mapping(self, state_dict):
        super().validate_action_mapping(state_dict)
        current = self.state_dict()
        for name, value in current.items():
            if ".decoder." in name and not torch.equal(
                state_dict[name].to(value), value
            ):
                raise ValueError("Saved frozen decoder differs")
