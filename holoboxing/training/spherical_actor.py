"""Causal observations select a unit latent for a frozen decoder."""

import copy
from dataclasses import asdict

import torch
from torch import nn

from holoboxing.data.common import sha256
from holoboxing.data.contract import same_data_assets
from holoboxing.training.motion_prior import (
    MotionDecoder,
    PriorConfig,
    load_prior,
    state_sha256,
    unit_vector,
)
from holoboxing.training.rsl_rl.modules.actor_critic import mlp

FORMAT = "holoboxing_frozen_sphere32_supervised_v1"
KIND = "frozen_reference_sphere32"


class SphericalActor(nn.Module):
    actor_dim = 1097

    def __init__(self, decoder, config, spec):
        super().__init__()
        self.decoder = copy.deepcopy(decoder).eval().requires_grad_(False)
        self.latent_policy = mlp(1095, 32, config.hidden_dims)
        self.spec = copy.deepcopy(spec)
        if self.spec.get("self_observation_mode", "history10") not in (
            "history10",
            "current_state",
        ) or not self.spec.get("opponent_condition", False):
            raise ValueError("Unsupported frozen-prior observation contract")
        # Explicit mapping buffers keep generic checkpoint checks usable.
        self.register_buffer(
            "neutral_reference", decoder.reference_mean.detach().clone()
        )
        self.register_buffer(
            "physical_scale",
            torch.ones(32, device=decoder.reference_mean.device),
        )
        self.check_frozen()

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        return self

    def latent(self, observations):
        causal = observations[..., :1095]
        mode = self.spec.get("self_observation_mode", "history10")
        if mode == "current_state":
            # Current108: angular velocity3, pose45, linear velocity15,
            # previous measured pose45. Remove it and the nine old frames.
            causal = torch.cat(
                (
                    torch.zeros_like(causal[..., :972]),
                    causal[..., 972:1035],
                    torch.zeros_like(causal[..., 1035:1080]),
                    causal[..., 1080:],
                ),
                -1,
            )
        return unit_vector(self.latent_policy(causal))

    def forward(self, observations):
        return self.decoder(self.latent(observations))

    def check_frozen(self):
        if state_sha256(self.decoder) != self.spec["decoder_sha256"]:
            raise ValueError("Frozen motion decoder changed")
        if any(p.requires_grad for p in self.decoder.parameters()):
            raise ValueError("Motion decoder parameters must remain frozen")


def actor_from_state(state, spec, config):
    if spec.get("kind") != KIND:
        raise ValueError("Unknown frozen-prior policy specification")
    decoder = MotionDecoder(
        state["decoder.reference_mean"],
        state["decoder.reference_scale"],
        PriorConfig(**spec["prior_config"]),
    )
    decoder.load_state_dict(
        {
            k.removeprefix("decoder."): v
            for k, v in state.items()
            if k.startswith("decoder.")
        }
    )
    actor = SphericalActor(decoder, config, spec)
    actor.load_state_dict(state, strict=True)
    actor.check_frozen()
    return actor


def actor_from_prior(
    path,
    config,
    records,
    validation_records,
):
    prior, saved = load_prior(path)
    if not same_data_assets(
        saved.get("asset_identity"), records.manifest["asset_identity"]
    ):
        raise ValueError("Motion prior and imitation kinematics differ")
    if saved.get("source_selection", {}).get(
        "split"
    ) != "train" or not saved.get("training_groups"):
        raise ValueError("Prior requires verified training-group provenance")
    groups = {tuple(row) for row in saved["training_groups"]}
    if validation_records is not None:
        heldout = {
            (r["dataset"], r["group"])
            for r in validation_records.source_metadata()
            if r["split"] == "validation"
        }
        if groups & heldout:
            raise ValueError(
                "Prior pretraining leaked imitation validation groups"
            )
    spec = {
        "kind": KIND,
        "prior_checkpoint_sha256": sha256(path),
        "prior_config": asdict(prior.config),
        "decoder_sha256": state_sha256(prior.decoder),
        "opponent_condition": True,
        "self_observation_mode": "current_state",
        "preserve_input_mask": True,
        "training_groups": sorted(groups),
    }
    return SphericalActor(prior.decoder, config, spec)
