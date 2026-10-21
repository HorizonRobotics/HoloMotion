"""Reference-only 225 -> unit-sphere 32 -> 225 motion representation."""

import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from holoboxing.training.rsl_rl.modules.actor_critic import mlp
from holoboxing.training.tasks.combat.mdp.geometry import project_xz, rot_xz

FORMAT = "holoboxing_reference_sphere32_v1"


def unit_vector(value):
    """The zero vector uses a tangent chart around the first sphere axis."""
    scaled = value / value.abs().amax(-1, keepdim=True).clamp_min(1)
    fallback = torch.zeros_like(scaled)
    fallback[..., 0] = 1
    scaled = torch.where(
        scaled.norm(dim=-1, keepdim=True) > 1e-8, scaled, scaled + fallback
    )
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@dataclass(frozen=True)
class PriorConfig:
    latent_dim: int = 32
    hidden_dims: tuple = (512, 256)

    def __post_init__(self):
        if self.latent_dim != 32:
            raise ValueError(
                "This representation requires a 32-dimensional unit vector"
            )
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not self.hidden_dims or any(
            type(v) is not int or v < 1 for v in self.hidden_dims
        ):
            raise ValueError(
                "Prior hidden layers must have positive integer widths"
            )


class MotionDecoder(nn.Module):
    def __init__(self, mean, scale, config):
        super().__init__()
        if any(
            v.shape != (225,) or not torch.isfinite(v).all()
            for v in (mean, scale)
        ):
            raise ValueError("Expected finite 225D reference statistics")
        if not (scale > 0).all():
            raise ValueError("Reference scales must be positive")
        self.config = config
        self.network = mlp(
            config.latent_dim, 225, tuple(reversed(config.hidden_dims))
        )
        self.register_buffer("reference_mean", mean.detach().clone())
        self.register_buffer("reference_scale", scale.detach().clone())

    def forward(self, latent):
        value = self.reference_mean + self.reference_scale * self.network(
            unit_vector(latent)
        )
        points = value.reshape(-1, 5, 5, 9)
        return torch.cat(
            (points[..., :3], rot_xz(project_xz(points[..., 3:]))), -1
        ).flatten(1)


class MotionPrior(nn.Module):
    def __init__(self, mean, scale, config=None):
        super().__init__()
        self.config = config or PriorConfig()
        self.encoder = mlp(
            225, self.config.latent_dim, self.config.hidden_dims
        )
        self.decoder = MotionDecoder(mean, scale, self.config)

    def encode(self, reference):
        return unit_vector(
            self.encoder(
                (reference - self.decoder.reference_mean)
                / self.decoder.reference_scale
            )
        )

    def forward(self, reference):
        return self.decoder(self.encode(reference))


def model_payload(model):
    return {
        "format": FORMAT,
        "config": asdict(model.config),
        "model": model.state_dict(),
    }


def state_sha256(module):
    digest = hashlib.sha256()
    state = module if isinstance(module, dict) else module.state_dict()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_prior(path, *, device="cpu", frozen=True):
    from holoboxing.training.rsl_rl.algorithms.ppo import require_finite

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != FORMAT:
        raise ValueError("Expected a reference-only sphere32 checkpoint")
    require_finite(saved["model"], "motion prior")
    state = saved["model"]
    with torch.random.fork_rng(devices=[]):
        model = MotionPrior(
            state["decoder.reference_mean"],
            state["decoder.reference_scale"],
            PriorConfig(**saved["config"]),
        )
    model.load_state_dict(state, strict=True)
    model.to(device)
    if frozen:
        model.eval().requires_grad_(False)
    return model, saved
