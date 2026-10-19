import torch
from torch import nn

SELF_DIM = 108
TASK_DIM = 123


def unpack_actor_observations(observations):
    """Split one packed snapshot without changing its normalization."""
    if observations.shape[-1] != 1097:
        raise ValueError("Actor input must be 1097")
    history = observations[..., :1080]
    current = history[..., -108:]
    task = torch.cat((current, observations[..., 1080:1095]), dim=-1)
    return current, task, history


def validate_statistics(state):
    """Reject finite but impossible statistics before changing a live policy."""
    for name, value in state.items():
        if name.endswith(("_variance", "_var", "_std", "count")) and (value < 0).any():
            raise ValueError(f"Invalid saved normalization statistic: {name}")
        if name.endswith("until") and (value <= 0).any():
            raise ValueError(f"Invalid saved normalization cap: {name}")


class ActorNormalizer(nn.Module):
    """Update current measurements once; store the returned snapshot in PPO."""

    def __init__(self, until=100_000_000):
        super().__init__()
        self.actor_dim = 1097
        self.task_dim = TASK_DIM
        if type(until) is not int or until <= 0:
            raise ValueError("Expected a positive observation count cap")
        self.register_buffer("until", torch.tensor(until, dtype=torch.long))
        for name, width in (("self", SELF_DIM), ("task", self.task_dim)):
            self.register_buffer(
                name + "_mean", torch.zeros(width, dtype=torch.float64)
            )
            self.register_buffer(
                name + "_variance", torch.zeros(width, dtype=torch.float64)
            )
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, current, task):
        self._check(current, task)
        if self.count >= self.until:
            return
        size = current.shape[0]
        if size == 0:
            raise ValueError("Cannot update statistics from an empty batch")
        total = self.count + size
        for name, value in (("self", current), ("task", task)):
            value = value.double()
            mean = getattr(self, name + "_mean")
            variance = getattr(self, name + "_variance")
            delta = value.mean(0) - mean
            ratio = size / total.double()
            variance.mul_(1 - ratio).add_(value.var(0, unbiased=False) * ratio)
            variance.add_(delta.square() * ratio * (1 - ratio))
            mean.add_(delta * ratio)
        self.count.copy_(total)

    def _check(self, current, task):
        if (
            current.ndim != 2
            or current.shape[1] != SELF_DIM
            or task.shape != (current.shape[0], self.task_dim)
            or not torch.isfinite(current).all()
            or not torch.isfinite(task).all()
        ):
            raise ValueError("Expected finite, aligned current self/task measurements")

    def forward(self, batch):
        current, task, history = batch["self"], batch["task"], batch["history"]
        self._check(current, task)
        if history.shape not in (
            (len(current), 1080),
            (len(current), 10, SELF_DIM),
        ):
            raise ValueError("Expected the true ten-frame self history")
        if not torch.isfinite(history).all() or self.count <= 0:
            raise ValueError("Nonfinite history or unfitted statistics")

        def normalize(name, value):
            mean = getattr(self, name + "_mean").to(value.dtype)
            std = (
                getattr(self, name + "_variance").clamp_min(1e-6).sqrt().to(value.dtype)
            )
            return ((value - mean) / std).clamp(-10, 10)

        # Match context is accepted but deliberately unused by this experiment.
        return torch.cat((
            normalize("self", history.reshape(-1, 10, SELF_DIM)).flatten(1),
            normalize("task", task)[..., -15:],
            torch.zeros_like(current[..., :2]),
        ), dim=-1)
