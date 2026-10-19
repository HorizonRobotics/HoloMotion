"""One C-network task with explicit resources, action encoding and budget."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from holoboxing.evaluation.runtime.model_contract import (
    INPUT_LINK_NAMES,
    sha256 as sha256,
)

AGENTS = ("robot_A", "robot_B")
ROLES = ("ego", "opponent")
POINTS = INPUT_LINK_NAMES
DT = 0.02
ACTION_DIM = 225
ACTOR_DIM = 1097
CRITIC_DIM = 225
ASSETS = Path(__file__).resolve().parents[3] / "assets"
SCHEMA = json.loads((ASSETS / "robot_schema.json").read_text())
JOINT_NAMES = tuple(SCHEMA["joint_names"])
BODY_NAMES = tuple(SCHEMA["body_names"])
DEFAULTS = json.loads((ASSETS.parent / "configs/training.json").read_text())


@dataclass(frozen=True)
class TrainConfig:
    actor_dim: int = DEFAULTS["actor_dim"]
    action_dim: int = DEFAULTS["action_dim"]
    seed: int = DEFAULTS["seed"]
    num_envs: int = DEFAULTS["num_envs"]
    num_steps_per_env: int = DEFAULTS["num_steps_per_env"]
    num_learning_epochs: int = DEFAULTS["num_learning_epochs"]
    num_mini_batches: int = DEFAULTS["num_mini_batches"]
    learning_rate: float = DEFAULTS["learning_rate"]
    min_learning_rate: float = DEFAULTS["min_learning_rate"]
    max_learning_rate: float = DEFAULTS["max_learning_rate"]
    desired_kl: float = DEFAULTS["desired_kl"]
    gamma: float = DEFAULTS["gamma"]
    lam: float = DEFAULTS["lam"]
    clip_param: float = DEFAULTS["clip_param"]
    value_loss_coef: float = DEFAULTS["value_loss_coef"]
    entropy_coef: float = DEFAULTS["entropy_coef"]
    max_grad_norm: float = DEFAULTS["max_grad_norm"]
    exploration_rho: float = DEFAULTS["exploration_rho"]
    horizon_rho: float = DEFAULTS["horizon_rho"]
    init_noise_std: float = DEFAULTS["init_noise_std"]
    # Physical residual units, not statistics from an old supervised dataset.
    position_scale_m: float = DEFAULTS["position_scale_m"]
    rotation_scale: float = DEFAULTS["rotation_scale"]
    actor_forward_rows: int = DEFAULTS["actor_forward_rows"]
    decoder_init_gain: float = DEFAULTS["decoder_init_gain"]
    hidden_dims: tuple = tuple(DEFAULTS["hidden_dims"])
    disc_hidden_dims: tuple = tuple(DEFAULTS["disc_hidden_dims"])
    disc_learning_rate: float = DEFAULTS["disc_learning_rate"]
    disc_weight_decay: float = DEFAULTS["disc_weight_decay"]
    disc_max_grad_norm: float = DEFAULTS["disc_max_grad_norm"]
    disc_coef: float = DEFAULTS["disc_coef"]
    disc_logit_reg: float = DEFAULTS["disc_logit_reg"]
    disc_grad_penalty: float = DEFAULTS["disc_grad_penalty"]
    disc_reward_scale: float = DEFAULTS["disc_reward_scale"]
    task_reward_weight: float = DEFAULTS["task_reward_weight"]
    style_reward_weight: float = DEFAULTS["style_reward_weight"]
    amp_batch_size: int = DEFAULTS["amp_batch_size"]
    amp_replay_buffer_size: int = DEFAULTS["amp_replay_buffer_size"]

    def __post_init__(self):
        if DEFAULTS.keys() != {field.name for field in fields(self)}:
            raise ValueError("training.json fields must match TrainConfig")
        if type(self.actor_dim) is not int or self.actor_dim != ACTOR_DIM:
            raise ValueError("Actor input must be 1097")
        for name in ("hidden_dims", "disc_hidden_dims"):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ValueError(f"{name} must be a sequence of widths")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.action_dim) is not int or self.action_dim != 225:
            raise ValueError("This experiment requires C225")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Expected a nonnegative 32-bit seed")
        if any(
            type(width) is not int or width < 1
            for dims in (self.hidden_dims, self.disc_hidden_dims)
            for width in dims
        ):
            raise ValueError("Network widths must be positive integers")
        if not self.hidden_dims or not self.disc_hidden_dims:
            raise ValueError("Network hidden layers must not be empty")
        for name in (
            "num_envs",
            "num_steps_per_env",
            "num_learning_epochs",
            "num_mini_batches",
            "actor_forward_rows",
            "amp_batch_size",
            "amp_replay_buffer_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        batch = self.num_envs * self.num_steps_per_env
        if batch < 2 or batch % self.num_mini_batches:
            raise ValueError(
                "Rollout must contain >=2 rows and divide into minibatches"
            )
        if self.amp_batch_size < 2 or self.amp_batch_size % 2:
            raise ValueError("AMP batch size must be positive and even")
        for field in fields(self):
            if field.type is float:
                value = getattr(self, field.name)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid {field.name}")
        for name in (
            "position_scale_m",
            "rotation_scale",
            "init_noise_std",
            "learning_rate",
            "min_learning_rate",
            "max_learning_rate",
            "decoder_init_gain",
            "max_grad_norm",
            "disc_max_grad_norm",
            "disc_learning_rate",
            "desired_kl",
            "clip_param",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0 <= self.exploration_rho < 1 and 0 <= self.horizon_rho < 1):
            raise ValueError("Invalid exploration correlation")
        if (
            not self.min_learning_rate
            <= self.learning_rate
            <= self.max_learning_rate
        ):
            raise ValueError(
                "Learning rate must lie within its adaptive bounds"
            )
        if not 0 < self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("Invalid return discount")

    def to_dict(self):
        return asdict(self)
