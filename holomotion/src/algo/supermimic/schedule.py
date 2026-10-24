"""Exact update clocks and explicit model-data reuse ratios."""

from dataclasses import asdict, dataclass, replace
from fractions import Fraction


def rate(value):
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "Update rates must be finite nonnegative rational numbers"
        ) from error
    if result < 0:
        raise ValueError("Update rates must be nonnegative")
    return result


@dataclass(frozen=True)
class ScheduleConfig:
    physical_envs: int
    actor_batch_size: int
    imagination_horizon: int
    world_batch_size: int
    world_window_length: int
    world_batches_per_update: int
    actor_updates_per_vector_step: str = "0.4"
    world_updates_per_vector_step: str = "0.4"
    warmup_vector_steps: int = 64
    world_only_updates: int = 0

    @classmethod
    def from_reuse_ratios(
        cls,
        *,
        imagined_steps_per_real_transition,
        fitted_steps_per_real_transition,
        **dimensions,
    ):
        """Keep sample reuse explicit when environment or logical batches change.

        Microbatch size does not enter these ratios: it changes execution,
        not the number of samples in one optimizer update.
        """
        config = cls(**dimensions)
        actor_steps = config.actor_batch_size * config.imagination_horizon
        world_steps = (
            config.world_batch_size
            * config.world_window_length
            * config.world_batches_per_update
        )
        if min(actor_steps, world_steps, config.physical_envs) <= 0:
            raise ValueError(
                "Reuse ratios require positive batch, horizon and environment dimensions"
            )
        return replace(
            config,
            actor_updates_per_vector_step=str(
                rate(imagined_steps_per_real_transition)
                * config.physical_envs
                / actor_steps
            ),
            world_updates_per_vector_step=str(
                rate(fitted_steps_per_real_transition)
                * config.physical_envs
                / world_steps
            ),
        )


class UpdateSchedule:
    """Accrue updates after each vectorized physical step, never per environment.

    Initial replay warmup earns no optimizer credits. Actor credits begin on
    the first physical step AFTER the requested world-only updates complete.
    Skipped early actor updates never become a later catch-up burst.
    Pending events must finish before collecting another physical step.
    Save only at quiescent boundaries; native physics restoration is separate.
    """

    def __init__(self, config: ScheduleConfig):
        dimensions = (
            config.physical_envs,
            config.actor_batch_size,
            config.imagination_horizon,
            config.world_batch_size,
            config.world_window_length,
            config.world_batches_per_update,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in dimensions
        ):
            raise ValueError("Schedule dimensions must be positive integers")
        if any(
            not isinstance(value, int) or value < 0
            for value in (
                config.warmup_vector_steps,
                config.world_only_updates,
            )
        ):
            raise ValueError(
                "Warmup steps and world-only updates must be nonnegative integers"
            )
        self.actor_rate, self.world_rate = (
            rate(config.actor_updates_per_vector_step),
            rate(config.world_updates_per_vector_step),
        )
        if not self.actor_rate and not self.world_rate:
            raise ValueError("At least one optimizer clock must be enabled")
        if config.world_only_updates and not self.world_rate:
            raise ValueError(
                "World-only warmup cannot complete with a disabled world optimizer clock"
            )
        self.config = replace(
            config,
            actor_updates_per_vector_step=str(self.actor_rate),
            world_updates_per_vector_step=str(self.world_rate),
        )
        self.vector_steps = self.actor_eligible_steps = 0
        self.world_updates = self.actor_updates = 0
        self.pending_world = self.pending_actor = 0
        self.world_credit = self.actor_credit = Fraction(0)

    def advance(self):
        if self.pending_world or self.pending_actor:
            raise RuntimeError(
                "Finish pending optimizer events before collecting another physical step"
            )
        self.vector_steps += 1
        if self.vector_steps <= self.config.warmup_vector_steps:
            return 0, 0
        actor_ready = self.world_updates >= self.config.world_only_updates
        self.world_credit += self.world_rate
        self.pending_world = (
            self.world_credit.numerator // self.world_credit.denominator
        )
        self.world_credit -= self.pending_world
        if actor_ready:
            self.actor_eligible_steps += 1
            self.actor_credit += self.actor_rate
            self.pending_actor = (
                self.actor_credit.numerator // self.actor_credit.denominator
            )
            self.actor_credit -= self.pending_actor
        return self.pending_world, self.pending_actor

    def complete_world(self):
        if self.pending_world < 1:
            raise RuntimeError("No pending world optimizer event")
        self.pending_world -= 1
        self.world_updates += 1

    def complete_actor(self):
        if self.pending_actor < 1:
            raise RuntimeError("No pending actor optimizer event")
        self.pending_actor -= 1
        self.actor_updates += 1

    def ratios(self):
        world_samples = (
            self.config.world_batch_size
            * self.config.world_window_length
            * self.config.world_batches_per_update
        )
        actor_samples = (
            self.config.actor_batch_size * self.config.imagination_horizon
        )
        return dict(
            steady_imagined_steps_per_real_transition=str(
                self.actor_rate * actor_samples / self.config.physical_envs
            ),
            steady_fitted_steps_per_real_transition=str(
                self.world_rate * world_samples / self.config.physical_envs
            ),
            collected_real_transitions=self.vector_steps
            * self.config.physical_envs,
            imagined_steps=self.actor_updates * actor_samples,
            fitted_transition_targets=self.world_updates * world_samples,
        )

    def state_dict(self):
        if self.pending_world or self.pending_actor:
            raise RuntimeError(
                "Checkpoint the update clock only after pending optimizer events finish"
            )
        return dict(
            configuration=asdict(self.config),
            vector_steps=self.vector_steps,
            actor_eligible_steps=self.actor_eligible_steps,
            world_updates=self.world_updates,
            actor_updates=self.actor_updates,
            world_credit=str(self.world_credit),
            actor_credit=str(self.actor_credit),
        )

    def load_state_dict(self, value):
        if self.pending_world or self.pending_actor:
            raise RuntimeError(
                "Cannot replace a clock with pending optimizer events"
            )
        if value["configuration"] != asdict(self.config):
            raise ValueError(
                "Saved update schedule differs from the requested dimensions or rates"
            )
        names = (
            "vector_steps",
            "actor_eligible_steps",
            "world_updates",
            "actor_updates",
        )
        if any(
            not isinstance(value[name], int) or value[name] < 0
            for name in names
        ):
            raise ValueError("Invalid saved update counters")
        active = max(
            0, value["vector_steps"] - self.config.warmup_vector_steps
        )
        delay = (
            (
                Fraction(self.config.world_only_updates) / self.world_rate
            ).__ceil__()
            if self.config.world_only_updates
            else 0
        )
        actor_active = max(0, active - delay)
        world_credit, actor_credit = (
            rate(value["world_credit"]),
            rate(value["actor_credit"]),
        )
        if (
            value["actor_eligible_steps"] != actor_active
            or world_credit
            != active * self.world_rate - value["world_updates"]
            or actor_credit
            != actor_active * self.actor_rate - value["actor_updates"]
            or world_credit >= 1
            or actor_credit >= 1
        ):
            raise ValueError(
                "Saved update credits do not match completed physical/optimizer events"
            )
        for name in names:
            setattr(self, name, value[name])
        self.world_credit, self.actor_credit = world_credit, actor_credit
