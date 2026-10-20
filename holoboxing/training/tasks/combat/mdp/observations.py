from __future__ import annotations

import torch

from holoboxing.training.tasks.combat.mdp.geometry import (
    heading,
    matrix,
    rotate,
)

POINT_INDICES = (0, 28, 29, 18, 19)


def a_half_width(like):
    scale = like.new_zeros(192)
    scale[:3], scale[3:32], scale[32:61], scale[90:] = 0.2, 0.01, 0.5, 0.2
    return scale


def apply_record(measured, record):
    """Re-express saved errors, never resample or perturb real state."""
    delta = record["a_delta"]
    history = record["raw_history"]
    inverse = heading(measured["quaternion"][:, 0]).transpose(-1, -2)[:, None, None]

    def pose(values):
        values = values.reshape(-1, 10, 5, 9)
        return torch.cat(
            (rotate(inverse, values[..., :3]), values[..., 3:]), -1
        ).flatten(-2)

    c_delta = torch.cat(
        (
            history[..., :3],
            pose(history[..., 3:48]),
            history[..., 48:63],
            pose(history[..., 63:]),
        ),
        -1,
    )
    own = delta[:, 90:180].reshape(-1, 30, 3)[:, POINT_INDICES].flatten(1)
    current = measured["self"] + c_delta[:, -1]
    return {
        "actor": measured["actor"] + delta,
        "self": current,
        "task": torch.cat(
            (
                current,
                measured["task"][:, 108:123] + own,
                measured["task"][:, 123:] + record["opponent_delta"],
            ),
            -1,
        ),
        "history": measured["history"] + c_delta.flatten(1),
    }


class ActorObservationNoise:
    def __init__(self, device, seed=42):
        if type(seed) is not int or seed < 0:
            raise ValueError("Expected a nonnegative random seed")
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.reset_history()

    def reset_history(self):
        self.raw_history = self.previous_position_delta = None
        self.last_step = self.last_reset = self.cached = self.record = None

    def rng_state(self):
        return self.generator.get_state().cpu().clone()

    def observe(self, measured, step, reset, *, refresh=False):
        if refresh:
            if self.last_step != step:
                raise ValueError("Local refresh requires the current measurement")
        elif self.last_step == step:
            if not torch.equal(reset, self.last_reset):
                raise ValueError("Repeated measurement changed reset identity")
            return self.cached
        if not refresh and self.last_step is not None and step != self.last_step + 1:
            raise ValueError("Actor-noise history must advance once per decision")
        actor = measured["actor"]
        count = len(actor)
        if (
            actor.shape != (count, 192)
            or reset.shape != (count // 2,)
            or reset.dtype != torch.bool
        ):
            raise ValueError("Expected paired clean A192 measurement and reset mask")
        if refresh:
            # Reuse this decision's noise; only reset the affected histories.
            delta, opponent = self.record["a_delta"], self.record["opponent_delta"]
        else:
            draws = (
                torch.rand(
                    (count, 207),
                    device=actor.device,
                    dtype=actor.dtype,
                    generator=self.generator,
                )
                * 2
                - 1
            )
            delta, opponent = (
                draws[:, :192] * a_half_width(actor),
                draws[:, 192:] * 0.2,
            )
        full = delta[:, 90:180].reshape(count, 30, 3)[:, POINT_INDICES]
        world = rotate(matrix(measured["quaternion"][:, 0])[:, None], full)
        reset = reset.repeat_interleave(2)
        previous = (
            world
            if self.previous_position_delta is None
            else torch.where(reset[:, None, None], world, self.previous_position_delta)
        )
        zeros = actor.new_zeros(count, 5, 6)
        raw = torch.cat(
            (
                delta[:, :3],
                torch.cat((world, zeros), -1).flatten(1),
                actor.new_zeros(count, 15),
                torch.cat((previous, zeros), -1).flatten(1),
            ),
            -1,
        )
        repeated = raw[:, None].expand(-1, 10, -1)
        if self.raw_history is None:
            shifted = repeated
        elif refresh:
            shifted = self.raw_history
        else:
            shifted = torch.cat((self.raw_history[:, 1:], raw[:, None]), 1)
        self.raw_history = torch.where(reset[:, None, None], repeated, shifted).clone()
        self.previous_position_delta = world.clone()
        self.record = {
            "a_delta": delta,
            "opponent_delta": opponent,
            "raw_history": self.raw_history,
        }
        self.cached = apply_record(measured, self.record)
        self.last_step = step
        self.last_reset = (
            self.last_reset | reset[::2] if refresh else reset[::2].clone()
        )
        return self.cached
