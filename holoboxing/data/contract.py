"""Competition assets and causal observation history for prepared records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.runtime import representation as geometry
from holoboxing.evaluation.runtime.hand_hit_rules import (
    SCORING_VERSION,
    scoring_config,
)
from holoboxing.evaluation.runtime.model_contract import (
    CONTRACT,
    sha256,
)

OBSERVATION_DIM = 1097
HISTORY_LENGTH = 10


@dataclass(frozen=True)
class CompetitionAssets:
    """Resolve both simulator and display assets from one evaluator inventory."""

    resources: ResourceTree

    @classmethod
    def load(cls, directory: str | Path) -> CompetitionAssets:
        return cls(ResourceTree(directory))

    def identity(self) -> dict:
        tree = self.resources
        return {
            "robot_kinematics": "competition_xml_v1",
            "policy_contract": CONTRACT,
            "scoring_version": SCORING_VERSION,
            "scoring": scoring_config(),
            "resource_manifest_sha256": tree.manifest_sha256,
            "scene_sha256": sha256(tree.scene),
            "initial_state_sha256": sha256(tree.initial_state),
            "display_manifest_sha256": sha256(tree.path("manifest.json")),
        }


def target_from_world(
    position, quaternion, future_position, future_quaternion
):
    """Five future frames in the current pelvis's FULL 3-D coordinate frame."""
    return geometry.encode_output(
        position, quaternion, future_position, future_quaternion
    )


class CompetitionHistory:
    """Independent paired worlds with the evaluator's causal ten-frame history."""

    def __init__(self, worlds: int, device="cpu"):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("worlds must be a positive integer")
        self.worlds = worlds
        self.device = torch.device(device)
        self.history = self.previous_position = self.previous_quaternion = None
        self.last_step = None

    def observe(
        self,
        position,
        quaternion,
        link_origin_velocity,
        pelvis_angular_velocity_b,
        remaining_seconds,
        scores,
        *,
        reset,
        step: int,
        advance: bool = True,
    ):
        expected = (
            None if self.last_step is None else self.last_step + int(advance)
        )
        if expected is not None and step != expected:
            raise ValueError(
                "History must advance exactly once per control observation"
            )
        shapes = (
            (position, (self.worlds, 2, 5, 3)),
            (quaternion, (self.worlds, 2, 5, 4)),
            (link_origin_velocity, (self.worlds, 2, 5, 3)),
            (pelvis_angular_velocity_b, (self.worlds, 2, 3)),
            (remaining_seconds, (self.worlds,)),
            (scores, (self.worlds, 2)),
            (reset, (self.worlds,)),
        )
        for value, shape in shapes:
            if value.shape != shape or value.device != self.device:
                raise ValueError(f"Expected shape {shape} on {self.device}")
        if reset.dtype != torch.bool:
            raise ValueError("reset must be a Boolean per-world mask")
        for value, _ in shapes[:-1]:
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(
                    "Physical measurements and context must be finite float32"
                )
        if (quaternion.norm(dim=-1) < 1e-8).any():
            raise ValueError("Physical quaternions must be nonzero")
        p, q, velocity, omega = (
            value.flatten(0, 1)
            for value in (
                position,
                quaternion,
                link_origin_velocity,
                pelvis_angular_velocity_b,
            )
        )
        mask = reset.repeat_interleave(2)
        if self.history is None:
            mask = torch.ones_like(mask)
            previous_p, previous_q = p, q
        else:
            previous_p = torch.where(
                mask[:, None, None], p, self.previous_position
            )
            previous_q = torch.where(
                mask[:, None, None], q, self.previous_quaternion
            )
        raw = geometry.raw_self(p, q, velocity, omega, previous_p, previous_q)
        repeated = raw[:, None].expand(-1, HISTORY_LENGTH, -1)
        shifted = (
            repeated
            if self.history is None
            else torch.cat((self.history[:, 1:], raw[:, None]), 1)
            if advance
            else self.history
        )
        self.history = torch.where(
            mask[:, None, None], repeated, shifted
        ).clone()
        state = geometry.reframe_c(
            self.history, p, q, position.flip(1).flatten(0, 1)
        )
        obs = torch.cat(
            (
                state["history"],
                state["task"][:, -15:],
                remaining_seconds[:, None].expand(-1, 2).reshape(-1, 1),
                (scores - scores.flip(1)).reshape(-1, 1),
            ),
            -1,
        )
        self.previous_position, self.previous_quaternion = p.clone(), q.clone()
        self.last_step = step
        return obs.reshape(self.worlds, 2, OBSERVATION_DIM)
