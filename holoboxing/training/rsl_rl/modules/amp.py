from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from holoboxing.data.contract import ROBOT_KINEMATICS
from holoboxing.training.rsl_rl.modules.amp_discriminator import (
    AMPDiscriminator,
)
from holoboxing.training.rsl_rl.modules.normalizer import (
    EmpiricalNormalization,
)
from holoboxing.training.rsl_rl.storage.circular_buffer import (
    CircularBuffer,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    DT,
    JOINT_NAMES,
    sha256,
)
from holoboxing.training.tasks.combat.mdp.geometry import (
    normalize_quaternion_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
)


def motion_files(path, *, resource_manifest_sha256=None):
    from holoboxing.data.dataset import PreparedDataset

    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(
            "AMP requires the prepared directory from holoboxing.data prepare"
        )
    dataset = PreparedDataset(path / "manifest.json", validate_contents=False)
    identity = dataset.manifest["asset_identity"]
    if identity.get("robot_kinematics") != ROBOT_KINEMATICS:
        raise ValueError(
            "Regenerate AMP data with competition XML reference kinematics"
        )
    if (
        resource_manifest_sha256 is not None
        and identity.get("resource_manifest_sha256")
        != resource_manifest_sha256
    ):
        raise ValueError("Prepared AMP data and training assets differ")
    dataset.select_sources(split="train")
    return sorted(dataset.root / record["path"] for record in dataset.records)


def amp_frame(
    angular_velocity_b,
    joint_position,
    joint_velocity,
    default_position,
    default_velocity,
):
    return torch.cat(
        (
            angular_velocity_b,
            joint_position - default_position,
            joint_velocity - default_velocity,
        ),
        dim=-1,
    )


def amp_pairs(frames):
    if frames.ndim != 2 or frames.shape[1] != 61 or len(frames) < 2:
        raise ValueError("AMP needs at least two 61D frames")
    if not torch.isfinite(frames).all():
        raise ValueError("AMP frames must be finite")
    return torch.cat((frames[:-1], frames[1:]), dim=-1)


def load_amp_motions(
    path,
    default_position,
    default_velocity,
    device="cpu",
    *,
    resource_manifest_sha256=None,
):
    """Read prepared world states using the original AMP rotation and offsets."""
    defaults = [
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (default_position, default_velocity)
    ]
    if any(
        value.shape != (29,) or not torch.isfinite(value).all()
        for value in defaults
    ):
        raise ValueError("AMP requires finite 29-joint nominal offsets")
    pairs, receipt = [], []
    for file in motion_files(
        path, resource_manifest_sha256=resource_manifest_sha256
    ):
        with np.load(file, allow_pickle=False) as data:
            if not {
                "amp_reference",
                "amp_joint_names",
                "amp_fps",
                "amp_root_quaternion_wxyz",
                "amp_angular_velocity_frame",
            }.issubset(data.files):
                raise ValueError(
                    f"{file.name}: run holoboxing.data prepare and pass its prepared directory"
                )
            if data["amp_joint_names"].tolist() != list(JOINT_NAMES):
                raise ValueError(
                    f"{file.name}: AMP joint order differs from training"
                )
            if data["amp_fps"].shape != () or data["amp_fps"].item() != 1 / DT:
                raise ValueError(
                    f"{file.name}: expected 50 Hz AMP transitions"
                )
            if str(data["amp_angular_velocity_frame"]) != "world":
                raise ValueError(
                    f"{file.name}: AMP angular velocity must be world-frame"
                )
            values = data["amp_reference"]
            if (
                values.ndim != 2
                or values.shape[1] != 122
                or not len(values)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{file.name}: invalid AMP reference transitions"
                )
            frames = torch.as_tensor(values, device=device).reshape(-1, 61)
            quaternion = data["amp_root_quaternion_wxyz"]
            if (
                quaternion.shape != (len(values), 2, 4)
                or quaternion.dtype != np.float32
                or not np.isfinite(quaternion).all()
                or not np.allclose(
                    np.linalg.norm(quaternion.astype(np.float64), axis=-1),
                    1,
                    atol=1e-3,
                    rtol=0,
                )
            ):
                raise ValueError(f"{file.name}: invalid AMP root quaternions")
            quaternion = torch.as_tensor(quaternion, device=device).reshape(
                -1, 4
            )
        angular = quat_apply_wxyz(
            quat_conjugate_wxyz(normalize_quaternion_wxyz(quaternion)),
            frames[:, :3],
        )
        pair = amp_frame(
            angular, frames[:, 3:32], frames[:, 32:], *defaults
        ).reshape(-1, 122)
        if not torch.isfinite(pair).all():
            raise ValueError(f"{file.name}: AMP conversion must remain finite")
        pairs.append(pair)
        receipt.append(
            {
                "file": file.name,
                "sha256": sha256(file),
                "transitions": len(values),
            }
        )
    return torch.cat(pairs), receipt


class AMP:
    """RoboStriker BCE discriminator, demo-fitted frozen feature statistics."""

    def __init__(self, demo, config, device):
        if (
            demo.ndim != 2
            or demo.shape[1] != 122
            or not len(demo)
            or not torch.isfinite(demo).all()
        ):
            raise ValueError("Expected finite 122D AMP demonstrations")
        self.config, self.device = config, device
        self.discriminator = AMPDiscriminator(
            122, config.disc_hidden_dims, "relu"
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.disc_learning_rate,
            weight_decay=config.disc_weight_decay,
        )
        self.normalizer = EmpiricalNormalization([61], until=100_000_000).to(
            device
        )
        with torch.no_grad():
            self.normalizer(demo.to(device).reshape(-1, 61))
        self.normalizer.eval()
        self.demo = CircularBuffer(len(demo), device)
        self.demo.store(demo.to(device))
        self.replay = CircularBuffer(config.amp_replay_buffer_size, device)
        self.optimizer_steps = 0

    def normalize(self, pairs):
        return torch.cat(
            (self.normalizer(pairs[:, :61]), self.normalizer(pairs[:, 61:])),
            dim=-1,
        )

    @torch.no_grad()
    def reward(self, pairs):
        logits = self.discriminator(self.normalize(pairs))
        prob = 1 / (1 + torch.exp(-logits))
        return (
            -torch.log((1 - prob).clamp_min(0.0001))
            * self.config.disc_reward_scale
        ).squeeze(-1)

    def loss(self, current, replay, demo):
        # Source BCE arithmetic and demo gradient penalty, unchanged.
        bce = torch.nn.BCEWithLogitsLoss()
        agent_logits = self.discriminator(current)
        replay_logits = self.discriminator(replay)
        demo_logits = self.discriminator(demo)
        loss = 0.5 * (
            bce(agent_logits, torch.zeros_like(agent_logits))
            + bce(replay_logits, torch.zeros_like(replay_logits))
        )
        loss = loss + bce(demo_logits, torch.ones_like(demo_logits))
        output = [
            layer
            for layer in self.discriminator.modules()
            if isinstance(layer, torch.nn.Linear)
        ][-1]
        loss = loss + self.config.disc_logit_reg * output.weight.square().sum()
        grad = torch.autograd.grad(
            demo_logits,
            demo,
            grad_outputs=torch.ones_like(demo_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        penalty = grad.square().sum(-1).mean()
        loss = loss + self.config.disc_grad_penalty * penalty
        return loss, penalty

    def update(self, ego, opponent, num_updates):
        from holoboxing.training.rsl_rl.algorithms.ppo import (
            require_finite,
        )

        cfg = self.config
        metrics = {"amp/loss": 0.0, "amp/gradient_penalty": 0.0}
        for _ in range(num_updates):
            current = torch.cat(
                [
                    values[
                        torch.randperm(len(values), device=self.device)[
                            : cfg.amp_batch_size // 2
                        ]
                    ]
                    for values in (ego, opponent)
                ],
                dim=0,
            )
            replay = (
                self.replay.sample(cfg.amp_batch_size)
                if len(self.replay)
                else current.clone()
            )
            demo = self.normalize(
                self.demo.sample(cfg.amp_batch_size)
            ).requires_grad_(True)
            loss, penalty = self.loss(
                self.normalize(current), self.normalize(replay), demo
            )
            require_finite((loss, penalty), "AMP loss")
            self.optimizer.zero_grad(set_to_none=True)
            (cfg.disc_coef * loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                cfg.disc_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.optimizer_steps += 1
            require_finite(
                (self.discriminator.state_dict(), self.optimizer.state),
                "AMP update",
            )
            metrics["amp/loss"] += loss.item()
            metrics["amp/gradient_penalty"] += penalty.item()
        self.replay.store(ego)
        self.replay.store(opponent)
        return {key: value / num_updates for key, value in metrics.items()}
