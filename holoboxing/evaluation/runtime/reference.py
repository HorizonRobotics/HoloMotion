"""Map all five public future poses into the saved HoloMotion V2 schema."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holomotion.src.motion_tracking import link_actor_packing, link_observation

from .model_contract import INPUT_LINK_NAMES, sha256

COMMAND_LINK_NAMES = link_observation.DEFAULT_COMMAND_LINK_NAMES
FIVE_POINT_MASK = np.array(
    [name in INPUT_LINK_NAMES for name in COMMAND_LINK_NAMES], dtype=np.float32
)


@dataclass(frozen=True)
class FivePointTargets:
    current_root_position: np.ndarray
    current_root_rotation_wxyz: np.ndarray
    future_positions: np.ndarray
    future_rotations_wxyz: np.ndarray
    frame_offsets: np.ndarray


def rotation_from_xz(values):
    values = np.asarray(values, dtype=np.float64)
    x = values[..., :3]
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    z = values[..., 3:] - (values[..., 3:] * x).sum(-1, keepdims=True) * x
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    return np.stack((x, np.cross(z, x), z), axis=-1)


class ReferenceSchema:
    input_dim = 1097
    input_history_frames = 10
    frame_offsets = (1, 2, 3, 4, 5)
    control_dt = 0.02
    link_names = COMMAND_LINK_NAMES
    point_names = INPUT_LINK_NAMES

    def __init__(self):
        self.active = [
            self.link_names.index(name) for name in self.point_names
        ]
        self.frames, self.num_links = (
            len(self.frame_offsets),
            len(self.link_names),
        )
        self.width = 9 * self.num_links + 1
        self.task_dim = self.frames * self.width
        positions = torch.zeros(1, self.frames, self.num_links, 3)
        rotations = torch.zeros(1, self.frames, self.num_links, 4)
        rotations[..., 0] = 1
        command = link_observation.build_link_observation(
            link_observation.LinkObservationInput(
                reference_link_pos=positions,
                reference_link_quat_wxyz=rotations,
                reference_root_pos=torch.zeros(1, 3),
                robot_root_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                frame_offsets=torch.tensor(self.frame_offsets),
                link_mask=torch.from_numpy(FIVE_POINT_MASK),
            ),
            root_link_index=0,
            include_error_terms=False,
        )
        self.template = command.flat_reference_command[0]
        self.output_dim = self.template.numel()
        indices = []
        for frame in range(self.frames):
            base = frame * self.width
            for link in self.active:
                indices.extend(range(base + link * 3, base + link * 3 + 3))
                indices.extend(
                    range(
                        base + self.num_links * 3 + link * 6,
                        base + self.num_links * 3 + link * 6 + 6,
                    )
                )
        indices.append(self.task_dim)
        self.learn_indices = torch.tensor(indices, dtype=torch.long)
        self.source_hashes = {
            "reference_schema": sha256(Path(__file__)),
            "link_observation": sha256(Path(link_observation.__file__)),
            "link_actor_packing": sha256(Path(link_actor_packing.__file__)),
        }

    def to_targets(
        self, reference, current_root_position, current_root_rotation_wxyz
    ):
        reference = np.asarray(reference, dtype=np.float64)
        if (
            reference.shape != (self.output_dim,)
            or not np.isfinite(reference).all()
        ):
            raise ValueError(
                f"Expected finite internal reference[{self.output_dim}]"
            )
        fixed = np.ones(self.output_dim, dtype=bool)
        fixed[self.learn_indices.numpy()] = False
        if not np.allclose(
            reference[fixed], self.template.numpy()[fixed], atol=1e-6, rtol=0
        ):
            raise ValueError(
                "Masked reference slots, offsets and validity bits must be preserved"
            )
        task = reference[: self.task_dim].reshape(self.frames, self.width)
        positions = task[:, : self.num_links * 3].reshape(
            self.frames, self.num_links, 3
        )[:, self.active]
        axes = task[:, self.num_links * 3 : -1].reshape(
            self.frames, self.num_links, 6
        )[:, self.active]
        root_quaternion = np.asarray(
            current_root_rotation_wxyz, dtype=np.float64
        )
        root_rotation = Rotation.from_quat(root_quaternion[[1, 2, 3, 0]])
        root_position = np.asarray(
            current_root_position, dtype=np.float64
        ).copy()
        root_position[2] = reference[self.task_dim]
        world_positions = (
            root_rotation.apply(positions.reshape(-1, 3)).reshape(
                positions.shape
            )
            + root_position
        )
        world_rotations = (
            root_rotation
            * Rotation.from_matrix(rotation_from_xz(axes).reshape(-1, 3, 3))
        ).as_quat()
        return FivePointTargets(
            root_position,
            root_quaternion,
            world_positions,
            world_rotations[:, [3, 0, 1, 2]].reshape(self.frames, 5, 4),
            np.asarray(self.frame_offsets),
        )
