import numpy as np
import torch

from typing import Dict, List, Sequence


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Calculate gravity orientation from quaternion.

    Args:
        quaternion: Array-like [w, x, y, z]

    Returns:
        np.ndarray of shape (3,) representing gravity projection.
    """
    qw = float(quaternion[0])
    qx = float(quaternion[1])
    qy = float(quaternion[2])
    qz = float(quaternion[3])

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


class _CircularBuffer:
    """History buffer for batched tensor data (batch==1 in our eval/deploy).

    Stores history in oldest->newest order when accessed via .buffer.
    """

    def __init__(self, max_len: int, feat_dim: int, device: str):
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self._max_len = int(max_len)
        self._feat_dim = int(feat_dim)
        self._device = device
        self._pointer = -1
        self._num_pushes = 0
        self._buffer: torch.Tensor = torch.zeros(
            (self._max_len, 1, self._feat_dim),
            dtype=torch.float32,
            device=self._device,
        )

    @property
    def buffer(self) -> torch.Tensor:
        """Tensor of shape [1, max_len, feat_dim], oldest->newest along dim=1."""
        if self._num_pushes == 0:
            raise RuntimeError(
                "Attempting to read from an empty history buffer."
            )
        # roll such that oldest is at index=0 along the history axis
        rolled = torch.roll(
            self._buffer, shifts=self._max_len - self._pointer - 1, dims=0
        )
        return torch.transpose(rolled, 0, 1)  # [1, max_len, feat]

    def append(self, data: torch.Tensor) -> None:
        """Append one step: data shape [1, feat_dim] on the configured device."""
        if (
            data.ndim != 2
            or data.shape[0] != 1
            or data.shape[1] != self._feat_dim
        ):
            raise ValueError(
                f"Expected data with shape [1, {self._feat_dim}], got {tuple(data.shape)}"
            )
        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data
        if self._num_pushes == 0:
            # duplicate first push across entire history for warm start
            self._buffer[:] = data
        self._num_pushes += 1


class PolicyObsBuilder:
    """Builds policy observations from Unitree lowstate with temporal history.

    Designed to be shared between MuJoCo sim2sim evaluation and ROS2 deployment.
    History management is internal and produces a flattened vector of size
    sum_i(context_length * feat_i) across the configured observation items.

    Supports two command modes:
    - "motion_tracking": uses reference motion states
    - "velocity_tracking": uses velocity commands [vx, vy, vyaw]
    """

    def __init__(
        self,
        dof_names_onnx: Sequence[str],
        default_angles_onnx: np.ndarray,
        context_length: int,
        device: str,
        command_mode: str = "motion_tracking",
    ) -> None:
        self.dof_names_onnx: List[str] = list(dof_names_onnx)
        self.num_actions: int = len(self.dof_names_onnx)
        self.device: str = device
        self.context_length: int = int(context_length)
        self.command_mode: str = command_mode

        if self.command_mode not in ["motion_tracking", "velocity_tracking"]:
            raise ValueError(
                f"command_mode must be 'motion_tracking' or 'velocity_tracking', got {self.command_mode}"
            )

        if default_angles_onnx.shape[0] != self.num_actions:
            raise ValueError(
                "default_angles_onnx length must match num actions"
            )
        self.default_angles_onnx = default_angles_onnx.astype(np.float32)
        self.default_angles_dict: Dict[str, float] = {
            name: float(self.default_angles_onnx[idx])
            for idx, name in enumerate(self.dof_names_onnx)
        }

        # Build observation schema based on command mode
        if self.command_mode == "motion_tracking":
            self.obs_order: List[str] = [
                "ref_motion_states",
                "projected_gravity",
                "rel_robot_root_ang_vel",
                "dof_pos",
                "dof_vel",
                "last_action",
            ]
            obs_dims_map = {
                "ref_motion_states": 2 * self.num_actions,
                "projected_gravity": 3,
                "rel_robot_root_ang_vel": 3,
                "dof_pos": self.num_actions,
                "dof_vel": self.num_actions,
                "last_action": self.num_actions,
            }
        else:  # velocity_tracking
            self.obs_order: List[str] = [
                "velocity_command",
                "projected_gravity",
                "rel_robot_root_ang_vel",
                "dof_pos",
                "dof_vel",
                "last_action",
            ]
            obs_dims_map = {
                "velocity_command": 4,  # [vx, vy, vyaw]
                "projected_gravity": 3,
                "rel_robot_root_ang_vel": 3,
                "dof_pos": self.num_actions,
                "dof_vel": self.num_actions,
                "last_action": self.num_actions,
            }

        if self.context_length > 0:
            self._buffers: Dict[str, _CircularBuffer] = {
                key: _CircularBuffer(
                    self.context_length, obs_dims_map[key], self.device
                )
                for key in self.obs_order
            }

    def reset(self) -> None:
        if self.context_length > 0:
            for buf in self._buffers.values():
                buf._pointer = -1
                buf._num_pushes = 0
                buf._buffer.zero_()

    def update_from_lowstate(
        self,
        q_by_name: Dict[str, float],
        dq_by_name: Dict[str, float],
        imu_quat: np.ndarray,
        imu_gyro: np.ndarray,
        last_action: np.ndarray,
        ref_dof_pos_mu: np.ndarray = None,
        ref_dof_vel_mu: np.ndarray = None,
        ref_to_onnx: Sequence[int] = None,
        velocity_command: np.ndarray = None,
    ) -> np.ndarray:
        """Append one step using Unitree lowstate and command (motion or velocity).

        Args:
            q_by_name: Current joint positions keyed by joint name (MuJoCo order names)
            dq_by_name: Current joint velocities keyed by joint name
            imu_quat: IMU quaternion [w, x, y, z]
            imu_gyro: IMU body angular velocity [x, y, z]
            last_action: Previous policy action in ONNX joint order [n_dofs]
            ref_dof_pos_mu: Reference joint positions in MuJoCo order [n_dofs] (motion_tracking only)
            ref_dof_vel_mu: Reference joint velocities in MuJoCo order [n_dofs] (motion_tracking only)
            ref_to_onnx: Indices mapping MuJoCo->ONNX order (motion_tracking only)
            velocity_command: Velocity command [vx, vy, vyaw] (velocity_tracking only)

        Returns:
            Flattened observation vector with history: np.ndarray [context_length * feat_dim]
        """
        dof_pos_onnx = np.array(
            [
                q_by_name[name] - self.default_angles_dict[name]
                for name in self.dof_names_onnx
            ],
            dtype=np.float32,
        )
        dof_vel_onnx = np.array(
            [dq_by_name[name] for name in self.dof_names_onnx],
            dtype=np.float32,
        )

        obs_items = {
            "projected_gravity": get_gravity_orientation(imu_quat).astype(
                np.float32
            ),
            "rel_robot_root_ang_vel": np.asarray(imu_gyro, dtype=np.float32),
            "dof_pos": dof_pos_onnx.astype(np.float32),
            "dof_vel": dof_vel_onnx.astype(np.float32),
            "last_action": last_action.astype(np.float32),
        }

        if self.context_length > 0:
            if self.command_mode == "motion_tracking":
                if (
                    ref_dof_pos_mu is None
                    or ref_dof_vel_mu is None
                    or ref_to_onnx is None
                ):
                    raise ValueError(
                        "ref_dof_pos_mu, ref_dof_vel_mu, and ref_to_onnx must be provided for motion_tracking mode"
                    )
                ref_dof_pos_onnx = ref_dof_pos_mu[ref_to_onnx].astype(
                    np.float32
                )
                ref_dof_vel_onnx = ref_dof_vel_mu[ref_to_onnx].astype(
                    np.float32
                )
                obs_items["ref_motion_states"] = np.concatenate(
                    [ref_dof_pos_onnx, ref_dof_vel_onnx]
                ).astype(np.float32)
            else:  # velocity_tracking
                if velocity_command is None:
                    raise ValueError(
                        "velocity_command must be provided for velocity_tracking mode"
                    )
                extended_velo_command = np.zeros(4, dtype=np.float32)
                extended_velo_command[1:] = velocity_command
                extended_velo_command[0] = np.linalg.norm(velocity_command) > 0.1
                obs_items["velocity_command"] = extended_velo_command.astype(
                    np.float32
                )

            for key in self.obs_order:
                item = torch.as_tensor(
                    obs_items[key], dtype=torch.float32, device=self.device
                )[None, :]
                self._buffers[key].append(item)

            flat_list: List[np.ndarray] = []
            for key in self.obs_order:
                buf = self._buffers[key].buffer[0]  # [max_len, feat]
                flat_list.append(buf.reshape(-1).detach().cpu().numpy())
            hist_obs = np.concatenate(flat_list, axis=0)
            return hist_obs.astype(np.float32)
        else:
            flat_list: List[np.ndarray] = []
            if self.command_mode == "motion_tracking":
                if (
                    ref_dof_pos_mu is None
                    or ref_dof_vel_mu is None
                    or ref_to_onnx is None
                ):
                    raise ValueError(
                        "ref_dof_pos_mu, ref_dof_vel_mu, and ref_to_onnx must be provided for motion_tracking mode"
                    )
                ref_dof_pos_onnx = ref_dof_pos_mu[ref_to_onnx].astype(
                    np.float32
                )
                ref_dof_vel_onnx = ref_dof_vel_mu[ref_to_onnx].astype(
                    np.float32
                )
                obs_items["ref_motion_states"] = np.concatenate(
                    [ref_dof_pos_onnx, ref_dof_vel_onnx]
                ).astype(np.float32)
                flat_list.append(obs_items["ref_motion_states"])
            else:
                flat_list.append(obs_items["velocity_command"])
            flat_list.append(obs_items["projected_gravity"])
            flat_list.append(obs_items["rel_robot_root_ang_vel"])
            flat_list.append(obs_items["dof_pos"])
            flat_list.append(obs_items["dof_vel"])
            flat_list.append(obs_items["last_action"])
            return np.concatenate(flat_list, axis=0).astype(np.float32)
