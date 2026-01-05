import numpy as np
import torch

from typing import Dict, List, Sequence, Any, Optional


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

    def __init__(self, max_len: int, feat_dim: int):
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self._max_len = int(max_len)
        self._feat_dim = int(feat_dim)
        self._pointer = -1
        self._num_pushes = 0
        self._buffer: torch.Tensor = torch.zeros(
            (self._max_len, 1, self._feat_dim),
            dtype=torch.float32,
            device="cpu",
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
        evaluator: Optional[Any] = None,
        obs_policy_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dof_names_onnx: List[str] = list(dof_names_onnx)
        self.num_actions: int = len(self.dof_names_onnx)
        self.evaluator = evaluator
        self.obs_policy_cfg = obs_policy_cfg

        if default_angles_onnx.shape[0] != self.num_actions:
            raise ValueError(
                "default_angles_onnx length must match num actions"
            )
        self.default_angles_onnx = default_angles_onnx.astype(np.float32)
        self.default_angles_dict: Dict[str, float] = {
            name: float(self.default_angles_onnx[idx])
            for idx, name in enumerate(self.dof_names_onnx)
        }

        # Build observation schema from config if provided
        self.term_specs: List[Dict[str, Any]] = []

        for term_dict in self.obs_policy_cfg["atomic_obs_list"]:
            for name, cfg in term_dict.items():
                term_dict = {**cfg}
                term_dict["name"] = name
                self.term_specs.append(term_dict)

        # Buffers are created lazily after first dimension inference
        self._buffers: Dict[str, _CircularBuffer] = {}

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf._pointer = -1
            buf._num_pushes = 0
            buf._buffer.zero_()

    def _compute_term(
        self,
        name: str,
    ) -> np.ndarray:
        # Prefer evaluator-provided methods; no legacy fallbacks
        if self.evaluator is not None:
            meth = getattr(self.evaluator, f"_get_obs_{name}", None)
            if callable(meth):
                out = meth()
                return np.asarray(out, dtype=np.float32).reshape(-1)
        raise ValueError(
            f"Unknown observation term '{name}' or evaluator method missing."
        )

    def build_policy_obs(self) -> np.ndarray:
        """Append one step using evaluator-provided observation terms and return flattened obs."""
        # Compute per-term outputs
        values: Dict[str, np.ndarray] = {}
        for spec in self.term_specs:
            name = spec["name"]
            scale = float(spec.get("scale", 1.0))
            values[name] = self._compute_term(name) * scale

        # Lazily initialize buffers with inferred feature dims
        if len(self._buffers) == 0:
            for spec in self.term_specs:
                name = spec["name"]
                hist_len = int(spec.get("history_length", 0))
                if hist_len <= 0:
                    continue
                feat_dim = int(values[name].reshape(-1).shape[0])
                self._buffers[name] = _CircularBuffer(
                    hist_len,
                    feat_dim,
                )

        # Append current step to buffers (skip terms without history)
        for spec in self.term_specs:
            name = spec["name"]
            if name in self._buffers:
                item = torch.as_tensor(
                    values[name],
                    dtype=torch.float32,
                    device="cpu",
                )[None, :]
                self._buffers[name].append(item)

        # Assemble flat list according to term ordering and history flatten rules
        flat_list: List[np.ndarray] = []
        for spec in self.term_specs:
            name = spec["name"]
            if name in self._buffers:
                buf = self._buffers[name].buffer[0]  # [hist, feat]
                arr = buf.reshape(-1).detach().cpu().numpy()
                flat_list.append(arr.astype(np.float32))
            else:
                # no history -> use computed value directly
                flat_list.append(values[name].reshape(-1).astype(np.float32))

        if len(flat_list) == 0:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(flat_list, axis=0).astype(np.float32)
