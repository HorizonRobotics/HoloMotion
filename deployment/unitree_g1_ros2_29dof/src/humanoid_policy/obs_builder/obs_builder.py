from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Optional

import numpy as np


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
    """History buffer for batched data (batch==1 in our eval/deploy).

    Stores history in oldest->newest order when accessed via .buffer.
    """

    def __init__(self, max_len: int, feat_dim: int):
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self._max_len = int(max_len)
        self._feat_dim = int(feat_dim)
        self._pointer = -1
        self._num_pushes = 0
        self._buffer: np.ndarray = np.zeros(
            (self._max_len, self._feat_dim),
            dtype=np.float32,
        )
        self._flat_buffer = np.zeros(self._max_len * self._feat_dim, dtype=np.float32)

    @property
    def buffer(self) -> np.ndarray:
        """Array of shape [1, max_len, feat_dim], oldest->newest along dim=1."""
        if self._num_pushes == 0:
            raise RuntimeError(
                "Attempting to read from an empty history buffer."
            )
        return self.flatten_oldest_first().reshape(1, self._max_len, self._feat_dim)

    def append(self, data: np.ndarray) -> None:
        """Append one step: data shape [feat_dim] or [1, feat_dim]."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2 and data.shape[0] == 1:
            data = data.reshape(-1)
        if data.ndim != 1 or data.shape[0] != self._feat_dim:
            raise ValueError(
                f"Expected data with shape [{self._feat_dim}], got {tuple(data.shape)}"
            )
        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data
        if self._num_pushes == 0:
            # duplicate first push across entire history for warm start
            self._buffer[:] = data
        self._num_pushes += 1

    def flatten_oldest_first(self) -> np.ndarray:
        """Return a reusable flat buffer in oldest->newest order."""
        if self._num_pushes == 0:
            raise RuntimeError(
                "Attempting to read from an empty history buffer."
            )
        start = (self._pointer + 1) % self._max_len
        if start == 0:
            self._flat_buffer[:] = self._buffer.reshape(-1)
            return self._flat_buffer

        first_len = self._max_len - start
        first_flat_len = first_len * self._feat_dim
        self._flat_buffer[:first_flat_len] = self._buffer[start:].reshape(-1)
        self._flat_buffer[first_flat_len:] = self._buffer[:start].reshape(-1)
        return self._flat_buffer


@dataclass
class _RuntimeTerm:
    name: str
    getter: Any
    scale: float
    scale_buffer: np.ndarray | None
    history_buffer: _CircularBuffer | None
    start: int
    end: int


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
        self._term_entries: List[tuple[str, float, int]] = [
            (
                str(spec["name"]),
                float(spec.get("scale", 1.0)),
                int(spec.get("history_length", 0)),
            )
            for spec in self.term_specs
        ]

        # Buffers are created lazily after first dimension inference
        self._buffers: Dict[str, _CircularBuffer] = {}
        self._history_names: List[str] = []
        self._obs_slices: List[tuple[str, int, int, bool]] = []
        self._obs_buffer: np.ndarray | None = None
        self._obs_batch_buffer: np.ndarray | None = None
        self._term_getters: Dict[str, Any] = {}
        self._runtime_terms: List[_RuntimeTerm] = []

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf._pointer = -1
            buf._num_pushes = 0
            buf._buffer.fill(0.0)
            buf._flat_buffer.fill(0.0)

    def batch_view(self) -> np.ndarray:
        if self._obs_batch_buffer is None:
            if self._obs_buffer is None:
                raise RuntimeError("Observation buffer has not been initialized.")
            self._obs_batch_buffer = self._obs_buffer.reshape(1, -1)
        return self._obs_batch_buffer

    def _compute_term(
        self,
        name: str,
    ) -> np.ndarray:
        # Prefer evaluator-provided methods; no legacy fallbacks
        meth = self._term_getters.get(name)
        if meth is None and self.evaluator is not None:
            meth = getattr(self.evaluator, f"_get_obs_{name}", None)
            if callable(meth):
                self._term_getters[name] = meth
        if meth is not None:
            out = meth()
            if isinstance(out, np.ndarray) and out.dtype == np.float32 and out.ndim == 1:
                return out
            return np.asarray(out, dtype=np.float32).reshape(-1)
        raise ValueError(
            f"Unknown observation term '{name}' or evaluator method missing."
        )

    def _resolve_term_getter(self, name: str):
        meth = self._term_getters.get(name)
        if meth is None and self.evaluator is not None:
            meth = getattr(self.evaluator, f"_get_obs_{name}", None)
            if callable(meth):
                self._term_getters[name] = meth
        if meth is None:
            raise ValueError(
                f"Unknown observation term '{name}' or evaluator method missing."
            )
        return meth

    @staticmethod
    def _coerce_term_output(out: Any) -> np.ndarray:
        if isinstance(out, np.ndarray) and out.dtype == np.float32 and out.ndim == 1:
            return out
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _compute_runtime_term(self, term: _RuntimeTerm) -> np.ndarray:
        value = self._coerce_term_output(term.getter())
        if term.scale == 1.0:
            return value
        if term.scale_buffer is None:
            return value * term.scale
        np.multiply(value, term.scale, out=term.scale_buffer)
        return term.scale_buffer

    def _build_policy_obs_fast(self) -> np.ndarray:
        obs = self._obs_buffer
        if obs is None:
            raise RuntimeError("Observation buffer has not been initialized.")

        for term in self._runtime_terms:
            value = self._compute_runtime_term(term)
            if term.history_buffer is None:
                obs[term.start : term.end] = value
            else:
                term.history_buffer.append(value)
                obs[term.start : term.end] = (
                    term.history_buffer.flatten_oldest_first()
                )
        return obs

    def build_policy_obs(self) -> np.ndarray:
        """Append one step using evaluator-provided observation terms and return flattened obs."""
        if self._obs_buffer is not None and self._runtime_terms:
            return self._build_policy_obs_fast()

        # Compute per-term outputs
        values: Dict[str, np.ndarray] = {}
        for name, scale, _hist_len in self._term_entries:
            value = self._compute_term(name)
            if scale != 1.0:
                value = value * scale
            values[name] = value

        # Lazily initialize buffers with inferred feature dims
        if self._obs_buffer is None:
            offset = 0
            obs_slices: List[tuple[str, int, int, bool]] = []
            history_names: List[str] = []
            runtime_terms: List[_RuntimeTerm] = []
            for name, _scale, hist_len in self._term_entries:
                feat_dim = int(values[name].shape[0])
                scale = float(_scale)
                scale_buffer = (
                    np.zeros(feat_dim, dtype=np.float32)
                    if scale != 1.0
                    else None
                )
                if hist_len <= 0:
                    obs_slices.append((name, offset, offset + feat_dim, False))
                    runtime_terms.append(
                        _RuntimeTerm(
                            name=name,
                            getter=self._resolve_term_getter(name),
                            scale=scale,
                            scale_buffer=scale_buffer,
                            history_buffer=None,
                            start=offset,
                            end=offset + feat_dim,
                        )
                    )
                    offset += feat_dim
                    continue
                history_buffer = _CircularBuffer(
                    hist_len,
                    feat_dim,
                )
                self._buffers[name] = history_buffer
                dim = hist_len * feat_dim
                obs_slices.append((name, offset, offset + dim, True))
                runtime_terms.append(
                    _RuntimeTerm(
                        name=name,
                        getter=self._resolve_term_getter(name),
                        scale=scale,
                        scale_buffer=scale_buffer,
                        history_buffer=history_buffer,
                        start=offset,
                        end=offset + dim,
                    )
                )
                history_names.append(name)
                offset += dim
            self._history_names = history_names
            self._obs_slices = obs_slices
            self._obs_buffer = np.zeros(offset, dtype=np.float32)
            self._obs_batch_buffer = self._obs_buffer.reshape(1, -1)
            self._runtime_terms = runtime_terms

        # Append current step to buffers (skip terms without history)
        for name in self._history_names:
            self._buffers[name].append(values[name])

        # Assemble flat list according to term ordering and history flatten rules
        obs = self._obs_buffer
        if obs is None:
            obs = np.zeros(0, dtype=np.float32)
            self._obs_buffer = obs
        for name, start, end, has_history in self._obs_slices:
            if has_history:
                obs[start:end] = self._buffers[name].flatten_oldest_first()
            else:
                obs[start:end] = values[name].reshape(-1)

        if obs.shape[0] == 0:
            return obs

        return obs
