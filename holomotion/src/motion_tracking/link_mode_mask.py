"""Named command modes for link observations, independent of the simulator.

A mode selects a complete, configured set of links. No arbitrary link subset
is generated. The underlying adapter still owns feature masking and caching.
"""

import json
import math
from collections.abc import Mapping

import torch


def resolve_control_links(
    link_names, *, active_link_names=None, mask_mode="all14"
):
    """Return canonical active names and a plain boolean mask in schema order.

    Shared by named training modes and fixed evaluation controls. This helper
    neither samples modes nor depends on an evaluation configuration object.
    """
    if isinstance(link_names, str):
        raise ValueError("Link schema must be a sequence of names")
    names = list(link_names)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("Command link names must be nonempty and unique")
    if len(set(names)) != len(names):
        raise ValueError("Command link names must be nonempty and unique")
    if mask_mode == "all14":
        if active_link_names is not None and list(active_link_names) != names:
            raise ValueError(
                "all14 cannot specify a different active-link set"
            )
        selected = names
    elif mask_mode == "custom":
        if active_link_names is None or isinstance(active_link_names, str):
            raise ValueError(
                "custom requires an explicit active_link_names list"
            )
        selected = list(active_link_names)
    else:
        raise ValueError("Use all14 or custom with explicit active_link_names")
    if not selected or not all(
        isinstance(name, str) and name for name in selected
    ):
        raise ValueError(
            "Active links must be a nonempty unique list of names"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(
            "Active links must be a nonempty unique list of names"
        )
    selected = set(selected)
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f"unknown command links: {sorted(unknown)}")
    mask = [name in selected for name in names]
    return [name for name in names if name in selected], mask


class ModeMaskSampler:
    """Compile modes and sample one binary-mask row per environment."""

    def __init__(self, link_names, config, device="cpu"):
        names, _ = resolve_control_links(link_names)
        self.link_names = tuple(names)
        self.config = dict(config)
        if self.config.get("strategy") != "modes":
            raise ValueError(
                "Named mode sampling requires mask.strategy=modes"
            )
        candidates = self.config.get("mode_candidates")
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError(
                "mode_candidates must be a nonempty name-to-links mapping"
            )
        self.mode_names = tuple(candidates)
        if any(
            not isinstance(name, str) or not name for name in self.mode_names
        ):
            raise ValueError("Mode names must be nonempty strings")
        table = torch.zeros(
            len(candidates), len(self.link_names), device=device
        )
        for index, (_name, active) in enumerate(candidates.items()):
            _, mask = resolve_control_links(
                self.link_names, active_link_names=active, mask_mode="custom"
            )
            table[index] = torch.tensor(mask, dtype=table.dtype, device=device)
        self.table = table
        weights = self.config.get("mode_weights")
        self.uniform = weights is None
        if weights is None:
            values = [1.0] * len(self.mode_names)
        else:
            if not isinstance(weights, Mapping) or set(weights) != set(
                self.mode_names
            ):
                raise ValueError(
                    "mode_weights must name every configured mode exactly once"
                )
            values = [float(weights[name]) for name in self.mode_names]
            if (
                any(not math.isfinite(value) or value < 0 for value in values)
                or sum(values) <= 0
            ):
                raise ValueError(
                    "Mode weights must be finite, nonnegative, and nonzero"
                )
        # Normalize before float32 conversion to avoid overflow.
        peak = max(values)
        values = [value / peak for value in values]
        self.probabilities = torch.tensor(
            values, device=device, dtype=torch.float32
        )
        self.probabilities /= self.probabilities.sum()

    def sample(self, count):
        if not isinstance(count, int) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        if self.uniform:
            indices = torch.randint(
                len(self.mode_names), (count,), device=self.table.device
            )
        elif count:
            indices = torch.multinomial(
                self.probabilities, count, replacement=True
            )
        else:
            indices = torch.empty(
                0, device=self.table.device, dtype=torch.long
            )
        return indices, self.table[indices].clone()

    def mode_index(self, name_or_id):
        if isinstance(name_or_id, str):
            if name_or_id not in self.mode_names:
                raise ValueError(f"Unknown command mode: {name_or_id}")
            return self.mode_names.index(name_or_id)
        if isinstance(name_or_id, bool) or not isinstance(name_or_id, int):
            raise TypeError("Mode must be a configured name or integer index")
        if not 0 <= name_or_id < len(self.mode_names):
            raise ValueError("Mode index is outside the configured table")
        return name_or_id

    def description(self):
        return {
            "mode_names": list(self.mode_names),
            "link_names": list(self.link_names),
            "mode_table": self.table.cpu().tolist(),
            "probabilities": self.probabilities.cpu().tolist(),
            "sampling": "uniform_mode" if self.uniform else "weighted_mode",
        }


class ModeMaskLinkAdapter:
    """Delegate observations, replacing only command-mask selection.

    All reset/motion reassignment call sites use resample_mask on this wrapper.
    A selected mode is shared by every future target of the environment.
    """

    def __init__(self, adapter, config):
        self.base = adapter
        self.command = adapter.command
        self.sampler = ModeMaskSampler(
            adapter.link_names, config, device=self.command.device
        )
        self.sampled_mode_ids = torch.full(
            (self.command.num_envs,),
            -1,
            device=self.command.device,
            dtype=torch.long,
        )
        self.resample_calls = 0
        self.report_every = int(config.get("report_every_resamples", 0))
        if self.report_every < 0:
            raise ValueError("report_every_resamples must be nonnegative")
        self.resample_mask(
            torch.arange(self.command.num_envs, device=self.command.device)
        )

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _env_ids(self, env_ids):
        if env_ids is None:
            return torch.arange(
                self.command.num_envs, device=self.command.device
            )
        ids = torch.as_tensor(
            env_ids, device=self.command.device, dtype=torch.long
        )
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        return ids

    @torch.no_grad()
    def resample_mask(self, env_ids):
        ids = self._env_ids(env_ids)
        if not ids.numel():
            return
        modes, masks = self.sampler.sample(ids.numel())
        # Rows are already validated binary masks; avoid per-reset GPU-to-CPU
        # validation synchronizations in the public explicit-control setter.
        self.base.link_mask[ids] = masks
        self.base.invalidate()
        self.sampled_mode_ids[ids] = modes
        self.resample_calls += 1
        if self.report_every and self.resample_calls % self.report_every == 0:
            print(
                "LINK_MODE_SAMPLE " + json.dumps(self.statistics()), flush=True
            )

    @torch.no_grad()
    def set_mode(self, name_or_id, env_ids=None):
        ids = self._env_ids(env_ids)
        mode = self.sampler.mode_index(name_or_id)
        self.base.set_link_mask(self.sampler.table[mode], env_ids=ids)
        self.sampled_mode_ids[ids] = mode

    @torch.no_grad()
    def set_link_mask(self, mask, env_ids=None):
        """Custom evaluation masks have mode ID -1 when unmatched."""
        ids = self._env_ids(env_ids)
        self.base.set_link_mask(mask, env_ids=ids)
        value = self.base.link_mask[ids]
        matches = (value[:, None, :] == self.sampler.table[None, :, :]).all(-1)
        self.sampled_mode_ids[ids] = torch.where(
            matches.any(-1), matches.long().argmax(-1), -1
        )

    def statistics(self):
        selected = self.sampled_mode_ids
        counts = torch.bincount(
            selected[selected >= 0], minlength=len(self.sampler.mode_names)
        )
        return {
            "mode_names": list(self.sampler.mode_names),
            "mode_counts": counts.cpu().tolist(),
            "custom_count": int((selected < 0).sum().item()),
            "num_envs": self.command.num_envs,
            "resample_calls": self.resample_calls,
        }
