# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Actor-only command feature corruption.

Noise is added to the already encoded local position/error and XZ rotation-6D
features, before actor statistical normalization. Rotation noise is unitless,
not an angular perturbation. Reference tensors, critic, timestamps, masks and
reference height remain clean. No tensor is modified in place.
"""

import math
from dataclasses import replace

import torch

FEATURES = ("target_pos", "position_error", "target_rot6d", "rotation_error6d")


def validate_feature_noise(config):
    config = dict(config or {})
    if not isinstance(config.get("enabled", False), bool):
        raise TypeError("command_noise.enabled must be boolean")
    unknown = set(config) - {"enabled", "distribution", "features"}
    if unknown:
        raise ValueError(f"Unknown command noise settings: {sorted(unknown)}")
    if config.get("distribution", "uniform") != "uniform":
        raise ValueError("Only uniform feature noise is supported")
    bounds = config.get("features", {})
    if set(bounds) - set(FEATURES):
        raise ValueError("Noise may only affect link pose/error features")
    for name, interval in bounds.items():
        if len(interval) != 2:
            raise ValueError(f"{name}: expected [minimum, maximum]")
        low, high = map(float, interval)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"{name}: invalid uniform noise bounds")
    return config


def add_link_feature_noise(observation, config, *, enabled=True):
    """Return a corrupted copy; exact identity and RNG no-op when disabled."""
    config = validate_feature_noise(config)
    if not enabled or not config.get("enabled", False):
        return observation
    updates = {}
    valid = observation.effective_mask.bool().unsqueeze(-1)
    for name, interval in config.get("features", {}).items():
        value = getattr(observation, name)
        if value is None:
            continue
        low, high = map(float, interval)
        if low == high == 0.0:
            continue
        noise = torch.empty_like(value).uniform_(low, high)
        updates[name] = torch.where(valid, value + noise, 0.0)
    return replace(observation, **updates) if updates else observation


class FeatureNoiseLinkAdapter:
    """Wrap one existing cached adapter, preserving its feedback, FK and masks.

    Each clean snapshot receives exactly one independent actor corruption.
    Repeated actor terms/router reads share it. A reset or invalidation changes
    the underlying snapshot identity and therefore resamples feature noise.
    """

    def __init__(self, adapter, config):
        self.base = adapter
        self.command = adapter.command
        self.noise_config = validate_feature_noise(config)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @torch.no_grad()
    def snapshot(self):
        clean = self.base.snapshot()
        cache = self.command._reference_observation_cache
        group = getattr(
            getattr(self.command._env.cfg, "observations", None),
            "unified",
            None,
        )
        enabled = not self.command._is_evaluating and (
            group is None or getattr(group, "enable_corruption", True)
        )
        saved = cache.get("link_feature_noise_snapshot")
        if saved is not None and saved[0] is clean and saved[1] == enabled:
            return saved[2]
        actor = add_link_feature_noise(
            clean["actor"], self.noise_config, enabled=enabled
        )
        result = dict(clean, actor=actor)
        cache["link_feature_noise_snapshot"] = (clean, enabled, result)
        return result

    def get_term(self, name, actor=True):
        value = getattr(self.snapshot()["actor" if actor else "critic"], name)
        if value is None:
            raise ValueError(f"Link observation term {name!r} is disabled")
        return value.flatten(start_dim=2) if value.ndim >= 4 else value

    def get_state(self, name, actor=True):
        return self.base.get_state(name, actor=actor)
