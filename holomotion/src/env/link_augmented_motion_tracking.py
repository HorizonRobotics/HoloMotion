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

"""Opt-in link training extensions without changing the base environment.

Adds generic world-frame tracking rewards and actor command feature noise.
The existing managers/adapter are fully initialized first; the reward manager
is rebuilt once before any rollout, preserving all configured base terms.
The effective Isaac Lab configuration is dumped again for provenance.
"""

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from holomotion.src.motion_tracking.global_tracking_rewards import (
    _body_indices,
    anchor_height_tracking_exp,
    global_body_rotation_tracking_exp,
)
from holomotion.src.motion_tracking.link_feature_noise import (
    FeatureNoiseLinkAdapter,
    validate_feature_noise,
)

REWARDS = {
    "anchor_height_tracking_exp": anchor_height_tracking_exp,
    "global_body_rotation_tracking_exp": global_body_rotation_tracking_exp,
}


def install_link_tracking_extensions(env):
    """Install rewards and actor noise around an already configured adapter.

    Mask wrappers must be installed first so their binding guards run before
    noise snapshots are read. Reward terms always access clean command state.
    """
    # Keep the default, disabled AllModes import path independent of Isaac Lab.
    from isaaclab.managers import RewardManager, RewardTermCfg
    from isaaclab.utils.io import dump_yaml

    extra = env.config.get("tracking_extensions", {})
    if OmegaConf.is_config(extra):
        extra = OmegaConf.to_container(extra, resolve=True)
    if not isinstance(extra, Mapping):
        raise TypeError("tracking_extensions must be a mapping")
    extra = dict(extra)
    command_name = extra.get("command_name", "ref_motion")
    if not isinstance(command_name, str) or not command_name:
        raise ValueError("tracking_extensions.command_name must be nonempty")
    command = env._env.command_manager.get_term(command_name)
    adapter = getattr(command, "_link_observation_adapter", None)
    if adapter is None:
        raise ValueError(
            "Link training extensions require an enabled "
            "link observation adapter"
        )
    noise_config = validate_feature_noise(extra.get("command_noise", {}))
    noise_enabled = noise_config.get("enabled", False)
    noise_adapter = (
        FeatureNoiseLinkAdapter(adapter, noise_config)
        if noise_enabled
        else adapter
    )
    terms = extra.get("rewards", {})
    if not isinstance(terms, Mapping):
        raise TypeError("tracking_extensions.rewards must be a mapping")
    reward_cfg = copy.deepcopy(env._env.cfg.rewards) if terms else None
    validated_weights = {}
    for name, spec in terms.items():
        if name not in REWARDS:
            raise ValueError(f"Unknown link tracking extension reward: {name}")
        if not isinstance(spec, Mapping):
            raise TypeError(f"{name}: reward specification must be a mapping")
        if hasattr(reward_cfg, name):
            raise ValueError(
                f"Extension cannot overwrite existing reward: {name}"
            )
        weight = float(spec["weight"])
        if not math.isfinite(weight):
            raise ValueError(f"{name}: reward weight must be finite")
        raw_params = spec.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise TypeError(f"{name}: params must be a mapping")
        params = dict(raw_params)
        std = float(params["std"])
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"{name}: std must be finite and positive")
        if name == "anchor_height_tracking_exp":
            _body_indices(command, [params.get("anchor_body_name", "pelvis")])
        else:
            _body_indices(command, params["body_names"])
        params.setdefault("ref_motion_command_name", command_name)
        setattr(
            reward_cfg,
            name,
            RewardTermCfg(func=REWARDS[name], weight=weight, params=params),
        )
        validated_weights[name] = weight
    if terms:
        previous_reward_cfg = env._env.cfg.rewards
        env._env.cfg.rewards = reward_cfg
        try:
            reward_manager = RewardManager(reward_cfg, env._env)
            actual = set(reward_manager.active_terms)
            for name, weight in validated_weights.items():
                if weight != 0.0 and name not in actual:
                    raise RuntimeError(
                        f"Configured extension reward is inactive: {name}"
                    )
        except Exception:
            env._env.cfg.rewards = previous_reward_cfg
            raise
        env._env.reward_manager = reward_manager
    else:
        actual = set(env._env.reward_manager.active_terms)
    if noise_enabled:
        command._link_observation_adapter = noise_adapter
    rank = (
        int(env.accelerator.process_index)
        if env.accelerator is not None
        else int(env.config.get("process_id", 0))
    )
    dump_yaml(
        str(Path(env.log_dir) / f"isaaclab_env_cfg_rank{rank}.yaml"),
        env._env.cfg,
    )
    # Every parent constructor must finish its legacy canonical dump first.
    if env.accelerator is not None:
        env.accelerator.wait_for_everyone()
    if rank == 0:
        dump_yaml(
            str(Path(env.log_dir) / "isaaclab_env_cfg.yaml"), env._env.cfg
        )
    receipt = {
        "command_name": command_name,
        "command_noise": noise_config,
        "rewards": terms,
        "active_reward_terms": sorted(actual),
        "actor_command_noise_only": noise_enabled,
        "critic_and_reward_reference_clean": True,
        "noise_before_actor_normalization": noise_enabled,
        "num_envs_per_rank": env._env.num_envs,
        "process_id": rank,
    }
    path = Path(env.log_dir) / f"link_training_extensions_rank{rank}.json"
    path.write_text(json.dumps(receipt, indent=2))
    if rank == 0:
        (Path(env.log_dir) / "link_training_extensions.json").write_text(
            json.dumps(receipt, indent=2)
        )
    print("LINK_TRAINING_EXTENSIONS " + json.dumps(receipt), flush=True)
