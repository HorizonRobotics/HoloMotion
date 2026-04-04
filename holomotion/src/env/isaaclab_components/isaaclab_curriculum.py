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



from isaaclab.envs import ManagerBasedRLEnv
import torch
from typing import Sequence
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as isaaclab_mdp
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, Callable, Dict
from loguru import logger
from .isaaclab_domain_rand import DomainRandFunctions


def _completion_rate_curriculum_get_level(
    env,
    *,
    term_tag: str = "default",
    metric_key: str = "Metrics/ref_motion/Task/Completion_Rate",
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    apply_on_startup: bool = True,
    startup_level: int = 0,
    state_prefix: str = "_cr_curr",
):
    base_env = getattr(env, "unwrapped", env)

    level_key = f"{state_prefix}_level"
    startup_key = f"{state_prefix}_startup_applied"
    last_up_key = f"{state_prefix}_last_upgrade_step"
    level_start_step_key = f"{state_prefix}_level_start_step"

    if not hasattr(base_env, level_key):
        setattr(base_env, level_key, -1)
    if not hasattr(base_env, startup_key):
        setattr(base_env, startup_key, False)
    if not hasattr(base_env, last_up_key):
        setattr(base_env, last_up_key, -(10**18))
    if not hasattr(base_env, level_start_step_key):
        setattr(base_env, level_start_step_key, 0)

    step = int(
        getattr(
            base_env,
            "common_step_counter",
            getattr(env, "common_step_counter", 0),
        )
    )

    def _get_completion_stats():
        metrics = getattr(base_env, "metrics", None)
        if isinstance(metrics, dict) and metric_key in metrics:
            val = metrics[metric_key]
            val = float(val.item()) if hasattr(val, "item") else float(val)
            return val, step
        return None

    def _thr_for_next(next_level: int) -> float:
        if not cr_thresholds:
            return 1.0
        idx = max(0, min(next_level - 1, len(cr_thresholds) - 1))
        return float(cr_thresholds[idx])

    stats = _get_completion_stats()
    cur_level = int(getattr(base_env, level_key))
    changed = False

    # -------- startup init --------
    if apply_on_startup and not bool(getattr(base_env, startup_key)):
        init_level = int(max(0, min(int(startup_level), int(num_updates))))
        setattr(base_env, level_key, max(cur_level, init_level))
        setattr(base_env, startup_key, True)
        setattr(base_env, last_up_key, step)
        setattr(base_env, level_start_step_key, step)
        cur_level = int(getattr(base_env, level_key))
        changed = True

    # -------- level upgrade --------
    if cur_level < int(num_updates):
        if stats is not None:
            cr_val, _ = stats
            level_start_step = int(getattr(base_env, level_start_step_key))
            stayed_steps = int(step - level_start_step)

            cooldown_ok = True
            if int(cooldown_steps) > 0:
                last_up = int(getattr(base_env, last_up_key))
                cooldown_ok = (step - last_up) >= int(cooldown_steps)

            if cooldown_ok and stayed_steps >= int(min_steps_per_level):
                next_level = min(cur_level + 1, int(num_updates))
                thr = _thr_for_next(next_level)
                if float(cr_val) >= float(thr):
                    setattr(base_env, level_key, next_level)
                    setattr(base_env, last_up_key, step)
                    setattr(base_env, level_start_step_key, step)
                    cur_level = next_level
                    changed = True

    applied_key = (
        f"{state_prefix}_applied_{str(term_tag)}_level_{int(cur_level)}"
    )
    if not hasattr(base_env, applied_key):
        setattr(base_env, applied_key, False)

    already_applied = bool(getattr(base_env, applied_key))
    need_apply = bool(changed) or (not already_applied)

    return int(cur_level), stats, bool(changed), bool(need_apply)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device)
                + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device)
                + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device)
                + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def robot_friction_range_by_completion_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
    static_friction_target=(0.3, 1.6),
    dynamic_friction_target=(0.3, 1.2),
    enforce_dynamic_le_static: bool = True,
    asset_name: str = "robot",
    body_names: str = ".*",
    restitution_range=(0.0, 0.5),
    num_buckets: int = 64,
    anchor_quantile: float = 0.5,
    min_expand_frac: float = 0.0,
):
    base_env = getattr(env, "unwrapped", env)

    def _quantile(lo: float, hi: float, q: float) -> float:
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        q = float(max(0.0, min(1.0, q)))
        return lo + (hi - lo) * q

    def _compute_ranges(level: int):
        level_i = int(max(0, min(level, int(num_updates))))
        frac = (
            1.0
            if int(num_updates) <= 0
            else (level_i / float(int(num_updates)))
        )

        s_lo_t, s_hi_t = map(float, static_friction_target)
        d_lo_t, d_hi_t = map(float, dynamic_friction_target)
        s_lo_t, s_hi_t = min(s_lo_t, s_hi_t), max(s_lo_t, s_hi_t)
        d_lo_t, d_hi_t = min(d_lo_t, d_hi_t), max(d_lo_t, d_hi_t)

        s_anchor = _quantile(s_lo_t, s_hi_t, anchor_quantile)
        d_anchor = _quantile(d_lo_t, d_hi_t, anchor_quantile)

        eps = float(min_expand_frac)
        band = eps + (1.0 - eps) * float(max(frac, 0.0))

        s_lo = s_anchor - (s_anchor - s_lo_t) * band
        s_hi = s_anchor + (s_hi_t - s_anchor) * band
        d_lo = d_anchor - (d_anchor - d_lo_t) * band
        d_hi = d_anchor + (d_hi_t - d_anchor) * band

        s_lo, s_hi = min(s_lo, s_hi), max(s_lo, s_hi)
        d_lo, d_hi = min(d_lo, d_hi), max(d_lo, d_hi)

        if enforce_dynamic_le_static:
            d_hi = min(d_hi, s_hi)
            d_lo = min(d_lo, d_hi)

        return (
            float(s_lo),
            float(s_hi),
            float(d_lo),
            float(d_hi),
            float(frac),
            int(level_i),
        )

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag="fric",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    if not need_apply:
        return float(level)

    s_lo, s_hi, d_lo, d_hi, frac, level_i = _compute_ranges(int(level))

    DomainRandFunctions._get_dr_rigid_body_material(
        env=env,
        env_ids=None,
        asset_name=asset_name,
        body_names=body_names,
        static_friction_range=(s_lo, s_hi),
        dynamic_friction_range=(d_lo, d_hi),
        restitution_range=tuple(restitution_range),
        num_buckets=int(num_buckets),
    )

    setattr(base_env, f"{state_prefix}_applied_fric_level_{int(level)}", True)
    return float(level)


def rigid_body_com_by_completion_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
    asset_name: str = "robot",
    body_names: str = "torso_link",
    com_range_target: dict = {
        "x": (-0.025, 0.025),
        "y": (-0.05, 0.05),
        "z": (-0.05, 0.05),
    },
    anchor_quantile: float = 0.5,
    min_expand_frac: float = 0.0,
):
    base_env = getattr(env, "unwrapped", env)

    def _quantile(lo: float, hi: float, q: float) -> float:
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        q = float(max(0.0, min(1.0, q)))
        return lo + (hi - lo) * q

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag="com",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    if not need_apply:
        return float(level)

    level_i = int(max(0, min(int(level), int(num_updates))))
    frac = (
        1.0 if int(num_updates) <= 0 else (level_i / float(int(num_updates)))
    )
    band = float(min_expand_frac) + (1.0 - float(min_expand_frac)) * float(
        max(frac, 0.0)
    )

    com_range = {}
    for axis, (lo_t, hi_t) in com_range_target.items():
        lo_t, hi_t = float(lo_t), float(hi_t)
        lo_t, hi_t = min(lo_t, hi_t), max(lo_t, hi_t)

        anchor = _quantile(lo_t, hi_t, anchor_quantile)
        lo = anchor - (anchor - lo_t) * band
        hi = anchor + (hi_t - anchor) * band
        com_range[axis] = (float(min(lo, hi)), float(max(lo, hi)))

    DomainRandFunctions._get_dr_rigid_body_com(
        env=env,
        env_ids=None,
        com_range=com_range,
        asset_name=asset_name,
        body_names=body_names,
    )

    setattr(base_env, f"{state_prefix}_applied_com_level_{int(level)}", True)
    return float(level)


def default_dof_pos_bias_by_completion_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
    asset_name: str = "robot",
    joint_names: list[str] = (".*"),
    pos_distribution_params_target: tuple[float, float] = (-0.01, 0.01),
    operation: str = "add",
    distribution: str = "uniform",
    anchor_quantile: float = 0.5,
    min_expand_frac: float = 0.0,
):
    base_env = getattr(env, "unwrapped", env)

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag="dof",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    if not need_apply:
        return float(level)

    def _quantile(lo: float, hi: float, q: float) -> float:
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        q = float(max(0.0, min(1.0, q)))
        return lo + (hi - lo) * q

    lo_t, hi_t = map(float, pos_distribution_params_target)
    lo_t, hi_t = min(lo_t, hi_t), max(lo_t, hi_t)

    level_i = int(max(0, min(int(level), int(num_updates))))
    frac = (
        1.0 if int(num_updates) <= 0 else (level_i / float(int(num_updates)))
    )
    band = float(min_expand_frac) + (1.0 - float(min_expand_frac)) * float(
        max(frac, 0.0)
    )

    anchor = _quantile(lo_t, hi_t, anchor_quantile)
    lo = anchor - (anchor - lo_t) * band
    hi = anchor + (hi_t - anchor) * band
    lo, hi = float(min(lo, hi)), float(max(lo, hi))

    DomainRandFunctions._get_dr_default_dof_pos_bias(
        env=env,
        env_ids=None,
        asset_name=asset_name,
        joint_names=joint_names,
        pos_distribution_params=(lo, hi),
        operation=operation,
        distribution=distribution,
    )

    setattr(base_env, f"{state_prefix}_applied_dof_level_{int(level)}", True)
    return float(level)


def push_by_setting_velocity_range_by_completion_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value,
    *,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
    velocity_range_target: dict = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    },
    anchor_quantile: float = 0.5,
    min_expand_frac: float = 0.0,
):
    base_env = getattr(env, "unwrapped", env)

    def _quantile(lo: float, hi: float, q: float) -> float:
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        q = float(max(0.0, min(1.0, q)))
        return lo + (hi - lo) * q

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag="push",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    if not need_apply:
        return isaaclab_mdp.modify_term_cfg.NO_CHANGE

    level_i = int(max(0, min(int(level), int(num_updates))))
    frac = (
        1.0 if int(num_updates) <= 0 else (level_i / float(int(num_updates)))
    )
    band = float(min_expand_frac) + (1.0 - float(min_expand_frac)) * float(
        max(frac, 0.0)
    )

    new_params = dict(old_value) if isinstance(old_value, dict) else old_value

    current_velocity_range = {}
    for axis, (lo_t, hi_t) in velocity_range_target.items():
        lo_t, hi_t = float(lo_t), float(hi_t)
        lo_t, hi_t = min(lo_t, hi_t), max(lo_t, hi_t)

        anchor = _quantile(lo_t, hi_t, anchor_quantile)
        lo = anchor - (anchor - lo_t) * band
        hi = anchor + (hi_t - anchor) * band
        current_velocity_range[axis] = [float(min(lo, hi)), float(max(lo, hi))]

    if isinstance(new_params, dict) or hasattr(new_params, "__setitem__"):
        if isinstance(new_params, dict):
            new_params = dict(new_params)
            new_params["velocity_range"] = current_velocity_range
        else:
            new_params["velocity_range"] = current_velocity_range
    else:
        setattr(new_params, "velocity_range", current_velocity_range)

    setattr(base_env, f"{state_prefix}_applied_push_level_{int(level)}", True)
    return new_params


def randomize_actuator_gains_by_completion_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
    asset_name: str = "robot",
    body_names: str = ".*",
    stiffness_distribution_params_target: tuple[float, float] = (0.9, 1.1),
    damping_distribution_params_target: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
    distribution: str = "uniform",
    anchor_quantile: float = 0.5,
    min_expand_frac: float = 0.0,
):
    base_env = getattr(env, "unwrapped", env)

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag="gains",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    if not need_apply:
        return float(level)

    def _quantile(lo: float, hi: float, q: float) -> float:
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        q = float(max(0.0, min(1.0, q)))
        return lo + (hi - lo) * q

    level_i = int(max(0, min(int(level), int(num_updates))))
    frac = (
        1.0 if int(num_updates) <= 0 else (level_i / float(int(num_updates)))
    )
    band = float(min_expand_frac) + (1.0 - float(min_expand_frac)) * float(
        max(frac, 0.0)
    )

    # stiffness
    ks_lo_t, ks_hi_t = map(float, stiffness_distribution_params_target)
    ks_lo_t, ks_hi_t = min(ks_lo_t, ks_hi_t), max(ks_lo_t, ks_hi_t)
    ks_anchor = _quantile(ks_lo_t, ks_hi_t, anchor_quantile)
    ks_lo = ks_anchor - (ks_anchor - ks_lo_t) * band
    ks_hi = ks_anchor + (ks_hi_t - ks_anchor) * band
    ks_lo, ks_hi = float(min(ks_lo, ks_hi)), float(max(ks_lo, ks_hi))

    # damping
    kd_lo_t, kd_hi_t = map(float, damping_distribution_params_target)
    kd_lo_t, kd_hi_t = min(kd_lo_t, kd_hi_t), max(kd_lo_t, kd_hi_t)
    kd_anchor = _quantile(kd_lo_t, kd_hi_t, anchor_quantile)
    kd_lo = kd_anchor - (kd_anchor - kd_lo_t) * band
    kd_hi = kd_anchor + (kd_hi_t - kd_anchor) * band
    kd_lo, kd_hi = float(min(kd_lo, kd_hi)), float(max(kd_lo, kd_hi))

    DomainRandFunctions._get_dr_randomize_actuator_gains(
        env=env,
        env_ids=None,
        asset_name=asset_name,
        body_names=body_names,
        stiffness_distribution_params=(ks_lo, ks_hi),
        damping_distribution_params=(kd_lo, kd_hi),
        operation=operation,
        distribution=distribution,
    )

    setattr(base_env, f"{state_prefix}_applied_gains_level_{int(level)}", True)
    return float(level)


def reward_term_weight_by_completion_rate(
    env,
    env_ids,
    *,
    reward_term_name: str,
    final_weight: float,
    start_scale: float = 0.1,
    num_updates: int = 5,
    cr_thresholds=(0.10, 0.20, 0.28, 0.34, 0.40),
    min_steps_per_level: int = 300,
    cooldown_steps: int = 0,
    state_prefix: str = "_cr_curr",
):
    base_env = getattr(env, "unwrapped", env)

    level, stats, changed, need_apply = _completion_rate_curriculum_get_level(
        env,
        term_tag=f"reward_{reward_term_name}",
        num_updates=num_updates,
        cr_thresholds=cr_thresholds,
        min_steps_per_level=min_steps_per_level,
        cooldown_steps=cooldown_steps,
        state_prefix=state_prefix,
    )

    progress = 1.0 if num_updates <= 0 else float(level) / float(num_updates)
    start_weight = float(final_weight) * float(start_scale)
    new_weight = start_weight + progress * (float(final_weight) - start_weight)

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    old_weight = float(reward_cfg.weight)

    if not need_apply:
        return float(level)

    reward_cfg.weight = float(new_weight)
    env.reward_manager.set_term_cfg(reward_term_name, reward_cfg)

    setattr(
        base_env,
        f"{state_prefix}_reward_weight_{reward_term_name}",
        float(new_weight),
    )
    setattr(
        base_env,
        f"{state_prefix}_applied_reward_{reward_term_name}_level_{int(level)}",
        True,
    )
    return float(level)


@configclass
class CurriculumCfg:
    pass


def build_curriculum_config(curriculum_config_dict: dict) -> CurriculumCfg:
    """
    Build IsaacLab-compatible CurriculumCfg from a config dictionary.
    """
    if isinstance(curriculum_config_dict, (DictConfig, ListConfig)):
        curriculum_config_dict = OmegaConf.to_container(
            curriculum_config_dict, resolve=True
        )

    curriculum_cfg = CurriculumCfg()
    cfg_dict: Dict[str, Any] = dict(curriculum_config_dict or {})

    def _resolve_callable(name: Any) -> Callable:
        if callable(name):
            return name

        if isinstance(name, str) and name.startswith("isaaclab_mdp."):
            name = name.split(".", 1)[1]

        fn = globals().get(name)
        if callable(fn):
            return fn

        fn = getattr(isaaclab_mdp, name, None)
        if callable(fn):
            return fn

        if hasattr(isaaclab_mdp, "curriculums"):
            fn = getattr(isaaclab_mdp.curriculums, name, None)
            if callable(fn):
                return fn

        raise ValueError(f"Unknown curriculum function: {name}")

    def _normalize_modify_params(x: Any) -> Any:
        if isinstance(x, list):
            # many configs express tuples as YAML lists
            return tuple(_normalize_modify_params(v) for v in x)
        if isinstance(x, dict):
            return {k: _normalize_modify_params(v) for k, v in x.items()}
        return x

    def _fix_params(params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params or {})

        if "modify_fn" in params and isinstance(
            params["modify_fn"], (str, Callable)
        ):
            params["modify_fn"] = _resolve_callable(params["modify_fn"])

        if "modify_params" in params and isinstance(
            params["modify_params"], dict
        ):
            params["modify_params"] = _normalize_modify_params(
                params["modify_params"]
            )

        return params

    global_enabled = cfg_dict.pop("enabled", True)
    if not global_enabled:
        return curriculum_cfg

    for term_name, term_cfg in cfg_dict.items():
        if term_cfg is None:
            term_cfg = {}

        if isinstance(term_cfg, bool):
            if not term_cfg:
                continue
            term_cfg = {}

        if not isinstance(term_cfg, dict):
            raise TypeError(
                f"[build_curriculum_config] term '{term_name}' must be a dict/bool/None, got {type(term_cfg)}"
            )

        if not term_cfg.get("enabled", True):
            continue

        func_field = term_cfg.get("func", None)
        if func_field is None:
            func = _resolve_callable(term_name)
        else:
            func = _resolve_callable(func_field)

        params = _fix_params(term_cfg.get("params", {}) or {})

        setattr(
            curriculum_cfg,
            term_name,
            CurriculumTermCfg(func=func, params=params),
        )

    return curriculum_cfg
