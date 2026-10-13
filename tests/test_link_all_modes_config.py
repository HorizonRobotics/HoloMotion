"""CPU composition checks for the default AllModes training recipe."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from holomotion.src.motion_tracking.link_actor_packing import link_actor_layout
from holomotion.src.motion_tracking.link_mode_mask import ModeMaskSampler
from holomotion.src.utils.config import setup_hydra_resolvers

CONFIG_DIR = Path(__file__).resolve().parents[1] / "holomotion/config"
BASE = "training/motion_tracking/motion_tracking_v2_0_0"
ALL_MODES = BASE
EXPECTED_MODES = {
    "Pelvis-1": ("pelvis",),
    "Keypoint-3": ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    "Keypoint-5": (
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
    "UpperBody-6": (
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    "UpperBody-Mobile-7": (
        "pelvis",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
    "WholeBody-14": (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
}


def _compose(name, overrides=()):
    setup_hydra_resolvers()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def test_default_recipe_matches_mode_table():
    modes = _compose(BASE)

    assert modes.env._target_ == (
        "holomotion.src.env.mode_masked_motion_tracking.ModeMaskedMotionTrackingEnv"
    )
    mask = modes.link_observation.mask
    assert mask.strategy == "modes"
    assert mask.mode_weights is None
    assert mask.clip_mode_labels is None
    assert mask.report_every_resamples == 1000
    assert (
        mask.min_active_links,
        mask.max_active_links,
        mask.full_probability,
    ) == (
        14,
        14,
        1.0,
    )
    actual = OmegaConf.to_container(mask.mode_candidates, resolve=True)
    assert list(actual) == list(EXPECTED_MODES)
    assert {
        name: tuple(links) for name, links in actual.items()
    } == EXPECTED_MODES
    assert [len(links) for links in actual.values()] == [
        1,
        3,
        5,
        6,
        7,
        14,
    ]
    assert set(actual["WholeBody-14"]) == set(
        modes.link_observation.link_names
    )
    sampler = ModeMaskSampler(
        modes.link_observation.link_names,
        OmegaConf.to_container(mask, resolve=True),
    )
    assert sampler.table.shape == (6, 14)
    assert sampler.probabilities.tolist() == pytest.approx([1 / 6] * 6)
    assert sampler.description()["sampling"] == "uniform_mode"


def test_all_modes_five_frame_observation_contract():
    modes = _compose(ALL_MODES)
    link = modes.link_observation
    assert list(link.frame_offsets) == [1, 2, 3, 4, 5]
    assert list(link.critic_frame_offsets) == [1, 2, 3, 4, 5]
    assert modes.obs.n_fut_frames == 5
    assert modes.obs.critic_n_fut_frames == 5
    assert modes.env.config.commands.ref_motion.params.n_fut_frames == 5
    assert OmegaConf.to_container(
        modes.env.config.commands.ref_motion.params.link_observation,
        resolve=True,
    ) == OmegaConf.to_container(link, resolve=True)
    assert modes.modules.actor.obs_schema.flattened_obs_fut.seq_len == 5
    assert modes.modules.critic.obs_schema.flattened_obs_fut.seq_len == 5
    assert (
        link_actor_layout(link, modes.modules.actor.obs_schema)["actor_dim"]
        == 1378
    )


def test_all_modes_respects_existing_error_switch():
    modes = _compose(ALL_MODES, ["link_observation.include_error_terms=false"])
    assert modes.link_observation.include_error_terms is False
    assert (
        modes.env.config.commands.ref_motion.params.link_observation.include_error_terms
        is False
    )
    assert (
        link_actor_layout(
            modes.link_observation, modes.modules.actor.obs_schema
        )["actor_dim"]
        == 748
    )
    for role in ("actor", "critic"):
        terms = modes.modules[role].obs_schema.flattened_obs_fut.terms
        assert all("position_error" not in term for term in terms)
        assert all("rotation_error6d" not in term for term in terms)


def test_default_recipe_requires_user_source_probabilities():
    config = _compose(BASE)
    assert OmegaConf.is_missing(
        config.robot.motion.source_sampling, "probabilities"
    )
    assert config.algo.config.sampling_strategy == "source_adaptive"
    assert config.robot.motion.source_adaptive.enabled is True
    assert config.robot.motion.failure_replay.enabled is True
    assert config.robot.motion.failure_replay.replay_start_mode == "next_frame"


def test_default_recipe_rough_terrain_rewards_cache_and_optional_dumps():
    config = _compose(BASE)
    assert config.num_envs == 8192
    assert config.robot.motion.cache.max_num_clips == 8192
    assert config.terrain.terrain_type == "generator"
    assert config.terrain.generator.sub_terrains.rough.type == "random_uniform"
    assert list(config.terrain.generator.sub_terrains.rough.noise_range) == [
        0.0,
        0.04,
    ]
    assert config.algo.config.num_learning_iterations == 18001
    assert config.algo.config.save_interval == 500
    assert config.algo.config.distributed_update.lr_scale.mode == "none"
    assert config.robot.motion.dump_sampled_motion_keys is False
    assert config.algo.config.curriculum.dump_whole_window_scores_json is False
    assert config.rewards.is_alive.weight == 1.0
    assert config.rewards.root_pos_xy_tracking_exp.weight == 1.0
    assert config.rewards.root_pos_xy_tracking_exp.params.std == 0.5
    assert config.rewards.root_pos_rel_z_tracking_exp.weight == 0.5
    assert config.rewards.root_pos_rel_z_tracking_exp.params.std == 0.3
    assert config.env.config.tracking_extensions.enabled is True
    assert config.env.config.tracking_extensions.command_noise.enabled is True
    assert (
        config.env.config.tracking_extensions.rewards.anchor_height_tracking_exp.weight
        == 0.5
    )
    assert (
        config.env.config.tracking_extensions.rewards.global_body_rotation_tracking_exp.weight
        == 1.0
    )
