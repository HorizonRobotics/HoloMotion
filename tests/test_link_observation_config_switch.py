"""CPU checks for the link observation and agent module schema."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.modules.agent_modules import TensorDictAssembler
from holomotion.src.motion_tracking.link_actor_packing import (
    LINK_CURRENT_TERMS,
    link_actor_atomic_terms,
    link_actor_layout,
    pack_link_actor_observation,
)
from holomotion.src.motion_tracking.link_observation import (
    DEFAULT_COMMAND_LINK_NAMES,
    LinkObservation,
)
from holomotion.src.utils.config import setup_hydra_resolvers

ROOT = Path(__file__).resolve().parents[1]
ERROR_NAMES = {
    f"{role}_link_{name}"
    for role in ("actor", "critic")
    for name in ("position_error", "rotation_error6d")
}


def link_config(
    *, include_errors: bool, frames: int, critic_frames: int | None = None
):
    """Compose link schema groups on the existing training base."""
    if critic_frames is None:
        critic_frames = frames
    setup_hydra_resolvers()
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        config = compose(
            overrides=[
                "+training=train_base",
                "+algo=ppo_tf",
                "+robot=unitree/G1/29dof/29dof_training_isaaclab",
                "+env=motion_tracking",
                "+env/terminations=termination_motion_tracking",
                "+env/observations=motion_tracking/obs_motion_tracking_link",
                "+env/rewards=motion_tracking/rew_motion_tracking",
                "+env/domain_randomization=domain_rand_medium",
                "+env/terrain=isaaclab_rough",
                "+modules=motion_tracking/motion_tracking_link_tf-moe",
            ],
        )
    OmegaConf.update(
        config,
        "link_observation",
        {
            "enabled": True,
            "schema_version": "g1_link_xz6d_v1",
            "link_names": list(DEFAULT_COMMAND_LINK_NAMES),
            "frame_offsets": list(range(1, frames + 1)),
            "critic_frame_offsets": list(range(1, critic_frames + 1)),
            "include_error_terms": include_errors,
        },
        force_add=True,
    )
    return config


def example_observations(config):
    """Construct atomics with the shapes produced by the observation group."""
    links = len(config.link_observation.link_names)
    actor_frames = len(config.link_observation.frame_offsets)
    critic_frames = len(config.link_observation.critic_frame_offsets)
    dofs = len(config.robot.dof_names)
    bodies = int(config.robot.num_bodies)
    feedback = {
        "projected_gravity": 3,
        "root_ang_vel": 3,
        "dof_pos": dofs,
        "dof_vel": dofs,
        "last_action": dofs,
        "root_height": 1,
    }
    command = {
        "reference_height": 1,
        "link_mask": links,
        "target_pos": 3 * links,
        "position_error": 3 * links,
        "target_rot6d": 6 * links,
        "rotation_error6d": 6 * links,
        "time_offsets": 1,
    }
    privileged = {
        "critic_rel_robot_root_lin_vel": 3,
        "critic_rel_robot_root_ang_vel": 3,
        "critic_global_robot_bodylink_lin_vel_flat": 3 * bodies,
        "critic_global_robot_bodylink_ang_vel_flat": 3 * bodies,
        "critic_root_rel_robot_bodylink_pos_flat": 3 * bodies,
        "critic_root_rel_robot_bodylink_rot_mat_flat": 6 * bodies,
        "critic_global_anchor_diff": 9,
    }
    future = {
        "target_pos",
        "position_error",
        "target_rot6d",
        "rotation_error6d",
        "time_offsets",
    }
    terms = {}
    for row in config.obs.obs_groups.unified.atomic_obs_list:
        name, spec = next(iter(row.items()))
        if spec.func == "link_feedback":
            shape = (2, feedback[spec.params.name])
        elif spec.func == "link_command_term":
            frames = actor_frames if spec.params.actor else critic_frames
            width = (
                frames
                if spec.params.name == "time_valid"
                else command[spec.params.name]
            )
            shape = (
                (2, frames, width)
                if spec.params.name in future
                else (2, width)
            )
        else:
            shape = (2, privileged[name])
        if spec.get("history_length", 0):
            shape = (shape[0], 1, *shape[1:])
        terms[name] = torch.zeros(shape)
    return TensorDict(
        {"unified": TensorDict(terms, batch_size=[2])}, batch_size=[2]
    )


def command_example(*, include_errors: bool, frames: int):
    links = len(DEFAULT_COMMAND_LINK_NAMES)
    positions = torch.arange(
        2 * frames * links * 3, dtype=torch.float32
    ).reshape(2, frames, links, 3)
    rotations = torch.arange(
        2 * frames * links * 6, dtype=torch.float32
    ).reshape(2, frames, links, 6)
    link_mask = torch.ones(2, links)
    link_mask[0, -1] = 0
    time_valid = torch.ones(2, frames)
    time_valid[1, -1] = 0
    return LinkObservation(
        target_pos=positions + 100,
        position_error=positions + 10_000 if include_errors else None,
        target_rot6d=rotations + 20_000,
        rotation_error6d=rotations + 30_000 if include_errors else None,
        time_offsets=torch.arange(1, frames + 1, dtype=torch.float32)
        .reshape(1, frames, 1)
        .expand(2, -1, -1)
        + 40_000,
        link_mask=link_mask,
        time_valid=time_valid,
        reference_height=torch.tensor([[0.8], [0.9]]),
        effective_mask=time_valid[:, :, None] * link_mask[:, None, :],
    )


@pytest.mark.parametrize(
    "include_errors,frames,actor_dim,critic_dim,router_dim",
    [
        (True, 2, 616, 1079, 271),
        (False, 2, 364, 827, 271),
        (True, 5, 1378, 1841, 655),
        (False, 5, 748, 1211, 655),
    ],
)
def test_hydra_schema_and_term_major_packing(
    include_errors, frames, actor_dim, critic_dim, router_dim
):
    config = link_config(include_errors=include_errors, frames=frames)
    assert len(config.link_observation.link_names) == 14
    assert len(config.robot.dof_names) == 29
    assert config.robot.num_bodies == 30
    assert config.obs.n_fut_frames == frames
    assert config.obs.critic_n_fut_frames == frames
    assert config.env.config.commands.ref_motion.params.n_fut_frames == frames

    atomics = [
        next(iter(row))
        for row in config.obs.obs_groups.unified.atomic_obs_list
    ]
    assert len(atomics) == len(set(atomics))
    assert (set(atomics) & ERROR_NAMES) == (
        ERROR_NAMES if include_errors else set()
    )
    assert config.obs.obs_groups.unified.concatenate_terms is False

    actor_schema = config.modules.actor.obs_schema
    layout = link_actor_layout(config.link_observation, actor_schema)
    assert layout["actor_dim"] == actor_dim
    observations = example_observations(config)
    actor_assembler = TensorDictAssembler(actor_schema)
    critic_assembler = TensorDictAssembler(config.modules.critic.obs_schema)
    assert actor_assembler(observations).shape == (2, actor_dim)
    assert critic_assembler(observations).shape == (2, critic_dim)
    assert (
        sum(
            observations[tuple(name.split("/"))].numel() // 2
            for name in config.modules.actor.router_obs_terms
        )
        == router_dim
    )

    command = command_example(include_errors=include_errors, frames=frames)
    proprioception = torch.arange(2 * 93, dtype=torch.float32).reshape(2, 93)
    for name, value in zip(
        LINK_CURRENT_TERMS[:5],
        torch.split(proprioception, (3, 3, 29, 29, 29), dim=-1),
        strict=True,
    ):
        observations["unified", name] = value[:, None, :]
    for name, value in link_actor_atomic_terms(
        command, include_error_terms=include_errors
    ).items():
        observations["unified", name] = (
            value[:, None, :] if name in LINK_CURRENT_TERMS else value
        )
    torch.testing.assert_close(
        actor_assembler(observations),
        pack_link_actor_observation(
            proprioception, command, include_error_terms=include_errors
        ),
    )


@pytest.mark.parametrize(
    "include_errors,actor_dim,critic_dim,future_width",
    [(True, 616, 1333, 253), (False, 364, 955, 127)],
)
def test_critic_horizon_can_exceed_actor_horizon(
    include_errors, actor_dim, critic_dim, future_width
):
    config = link_config(
        include_errors=include_errors, frames=2, critic_frames=3
    )
    assert config.obs.n_fut_frames == 2
    assert config.obs.critic_n_fut_frames == 3
    assert config.env.config.commands.ref_motion.params.n_fut_frames == 3
    actor_schema = config.modules.actor.obs_schema
    critic_schema = config.modules.critic.obs_schema
    assert actor_schema.flattened_obs_fut.seq_len == 2
    assert critic_schema.flattened_obs_fut.seq_len == 3

    observations = example_observations(config)
    assert TensorDictAssembler(actor_schema)(observations).shape == (
        2,
        actor_dim,
    )
    assert TensorDictAssembler(critic_schema)(observations).shape == (
        2,
        critic_dim,
    )
    future_schema = {"flattened_obs_fut": critic_schema.flattened_obs_fut}
    assert TensorDictAssembler(future_schema, output_mode="seq")(
        observations
    ).shape == (2, 3, future_width)


@pytest.mark.parametrize("longer_critic", [False, True])
def test_training_recipe_wires_link_config_to_command(longer_critic):
    setup_hydra_resolvers()
    overrides = (
        ["link_observation.critic_frame_offsets=[1,2,3,4,5,6]"]
        if longer_critic
        else []
    )
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        config = compose(
            config_name="training/motion_tracking/motion_tracking_v2_0_0",
            overrides=overrides,
        )
    command_link = (
        config.env.config.commands.ref_motion.params.link_observation
    )
    assert command_link.enabled is True
    assert OmegaConf.to_container(command_link, resolve=True) == (
        OmegaConf.to_container(config.link_observation, resolve=True)
    )
    critic_frames = 6 if longer_critic else 5
    assert config.obs.n_fut_frames == 5
    assert config.obs.critic_n_fut_frames == critic_frames
    assert config.env.config.commands.ref_motion.params.n_fut_frames == (
        critic_frames
    )
    assert config.modules.actor.obs_schema.flattened_obs_fut.seq_len == 5
    assert (
        config.modules.critic.obs_schema.flattened_obs_fut.seq_len
        == critic_frames
    )
    observations = example_observations(config)
    assert TensorDictAssembler(config.modules.actor.obs_schema)(
        observations
    ).shape == (2, 1378)
    assert TensorDictAssembler(config.modules.critic.obs_schema)(
        observations
    ).shape == (2, 2095 if longer_critic else 1841)


def test_shared_motion_command_requires_explicit_link_schema():
    setup_hydra_resolvers()
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        config = compose(overrides=["+env=motion_tracking"])
    assert (
        OmegaConf.select(
            config,
            "env.config.commands.ref_motion.params.link_observation",
            default=None,
        )
        is None
    )


def test_error_switch_preserves_non_error_observations():
    enabled = link_config(include_errors=True, frames=2)
    disabled = link_config(include_errors=False, frames=2)
    active_rows = OmegaConf.to_container(
        enabled.obs.obs_groups.unified.atomic_obs_list, resolve=True
    )
    without_errors = [
        row for row in active_rows if next(iter(row)) not in ERROR_NAMES
    ]
    assert (
        OmegaConf.to_container(
            disabled.obs.obs_groups.unified.atomic_obs_list, resolve=True
        )
        == without_errors
    )
    for key in ("robot", "rewards", "domain_rand", "algo"):
        assert OmegaConf.to_container(enabled[key], resolve=False) == (
            OmegaConf.to_container(disabled[key], resolve=False)
        )
