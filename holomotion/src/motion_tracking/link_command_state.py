"""Isaac Lab adapter for shared link observations.

Samples actor feedback once per control
observation and reuses it for proprioception and FK. The critic may observe
clean simulation poses; the actor builder never receives robot translation.
"""

from dataclasses import replace

import torch
from omegaconf import OmegaConf

from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from holomotion.src.motion_tracking.link_observation import (
    LinkObservationInput,
    build_link_observation,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
)
from holomotion.src.utils.reference_prefix import resolve_reference_tensor_key


def root_orientation_from_gravity(root_quat, measured_gravity):
    """Apply the smallest tilt change consistent with measured gravity.

    Preserve the actor's original additive gravity observation (including its
    magnitude); only normalize a copy when using it as an orientation sensor.
    """
    root_quat = torch.nn.functional.normalize(root_quat, dim=-1)
    world_down = torch.zeros_like(measured_gravity)
    world_down[..., 2] = -1.0
    true_gravity = quat_apply_wxyz(quat_conjugate_wxyz(root_quat), world_down)
    measured = torch.nn.functional.normalize(measured_gravity, dim=-1)
    cross = torch.cross(measured, true_gravity, dim=-1)
    dot = (measured * true_gravity).sum(dim=-1, keepdim=True)
    correction = torch.cat((1.0 + dot, cross), dim=-1)
    # A deterministic orthogonal axis also handles a hypothetical 180deg tilt.
    axis_seed = torch.nn.functional.one_hot(
        measured.abs().argmin(dim=-1), num_classes=3
    ).to(measured.dtype)
    axis = torch.nn.functional.normalize(
        torch.cross(measured, axis_seed, dim=-1), dim=-1
    )
    opposite = torch.cat((torch.zeros_like(dot), axis), dim=-1)
    correction = torch.where(dot < -0.999999, opposite, correction)
    correction = torch.nn.functional.normalize(correction, dim=-1)
    return torch.nn.functional.normalize(
        quat_multiply_wxyz(root_quat, correction), dim=-1
    )


class LinkCommandObservationAdapter:
    """Per-command cached actor/critic observations and episode-level masks."""

    def __init__(self, command):
        self.command = command
        raw_config = command.cfg.link_observation
        if not OmegaConf.is_config(raw_config):
            raw_config = OmegaConf.create(raw_config)
        self.config = OmegaConf.to_container(raw_config, resolve=True)
        self.link_names = tuple(self.config["link_names"])
        self.include_error_terms = self.config.get("include_error_terms", True)
        if not isinstance(self.include_error_terms, bool):
            raise TypeError("include_error_terms must be a bool")
        self.actor_frame_offsets = tuple(
            int(v) for v in self.config.get("frame_offsets", (1, 2))
        )
        self.critic_frame_offsets = tuple(
            int(v)
            for v in self.config.get(
                "critic_frame_offsets", self.actor_frame_offsets
            )
        )
        for name, offsets in (
            ("frame_offsets", self.actor_frame_offsets),
            ("critic_frame_offsets", self.critic_frame_offsets),
        ):
            if not offsets or min(offsets) < 1:
                raise ValueError(
                    f"Link {name} must contain positive future frame indices"
                )
            if len(set(offsets)) != len(offsets):
                raise ValueError(f"Link {name} must be unique")
        # Backward-compatible alias: frame_offsets remains the deployed
        # actor schema.
        self.frame_offsets = self.actor_frame_offsets
        required_horizon = max(
            max(self.actor_frame_offsets), max(self.critic_frame_offsets)
        )
        if required_horizon > int(command.cfg.n_fut_frames):
            raise ValueError(
                "Continuous reference horizon must cover every actor and "
                "critic link frame offset"
            )
        if float(command.cfg.target_fps) != 50.0:
            raise ValueError(
                "The first link observation schema requires a 50Hz reference"
            )
        if (
            len(set(self.link_names)) != len(self.link_names)
            or "pelvis" not in self.link_names
        ):
            raise ValueError(
                "Unique command links including pelvis are required"
            )
        self.root_link_index = self.link_names.index("pelvis")
        self.reference_indices = [
            command.urdf_body_names.index(n) for n in self.link_names
        ]
        self.sim_indices = [
            command.simulator_body_names.index(n) for n in self.link_names
        ]
        self.fk = None
        if self.include_error_terms:
            self.fk = SharedLinkFK(
                str(command.mcfg.fk_robot_file_path), device=command.device
            )
            self.fk_joint_indices = [
                command.simulator_dof_names.index(n) for n in self.fk.dof_names
            ]
            self.fk_link_indices = [
                self.fk.body_names.index(n) for n in self.link_names
            ]
        self.link_mask = torch.ones(
            command.num_envs, len(self.link_names), device=command.device
        )
        self.mask_config = self.config.get("mask", {})
        self.min_links = int(
            self.mask_config.get("min_active_links", len(self.link_names))
        )
        self.max_links = int(
            self.mask_config.get("max_active_links", len(self.link_names))
        )
        self.full_probability = float(
            self.mask_config.get("full_probability", 1.0)
        )
        if not 1 <= self.min_links <= self.max_links <= len(self.link_names):
            raise ValueError(
                "Mask active-link bounds must be within [1, number of links]"
            )
        if not 0 <= self.full_probability <= 1:
            raise ValueError("Mask full_probability must be in [0,1]")
        self.resample_mask(
            torch.arange(command.num_envs, device=command.device)
        )

    def resample_mask(self, env_ids):
        env_ids = torch.as_tensor(
            env_ids, device=self.command.device, dtype=torch.long
        )
        count = env_ids.numel()
        if count == 0:
            return
        if self.min_links == self.max_links == len(self.link_names):
            self.link_mask[env_ids] = 1.0
        else:
            counts = torch.randint(
                self.min_links,
                self.max_links + 1,
                (count,),
                device=self.command.device,
            )
            full = (
                torch.rand(count, device=self.command.device)
                < self.full_probability
            )
            counts = torch.where(full, len(self.link_names), counts)
            ranks = (
                torch.rand(
                    count, len(self.link_names), device=self.command.device
                )
                .argsort(-1)
                .argsort(-1)
            )
            self.link_mask[env_ids] = (ranks < counts[:, None]).float()
        self.invalidate()

    def set_link_mask(self, mask, env_ids=None):
        """Set explicit simulation controls without resampling motion."""
        if env_ids is None:
            env_ids = torch.arange(
                self.command.num_envs, device=self.command.device
            )
        env_ids = torch.as_tensor(
            env_ids, device=self.command.device, dtype=torch.long
        )
        value = torch.as_tensor(
            mask, device=self.command.device, dtype=torch.float32
        )
        value = value.expand(env_ids.numel(), len(self.link_names))
        if not bool(torch.isfinite(value).all()) or not bool(
            ((value == 0) | (value == 1)).all()
        ):
            raise ValueError("Explicit link mask must contain only 0/1")
        if not bool((value.sum(-1) > 0).all()):
            raise ValueError("At least one link must be active")
        self.link_mask[env_ids] = value
        self.invalidate()

    def invalidate(self):
        cache = getattr(self.command, "_reference_observation_cache", None)
        if cache is not None:
            cache.pop("link_snapshot", None)

    def _noise(self, name, value):
        if self.command._is_evaluating:
            return value
        group = getattr(
            getattr(self.command._env.cfg, "observations", None),
            "unified",
            None,
        )
        if group is not None and not getattr(group, "enable_corruption", True):
            return value
        bounds = self.config.get("proprio_noise", {}).get(name, {})
        low, high = (
            float(bounds.get("n_min", 0.0)),
            float(bounds.get("n_max", 0.0)),
        )
        if low == high == 0.0:
            return value
        return value + torch.empty_like(value).uniform_(low, high)

    def _gather(self, base_key, offsets):
        command = self.command
        key = resolve_reference_tensor_key(
            batch_tensors=command._motion_cache.current_batch.tensors,
            base_key=base_key,
            prefix="ref_",
        )
        return command._motion_cache.gather_tensor_offsets(
            key,
            clip_indices=command._clip_indices,
            frame_indices=command._frame_indices,
            frame_offsets=offsets,
        )

    @torch.no_grad()
    def snapshot(self):
        command = self.command
        cache = getattr(command, "_reference_observation_cache", None)
        if cache is None:
            cache = {}
            command._reference_observation_cache = cache
        if "link_snapshot" in cache:
            return cache["link_snapshot"]
        data = command.robot.data
        root_quat = data.root_quat_w.float()
        gravity = torch.zeros_like(data.root_ang_vel_b)
        gravity[:, 2] = -1.0
        gravity = quat_apply_wxyz(quat_conjugate_wxyz(root_quat), gravity)
        clean = {
            "projected_gravity": gravity,
            "root_ang_vel": data.root_ang_vel_b.float(),
            "dof_pos": (data.joint_pos - data.default_joint_pos).float(),
            "dof_vel": data.joint_vel.float(),
            "last_action": command._env.action_manager.action.float(),
            "root_height": (data.root_pos_w - command._env.scene.env_origins)[
                :, 2:3
            ].float(),
        }
        actor_state = {
            name: self._noise(name, value)
            for name, value in clean.items()
            if name != "root_height"
        }
        measured_root = root_orientation_from_gravity(
            root_quat, actor_state["projected_gravity"]
        )
        actor_local_pos = None
        actor_local_quat = None
        if self.include_error_terms:
            measured_q = actor_state["dof_pos"] + data.default_joint_pos
            fk_pos, fk_quat = self.fk(measured_q[:, self.fk_joint_indices])
            actor_local_pos = fk_pos[:, self.fk_link_indices]
            actor_local_quat = fk_quat[:, self.fk_link_indices]
        ref_root, root_valid = self._gather("root_pos", (0,))

        def build_reference_args(offsets):
            ref_pos, pos_valid = self._gather("rg_pos", offsets)
            ref_quat, rot_valid = self._gather("rb_rot", offsets)
            valid = pos_valid & rot_valid & root_valid
            return dict(
                reference_link_pos=ref_pos[:, :, self.reference_indices],
                reference_link_quat_wxyz=ref_quat[
                    :, :, self.reference_indices
                ][..., [3, 0, 1, 2]],
                reference_root_pos=ref_root[:, 0],
                reference_root_height=ref_root[:, 0, 2:3],
                frame_offsets=torch.tensor(offsets, device=command.device),
                time_valid=valid,
            )

        actor_reference_args = build_reference_args(self.actor_frame_offsets)
        critic_reference_args = (
            actor_reference_args
            if self.critic_frame_offsets == self.actor_frame_offsets
            else build_reference_args(self.critic_frame_offsets)
        )
        actor = build_link_observation(
            LinkObservationInput(
                **actor_reference_args,
                robot_root_quat_wxyz=measured_root,
                robot_link_pos_local=actor_local_pos,
                robot_link_quat_local_wxyz=actor_local_quat,
                link_mask=self.link_mask,
            ),
            root_link_index=self.root_link_index,
            include_error_terms=self.include_error_terms,
        )
        clean_local_pos = None
        clean_local_quat = None
        if self.include_error_terms:
            inv_root = quat_conjugate_wxyz(root_quat)[:, None]
            clean_local_pos = quat_apply_wxyz(
                inv_root,
                data.body_pos_w[:, self.sim_indices]
                - data.root_pos_w[:, None],
            )
            clean_local_quat = quat_multiply_wxyz(
                inv_root, data.body_quat_w[:, self.sim_indices]
            )
        # Keep full clean targets for value estimation under the unchanged
        # full-body reward, while identifying the actor's actual control mask.
        critic = build_link_observation(
            LinkObservationInput(
                **critic_reference_args,
                robot_root_quat_wxyz=root_quat,
                robot_link_pos_local=clean_local_pos,
                robot_link_quat_local_wxyz=clean_local_quat,
            ),
            root_link_index=None,
            include_error_terms=self.include_error_terms,
        )
        critic = replace(critic, link_mask=actor.link_mask)
        result = {
            "actor": actor,
            "critic": critic,
            "actor_state": actor_state,
            "critic_state": clean,
        }
        cache["link_snapshot"] = result
        return result

    def get_term(self, name, actor=True):
        value = getattr(self.snapshot()["actor" if actor else "critic"], name)
        if value is None:
            raise ValueError(f"Link observation term {name!r} is disabled")
        if value.ndim >= 4:
            value = value.flatten(start_dim=2)
        return value

    def get_state(self, name, actor=True):
        return self.snapshot()["actor_state" if actor else "critic_state"][
            name
        ]
