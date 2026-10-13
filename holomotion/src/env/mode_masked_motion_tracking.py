"""Opt-in named link-mask modes for a motion-tracking environment.

The base environment keeps its full-link mask and reward behavior. This wrapper
replaces only the command-mask sampler after the command exists.
"""

import json
from pathlib import Path

from omegaconf import OmegaConf

from holomotion.src.env.motion_tracking import MotionTrackingEnv
from holomotion.src.motion_tracking.link_mode_mask import ModeMaskLinkAdapter


class ModeMaskedMotionTrackingEnv(MotionTrackingEnv):
    """Sample one named control mode per environment and motion assignment."""

    def _init_isaaclab_env(self):
        raw_extensions = self.config.get("tracking_extensions")
        extensions_enabled = False
        if raw_extensions:
            extension_config = (
                OmegaConf.to_container(raw_extensions, resolve=True)
                if OmegaConf.is_config(raw_extensions)
                else raw_extensions
            )
            if not isinstance(extension_config, dict):
                raise TypeError("tracking_extensions must be a mapping")
            extensions_enabled = extension_config.get("enabled", True)
            if not isinstance(extensions_enabled, bool):
                raise TypeError("tracking_extensions.enabled must be boolean")

        result = super()._init_isaaclab_env()
        command = self._env.command_manager.get_term("ref_motion")
        base = getattr(command, "_link_observation_adapter", None)
        if base is None:
            raise ValueError("Mode masks require enabled link observations")

        raw_config = command.cfg.link_observation
        config = (
            OmegaConf.to_container(raw_config, resolve=True)
            if OmegaConf.is_config(raw_config)
            else dict(raw_config)
        )
        adapter = ModeMaskLinkAdapter(base, config["mask"])
        command._link_observation_adapter = adapter

        rank = (
            int(self.accelerator.process_index)
            if self.accelerator is not None
            else int(self.config.get("process_id", 0))
        )
        receipt = {
            **adapter.sampler.description(),
            **adapter.statistics(),
            "process_id": rank,
            "actor_features_masked": True,
            "critic_targets_and_rewards": "complete_clean_reference",
            "resample_on": "existing_command_reset_and_motion_reassignment",
            "future_frames_share_mode": True,
            "frame_offsets": list(base.actor_frame_offsets),
            "actor_frame_offsets": list(base.actor_frame_offsets),
            "critic_frame_offsets": list(base.critic_frame_offsets),
        }
        log_dir = Path(self.log_dir)
        (log_dir / f"link_command_modes_rank{rank}.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        if rank == 0:
            (log_dir / "link_command_modes.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        print("LINK_COMMAND_MODES " + json.dumps(receipt), flush=True)
        if extensions_enabled:
            from holomotion.src.env.link_augmented_motion_tracking import (
                install_link_tracking_extensions,
            )

            install_link_tracking_extensions(self)
        return result
