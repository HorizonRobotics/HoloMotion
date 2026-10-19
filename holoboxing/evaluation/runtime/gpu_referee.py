"""GPU physics adapter to the single shared hand/KO referee."""

import numpy as np
from .rules import MatchStateMachine


class CudaReferee(MatchStateMachine):
    def __init__(self, rules, scene):
        super().__init__(rules)
        self.scene = scene
        self.physics = scene._gpu_physics
        if self.physics.slot_map is None:
            self.physics.prepare_contacts()
        self.set_thresholds(
            rules.minimum_contact_force_n, rules.minimum_closing_speed_mps
        )

    def set_thresholds(self, force_n, speed_mps):
        super().set_thresholds(force_n, speed_mps)
        if hasattr(self, "physics"):
            self.physics.hand_thresholds.assign(
                np.asarray([force_n, speed_mps], np.float32)
            )

    def observe_frame(
        self, *, sim_time_s, contacts, root_pos, fallen, physics_samples=None
    ):
        if fallen is None:
            from .fall_rules import fallen_from_scene

            fallen = fallen_from_scene(self.scene)
        samples = (
            physics_samples
            if physics_samples is not None
            else getattr(self.scene, "_hand_contact_samples", None)
        )
        if samples is None:
            raise RuntimeError(
                "Raw physics contacts missing; install hand_rule_runtime before starting the match"
            )
        return super().observe_frame(
            sim_time_s=sim_time_s,
            contacts=contacts,
            root_pos=root_pos,
            fallen=fallen,
            physics_samples=samples,
        )
