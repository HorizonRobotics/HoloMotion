import unittest
import numpy as np
from holoboxing.evaluation.runtime.rules import MatchStateMachine
from holoboxing.evaluation.runtime.hand_hit_rules import (
    HandHitGate,
    FORCE_LEVELS,
    SPEED_LEVELS,
)


def contact(hand="left_hand", body="head", force=4.0, speed=0.4, actor=0):
    return dict(
        attacker=actor,
        defender=1 - actor,
        attacking_body=hand,
        target_body=body,
        peak_force=force,
        closing_speed=speed,
        physical_contact=True,
        position=[0, 0, 1],
    )


class HandRulesTest(unittest.TestCase):
    def setUp(self):
        self.ref = MatchStateMachine(scoring_mode="force_speed")

    def step(self, time, *contacts):
        return self.ref.observe_frame(
            sim_time_s=time,
            contacts=contacts,
            root_pos=[[-1, 0, 0.8], [1, 0, 0.8]],
            fallen=[False, False],
        )

    def test_feet_do_not_score(self):
        self.assertEqual(
            self.step(
                0.005, contact("left_foot"), contact("right_ankle")
            ).score,
            (0, 0),
        )
        self.assertEqual(
            [c["reason"] for c in self.ref.contact_diagnostics],
            ["foot_not_scored", "foot_not_scored"],
        )

    def test_hands_independent_and_target_switch_not_new_hit(self):
        self.assertEqual(self.step(0.005, contact()).score, (1, 0))
        self.assertEqual(
            self.step(
                0.01, contact(body="pelvis"), contact("right_hand")
            ).score,
            (2, 0),
        )
        self.assertEqual(
            self.step(
                0.015,
                contact(body="chest"),
                contact("right_hand", body="head"),
            ).score,
            (2, 0),
        )

    def test_99ms_not_enough_exact_100ms_allowed(self):
        self.step(0.005, contact())
        self.step(0.010)
        self.assertEqual(self.step(0.109, contact()).score, (1, 0))
        self.step(0.114)
        self.assertEqual(self.step(0.214, contact()).score, (2, 0))

    def test_low_force_or_speed_contact_interrupts_gap(self):
        self.step(0.005, contact())
        self.step(0.01)
        self.step(0.10, contact(force=0.001, speed=0))
        self.step(0.105)
        self.assertEqual(self.step(0.20, contact()).score, (1, 0))

    def test_threshold_changes_do_not_rearm(self):
        self.step(0.005, contact())
        self.ref.set_thresholds(128, 1.6)
        self.step(0.01, contact())
        self.ref.set_thresholds(1, 0.1)
        self.assertEqual(self.step(0.015, contact()).score, (1, 0))

    def test_weak_initial_contact_can_score_later_once(self):
        self.assertEqual(self.step(0.005, contact(force=0.1)).score, (0, 0))
        self.assertEqual(self.step(0.01, contact()).score, (1, 0))
        self.assertEqual(self.step(0.015, contact()).score, (1, 0))

    def test_large_timestamp_jump_in_contact_does_not_rearm(self):
        self.step(0.005, contact())
        self.assertEqual(self.step(1, contact()).score, (1, 0))

    def test_force_and_speed_must_pass_on_same_contact(self):
        self.ref.set_thresholds(16, 0.4)
        self.assertEqual(
            self.step(
                0.005,
                contact(force=64, speed=0.1),
                contact(force=1, speed=1.6),
            ).score,
            (0, 0),
        )

    def test_all_15_threshold_combinations(self):
        for force in FORCE_LEVELS:
            for speed in SPEED_LEVELS:
                self.ref = MatchStateMachine(scoring_mode="force_speed")
                self.ref.set_thresholds(force, speed)
                result = self.step(0.005, contact(force=force, speed=speed))
                self.assertEqual(result.score, (1, 0))
                self.assertEqual(result.events[0]["force_threshold_n"], force)

    def test_every_physics_substep_processed(self):
        self.step(0.005, contact())
        self.step(0.01)
        samples = [
            (0.10, [contact(force=0.001, speed=0)]),
            (0.105, []),
            (0.11, [contact()]),
            (0.115, []),
        ]
        result = self.ref.observe_frame(
            sim_time_s=0.12,
            contacts=[],
            physics_samples=samples,
            root_pos=[[-1, 0, 0.8], [1, 0, 0.8]],
            fallen=[False, False],
        )
        self.assertEqual(result.score, (1, 0))

    def test_recovery_and_ko_unchanged(self):
        self.ref.observe_frame(
            sim_time_s=1,
            contacts=[],
            root_pos=[[-1, 0, 0.3], [1, 0, 0.8]],
            fallen=[True, False],
        )
        result = self.ref.observe_frame(
            sim_time_s=4,
            contacts=[contact(actor=1)],
            root_pos=[[-1, 0, 0.3], [1, 0, 0.8]],
            fallen=[True, False],
        )
        self.assertTrue(result.match_over)
        self.assertEqual(result.score, (0, 1000))
        self.assertNotIn("HIT", [e["type"] for e in result.events])

    def test_torch_and_numpy_share_gate(self):
        import torch

        rng = np.random.default_rng(12)
        cpu, gpu = HandHitGate(4, 0.4), HandHitGate(4, 0.4)
        for i in range(100):
            physical = rng.random((3, 2, 2, 4)) > 0.8
            force = rng.random(physical.shape) * 16
            speed = rng.random(physical.shape) * 1.6
            a = cpu.step(i * 0.005, physical, force, speed)[0]
            b = gpu.step(
                i * 0.005,
                torch.as_tensor(physical),
                torch.as_tensor(force),
                torch.as_tensor(speed),
            )[0]
            np.testing.assert_array_equal(a, b.numpy())
        cpu.reset([1])
        gpu.reset([1])
        np.testing.assert_array_equal(cpu.armed, gpu.armed.numpy())


if __name__ == "__main__":
    unittest.main()
