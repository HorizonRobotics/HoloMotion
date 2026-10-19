import unittest
from holoboxing.evaluation.runtime.rules import MatchStateMachine
from holoboxing.evaluation.runtime.impulse_hit_rules import (
    RollingNormalImpulse,
)


def contact(
    force, *, actor=0, hand="left_hand", target="head", pair=0, speed=0
):
    return dict(
        attacker=actor,
        defender=1 - actor,
        attacking_body=hand,
        target_body=target,
        physical_contact=True,
        peak_force=force,
        normal_force_n=force,
        sample_dt_s=0.005,
        closing_speed=speed,
        pair_id=pair,
        position=[0, 0, 1],
    )


class ImpulseRulesTest(unittest.TestCase):
    def setUp(self):
        self.ref = MatchStateMachine()

    def step(self, t, *contacts):
        return self.ref.observe_frame(
            sim_time_s=t,
            contacts=contacts,
            root_pos=[[-0.5, 0, 0.8], [0.5, 0, 0.8]],
            fallen=[False, False],
        )

    def test_exact_10ns_zero_speed_scores(self):
        result = self.step(0.005, contact(2000, speed=0))
        self.assertEqual(result.score, (1, 0))
        hit = result.events[0]
        self.assertEqual(hit["normal_impulse_ns"], 10)
        self.assertEqual(hit["impulse_threshold_ns"], 10)
        self.assertIsNone(hit["speed_threshold_mps"])

    def test_old_5ns_threshold_no_longer_scores(self):
        self.assertEqual(self.step(0.005, contact(1000)).score, (0, 0))

    def test_below_threshold_then_accumulate(self):
        self.assertEqual(self.step(0.005, contact(1999)).score, (0, 0))
        self.assertEqual(self.step(0.010, contact(1)).score, (1, 0))

    def test_20_substeps_not_peak_force_times_100ms(self):
        for i in range(1, 20):
            self.assertEqual(self.step(i * 0.005, contact(100)).score, (0, 0))
        self.assertEqual(self.step(0.1, contact(100)).score, (1, 0))
        self.assertAlmostEqual(
            self.ref.contact_diagnostics[0]["normal_impulse_ns"], 10
        )

    def test_half_open_window_expiry_and_empty_steps(self):
        clock = RollingNormalImpulse()
        self.assertAlmostEqual(clock.update(0.005, [contact(800)])[0][0, 0], 4)
        self.assertEqual(clock.update(0.104, [])[0][0, 0], 4)
        self.assertEqual(clock.update(0.105, [contact(200)])[0][0, 0], 1)

    def test_duplicate_points_not_double_counted_and_multiple_points_sum(self):
        clock = RollingNormalImpulse()
        item = contact(500)
        self.assertEqual(clock.update(0.005, [item, item])[0][0, 0], 2.5)
        self.assertEqual(
            clock.update(0.01, [contact(250, pair=1), contact(250, pair=2)])[
                0
            ][0, 0],
            5,
        )

    def test_actors_hands_and_targets(self):
        result = self.step(
            0.005,
            contact(1200, hand="left_hand", target="head", pair=0),
            contact(800, hand="left_hand", target="chest", pair=1),
            contact(1999, hand="right_hand", pair=2),
            contact(2000, actor=1, pair=3),
        )
        self.assertEqual(result.score, (1, 1))
        self.assertEqual(
            len([e for e in result.events if e["type"] == "HIT"]), 2
        )

    def test_foot_and_illegal_targets_excluded(self):
        self.assertEqual(
            self.step(
                0.005,
                contact(200000, hand="left_foot"),
                contact(200000, target="left_hand"),
            ).score,
            (0, 0),
        )

    def test_cooldown_does_not_require_separation(self):
        self.step(0.005, contact(2000))
        self.step(0.010)
        self.step(0.10, contact(0))
        self.step(0.105)
        self.assertEqual(self.step(0.20, contact(2000)).score, (2, 0))
        self.step(0.205)
        self.assertEqual(self.step(0.305, contact(2000)).score, (3, 0))

    def test_sustained_contact_scores_every_100ms(self):
        for i in range(1, 101):
            self.assertEqual(
                self.step(i * 0.005, contact(2000)).score,
                ((i - 1) // 20 + 1, 0),
            )

    def test_scored_impulse_cannot_be_reused(self):
        self.step(0.005, contact(2000))
        for i in range(2, 22):
            self.assertEqual(self.step(i * 0.005, contact(0)).score, (1, 0))
        self.assertEqual(self.step(0.110, contact(1999)).score, (1, 0))
        self.assertEqual(self.step(0.115, contact(1)).score, (2, 0))

    def test_hands_have_independent_cooldowns(self):
        self.assertEqual(self.step(0.005, contact(2000)).score, (1, 0))
        self.assertEqual(
            self.step(0.01, contact(2000, hand="right_hand")).score, (2, 0)
        )
        self.assertEqual(self.step(0.105, contact(2000)).score, (3, 0))
        self.assertEqual(
            self.step(0.1055, contact(2000, hand="right_hand")).score, (3, 0)
        )
        self.assertEqual(
            self.step(0.11, contact(0, hand="right_hand")).score, (4, 0)
        )

    def test_reset_clears_old_impulse(self):
        self.step(0.005, contact(1999))
        self.ref.reset_motion_state()
        self.assertEqual(self.step(0.01, contact(1)).score, (0, 0))

    def observe_fall(self, time, fallen):
        return self.ref.observe_frame(
            sim_time_s=time, contacts=[],
            root_pos=[[-0.5, 0, 0.2], [0.5, 0, 0.2]], fallen=fallen,
        )

    def test_simultaneous_confirmed_falls_draw_despite_unequal_hits(self):
        self.ref.score[0] = 7
        self.observe_fall(1, [True, True])
        result = self.observe_fall(4, [True, True])
        self.assertTrue(result.match_over)
        self.assertIsNone(result.winner)
        self.assertEqual({e['actor'] for e in result.events if e['type'] == 'KO'}, {0, 1})

    def test_delayed_observation_uses_earliest_fall_not_fighter_order(self):
        self.observe_fall(1, [False, True])
        self.observe_fall(1.02, [True, True])
        result = self.observe_fall(4.02, [True, True])
        self.assertEqual(result.winner, 0)
        self.assertEqual([e['actor'] for e in result.events if e['type'] == 'KO'], [0])

    def test_equal_scores_without_falls_draw_at_time_limit(self):
        self.ref.score[:] = [4, 4]
        result = self.step(60)
        self.assertTrue(result.match_over)
        self.assertIsNone(result.winner)
        self.assertNotIn('KO', [e['type'] for e in result.events])

    def test_summary_retains_simultaneous_fall_draw(self):
        from holoboxing.evaluation.runtime.rules import MatchLogger
        self.ref.score[0] = 7
        self.observe_fall(1, [True, True])
        result = self.observe_fall(4, [True, True])
        logger = MatchLogger(model_id=['red', 'blue'], match_id='draw')
        logger.frames.append(dict(events=result.events, policy_start_ns=[0, 0], policy_end_ns=[1, 1]))
        self.assertIsNone(logger.summary(result.score)['winner'])

    def test_missing_measured_normal_force_is_error(self):
        item = contact(2000)
        del item["normal_force_n"]
        with self.assertRaises(ValueError):
            self.step(0.005, item)

    def test_aggregate_normal_impulse_not_peak_magnitude(self):
        item = contact(99999)
        item["normal_impulse_step_ns"] = 0.005
        self.assertEqual(self.step(0.005, item).score, (0, 0))


if __name__ == "__main__":
    unittest.main()
