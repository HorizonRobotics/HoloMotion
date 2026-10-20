import unittest
from holoboxing.evaluation.runtime.rules import MatchStateMachine
from holoboxing.evaluation.runtime.impulse_hit_rules import (
    RollingNormalImpulse,
)
from holoboxing.evaluation.runtime.hand_hit_rules import SCORING_VERSION, scoring_config


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

    def test_exact_5ns_zero_speed_scores(self):
        result = self.step(0.005, contact(1000, speed=0))
        self.assertEqual(result.score, (1, 0))
        hit = result.events[0]
        self.assertEqual(hit["normal_impulse_ns"], 5)
        self.assertEqual(hit["impulse_threshold_ns"], 5)
        self.assertEqual(hit["impulse_window_ms"], 20)
        self.assertIsNone(hit["speed_threshold_mps"])

    def test_below_5ns_does_not_score(self):
        self.assertEqual(self.step(0.005, contact(999)).score, (0, 0))

    def test_below_threshold_then_accumulate(self):
        self.assertEqual(self.step(0.005, contact(999)).score, (0, 0))
        self.assertEqual(self.step(0.010, contact(1)).score, (1, 0))

    def test_four_physics_substeps_accumulate_exact_5ns(self):
        for i in range(1, 4):
            self.assertEqual(self.step(i * 0.005, contact(250)).score, (0, 0))
        self.assertEqual(self.step(0.020, contact(250)).score, (1, 0))
        self.assertAlmostEqual(
            self.ref.contact_diagnostics[0]["normal_impulse_ns"], 5
        )

    def test_control_frame_integrates_four_physics_samples(self):
        result = self.ref.observe_frame(
            sim_time_s=.020, contacts=[],
            root_pos=[[-.5, 0, .8], [.5, 0, .8]], fallen=[False, False],
            physics_samples=[(i * .005, [contact(250)]) for i in range(1, 5)],
        )
        self.assertEqual(result.score, (1, 0))
        hits = [event for event in result.events if event['type'] == 'HIT']
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['physics_time_s'], .020)
        self.assertEqual(hits[0]['normal_impulse_ns'], 5)

    def test_recorded_rules_match_window_threshold_and_cooldown(self):
        config = scoring_config()
        self.assertEqual(config['window_ms'], 20)
        self.assertEqual(config['normal_impulse_threshold_ns'], 5)
        self.assertEqual(config['per_hand_cooldown_ms'], 200)
        self.assertTrue(SCORING_VERSION.endswith('_v6'))

    def test_legacy_mode_spelling_uses_current_impulse_rules(self):
        self.ref = MatchStateMachine(scoring_mode='impulse100ms')
        self.assertEqual(self.step(.005, contact(1000)).score, (1, 0))
        self.assertEqual(self.step(.105, contact(1000)).score, (1, 0))

    def test_force_cannot_accumulate_beyond_20ms_window(self):
        for i in range(1, 41):
            self.assertEqual(self.step(i * .005, contact(200)).score, (0, 0))

    def test_half_open_window_expiry_and_empty_steps(self):
        clock = RollingNormalImpulse()
        self.assertAlmostEqual(clock.update(0.005, [contact(800)])[0][0, 0], 4)
        self.assertEqual(clock.update(0.024, [])[0][0, 0], 4)
        self.assertEqual(clock.update(0.025, [contact(200)])[0][0, 0], 1)

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
            contact(600, hand="left_hand", target="head", pair=0),
            contact(400, hand="left_hand", target="chest", pair=1),
            contact(999, hand="right_hand", pair=2),
            contact(1000, actor=1, pair=3),
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
        self.step(0.005, contact(1000))
        self.step(0.010)
        self.assertEqual(self.step(0.105, contact(1000)).score, (1, 0))
        self.assertEqual(self.step(0.200, contact(1000)).score, (1, 0))
        self.assertEqual(self.step(0.205, contact(1000)).score, (2, 0))
        self.assertEqual(self.step(0.400, contact(1000)).score, (2, 0))
        self.assertEqual(self.step(0.405, contact(1000)).score, (3, 0))

    def test_sustained_contact_scores_every_200ms(self):
        for i in range(1, 101):
            self.assertEqual(
                self.step(i * 0.005, contact(1000)).score,
                ((i - 1) // 40 + 1, 0),
            )

    def test_scored_impulse_cannot_be_reused(self):
        self.step(0.005, contact(1000))
        for i in range(2, 41):
            self.assertEqual(self.step(i * 0.005, contact(0)).score, (1, 0))
        self.assertEqual(self.step(0.205, contact(999)).score, (1, 0))
        self.assertEqual(self.step(0.210, contact(1)).score, (2, 0))

    def test_hands_have_independent_cooldowns(self):
        self.assertEqual(self.step(0.005, contact(1000)).score, (1, 0))
        self.assertEqual(
            self.step(0.01, contact(1000, hand="right_hand")).score, (2, 0)
        )
        self.assertEqual(
            self.step(0.205, contact(1000), contact(1000, hand="right_hand")).score,
            (3, 0),
        )
        self.assertEqual(
            self.step(0.210, contact(0, hand="right_hand")).score, (4, 0)
        )

    def test_reset_clears_old_impulse(self):
        self.step(0.005, contact(999))
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
