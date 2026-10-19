"""Deadline checks must retain tails and scheduler overshoot."""

import unittest

from holoboxing.evaluation.benchmark import reserve_budget
from holoboxing.evaluation.runtime.chain_benchmark import control_loop_report


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1_000
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9) + 400_000


class ControlDeadlineTest(unittest.TestCase):
    def test_p99_uses_an_observed_tail_and_strict_deadline(self):
        durations = [19_000_000] * 99 + [20_000_000]
        frames = [
            dict(control_step_start_ns=1, control_step_end_ns=1 + x)
            for x in durations
        ]
        report = control_loop_report(frames, warmup=0)
        self.assertEqual(report["p99_ms"], 20.0)
        self.assertFalse(report["p99_within_budget"])
        self.assertTrue(report["mean_within_budget"])
        self.assertEqual(report["over_budget_steps"], 1)

    def test_reservation_counts_sleep_overshoot(self):
        clock = Clock()
        elapsed = reserve_budget(
            2_000_000, 10_000_000, clock=clock, sleep=clock.sleep
        )
        self.assertGreater(elapsed, 10_000_000)

    def test_an_over_budget_policy_is_not_clipped(self):
        clock = Clock()
        elapsed = reserve_budget(
            12_000_000, 10_000_000, clock=clock, sleep=clock.sleep
        )
        self.assertGreaterEqual(elapsed, 12_000_000)

    def test_invalid_timestamps_and_warmup_fail(self):
        with self.assertRaises(ValueError):
            control_loop_report([], warmup=-1)
        with self.assertRaises(ValueError):
            control_loop_report(
                [dict(control_step_start_ns=2, control_step_end_ns=1)],
                warmup=0,
            )


if __name__ == "__main__":
    unittest.main()
