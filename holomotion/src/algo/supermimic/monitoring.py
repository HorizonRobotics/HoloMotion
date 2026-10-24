"""Measured training counters shared by the reproduction and PPO control.

Episode summaries describe sampled training episodes, rather than the fixed
complete-clip evaluation population. Keep their names and denominators distinct.
"""

import math


class TrainingRewardArea:
    """Integrate measured physical reward on a fixed training wall clock.

    Each observation is the mean reward of one real vector step. Hold that
    value until the next observation. Before the first observation, the curve
    is zero, so startup cannot earn unmeasured reward. Finalization is excluded
    by ending the curve at the last completed training boundary.
    """

    def __init__(self, horizon_seconds=1800.0, tail_seconds=300.0):
        if not (
            math.isfinite(horizon_seconds)
            and math.isfinite(tail_seconds)
            and 0 < tail_seconds <= horizon_seconds
        ):
            raise ValueError("Reward area requires finite positive durations")
        self.horizon_seconds = float(horizon_seconds)
        self.tail_seconds = float(tail_seconds)
        self.points = []

    def observe(self, training_seconds, mean_real_step_reward):
        time, reward = float(training_seconds), float(mean_real_step_reward)
        if not math.isfinite(time) or time < 0 or not math.isfinite(reward):
            raise ValueError(
                "Reward observations require finite time and reward"
            )
        if self.points and time <= self.points[-1][0]:
            raise ValueError("Reward observation times must strictly increase")
        self.points.append((time, reward))

    def summary(self, training_seconds):
        end = float(training_seconds)
        if not math.isfinite(end) or end < 0:
            raise ValueError("Reward summary requires a finite training clock")
        if self.points and end < self.points[-1][0]:
            raise ValueError("Reward summary cannot precede its observations")
        end = min(end, self.horizon_seconds)
        area = tail_area = 0.0
        tail_start = self.horizon_seconds - self.tail_seconds
        for index, (time, reward) in enumerate(self.points):
            right = min(
                self.points[index + 1][0]
                if index + 1 < len(self.points)
                else end,
                end,
            )
            area += reward * max(0.0, right - time)
            tail_area += reward * max(0.0, right - max(time, tail_start))
        complete = end == self.horizon_seconds and bool(self.points)
        return dict(
            metric="real_training_reward_wall_time_area_v1",
            source="Mean physical step reward across all training environments",
            integration="Previous observation held until the next; zero before the first",
            horizon_seconds=self.horizon_seconds,
            observed_training_seconds=end,
            complete=complete,
            real_vector_observations=len(self.points),
            first_observation_seconds=self.points[0][0]
            if self.points
            else None,
            reward_seconds=area,
            time_mean_reward=area / self.horizon_seconds if complete else None,
            final_window_seconds=self.tail_seconds,
            final_window_mean_reward=tail_area / self.tail_seconds
            if complete
            else None,
        )


class EpisodeMonitor:
    def __init__(self):
        self.transitions = self.terminated = self.ended = self.completed = 0
        self.length_sum = self.return_sum = 0.0

    def add(self, transition):
        self.transitions += transition["done"].numel()
        self.terminated += int(transition["term"].sum())
        self.ended += int(transition["done"].sum())
        if "ep_len" in transition:
            lengths, returns = transition["ep_len"], transition["ep_ret_norm"]
            if lengths.shape != returns.shape or len(lengths) != int(
                transition["done"].sum()
            ):
                raise ValueError(
                    "Episode monitoring requires one length and return per completed episode"
                )
            self.completed += len(lengths)
            self.length_sum += float(lengths.sum())
            self.return_sum += float(returns.sum())

    def summary(self, *, vector_steps, training_seconds, gpu_count=1):
        return dict(
            real_transitions=self.transitions,
            completed_episodes=self.completed,
            physical_terminations=self.terminated,
            physical_episode_endings=self.ended,
            termination_fraction_among_endings=self.terminated / self.ended
            if self.ended
            else None,
            sampled_episode_mean_length=self.length_sum / self.completed
            if self.completed
            else None,
            sampled_episode_mean_normalized_return=self.return_sum
            / self.completed
            if self.completed
            else None,
            real_transitions_per_training_second=self.transitions
            / training_seconds
            if training_seconds
            else None,
            vector_steps_per_training_second=vector_steps / training_seconds
            if training_seconds
            else None,
            training_gpu_hours=gpu_count * training_seconds / 3600,
            episode_population="Return-sampled training episodes; distinct from complete-clip evaluation",
        )
