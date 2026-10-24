"""Deterministic complete-clip scoring for the small reproduction benchmark."""

import time

import torch

from holomotion.src.env.reference_tracking.mdp import MAX_STEP_REWARD


def full_clip_scores(returns, planned_transitions):
    if (
        returns.shape != planned_transitions.shape
        or (planned_transitions <= 0).any()
        or not torch.isfinite(returns).all()
    ):
        raise ValueError(
            "Complete-clip scores require finite returns and positive planned lengths"
        )
    return returns / (MAX_STEP_REWARD * planned_transitions)


class FullClipEvaluator:
    def __init__(
        self,
        model,
        motion,
        mdp,
        *,
        device,
        history_length=10,
        future_frames=10,
        capture_graph=True,
        evaluation_mode="native",
    ):
        from holomotion.src.env.reference_tracking.warp_backend import (
            WarpTrackingBackend,
        )

        if evaluation_mode == "canonical_gpu_v1":
            from .canonical_gpu.backend import CanonicalEvaluationBackend

            backend = CanonicalEvaluationBackend
        elif evaluation_mode == "native":
            backend = WarpTrackingBackend
        else:
            raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.motion, self.mdp = motion, mdp
        self.history_length, self.future_frames = history_length, future_frames
        self.env = backend(
            model,
            motion,
            mdp,
            len(motion.names),
            device=device,
            evaluation=True,
            capture_graph=capture_graph,
        )

    @torch.no_grad()
    def evaluate(self, policy, *, progress=None):
        start = time.monotonic()
        motion, mdp, env = self.motion, self.mdp, self.env
        ids = torch.arange(env.N, device=env.device)
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.reset(ids, motion.clip_start)
        hist = mdp.obs(env.state(), env.a_prev, env.xi, env.f)[:, None].repeat(
            1, self.history_length, 1
        )
        returns = torch.zeros(env.N, device=env.device)
        elapsed = torch.zeros(env.N, dtype=torch.long, device=env.device)
        planned = motion.clip_len - 1
        body_error_sum = torch.zeros_like(returns)
        root_error_sum = torch.zeros_like(returns)
        trace = {
            name: []
            for name in (
                "clip_id",
                "reference_frame",
                "next_state",
                "action",
                "reward",
                "terminated",
                "ended",
            )
        }
        max_steps = int(planned.max())
        for step in range(max_steps):
            active = ~env.done_eval
            observation = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            if step:
                hist = torch.cat((hist[:, 1:], observation[:, None]), 1)
            action = policy(hist, motion.goal(env.f, self.future_frames))
            transition = env.step(action)
            packet = dict(
                clip_id=ids,
                reference_frame=env.f,
                next_state=transition["s1"],
                action=transition["a"],
                reward=transition["r"],
                terminated=transition["term"],
                ended=transition["done"],
            )
            for name, value in packet.items():
                trace[name].append(value[active].cpu())
            returns += active * transition["r"]
            elapsed += active
            root = env.state()[0]
            root_error_sum += active * (root - motion.root_p[env.f]).norm(
                dim=-1
            )
            # These errors supplement the paper's score and failure counts.
            bp = mdp.kin.forward(*env.state())[0]
            body_error_sum += active * (bp - motion.body_p[env.f]).norm(
                dim=-1
            ).mean(-1)
            if progress is not None and (step + 1) % 1000 == 0:
                progress(
                    dict(
                        evaluation_step=step + 1,
                        completed_clips=int(env.done_eval.sum()),
                        total_clips=env.N,
                    )
                )
            if env.done_eval.all():
                break
        if not env.done_eval.all():
            raise RuntimeError(
                "Complete-clip evaluator did not finish every planned trial"
            )
        scores = full_clip_scores(returns, planned)
        self.last_trace = {
            name: torch.cat(values) for name, values in trace.items()
        }
        self.last_trace.update(
            clip_names=motion.names,
            clip_start=motion.clip_start.cpu(),
            clip_length=motion.clip_len.cpu(),
            state_convention="root XYZ, WXYZ, world linear/angular velocities, 29 SDK joints, 29 velocities",
            control_interval_s=0.02,
        )
        clips = [
            dict(
                name=name,
                normalized_return=float(scores[i]),
                total_reward=float(returns[i]),
                failed=bool(env.failed_eval[i]),
                planned_steps=int(planned[i]),
                executed_steps=int(elapsed[i]),
                mean_global_tracked_body_error_mm=float(
                    1000 * body_error_sum[i] / elapsed[i].clamp_min(1)
                ),
                mean_root_position_error_mm=float(
                    1000 * root_error_sum[i] / elapsed[i].clamp_min(1)
                ),
            )
            for i, name in enumerate(motion.names)
        ]
        return dict(
            evaluation_protocol=(
                env.canonical.metadata()
                if self.evaluation_mode == "canonical_gpu_v1"
                else {"mode": "native"}
            ),
            evaluation_mode=self.evaluation_mode,
            new_evaluation_protocol=self.evaluation_mode != "native",
            baseline_rebuild_required=self.evaluation_mode != "native",
            scope="Training-set benchmark; deterministic complete clips starting at frame zero",
            reward_denominator="0.1 multiplied by full planned transitions, including unexecuted failure tails",
            normalized_return=float(scores.mean()),
            failed_clips=int(env.failed_eval.sum()),
            frame_weighted_normalized_return=float(
                returns.sum() / (MAX_STEP_REWARD * planned.sum())
            ),
            evaluated_clips=env.N,
            clips=clips,
            elapsed_s=time.monotonic() - start,
            observation_noise=False,
            reset_noise=False,
            domain_randomization=False,
            foot_friction=1.0,
            physics_status=env.status_summary(),
        )
