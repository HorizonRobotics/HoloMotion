"""Recorded-action world-model errors in physical units and named scales."""

import torch

from holomotion.src.env.reference_tracking.mdp import (
    normalize_xi,
    unflatten_state,
)


@torch.no_grad()
def dynamics_diagnostics(learner, *, seed, batch_size=256, horizon=8):
    replay = learner.buffer
    if replay.size < horizon:
        return dict(
            available=False,
            reason="Replay does not yet contain the requested physical window",
        )
    generator = torch.Generator(device=replay.device).manual_seed(seed)
    window = replay.sample_windows(batch_size, horizon, generator=generator)
    state = unflatten_state(window["s"][:, 0])
    delta_scale = torch.maximum(learner.wm.out_norm.std(), learner.ERR_FLOOR)
    # These fixed scales come from the second community reference. The paper
    # does not publish its complete nominal-error denominator.
    nominal_scale = torch.tensor(
        [0.05] * 3
        + [0.1] * 3
        + [0.5] * 3
        + [1.0] * 3
        + [0.1] * 29
        + [2.0] * 29,
        device=replay.device,
    )
    steps = []
    for step in range(horizon):
        state = learner.wm.step(
            state, window["a"][:, step], normalize_xi(window["xi"][:, step])
        )
        target = unflatten_state(window["s1"][:, step])
        error = learner.wm.state_error(state, target)
        valid = window["mask"][:, step] > 0
        if not valid.any():
            steps.append(dict(step=step + 1, valid_windows=0))
            continue
        error = error[valid]
        if not torch.isfinite(error).all():
            raise FloatingPointError(
                "Nonfinite predicted physical error on valid recorded-action windows"
            )
        steps.append(
            dict(
                step=step + 1,
                valid_windows=int(valid.sum()),
                empirical_delta_normalized_mse=float(
                    (error / delta_scale).square().mean()
                ),
                reproduced_nominal_normalized_mse=float(
                    (error / nominal_scale).square().mean()
                ),
                root_position_component_rmse_m=float(
                    error[:, :3].square().mean().sqrt()
                ),
                root_orientation_angle_rmse_rad=float(
                    error[:, 3:6].square().sum(-1).mean().sqrt()
                ),
                root_linear_velocity_component_rmse_m_s=float(
                    error[:, 6:9].square().mean().sqrt()
                ),
                root_angular_velocity_component_rmse_rad_s=float(
                    error[:, 9:12].square().mean().sqrt()
                ),
                joint_position_component_rmse_rad=float(
                    error[:, 12:41].square().mean().sqrt()
                ),
                joint_velocity_component_rmse_rad_s=float(
                    error[:, 41:].square().mean().sqrt()
                ),
            )
        )
    return dict(
        available=True,
        scope="Online recorded-action replay diagnostic; samples may have entered model fitting",
        unseen_physical_holdout=False,
        batch_size=batch_size,
        horizon=horizon,
        sampler_seed=seed,
        empirical_denominator="Current Vibed output increment standard deviation with its fixed per-component floors",
        nominal_denominator="Reproduced StateSpecCfg nominal errors: position .05 m, rotation .1 rad, velocity .5 m/s, angular velocity 1 rad/s, joint .1 rad, joint velocity 2 rad/s",
        directly_comparable_to_paper_error=False,
        steps=steps,
    )
