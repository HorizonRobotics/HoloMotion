"""Measure completed matches with a reserved two-fighter policy-chain budget.

Actual models, observations, recurrent state, physics, referee and replay prep
still execute. A wall-clock reservation pads the completed policy work inside
each measured frame. This is a budget stress probe, not a substitute for timing
an actual policy on the target GPU. Cold setup and final file encoding are
reported separately from completed steady-state frames.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def reserve_budget(
    work_ns, budget_ns, *, clock=time.perf_counter_ns, sleep=time.sleep
):
    """Count actual scheduler overshoot; never replace measured time with a target."""
    if work_ns < 0 or budget_ns < 0:
        raise ValueError("Policy work and budget must be nonnegative")
    start = clock()
    deadline = start + max(0, budget_ns - work_ns)
    remaining = (deadline - clock()) / 1e9
    if remaining > 0.0002:
        sleep(remaining - 0.0002)
    while clock() < deadline:
        pass
    return work_ns + clock() - start


@contextmanager
def policy_reservation(budget_ms):
    from .runtime import upper, lower

    if not math.isfinite(budget_ms) or budget_ms <= 0:
        raise ValueError("Combined policy budget must be positive and finite")
    previous_window = upper.FivePointReferencePair.window
    previous_act = lower.LinkPolicy.act
    accumulated, samples = [None], []

    def window(pair, *args, **kwargs):
        start = time.perf_counter_ns()
        value = previous_window(pair, *args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        sanitation = pair.frame_data["competitor_sanitization_ns"]
        if len(sanitation) != 2 or any(
            type(x) is not int or x < 0 for x in sanitation
        ):
            raise ValueError(
                "Two completed output-sanitization timings are required"
            )
        # Qualification excludes only competitor-output sanitation. The full
        # control frame includes it, so do not charge it against the 10 ms reserve.
        accumulated[0] = elapsed - sum(sanitation)
        if accumulated[0] < 0:
            raise ValueError(
                "Sanitization exceeds completed upper-policy work"
            )
        return value

    def act(policy, *args, **kwargs):
        if accumulated[0] is None:
            raise RuntimeError(
                "Upper work must precede both lower-policy calls"
            )
        start = time.perf_counter_ns()
        value = previous_act(policy, *args, **kwargs)
        accumulated[0] += time.perf_counter_ns() - start
        if kwargs["fighter"] == 1:
            work = accumulated[0]
            reserved = reserve_budget(work, round(budget_ms * 1e6))
            samples.append(
                dict(work_ms=work / 1e6, reserved_ms=reserved / 1e6)
            )
            accumulated[0] = None
        return value

    upper.FivePointReferencePair.window = window
    lower.LinkPolicy.act = act
    try:
        yield samples
    finally:
        upper.FivePointReferencePair.window = previous_window
        lower.LinkPolicy.act = previous_act


def benchmark_match(
    red, blue, assets, lower, output_dir, *, policy_budget_ms=10.0
):
    if not math.isfinite(policy_budget_ms) or policy_budget_ms <= 0:
        raise ValueError("Combined policy budget must be positive and finite")
    from .validation import validate_model
    from .match import evaluate_match
    from .runtime.chain_benchmark import control_loop_report, gpu_telemetry
    from .runtime.model_contract import sha256

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(p).expanduser().resolve(strict=True) for p in (red, blue)]
    validation = {}
    # This installs the exact production runtime before the temporary wrappers.
    # Local hardware compatibility and the 5 ms performance criterion remain
    # separate: official target-hardware qualification is a different command.
    for path in dict.fromkeys(paths):
        validation[str(path)] = validate_model(
            path, assets, lower, qualification_mode="local"
        )
    valid = all(
        v["status"] == "PASS"
        and v.get("chain_benchmark", {}).get("passed") is True
        for v in validation.values()
    )
    if not valid:
        report = dict(
            status="FAIL",
            reason="single-fighter complete chain must pass mean <= 5 ms",
            validation=validation,
        )
        (output / "benchmark.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report
    telemetry = gpu_telemetry()
    started = time.perf_counter_ns()
    with policy_reservation(policy_budget_ms) as policy_samples:
        result = evaluate_match(
            paths[0], paths[1], assets, lower, output / "match"
        )
    wall_ns = time.perf_counter_ns() - started
    packet = json.loads((output / "match" / "match.json").read_text())
    if packet["metadata"].get("match_completed") is not True:
        raise ValueError("A complete match is required for the latency gate")
    frames = packet["frames"]
    if len(policy_samples) != len(frames):
        raise ValueError(
            "A completed paired-policy sample is required for every frame"
        )
    control = control_loop_report(frames)
    warm = policy_samples[control["warmup"] :]
    values = np.asarray([s["reserved_ms"] for s in warm])
    if not len(values):
        raise ValueError("Insufficient policy budget samples after warmup")
    report = dict(
        status="PASS" if control.get("p99_within_budget") is True else "FAIL",
        scope="complete control-frame p99 under an in-frame paired-policy time reservation",
        reservation_method="wall_clock_padding_after_completed_GPU_policy_work; scheduler_overshoot_included",
        reservation_boundary="upper_pair_window+both_lower_act-minus_output_sanitization; independent_FK_is_additional_frame_work",
        policy_budget_ms=policy_budget_ms,
        required_single_fighter_mean_ms=5.0,
        policy_reservation=dict(
            mean_ms=float(values.mean()),
            p99_ms=float(np.percentile(values, 99, method="higher")),
            maximum_ms=float(values.max()),
        ),
        control_loop=control,
        match_wall_seconds=wall_ns / 1e9,
        match=result,
        models=[dict(path=str(p), sha256=sha256(p)) for p in paths],
        validation=validation,
        gpu_telemetry_before=telemetry,
        gpu_telemetry_after=gpu_telemetry(),
        target_hardware_notice="Applies only to the recorded GPU, driver, image, models and workload. Re-run on the competition target.",
    )
    (output / "policy-budget-samples.json").write_text(
        json.dumps(policy_samples) + "\n"
    )
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("red_model", metavar="YOUR_POLICY")
    parser.add_argument("blue_model", metavar="OPPONENT_POLICY")
    parser.add_argument(
        "--assets",
        default=os.environ.get("HOLOBOXING_ASSETS"),
        required=not bool(os.environ.get("HOLOBOXING_ASSETS")),
    )
    parser.add_argument(
        "--lower",
        default=os.environ.get("HOLOBOXING_LOWER"),
        required=not bool(os.environ.get("HOLOBOXING_LOWER")),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-budget-ms", type=float, default=10.0)
    args = parser.parse_args(argv)
    report = benchmark_match(
        args.red_model,
        args.blue_model,
        args.assets,
        args.lower,
        args.output_dir,
        policy_budget_ms=args.policy_budget_ms,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
