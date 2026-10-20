"""Completed single-fighter upper + frozen-lower timing, without simulation.

The fixture uses real frozen-controller state, scene and production adapters.
Physics stays at its reset pose. Recurrent state, history, observation building,
I/O and target decoding run normally. Only output sanitation is excluded.
"""

import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100

BENCHMARK_VERSION = "upper_lower_completed_mean5ms_v1"
MEASUREMENT_SCOPE = "single_fighter_upper_observation_inference_decode_lower_observation_inference_cache_target; excludes_output_sanitization_and_physics"


def measure_chain(
    step,
    synchronize,
    *,
    warmup=DEFAULT_WARMUP,
    iterations=DEFAULT_ITERATIONS,
    clock=time.perf_counter_ns,
):
    if warmup < 0 or iterations < 1:
        raise ValueError(
            "Warmup must be nonnegative; iterations must be positive"
        )
    samples, sanitation = [], []
    for index in range(warmup + iterations):
        synchronize()
        started = clock()
        excluded_ns = step()
        synchronize()
        elapsed = clock() - started
        if (
            not isinstance(excluded_ns, int)
            or excluded_ns < 0
            or excluded_ns > elapsed
        ):
            raise ValueError("Invalid excluded sanitizer timing")
        if index >= warmup:
            samples.append((elapsed - excluded_ns) / 1e6)
            sanitation.append(excluded_ns / 1e6)
    values = np.asarray(samples)
    mean = float(values.mean())
    return dict(
        version=BENCHMARK_VERSION,
        scope=MEASUREMENT_SCOPE,
        warmup=warmup,
        iterations=iterations,
        mean_ms=mean,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        maximum_ms=float(values.max()),
        sanitizer_mean_ms=float(np.mean(sanitation)),
        threshold_ms=5.0,
        criterion="mean_ms <= 5.0; individual tails are diagnostic",
        passed=mean <= 5.0,
        timing_completed=True,
        fps=1000 / mean if mean > 0 else None,
    )


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def gpu_telemetry():
    """Record observed driver, load and clock state without changing GPU settings."""
    fields = "index,uuid,name,driver_version,pstate,utilization.gpu,memory.used,memory.total,clocks.current.sm,clocks.current.memory,power.draw"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + fields,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return dict(
            fields=fields,
            rows=result.stdout.strip().splitlines(),
            units="utilization percent; memory MiB; clocks MHz; power W",
        )
    except (OSError, subprocess.SubprocessError) as error:
        return dict(status="UNAVAILABLE", reason=type(error).__name__)


def control_loop_report(frames, *, warmup=100):
    """Completed frames, including both fighters, physics, referee and replay prep."""
    if type(warmup) is not int or warmup < 0:
        raise ValueError("Warmup must be a nonnegative integer")
    samples = []
    for frame in frames[warmup:]:
        start, end = (
            frame["control_step_start_ns"],
            frame["control_step_end_ns"],
        )
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError("Invalid completed control-step timestamps")
        samples.append((end - start) / 1e6)
    result = dict(
        scope="A_upper_lower+B_upper_lower+physics+referee+in_memory_replay; includes_sanitization; excludes_final_file_encoding_and_browser_playback",
        warmup=warmup,
        samples=len(samples),
        target_hz=50,
        budget_ms=20.0,
        criterion="p99_ms < 20.0",
        percentile_method="higher",
    )
    if not samples:
        return dict(result, status="INSUFFICIENT_SAMPLES")
    data = np.asarray(samples)
    return dict(
        result,
        status="MEASURED",
        mean_ms=float(data.mean()),
        p50_ms=float(np.percentile(data, 50)),
        p95_ms=float(np.percentile(data, 95)),
        p99_ms=float(np.percentile(data, 99, method="higher")),
        maximum_ms=float(data.max()),
        mean_within_budget=bool(data.mean() <= 20),
        p99_within_budget=bool(np.percentile(data, 99, method="higher") < 20),
        over_budget_steps=int((data >= 20).sum()),
    )


def benchmark_model(
    model,
    *,
    assets,
    lower_directory,
    warmup=DEFAULT_WARMUP,
    iterations=DEFAULT_ITERATIONS,
    device="cuda:0",
):
    # Loading this entry installs precisely the production CUDA/observation adapters.
    from .holomotion_v2_runtime import install_runtime

    runtime = install_runtime(lower_directory)
    from .upper import FivePointReferencePair
    from .lower import MODEL_ROOT, VERSION, verify_checkpoint
    from ..resources import ResourceTree
    from .compiled_artifacts import available
    from .onnx_device import require_gpu

    resources = ResourceTree(assets)
    import onnxruntime as ort

    if not device.startswith("cuda"):
        raise RuntimeError(
            "CUDA chain benchmarking is required; CPU fallback is forbidden"
        )
    require_gpu(device)
    compiled = available()
    if compiled:
        from .lean_runtime import (
            METRICS,
            runtime_environment,
            set_comparison,
            synchronize,
        )
    else:
        import torch

        from .fast_runtime import METRICS

        torch.set_num_threads(1)

        def synchronize(selected):
            torch.cuda.synchronize(selected)

        def runtime_environment(selected):
            return dict(
                gpu=torch.cuda.get_device_name(torch.device(selected)),
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                onnxruntime_version=ort.__version__,
            )

    lower = MODEL_ROOT / "exported/model_10500.onnx"
    verify_checkpoint(lower)
    motion = runtime.ResetPose.load(resources.initial_state)
    from .hand_hit_rules import configured_thresholds

    scene = runtime.BoxingMatchScene(
        model_path=resources.scene,
        mesh_root=resources.scene.parent / "meshes",
        simulation_frequency=200,
        control_decimation=4,
        contact_force_threshold=configured_thresholds()[0],
        show_viewer=False,
        realtime=False,
        physics_device=device,
        inference_benchmark=True,
    )
    previous_compare = os.environ.get("BOXING_COMPARE_OBSERVATIONS")
    try:
        policies = tuple(
            runtime.HoloMotionV2Policy(
                lower,
                device=device,
                articulation_joint_names=scene.articulation_joint_names,
                retarget_dof_names=motion.dof_names,
            )
            for _ in range(2)
        )
        scene._kp = (
            policies[0]
            .kp[policies[0].model_to_articulation]
            .astype(np.float64)
        )
        scene._kd = (
            policies[0]
            .kd[policies[0].model_to_articulation]
            .astype(np.float64)
        )
        order = runtime._name_order(
            motion.dof_names,
            scene.articulation_joint_names,
            "retargeted motion",
        )
        runtime._reset_scene(scene, motion, order)
        for policy in policies:
            policy.reset()
        pair = FivePointReferencePair(
            [Path(model), Path(model)], scene, policies, device=device
        )
        # No contacts or referee are measured in this fixed-state fixture.
        # Keep the production rule state without allocating a collision engine.
        from .rules import MatchStateMachine

        pair.match = MatchStateMachine(runtime.MatchRules())
        for policy in policies:
            policy.retarget_to_model = np.arange(
                runtime.DOF_COUNT, dtype=np.int64
            )
        before = scene.read_kinematics()

        def step():
            scene.prepare_actor_observation()
            reference = pair.window(before, fighters=(0,))[0]
            policies[0].act(
                reference_qpos=reference,
                kinematics=before,
                fighter=0,
                reference_current_index=0,
            )
            return pair.frame_data["competitor_sanitization_ns"][0]

        # Compare the observation, actions AND returned cache through the
        # production reference path before collecting timing samples.
        METRICS.clear()
        if compiled:
            set_comparison(pair, policies[0], True)
        else:
            os.environ["BOXING_COMPARE_OBSERVATIONS"] = "1"
        for _ in range(5):
            step()
            synchronize(device)
        parity = dict(
            steps=5,
            observation_atol=3e-5,
            observation_rtol=3e-5,
            action_and_cache_atol=1e-6,
            action_and_cache_rtol=1e-6,
            max_observation_error=METRICS.get("max_observation_error"),
            max_binding_action_error=METRICS.get("max_binding_action_error"),
        )
        if compiled:
            set_comparison(pair, policies[0], False)
            parity["scope"] = (
                "compiled observation CPU/CUDA parity; CUDA model feed/binding "
                "action and cache parity; canonical PyTorch/graph parity "
                "recorded at export"
            )
        else:
            os.environ["BOXING_COMPARE_OBSERVATIONS"] = "0"
        telemetry_before = gpu_telemetry()
        report = measure_chain(
            step,
            lambda: synchronize(device),
            warmup=warmup,
            iterations=iterations,
        )
        report.update(
            **runtime_environment(device),
            gpu_telemetry_before=telemetry_before,
            gpu_telemetry_after=gpu_telemetry(),
            parity_preflight=parity,
            execution_provider=pair.sessions[0].get_providers()[0],
            platform=platform.platform(),
            upper_sha256=file_sha256(model),
            lower_sha256=file_sha256(lower),
            lower_version=VERSION,
            model_config_sha256=file_sha256(MODEL_ROOT / "config.yaml"),
            batch=1,
            precision="float32",
            fixture="actual_scene_reset_pose_kinematics_only_no_physics_stepping",
            cache="returned recurrent state; production reset at maximum context length",
            sanitization_components=pair.sanitization_total[0],
            target_hardware_gate="RTX 5090 initial round; S600 subsequent round; report GPU is actual hardware",
        )
        return report
    finally:
        if previous_compare is None:
            os.environ.pop("BOXING_COMPARE_OBSERVATIONS", None)
        else:
            os.environ["BOXING_COMPARE_OBSERVATIONS"] = previous_compare
        scene.close()
