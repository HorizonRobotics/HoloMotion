"""Own the optional Isaac application before importing simulator components."""

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json


@contextmanager
def physics_runtime(options):
    """Leave MuJoCo independent and close Isaac on success or failure."""
    if options.physics_backend != "isaac_physx":
        yield
        return

    # Select Warp before Kit loads bundled Python modules.
    import warp as wp
    from isaaclab.app import AppLauncher

    selected_warp = wp.__version__
    log_path = (options.output / "isaac_kit.log").resolve()
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite Kit log: {log_path}")
    original_argv = sys.argv[:]
    launcher = None
    try:
        sys.argv = [original_argv[0], f"--/log/file={log_path}"]
        launcher = AppLauncher(
            headless=True, device=options.device, fast_shutdown=False
        )
        import carb.settings

        settings = carb.settings.get_settings()
        configured = settings.get("/log/file")
        if not configured or Path(configured).resolve() != log_path:
            raise RuntimeError(
                "Isaac did not retain the requested Kit log path"
            )
        atomic_json(
            options.output / "isaac_logging.json",
            dict(requested_path=str(log_path), configured_path=configured),
        )
        wp.init()
        if wp.__version__ != selected_warp or wp.get_cuda_device_count() < 1:
            raise RuntimeError("Isaac changed Warp or lost CUDA")
        print(
            json.dumps(dict(phase="isaac_ready", warp=selected_warp)),
            flush=True,
        )
        yield
    finally:
        sys.argv = original_argv
        if launcher is not None:
            from holomotion.src.env.reference_tracking.isaac_backend import (
                IsaacTrackingBackend,
            )

            active = IsaacTrackingBackend._active
            try:
                if active is not None:
                    active.close()
            finally:
                launcher.app.close(wait_for_replicator=False)


def _isaac_worker(options, completion, evaluation, training_operation=None):
    from holomotion.src.algo.supermimic.training import execute_local, run

    operation = run if training_operation is None else training_operation
    if evaluation:
        from holomotion.src.algo.supermimic.evaluation import (
            evaluate_checkpoint,
        )

        operation = evaluate_checkpoint
    try:
        execute_local(options, completion=completion, operation=operation)
    finally:
        completion.close()


def _supervise(options, *, evaluation=False, training_operation=None):
    """Bound SDK cleanup only after explicit training/evaluation completion."""
    import multiprocessing
    from multiprocessing.connection import wait

    options.output.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isaac_worker,
        args=(options, sender, evaluation, training_operation),
    )
    process.start()
    sender.close()
    outcome = None
    try:
        wait([receiver, process.sentinel])
        if receiver.poll():
            try:
                outcome = receiver.recv()
            except EOFError:
                pass
        if outcome is None:
            process.join()
            message = f"Isaac exited without a completion message: {process.exitcode}"
            atomic_json(
                options.output / "run_status.json",
                dict(completed=False, phase="worker_exited"),
            )
            atomic_json(
                options.output / "isaac_process.json",
                dict(
                    operation_succeeded=False,
                    completion_received=False,
                    exit_code=process.exitcode,
                ),
            )
            (options.output / "error.txt").write_text(message + "\n")
            raise RuntimeError(message)
        shutdown_started = time.monotonic()
        process.join(timeout=30)
        terminated = process.is_alive()
        if terminated:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        atomic_json(
            options.output / "isaac_process.json",
            dict(
                operation_succeeded=outcome["succeeded"],
                shutdown_timeout=terminated,
                shutdown_wall_s=time.monotonic() - shutdown_started,
                exit_code=process.exitcode,
            ),
        )
        if not outcome["succeeded"]:
            raise RuntimeError(
                f"Isaac operation failed; see {options.output / 'error.txt'}"
            )
        status = json.loads((options.output / "run_status.json").read_text())
        if status != dict(completed=True, phase="finished"):
            raise RuntimeError(
                "Isaac completion message disagrees with run status"
            )
        return json.loads((options.output / "report.json").read_text())
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def execute_isaac(options, *, training_operation=None):
    """Run training and final evaluation in independent owned SDK processes."""
    from dataclasses import replace

    started = time.monotonic()
    report = (
        _supervise(options)
        if training_operation is None
        else _supervise(options, training_operation=training_operation)
    )
    report["training_process_wall_s"] = report["process_wall_s"]
    report["process_wall_s"] = time.monotonic() - started
    atomic_json(options.output / "report.json", report)
    if options.skip_evaluation or options.preflight_only:
        return report
    atomic_json(
        options.output / "run_status.json",
        dict(completed=False, phase="evaluation"),
    )
    evaluation_started = time.monotonic()
    try:
        result = _supervise(
            replace(options, output=options.output / "evaluation"),
            evaluation=True,
        )
        result["rollout_file"] = "evaluation/" + result["rollout_file"]
        report.update(
            evaluations=[result],
            evaluation_wall_s=time.monotonic() - evaluation_started,
            process_wall_s=time.monotonic() - started,
        )
        report["paper_reference_target_observed"] = bool(
            result["evaluated_clips"] == 40
            and result["normalized_return"] >= 0.82
        )
        atomic_json(
            options.output / "evaluation_final.json",
            dict(result, final_policy_evaluation=True),
        )
        atomic_json(options.output / "report.json", report)
        atomic_json(
            options.output / "run_status.json",
            dict(completed=True, phase="finished"),
        )
        return report
    except BaseException:
        atomic_json(
            options.output / "run_status.json",
            dict(completed=False, phase="evaluation_failed"),
        )
        raise
