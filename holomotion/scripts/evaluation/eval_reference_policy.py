"""Evaluate a compact SM or PPO checkpoint in a dedicated MuJoCo process."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json, fingerprint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--robot-urdf", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode", choices=("native", "canonical_gpu_v1"), default="native"
    )
    parser.add_argument(
        "--capture-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def verify_inputs(specification, manifest, robot_urdf):
    """Allow relocation while requiring the complete original input contents."""
    import xml.etree.ElementTree as ET

    metadata = json.loads(manifest.read_text())
    paths = [manifest, robot_urdf]
    paths.extend(
        (manifest.parent / item["file"]).resolve()
        for item in metadata["hdf5_shards"]
    )
    paths.extend(
        sorted(
            {
                (robot_urdf.parent / mesh.get("filename")).resolve()
                for mesh in ET.parse(robot_urdf).getroot().findall(".//mesh")
            }
        )
    )
    hashes = {str(path): fingerprint(path) for path in paths}
    if Counter(hashes.values()) != Counter(
        specification["input_sha256"].values()
    ):
        raise ValueError("Evaluation inputs differ from checkpoint inputs")
    return hashes


def verify_task_source(specification, repository):
    """Require the checkpoint's shared task, reader and policy implementation."""
    hashes = specification["source_sha256"]
    required = (
        "holomotion/src/env/reference_tracking/task.py",
        "holomotion/src/algo/supermimic/networks.py",
        "holomotion/src/training/h5_dataloader.py",
        "holomotion/src/training/reference_fk.py",
    )
    if any(name not in hashes for name in required):
        raise ValueError("Checkpoint lacks the shared task source contract")
    selected = {
        name: digest
        for name, digest in hashes.items()
        if name in required
        or name.startswith("holomotion/src/env/reference_tracking/")
    }
    for name, digest in selected.items():
        if fingerprint(repository / name) != digest:
            raise ValueError(f"Evaluation task source changed: {name}")
    return selected


def run_worker(options):
    import hashlib
    import os

    import torch

    from holomotion.src.algo.supermimic import networks
    from holomotion.src.env.reference_tracking.task import load_task
    from holomotion.src.evaluation.reference_tracking.evaluation import (
        FullClipEvaluator,
    )

    started = time.monotonic()
    checkpoint_sha = fingerprint(options.checkpoint)
    bundle = torch.load(
        options.checkpoint, map_location="cpu", weights_only=False
    )
    specification = bundle["specification"]
    arguments = specification["arguments"]
    repository = Path(__file__).resolve().parents[3]
    manifest = (options.manifest or Path(arguments["manifest"])).resolve()
    robot = (options.robot_urdf or Path(arguments["robot_urdf"])).resolve()
    inputs = verify_inputs(specification, manifest, robot)
    source = verify_task_source(specification, repository)
    torch.manual_seed(arguments["seed"])
    torch.set_num_threads(8)
    torch.cuda.set_device(options.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if arguments["amp"] == "bf16" else None
    model, motion, mdp = load_task(
        robot_urdf=robot,
        manifest=manifest,
        output=options.output,
        device=options.device,
        profile=arguments["profile"],
    )
    policy = networks.Policy(
        mdp.obs_dim, 73 * arguments["goal_frames"], arguments["hist"]
    ).to(options.device)
    policy.load_state_dict(bundle["pi"], strict=True)
    policy.eval()

    def policy_digest():
        return hashlib.sha256(
            b"".join(
                name.encode()
                + value.detach().contiguous().cpu().numpy().tobytes()
                for name, value in sorted(policy.state_dict().items())
            )
        ).hexdigest()

    before = policy_digest()
    setup_seconds = time.monotonic() - started
    evaluator = FullClipEvaluator(
        model,
        motion,
        mdp,
        device=options.device,
        history_length=arguments["hist"],
        future_frames=arguments["goal_frames"],
        capture_graph=options.capture_graph,
        evaluation_mode=options.mode,
    )
    constructor_seconds = time.monotonic() - started - setup_seconds
    result = evaluator.evaluate(
        policy, progress=lambda event: print(json.dumps(event), flush=True)
    )
    if before != policy_digest() or checkpoint_sha != fingerprint(
        options.checkpoint
    ):
        raise ValueError("Evaluation modified policy or checkpoint")
    if inputs != verify_inputs(specification, manifest, robot):
        raise ValueError("Inputs changed during evaluation")
    verify_task_source(specification, repository)
    trace = options.output / "evaluation_trace.pt"
    torch.save(evaluator.last_trace, trace)
    evaluation_root = (
        repository / "holomotion/src/evaluation/reference_tracking"
    )
    evaluation_source = {
        str(path.relative_to(repository)): fingerprint(path)
        for path in sorted(evaluation_root.rglob("*"))
        if path.suffix in (".py", ".json")
    }
    return dict(
        completed=True,
        worker_pid=os.getpid(),
        parent_pid=os.getppid(),
        evaluation=result,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_provenance=dict(
            qualification_only=bool(bundle.get("qualification_only", False)),
            original_checkpoint_sha256=bundle.get(
                "original_checkpoint_sha256"
            ),
            original_policy_sha256=bundle.get("original_policy_sha256"),
        ),
        policy_sha256=before,
        input_sha256=inputs,
        task_source_sha256=source,
        evaluation_source_sha256=evaluation_source,
        training_backend=arguments["physics_backend"],
        evaluated_backend="mujoco_warp",
        evaluation_mode=options.mode,
        capture_graph=options.capture_graph,
        baseline_rebuild_required=options.mode != "native",
        learning_performed=False,
        gpu=torch.cuda.get_device_name(options.device),
        setup_seconds=setup_seconds,
        constructor_seconds=constructor_seconds,
        worker_wall_seconds=time.monotonic() - started,
        trace_file=trace.name,
        trace_sha256=fingerprint(trace),
    )


def main(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    options.output = options.output.expanduser().resolve()
    options.checkpoint = options.checkpoint.expanduser().resolve()
    if (
        not options.checkpoint.is_file()
        or not 0 < options.timeout_seconds < float("inf")
    ):
        parser.error(
            "Checkpoint must exist and timeout must be finite and positive"
        )
    if options.worker:
        atomic_json(options.output / "report.json", run_worker(options))
        return
    if options.output.exists() and any(options.output.iterdir()):
        parser.error("Choose an empty evaluation output directory")
    options.output.mkdir(parents=True, exist_ok=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", __name__, *arguments, "--worker"]
    # __name__ is __main__ under -m; use the importable module name explicitly.
    command[2] = "holomotion.scripts.evaluation.eval_reference_policy"
    started = time.monotonic()
    try:
        with (options.output / "evaluation.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=options.timeout_seconds,
                check=True,
            )
        report = json.loads((options.output / "report.json").read_text())
        if report.get("completed") is not True:
            raise RuntimeError("Evaluation worker did not report completion")
        atomic_json(
            options.output / "run_status.json",
            dict(
                completed=True,
                process_wall_seconds=time.monotonic() - started,
            ),
        )
    except BaseException as error:
        atomic_json(
            options.output / "run_status.json",
            dict(
                completed=False,
                error=str(error),
                process_wall_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
