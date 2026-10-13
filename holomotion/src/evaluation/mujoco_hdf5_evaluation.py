# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Full-clip HDF5 evaluation on the existing MuJoCo evaluator.

One isolated FP32 gate precedes all rollout subprocesses in this job. Workers
consume the checkpoint-bound attestation; they never construct a Torch actor.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from holomotion.src.evaluation.aggregate_evaluation_results import (
    aggregate_shards,
    write_json,
)
from holomotion.src.evaluation.checkpoint_onnx_parity import file_sha256
from holomotion.src.evaluation.mujoco_metric_tracker import MujocoMetricTracker
from holomotion.src.motion_tracking.link_mode_mask import resolve_control_links


def build_hdf5_evaluation_dataset(config):
    """Reuse the training HDF5 reader, retaining exactly one full window/clip.

    Return (dataset, clip_index). The caller owns dataset.close(). FK and
    normalization stay in memory; no NPZ conversion or derived dataset is made.
    """
    from holomotion.src.training.h5_dataloader import Hdf5RootDofDataset

    data = config["evaluation_data"]
    manifest_path = Path(data["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = data.get("clip_names")
    selection_path = data.get("selection_path")
    if requested is not None and selection_path:
        raise ValueError("Specify clip_names or selection_path, not both")
    if selection_path:
        selection = json.loads(
            Path(selection_path).read_text(encoding="utf-8")
        )
        requested = selection["val_handpicked_motion_names"]
        if set(requested).intersection(
            selection.get("train_handpicked_motion_names", [])
        ):
            raise ValueError("Selection has train/val overlap")
    if requested is None or isinstance(requested, str):
        raise ValueError(
            "Evaluation requires explicit clip names or selection"
        )
    requested = list(requested)
    if not requested or not all(
        isinstance(name, str) and name for name in requested
    ):
        raise ValueError("Clip names must be nonempty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate evaluation clip names")
    missing = set(requested) - set(manifest["clips"])
    if missing:
        raise ValueError(f"Clips absent from manifest: {sorted(missing)}")
    horizon = int(data.get("reference_horizon", 5))
    lengths = [int(manifest["clips"][name]["length"]) for name in requested]
    if horizon < 0 or min(lengths) <= horizon:
        raise ValueError("Every complete clip must exceed the future horizon")
    dataset = Hdf5RootDofDataset(
        manifest_path=str(manifest_path),
        max_frame_length=max(lengths),
        min_window_length=horizon + 1,
        handpicked_motion_names=requested,
        fk_robot_file_path=data["fk_robot_file_path"],
        expected_dof_names=data.get("expected_dof_names"),
        allowed_prefixes=("ref_",),
    )
    index = {
        window.raw_motion_key: i for i, window in enumerate(dataset.windows)
    }
    if set(index) != set(requested) or len(dataset) != len(requested):
        dataset.close()
        raise ValueError("Expected exactly one full HDF5 window per clip")
    return dataset, index


def evaluation_groups(config):
    groups = config.evaluation_run.get("groups")
    if groups is None:
        groups = OmegaConf.create(
            [
                {
                    "name": config.evaluation_run.phase,
                    "link_eval": config.link_eval,
                }
            ]
        )
    names = [str(group.name) for group in groups]
    if not names or len(names) != len(set(names)):
        raise ValueError("Evaluation groups must have unique names")
    for name in names:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Group names must be single safe path components")
    return [(str(group.name), group.link_eval) for group in groups]


def resolve_hdf5_config(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    data, run = config.evaluation_data, config.evaluation_run
    if data.get("backend") != "hdf5_v2":
        raise ValueError(
            "Compact root/DOF evaluation requires backend=hdf5_v2"
        )
    if (
        config.get("ckpt_onnx_root_dir")
        or config.get("ckpt_onnx_names") is not None
    ):
        raise ValueError(
            "HDF5 evaluation supports one explicit ckpt_onnx_path per job; "
            "ckpt_onnx_root_dir and ckpt_onnx_names are NPZ-only"
        )
    if config.get("metric_calculation", "per_clip") != "per_clip":
        raise ValueError(
            "HDF5 evaluation requires equal-clip metric_calculation=per_clip"
        )
    for option in ("generate_report", "dump_onnx_io_npy"):
        if config.get(option, False):
            raise ValueError(
                f"{option} is unsupported by HDF5 quantitative evaluation"
            )
    if int(config.get("max_policy_steps", 0)) != 0:
        raise ValueError(
            "HDF5 evaluation scores full clips; max_policy_steps must be 0"
        )
    npz_only = [
        key
        for key in config
        if key.startswith("ray_")
        or key
        in (
            "dof_mode",
            "metrics_threadpool_max_workers",
            "calc_per_clip_metrics",
            "dump_npzs",
        )
    ]
    if npz_only:
        warnings.warn(
            "HDF5 evaluation ignores NPZ-only options: "
            + ", ".join(sorted(npz_only))
            + ". Use link_eval for control points and "
            "evaluation_run.workers for parallelism; "
            "metrics are always computed and results are JSON, not NPZ.",
            UserWarning,
            stacklevel=2,
        )
    if not config.get("link_observation", {}).get("enabled", False):
        raise ValueError(
            "HDF5 control adapter requires a saved link-policy schema"
        )
    if config.get("motion_npz_dir") or config.get("motion_npz_path"):
        raise ValueError("HDF5 and legacy NPZ inputs cannot be mixed")
    if config.get("record_video", False) or config.get(
        "isaac_action_playback", False
    ):
        raise ValueError(
            "Quantitative HDF5 evaluation does not record video or "
            "replay actions"
        )
    if int(config.get("policy_action_delay_step", 0)) != 0:
        raise ValueError(
            "Formal evaluation requires zero additional action delay"
        )
    if not run.output_dir:
        raise ValueError("evaluation_run.output_dir must be a fresh directory")
    output = Path(run.output_dir).resolve()
    config.ckpt_onnx_path = str(
        Path(config.ckpt_onnx_path).resolve(strict=True)
    )
    onnx = Path(config.ckpt_onnx_path)
    manifest = Path(data.manifest_path).resolve(strict=True)
    for protected in (manifest.parent, onnx.parent.parent):
        if output.is_relative_to(protected) or protected.is_relative_to(
            output
        ):
            raise ValueError(
                "Evaluation output must not overlap checkpoint or data roots"
            )
    data.manifest_path = str(manifest)
    if data.get("selection_path"):
        data.selection_path = str(
            Path(data.selection_path).resolve(strict=True)
        )
        selection = json.loads(Path(data.selection_path).read_text())
        expected_hash = selection.get("manifest_sha256")
        if (
            data.get("require_manifest_fingerprint", False)
            and not expected_hash
        ):
            raise ValueError(
                "Formal selection must bind the HDF5 manifest SHA256"
            )
        if expected_hash and expected_hash != file_sha256(manifest):
            raise ValueError(
                "Selection fingerprint differs from the HDF5 manifest"
            )
    if data.get("require_manifest_fingerprint", False) and not data.get(
        "selection_path"
    ):
        raise ValueError(
            "Formal evaluation requires a fingerprinted selection file"
        )
    horizon = max(int(x) for x in config.link_observation.frame_offsets)
    if int(data.reference_horizon) != horizon:
        raise ValueError(
            "evaluation_data.reference_horizon differs from the saved actor"
        )
    if data.get("expected_dof_names") is not None and list(
        data.expected_dof_names
    ) != list(config.robot.dof_names):
        raise ValueError("HDF5 DOF order differs from the saved robot")
    data.expected_dof_names = list(config.robot.dof_names)
    data.fk_robot_file_path = str(
        Path(
            data.get("fk_robot_file_path")
            or config.robot.motion.fk_robot_file_path
        ).resolve(strict=True)
    )
    config.robot.motion.fk_robot_file_path = data.fk_robot_file_path
    config.robot_xml_path = str(
        Path(config.robot_xml_path).resolve(strict=True)
    )
    config.headless = True
    config.dump_npzs = False
    run.output_dir = str(output)
    if not run.model_label:
        run.model_label = onnx.stem
    workers, gpus = int(run.workers), int(run.gpus)
    if workers < 1 or gpus < 1 or gpus > 4:
        raise ValueError(
            "Use positive workers and 1..4 GPUs per evaluation job"
        )
    for _, options in evaluation_groups(config):
        resolve_control_links(config.link_observation.link_names, **options)
    dataset, index = build_hdf5_evaluation_dataset(config)
    try:
        if data.get("expected_clips") is not None and len(index) != int(
            data.expected_clips
        ):
            raise ValueError(
                "Selection count differs from the evaluation profile"
            )
        if workers > len(index):
            raise ValueError("Do not create empty rollout shards")
        manifest_doc = json.loads(manifest.read_text())
        candidates = [
            name
            for name in index
            if int(manifest_doc["clips"][name]["length"]) >= horizon + 8
        ]
        if not candidates:
            raise ValueError(
                "Selection must contain a clip covering eight parity steps"
            )
        run.gate_clip = candidates[0]
        if list(dataset._fk_transform._fk.body_names) != list(
            config.robot.body_names
        ):
            raise ValueError("HDF5 FK body order differs from the saved robot")
    finally:
        dataset.close()
    return config, list(index)


def runtime_identity(config):
    """Bind gate and workers to the same mounted inputs and source checkout."""
    onnx = Path(config.ckpt_onnx_path)
    actor_dir = onnx.parent.parent / onnx.stem / "actor"
    actor_file = next(
        (
            actor_dir / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (actor_dir / name).is_file()
        ),
        None,
    )
    if actor_file is None:
        raise FileNotFoundError(
            f"Actor checkpoint sidecar missing: {actor_dir}"
        )
    inputs = [
        onnx,
        onnx.parent.parent / "config.yaml",
        actor_file,
        Path(config.evaluation_data.manifest_path),
        Path(config.evaluation_data.fk_robot_file_path),
        Path(config.robot_xml_path),
    ]
    if config.evaluation_data.get("selection_path"):
        inputs.append(Path(config.evaluation_data.selection_path))
    # Include the packages implementing policy, FK, observation and metrics.
    root = Path(__file__).resolve().parents[1]
    source = sorted(
        p
        for folder in (
            "evaluation",
            "motion_tracking",
            "modules",
            "algo",
            "training",
        )
        for p in (root / folder).rglob("*.py")
    )
    body = {
        "inputs": {str(p): file_sha256(p) for p in inputs},
        "runtime": {str(p.relative_to(root)): file_sha256(p) for p in source},
        "config": OmegaConf.to_container(config, resolve=True),
    }
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest()
    return body


def verify_attestation(path, identity):
    gate = json.loads(Path(path).read_text())
    if (
        gate.get("status") != "passed"
        or gate.get("checked_steps") != 8
        or gate.get("identity_sha256") != identity["sha256"]
        or gate.get("torch_allow_tf32") is not False
    ):
        raise ValueError(
            "Missing, failed or mismatched checkpoint parity gate"
        )
    return gate


def setup_evaluator(config, *, gpu_id, tf32):
    from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.gpu_id, cfg.onnx_use_tf32 = gpu_id, bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    random.seed(int(cfg.evaluation_run.seed))
    np.random.seed(int(cfg.evaluation_run.seed))
    torch.manual_seed(int(cfg.evaluation_run.seed))
    evaluator = MujocoEvaluator(cfg)
    evaluator.setup()
    if (
        cfg.use_gpu
        and "CUDAExecutionProvider"
        not in evaluator.policy_session.get_providers()
    ):
        raise RuntimeError("GPU evaluation cannot silently fall back to CPU")
    if cfg.use_gpu:
        value = evaluator.policy_session.get_provider_options()[
            "CUDAExecutionProvider"
        ].get("use_tf32")
        if value != ("1" if tf32 else "0"):
            raise RuntimeError(
                "ONNX provider did not honor requested TF32 precision"
            )
    if not evaluator.use_kv_cache:
        raise ValueError(
            "Formal link evaluation requires the exported recurrent "
            "KV interface"
        )
    return evaluator


def load_clip(evaluator, dataset, clip_index, name):
    sample = dataset[clip_index[name]]
    fps = sample.tensors["motion_fps"].numpy()
    if not np.allclose(fps, 1.0 / evaluator.policy_dt):
        raise ValueError("HDF5 clip rate differs from the saved policy rate")
    arrays = {
        key: value.numpy().astype(np.float32, copy=False)
        for key, value in sample.tensors.items()
        if key.startswith("ref_")
    }
    evaluator.load_motion_arrays(
        arrays,
        motion_key=name,
        reference_horizon=max(evaluator.config.link_observation.frame_offsets),
    )
    evaluator.reset_state_teleport()
    return sample.length


def advance(evaluator, frame):
    evaluator.motion_frame_idx = frame
    evaluator._update_policy()
    if not np.isfinite(evaluator.actions_onnx).all():
        raise FloatingPointError("Non-finite policy action")
    evaluator._apply_control(sleep=False)
    evaluator.counter += 1
    if (
        not np.isfinite(evaluator.d.qpos).all()
        or not np.isfinite(evaluator.d.qvel).all()
    ):
        raise FloatingPointError("Non-finite MuJoCo state")


def run_parity_gate(config, names, identity):
    from holomotion.src.evaluation.checkpoint_onnx_parity import (
        CheckpointOnnxParity,
        actor_example_from_flat,
        construct_checkpoint_actor,
        load_actor_strict,
    )

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.link_eval = evaluation_groups(config)[0][1]
    evaluator = setup_evaluator(cfg, gpu_id=0, tf32=False)
    dataset, index = build_hdf5_evaluation_dataset(cfg)
    destination = (
        Path(config.evaluation_run.output_dir) / "checkpoint_parity.json"
    )
    parity = None
    try:
        load_clip(evaluator, dataset, index, config.evaluation_run.gate_clip)
        actor = construct_checkpoint_actor(
            cfg,
            actor_example_from_flat(
                np.zeros(
                    (1, evaluator.obs_builder.link_adapter.actor_dim),
                    np.float32,
                ),
                cfg,
            ),
        )
        onnx = Path(cfg.ckpt_onnx_path)
        metadata = load_actor_strict(
            actor, onnx.parent.parent / onnx.stem / "actor"
        )
        parity = CheckpointOnnxParity(
            actor,
            evaluator.policy_session,
            {
                "obs": evaluator.policy_input_name,
                "past": evaluator.policy_kv_input_name,
                "step": evaluator.policy_step_input_name,
                "actions": evaluator.policy_output_name,
                "present": evaluator.policy_kv_output_name,
            },
            steps=8,
            device="cuda:0" if cfg.use_gpu else "cpu",
            checkpoint_metadata=metadata,
            actor_config=cfg.algo.config.module_dict.actor,
        )

        def check(inputs, outputs):
            parity.check_step(inputs, outputs)
            report = parity.report()
            if report["error"] or any(
                not row["actions"]["passed"]
                or not row["present_key_values"]["passed"]
                for row in report["steps"]
            ):
                raise ValueError("FP32 checkpoint parity mismatch")

        evaluator._policy_parity_callback = check
        for frame in range(8):
            advance(evaluator, frame)
        report = parity.report()
        report["identity_sha256"] = identity["sha256"]
        write_json(destination, report)
        verify_attestation(destination, identity)
    except Exception as exc:
        write_json(
            destination,
            {
                **(parity.report() if parity is not None else {}),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "identity_sha256": identity["sha256"],
            },
        )
        raise
    finally:
        dataset.close()


def run_rollout_worker(config, names, group_index, worker):
    identity = runtime_identity(config)
    output = Path(config.evaluation_run.output_dir)
    verify_attestation(output / "checkpoint_parity.json", identity)
    phase, options = evaluation_groups(config)[group_index]
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.link_eval = options
    assigned = names[worker :: int(cfg.evaluation_run.workers)]
    cfg.evaluation_data.clip_names = assigned
    cfg.evaluation_data.selection_path = None
    evaluator = setup_evaluator(
        cfg,
        gpu_id=worker % int(cfg.evaluation_run.gpus),
        tf32=bool(cfg.evaluation_run.rollout_tf32),
    )
    dataset, index = build_hdf5_evaluation_dataset(cfg)
    root = output / phase / "results" / f"worker{worker}"
    root.mkdir(parents=True, exist_ok=False)
    records = []
    horizon = int(cfg.evaluation_data.reference_horizon)
    try:
        for name in assigned:
            row = {
                "clip": name,
                "status": "running",
                "executed_policy_steps": 0,
            }
            try:
                frames = load_clip(evaluator, dataset, index, name)
                steps = frames - horizon
                row.update(
                    reference_frames=frames, expected_policy_steps=steps
                )
                adapter = evaluator.obs_builder.link_adapter
                tracker = MujocoMetricTracker.from_config(
                    evaluator.m, adapter.link_names, cfg
                )
                if not np.array_equal(
                    tracker.control_mask,
                    adapter.requested_mask[0].numpy().astype(bool),
                ):
                    raise ValueError("Control mask and metric points differ")
                for frame in range(steps):
                    evaluator.motion_frame_idx = frame
                    pos = evaluator.ref_global_translation[
                        frame, adapter.ref_indices
                    ]
                    quat = evaluator.ref_global_rotation_quat_xyzw[
                        frame, adapter.ref_indices
                    ][:, [3, 0, 1, 2]]
                    tracker.observe(evaluator.d, pos, quat, frame)
                    advance(evaluator, frame)
                    row["executed_policy_steps"] += 1
                row.update(
                    status="completed", common_metrics=tracker.report(steps)
                )
            except Exception as exc:
                row.update(
                    status="failed", error=f"{type(exc).__name__}: {exc}"
                )
                raise
            finally:
                records.append(row)
                write_json(root / "per_clip.json", records)
        write_json(
            root / "summary.json",
            {
                "status": "completed",
                "clips": len(records),
                "identity_sha256": identity["sha256"],
                "parity_steps": 0,
            },
        )
    finally:
        dataset.close()


def launch_stage(config_path, *args):
    """Fresh processes release the gate's CUDA/Torch state before rollout."""
    log_path = config_path.parent / (
        "_".join(str(x).lstrip("-") for x in args) + ".log"
    )
    with log_path.open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                "--resolved-config",
                str(config_path),
                *map(str, args),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_hdf5_evaluation(config):
    config, names = resolve_hdf5_config(config)
    output = Path(config.evaluation_run.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "resolved_config.yaml"
    OmegaConf.save(config, config_path)
    identity = runtime_identity(config)
    write_json(output / "identity.json", identity)
    selection = output / "selection.json"
    write_json(selection, {"val_handpicked_motion_names": names})
    try:
        launch_stage(config_path, "--gate")
        verify_attestation(output / "checkpoint_parity.json", identity)
        summaries = {}
        for group, (phase, options) in enumerate(evaluation_groups(config)):
            with ThreadPoolExecutor(
                max_workers=int(config.evaluation_run.workers)
            ) as pool:
                futures = [
                    pool.submit(
                        launch_stage,
                        config_path,
                        "--group",
                        group,
                        "--worker",
                        worker,
                    )
                    for worker in range(int(config.evaluation_run.workers))
                ]
                for future in futures:
                    future.result()
            summary = aggregate_shards(
                phase_dir=output / phase,
                workers=int(config.evaluation_run.workers),
                expected_clips=len(names),
                selection_path=selection,
                model=str(config.evaluation_run.model_label),
                phase=phase,
                output_dir=output / phase / "aggregate",
                failure_pos_err_thresh_m=float(
                    config.failure_pos_err_thresh_m
                ),
                link_eval=OmegaConf.to_container(options),
                reference_horizon=int(
                    config.evaluation_data.reference_horizon
                ),
            )
            if summary["status"] != "completed":
                raise RuntimeError(f"Incomplete evaluation group: {phase}")
            summaries[phase] = summary
        write_json(
            output / "summary.json",
            {
                "status": "completed",
                "groups": summaries,
                "identity_sha256": identity["sha256"],
            },
        )
        write_json(output / "done.json", {"status": "completed"})
    except Exception as exc:
        write_json(
            output / "failed.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Internal gate/worker for the MuJoCo HDF5 entry"
    )
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--group", type=int)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    config = OmegaConf.load(args.resolved_config)
    names = json.loads(
        (args.resolved_config.parent / "selection.json").read_text()
    )["val_handpicked_motion_names"]
    if args.gate:
        if args.group is not None or args.worker is not None:
            parser.error("The gate cannot be a rollout worker")
        run_parity_gate(config, names, runtime_identity(config))
    else:
        if (
            args.group is None
            or args.worker is None
            or not 0 <= args.worker < config.evaluation_run.workers
        ):
            parser.error("Rollout requires a valid group and worker")
        run_rollout_worker(config, names, args.group, args.worker)


if __name__ == "__main__":
    main()
