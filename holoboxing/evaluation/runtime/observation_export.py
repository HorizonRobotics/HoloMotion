"""Build and verify observation graphs; PyTorch is a build-only dependency.

The source backend remains the independent numerical reference. Model weights
and recurrent inference are not converted or quantized by this exporter.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import json
import os
import tempfile
import warnings

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.spatial.transform import Rotation

from ..resources import ResourceTree
from . import lower, representation, scene as scene_module, upper
from .compiled_artifacts import APP_ROOT, available, scene_layout
from .gpu_physics import ResidentPhysics
from .model_contract import INPUT_LINK_NAMES, sha256
from .onnx_device import create_device_session, verify_cuda_execution
from .reference import ReferenceSchema


def _fingerprint(path):
    return dict(bytes=path.stat().st_size, sha256=sha256(path))


def _host_scene(resources):
    # This is build-time geometry and FK, never a CPU evaluation fallback.
    # The normal scene constructor still requires a real CUDA device.
    original_init = ResidentPhysics.__init__
    original_device = scene_module.require_gpu
    try:
        ResidentPhysics.__init__ = lambda *args, **kwargs: None
        scene_module.require_gpu = lambda device: torch.device(device)
        scene = scene_module.BoxingMatchScene(model_path=resources.scene)
    finally:
        ResidentPhysics.__init__ = original_init
        scene_module.require_gpu = original_device
    reset = json.loads(resources.initial_state.read_text())
    order = lower.name_order(
        reset["dof_names"], scene.articulation_joint_names, "reset"
    )
    qpos = np.asarray(reset["qpos"], np.float32)
    qpos = np.concatenate((qpos[:, :7], qpos[:, 7:][:, order]), axis=-1)
    for fighter, qi in enumerate(scene._root_qpos):
        scene.data.qpos[qi : qi + 36] = qpos[fighter]
    mujoco.mj_forward(scene.model, scene.data)
    return scene


def _algebraic_heading(q):
    # ORT 1.22 has no CUDA Atan kernel. Normalized projected XY gives the
    # same heading matrix without atan2/cos/sin. The unchanged trigonometric
    # function is the oracle for random and near-vertical export checks.
    r = representation.matrix(q)
    x, y = r[..., 0, 0], r[..., 1, 0]
    norm = torch.sqrt(x * x + y * y)
    safe = norm.clamp_min(torch.finfo(x.dtype).tiny)
    c = torch.where(norm > 0, x / safe, torch.ones_like(x))
    s = torch.where(norm > 0, y / safe, torch.zeros_like(y))
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(
        *c.shape, 3, 3
    )


def _cases(arguments, names, root_qpos, count=128):
    rng = np.random.default_rng(0x484F4C4F)
    # Include heading wraparound and nearly vertical pelvis orientations.
    rotations = [
        Rotation.from_euler("zyx", [yaw, pitch, 0]).as_quat()[[3, 0, 1, 2]]
        for yaw in (0, np.pi / 2, -np.pi / 2, np.pi, -np.pi)
        for pitch in (
            0,
            np.pi / 2,
            np.pi / 2 - 1e-7,
            np.pi / 2 + 1e-7,
            -np.pi / 2,
            -np.pi / 2 - 1e-7,
            -np.pi / 2 + 1e-7,
        )
    ]
    for sample in range(count + len(rotations)):
        args = [x.clone() for x in arguments]
        for index, (key, value) in enumerate(zip(names, args)):
            if key in ("qvel", "cvel", "last_action", "raw_history"):
                args[index] = value + torch.from_numpy(
                    rng.normal(0, 0.01, value.shape).astype(np.float32)
                )
            elif key == "reference_root":
                # Exercise the independent offline anchor, including height.
                args[index] = value + torch.from_numpy(
                    rng.normal(0, 0.3, value.shape).astype(np.float32)
                )
            elif key in ("xpos", "qpos", "previous_p"):
                args[index] = value + torch.from_numpy(
                    rng.normal(0, 0.001, value.shape).astype(np.float32)
                )
            elif key in ("xquat", "previous_q"):
                q = rng.normal(0, 1, value.shape).astype(np.float32)
                q /= np.linalg.norm(q, axis=-1, keepdims=True)
                if sample >= count:
                    q[...] = np.asarray(rotations[sample - count], np.float32)
                args[index] = torch.from_numpy(q)
            elif key == "target_pose":
                points = rng.normal(0, 0.2, (5, 5, 9)).astype(np.float32)
                q = torch.from_numpy(
                    rng.normal(0, 1, (5, 5, 4)).astype(np.float32)
                )
                points[..., 3:] = representation.rot_xz(
                    representation.matrix(q)
                ).numpy()
                args[index] = torch.from_numpy(points.reshape(1, 225))
            elif key == "reset_history":
                args[index] = torch.tensor([sample % 5 == 0])
            elif key == "clock_score":
                args[index] = torch.tensor(
                    [
                        max(0, 60 - sample * 0.5),
                        float(sample),
                        float(sample // 2),
                    ]
                )
        if "qpos" in names and sample >= count:
            qpos = args[names.index("qpos")]
            quaternion = torch.tensor(
                rotations[sample - count], dtype=torch.float32
            )
            for qi in root_qpos:
                qpos[qi + 3 : qi + 7] = quaternion
        yield args


def _verify_graph(
    path, module, arguments, names, outputs, root_qpos, cuda_check
):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cpu = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if {x.name for x in cpu.get_inputs()} != set(names):
        raise ValueError("Export dropped a required dynamic observation input")
    cuda = create_device_session(path) if cuda_check else None
    maxima, maxima_gpu = [0.0] * len(outputs), [0.0] * len(outputs)
    samples = 0
    for args in _cases(arguments, names, root_qpos):
        # This calls the original Pair methods, not the export wrappers.
        expected = module.eager_reference(*args)
        feed = {key: value.numpy() for key, value in zip(names, args)}
        actual = cpu.run(None, feed)
        gpu_values = cuda.run(None, feed) if cuda is not None else None
        if cuda is not None:
            verify_cuda_execution(cuda)
        for i, (baseline, value) in enumerate(zip(expected, actual)):
            baseline = baseline.numpy()
            maxima[i] = max(maxima[i], float(np.max(np.abs(baseline - value))))
            np.testing.assert_allclose(value, baseline, atol=3e-5, rtol=3e-5)
            if gpu_values is not None:
                value = gpu_values[i]
                maxima_gpu[i] = max(
                    maxima_gpu[i], float(np.max(np.abs(baseline - value)))
                )
                np.testing.assert_allclose(
                    value, baseline, atol=3e-5, rtol=3e-5
                )
        samples += 1
    return dict(
        name=path.stem,
        samples=samples,
        random_samples=128,
        adversarial_heading_samples=35,
        cpu_max_abs_error=maxima,
        cuda_max_abs_error=maxima_gpu if cuda is not None else None,
        cuda_verified=cuda.boxing_cuda_verified if cuda is not None else False,
        cuda_check="PASS" if cuda is not None else "NOT_RUN_BUILD_ONLY",
        inputs=[
            dict(name=x.name, shape=x.shape, type=x.type)
            for x in cpu.get_inputs()
        ],
        outputs=[
            dict(name=x.name, shape=x.shape, type=x.type)
            for x in cpu.get_outputs()
        ],
    )


def export_observations(assets, lower_directory, output, *, cuda_check=False):
    if available():
        raise ValueError(
            "Export from source without an existing compiled bundle"
        )
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    lower.configure_bundle(lower_directory)
    resources = ResourceTree(assets)
    scene = _host_scene(resources)
    graph = onnx.load(lower.MODEL_ROOT / "exported/model_10500.onnx")
    metadata = {x.key: x.value for x in graph.metadata_props}
    del graph
    gc.collect()
    joint_names = tuple(metadata["joint_names"].split(","))
    mapping = lower.name_order(
        scene.articulation_joint_names, joint_names, "graph"
    )
    body_ids = np.asarray(
        [
            [scene.model.body(prefix + name).id for name in INPUT_LINK_NAMES]
            for prefix in ("p1_", "p2_")
        ]
    )
    schema = ReferenceSchema()
    # Resolve the exact deployed lower observation method, including its
    # float64 decoding and canonical HoloMotion term-major packing.
    from .fast_runtime import install

    install()

    class UpperFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("ids", torch.tensor(body_ids))
            self.register_buffer(
                "roots", torch.tensor(scene.model.body_rootid[body_ids])
            )

        def physical_features(self, xpos, xquat, cvel, com, qvel):
            p = xpos[self.ids].float()
            q = xquat[self.ids].float()
            q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
            v = cvel[self.ids].float()
            velocity = v[..., 3:] + torch.cross(
                v[..., :3], p - com[self.roots].float(), dim=-1
            )
            omega = torch.stack(
                [qvel[vi + 3 : vi + 6].float() for vi in scene._root_dof]
            )
            return p, q, velocity, omega

        def forward(
            self,
            xpos,
            xquat,
            cvel,
            com,
            qvel,
            previous_p,
            previous_q,
            raw_history,
            clock_score,
            reset_history,
        ):
            p, q, velocity, omega = self.physical_features(
                xpos, xquat, cvel, com, qvel
            )
            pp = torch.where(reset_history, p, previous_p)
            pq = torch.where(reset_history, q, previous_q)
            raw = representation.raw_self(p, q, velocity, omega, pp, pq)
            history = torch.where(
                reset_history,
                raw[:, None].expand(-1, 10, -1),
                torch.cat((raw_history[:, 1:], raw[:, None]), 1),
            )
            state = representation.reframe_c(history, p, q, p.flip(0))
            score = clock_score[1:3]
            obs = torch.cat(
                (
                    state["history"],
                    state["task"][:, -15:],
                    clock_score[0].expand(2, 1),
                    (score - score.flip(0))[:, None],
                ),
                -1,
            )
            return obs, history, p, q

        def eager_reference(
            self,
            xpos,
            xquat,
            cvel,
            com,
            qvel,
            previous_p,
            previous_q,
            raw_history,
            clock_score,
            reset_history,
        ):
            p, q, velocity, omega = self.physical_features(
                xpos, xquat, cvel, com, qvel
            )
            reset = bool(reset_history.item())
            pair = SimpleNamespace(
                previous_position=None if reset else previous_p,
                previous_quaternion=None if reset else previous_q,
                raw_history=None if reset else raw_history,
                match=SimpleNamespace(
                    rules=SimpleNamespace(
                        round_duration_s=float(clock_score[0])
                    ),
                    _last_sim_time_s=0.0,
                    round_start_s=0.0,
                    score=clock_score[1:3],
                ),
            )
            obs = upper.FivePointReferencePair.observations(
                pair, p, q, velocity, omega
            )
            return (
                obs,
                pair.raw_history,
                pair.previous_position,
                pair.previous_quaternion,
            )

    class LowerFeatures(torch.nn.Module):
        def __init__(self, fighter):
            super().__init__()
            self.fighter = fighter
            self.register_buffer("template", schema.template.clone())
            self.register_buffer("indices", schema.learn_indices.clone())
            self.policy = SimpleNamespace(
                schema=schema,
                scene=SimpleNamespace(
                    _root_qpos=scene._root_qpos,
                    _root_dof=scene._root_dof,
                ),
                body_gpu=[
                    torch.tensor(
                        [
                            scene.model.body(prefix + name).id
                            for name in lower.CONFIG["link_observation"][
                                "link_names"
                            ]
                        ]
                    )
                    for prefix in ("p1_", "p2_")
                ],
                upper_body_host=body_ids,
                offsets=torch.tensor(schema.frame_offsets),
                mask=torch.tensor(
                    [
                        name in INPUT_LINK_NAMES
                        for name in lower.CONFIG["link_observation"][
                            "link_names"
                        ]
                    ]
                ),
                active=torch.tensor(schema.active),
                gravity=torch.tensor([[0.0, 0.0, -1.0]]),
                joint_ids=[
                    torch.tensor(scene._joint_qpos[f][mapping])
                    for f in range(2)
                ],
                vel_ids=[
                    torch.tensor(scene._joint_dof[f][mapping])
                    for f in range(2)
                ],
                default_gpu=torch.tensor(
                    [
                        float(x)
                        for x in metadata["default_joint_pos"].split(",")
                    ]
                ),
            )
            self.pair = SimpleNamespace(
                schemas=[schema, schema],
                references=torch.stack(
                    [schema.template.clone(), schema.template.clone()]
                ),
            )

        def forward(
            self,
            target_pose,
            qpos,
            qvel,
            xpos,
            xquat,
            last_action,
            reference_root,
        ):
            p = self.policy
            p.scene._source_reference_root_anchor = reference_root
            p.qpos, p.qvel, p.xpos, p.xquat, p.last_gpu = (
                qpos,
                qvel,
                xpos,
                xquat,
                last_action,
            )
            values = torch.cat(
                (
                    target_pose.reshape(-1),
                    xpos[body_ids[self.fighter, 0], 2].reshape(1),
                )
            )
            reference = self.template.scatter(0, self.indices, values)
            return lower.LinkPolicy.gpu_observation(
                p, reference, None, self.fighter, 0
            ), reference

        def eager_reference(
            self,
            target_pose,
            qpos,
            qvel,
            xpos,
            xquat,
            last_action,
            reference_root,
        ):
            p = self.policy
            p.scene._source_reference_root_anchor = reference_root
            p.qpos, p.qvel, p.xpos, p.xquat, p.last_gpu = (
                qpos,
                qvel,
                xpos,
                xquat,
                last_action,
            )
            reference = upper.FivePointReferencePair.reference(
                self.pair,
                target_pose[0],
                self.fighter,
                xpos[body_ids[self.fighter, 0], 2],
            )
            return lower.LinkPolicy.gpu_observation(
                p, reference, None, self.fighter, 0
            ), reference

    state = {
        name: torch.tensor(
            np.asarray(getattr(scene.data, name)), dtype=torch.float32
        )
        for name in ("qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com")
    }
    target = torch.zeros((5, 5, 9))
    target[..., 3], target[..., 8] = 1, 1
    upper_args = (
        state["xpos"],
        state["xquat"],
        state["cvel"],
        state["subtree_com"],
        state["qvel"],
        state["xpos"][torch.tensor(body_ids)],
        state["xquat"][torch.tensor(body_ids)],
        torch.zeros(2, 10, 108),
        torch.tensor([60.0, 0.0, 0.0]),
        torch.tensor([True]),
    )
    lower_args = (
        target.reshape(1, 225),
        state["qpos"],
        state["qvel"],
        state["xpos"],
        state["xquat"],
        torch.zeros(29),
        state["xpos"][torch.tensor(body_ids[:, 0])],
    )
    upper_names = [
        "xpos",
        "xquat",
        "cvel",
        "com",
        "qvel",
        "previous_p",
        "previous_q",
        "raw_history",
        "clock_score",
        "reset_history",
    ]
    lower_names = [
        "target_pose",
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "last_action",
        "reference_root",
    ]
    models = [
        (
            "upper_features",
            UpperFeatures(),
            upper_args,
            upper_names,
            ["obs", "history", "position", "quaternion"],
        )
    ]
    models.extend(
        (
            f"lower_features_{f}",
            LowerFeatures(f),
            lower_args,
            lower_names,
            ["obs", "reference"],
        )
        for f in range(2)
    )

    with tempfile.TemporaryDirectory(
        prefix=".observations-", dir=destination.parent
    ) as temporary:
        output = Path(temporary)
        reports = []
        for name, module, arguments, names, outputs in models:
            module.eval()
            path = output / (name + ".onnx")
            original_heading = representation.heading
            representation.heading = _algebraic_heading
            try:
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore", torch.jit.TracerWarning)
                    torch.onnx.export(
                        module,
                        arguments,
                        path,
                        input_names=names,
                        output_names=outputs,
                        opset_version=17,
                        dynamo=False,
                        do_constant_folding=True,
                    )
            finally:
                representation.heading = original_heading
            report = _verify_graph(
                path,
                module,
                arguments,
                names,
                outputs,
                scene._root_qpos,
                cuda_check,
            )
            reports.append(report)
            print(json.dumps(report), flush=True)
        from holomotion.src.motion_tracking.link_actor_packing import (
            link_actor_layout,
        )

        saved_schema = dict(
            frozen_files=lower.frozen_manifest()["files"],
            layout=link_actor_layout(
                lower.CONFIG["link_observation"],
                lower.CONFIG["modules"]["actor"]["obs_schema"],
            ),
            scene_layout=scene_layout(
                scene, lower.CONFIG["link_observation"]["link_names"]
            ),
            lower_link_names=list(
                lower.CONFIG["link_observation"]["link_names"]
            ),
            schema_template=schema.template.tolist(),
            schema_indices=schema.learn_indices.tolist(),
            schema_hashes=schema.source_hashes,
        )
        (output / "schema.json").write_text(json.dumps(saved_schema, indent=2))
        np.save(
            output / "neutral_observation.npy", upper.neutral_observation()
        )
        (output / "export-parity.json").write_text(
            json.dumps(
                dict(status="PASS", atol=3e-5, rtol=3e-5, graphs=reports),
                indent=2,
            )
        )
        sources = list(Path(__file__).parent.glob("*.py"))
        sources.extend(
            [
                APP_ROOT
                / "holomotion/src/motion_tracking/link_observation.py",
                APP_ROOT
                / "holomotion/src/motion_tracking/link_actor_packing.py",
            ]
        )
        record = dict(
            schema_version=1,
            frozen_files=lower.frozen_manifest()["files"],
            files={p.name: _fingerprint(p) for p in sorted(output.iterdir())},
            sources={
                p.relative_to(APP_ROOT).as_posix(): _fingerprint(p)
                for p in sorted(sources)
            },
        )
        (output / "manifest.json").write_text(json.dumps(record, indent=2))
        os.replace(output, destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--lower", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda-check", action="store_true")
    args = parser.parse_args()
    export_observations(
        args.assets, args.lower, args.output, cuda_check=args.cuda_check
    )


if __name__ == "__main__":
    main()
