"""Warp buffers and compiled observations for the shared GPU evaluation core."""

from pathlib import Path
from collections import OrderedDict
import importlib.metadata
import json
import math
import time

import numpy as np
import onnxruntime as ort
import warp as wp

from .onnx_device import create_device_session, verify_cuda_execution
from .compiled_artifacts import (
    DIRECTORY as GRAPHS,
    schema as compiled_schema,
    scene_layout,
)
from .model_contract import INPUT_LINK_NAMES, sha256
from .observation_fk import CurrentObservationFK

SESSIONS = OrderedDict()
SCENES = {}
METRICS = {}


def view_float(value, shape=None):
    count = value.size * (
        value.dtype._length_ if hasattr(value.dtype, "_length_") else 1
    )
    # Spatial vectors and matrices expose scalar width through _length_.
    if shape is None:
        shape = (count,)
    return wp.array(
        ptr=value.ptr,
        dtype=wp.float32,
        shape=shape,
        device=value.device,
        copy=False,
        deleter=None,
    )


@wp.kernel
def check_finite_kernel(
    values: wp.array(dtype=float), failed: wp.array(dtype=int)
):
    i = wp.tid()
    if not wp.isfinite(values[i]):
        wp.atomic_max(failed, 0, 1)


def check_finite(*arrays):
    device = arrays[0].device
    failed = wp.zeros(1, dtype=int, device=device)
    for array in arrays:
        values = view_float(array)
        wp.launch(
            check_finite_kernel,
            dim=values.size,
            inputs=[values, failed],
            device=device,
        )
    if failed.numpy()[0]:
        raise ValueError(
            "Nonfinite physical state, observation, frozen action or recurrent cache"
        )


class Bound:
    def __init__(self, session):
        self.session = session
        self.device = (
            "cuda:"
            + session.get_provider_options()["CUDAExecutionProvider"][
                "device_id"
            ]
        )
        self.binding = session.io_binding()
        self.inputs, self.outputs = {}, {}
        self.input_shapes, self.output_shapes = {}, {}
        for nodes, destination, output in (
            (session.get_inputs(), self.inputs, False),
            (session.get_outputs(), self.outputs, True),
        ):
            for node in nodes:
                # Competitor models may name their batch dimension. The
                # public contract fixes the executed batch to one.
                shape = tuple(
                    x if isinstance(x, int) else 1 for x in node.shape
                )
                types = {
                    "tensor(float)": (wp.float32, np.float32),
                    "tensor(int64)": (wp.int64, np.int64),
                    "tensor(bool)": (wp.bool, np.bool_),
                }
                if node.type not in types:
                    raise ValueError(
                        f"Unsupported bound dtype: {node.name}: {node.type}"
                    )
                dtype, ndtype = types[node.type]
                storage_shape = (
                    shape if len(shape) <= 4 else (math.prod(shape),)
                )
                value = wp.zeros(
                    storage_shape, dtype=dtype, device=self.device
                )
                destination[node.name] = value
                if output:
                    self.output_shapes[node.name] = shape
                    self.binding.bind_output(
                        node.name,
                        "cuda",
                        int(self.device.split(":")[1]),
                        ndtype,
                        shape,
                        value.ptr,
                    )
                else:
                    self.input_shapes[node.name] = shape
                    self.binding.bind_input(
                        node.name,
                        "cuda",
                        int(self.device.split(":")[1]),
                        ndtype,
                        shape,
                        value.ptr,
                    )

    def run(self):
        self.session.run_with_iobinding(self.binding)
        wp.synchronize_device(self.device)
        verify_cuda_execution(self.session)
        return self.outputs

    def physical_inputs(self, scene):
        data = scene._gpu_physics.data
        for name in ("qpos", "qvel", "xpos", "xquat", "cvel", "com"):
            if name in self.inputs:
                if name in ("xpos", "xquat"):
                    source = getattr(
                        scene._observation_fk,
                        "positions" if name == "xpos" else "rotations",
                    )
                else:
                    source = getattr(
                        data, "subtree_com" if name == "com" else name
                    )
                wp.copy(
                    self.inputs[name],
                    view_float(source, self.inputs[name].shape),
                )

        if "reference_root" in self.inputs:
            anchor = getattr(scene, "_source_reference_root_anchor", None)
            destination = self.inputs["reference_root"]
            if anchor is None:
                positions = view_float(scene._observation_fk.positions)
                for fighter in range(2):
                    body = scene.model.body(
                        f"p{fighter + 1}_" + INPUT_LINK_NAMES[0]
                    ).id
                    wp.copy(
                        view_float(destination),
                        positions,
                        dest_offset=fighter * 3,
                        src_offset=body * 3,
                        count=3,
                    )
            else:
                if hasattr(anchor, "detach"):
                    anchor = anchor.detach().cpu().numpy()
                elif isinstance(anchor, wp.array):
                    anchor = anchor.numpy()
                anchor = np.asarray(anchor, dtype=np.float32)
                if anchor.shape != (2, 3) or not np.isfinite(anchor).all():
                    raise ValueError(
                        "Source reference root must be finite [2, 3]"
                    )
                destination.assign(anchor)

    def compare_cpu(self):
        if not hasattr(self, "cpu_session"):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self.cpu_session = ort.InferenceSession(
                self.session.boxing_model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        feed = {
            key: value.numpy().reshape(self.input_shapes[key])
            for key, value in self.inputs.items()
        }
        expected = self.cpu_session.run(None, feed)
        for node, value in zip(self.cpu_session.get_outputs(), expected):
            actual = (
                self.outputs[node.name]
                .numpy()
                .reshape(self.output_shapes[node.name])
            )
            error = float(np.max(np.abs(value - actual)))
            METRICS["max_observation_error"] = max(
                METRICS.get("max_observation_error", 0.0), error
            )
            np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)


@wp.kernel
def scale_targets(
    action: wp.array2d(dtype=float),
    default: wp.array(dtype=float),
    scale: wp.array(dtype=float),
    order: wp.array(dtype=int),
    fighter: int,
    targets: wp.array(dtype=float),
    packet: wp.array2d(dtype=float),
):
    i = wp.tid()
    j = order[i]
    value = default[j] + scale[j] * action[0, j]
    targets[fighter * 29 + i] = value
    packet[0, i] = action[0, i]
    packet[1, i] = value


def install():
    from . import lower, upper, match_runtime as match
    from .hand_rule_runtime import install as install_hands

    install_hands()
    original_session = create_device_session

    def cached_session(path, *args, **kwargs):
        key = (str(Path(path).resolve()), repr(args), repr(kwargs))
        if key not in SESSIONS:
            SESSIONS[key] = original_session(path, *args, **kwargs)
            while len(SESSIONS) > 16:
                SESSIONS.popitem(last=False)
        return SESSIONS[key]

    lower.create_device_session = upper.create_device_session = cached_session
    original_scene = match.BoxingMatchScene
    current = {}

    def cached_scene(**kwargs):
        key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
        if key not in SCENES:
            s = original_scene(**kwargs)
            s._actual_close = s.close
            s.close = lambda: None
            SCENES[key] = s
        s = SCENES[key]
        s.__dict__.pop("_source_reference_root_anchor", None)
        if not hasattr(s, "_observation_fk"):
            s._observation_fk = CurrentObservationFK(s)
        s._observation_fk.refreshes = 0
        s.prepare_actor_observation = s._observation_fk.refresh
        s._gpu_referee, s._upper_pair = None, None
        s._live_policies = []
        s._pending_contacts = None
        s._gpu_physics.steps = s._gpu_physics.gpu_frames = 0
        current["scene"] = s
        return s

    match.BoxingMatchScene = cached_scene
    old_referee = match.MatchStateMachine

    def referee(rules):
        if rules.num_rounds != 1:
            return old_referee(rules)
        from .gpu_referee import CudaReferee

        s = current["scene"]
        result = CudaReferee(rules, s)
        s._gpu_referee = result
        return result

    match.MatchStateMachine = referee
    old_init, old_reset, old_attach = (
        lower.LinkPolicy.__init__,
        lower.LinkPolicy.reset,
        lower.LinkPolicy.attach_scene,
    )

    def policy_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.bound = Bound(self.session)
        self.default_gpu = wp.array(
            self.default_joint_position, device=str(self.device)
        )
        self.scale_gpu = wp.array(self.action_scale, device=str(self.device))
        self.order_gpu = wp.array(
            self.model_to_articulation.astype(np.int32),
            device=str(self.device),
        )
        self.packet = wp.zeros((2, 29), dtype=float, device=str(self.device))
        self.packet_host = wp.zeros(
            (2, 29), dtype=float, device="cpu", pinned=True
        )
        self.compare = False

    def policy_reset(self):
        old_reset(self)
        if hasattr(self, "bound"):
            for name in ("past_key_values", "step_idx"):
                self.bound.inputs[name].zero_()
            self.bound.outputs["actions"].zero_()

    def policy_attach(self, scene):
        old_attach(self, scene)
        scene._live_policies.append(self)

    def act(
        self, *, reference_qpos, kinematics, fighter, reference_current_index=0
    ):
        self.features.physical_inputs(self.scene)
        self.features.inputs["target_pose"].assign(reference_qpos)
        wp.copy(
            self.features.inputs["last_action"],
            view_float(self.bound.outputs["actions"], (29,)),
        )
        self.features.run()
        check_finite(self.features.outputs["obs"])
        if self.compare:
            self.features.compare_cpu()
        wp.copy(self.bound.inputs["obs"], self.features.outputs["obs"])
        self.bound.inputs["step_idx"].assign(
            np.asarray([self.step_index], np.int64)
        )
        feed = (
            {
                key: value.numpy().reshape(self.bound.input_shapes[key])
                for key, value in self.bound.inputs.items()
            }
            if self.compare
            else None
        )
        start = time.perf_counter_ns()
        output = self.bound.run()
        end = time.perf_counter_ns()
        if self.compare:
            expected = self.session.run(
                ["actions", "present_key_values"], feed
            )
            for name, value in zip(
                ("actions", "present_key_values"), expected
            ):
                actual = (
                    output[name]
                    .numpy()
                    .reshape(self.bound.output_shapes[name])
                )
                error = float(np.max(np.abs(actual - value)))
                METRICS["max_binding_action_error"] = max(
                    METRICS.get("max_binding_action_error", 0.0), error
                )
                np.testing.assert_allclose(actual, value, atol=1e-6, rtol=1e-6)
        check_finite(output["actions"], output["present_key_values"])
        wp.copy(
            self.bound.inputs["past_key_values"], output["present_key_values"]
        )
        self.step_index += 1
        if self.step_index >= self.maximum_context_length:
            self.step_index = 0
            self.bound.inputs["past_key_values"].zero_()
        wp.launch(
            scale_targets,
            dim=29,
            inputs=[
                output["actions"],
                self.default_gpu,
                self.scale_gpu,
                self.order_gpu,
                fighter,
                self.scene._gpu_physics.targets_gpu,
                self.packet,
            ],
            device=str(self.device),
        )
        self.scene._targets_on_gpu = True
        wp.copy(self.packet_host, self.packet)
        wp.synchronize_device(str(self.device))
        packet = self.packet_host.numpy()
        self.last_action = packet[0].copy()
        return self.last_action, packet[1].copy(), start, end

    lower.LinkPolicy.__init__ = policy_init
    lower.LinkPolicy.reset = policy_reset
    lower.LinkPolicy.attach_scene = policy_attach
    lower.LinkPolicy.act = act
    old_pair_init, old_pair_reset = (
        upper.FivePointReferencePair.__init__,
        upper.FivePointReferencePair.reset,
    )

    def pair_reset(self):
        old_pair_reset(self)
        if hasattr(self, "features"):
            self.features.inputs["reset_history"].assign(np.asarray([True]))

    def pair_init(self, paths, scene, policies, **kwargs):
        saved = compiled_schema()
        if (
            scene_layout(scene, saved["lower_link_names"])
            != saved["scene_layout"]
        ):
            raise ValueError(
                "Compiled observation graphs belong to another scene layout"
            )
        old_pair_init(self, paths, scene, policies, **kwargs)
        self.features = Bound(
            original_session(
                GRAPHS / "upper_features.onnx",
                device_id=self.device.index or 0,
                cuda_graph=True,
            )
        )
        self.bound = [Bound(s) for s in self.sessions]
        self.compare = False
        self.features.inputs["reset_history"].assign(np.asarray([True]))
        for f, policy in enumerate(policies):
            policy.features = Bound(
                original_session(
                    GRAPHS / f"lower_features_{f}.onnx",
                    device_id=self.device.index or 0,
                    cuda_graph=True,
                )
            )
        scene._upper_pair = self
        self.metadata.update(
            upper_observation_device=str(self.device),
            lower_observation_device=str(self.device),
            optimization_version="warp-ort-observation-v1",
            kv_cache_device=str(self.device),
            tensor_runtime="Warp + ONNX Runtime",
            compiled_observation_manifest_sha256=sha256(
                GRAPHS / "manifest.json"
            ),
            compiled_observation_export_parity=json.loads(
                (GRAPHS / "export-parity.json").read_text()
            ),
        )

    def window(self, kinematics, *, fighters=(0, 1)):
        from .policy_output import sanitize_target_pose

        data = self.scene._gpu_physics.data
        check_finite(data.qpos, data.qvel, data.xpos, data.xquat)
        self.features.physical_inputs(self.scene)
        remaining = max(
            0.0,
            self.match.rules.round_duration_s
            - (self.match._last_sim_time_s - self.match.round_start_s),
        )
        self.features.inputs["clock_score"].assign(
            np.asarray([remaining, *self.match.score], np.float32)
        )
        features = self.features.run()
        check_finite(features["obs"])
        if self.compare:
            self.features.compare_cpu()
        for output, input_name in (
            ("history", "raw_history"),
            ("position", "previous_p"),
            ("quaternion", "previous_q"),
        ):
            wp.copy(self.features.inputs[input_name], features[output])
        self.features.inputs["reset_history"].assign(np.asarray([False]))
        results, starts, ends, sanitized, timings = [], [], [], [], []
        for fighter in fighters:
            bound = self.bound[fighter]
            wp.copy(
                bound.inputs["obs"],
                view_float(features["obs"]),
                src_offset=fighter * 1097,
                count=1097,
            )
            start = time.perf_counter_ns()
            output = bound.run()["target_pose"]
            end = time.perf_counter_ns()
            host_output = output.numpy()
            sanitation_start = time.perf_counter_ns()
            cleaned, statistics = sanitize_target_pose(host_output)
            timings.append(time.perf_counter_ns() - sanitation_start)
            results.append(cleaned)
            starts.append(start)
            ends.append(end)
            sanitized.append(statistics)
            self.sanitization_total[fighter] += statistics[
                "nonfinite_components"
            ]
        self.history_steps += 1
        self.frame_data = dict(
            upper_policy_start_ns=starts,
            upper_policy_end_ns=ends,
            upper_history_step=self.history_steps,
            competitor_sanitization=sanitized,
            competitor_sanitization_total=self.sanitization_total.copy(),
            competitor_sanitization_ns=timings,
        )
        return results

    upper.FivePointReferencePair.__init__ = pair_init
    upper.FivePointReferencePair.reset = pair_reset
    upper.FivePointReferencePair.window = window

    def read_kinematics(s):
        from .scene import SceneKinematics, _quat_to_matrix_wxyz

        d = s.data
        root = np.stack([d.qpos[q : q + 3] for q in s._root_qpos]).astype(
            np.float32
        )[None]
        quat = np.stack([d.qpos[q + 3 : q + 7] for q in s._root_qpos]).astype(
            np.float32
        )[None]
        linear = np.stack([d.qvel[v : v + 3] for v in s._root_dof]).astype(
            np.float32
        )[None]
        angular = np.stack(
            [d.qvel[v + 3 : v + 6] for v in s._root_dof]
        ).astype(np.float32)[None]
        raw_q = np.stack([d.qpos[q + 3 : q + 7] for q in s._root_qpos])
        raw_w = np.stack([d.qvel[v + 3 : v + 6] for v in s._root_dof])
        world = np.stack(
            [_quat_to_matrix_wxyz(q) @ v for q, v in zip(raw_q, raw_w)]
        ).astype(np.float32)[None]
        jp = d.qpos[np.asarray(s._joint_qpos)].astype(np.float32)[None]
        jv = d.qvel[np.asarray(s._joint_dof)].astype(np.float32)[None]
        bp = np.zeros((1, 2, len(s.body_names), 3), dtype=np.float32)
        return SceneKinematics(
            root, quat, linear, angular, world, jp, jv, bp, np.zeros_like(bp)
        )

    original_scene.read_kinematics = read_kinematics


def set_comparison(pair, policy, enabled):
    pair.compare = policy.compare = enabled


def synchronize(device):
    wp.synchronize_device(device)


def runtime_environment(device):
    return dict(
        gpu=wp.get_device(device).name,
        torch_version=None,
        tensor_runtime="Warp + ONNX Runtime",
        warp_version=wp.__version__,
        cuda_version=importlib.metadata.version("nvidia-cuda-runtime-cu12"),
        onnxruntime_version=ort.__version__,
    )
