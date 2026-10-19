"""The pinned HoloMotion V2 lower controller and its reference observation."""

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import yaml

from ..models import frozen_manifest, verify_bundle
from .onnx_device import (
    create_device_session,
    require_gpu,
    require_gpu_session,
    verify_cuda_execution,
)
from .model_contract import INPUT_LINK_NAMES
from . import compiled_artifacts

VERSION = "2.0_link_10500"
MODEL_ROOT = None
CONFIG = None
DEPLOYMENT = None
_verified_root = None


def configure_bundle(directory):
    global MODEL_ROOT, CONFIG, DEPLOYMENT, _verified_root
    requested = Path(directory).expanduser().resolve(strict=True)
    if _verified_root != requested:
        MODEL_ROOT = verify_bundle(requested)
        CONFIG = yaml.safe_load((MODEL_ROOT / "config.yaml").read_text())
        schema = CONFIG["link_observation"]
        if compiled_artifacts.available():
            layout = compiled_artifacts.schema()["layout"]
        else:
            from holomotion.src.motion_tracking.link_actor_packing import (
                link_actor_layout,
            )

            layout = link_actor_layout(
                schema, CONFIG["modules"]["actor"]["obs_schema"]
            )
        if (
            layout["actor_dim"] != 1378
            or schema["schema_version"] != "g1_link_xz6d_v1"
            or schema["frame_offsets"] != [1, 2, 3, 4, 5]
            or schema.get("include_error_terms") is not True
            or CONFIG["obs"]["target_fps"] != 50
        ):
            raise ValueError("The frozen HoloMotion V2 schema is incompatible")
        saved = frozen_manifest()["files"]
        DEPLOYMENT = {
            "model_sha256": saved["exported/model_10500.onnx"]["sha256"],
            "config_sha256": saved["config.yaml"]["sha256"],
        }
        _verified_root = requested
    return MODEL_ROOT


def verify_checkpoint(path):
    if MODEL_ROOT is None:
        raise RuntimeError(
            "An explicit verified --lower V2 bundle is required"
        )
    if (
        Path(path).resolve(strict=True)
        != (MODEL_ROOT / "exported/model_10500.onnx").resolve()
    ):
        raise ValueError(
            "Only the pinned HoloMotion V2 checkpoint is supported"
        )
    return MODEL_ROOT / "config.yaml"


def name_order(source, target, label):
    lookup = {name: index for index, name in enumerate(source)}
    if len(lookup) != len(source) or any(
        name not in lookup for name in target
    ):
        raise ValueError(f"Joint-name mapping differs: {label}")
    return np.asarray([lookup[name] for name in target], dtype=np.int64)


class LinkPolicy:
    def __init__(
        self,
        checkpoint,
        *,
        device,
        articulation_joint_names,
        retarget_dof_names,
    ):
        self.device = require_gpu(device)
        self.path = Path(checkpoint).resolve(strict=True)
        verify_checkpoint(self.path)
        options = ort.SessionOptions()
        self.session = create_device_session(
            self.path, device_id=self.device.index or 0, options=options
        )
        require_gpu_session(self.session)
        self.inputs = {node.name: node for node in self.session.get_inputs()}
        self.outputs = {node.name: node for node in self.session.get_outputs()}
        expected = {
            "obs": [1, 1378],
            "past_key_values": [1, 2, 1, 32, 4, 64],
            "step_idx": [1],
        }
        if {
            name: node.shape for name, node in self.inputs.items()
        } != expected:
            raise ValueError("The frozen V2 input contract differs")
        if (
            self.outputs["actions"].shape != [1, 29]
            or self.outputs["present_key_values"].shape
            != expected["past_key_values"]
        ):
            raise ValueError("The frozen V2 output contract differs")
        self.obs_name, self.action_name = "obs", "actions"
        self.cache_input_name, self.cache_output_name, self.step_name = (
            "past_key_values",
            "present_key_values",
            "step_idx",
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = tuple(metadata["joint_names"].split(","))
        for attribute, key in (
            ("default_joint_position", "default_joint_pos"),
            ("action_scale", "action_scale"),
            ("kp", "joint_stiffness"),
            ("kd", "joint_damping"),
        ):
            value = np.asarray(
                [float(x) for x in metadata[key].split(",")], dtype=np.float32
            )
            if value.shape != (29,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid frozen controller metadata: {key}")
            setattr(self, attribute, value)
        if len(set(self.joint_names)) != 29:
            raise ValueError(
                "The frozen V2 model must define 29 unique joints"
            )
        self.articulation_to_model = name_order(
            articulation_joint_names, self.joint_names, "scene to model"
        )
        self.model_to_articulation = name_order(
            self.joint_names, articulation_joint_names, "model to scene"
        )
        self.retarget_to_model = name_order(
            retarget_dof_names, self.joint_names, "reset pose to model"
        )
        self.maximum_context_length = min(
            32, int(metadata["rope_max_seq_len"])
        )
        if self.maximum_context_length != 32:
            raise ValueError("The frozen V2 context must support 32 steps")
        self.actor_observation_source = (
            "holomotion.src.motion_tracking.link_observation"
        )
        self.last_action = np.zeros(29, dtype=np.float32)
        self.reset()

    def reset(self):
        self.cache = np.zeros((1, 2, 1, 32, 4, 64), dtype=np.float32)
        self.step_index = 0
        self.last_action.fill(0.0)

    def attach_scene(self, scene):
        self.scene = scene
        self.body_ids = [
            np.asarray(
                [
                    scene.model.body(prefix + name).id
                    for name in CONFIG["link_observation"]["link_names"]
                ]
            )
            for prefix in ("p1_", "p2_")
        ]

    def observation(
        self, targets, kinematics, fighter, current_index, *, data=None
    ):
        import torch

        from holomotion.src.motion_tracking.link_actor_packing import (
            pack_link_actor_observation,
        )
        from holomotion.src.motion_tracking.link_observation import (
            LinkObservationInput,
            build_link_observation,
            quat_apply_wxyz,
            quat_conjugate_wxyz,
            quat_multiply_wxyz,
        )

        # Independent eager reference for the captured GPU observation path.
        ids = self.body_ids[fighter]
        scene = self.scene
        data = scene.data if data is None else data
        qi, vi = scene._root_qpos[fighter], scene._root_dof[fighter]
        for field in (data.qpos, data.qvel, data.xpos, data.xquat):
            if not np.isfinite(field).all():
                raise ValueError(
                    "Nonfinite measured physics state; no sanitation is allowed"
                )
        root = torch.tensor(data.qpos[qi : qi + 3], dtype=torch.float32)[None]
        quaternion = torch.tensor(
            data.qpos[qi + 3 : qi + 7], dtype=torch.float32
        )[None]
        inverse = quat_conjugate_wxyz(quaternion)
        actual_position = torch.tensor(data.xpos[ids], dtype=torch.float32)[
            None
        ]
        actual_quaternion = torch.tensor(data.xquat[ids], dtype=torch.float32)[
            None
        ]
        local_position = quat_apply_wxyz(
            inverse[:, None], actual_position - root[:, None]
        )
        local_quaternion = quat_multiply_wxyz(
            inverse[:, None], actual_quaternion
        )
        frames, links = 5, len(ids)
        future_position = (
            actual_position[:, None].expand(1, frames, links, 3).clone()
        )
        future_quaternion = (
            actual_quaternion[:, None].expand(1, frames, links, 4).clone()
        )
        active = [
            CONFIG["link_observation"]["link_names"].index(name)
            for name in INPUT_LINK_NAMES
        ]
        future_position[0][:, active] = torch.tensor(
            targets.future_positions, dtype=torch.float32
        )
        future_quaternion[0][:, active] = torch.tensor(
            targets.future_rotations_wxyz, dtype=torch.float32
        )
        mask = torch.tensor(
            [
                name in INPUT_LINK_NAMES
                for name in CONFIG["link_observation"]["link_names"]
            ]
        )
        command = build_link_observation(
            LinkObservationInput(
                reference_link_pos=future_position,
                reference_link_quat_wxyz=future_quaternion,
                reference_root_pos=torch.tensor(
                    targets.current_root_position, dtype=torch.float32
                )[None],
                reference_root_height=torch.tensor(
                    [[targets.current_root_position[2]]], dtype=torch.float32
                ),
                robot_root_quat_wxyz=quaternion,
                robot_link_pos_local=local_position,
                robot_link_quat_local_wxyz=local_quaternion,
                frame_offsets=torch.tensor(
                    targets.frame_offsets, dtype=torch.float32
                ),
                link_mask=mask,
            ),
            root_link_index=0,
            include_error_terms=True,
        )
        proprioception = torch.cat(
            (
                quat_apply_wxyz(inverse, torch.tensor([[0.0, 0.0, -1.0]])),
                torch.tensor(data.qvel[vi + 3 : vi + 6], dtype=torch.float32)[
                    None
                ],
                torch.tensor(
                    kinematics.joint_position[0, fighter][
                        self.articulation_to_model
                    ]
                    - self.default_joint_position
                )[None],
                torch.tensor(
                    kinematics.joint_velocity[0, fighter][
                        self.articulation_to_model
                    ]
                )[None],
                torch.tensor(self.last_action)[None],
            ),
            dim=-1,
        )
        result = pack_link_actor_observation(
            proprioception, command, include_error_terms=True
        ).numpy()
        if result.shape != (1, 1378) or not np.isfinite(result).all():
            raise ValueError("Invalid frozen V2 observation")
        return result

    def act(
        self, *, reference_qpos, kinematics, fighter, reference_current_index=0
    ):
        import torch

        observation = self.observation(
            reference_qpos, kinematics, fighter, reference_current_index
        )
        feed = {
            "obs": observation,
            "past_key_values": self.cache,
            "step_idx": np.asarray([self.step_index], dtype=np.int64),
        }
        torch.cuda.synchronize(self.device)
        start = time.perf_counter_ns()
        values = self.session.run(["actions", "present_key_values"], feed)
        torch.cuda.synchronize(self.device)
        end = time.perf_counter_ns()
        verify_cuda_execution(self.session)
        action = np.asarray(values[0], dtype=np.float32).reshape(29)
        self.cache = np.asarray(values[1]).copy()
        if not np.isfinite(action).all() or not np.isfinite(self.cache).all():
            raise ValueError("Nonfinite frozen V2 output or recurrent state")
        self.step_index += 1
        if self.step_index >= self.maximum_context_length:
            self.step_index = 0
            self.cache.fill(0.0)
        self.last_action = action.copy()
        target = self.default_joint_position + self.action_scale * action
        return action, target[self.model_to_articulation], start, end
