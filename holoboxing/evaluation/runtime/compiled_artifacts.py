"""Verify build-time observation graphs without importing the build framework."""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np

from .model_contract import sha256

DIRECTORY = Path(__file__).with_name("compiled")
APP_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FILES = {
    "upper_features.onnx",
    "lower_features_0.onnx",
    "lower_features_1.onnx",
    "schema.json",
    "neutral_observation.npy",
    "export-parity.json",
}


def available():
    # A partial or corrupt build must fail verification, rather than silently
    # selecting the source backend under different runtime assumptions.
    return DIRECTORY.exists()


def _verified_path(root, name, expected):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Compiled observation paths must stay in their root")
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError("Compiled observation path escapes its root")
    if (
        path.stat().st_size != expected["bytes"]
        or sha256(path) != expected["sha256"]
    ):
        raise ValueError(f"Compiled observation fingerprint differs: {name}")
    return path


@lru_cache(maxsize=1)
def manifest():
    from ..models import frozen_manifest

    record = json.loads((DIRECTORY / "manifest.json").read_text())
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported compiled observation manifest")
    if record.get("frozen_files") != frozen_manifest()["files"]:
        raise ValueError("Compiled observations belong to another controller")
    if set(record.get("files", {})) != REQUIRED_FILES:
        raise ValueError("Compiled observation bundle is incomplete")
    for name, expected in record["files"].items():
        _verified_path(DIRECTORY, name, expected)
    required_sources = {
        "holoboxing/evaluation/runtime/fast_runtime.py",
        "holoboxing/evaluation/runtime/representation.py",
        "holoboxing/evaluation/runtime/reference.py",
        "holoboxing/evaluation/runtime/observation_export.py",
        "holomotion/src/motion_tracking/link_observation.py",
        "holomotion/src/motion_tracking/link_actor_packing.py",
    }
    if not required_sources.issubset(record.get("sources", {})):
        raise ValueError(
            "Compiled observation source provenance is incomplete"
        )
    for name, expected in record["sources"].items():
        _verified_path(APP_ROOT, name, expected)
    return record


@lru_cache(maxsize=1)
def schema():
    manifest()
    saved = json.loads((DIRECTORY / "schema.json").read_text())
    if saved["frozen_files"] != manifest()["frozen_files"]:
        raise ValueError("Compiled schema belongs to another controller")
    return saved


def reference_schema():
    if not available():
        from .reference import ReferenceSchema

        return ReferenceSchema()
    return CompiledReferenceSchema()


def scene_layout(scene, lower_link_names):
    from .model_contract import INPUT_LINK_NAMES

    return dict(
        nbody=scene.model.nbody,
        nq=scene.model.nq,
        nv=scene.model.nv,
        body_ids=[
            [scene.model.body(prefix + name).id for name in INPUT_LINK_NAMES]
            for prefix in ("p1_", "p2_")
        ],
        lower_body_ids=[
            [scene.model.body(prefix + name).id for name in lower_link_names]
            for prefix in ("p1_", "p2_")
        ],
        body_roots=scene.model.body_rootid.tolist(),
        root_qpos=scene._root_qpos,
        root_dof=scene._root_dof,
        joint_qpos=[x.tolist() for x in scene._joint_qpos],
        joint_dof=[x.tolist() for x in scene._joint_dof],
        articulation_joint_names=list(scene.articulation_joint_names),
    )


class CompiledReferenceSchema:
    """The verified schema constants consumed by compiled GPU observations."""

    input_dim = 1097
    input_history_frames = 10
    frame_offsets = (1, 2, 3, 4, 5)
    control_dt = 0.02

    def __init__(self):
        from .model_contract import INPUT_LINK_NAMES

        saved = schema()
        self.link_names = tuple(saved["lower_link_names"])
        self.point_names = INPUT_LINK_NAMES
        self.active = [
            self.link_names.index(name) for name in self.point_names
        ]
        self.frames = len(self.frame_offsets)
        self.num_links = len(self.link_names)
        self.width = 9 * self.num_links + 1
        self.task_dim = self.frames * self.width
        self.template = np.asarray(saved["schema_template"], np.float32)
        self.output_dim = self.template.size
        self.learn_indices = np.asarray(saved["schema_indices"], np.int64)
        self.source_hashes = saved["schema_hashes"]
