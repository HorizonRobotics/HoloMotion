#!/usr/bin/env python3
"""Modular Unitree G1 URDF composition.

The public entry point is :func:`compose_g1_urdf`.  A robot is assembled from
one body, one head, and one hand module while keeping the training body at
exactly 29 movable joints.  Standard hands attach below ``wrist_yaw_link``;
BrainCo is an atomic wrist-plus-hand module that replaces the wrist subtree at
``elbow_link``.
"""

from __future__ import annotations
import argparse
import copy
import filecmp
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

CODE_ROOT = Path(__file__).resolve().parent


def _resolve_asset_root() -> Path:
    override = os.environ.get("HOLOMOTION_G1_MODULAR_ASSET_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (
            not (root / "components").is_dir()
            or not (root / "mesh_index.json").is_file()
        ):
            raise FileNotFoundError(
                f"invalid HOLOMOTION_G1_MODULAR_ASSET_ROOT: {root}"
            )
        return root

    if len(CODE_ROOT.parents) > 4:
        repository_root = CODE_ROOT.parents[4]
        huggingface_root = (
            repository_root
            / "thirdparties"
            / "HoloMotion_assets"
            / "assets"
            / "robots"
            / "unitree"
            / "G1"
            / "modular"
        )
        if (huggingface_root / "components").is_dir() and (
            huggingface_root / "mesh_index.json"
        ).is_file():
            return huggingface_root.resolve()
    return CODE_ROOT


ASSET_ROOT = _resolve_asset_root()
RUNTIME_MANIFEST_SCHEMA = "holomotion-g1-variant-manifest-v1"
MESH_INDEX_SCHEMA = "holomotion-g1-mesh-index-v1"
MESH_INDEX_PATH = ASSET_ROOT / "mesh_index.json"


@dataclass(frozen=True)
class BodyModule:
    key: str
    source: Path


@dataclass(frozen=True)
class HeadModule:
    key: str
    source: Path | None


@dataclass(frozen=True)
class HandModule:
    key: str
    source: Path | None
    replaces_wrist: bool = False


@dataclass(frozen=True)
class CompositionResult:
    output_path: Path
    body: str
    head: str
    hand: str
    movable_joints: tuple[str, ...]
    nq: int
    nv: int


@dataclass(frozen=True)
class ModelBundleResult:
    urdf: CompositionResult
    mjcf_path: Path
    mjcf_nq: int
    mjcf_nv: int


@dataclass(frozen=True)
class RuntimeDependencySet:
    """Minimal component closure required to compose every unique robot."""

    component_descriptions: tuple[Path, ...]
    meshes: tuple[Path, ...]


BODY_MODULES = {
    key: BodyModule(key, ASSET_ROOT / "components" / "bodies" / f"{key}.xml")
    for key in ("mode5", "mode11", "mode13", "mode15", "mode18")
}

HEAD_MODULES = {
    key: HeadModule(key, ASSET_ROOT / "components" / "heads" / f"{key}.xml")
    for key in ("normal", "comp")
}

HAND_MODULES = {
    "none": HandModule("none", None),
    **{
        key: HandModule(
            key, ASSET_ROOT / "components" / "hands" / f"{key}.xml"
        )
        for key in (
            "rubber",
            "dex1_1",
            "dex3_1",
            "inspire_dfq",
            "inspire_ftp",
        )
    },
    "brainco": HandModule(
        "brainco",
        ASSET_ROOT / "components" / "hands" / "brainco.xml",
        True,
    ),
}

# Replacing the complete wrist subtree makes these source-body pairs
# structurally identical for BrainCo. Keep one canonical representative rather
# than generating four duplicate training assets.
BRAINCO_BODY_ALIASES = {
    "mode13": "mode5",
    "mode15": "mode11",
}

HEAD_ROOT_JOINTS = (
    "head_joint",
    "d435_joint",
    "mid360_joint",
    "head_servo_joint",
)
WRIST_ROOT_JOINTS = ("left_wrist_roll_joint", "right_wrist_roll_joint")
BRAINCO_ACTIVE_WRISTS = {
    f"{side}_wrist_{axis}_joint"
    for side in ("left", "right")
    for axis in ("roll", "pitch", "yaw")
}
CANONICAL_DOF_NAMES = tuple(
    joint.get("name")
    for joint in ET.parse(BODY_MODULES["mode5"].source)
    .getroot()
    .findall("joint")
    if joint.get("type") != "fixed"
)


def _joint_maps(root: ET.Element):
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        by_parent.setdefault(joint.find("parent").get("link"), []).append(
            joint
        )
    return joints, by_parent


def _subtree_names(
    root: ET.Element, root_joint_name: str
) -> tuple[set[str], set[str]]:
    joints, by_parent = _joint_maps(root)
    if root_joint_name not in joints:
        return set(), set()
    joint_names: set[str] = set()
    link_names: set[str] = set()
    stack = [joints[root_joint_name]]
    while stack:
        joint = stack.pop()
        name = joint.get("name")
        if name in joint_names:
            continue
        joint_names.add(name)
        child = joint.find("child").get("link")
        link_names.add(child)
        stack.extend(by_parent.get(child, []))
    return joint_names, link_names


def _remove_subtree(root: ET.Element, root_joint_name: str) -> None:
    joint_names, link_names = _subtree_names(root, root_joint_name)
    for joint in list(root.findall("joint")):
        if joint.get("name") in joint_names:
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") in link_names:
            root.remove(link)


def _roots_attached_to(root: ET.Element, parent_link: str) -> list[str]:
    return [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.find("parent").get("link") == parent_link
    ]


def _make_fixed(joint: ET.Element) -> None:
    joint.set("type", "fixed")
    for tag in (
        "axis",
        "limit",
        "dynamics",
        "safety_controller",
        "mimic",
        "calibration",
    ):
        child = joint.find(tag)
        if child is not None:
            joint.remove(child)


def _rewrite_meshes(element: ET.Element, prefix: str = "") -> None:
    for mesh in element.iter("mesh"):
        mesh.set(
            "filename", f"meshes/{prefix}{Path(mesh.get('filename')).name}"
        )


def _load_mesh_index() -> dict[str, str]:
    if not MESH_INDEX_PATH.is_file():
        return {}
    payload = json.loads(MESH_INDEX_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != MESH_INDEX_SCHEMA:
        raise ValueError(f"unsupported mesh index: {MESH_INDEX_PATH}")
    return payload["meshes"]


def _resolve_mesh_source(logical_name: str) -> Path:
    relative_path = _load_mesh_index().get(
        logical_name, f"meshes/{logical_name}"
    )
    source_path = ASSET_ROOT / relative_path
    g1_root = ASSET_ROOT.parent.absolute()
    if not source_path.absolute().is_relative_to(g1_root):
        raise ValueError(
            "mesh index escapes the G1 asset root: "
            f"{logical_name} -> {source_path}"
        )
    source = source_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"missing modular mesh source for {logical_name}: {source}"
        )
    return source


def _append_subtree(
    target: ET.Element,
    source: ET.Element,
    root_joint_name: str,
    mesh_prefix: str = "",
    keep_active: set[str] | None = None,
) -> None:
    joint_names, link_names = _subtree_names(source, root_joint_name)
    keep_active = keep_active or set()
    for link in source.findall("link"):
        if link.get("name") in link_names:
            element = copy.deepcopy(link)
            _rewrite_meshes(element, mesh_prefix)
            target.append(element)
    for joint in source.findall("joint"):
        if joint.get("name") in joint_names:
            element = copy.deepcopy(joint)
            if (
                element.get("name") not in keep_active
                and element.get("type") != "fixed"
            ):
                _make_fixed(element)
            target.append(element)


def _symlink_checked(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return
        raise FileExistsError(
            "mesh target already exists and is not the expected symlink: "
            f"{target}"
        )
    os.symlink(source, target)


def _materialize_meshes(output_dir: Path, root: ET.Element) -> None:
    """Link only meshes referenced by one composed result."""

    for mesh in root.iter("mesh"):
        relative_path = Path(mesh.get("filename"))
        source = _resolve_mesh_source(relative_path.name)
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _symlink_checked(source, target)


def _prepare_body(body: BodyModule) -> ET.Element:
    root = copy.deepcopy(ET.parse(body.source).getroot())
    root.set("name", "g1_modular")
    _rewrite_meshes(root)
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.Element("mujoco")
        root.insert(0, mujoco_element)
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.set("meshdir", "meshes")
    compiler.set("discardvisual", "false")
    return root


def _compose_g1_root(body: str, head: str, hand: str) -> ET.Element:
    """Compose an XML tree without writing files or materializing meshes."""

    if body not in BODY_MODULES:
        raise KeyError(
            f"unknown body {body!r}; choose from {tuple(BODY_MODULES)}"
        )
    if head not in HEAD_MODULES:
        raise KeyError(
            f"unknown head {head!r}; choose from {tuple(HEAD_MODULES)}"
        )
    if hand not in HAND_MODULES:
        raise KeyError(
            f"unknown hand {hand!r}; choose from {tuple(HAND_MODULES)}"
        )

    root = _prepare_body(BODY_MODULES[body])
    head_module = HEAD_MODULES[head]
    if head_module.source is not None:
        for name in HEAD_ROOT_JOINTS:
            _remove_subtree(root, name)
        source_root = ET.parse(head_module.source).getroot()
        for name in HEAD_ROOT_JOINTS:
            _append_subtree(root, source_root, name)

    hand_module = HAND_MODULES[hand]
    if hand_module.replaces_wrist:
        for name in WRIST_ROOT_JOINTS:
            _remove_subtree(root, name)
        source_root = ET.parse(hand_module.source).getroot()
        for name in WRIST_ROOT_JOINTS:
            _append_subtree(
                root,
                source_root,
                name,
                "brainco__",
                BRAINCO_ACTIVE_WRISTS,
            )
    else:
        for parent in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            for name in _roots_attached_to(root, parent):
                _remove_subtree(root, name)
        if hand_module.source is not None:
            source_root = ET.parse(hand_module.source).getroot()
            for parent in (
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ):
                for name in _roots_attached_to(source_root, parent):
                    _append_subtree(root, source_root, name)

    _collapse_fixed_joints(root)
    _canonicalize_joint_order(root)
    return root


def unique_g1_combinations() -> tuple[tuple[str, str, str], ...]:
    """Return the deterministic 66-model training matrix."""

    combinations = []
    for body in BODY_MODULES:
        for head in HEAD_MODULES:
            for hand in HAND_MODULES:
                if hand == "brainco" and body in BRAINCO_BODY_ALIASES:
                    continue
                combinations.append((body, head, hand))
    result = tuple(combinations)
    if len(result) != 66:
        raise AssertionError(
            f"expected 66 unique combinations, got {len(result)}"
        )
    return result


def runtime_component_dependencies_for(
    combinations: tuple[tuple[str, str, str], ...],
) -> RuntimeDependencySet:
    """Compute the component and mesh closure for selected combinations."""

    component_descriptions = set()
    mesh_names = set()
    for body, head, hand in combinations:
        for module in (
            BODY_MODULES[body],
            HEAD_MODULES[head],
            HAND_MODULES[hand],
        ):
            if module.source is not None:
                component_descriptions.add(module.source.resolve())
        root = _compose_g1_root(body, head, hand)
        mesh_names.update(
            Path(mesh.get("filename")).name for mesh in root.iter("mesh")
        )
    meshes = tuple(sorted({_resolve_mesh_source(name) for name in mesh_names}))
    missing = [
        path
        for path in (*component_descriptions, *meshes)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "runtime component closure contains missing files: "
            + ", ".join(map(str, missing))
        )
    return RuntimeDependencySet(tuple(sorted(component_descriptions)), meshes)


def runtime_component_dependencies() -> RuntimeDependencySet:
    """Compute the exact component-description and mesh closure."""

    return runtime_component_dependencies_for(unique_g1_combinations())


def _canonicalize_joint_order(root: ET.Element) -> None:
    """Keep policy dimensions independent of the source layout."""
    joints = list(root.findall("joint"))
    movable_by_name = {
        joint.get("name"): joint
        for joint in joints
        if joint.get("type") != "fixed"
    }
    if set(movable_by_name) != set(CANONICAL_DOF_NAMES):
        missing = sorted(set(CANONICAL_DOF_NAMES) - set(movable_by_name))
        extra = sorted(set(movable_by_name) - set(CANONICAL_DOF_NAMES))
        raise ValueError(
            "composed movable joints differ from canonical G1: "
            f"missing={missing}, extra={extra}"
        )
    for joint in joints:
        root.remove(joint)
    for name in CANONICAL_DOF_NAMES:
        root.append(movable_by_name[name])
    for joint in joints:
        if joint.get("type") == "fixed":
            root.append(joint)


def _parse_vector(value: str | None, size: int = 3) -> np.ndarray:
    if not value:
        return np.zeros(size, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"expected {size} values, got {value!r}")
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _origin_transform(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    transform = np.eye(4, dtype=np.float64)
    if origin is not None:
        transform[:3, 3] = _parse_vector(origin.get("xyz"))
        transform[:3, :3] = _rpy_matrix(_parse_vector(origin.get("rpy")))
    return transform


def _format_vector(value: np.ndarray) -> str:
    return " ".join(f"{float(item):.12g}" for item in value)


def _set_origin_transform(element: ET.Element, transform: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        element.insert(0, origin)
    origin.set("xyz", _format_vector(transform[:3, 3]))
    origin.set("rpy", _format_vector(_matrix_rpy(transform[:3, :3])))


def _read_inertial(
    link: ET.Element, link_to_target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray] | None:
    inertial = link.find("inertial")
    if inertial is None or inertial.find("mass") is None:
        return None
    mass = float(inertial.find("mass").get("value"))
    if mass <= 0.0:
        return None
    target_to_inertial = link_to_target @ _origin_transform(inertial)
    com = target_to_inertial[:3, 3]
    rotation = target_to_inertial[:3, :3]
    tensor = inertial.find("inertia")
    inertia_local = np.array(
        [
            [
                float(tensor.get("ixx")),
                float(tensor.get("ixy")),
                float(tensor.get("ixz")),
            ],
            [
                float(tensor.get("ixy")),
                float(tensor.get("iyy")),
                float(tensor.get("iyz")),
            ],
            [
                float(tensor.get("ixz")),
                float(tensor.get("iyz")),
                float(tensor.get("izz")),
            ],
        ],
        dtype=np.float64,
    )
    return mass, com, rotation @ inertia_local @ rotation.T


def _write_combined_inertial(
    parent: ET.Element,
    components: list[tuple[float, np.ndarray, np.ndarray]],
) -> None:
    existing = parent.find("inertial")
    if existing is not None:
        parent.remove(existing)
    if not components:
        return
    total_mass = sum(component[0] for component in components)
    center = sum(mass * com for mass, com, _ in components) / total_mass
    inertia = np.zeros((3, 3), dtype=np.float64)
    for mass, com, inertia_at_com in components:
        offset = com - center
        inertia += inertia_at_com + mass * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
    inertial = ET.Element("inertial")
    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", _format_vector(center))
    origin.set("rpy", "0 0 0")
    ET.SubElement(inertial, "mass", value=f"{total_mass:.12g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.12g}",
        ixy=f"{inertia[0, 1]:.12g}",
        ixz=f"{inertia[0, 2]:.12g}",
        iyy=f"{inertia[1, 1]:.12g}",
        iyz=f"{inertia[1, 2]:.12g}",
        izz=f"{inertia[2, 2]:.12g}",
    )
    parent.insert(0, inertial)


def _collapse_fixed_joints(root: ET.Element) -> None:
    """Fold fixed subtrees into their movable parent."""

    while True:
        fixed_joint = next(
            (
                joint
                for joint in root.findall("joint")
                if joint.get("type") == "fixed"
            ),
            None,
        )
        if fixed_joint is None:
            break
        parent_name = fixed_joint.find("parent").get("link")
        child_name = fixed_joint.find("child").get("link")
        links = {link.get("name"): link for link in root.findall("link")}
        parent = links[parent_name]
        child = links[child_name]
        parent_to_child = _origin_transform(fixed_joint)

        components = []
        parent_inertial = _read_inertial(parent, np.eye(4))
        child_inertial = _read_inertial(child, parent_to_child)
        if parent_inertial is not None:
            components.append(parent_inertial)
        if child_inertial is not None:
            components.append(child_inertial)
        _write_combined_inertial(parent, components)

        for tag in ("visual", "collision"):
            for geometry in child.findall(tag):
                moved = copy.deepcopy(geometry)
                if moved.get("name"):
                    moved.set("name", f"{child_name}__{moved.get('name')}")
                _set_origin_transform(
                    moved, parent_to_child @ _origin_transform(moved)
                )
                parent.append(moved)

        for joint in root.findall("joint"):
            if joint is fixed_joint:
                continue
            joint_parent = joint.find("parent")
            if joint_parent.get("link") == child_name:
                joint_parent.set("link", parent_name)
                _set_origin_transform(
                    joint, parent_to_child @ _origin_transform(joint)
                )

        for gazebo in root.findall("gazebo"):
            if gazebo.get("reference") == child_name:
                gazebo.set("reference", parent_name)
        root.remove(fixed_joint)
        root.remove(child)


def validate_composed_urdf(path: str | Path) -> CompositionResult:
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    joints = [joint.get("name") for joint in root.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError(f"duplicate links in {path}")
    if len(joints) != len(set(joints)):
        raise ValueError(f"duplicate joints in {path}")
    known_links = set(links)
    if len(links) != 30:
        raise ValueError(
            f"expected one shared 30-body topology in {path}; "
            f"got {len(links)} links"
        )
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent not in known_links or child not in known_links:
            raise ValueError(
                f"invalid joint tree at {joint.get('name')}: "
                f"{parent} -> {child}"
            )
    movable = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    )
    fixed = tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") == "fixed"
    )
    if fixed:
        raise ValueError(
            f"fixed joints must be collapsed in {path}; got {fixed}"
        )
    if movable != CANONICAL_DOF_NAMES:
        raise ValueError(
            f"expected canonical 29-DOF order in {path}; got {movable}"
        )
    meshdir = root.find("mujoco/compiler").get("meshdir")
    for mesh in root.iter("mesh"):
        filename = Path(mesh.get("filename"))
        mesh_path = path.parent / filename
        if not mesh_path.exists():
            mesh_path = path.parent / meshdir / filename.name
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nq != 29 or model.nv != 29:
        raise ValueError(
            f"MuJoCo expected nq=nv=29, got nq={model.nq}, nv={model.nv}"
        )
    return CompositionResult(path, "", "", "", movable, model.nq, model.nv)


def compose_g1_urdf(
    body: str,
    head: str,
    hand: str,
    output_path: str | Path,
    *,
    validate: bool = True,
) -> CompositionResult:
    """Assemble one modular 29DOF G1 URDF.

    Args:
        body: One of ``mode5``, ``mode11``, ``mode13``, ``mode15``, ``mode18``.
        head: ``normal`` or ``comp``.
        hand: ``none``, ``rubber``, ``dex1_1``, ``dex3_1``, ``inspire_dfq``,
            ``inspire_ftp``, or ``brainco``.
        output_path: Destination URDF. A shared ``meshes`` directory containing
            symlinks is materialized next to it.
        validate: Load the result with MuJoCo and enforce exactly 29 DOF.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = _compose_g1_root(body, head, hand)
    _materialize_meshes(output_path.parent, root)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        output_path, encoding="unicode", xml_declaration=False
    )
    checked = (
        validate_composed_urdf(output_path)
        if validate
        else CompositionResult(
            output_path,
            "",
            "",
            "",
            tuple(
                j.get("name")
                for j in root.findall("joint")
                if j.get("type") != "fixed"
            ),
            -1,
            -1,
        )
    )
    return CompositionResult(
        checked.output_path,
        body,
        head,
        hand,
        checked.movable_joints,
        checked.nq,
        checked.nv,
    )


def validate_generated_urdf_bundle(
    output_dir: str | Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Assert a generated directory contains exactly its URDF closure."""

    output_dir = Path(output_dir).resolve()
    expected_names = {
        f"g1_{body}_{head}_{hand}.urdf"
        for body, head, hand in unique_g1_combinations()
    }
    urdfs = tuple(sorted(output_dir.glob("g1_*.urdf")))
    actual_names = {path.name for path in urdfs}
    if actual_names != expected_names:
        raise ValueError(
            "generated URDF set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    referenced_meshes = set()
    for urdf in urdfs:
        root = ET.parse(urdf).getroot()
        for mesh in root.iter("mesh"):
            mesh_path = (urdf.parent / mesh.get("filename")).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            referenced_meshes.add(mesh_path)

    mesh_dir = output_dir / "meshes"
    actual_meshes = {
        path.resolve()
        for path in mesh_dir.iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual_meshes != referenced_meshes:
        raise ValueError(
            "generated mesh closure mismatch: "
            f"missing={sorted(map(str, referenced_meshes - actual_meshes))}, "
            f"extra={sorted(map(str, actual_meshes - referenced_meshes))}"
        )
    return urdfs, tuple(sorted(referenced_meshes))


def compose_all_g1_urdfs(
    output_dir: str | Path,
    *,
    validate: bool = False,
    clean: bool = True,
) -> tuple[CompositionResult, ...]:
    """Generate the 66 unique training URDFs from the component sources."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for path in output_dir.glob("g1_*.urdf"):
            path.unlink()
        mesh_dir = output_dir / "meshes"
        if mesh_dir.exists() or mesh_dir.is_symlink():
            if mesh_dir.is_symlink():
                mesh_dir.unlink()
            else:
                shutil.rmtree(mesh_dir)

    results = []
    for body, head, hand in unique_g1_combinations():
        results.append(
            compose_g1_urdf(
                body,
                head,
                hand,
                output_dir / f"g1_{body}_{head}_{hand}.urdf",
                validate=validate,
            )
        )
    validate_generated_urdf_bundle(output_dir)
    manifest_path = output_dir / "variant_manifest.json"
    manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "variants": [
            {
                "variant_id": result.output_path.stem,
                "body": result.body,
                "head": result.head,
                "hand": result.hand,
                "urdf_path": result.output_path.name,
                "urdf_sha256": hashlib.sha256(
                    result.output_path.read_bytes()
                ).hexdigest(),
            }
            for result in results
        ],
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_manifest, manifest_path)
    return tuple(results)


def compose_g1_urdf_and_mjcf(
    body: str,
    head: str,
    hand: str,
    output_dir: str | Path,
    *,
    name: str | None = None,
    validate: bool = True,
    simulation_steps: int = 20,
) -> ModelBundleResult:
    """Compose matching URDF and MuJoCo MJCF files."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"g1_{body}_{head}_{hand}"
    urdf_result = compose_g1_urdf(
        body, head, hand, output_dir / f"{stem}.urdf", validate=validate
    )

    model = mujoco.MjModel.from_xml_path(str(urdf_result.output_path))
    mjcf_path = output_dir / f"{stem}.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), model)

    reloaded = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if reloaded.nq != 29 or reloaded.nv != 29:
        raise ValueError(
            "MuJoCo MJCF expected nq=nv=29, "
            f"got nq={reloaded.nq}, nv={reloaded.nv}: {mjcf_path}"
        )
    if validate:
        data = mujoco.MjData(reloaded)
        for _ in range(simulation_steps):
            mujoco.mj_step(reloaded, data)
        if any(
            value != value or abs(value) == float("inf")
            for value in (*data.qpos, *data.qvel)
        ):
            raise ValueError(
                f"non-finite MuJoCo state after stepping: {mjcf_path}"
            )

    return ModelBundleResult(urdf_result, mjcf_path, reloaded.nq, reloaded.nv)


__all__ = [
    "BODY_MODULES",
    "HEAD_MODULES",
    "HAND_MODULES",
    "CompositionResult",
    "ModelBundleResult",
    "RuntimeDependencySet",
    "BRAINCO_BODY_ALIASES",
    "compose_g1_urdf",
    "compose_all_g1_urdfs",
    "compose_g1_urdf_and_mjcf",
    "runtime_component_dependencies",
    "runtime_component_dependencies_for",
    "unique_g1_combinations",
    "validate_composed_urdf",
    "validate_generated_urdf_bundle",
]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose the 66 unique modular G1 training URDFs."
    )
    parser.add_argument(
        "command", choices=("generate-all", "generate-one", "dependencies")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--body", choices=tuple(BODY_MODULES))
    parser.add_argument("--head", choices=tuple(HEAD_MODULES))
    parser.add_argument("--hand", choices=tuple(HAND_MODULES))
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load every generated URDF with MuJoCo (default: false).",
    )
    args = parser.parse_args()

    selected = None
    if any(value is not None for value in (args.body, args.head, args.hand)):
        if not all(
            value is not None for value in (args.body, args.head, args.hand)
        ):
            parser.error(
                "--body, --head, and --hand must be provided together"
            )
        selected = ((args.body, args.head, args.hand),)

    if args.command == "dependencies":
        dependencies = (
            runtime_component_dependencies_for(selected)
            if selected is not None
            else runtime_component_dependencies()
        )
        for path in (
            *dependencies.component_descriptions,
            *dependencies.meshes,
        ):
            print(os.path.relpath(path, ASSET_ROOT))
        return

    if args.output_dir is None:
        parser.error(f"{args.command} requires --output-dir")
    if args.command == "generate-one":
        if selected is None:
            parser.error("generate-one requires --body, --head, and --hand")
        body, head, hand = selected[0]
        result = compose_g1_urdf(
            body,
            head,
            hand,
            args.output_dir / f"g1_{body}_{head}_{hand}.urdf",
            validate=args.validate,
        )
        print(f"READY: generated {result.output_path}")
        return
    results = compose_all_g1_urdfs(
        args.output_dir,
        validate=args.validate,
    )
    urdfs, meshes = validate_generated_urdf_bundle(args.output_dir)
    print(
        f"READY: generated {len(results)} unique G1 URDFs with "
        f"{len(meshes)} referenced meshes under {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    _main()
