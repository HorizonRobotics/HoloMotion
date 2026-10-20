"""Competition XML kinematics and a runtime adapter for HoloRetarget."""

import shutil
import copy
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path

import mujoco


def robot_xml(path, side="red"):
    """Read the same robot XML attached by the evaluator and replay player."""
    path = Path(path).resolve(strict=True)
    if side not in ("red", "blue"):
        raise ValueError("Choose red or blue")
    prefix = "p1_" if side == "red" else "p2_"
    root = ET.parse(path).getroot()
    attachment = root.find(f"worldbody/attach[@prefix='{prefix}']")
    if attachment is None:
        raise ValueError("Scene must attach the existing replay robot XML")
    model = root.find(f"asset/model[@name='{attachment.get('model')}']")
    if model is None:
        raise ValueError("Scene attachment is missing its replay model")
    path = (path.parent / model.get("file")).resolve(strict=True)
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    robot = world.find("body[@name='pelvis']")
    if robot is None:
        raise ValueError("Scene is missing the named floating robot")
    robot = copy.deepcopy(robot)
    world.clear()
    world.append(robot)
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            compiler.set(
                key, str((path.parent / compiler.get(key, "")).resolve())
            )
    return ET.tostring(root, encoding="unicode")


def robot_spec(path, side="red"):
    spec = mujoco.MjSpec.from_string(robot_xml(path, side))
    spec.compiler.fusestatic = False
    return spec


@contextmanager
def robot_file(scene, side="red"):
    """A temporary single-robot MJCF for existing HoloMotion FK readers."""
    root = ET.fromstring(robot_xml(scene, side))
    if root.find("actuator") is None:
        # Replay XML has no motors; evaluation adds them when loading the scene.
        # FK readers use motor targets to identify and order the controlled DOFs.
        # Add that metadata only to this temporary kinematic model.
        actuator = ET.SubElement(root, "actuator")
        for joint in root.findall("worldbody/.//joint"):
            if joint.get("type", "hinge") != "free":
                ET.SubElement(actuator, "motor", joint=joint.attrib["name"])
    with TemporaryDirectory(prefix="boxing-xml-fk-") as directory:
        path = Path(directory) / "robot.xml"
        path.write_text(ET.tostring(root, encoding="unicode"))
        yield path


def load_scene_model(assets):
    """Use exactly the evaluator's robot geometry and physical scene."""
    from holoboxing.evaluation.runtime.scene import load_scene_model as load

    return load(assets.resources.scene)


def retarget_assets(assets, config, directory):
    """Adapt the scene robot to HoloRetarget's asset-root interface."""
    from holoretarget._engine_impl import HoloNewtonG1Retargeter

    directory = Path(directory)
    robot_dir = directory / "unitree_g1"
    robot_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        config.resolved_asset_root / "target_configs",
        directory / "target_configs",
    )
    spec = robot_spec(assets.resources.scene)
    # Existing IK toe targets are fixed markers, not physical robot changes.
    for name, parent in HoloNewtonG1Retargeter._BODY_REMAP.items():
        spec.body(parent).add_body(
            name=name, pos=HoloNewtonG1Retargeter._BODY_LINK_OFFSETS[name]
        )
    # The existing gloved XML already contains both fixed hand frames.
    spec.compile()
    (robot_dir / "g1_mocap_29dof.xml").write_text(spec.to_xml())
    return directory
