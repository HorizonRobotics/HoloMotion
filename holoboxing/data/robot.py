"""Competition XML kinematics and a runtime adapter for HoloRetarget."""

import shutil
import copy
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path

import mujoco


def robot_xml(path, side="red"):
    """Extract one robot from the simulation scene, not the display XML."""
    path = Path(path).resolve(strict=True)
    if side not in ("red", "blue"):
        raise ValueError("Choose red or blue")
    prefix = "p1_" if side == "red" else "p2_"
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    robot = world.find(f"body[@name='{prefix}pelvis']")
    if robot is None:
        raise ValueError("Scene is missing the named floating robot")
    robot = copy.deepcopy(robot)
    world.clear()
    world.append(robot)
    # Scene-wide pairs and saved states refer to the removed robot or arena.
    for tag in ("contact", "equality", "keyframe", "sensor", "tendon"):
        section = root.find(tag)
        if section is not None:
            root.remove(section)
    actuators = root.find("actuator")
    if actuators is not None:
        for node in list(actuators):
            if not node.get("joint", "").startswith(prefix):
                actuators.remove(node)
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith(prefix):
                node.set(key, value[len(prefix) :])
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
    with TemporaryDirectory(prefix="boxing-xml-fk-") as directory:
        path = Path(directory) / "robot.xml"
        path.write_text(robot_xml(scene, side))
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
    # The scene folds the palm into the wrist; IK still names its fixed frame.
    for side in ("left", "right"):
        hand = spec.geom(f"{side}_boxing_hand_collision")
        spec.body(f"{side}_wrist_yaw_link").add_body(
            name=f"{side}_rubber_hand", pos=hand.pos, quat=hand.quat
        )
    spec.compile()
    (robot_dir / "g1_mocap_29dof.xml").write_text(spec.to_xml())
    return directory
