"""Build a reference G1 MuJoCo model without changing the source robot asset."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .robot import G1_JOINT_NAMES, _kd, _kp, motor_of


def build_model(urdf, output, *, self_collisions=True):
    urdf, output = Path(urdf).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(urdf).getroot()
    extension = robot.find("mujoco")
    if extension is not None:
        robot.remove(extension)
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        balanceinertia="true",
        discardvisual="false",
        fusestatic="false",
        strippath="false",
    )
    for mesh in robot.findall(".//mesh"):
        mesh.set(
            "filename", str((urdf.parent / mesh.get("filename")).resolve())
        )
    compiled = mujoco.MjModel.from_xml_string(
        ET.tostring(robot, encoding="unicode")
    )
    intermediate = output.with_suffix(".compiled.xml")
    mujoco.mj_saveLastXML(str(intermediate), compiled)
    root = ET.parse(intermediate).getroot()
    pelvis = root.find('.//body[@name="pelvis"]')
    if pelvis is None:
        raise ValueError("URDF conversion did not retain the pelvis link")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep="0.005",
        integrator="implicitfast",
        cone="pyramidal",
        iterations="10",
        ls_iterations="10",
    )
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for name in G1_JOINT_NAMES:
        effort, _, armature = motor_of(name)
        joint = root.find(f'.//joint[@name="{name}"]')
        if joint is None:
            raise ValueError(f"URDF conversion lost joint {name}")
        joint.attrib.update(
            armature=str(armature), damping="0", frictionloss="0"
        )
        ET.SubElement(
            actuator,
            "general",
            name=name + "_drive",
            joint=name,
            gainprm=str(_kp(armature)),
            biastype="affine",
            biasprm=f"0 {-_kp(armature)} {-_kd(armature)}",
            forcelimited="true",
            forcerange=f"{-effort} {effort}",
        )
    world = root.find("worldbody")
    # Explicit masks separate disabling the penalty from disabling collisions.
    for geom in root.findall(".//body//geom"):
        if (
            geom.get("contype", "1") == "0"
            and geom.get("conaffinity", "1") == "0"
        ):
            continue
        geom.attrib.update(friction="0.8 0.005 0.0001")
        if not self_collisions:
            geom.attrib.update(contype="1", conaffinity="0")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="0.3 0.005 0.0001",
        contype="0" if not self_collisions else "1",
        conaffinity="1",
    )
    ET.indent(root)
    output.write_text(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(output))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(
            f"Unexpected G1 model dimensions: {model.nq}/{model.nv}/{model.nu}"
        )
    if [model.joint(i).name for i in range(1, model.njnt)] != G1_JOINT_NAMES:
        raise ValueError(
            "MuJoCo joint order differs from the declared reference data order"
        )
    return model
