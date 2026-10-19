"""Export full-precision replay MJCF from the canonical gloved URDFs.

Exports are maintainer artifacts. Evaluation and playback both load the saved
XML directly; neither translates or copies a robot at runtime.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .runtime.model_contract import sha256


def urdf_spec(path):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(path))
    spec.compiler.fusestatic = False
    spec.body("pelvis").add_joint(
        name="floating_base", type=mujoco.mjtJoint.mjJNT_FREE
    )
    return spec


def _numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


def export_xml(urdf):
    """Avoid the XML writer's six-digit rounding of robot parameters."""
    spec = urdf_spec(urdf)
    model = spec.compile()
    root = ET.fromstring(spec.to_xml())
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="source_urdf_sha256", data=sha256(urdf))
    for body in root.findall(".//body"):
        name = body.get("name")
        index = model.body(name).id
        body.set("pos", _numbers(model.body_pos[index]))
        body.set("quat", _numbers(model.body_quat[index]))
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", _numbers([model.body_mass[index]]))
            inertial.set("pos", _numbers(model.body_ipos[index]))
            inertial.set("quat", _numbers(model.body_iquat[index]))
            inertial.set("diaginertia", _numbers(model.body_inertia[index]))
        for geom, source in zip(
            body.findall("geom"), spec.body(name).geoms, strict=True
        ):
            # Compiled mesh geoms include a centering transform. Preserve the
            # original spec transform here, so it is applied only once.
            geom.set("pos", _numbers(source.pos))
            geom.set("quat", _numbers(source.quat))
            geom.set("rgba", _numbers(source.rgba))
            if "size" in geom.attrib:
                width = len(geom.get("size").split())
                geom.set("size", _numbers(source.size[:width]))
    for joint in root.findall(".//joint"):
        index = model.joint(joint.get("name")).id
        joint.set("pos", _numbers(model.jnt_pos[index]))
        joint.set("axis", _numbers(model.jnt_axis[index]))
        joint.set("range", _numbers(model.jnt_range[index]))
        joint.set("actuatorfrcrange", _numbers(model.jnt_actfrcrange[index]))
    ET.indent(root, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def verify_robot(urdf, xml):
    """Compare compiled kinematics, inertias, force limits and all geoms."""
    import mujoco
    import numpy as np

    source = urdf_spec(urdf).compile()
    target = mujoco.MjModel.from_xml_path(str(xml))
    if (source.nq, source.nv, source.nbody, source.ngeom) != (
        target.nq,
        target.nv,
        target.nbody,
        target.ngeom,
    ):
        raise ValueError(f"URDF/XML dimensions differ: {xml.name}")
    maximum = 0.0

    def equal(a, b, label):
        nonlocal maximum
        if np.asarray(a).shape != np.asarray(b).shape or not np.allclose(
            a, b, atol=1e-12, rtol=0
        ):
            raise ValueError(
                f"URDF/XML parameters differ: {xml.name}: {label}"
            )
        maximum = max(
            maximum, float(np.max(np.abs(np.asarray(a) - b), initial=0))
        )

    for index in range(1, source.nbody):
        name = source.body(index).name
        other = target.body(name).id
        if (
            source.body(int(source.body_parentid[index])).name
            != target.body(int(target.body_parentid[other])).name
        ):
            raise ValueError(f"URDF/XML body parent differs: {name}")
        for field in (
            "body_pos",
            "body_quat",
            "body_mass",
            "body_ipos",
            "body_iquat",
            "body_inertia",
        ):
            equal(
                getattr(source, field)[index],
                getattr(target, field)[other],
                f"{name}/{field}",
            )
        left = np.flatnonzero(source.geom_bodyid == index)
        right = np.flatnonzero(target.geom_bodyid == other)
        if len(left) != len(right):
            raise ValueError(f"URDF/XML geom count differs: {name}")
        for a, b in zip(left, right, strict=True):
            for field in (
                "geom_type",
                "geom_pos",
                "geom_quat",
                "geom_size",
                "geom_contype",
                "geom_conaffinity",
                "geom_group",
                "geom_friction",
                "geom_solref",
                "geom_solimp",
                "geom_rgba",
            ):
                equal(
                    getattr(source, field)[a],
                    getattr(target, field)[b],
                    f"{name}/{field}",
                )
            if source.geom_type[a] == mujoco.mjtGeom.mjGEOM_MESH:
                x, y = int(source.geom_dataid[a]), int(target.geom_dataid[b])
                if source.mesh(x).name != target.mesh(y).name:
                    raise ValueError(f"URDF/XML mesh differs: {name}")
                for field in (
                    "mesh_vertadr",
                    "mesh_faceadr",
                    "mesh_vertnum",
                    "mesh_facenum",
                    "mesh_pos",
                    "mesh_quat",
                    "mesh_scale",
                ):
                    equal(
                        getattr(source, field)[x],
                        getattr(target, field)[y],
                        f"{name}/{field}",
                    )
                for field, address, count in (
                    ("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                    ("mesh_face", "mesh_faceadr", "mesh_facenum"),
                ):
                    start_a, start_b = (
                        getattr(source, address)[x],
                        getattr(target, address)[y],
                    )
                    size = getattr(source, count)[x]
                    equal(
                        getattr(source, field)[start_a : start_a + size],
                        getattr(target, field)[start_b : start_b + size],
                        f"{name}/{field}",
                    )
    for index in range(source.njnt):
        name = source.joint(index).name
        other = target.joint(name).id
        for field in (
            "jnt_type",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "jnt_actfrcrange",
        ):
            equal(
                getattr(source, field)[index],
                getattr(target, field)[other],
                f"{name}/{field}",
            )
    return dict(
        nq=source.nq,
        nv=source.nv,
        bodies=source.nbody - 1,
        geoms=source.ngeom,
        max_parameter_difference=maximum,
    )


def verify_robots(resources):
    return {
        side: verify_robot(
            resources.path(str(Path(entry).with_suffix(".urdf"))),
            resources.path(entry),
        )
        for side, entry in resources.entries.items()
    }


def main():
    from .resources import ResourceTree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_robots(ResourceTree(args.assets)), indent=2))


if __name__ == "__main__":
    main()
