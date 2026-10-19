"""Small self-contained gloved robots for resource and server checks."""

import json
from pathlib import Path
import struct

from holoboxing.evaluation.robot_models import export_xml
from holoboxing.evaluation.runtime.model_contract import sha256


def resign(root):
    record = json.loads((root / "resources.json").read_text())
    for item in record["files"]:
        path = root / item["path"]
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
    (root / "resources.json").write_text(json.dumps(record))
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                entry=record["entry"],
                entries=record["entries"],
                files=[item["path"] for item in record["files"]],
                sha256={
                    item["path"]: item["sha256"] for item in record["files"]
                },
            )
        )
    )


def make_resources(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vertices = [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, 0, 0.05)]
    mesh = b"fixture".ljust(80, b"\0") + struct.pack("<I", 4)
    for face in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
        mesh += struct.pack(
            "<12fH", 0, 0, 0, *[v for i in face for v in vertices[i]], 0
        )
    names = []
    for hand in ("left", "right"):
        for suffix in ("", "_collision"):
            name = f"{hand}_boxing_glove{suffix}.STL"
            (root / name).write_bytes(mesh)
            names.append(name)
    for side in ("red", "blue"):
        links = [
            '<link name="pelvis"><inertial><mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/></inertial></link>'
        ]
        for hand in ("left", "right"):
            links.append(f'''<link name="{hand}_rubber_hand">
              <inertial><mass value=".17"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial>
              <visual name="{hand}_glove_visual"><geometry><mesh filename="{hand}_boxing_glove.STL"/></geometry></visual>
              <collision name="{hand}_glove_collision"><geometry><mesh filename="{hand}_boxing_glove_collision.STL"/></geometry></collision>
              </link><joint name="{hand}_wrist_yaw_joint" type="revolute">
              <parent link="pelvis"/><child link="{hand}_rubber_hand"/>
              <origin xyz="0 .2 0"/><axis xyz="0 0 1"/>
              <limit lower="-1.614429558" upper="1.614429558" effort="13.4" velocity="27"/></joint>''')
        urdf = root / f"{side}.urdf"
        urdf.write_text(
            '<robot name="fixture"><mujoco><compiler strippath="false" discardvisual="false" inertiafromgeom="false"/></mujoco>'
            + "".join(links)
            + "</robot>"
        )
        (root / f"{side}.xml").write_text(export_xml(urdf))
        names.extend((f"{side}.urdf", f"{side}.xml"))
    (root / "scene.xml").write_text("""<mujoco><asset>
      <model name="red" file="red.xml"/><model name="blue" file="blue.xml"/>
      </asset><worldbody><geom name="smpl_ring_baseLink" type="box" pos="0 0 -.52" size="2.5 2.5 .5"/>
      <attach model="red" body="pelvis" prefix="p1_"/>
      <attach model="blue" body="pelvis" prefix="p2_"/>
      </worldbody></mujoco>""")
    (root / "reset.json").write_text("{}")
    (root / "arena.json").write_text(
        json.dumps(
            dict(
                collision_boxes=[
                    dict(
                        name="smpl_ring_baseLink",
                        position=[0, 0, -0.52],
                        size=[5, 5, 1],
                    )
                ],
                collision_meshes=[],
            )
        )
    )
    names.extend(("scene.xml", "reset.json", "arena.json"))
    record = dict(
        schema_version=1,
        entry="red.xml",
        entries=dict(red="red.xml", blue="blue.xml"),
        simulation_entry="scene.xml",
        arena_entry="arena.json",
        initial_state="reset.json",
        files=[dict(path=name) for name in names],
    )
    (root / "resources.json").write_text(json.dumps(record))
    resign(root)
    return root
