"""Verify that evaluation attaches the existing replay robots and arena.

This module uses only the standard library so asset checks and image-context
preparation do not need a simulator, GPU, or training framework.
"""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def verify_scene(resources):
    scene = ET.parse(resources.scene).getroot()
    world = scene.find("worldbody")
    models = {node.get("name"): node for node in scene.findall("asset/model")}
    attachments = [] if world is None else world.findall("attach")
    if (
        len(models) != 2
        or len(attachments) != 2
        or world.find(".//body") is not None
        or {node.get("prefix") for node in attachments} != {"p1_", "p2_"}
    ):
        raise ValueError("Evaluation must attach the two existing replay XMLs")
    allowed = {resources.manifest["simulation_entry"]}
    for side, prefix in (("red", "p1_"), ("blue", "p2_")):
        attachment = next(n for n in attachments if n.get("prefix") == prefix)
        model = models.get(attachment.get("model"))
        entry = resources.path(resources.entries[side])
        if (
            model is None
            or attachment.get("body") != "pelvis"
            or (resources.scene.parent / model.get("file", "")).resolve()
            != entry
        ):
            raise ValueError(f"Evaluation and replay robot differ: {side}")
        robot = ET.parse(entry).getroot()
        source = resources.path(
            str(Path(resources.entries[side]).with_suffix(".urdf"))
        )
        fingerprint = robot.find("custom/text[@name='source_urdf_sha256']")
        from .runtime.model_contract import sha256

        if fingerprint is None or fingerprint.get("data") != sha256(source):
            raise ValueError(
                f"Replay XML must match its canonical gloved URDF: {side}"
            )
        for hand in ("left", "right"):
            for kind in ("visual", "collision"):
                geom = robot.find(f".//geom[@name='{hand}_glove_{kind}']")
                if geom is None or geom.get(
                    "mesh"
                ) != f"{hand}_boxing_glove" + (
                    "_collision" if kind == "collision" else ""
                ):
                    raise ValueError(
                        f"Replay robot must use boxing gloves: {side}"
                    )
        allowed.update(
            (
                resources.entries[side],
                str(Path(resources.entries[side]).with_suffix(".urdf")),
            )
        )
    for item in resources.manifest["files"]:
        name = item["path"]
        if (
            Path(name).suffix.lower() in (".urdf", ".xml")
            and name not in allowed
        ):
            raise ValueError(f"Legacy robot/scene must not be bundled: {name}")
        if Path(name).stem in ("left_rubber_hand", "right_rubber_hand"):
            raise ValueError(f"Bare-hand mesh must not be bundled: {name}")
    replay = json.loads(resources.path("manifest.json").read_text())
    expected_hashes = {
        item["path"]: item["sha256"] for item in resources.manifest["files"]
    }
    if (
        replay.get("entries") != resources.entries
        or replay.get("entry") != resources.manifest.get("entry")
        or replay.get("sha256") != expected_hashes
        or set(replay.get("files", [])) != set(expected_hashes)
    ):
        raise ValueError("Replay and evaluation resource manifests differ")

    # The player consumes this same arena description. Physics uses its
    # declared collision proxies, rather than a second independent ring.
    arena = json.loads(
        resources.path(resources.manifest["arena_entry"]).read_text()
    )
    geoms = {node.get("name"): node for node in world.findall("geom")}
    meshes = {node.get("name"): node for node in scene.findall("asset/mesh")}

    def equal(actual, expected):
        try:
            values = [float(x) for x in actual.split()]
        except (AttributeError, ValueError):
            return False
        return len(values) == len(expected) and all(
            math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
            for a, b in zip(values, expected)
        )

    names = set()
    for box in arena["collision_boxes"]:
        names.add(box["name"])
        geom = geoms.get(box["name"])
        if (
            geom is None
            or geom.get("type") != "box"
            or not equal(geom.get("pos"), box["position"])
            or not equal(geom.get("size"), [x / 2 for x in box["size"]])
        ):
            raise ValueError(
                f"Evaluation and replay arena differ: {box['name']}"
            )
    for mesh in arena["collision_meshes"]:
        name = mesh["name"]
        names.add(name)
        definition, geom = meshes.get(name), geoms.get(name)
        if (
            definition is None
            or geom is None
            or geom.get("mesh") != name
            or not equal(
                definition.get("vertex"),
                [v for xyz in mesh["vertices"] for v in xyz],
            )
            or not equal(geom.get("margin"), [mesh["margin"]])
        ):
            raise ValueError(f"Evaluation and replay arena differ: {name}")
    if set(geoms) != names:
        raise ValueError(
            "Evaluation contains an independent arena collision surface"
        )
