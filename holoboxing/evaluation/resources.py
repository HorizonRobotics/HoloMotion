"""Explicit, read-only boxing resources with relative-reference verification."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

from .runtime.model_contract import sha256


class ResourceTree:
    def __init__(self, directory, *, verify=True):
        self.root = Path(directory).expanduser().resolve(strict=True)
        manifest = self.path("resources.json")
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported boxing resource manifest")
        self.manifest_sha256 = sha256(manifest)
        self.entries = self.manifest["entries"]
        self.scene = self.path(self.manifest["simulation_entry"])
        self.initial_state = self.path(self.manifest["initial_state"])
        if verify:
            self.verify()

    def path(self, relative):
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(
                f"Resource path must remain inside --assets: {relative}"
            )
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Resource path escapes --assets: {relative}")
        return path

    def verify(self):
        inventory = self.manifest.get("files")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("A nonempty resource inventory is required")
        seen = set()
        for item in inventory:
            name = item["path"]
            if name in seen:
                raise ValueError(f"Duplicate resource inventory entry: {name}")
            seen.add(name)
            path = self.path(name)
            if (
                path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]
            ):
                raise ValueError(f"Resource size or SHA256 differs: {name}")
        required = [
            self.manifest["simulation_entry"],
            self.manifest["initial_state"],
            *self.entries.values(),
        ]
        if any(name not in seen for name in required):
            raise ValueError(
                "Scene and robot entries must be fingerprinted resources"
            )
        references = 0
        for name in seen:
            path = self.path(name)
            if path.suffix.lower() not in (".xml", ".urdf"):
                continue
            root = ET.parse(path).getroot()
            compiler = root.find("compiler")
            directories = {} if compiler is None else compiler.attrib
            for node in root.iter():
                reference = node.get("file") or node.get("filename")
                if not reference:
                    continue
                key = (
                    "meshdir"
                    if node.tag == "mesh"
                    else "texturedir"
                    if node.tag == "texture"
                    else ""
                )
                dependency = (
                    path.parent / directories.get(key, "") / reference
                ).resolve()
                if (
                    not dependency.is_relative_to(self.root)
                    or not dependency.is_file()
                ):
                    raise ValueError(
                        f"Unresolved relative resource: {name}: {reference}"
                    )
                relative = dependency.relative_to(self.root).as_posix()
                if relative not in seen:
                    raise ValueError(
                        f"Referenced file is absent from the inventory: {relative}"
                    )
                references += 1
        return {
            "files": len(inventory),
            "bytes": sum(item["bytes"] for item in inventory),
            "relative_references": references,
            "manifest_sha256": self.manifest_sha256,
        }
