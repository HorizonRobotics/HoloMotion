"""Synchronize saved replay XMLs and resource inventories from gloved URDFs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.robot_models import export_xml, verify_robots
from holoboxing.evaluation.runtime.model_contract import sha256


def synchronize(directory):
    directory = Path(directory).resolve(strict=True)
    record = json.loads((directory / "resources.json").read_text())

    def path(name):
        result = (directory / name).resolve(strict=True)
        if not result.is_relative_to(directory) or not result.is_file():
            raise ValueError(f"Resource escapes --assets: {name}")
        return result

    for entry in record["entries"].values():
        urdf = path(str(Path(entry).with_suffix(".urdf")))
        path(entry).write_text(export_xml(urdf))
    for item in record["files"]:
        source = path(item["path"])
        item.update(bytes=source.stat().st_size, sha256=sha256(source))
    (directory / "resources.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    replay = json.loads((directory / "manifest.json").read_text())
    replay["entry"], replay["entries"] = record["entry"], record["entries"]
    replay["files"] = [item["path"] for item in record["files"]]
    replay["sha256"] = {
        item["path"]: item["sha256"] for item in record["files"]
    }
    (directory / "manifest.json").write_text(
        json.dumps(replay, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update XMLs and inventories; otherwise verify only",
    )
    args = parser.parse_args()
    if args.write:
        synchronize(args.assets)
    print(json.dumps(verify_robots(ResourceTree(args.assets)), indent=2))


if __name__ == "__main__":
    main()
