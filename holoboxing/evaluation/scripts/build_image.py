"""Prepare an allowlisted Docker context and build a bundled evaluation image.

No source repository, runtime weight or resource download occurs inside the
participant container. This command never pushes or publishes an image.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from holoboxing.evaluation.models import frozen_manifest, verify_bundle
from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.runtime.model_contract import sha256


def git(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def prepare_context(directory, assets, lower, *, development=False):
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(
            "Use a new context directory; existing contents are never overwritten"
        )
    # Verify before copying; every runtime resource is selected by its inventory.
    resources = ResourceTree(assets)
    lower = verify_bundle(lower)
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    if dirty and not development:
        raise ValueError(
            "Release builds require a clean public source commit; use --development for an explicit snapshot"
        )
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError(
            "Keep the generated image context outside the public source repository"
        )
    directory.mkdir(parents=True)

    def copy(source, relative):
        source = Path(source).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Expected a regular build input: {source}")
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    package = ROOT / "holoboxing"
    for name in ("__init__.py", "__main__.py"):
        copy(package / name, Path("app/holoboxing") / name)
    for source in (package / "evaluation").rglob("*.py"):
        relative = source.relative_to(package / "evaluation")
        if relative.parts[0] in ("tests", "scripts", "viewer"):
            continue
        if source.is_symlink() or "__pycache__" in relative.parts:
            raise ValueError(
                f"Unexpected generated or linked package input: {relative}"
            )
        copy(source, Path("app/holoboxing/evaluation") / relative)
    copy(
        ROOT / "holoboxing/evaluation/frozen-model.json",
        "app/holoboxing/evaluation/frozen-model.json",
    )
    for name in (
        "__init__.py",
        "link_actor_packing.py",
        "link_observation.py",
    ):
        copy(
            ROOT / "holomotion/src/motion_tracking" / name,
            Path("app/holomotion/src/motion_tracking") / name,
        )
    for name in ("LICENSE", "NOTICE"):
        copy(ROOT / name, Path("app") / name)
    for source in (ROOT / "holoboxing/evaluation/viewer").rglob("*"):
        relative = source.relative_to(ROOT / "holoboxing/evaluation/viewer")
        if (
            relative.parts[0] in ("node_modules", "dist", "tests")
            or not source.is_file()
        ):
            continue
        if source.is_symlink():
            raise ValueError(f"Linked viewer input: {relative}")
        copy(source, Path("viewer") / relative)
    for item in resources.manifest["files"]:
        copy(resources.path(item["path"]), Path("assets") / item["path"])
    for name in (
        "resources.json",
        "manifest.json",
        "README.md",
        "UNITREE_LICENSE",
    ):
        copy(resources.path(name), Path("assets") / name)
    for name in frozen_manifest()["files"]:
        copy(lower / name, Path("lower") / name)
    copy(ROOT / "holoboxing/evaluation/docker/Dockerfile", "Dockerfile")
    copy(
        ROOT / "holoboxing/evaluation/docker/requirements.lock",
        "requirements.lock",
    )
    copy(
        ROOT / "holoboxing/evaluation/docker/build-requirements.lock",
        "build-requirements.lock",
    )
    files = [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    source_fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "image_version": "v9",
        "code_revision": revision,
        "development_snapshot": dirty,
        "source_fingerprint": source_fingerprint,
        "resource_manifest_sha256": resources.manifest_sha256,
        "robot_contract": "urdf-gloves-v1",
        "collision_precision": "epa-plane-fp64-v1",
        "frozen_model": frozen_manifest(),
        "files": files,
    }
    (directory / "build-inputs.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--lower", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--tag", "--image", dest="tag", required=True)
    parser.add_argument(
        "--python-base",
        default="python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d",
        help="public python image@sha256 digest",
    )
    parser.add_argument(
        "--node-base",
        default="node:22.14.0-bookworm-slim@sha256:1c18d9ab3af4585870b92e4dbc5cac5a0dc77dd13df1a5905cea89fc720eb05b",
        help="public node image@sha256 digest",
    )
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for reference in (args.python_base, args.node_base):
        if (
            "@sha256:" not in reference
            or len(reference.rsplit("@sha256:", 1)[1]) != 64
        ):
            parser.error(
                "Base references must include their verified SHA256 digest"
            )
    manifest = prepare_context(
        args.context, args.assets, args.lower, development=args.development
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "context": str(Path(args.context).resolve()),
                    "inputs": manifest["source_fingerprint"],
                }
            )
        )
        return
    command = [
        "docker",
        "build",
        "--progress=plain",
        "--tag",
        args.tag,
        "--build-arg",
        f"PYTHON_BASE={args.python_base}",
        "--build-arg",
        f"NODE_BASE={args.node_base}",
        "--build-arg",
        f"CODE_REVISION={manifest['code_revision']}",
        "--build-arg",
        f"INPUTS_SHA256={sha256(Path(args.context) / 'build-inputs.json')}",
    ]
    # Docker's predefined proxy arguments stay outside the image environment.
    # Pass argument names so values and any credentials never enter command logs.
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        if name in os.environ:
            command.extend(("--build-arg", name))
    command.append(args.context)
    subprocess.run(command, check=True)
    inspect = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", args.tag], text=True
        )
    )[0]
    receipt = {
        "image_version": "v9",
        "image": args.tag,
        "image_id": inspect["Id"],
        "bytes": inspect["Size"],
        "code_revision": manifest["code_revision"],
        "development_snapshot": manifest["development_snapshot"],
        "inputs_sha256": sha256(Path(args.context) / "build-inputs.json"),
        "python_base": args.python_base,
        "node_base": args.node_base,
    }
    (Path(args.context) / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
