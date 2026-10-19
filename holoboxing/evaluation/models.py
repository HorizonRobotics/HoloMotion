"""Fetch and verify the existing public HoloMotion V2 checkpoint bundle."""

from pathlib import Path
import json
import os
import tempfile
from urllib.request import urlopen

from .runtime.model_contract import sha256

MANIFEST_PATH = Path(__file__).with_name("frozen-model.json")


def frozen_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def verify_bundle(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = frozen_manifest()
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise FileNotFoundError(
                f"Frozen V2 bundle is missing {name}; use the download command"
            )
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(
                f"Frozen V2 bundle size or SHA256 mismatch: {name}"
            )
    return root


def download_bundle(directory):
    manifest = frozen_manifest()
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            path.is_file()
            and path.stat().st_size == expected["bytes"]
            and sha256(path) == expected["sha256"]
        ):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{manifest['repository']}/resolve/"
            f"{manifest['revision']}/{manifest['subdirectory']}/{name}?download=true"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".download-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                with urlopen(url, timeout=60) as response:
                    total = 0
                    while block := response.read(8 * 1024 * 1024):
                        total += len(block)
                        if total > expected["bytes"]:
                            raise ValueError(
                                f"Download exceeds the pinned size: {name}"
                            )
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if (
                temporary.stat().st_size != expected["bytes"]
                or sha256(temporary) != expected["sha256"]
            ):
                raise ValueError(
                    f"Downloaded V2 bundle size or SHA256 mismatch: {name}"
                )
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify_bundle(root)
