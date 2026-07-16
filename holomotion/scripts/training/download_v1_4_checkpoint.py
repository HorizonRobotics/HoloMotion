#!/usr/bin/env python3
"""Download and validate the HoloMotion v1.4 checkpoint package."""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

REQUIRED_FILES = (
    "config.yaml",
    "model_14000.pt",
    "model_14000/actor/model.safetensors",
    "model_14000/critic/model.safetensors",
    "exported/model_14000.onnx",
)


def _with_subfolder(patterns: tuple[str, ...], subfolder: str) -> list[str]:
    subfolder = subfolder.strip("/")
    if not subfolder:
        return list(patterns)
    return [f"{subfolder}/{pattern}" for pattern in patterns]


def validate_checkpoint_layout(root: Path) -> None:
    """Require every file consumed by finetuning and deployment."""
    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (root / relative_path).is_file()
        or (root / relative_path).stat().st_size == 0
    ]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Incomplete HoloMotion v1.4 checkpoint under {root}:\n  "
            f"{joined}"
        )


def validate_checksums(root: Path) -> None:
    """Validate SHA256SUMS when the model repository provides it."""
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        return

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        relative_path = relative_path.lstrip("*")
        model_file = root / relative_path
        if not model_file.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe checksum path: {relative_path}")
        digest = hashlib.sha256()
        with model_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the HoloMotion v1.4 Hugging Face model."
    )
    parser.add_argument("repo_id", help="Hugging Face model repo, owner/name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/holomotion_v1.4"),
        help="Local package directory",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--subfolder",
        default="HoloMotion_motion_tracking_model_v1.4.0",
        help="Model package subfolder inside the Hugging Face repo.",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    subfolder = args.subfolder.strip("/")
    download_dir = output_dir.parent / f".{output_dir.name}.download"
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=download_dir if subfolder else output_dir,
        allow_patterns=[
            *_with_subfolder(REQUIRED_FILES, subfolder),
            "README.md",
            "LICENSE*",
            "*.md",
            *_with_subfolder(("README.md", "LICENSE*", "*.md", "SHA256SUMS"), subfolder),
        ],
        force_download=args.force_download,
    )
    if subfolder:
        package_dir = download_dir / subfolder
        if not package_dir.is_dir():
            raise FileNotFoundError(
                f"Hugging Face subfolder not found after download: {subfolder}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in package_dir.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(package_dir)
                target = output_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())

    validate_checkpoint_layout(output_dir)
    validate_checksums(output_dir)

    checkpoint = output_dir / "model_14000.pt"
    print(f"Checkpoint package ready: {output_dir}")
    print(
        "Set the finetune checkpoint with:\n"
        f"export HOLOMOTION_FINETUNE_CHECKPOINT='{checkpoint}'"
    )


if __name__ == "__main__":
    main()
