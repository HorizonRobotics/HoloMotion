"""Shared resource arguments."""

import argparse
import os
from pathlib import Path


def existing_path(value):
    try:
        return Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise argparse.ArgumentTypeError(f"Cannot access {value}: {error.strerror}") from error


def add_controller_arguments(parser):
    for name, description in (
        ("assets", "competition asset directory containing resources.json"),
        (
            "lower",
            "public HoloMotion V2 bundle containing config.yaml "
            "and the exported ONNX model",
        ),
    ):
        default = os.environ.get("HOLOBOXING_" + name.upper())
        parser.add_argument(
            "--" + name,
            type=existing_path,
            default=default,
            required=not bool(default),
            help=description,
        )
