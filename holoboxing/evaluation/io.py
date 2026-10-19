"""Atomic report output for local and platform consumers."""

from pathlib import Path
import json
import os
import tempfile


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".report-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                payload, stream, indent=2, allow_nan=False, ensure_ascii=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
