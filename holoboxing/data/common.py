"""Small, dependency-light provenance and source-reading helpers."""

from __future__ import annotations
import json
import pickle
from pathlib import Path, PurePosixPath

from holoboxing.evaluation.runtime.model_contract import sha256 as sha256


def decode_annotation(data):
    """Keep source bytes unchanged while accepting legacy annotation punctuation."""
    for encoding in ("utf-8-sig", "gb18030", "latin1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise AssertionError("Latin-1 decodes every byte")


def read_annotation(path):
    return decode_annotation(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inside(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid relative data path: {relative}")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Data path escapes its root: {relative}")
    return result


def new_directory(output, *, source=None):
    output = Path(output).expanduser().resolve()
    if source is not None:
        source = Path(source).expanduser().resolve(strict=True)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Keep outputs outside the source data tree")
    output.mkdir(parents=True, exist_ok=False)
    return output


class _ArrayUnpickler(pickle.Unpickler):
    """InterHuman uses NumPy arrays; arbitrary pickled callables are rejected."""

    def find_class(self, module, name):
        import numpy as np

        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name in ("_reconstruct", "scalar"):
                return getattr(np.core.multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric"):
            if name == "_frombuffer":
                return np.core.numeric._frombuffer
        raise pickle.UnpicklingError(
            f"Unsupported source type: {module}.{name}"
        )


def read_interhuman(path):
    with Path(path).open("rb") as stream:
        result = _ArrayUnpickler(stream).load()
    if not isinstance(result, dict):
        raise ValueError(
            f"{Path(path).name}: expected an InterHuman dictionary"
        )
    return result


def clip_timing(packet):
    import numpy as np

    rate = np.asarray(packet["mocap_framerate"]).reshape(-1)
    count = np.asarray(packet["frames"]).reshape(-1)
    if (
        rate.size != 1
        or count.size != 1
        or not np.isfinite(rate[0])
        or not np.isfinite(count[0])
        or rate[0] <= 0
        or count[0] < 2
        or int(count[0]) != count[0]
    ):
        raise ValueError(
            "InterHuman frames and mocap_framerate must be positive scalars"
        )
    return int(count[0]), float(rate[0])
