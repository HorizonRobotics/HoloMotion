"""Write checkpoints and completion records durably."""

import hashlib
import json
import os
import random
from pathlib import Path


def _publish(path, write):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    # Persist the rename as well as the file contents.
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def save_checkpoint(path, state):
    import torch

    return _publish(path, lambda stream: torch.save(state, stream))


def write_json(path, record):
    payload = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    return _publish(path, lambda stream: stream.write(payload))


def random_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else None,
    }


def restore_random_state(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training resume requires CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def training_result(output, checkpoint, *, progress, budget=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "progress": progress,
        "budget": budget.report() if budget is not None else None,
    }
    write_json(Path(output) / "training-result.json", record)
    return record
