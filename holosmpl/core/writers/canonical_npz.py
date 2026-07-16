from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_canonical_npz(path: str | Path, clip: dict[str, Any]) -> None:
    """Write one standardized canonical clip NPZ."""

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(clip["metadata"])
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        path,
        root_orient=np.asarray(clip["root_orient"], dtype=np.float32),
        pose_body=np.asarray(clip["pose_body"], dtype=np.float32),
        trans=np.asarray(clip["trans"], dtype=np.float32),
        betas=np.asarray(clip["betas"], dtype=np.float32),
        gender=np.asarray(str(clip["gender"])),
        source_fps=np.asarray(float(clip["source_fps"]), dtype=np.float32),
        target_fps=np.asarray(float(clip["target_fps"]), dtype=np.float32),
        metadata=np.asarray(metadata_json),
    )
