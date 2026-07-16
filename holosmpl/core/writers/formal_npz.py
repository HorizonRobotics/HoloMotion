from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_formal_npz(path: str | Path, clip: dict[str, Any]) -> None:
    """Write one standardized formal human clip NPZ."""

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(clip["metadata"])
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        path,
        human_pose_aa=np.asarray(clip["human_pose_aa"], dtype=np.float32),
        human_shape_beta=np.asarray(clip["human_shape_beta"], dtype=np.float32),
        human_root_trans=np.asarray(clip["human_root_trans"], dtype=np.float32),
        human_root_height=np.asarray(clip["human_root_height"], dtype=np.float32),
        human_gravity_projection=np.asarray(
            clip["human_gravity_projection"], dtype=np.float32
        ),
        metadata=np.asarray(metadata_json),
    )
