"""Height-only fall detection used by the GPU match referee and logic tests."""

import numpy as np

FALL_HEIGHT_M = 0.30
FALL_BODY_NAMES = ("pelvis", "torso_link")


def fallen_from_heights(heights, *, floor_height=0.0, threshold=FALL_HEIGHT_M):
    heights = np.asarray(heights, dtype=np.float64)
    if heights.shape != (2, 2) or not np.isfinite(heights).all():
        raise ValueError(
            "Expected finite pelvis/torso heights for two fighters"
        )
    if not np.isfinite(floor_height) or threshold != FALL_HEIGHT_M:
        raise ValueError("Official fall threshold must be 0.30 m")
    return np.any(heights < floor_height + threshold, axis=1)


def fallen_from_scene(scene):
    ids = [
        [scene.model.body(prefix + name).id for name in FALL_BODY_NAMES]
        for prefix in ("p1_", "p2_")
    ]
    floor_id = scene.model.geom("fight_floor").id
    return fallen_from_heights(
        scene.data.xpos[np.asarray(ids), 2],
        floor_height=float(scene.data.geom_xpos[floor_id, 2]),
    )
