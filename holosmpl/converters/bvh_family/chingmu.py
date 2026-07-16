from __future__ import annotations

from pathlib import Path
from typing import Any

from holosmpl.converters.bvh_family.noitom import (
    classify_noitom_bvh_source,
    convert_bvh_family_sample,
)


CHINGMU_BVH_SOURCE_COORDINATE_FRAME = "chingmu_bvh_y_up_xyz_cm"
CHINGMU_BVH_CANONICAL_TRANSFORM = (
    "bvh_yup_cm_to_canonical_zup_m_root_orient_premultiply_and_trans_transform"
)
CHINGMU_BVH_UNIT_SCALE = 0.01
CHINGMU_BVH_DROP_FIRST_FRAME = True
CHINGMU_BVH_ZERO_BETA_DIM = 10


def classify_chingmu_bvh_source(source_path: str | Path) -> dict[str, Any]:
    """Classify whether one Chingmu BVH is convertible."""

    return classify_noitom_bvh_source(source_path)


def convert_chingmu_bvh_sample(
    source_path: str | Path,
    *,
    input_root: str | Path,
    target_fps: float,
) -> dict[str, Any]:
    """Convert one Chingmu BVH into canonical SMPL-like arrays.

    This is an approximate BVH-to-SMPL mapping.  The source skeleton uses
    SMPL-like semantic joint names, but it is not a fitted SMPL source.  The
    first frame is a T-pose calibration frame and is intentionally dropped.
    """

    result = convert_bvh_family_sample(
        source_path,
        input_root=input_root,
        target_fps=target_fps,
        dataset_name="ChingmuBVH",
        source_coordinate_frame=CHINGMU_BVH_SOURCE_COORDINATE_FRAME,
        unit_scale=CHINGMU_BVH_UNIT_SCALE,
        drop_first_frame=CHINGMU_BVH_DROP_FIRST_FRAME,
        zero_beta_dim=CHINGMU_BVH_ZERO_BETA_DIM,
        canonical_transform=CHINGMU_BVH_CANONICAL_TRANSFORM,
        repair_root_translation_spikes=True,
    )
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "dataset": "ChingmuBVH",
            "adapter": "chingmu_bvh",
            "source_dataset": "ChingmuBVH",
            "chingmu_source_policy": (
                "all Chingmu BVH files, 120Hz, first T-pose frame dropped"
            ),
            "root_translation_spike_repair_source": (
                "ported from pair_data_construction chingmu_bvh_adapter.py"
            ),
        }
    )
    result["metadata"] = metadata
    return result
