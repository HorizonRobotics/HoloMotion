"""Device/capture-system source configs for HoloSMPL conversion."""

from __future__ import annotations

from holosmpl.converters.bvh_family.noitom import (
    classify_noitom_bvh_source,
    convert_noitom_bvh_sample,
)
from holosmpl.converters.bvh_family.chingmu import (
    classify_chingmu_bvh_source,
    convert_chingmu_bvh_sample,
)
from holosmpl.converters.smpl_family.gvhmr import classify_gvhmr_source, convert_gvhmr_sample
from holosmpl.converters.pico.adapter import classify_pico_source, convert_pico_sample

SOURCE_CONFIGS = {
    "pico4_ultra_enterprise": {
        "source_name": "Pico4UltraEnterprise",
        "source_glob": "*.csv",
        "classify": classify_pico_source,
        "convert": convert_pico_sample,
        "convertible_status": "convertible_pico_csv",
    },
    "chingmu_bvh": {
        "source_name": "ChingmuBVH",
        "source_glob": "*.bvh",
        "classify": classify_chingmu_bvh_source,
        "convert": convert_chingmu_bvh_sample,
        "convertible_status": "convertible_bvh",
    },
    "noitom_bvh": {
        "source_name": "NoitomBVH",
        "source_glob": "*.bvh",
        "classify": classify_noitom_bvh_source,
        "convert": convert_noitom_bvh_sample,
        "convertible_status": "convertible_bvh",
    },
    "gvhmr": {
        "source_name": "GVHMR",
        "source_glob": "*.npz",
        "classify": classify_gvhmr_source,
        "convert": convert_gvhmr_sample,
        "convertible_status": "convertible_motion_npz",
    },
}
