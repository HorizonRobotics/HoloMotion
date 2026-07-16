"""Public dataset source configs for HoloSMPL conversion."""

from __future__ import annotations

from holosmpl.converters.bvh_family.lafan1 import classify_lafan1_source, convert_lafan1_sample
from holosmpl.converters.smpl_family.amass import classify_amass_source, convert_amass_sample
from holosmpl.converters.smpl_family.motionmillion import (
    classify_motionmillion_source,
    convert_motionmillion_sample,
)
from holosmpl.converters.smpl_family.omomo import classify_omomo_source, convert_omomo_sample
from holosmpl.converters.smpl_family.custom_smplx import (
    classify_custom_smplx_source,
    iter_convert_custom_smplx_source,
)
from holosmpl.converters.smpl_family.bones_seed_smpl import classify_bones_seed_smpl_source, convert_bones_seed_smpl_sample

SOURCE_CONFIGS = {
    "amass": {
        "source_name": "AMASS",
        "source_glob": "*.npz",
        "classify": classify_amass_source,
        "convert": convert_amass_sample,
        "convertible_status": "convertible_motion_npz",
    },
    "lafan1": {
        "source_name": "LaFAN1",
        "source_glob": "*.bvh",
        "classify": classify_lafan1_source,
        "convert": convert_lafan1_sample,
        "convertible_status": "convertible_bvh",
    },
    "motionmillion": {
        "source_name": "MotionMillion",
        "source_glob": "*.npz",
        "classify": classify_motionmillion_source,
        "convert": convert_motionmillion_sample,
        "convertible_status": "convertible_motion_npz",
    },
    "omomo": {
        "source_name": "OMOMO",
        "source_glob": "*.npz",
        "classify": classify_omomo_source,
        "convert": convert_omomo_sample,
        "convertible_status": "convertible_motion_npz",
    },
    "bones_seed_smpl": {
        "source_name": "BonesSeedSMPL",
        "source_glob": "*.pkl",
        "classify": classify_bones_seed_smpl_source,
        "convert": convert_bones_seed_smpl_sample,
        "convertible_status": "convertible_bones_seed_smpl_pkl",
    },
    "custom_smplx": {
        "source_name": "CustomSMPLX",
        "source_glob": "*.pkl",
        "classify": classify_custom_smplx_source,
        "iter_convert": iter_convert_custom_smplx_source,
        "convertible_status": "convertible_custom_smplx_pkl",
        "multi_clip_source": True,
    },
}
