"""Supported HoloSMPL source list and dispatch helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    key: str
    group: str
    module: str
    config_name: str = "SOURCE_CONFIG"
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    input_hint: str = ""
    notes: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return (self.key, *self.aliases)


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="amass",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.amass",
        display_name="AMASS",
        input_hint="AMASS raw SMPL/SMPL-H/SMPL-X NPZ tree",
    ),
    SourceSpec(
        key="lafan1",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.lafan1",
        aliases=("lafan",),
        display_name="LaFAN1",
        input_hint="LaFAN1 BVH files",
    ),
    SourceSpec(
        key="motionmillion",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.motionmillion",
        display_name="MotionMillion",
        input_hint="MotionMillion SMPL-family NPZ files",
    ),
    SourceSpec(
        key="omomo",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.omomo",
        aliases=("omomo_npz",),
        display_name="OMOMO",
        input_hint="OMOMO NPZ files",
    ),
    SourceSpec(
        key="gvhmr",
        group="devices",
        module="holosmpl.supported_devices.registry",
        config_name="SOURCE_CONFIGS.gvhmr",
        display_name="GVHMR",
        input_hint="SMPL NPZ files produced by GVHMR video reconstruction",
    ),
    SourceSpec(
        key="bones_seed_smpl",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.bones_seed_smpl",
        display_name="BonesSeedSMPL",
        input_hint="BonesSeedSMPL SMPL PKL files",
    ),
    SourceSpec(
        key="custom_smplx",
        group="datasets",
        module="holosmpl.supported_datasets.registry",
        config_name="SOURCE_CONFIGS.custom_smplx",
        display_name="CustomSMPLX",
        input_hint="Packed custom SMPL-X PKL files",
    ),
    SourceSpec(
        key="pico4_ultra_enterprise",
        group="devices",
        module="holosmpl.supported_devices.registry",
        config_name="SOURCE_CONFIGS.pico4_ultra_enterprise",
        aliases=("pico",),
        display_name="Pico 4 Ultra Enterprise/XRoboToolkit",
        input_hint="Pico 4 Ultra Enterprise/XRoboToolkit CSV files",
    ),
    SourceSpec(
        key="chingmu_bvh",
        group="devices",
        module="holosmpl.supported_devices.registry",
        config_name="SOURCE_CONFIGS.chingmu_bvh",
        aliases=("chingmu",),
        display_name="Chingmu BVH",
        input_hint="Chingmu BVH files",
    ),
    SourceSpec(
        key="noitom_bvh",
        group="devices",
        module="holosmpl.supported_devices.registry",
        config_name="SOURCE_CONFIGS.noitom_bvh",
        aliases=("noitom",),
        display_name="NoitomBVH",
        input_hint="Noitom BVH files",
    ),
)

_SOURCE_BY_NAME = {
    name.lower(): spec
    for spec in SOURCE_SPECS
    for name in spec.names
}


def list_sources(group: str | None = None) -> list[SourceSpec]:
    if group is None:
        return list(SOURCE_SPECS)
    normalized = group.lower()
    return [spec for spec in SOURCE_SPECS if spec.group == normalized]


def get_source_spec(name: str) -> SourceSpec | None:
    return _SOURCE_BY_NAME.get(str(name).lower())


def get_source_config(name: str) -> dict[str, Any] | None:
    spec = get_source_spec(name)
    if spec is None:
        return None
    module = importlib.import_module(spec.module)
    value: Any = module
    for part in spec.config_name.split("."):
        if isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    return dict(value)


def source_table() -> list[dict[str, Any]]:
    return [
        {
            "key": spec.key,
            "group": spec.group,
            "display_name": spec.display_name or spec.key,
            "aliases": list(spec.aliases),
            "input_hint": spec.input_hint,
            "notes": spec.notes,
        }
        for spec in SOURCE_SPECS
    ]
