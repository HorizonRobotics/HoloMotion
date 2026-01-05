import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def _eval_rule(val: float, op: str, thr: float) -> bool:
    if op == ">":
        return val > thr
    if op == ">=":
        return val >= thr
    if op == "<":
        return val < thr
    if op == "<=":
        return val <= thr
    if op == "==":
        return val == thr
    if op == "!=":
        return val != thr
    raise ValueError(f"Unsupported op: {op}")


def _deep_get(container: Dict[str, Any], parts: List[str]) -> Optional[float]:
    cur: Any = container
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _resolve_value(
    tags_root: Dict[str, Any],
    clip_group: Dict[str, Any],
    path: str,
) -> Optional[float]:
    """Resolve a threshold path to a numeric value.

    - dataset_stats.<feature>.<DS_stat> reads from tags_root
    - kinematic_features.<feature>.<stat> reads from clip_group
    - <feature>.<stat> (no prefix) also reads from clip_group for convenience.
    """
    parts = str(path).split(".")
    if len(parts) == 0:
        return None
    if parts[0] == "dataset_stats":
        return _deep_get(tags_root, parts)
    if parts[0] == "kinematic_features":
        return _deep_get(clip_group, parts[1:])
    return _deep_get(clip_group, parts)


def filter_with_schema(
    tags: Dict[str, Any],
    schema: Dict[str, Any],
) -> Tuple[Set[str], Dict[str, int], Dict[str, int]]:
    thresholds: Dict[str, Dict[str, Any]] = schema.get("thresholds", {}) or {}
    across_mode = str(schema.get("across", "union"))
    out: Set[str] = set()
    path_counts: Dict[str, int] = {}
    group_counts: Dict[str, int] = {}

    clips: Dict[str, Dict[str, Any]] = tags.get("clip_info", {}) or {}
    for motion_key, groups in tqdm(
        clips.items(), desc="Evaluating schema", unit="clip"
    ):
        hits: List[bool] = []
        hits_by_path: Dict[str, bool] = {}
        group_hit_any: Dict[str, bool] = {}
        for path, spec in thresholds.items():
            parts = str(path).split(".")
            if len(parts) == 0:
                continue
            val = _resolve_value(tags, groups, path)
            if val is None:
                continue
            op = str(spec.get("op", ">"))
            thr = float(spec["value"])
            hit = _eval_rule(val, op, thr)
            hits.append(hit)
            hits_by_path[path] = hit
            grp = parts[0]
            if hit:
                group_hit_any[grp] = True
        if len(hits) == 0:
            continue
        if across_mode == "union":
            excluded = any(hits)
        elif across_mode == "intersection":
            excluded = all(hits)
        else:
            raise ValueError(f"Invalid across mode: {across_mode}")
        if not excluded:
            continue
        out.add(motion_key)
        # accumulate counts for excluded clips
        for pth, hit in hits_by_path.items():
            if hit:
                path_counts[pth] = path_counts.get(pth, 0) + 1
        for grp, any_hit in group_hit_any.items():
            if any_hit:
                group_counts[grp] = group_counts.get(grp, 0) + 1
    return out, path_counts, group_counts


def _default_schema_path() -> Path:
    # holomotion/src/motion_retargeting/kinematic_filter.py
    # -> holomotion/config/motion_retargeting/kinematic_filtering_schema.yaml
    this_file = Path(__file__).resolve()
    holomotion_dir = this_file.parents[2]
    return (
        holomotion_dir
        / "config"
        / "motion_retargeting"
        / "kinematic_filtering_schema.yaml"
    )


def run(
    dataset_root: str,
    schema_yaml_path: Optional[str] = None,
    output_yaml_path: Optional[str] = None,
    schema_obj: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """Execute kinematic filtering using tags and a schema.

    - dataset_root: directory containing 'kinematic_tags.json'
    - schema_yaml_path: external YAML with 'across' and 'thresholds' (optional)
    - schema_obj: inline dict with 'across' and 'thresholds' (optional)
    - output_yaml_path: where to write the excluded list YAML (optional)
    """
    root = Path(dataset_root).expanduser().resolve()
    tags_path = root / "kinematic_tags.json"
    if not tags_path.is_file():
        raise FileNotFoundError(f"Missing kinematic tags JSON: {tags_path}")

    schema: Dict[str, Any]
    if schema_obj is not None:
        schema = dict(schema_obj)
    else:
        schema_path = (
            Path(schema_yaml_path).expanduser().resolve()
            if schema_yaml_path
            else _default_schema_path()
        )
        if not schema_path.is_file():
            raise FileNotFoundError(f"Missing schema YAML: {schema_path}")
        schema = yaml.safe_load(open(schema_path, "r", encoding="utf-8"))

    out_yaml = (
        Path(output_yaml_path).expanduser().resolve()
        if output_yaml_path
        else (root / "excluded_kinematic_motion_names.yaml")
    )

    logger.info(f"Dataset root: {root}")
    logger.info(f"Reading tags from: {tags_path}")
    logger.info(
        "Using schema from: inline config"
        if schema_obj is not None
        else "Using schema from YAML file"
    )
    # Pretty-print resolved schema to console
    try:
        logger.info(
            "Resolved schema:\n"
            + yaml.safe_dump(schema, sort_keys=True, default_flow_style=False)
        )
    except Exception:
        pass

    tags = json.load(open(tags_path, "r", encoding="utf-8"))

    # Dump the used filter config into dataset root
    try:
        used_cfg = {
            "dataset_root": str(root),
            "output_yaml": str(out_yaml),
            "schema": schema,
        }
        with open(
            root / "kinematic_filter_config_used.yaml", "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(
                used_cfg, f, sort_keys=True, default_flow_style=False
            )
    except Exception:
        pass

    excluded_keys, path_counts, group_counts = filter_with_schema(tags, schema)
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("# @package _global_\n\n")
        f.write("excluded_motion_names:\n")
        for k in sorted(excluded_keys):
            f.write(f"- {k}\n")

    logger.info(f"Excluded by config: {len(excluded_keys)}")
    if len(group_counts) > 0:
        logger.info("Excluded counts by category:")
        for grp, cnt in sorted(
            group_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            logger.info(f"- {grp}: {cnt}")
    if len(path_counts) > 0:
        logger.info("Excluded counts by threshold path:")
        for pth, cnt in sorted(
            path_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            logger.info(f"- {pth}: {cnt}")
    logger.info(f"Wrote excluded list to: {out_yaml}")
    return excluded_keys


@hydra.main(
    config_path="../../config",
    config_name="motion_retargeting/kinematic_filter",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    dataset_root = str(cfg.io.dataset_root)
    # Optional fields (external schema YAML override and output path)
    schema_val = ""
    out_val = ""
    schema_obj = None
    if "schema" in cfg:
        # Inline schema object
        schema_obj = OmegaConf.to_object(cfg.schema)
    if "filtering" in cfg and hasattr(cfg.filtering, "schema_yaml"):
        schema_val = str(cfg.filtering.get("schema_yaml", "") or "")
        out_val = str(cfg.filtering.get("output_yaml", "") or "")
    elif "filtering" in cfg:
        out_val = str(cfg.filtering.get("output_yaml", "") or "")

    schema_yaml_path = schema_val if len(schema_val) > 0 else None
    output_yaml_path = out_val if len(out_val) > 0 else None

    run(
        dataset_root=dataset_root,
        schema_yaml_path=schema_yaml_path,
        schema_obj=schema_obj,
        output_yaml_path=output_yaml_path,
    )


if __name__ == "__main__":
    main()
