import json
import math
from pathlib import Path
from typing import Dict, Any, List


JSON_INPUT_FILE = "logs/Holomotion/metrics_output_dataset/model_17500.json"
input_path = Path(JSON_INPUT_FILE).expanduser().resolve()
OUTPUT_JSON_FILE = str(input_path.parent / "bad_clips.json")

WORST_PERCENTAGE = 0.2

METRICS_INFO: Dict[str, Dict[str, str]] = {
    "whole_body_joints_dist": {
        "name": "Joint Angle Error (Whole Body Average)",
        "unit": "rad",
        "direction": "higher_is_worse",
    },
}


def find_and_save_bad_clips(
    data: Dict[str, Any],
    metrics_info: Dict[str, Dict[str, str]],
    percentage: float,
    output_file: str,
) -> None:
    per_clip_data: List[Dict[str, Any]] = data.get("per_clip", [])
    if not per_clip_data:
        print("Error: 'per_clip' not found in JSON data.")
        return

    total_clips = len(per_clip_data)
    num_to_select = math.ceil(total_clips * percentage)
    if num_to_select == 0 and total_clips > 0:
        num_to_select = 1

    bad_clips_report: Dict[str, List[Dict[str, Any]]] = {}

    for key, info in metrics_info.items():
        direction = info.get("direction")
        if not direction:
            continue

        sort_descending = direction == "higher_is_worse"

        clips_with_metric_value = [
            {"motion_key": clip["motion_key"], "value": clip[key]}
            for clip in per_clip_data
            if key in clip and "motion_key" in clip
        ]

        if not clips_with_metric_value:
            print(f"Warning: no values found for metric '{key}' in data.")
            continue

        sorted_clips = sorted(
            clips_with_metric_value,
            key=lambda x: x["value"],
            reverse=sort_descending,
        )

        worst_clips = sorted_clips[:num_to_select]
        bad_clips_report[key] = worst_clips

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bad_clips_report, f, indent=4, ensure_ascii=False)
    print(f"Saved bad-clips report to: {output_file}")


def main() -> None:
    if not Path(JSON_INPUT_FILE).is_file():
        print(f"Error: JSON input file '{JSON_INPUT_FILE}' not found.")
        return

    with open(JSON_INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    find_and_save_bad_clips(
        data, METRICS_INFO, WORST_PERCENTAGE, OUTPUT_JSON_FILE
    )


if __name__ == "__main__":
    main()
