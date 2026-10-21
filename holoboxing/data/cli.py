"""Download sample motions and prepare robot training inputs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Fetch original public upstream files"
    )
    download.add_argument("dataset", choices=("robostriker", "interhuman"))
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--repair", action="store_true")
    download.add_argument("--retries", type=int, default=2)
    download.add_argument("--limit-groups", type=int)
    download.add_argument("--limit-clips", type=int)
    download.add_argument(
        "--clip-id",
        action="append",
        dest="clip_ids",
        help="InterHuman clip ID; repeat to override the bundled sample list",
    )

    retarget = commands.add_parser(
        "retarget", help="Convert downloaded paired SMPL into G1 motion"
    )
    retarget.add_argument("--source", type=Path, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--smpl-model", type=Path, required=True)
    retarget.add_argument("--iterations", type=int, default=4)
    retarget.add_argument("--device", default="cuda:0")
    retarget.add_argument("--limit-clips", type=int)
    retarget.add_argument("--frame-limit", type=int)

    prepare = commands.add_parser(
        "prepare", help="Produce V2 robot H5 and current boxing records"
    )
    prepare.add_argument(
        "--source", action="append", type=Path, required=True, dest="sources"
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.0)

    verify = commands.add_parser(
        "verify", help="Check bytes, current training reader and V2 H5 layout"
    )
    verify.add_argument("--output", type=Path, required=True)
    for command in (retarget, prepare, verify):
        command.add_argument(
            "--assets",
            type=Path,
            default=Path(__file__).resolve().parents[1] / "assets",
        )
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "download":
        from .download import download_interhuman, download_robostriker

        dataset = args.pop("dataset")
        if dataset == "robostriker":
            if args.pop("limit_clips") is not None or args.pop("clip_ids"):
                parser.error("Use --limit-groups for RoboStriker")
            report = download_robostriker(**args)
        else:
            if args.pop("limit_groups") is not None:
                parser.error("Use --limit-clips or --clip-id for InterHuman")
            report = download_interhuman(**args)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("files", "requested_clip_ids")
                }
                | (
                    {
                        "selected_clip_ids": [
                            Path(row["path"]).stem
                            for row in report["files"]
                            if row["path"].startswith("motions/")
                            and row["path"].endswith(".pkl")
                        ],
                        "requested_clip_count": len(
                            report["requested_clip_ids"]
                        ),
                        "receipt": str(args["output"] / "download.json"),
                    }
                    if dataset == "interhuman"
                    else {}
                ),
                indent=2,
            )
        )
    elif command == "retarget":
        from .interhuman import retarget_dataset

        report = retarget_dataset(**args)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key != "records"
                },
                indent=2,
            )
        )
    elif command == "prepare":
        from .prepare import prepare_dataset

        print(json.dumps(prepare_dataset(**args), indent=2))
    else:
        from .prepare import verify_dataset

        print(json.dumps(verify_dataset(**args), indent=2))
    return 0
