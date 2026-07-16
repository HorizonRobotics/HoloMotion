#!/usr/bin/env python3
"""Convert one legacy offline-tracking NPZ for HoloMotion v1.4 deployment."""

from __future__ import annotations

import argparse
import json

from humanoid_policy.offline_motion_conversion import convert_legacy_offline_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Legacy offline-tracking NPZ")
    parser.add_argument("output", help="Output v1.4 offline-tracking NPZ")
    parser.add_argument("--fps", type=float, default=None, help="Override motion FPS")
    parser.add_argument("--expected-dofs", type=int, default=29)
    parser.add_argument("--expected-bodies", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = convert_legacy_offline_npz(
        args.source,
        args.output,
        expected_dof_count=args.expected_dofs,
        expected_body_count=args.expected_bodies,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
