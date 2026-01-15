# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_TEMPLATE_PATH = Path("index_wooden_static.html")
DEFAULT_OUT_HTML = Path("vis.html")

POSE_JOINTS = 22
EULER_FIX_DEG = (-90.0, 180.0, 0.0)
EULER_ORDER = "xyz"

# Empirical vertical offset (in meters) to align wooden_static visualization mesh
# with canonical SMPL coordinates (e.g., GVHMR pipelines).
WOODEN_SMPL_HEIGHT_OFFSET = 0.2


@dataclass(frozen=True)
class SmplSequence:
    """A minimal SMPL motion sequence loaded from npz."""
    poses: np.ndarray  # (T, 66) = root(3) + body(63), axis-angle
    trans: np.ndarray  # (T, 3)
    betas: np.ndarray  # (B,)
    fps: float
    gender: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate vis.html from a SMPL npz using a HTML template."
    )
    ap.add_argument("--npz", type=Path, help="Path to input .npz")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH, help="Path to HTML template")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_HTML, help="Path to output HTML")

    ap.add_argument(
        "--pose_joints",
        type=int,
        default=POSE_JOINTS,
        help=f"Number of pose joints in poses (default: {POSE_JOINTS}).",
    )
    ap.add_argument(
        "--height_axis",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Axis index for height in Th (default: 1 for Y-up).",
    )
    ap.add_argument(
        "--height_offset",
        type=float,
        default=WOODEN_SMPL_HEIGHT_OFFSET,
        help=(
            "Subtract from Th height axis (Y-up), in meters. "
            "Default is an empirical offset to align wooden_static mesh "
            "with canonical SMPL coordinates (e.g., GVHMR)."
        ),
    )
    return ap.parse_args()


def euler_fix_rot(euler_deg=EULER_FIX_DEG, order=EULER_ORDER) -> R:
    """Rotation for world-frame correction: R_new = R_fix * R_old."""
    return R.from_euler(order.lower(), euler_deg, degrees=True)


def _require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in npz. Available: {list(data.keys())}")
    return data[key]


def load_npz(path: Path) -> SmplSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    data = np.load(path, allow_pickle=False)

    poses = _require_key(data, "poses").astype(np.float32)
    trans = _require_key(data, "trans").astype(np.float32)
    betas = _require_key(data, "betas").astype(np.float32)

    fps = float(np.asarray(_require_key(data, "mocap_framerate")))
    gender = str(np.asarray(_require_key(data, "gender")))

    return SmplSequence(poses=poses, trans=trans, betas=betas, fps=fps, gender=gender)


def validate_sequence(seq: SmplSequence, pose_joints: int) -> int:
    """Validate shapes and return T."""
    if seq.poses.ndim != 2:
        raise ValueError(f"poses must be 2D, got shape={seq.poses.shape}")
    if seq.trans.ndim != 2 or seq.trans.shape[1] != 3:
        raise ValueError(f"trans must be (T,3), got shape={seq.trans.shape}")

    T = int(seq.poses.shape[0])
    exp_dim = int(pose_joints) * 3

    if seq.poses.shape[1] != exp_dim:
        raise ValueError(f"unexpected poses shape: {seq.poses.shape}, expected (T,{exp_dim})")
    if seq.trans.shape[0] != T:
        raise ValueError(f"poses frames ({T}) != trans frames ({seq.trans.shape[0]})")

    return T


def build_smpl_frames(
    seq: SmplSequence,
    *,
    pose_joints: int,
    height_axis: int,
    height_offset: float,
) -> Tuple[list, int]:
    """
    Build frames in the format expected by index_wooden_static.html template.

    Notes:
        height_offset is a visualization-only correction to compensate for the
        vertical origin mismatch between wooden_static mesh and canonical SMPL
        coordinates (e.g., GVHMR). Override via --height_offset if needed.
    """
    T = validate_sequence(seq, pose_joints)

    rot_fix = euler_fix_rot()
    root_aa = seq.poses[:, :3]
    body_aa = seq.poses[:, 3:]

    # root: left-multiply world rotation
    Rh = (rot_fix * R.from_rotvec(root_aa)).as_rotvec().astype(np.float32)

    # trans: rotate in world frame, then apply visualization height offset
    Th = rot_fix.apply(seq.trans).astype(np.float32)
    if height_offset != 0.0:
        Th[:, int(height_axis)] -= float(height_offset)

    # pad hands (6) -> body(63) + hand(6) = 69
    poses_js = np.concatenate([body_aa, np.zeros((T, 6), np.float32)], axis=1)

    shapes = seq.betas.reshape(-1).tolist()
    frames = [[{
        "id": 0,
        "gender": seq.gender,
        "Rh": [Rh[f].tolist()],
        "Th": [Th[f].tolist()],
        "poses": [poses_js[f].tolist()],
        "shapes": shapes,
    }] for f in range(T)]

    return frames, T


def render_html(template_path: Path, frames: list, T: int, fps: float) -> str:
    template = template_path.read_text(encoding="utf-8")

    smpl_data_json = json.dumps(frames, ensure_ascii=False)
    caption_html = (
        "<div class='caption-overlay'><div class='motion-info'>"
        f"Frames: {T} &nbsp;&nbsp; Framerate: {fps:.1f} fps"
        "</div></div>"
    )

    return (template
            .replace("{{ smpl_data_json }}", smpl_data_json)
            .replace("{{ caption_html }}", caption_html))


def main(
    npz_path: Path,
    template_path: Path,
    out_html: Path,
    *,
    pose_joints: int,
    height_axis: int,
    height_offset: float,
) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Missing {template_path}")

    seq = load_npz(npz_path)
    frames, T = build_smpl_frames(
        seq,
        pose_joints=pose_joints,
        height_axis=height_axis,
        height_offset=height_offset,
    )

    html = render_html(template_path, frames, T, seq.fps)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] wrote {out_html.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.npz,
        args.template,
        args.out,
        pose_joints=args.pose_joints,
        height_axis=args.height_axis,
        height_offset=args.height_offset,
    )
