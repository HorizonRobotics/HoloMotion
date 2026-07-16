from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from holosmpl.visualization.body_model import BodyModelOutput
from holosmpl.visualization.canonical_loader import CanonicalClip


SMPL_BODY_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)

VIEWS = (
    ("front", 10, -90),
    ("side", 10, 0),
    ("top", 90, -90),
)


def render_contact_sheet(
    *,
    clip: CanonicalClip,
    body_output: BodyModelOutput,
    frame_indices: list[int],
    output_path: str | Path,
    max_mesh_faces: int = 3000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = body_output.vertices
    joints = body_output.joints
    faces = _downsample_faces(body_output.faces, max_mesh_faces)
    limits = _axis_limits(vertices)

    rows = len(frame_indices)
    cols = len(VIEWS)
    fig = plt.figure(figsize=(cols * 4.3, rows * 3.6), dpi=130)
    fig.suptitle(
        (
            f"{clip.clip_id}\n"
            f"{body_output.model_type.upper()} mesh+skeleton | "
            f"{clip.source_fps:g}->{clip.target_fps:g} Hz | {clip.pose_body_layout}"
        ),
        fontsize=10,
    )

    for row, frame_idx in enumerate(frame_indices):
        for col, (view_name, elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(rows, cols, row * cols + col + 1, projection="3d")
            mesh = Poly3DCollection(
                vertices[row][faces],
                alpha=0.34,
                facecolor=(0.35, 0.66, 0.95),
                edgecolor=(0.15, 0.25, 0.32, 0.08),
                linewidth=0.08,
            )
            ax.add_collection3d(mesh)
            _draw_skeleton(ax, joints[row])
            _draw_ground(ax, limits)
            _draw_axes(ax, limits)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_box_aspect(
                (
                    limits[0][1] - limits[0][0],
                    limits[1][1] - limits[1][0],
                    limits[2][1] - limits[2][0],
                )
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("x", labelpad=-8, fontsize=7)
            ax.set_ylabel("y", labelpad=-8, fontsize=7)
            ax.set_zlabel("z", labelpad=-8, fontsize=7)
            ax.set_title(f"{view_name} | frame {frame_idx}", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)


def _downsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces
    step = max(1, int(np.ceil(len(faces) / max_faces)))
    return faces[::step]


def _axis_limits(vertices: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xyz_min = vertices.reshape(-1, 3).min(axis=0)
    xyz_max = vertices.reshape(-1, 3).max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = float(np.max(xyz_max - xyz_min))
    span = max(span, 1.5)
    half = span * 0.58
    z_min = min(0.0, float(center[2] - half))
    z_max = max(1.5, float(center[2] + half))
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (z_min, z_max),
    )


def _draw_skeleton(ax: Any, joints: np.ndarray) -> None:
    max_joint = min(len(joints), 24)
    for a, b in SMPL_BODY_EDGES:
        if a >= max_joint or b >= max_joint:
            continue
        pts = joints[[a, b]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=1.6)
    body = joints[:max_joint]
    ax.scatter(body[:, 0], body[:, 1], body[:, 2], color="black", s=5, depthshade=False)


def _draw_ground(ax: Any, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    xlim, ylim, _ = limits
    xs = np.linspace(xlim[0], xlim[1], 5)
    ys = np.linspace(ylim[0], ylim[1], 5)
    for x in xs:
        ax.plot([x, x], [ylim[0], ylim[1]], [0, 0], color=(0.55, 0.55, 0.55), linewidth=0.35)
    for y in ys:
        ax.plot([xlim[0], xlim[1]], [y, y], [0, 0], color=(0.55, 0.55, 0.55), linewidth=0.35)


def _draw_axes(ax: Any, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    origin = np.array([limits[0][0], limits[1][0], 0.0], dtype=np.float32)
    length = 0.3 * max(limits[0][1] - limits[0][0], limits[1][1] - limits[1][0])
    axes = (
        ((length, 0, 0), "r", "x"),
        ((0, length, 0), "g", "y"),
        ((0, 0, length), "b", "z"),
    )
    for direction, color, label in axes:
        end = origin + np.asarray(direction, dtype=np.float32)
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=color, linewidth=1.2)
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=7)
