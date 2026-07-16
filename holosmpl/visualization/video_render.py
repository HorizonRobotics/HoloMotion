from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from holosmpl.core.processing.derived_fields import gravity_projection_from_root_orient
from holosmpl.visualization.body_model import BodyModelCache, BodyModelOutput
from holosmpl.visualization.canonical_loader import CanonicalClip


@dataclass(frozen=True)
class VideoRenderResult:
    video_path: Path
    thumbnails: dict[str, Path]
    frame_indices: list[int]
    floor_z: float
    floor_reference_lowest_before_shift: float
    floor_reference_lowest_after_shift: float
    lowest_vertex_before_shift: float
    lowest_vertex_after_shift: float
    width: int
    height: int
    video_fps: float
    camera_mode: str
    floor_policy: str


@dataclass(frozen=True)
class FrameOverlayStats:
    source_frame_idx: int
    time_seconds: float
    smpl_fps: float
    yaw_deg: float
    yaw_rate_deg_s: float
    root_height_m: float
    root_vxy_m_s: float
    gravity_projection: tuple[float, float, float]
    floor_gap_m: float


def render_mesh_video(
    *,
    clip: CanonicalClip,
    body_cache: BodyModelCache,
    output_path: str | Path,
    thumbnail_dir: str | Path | None = None,
    thumbnail_prefix: str = "",
    video_fps: float = 50.0,
    max_seconds: float = 10.0,
    width: int = 960,
    height: int = 720,
    floor_policy: str = "first5_min",
    camera_mode: str = "fixed_3quarter",
    batch_size: int = 32,
    label: str = "Input SMPL mesh",
) -> VideoRenderResult:
    if abs(float(video_fps) - 50.0) > 1e-6:
        raise ValueError(f"this preview renderer is currently fixed to 50Hz, got {video_fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if max_seconds <= 0:
        raise ValueError(f"max_seconds must be positive, got {max_seconds}")

    frame_indices = _video_frame_indices(
        frame_count=clip.frame_count,
        target_fps=clip.target_fps,
        video_fps=video_fps,
        max_seconds=max_seconds,
    )
    if not frame_indices:
        raise ValueError(f"no frames selected for video: {clip.path}")

    floor_indices = list(range(min(5, clip.frame_count)))
    floor_output = body_cache.forward_clip_frames(clip, floor_indices, batch_size=batch_size)
    floor_z = _estimate_floor_z(floor_output.vertices, floor_policy)
    floor_reference_lowest = float(floor_output.vertices[:, :, 2].min())

    output = body_cache.forward_clip_frames(clip, frame_indices, batch_size=batch_size)
    vertices = output.vertices.copy()
    joints = output.joints.copy()
    lowest_before = float(vertices[:, :, 2].min())
    vertices[:, :, 2] -= floor_z
    joints[:, :, 2] -= floor_z
    lowest_after = float(vertices[:, :, 2].min())
    overlay_stats = _compute_overlay_stats(
        clip=clip,
        frame_indices=frame_indices,
        shifted_vertices=vertices,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_dir_path = output_path.parent if thumbnail_dir is None else Path(thumbnail_dir)
    thumbnail_dir_path.mkdir(parents=True, exist_ok=True)
    thumbnails = _render_frames_to_video(
        vertices=vertices,
        joints=joints,
        faces=output.faces,
        output_path=output_path,
        thumbnail_dir=thumbnail_dir_path,
        thumbnail_prefix=thumbnail_prefix,
        video_fps=video_fps,
        width=width,
        height=height,
        camera_mode=camera_mode,
        label=label,
        clip=clip,
        frame_indices=frame_indices,
        model_type=output.model_type,
        overlay_stats=overlay_stats,
    )

    return VideoRenderResult(
        video_path=output_path,
        thumbnails=thumbnails,
        frame_indices=frame_indices,
        floor_z=round(floor_z, 6),
        floor_reference_lowest_before_shift=round(floor_reference_lowest, 6),
        floor_reference_lowest_after_shift=round(floor_reference_lowest - floor_z, 6),
        lowest_vertex_before_shift=round(lowest_before, 6),
        lowest_vertex_after_shift=round(lowest_after, 6),
        width=width,
        height=height,
        video_fps=float(video_fps),
        camera_mode=camera_mode,
        floor_policy=floor_policy,
    )


def _render_frames_to_video(
    *,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    thumbnail_dir: Path,
    thumbnail_prefix: str,
    video_fps: float,
    width: int,
    height: int,
    camera_mode: str,
    label: str,
    clip: CanonicalClip,
    frame_indices: list[int],
    model_type: str,
    overlay_stats: list[FrameOverlayStats],
) -> dict[str, Path]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import imageio.v2 as imageio
    import pyrender
    import trimesh

    scene = pyrender.Scene(
        bg_color=np.array([0.94, 0.94, 0.91, 1.0], dtype=np.float32),
        ambient_light=np.array([0.24, 0.24, 0.24, 1.0], dtype=np.float32),
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    ground = pyrender.Mesh.from_trimesh(
        _make_ground_mesh(vertices),
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.45, 0.45, 0.41, 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.95,
        ),
        smooth=False,
    )
    scene.add(ground)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(43.0), aspectRatio=width / height)
    camera_node = scene.add(camera, pose=np.eye(4, dtype=np.float32))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.4)
    light_node = scene.add(light, pose=_look_at(np.array([3.0, -4.0, 5.0]), np.array([0.0, 0.0, 0.8])))

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.05, 0.48, 0.78, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.38,
    )
    mesh_node: Any | None = None

    thumbnail_frames = {
        f"{thumbnail_prefix}thumbnail_first.png": 0,
        f"{thumbnail_prefix}thumbnail_mid.png": len(frame_indices) // 2,
        f"{thumbnail_prefix}thumbnail_last.png": len(frame_indices) - 1,
    }
    thumbnails: dict[str, Path] = {}
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    try:
        with imageio.get_writer(
            output_path,
            fps=float(video_fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for local_idx, source_frame_idx in enumerate(frame_indices):
                if mesh_node is not None:
                    scene.remove_node(mesh_node)
                mesh = trimesh.Trimesh(vertices=vertices[local_idx], faces=faces, process=False)
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
                mesh_node = scene.add(render_mesh)

                target = _camera_target(joints[local_idx])
                eye = _camera_eye(target, vertices, camera_mode)
                scene.set_pose(camera_node, pose=_look_at(eye, target))
                scene.set_pose(light_node, pose=_look_at(eye + np.array([1.0, -1.5, 2.5]), target))

                color, _ = renderer.render(scene, flags=flags)
                frame = _overlay_label(
                    color,
                    label=label,
                    clip=clip,
                    model_type=model_type,
                    stats=overlay_stats[local_idx],
                )
                writer.append_data(frame)
                for name, thumb_idx in thumbnail_frames.items():
                    if local_idx == thumb_idx:
                        thumb_path = thumbnail_dir / name
                        imageio.imwrite(thumb_path, frame)
                        thumbnails[name] = thumb_path
    finally:
        renderer.delete()

    return thumbnails


def _video_frame_indices(
    *,
    frame_count: int,
    target_fps: float,
    video_fps: float,
    max_seconds: float,
) -> list[int]:
    if frame_count <= 0:
        return []
    max_source_frames = max(1, int(round(float(max_seconds) * float(target_fps))))
    end = min(frame_count, max_source_frames)
    step = max(1, int(round(float(target_fps) / float(video_fps))))
    return list(range(0, end, step))


def _estimate_floor_z(vertices: np.ndarray, floor_policy: str) -> float:
    if floor_policy == "first5_min":
        return float(vertices[:, :, 2].min())
    if floor_policy == "first5_percentile":
        return float(np.percentile(vertices[:, :, 2], 1.0))
    raise ValueError(f"unsupported floor_policy: {floor_policy}")


def _make_ground_mesh(vertices: np.ndarray) -> Any:
    import trimesh

    xy = vertices[:, :, :2].reshape(-1, 2)
    center = xy.mean(axis=0)
    extent = xy.max(axis=0) - xy.min(axis=0)
    size = float(max(8.0, extent.max() + 6.0))
    x0, y0 = center - size / 2.0
    x1, y1 = center + size / 2.0
    verts = np.array(
        [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _camera_target(joints: np.ndarray) -> np.ndarray:
    pelvis = joints[0].astype(np.float64)
    return np.array([pelvis[0], pelvis[1], max(0.9, pelvis[2] + 0.35)], dtype=np.float64)


def _camera_eye(target: np.ndarray, all_vertices: np.ndarray, camera_mode: str) -> np.ndarray:
    if camera_mode != "fixed_3quarter":
        raise ValueError(f"unsupported camera_mode: {camera_mode}")
    extent = all_vertices.reshape(-1, 3).ptp(axis=0)
    distance = max(4.2, float(np.linalg.norm(extent[:2])) * 0.42, float(extent[2]) * 2.3)
    direction = np.array([-0.72, -1.0, 0.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eye = target + direction * distance
    eye[2] = max(1.35, target[2] + 0.55)
    return eye


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose.astype(np.float32)


def _overlay_label(
    frame: np.ndarray,
    *,
    label: str,
    clip: CanonicalClip,
    model_type: str,
    stats: FrameOverlayStats,
) -> np.ndarray:
    import cv2

    out = np.asarray(frame[:, :, :3], dtype=np.uint8).copy()
    h, w = out.shape[:2]
    pad = max(12, int(round(w * 0.018)))
    label_height = max(52, int(round(h * 0.075)))
    label_width = min(w - 2 * pad, max(360, int(round(w * 0.50))))
    cv2.rectangle(
        out,
        (pad, pad),
        (pad + label_width, pad + label_height),
        color=(58, 58, 58),
        thickness=-1,
    )
    cv2.putText(
        out,
        label,
        (pad + 18, pad + int(label_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    lines = [
        f"frame {stats.source_frame_idx} | {stats.time_seconds:.2f}s | {stats.smpl_fps:g}Hz",
        f"yaw {stats.yaw_deg:+.1f} deg",
        f"yaw_rate {stats.yaw_rate_deg_s:+.1f} deg/s",
        f"root_height {stats.root_height_m:.3f} m",
        f"root_vxy {stats.root_vxy_m_s:.3f} m/s",
        (
            "gravity "
            f"[{stats.gravity_projection[0]:+0.2f}, "
            f"{stats.gravity_projection[1]:+0.2f}, "
            f"{stats.gravity_projection[2]:+0.2f}]"
        ),
        f"floor_gap {stats.floor_gap_m:+.3f} m",
    ]
    hud_top = pad + label_height + max(8, int(round(h * 0.012)))
    line_gap = max(18, int(round(h * 0.034)))
    hud_height = max(132, int(round(line_gap * len(lines) + 22)))
    hud_width = min(w - 2 * pad, max(430, int(round(w * 0.44))))
    cv2.rectangle(
        out,
        (pad, hud_top),
        (pad + hud_width, hud_top + hud_height),
        color=(66, 66, 66),
        thickness=-1,
    )

    base_y = hud_top + 24
    for idx, text in enumerate(lines):
        color = _overlay_line_color(idx=idx, stats=stats)
        cv2.putText(
            out,
            text,
            (pad + 16, base_y + idx * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    footer = f"{model_type.upper()} | frame {stats.source_frame_idx} | {clip.target_fps:g}Hz"
    cv2.putText(
        out,
        footer,
        (pad, h - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    return out


def _overlay_line_color(*, idx: int, stats: FrameOverlayStats) -> tuple[int, int, int]:
    white = (245, 245, 245)
    yellow = (126, 220, 240)
    red = (96, 96, 240)
    gravity_norm_error = abs(float(np.linalg.norm(np.asarray(stats.gravity_projection))) - 1.0)
    floor_gap_abs = abs(stats.floor_gap_m)
    yaw_rate_abs = abs(stats.yaw_rate_deg_s)

    if idx == 2 and yaw_rate_abs > 540.0:
        return red
    if idx == 2 and yaw_rate_abs > 360.0:
        return yellow
    if idx == 5 and gravity_norm_error > 0.08:
        return red
    if idx == 5 and gravity_norm_error > 0.03:
        return yellow
    if idx == 6 and floor_gap_abs > 0.05:
        return red
    if idx == 6 and floor_gap_abs > 0.02:
        return yellow
    return white


def _compute_overlay_stats(
    *,
    clip: CanonicalClip,
    frame_indices: list[int],
    shifted_vertices: np.ndarray,
) -> list[FrameOverlayStats]:
    from scipy.spatial.transform import Rotation as R

    dt = 1.0 / float(clip.target_fps)
    root_orient = np.asarray(clip.root_orient, dtype=np.float64)
    trans = np.asarray(clip.trans, dtype=np.float64)
    yaw_rad = R.from_rotvec(root_orient).as_euler("ZYX", degrees=False)[:, 0]
    yaw_deg = np.degrees(np.unwrap(yaw_rad))
    yaw_rate_deg_s = np.gradient(yaw_deg, dt)
    root_vxy_m_s = np.linalg.norm(np.gradient(trans[:, :2], dt, axis=0), axis=1)
    gravity = gravity_projection_from_root_orient(clip.root_orient)

    stats: list[FrameOverlayStats] = []
    for local_idx, source_frame_idx in enumerate(frame_indices):
        g = gravity[source_frame_idx]
        stats.append(
            FrameOverlayStats(
                source_frame_idx=int(source_frame_idx),
                time_seconds=float(source_frame_idx) / float(clip.target_fps),
                smpl_fps=float(clip.target_fps),
                yaw_deg=float(yaw_deg[source_frame_idx]),
                yaw_rate_deg_s=float(yaw_rate_deg_s[source_frame_idx]),
                root_height_m=float(trans[source_frame_idx, 2]),
                root_vxy_m_s=float(root_vxy_m_s[source_frame_idx]),
                gravity_projection=(float(g[0]), float(g[1]), float(g[2])),
                floor_gap_m=float(shifted_vertices[local_idx, :, 2].min()),
            )
        )
    return stats
