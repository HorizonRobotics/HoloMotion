"""Shared thin-face projection fix for MuJoCo-Warp 3.5.0 collision kernels."""

from importlib.metadata import version

import warp as wp

COLLISION_PRECISION_VERSION = "epa-plane-fp64-v1"


@wp.func
def project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):
    # Float32 cross products can misproject a nearly degenerate EPA face.
    # Promote before subtraction, and return the original float32 interface.
    a, b, c = wp.vec3d(v1), wp.vec3d(v2), wp.vec3d(v3)
    normal = wp.cross(b - a, c - a)
    squared_norm = wp.dot(normal, normal)
    if squared_norm == wp.float64(0.0):
        return wp.vec3(0.0), int(1)
    return wp.vec3(normal * (wp.dot(normal, a) / squared_norm)), int(0)


def install():
    """Install before any collision kernel is compiled, in both entrypoints."""
    from mujoco_warp._src import collision_gjk

    if version("mujoco-warp") != "3.5.0":
        raise RuntimeError(
            "Collision projection fix requires mujoco-warp 3.5.0"
        )
    collision_gjk._project_origin_plane = project_origin_plane
