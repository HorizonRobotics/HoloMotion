"""Thin-face projection regression on CPU and available CUDA devices."""

import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import collision_gjk

from holoboxing.evaluation.runtime.collision_precision import (
    install,
    project_origin_plane,
)


@wp.kernel
def project(
    vertices: wp.array2d(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
    status: wp.array(dtype=int),
):
    index = wp.tid()
    point, code = collision_gjk._project_origin_plane(
        vertices[index, 0], vertices[index, 1], vertices[index, 2]
    )
    result[index] = point
    status[index] = code


class CollisionPrecisionTest(unittest.TestCase):
    def test_thin_face_and_degenerate_face_preserve_the_float32_interface(
        self,
    ):
        install()
        self.assertIs(
            collision_gjk._project_origin_plane, project_origin_plane
        )
        vertices = np.asarray(
            [
                [
                    [
                        0.2637648582458496,
                        0.39773109555244446,
                        0.3818451762199402,
                    ],
                    [
                        0.16947799921035767,
                        0.42917612195014954,
                        0.5601202845573425,
                    ],
                    [
                        0.221335768699646,
                        0.41188135743141174,
                        0.46206897497177124,
                    ],
                ],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            ],
            dtype=np.float32,
        )
        expected = []
        for a, b, c in vertices.astype(np.float64):
            normal = np.cross(b - a, c - a)
            squared = normal @ normal
            expected.append(
                normal * ((normal @ a) / squared) if squared else np.zeros(3)
            )
        # This thin face produces a 0.69 m float32 projection error.
        a, b, c = vertices[0]
        normal = np.cross(c - b, b - a)
        legacy = normal * ((normal @ b) / (normal @ normal))
        self.assertGreater(np.linalg.norm(legacy - expected[0]), 0.6)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                output = wp.zeros(3, dtype=wp.vec3, device=device)
                status = wp.zeros(3, dtype=int, device=device)
                wp.launch(
                    project,
                    dim=3,
                    inputs=[wp.array(vertices, dtype=wp.vec3, device=device)],
                    outputs=[output, status],
                    device=device,
                )
                self.assertEqual(output.numpy().dtype, np.float32)
                np.testing.assert_allclose(
                    output.numpy(), expected, atol=3e-8, rtol=0
                )
                np.testing.assert_array_equal(status.numpy(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
