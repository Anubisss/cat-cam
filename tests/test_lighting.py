import numpy as np
import pytest

from catcam.head_pose import to_matrix
from catcam.lighting import SHADE_RANGE, shading, vertex_normals


def test_normals_of_a_sphere_point_outward_toward_the_camera(rng):
    direction = rng.normal(size=(400, 3))
    direction[:, 2] = -np.abs(direction[:, 2])  # the front half
    sphere = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    normals = vertex_normals(sphere * 100, face_width=100)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    assert (normals[:, 2] <= 0).all()
    front = sphere[:, 2] < -0.5  # away from the patch's rim
    assert (np.einsum("ij,ij->i", normals[front], sphere[front]) > 0.95).all()


def test_no_change_at_rest(rng):
    normals = vertex_normals(rng.normal(size=(50, 3)) * 10, face_width=10)
    np.testing.assert_allclose(shading(normals, np.eye(3)), 1)


@pytest.mark.gl
def test_the_side_turning_away_darkens(real_cats):
    from catcam.cat_mesh import CatMesh, Renderer

    renderer = Renderer((320, 180))
    for cat in real_cats:
        mesh = CatMesh(renderer, cat.sprite, cat.points, (0, 0, 0), snap=not cat.mouth_set)
        left = mesh.rest[:, 0] < mesh.pivot[0]
        sides = []
        for yaw in (0.3, -0.3):
            shade = shading(mesh.normals, to_matrix(np.array([0, yaw, 0])))
            assert SHADE_RANGE[0] <= shade.min() and shade.max() <= SHADE_RANGE[1]
            sides.append(shade[left].mean() - shade[~left].mean())
        assert abs(sides[0]) > 0.05 and np.sign(sides[0]) == -np.sign(sides[1]), (cat.name, sides)
