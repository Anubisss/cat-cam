"""Renders the real cats and checks the mesh has no holes or folds."""

import cv2
import numpy as np
import pytest

from catcam.cat_mesh import LOWER_INNER_LIP, CatMesh, Renderer
from catcam.head_pose import to_matrix

pytestmark = pytest.mark.gl
OUT = (640, 360)
HOLE = (255, 0, 255)  # cleared to this, so any gap in the cat shows up
# Brows, cheeks and chin: well inside every cat's face (the face outline itself
# can lie on the edge of the head).
INNER_FACE = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 187, 411, 205, 425, 18, 200, 199]


@pytest.fixture(scope="module")
def renderer():
    return Renderer(OUT)


@pytest.fixture(scope="module")
def meshes(renderer, real_cats):
    return [(cat, CatMesh(renderer, cat.sprite, cat.points, HOLE)) for cat in real_cats]


def _holes_in_face(mesh, image, offsets, rotation):
    ndc = mesh._project(offsets, rotation)
    px = np.column_stack([(ndc[:, 0] + 1) / 2 * OUT[0], (1 - ndc[:, 1]) / 2 * OUT[1]])
    mask = np.zeros(image.shape[:2], np.uint8)
    inner = px[INNER_FACE]
    # Shrunk, so the far cheek of a turned head (a real edge) stays outside.
    inner = inner.mean(axis=0) + 0.8 * (inner - inner.mean(axis=0))
    hull = cv2.convexHull(inner.astype(np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    # Also catches see-through spots, where the hole colour blends in.
    hole = np.linalg.norm(image.astype(float) - HOLE, axis=2) < 100
    return int((hole & (mask > 0)).sum())


POSES = [(0, 0, 0), (0.25, 0, 0), (-0.25, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 0.3)]


@pytest.mark.parametrize("vec", POSES)
def test_face_has_no_holes(meshes, vec):
    rotation = to_matrix(np.array(vec, float))
    for cat, mesh in meshes:
        offsets = np.zeros((478, 3))
        image = mesh.render(cat.sprite, offsets, rotation)
        assert _holes_in_face(mesh, image, offsets, rotation) == 0, cat.name


def test_rest_pose_reproduces_the_photo(meshes):
    for cat, mesh in meshes:
        image = mesh.render(cat.sprite, np.zeros((478, 3)), np.eye(3))
        # Sample the sprite where the nose tip lands on screen.
        nose = mesh.offset + cat.points[1, :2] * mesh.scale
        x, y = nose.astype(int)
        sx, sy = cat.points[1, :2].astype(int)
        diff = np.abs(image[y, x].astype(int) - cat.sprite[sy, sx, :3]).max()
        assert diff < 40, cat.name


def test_open_mouth_shows_the_inside(meshes):
    for cat, mesh in meshes:
        closed = mesh.render(cat.sprite, np.zeros((478, 3)), np.eye(3))
        offsets = np.zeros((478, 3))
        lower = mesh.rest[:478, 1] > mesh.rest[14, 1] - 1e-6  # lower lip, chin
        offsets[lower, 1] = 0.12 * mesh.face_width
        offsets[LOWER_INNER_LIP, 1] = 0.12 * mesh.face_width
        opened = mesh.render(cat.sprite, offsets, np.eye(3))
        lip = mesh.offset + (cat.points[13, :2] + [0, 0.05 * mesh.face_width]) * mesh.scale
        x, y = lip.astype(int)
        window = np.s_[y - 2:y + 3, x - 2:x + 3]
        assert np.abs(opened[window].astype(int) - closed[window]).mean() > 20, cat.name
        assert _holes_in_face(mesh, opened, offsets, np.eye(3)) == 0, cat.name
