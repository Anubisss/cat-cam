"""Renders the real cats and checks the mesh has no holes or folds."""

import cv2
import numpy as np
import pytest

from catcam.cat_mesh import LOWER_INNER_LIP, UPPER_INNER_LIP, CatMesh, Renderer
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
    return [(cat, CatMesh(renderer, cat.sprite, cat.points, HOLE, snap=not cat.mouth_set))
            for cat in real_cats]


def _holes_in_face(mesh, image, offsets, rotation):
    ndc = mesh._project(mesh._shape(offsets), rotation)
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


def test_eyelid_motion_does_not_bend_the_mesh(meshes):
    """Your eyelids travel further than the cats' eyes are tall; the eye
    regions move rigidly with their corners instead (the lids close in the
    texture)."""
    from catcam.cat_face import EYES

    for cat, mesh in meshes:
        offsets = np.zeros((478, 3))
        for upper, _ in EYES:
            offsets[upper[1:-1], 1] = 0.1 * mesh.face_width  # upper lids pulled far down
        brow = 105
        offsets[brow, 1] = -0.02 * mesh.face_width
        steady = mesh._steady_eyes(offsets)
        for upper, lower in EYES:
            lids = steady[upper + lower]
            np.testing.assert_allclose(lids - lids[0], 0, atol=1e-9)  # the eye doesn't deform
            assert np.abs(lids[0]).max() < 0.01 * mesh.face_width, cat.name  # ...nor follow the lids
        assert steady[brow, 1] < -0.005 * mesh.face_width, cat.name  # brows still move


def _open(mesh, amount):
    from catcam.cat_mesh import UPPER_INNER_LIP

    up = mesh.rest[UPPER_INNER_LIP]
    order = np.argsort(up[:, 0])
    line = np.interp(mesh.rest[:478, 0], up[order, 0], up[order, 1])
    weight = np.clip((mesh.rest[:478, 1] - line) / (0.01 * mesh.face_width), 0, 1)
    weight[LOWER_INNER_LIP[1:-1]] = 1
    weight[UPPER_INNER_LIP] = 0
    offsets = np.zeros((478, 3))
    offsets[:, 1] = weight * amount * mesh.face_width
    return offsets


def _lip_y(mesh, lip, x: float, drop: float = 0.0) -> int:
    """Image row of a lip contour of the mesh at image column x, dropped by drop (sprite px)."""
    points = mesh.rest[lip][np.argsort(mesh.rest[lip, 0])]
    sprite_x = (x - mesh.offset[0]) / mesh.scale
    return int(mesh.offset[1] + mesh.scale * (np.interp(sprite_x, points[:, 0], points[:, 1]) + drop))


def _tooth(mesh, image, x: int, amount: float) -> int:
    """Length (image px) of the longest tooth in column x, between the lips: fur, pale on a
    white cat, stays out, and enamel is pale and grey, unlike a pink tongue."""
    upper = _lip_y(mesh, UPPER_INNER_LIP, x)
    lower = _lip_y(mesh, LOWER_INNER_LIP, x, amount * mesh.face_width)
    pixels = image[upper:max(lower, upper + 1), x].astype(int)
    bright = (pixels.min(axis=1) > 100) & (np.ptp(pixels, axis=1) < 60)
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], bright, [0]]))))[::2]
    return runs.max() if len(runs) else 0


def test_teeth_keep_their_size_as_the_mouth_opens(meshes):
    measured = 0
    for cat, mesh in meshes:
        lengths = []
        for amount in (0.07, 0.14):
            image = mesh.render(cat.sprite, _open(mesh, amount), np.eye(3))
            middle = int(mesh.offset[0] + cat.points[13, 0] * mesh.scale)
            # Two spots on the front teeth, off the gap: a wide or a slanting mouth (a frown)
            # can put either between two teeth.
            spots = (0.025 * mesh.face_width, 0.08 * mesh.mouth_size * mesh.face_width)
            lengths.append(max(_tooth(mesh, image, middle + int(f * mesh.scale), amount) for f in spots))
        if lengths[0] <= 3:  # a small mouth, barely open: too little tooth shows there to measure
            continue
        measured += 1
        assert abs(lengths[0] - lengths[1]) <= max(2, 0.2 * lengths[0]), (cat.name, lengths)
    assert measured >= len(meshes) / 2


def test_small_mouths_are_enlarged(meshes):
    from catcam.cat_mesh import MOUTH_MAX_GROW, MOUTH_MIN
    from catcam.mouth_fit import snap_mouth

    for cat, mesh in meshes:
        fitted = snap_mouth(cat.sprite, cat.points, mesh.face_width)
        natural = np.linalg.norm(fitted[78, :2] - fitted[308, :2]) / mesh.face_width
        assert mesh.mouth_size >= min(MOUTH_MIN, MOUTH_MAX_GROW * natural) - 1e-6, cat.name
        # Away from the mouth the cat is unchanged; the nose barely moves.
        np.testing.assert_allclose(mesh.rest[[10, 33, 263], :2], cat.points[[10, 33, 263], :2])
        np.testing.assert_allclose(mesh.rest[1, :2], cat.points[1, :2], atol=0.015 * mesh.face_width)


def test_fangs_reach_below_the_front_teeth(meshes):
    for cat, mesh in meshes:
        image = mesh.render(cat.sprite, _open(mesh, 0.14), np.eye(3))
        # From just below the upper lip (pale lips would count as enamel).
        top = (mesh.offset + (mesh.rest[13, :2] + [0, 0.01 * mesh.face_width]) * mesh.scale).astype(int)
        depth = int(0.12 * mesh.face_width * mesh.scale)  # within the opening
        mouth = mesh.mouth_size * mesh.face_width * mesh.scale
        lengths = []
        for f in [0.08, *np.linspace(0.18, 0.3, 13)]:  # a front tooth, then around the fang
            column = image[top[1]:top[1] + depth, top[0] + int(f * mouth)].astype(int)
            bright = column.mean(axis=1) > 150  # enamel
            runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], bright, [0]]))))[::2]
            lengths.append(runs.max() if len(runs) else 0)  # the longest bright stretch: a tooth
        front, fang = lengths[0], max(lengths[1:])
        assert fang > 1.8 * front, (cat.name, front, fang)


def test_mouth_takes_on_the_lip_colour(meshes):
    for cat, mesh in meshes:
        assert mesh.lip_tint.shape == (3,) and abs(mesh.lip_tint.mean() - 1) < 1e-6
        assert np.ptp(mesh.lip_tint) > 0.1, cat.name  # a real colour, not neutral grey
