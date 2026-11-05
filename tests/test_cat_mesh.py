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
    return [(cat, CatMesh(renderer, cat.sprite, cat.points, HOLE, cat.depth, cat.kind == "dog"))
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


def test_teeth_keep_their_size_as_the_mouth_opens(meshes):
    for cat, mesh in meshes:
        if mesh.dog:  # its tongue lies over this column (the teeth are sized the same way)
            continue
        lengths = []
        for amount in (0.07, 0.14):
            image = mesh.render(cat.sprite, mesh.open_jaw(amount), np.eye(3))
            top = (mesh.offset + cat.points[13, :2] * mesh.scale).astype(int)
            x = top[0] + int(0.025 * mesh.face_width * mesh.scale)  # on a front tooth, off the gap
            column = image[top[1]:top[1] + int(0.2 * mesh.face_width * mesh.scale), x].astype(int)
            bright = column.mean(axis=1) > 150  # enamel
            runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], bright, [0]]))))[::2]
            lengths.append(runs.max() if len(runs) else 0)  # the longest bright stretch: a tooth
        assert lengths[0] > 3, cat.name
        assert abs(lengths[0] - lengths[1]) <= max(2, 0.2 * lengths[0]), (cat.name, lengths)


def test_small_mouths_are_enlarged(meshes):
    from catcam.cat_mesh import MOUTH_MAX_GROW, MOUTH_MIN
    from catcam.mouth_fit import snap_mouth

    for cat, mesh in meshes:
        fitted = snap_mouth(cat.sprite, cat.points, mesh.face_width)
        natural = np.linalg.norm(fitted[78, :2] - fitted[308, :2]) / mesh.face_width
        assert mesh.mouth_size >= min(MOUTH_MIN, MOUTH_MAX_GROW * natural) - 1e-6, cat.name
        # Away from the mouth the cat is unchanged; the nose barely moves.
        np.testing.assert_allclose(mesh.rest[[10, 33, 263], :2], cat.points[[10, 33, 263], :2])
        np.testing.assert_allclose(mesh.rest[1, :2], cat.points[1, :2], atol=0.01 * mesh.face_width)


def test_fangs_reach_below_the_front_teeth(meshes):
    for cat, mesh in meshes:
        image = mesh.render(cat.sprite, mesh.open_jaw(0.14), np.eye(3))
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


def _dome(radius=80, size=200):
    yy, xx = np.mgrid[:size, :size] + 0.5
    r = np.hypot(xx - size / 2, yy - size / 2)
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    outline = size / 2 + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return r, outline


def test_photo_depth_puts_the_nose_in_front_and_ignores_the_backdrop():
    from catcam.cat_mesh import HEAD_DEPTH, _photo_depth, _sample

    r, outline = _dome()
    alpha = np.where(r < 80, 255, 0).astype(np.uint8)
    disparity = np.sqrt(np.clip(1 - (r / 80) ** 2, 0, 1)).astype(np.float32)
    disparity[alpha == 0] = 5  # whatever the model saw around the cat
    z = _photo_depth(disparity, alpha, np.array([100.0, 100.0]), outline, face_width=100)
    rim = np.median(_sample(z, outline))
    assert _sample(z, [[100, 100]])[0] - rim == pytest.approx(-HEAD_DEPTH * 100, rel=1e-3)
    # Beyond the outline the depth carries on from just inside it.
    assert abs(z[100, 195] - z[100, 178]) < 0.05 * HEAD_DEPTH * 100


def test_normal_map_of_a_dome_points_outward_toward_the_camera():
    from catcam.cat_mesh import _normal_map

    r, _ = _dome()
    normals = _normal_map(-np.sqrt(np.clip(90 ** 2 - r ** 2, 0, None)))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=2), 1, rtol=1e-6)
    assert (normals[..., 2] < 0).all()
    np.testing.assert_allclose(normals[100, 100], [0, 0, -1], atol=0.02)
    assert normals[100, 40, 0] < -0.5 and normals[100, 160, 0] > 0.5  # left and right
    assert normals[40, 100, 1] < -0.5 and normals[160, 100, 1] > 0.5  # top and bottom


def test_a_depth_map_shapes_the_head(meshes):
    from catcam.cat_mesh import HEAD_DEPTH, OUTLINE_POINTS

    shaped = [(cat, mesh) for cat, mesh in meshes if cat.depth is not None]
    assert shaped
    for cat, mesh in shaped:
        outline = mesh.rest[-OUTLINE_POINTS:, 2]
        nose = mesh.rest[1, 2] - np.median(outline)
        assert nose == pytest.approx(-HEAD_DEPTH * mesh.face_width, rel=1e-3), cat.name
        # The muzzle stands out from the eyes further than on the human-like landmarks.
        eyes = mesh.rest[[33, 263], 2].mean()
        assert eyes - mesh.rest[1, 2] > 1.2 * (cat.points[[33, 263], 2].mean() - cat.points[1, 2]), cat.name


def test_a_depth_mapped_head_is_lit_per_pixel(meshes):
    for cat, mesh in meshes:
        if cat.depth is None:
            continue
        assert mesh.normal_map is not None, cat.name
        grey = cat.sprite.copy()
        grey[..., :3] = 160
        sides = []
        for yaw in (0.3, -0.3):
            rotation = to_matrix(np.array([0, yaw, 0]))
            image = mesh.render(grey, np.zeros((478, 3)), rotation).astype(float)
            ndc = mesh._project(mesh._shape(np.zeros((478, 3))), rotation)
            cheeks = []
            for i in (50, 280):  # image-left and image-right cheek
                x, y = int((ndc[i, 0] + 1) / 2 * OUT[0]), int((1 - ndc[i, 1]) / 2 * OUT[1])
                cheeks.append(image[y - 2:y + 3, x - 2:x + 3].mean())
            sides.append(cheeks[0] - cheeks[1])
        assert abs(sides[0]) > 10 and np.sign(sides[0]) == -np.sign(sides[1]), (cat.name, sides)


def test_breathing_in_lifts_the_head_on_a_swelling_chest(meshes):
    from catcam.cat_mesh import BREATH_RISE

    for cat, mesh in meshes:
        still, full = mesh._shape(np.zeros((478, 3))), mesh._shape(np.zeros((478, 3)), breath=1.0)
        rise = still[:, 1] - full[:, 1]
        np.testing.assert_allclose(rise[:478][mesh.rest[:478, 1] < mesh.rest[152, 1]],
                                   BREATH_RISE * mesh.face_width, err_msg=cat.name)  # the face rides up whole
        assert abs(rise[mesh.rest[:, 1].argmax()]) < 1e-9, cat.name  # the chest's bottom stays
        low = mesh.rest[:, 1] > mesh.rest[152, 1] + 0.05 * mesh.face_width
        if low.any():  # the chest widens
            assert np.ptp(full[low, 0]) > np.ptp(still[low, 0]), cat.name


def test_a_dogs_tongue_hangs_out_when_you_stick_yours_out(meshes):
    def pink_below_the_lip(mesh, image, offsets):
        lip = mesh.offset + (mesh.rest[14, :2] + offsets[14, :2] + [0.03 * mesh.face_width, 0]) * mesh.scale
        x, y = lip.astype(int)  # beside the groove down the tongue's middle
        below = image[y + 3:y + int(0.1 * mesh.face_width * mesh.scale), x - 3:x + 4].astype(int)
        return ((below[..., 2] > 170) & (below[..., 2] - below[..., 1] > 60)).mean()

    dogs = [(cat, mesh) for cat, mesh in meshes if mesh.dog]
    assert [cat.name for cat, _ in dogs] == ["Goldie"]
    for cat, mesh in dogs:
        closed, wide = np.zeros((478, 3)), mesh.open_jaw(0.14)
        for offsets, out, pink in ((closed, 0, False), (wide, 0, False), (closed, 1, True), (wide, 1, True)):
            image = mesh.render(cat.sprite, offsets, np.eye(3), tongue_out=out)
            share = pink_below_the_lip(mesh, image, offsets)
            assert share > 0.8 if pink else share < 0.05, (cat.name, out)
        turned = to_matrix(np.array([0, 0.3, 0]))
        image = mesh.render(cat.sprite, wide, turned, tongue_out=1)
        assert _holes_in_face(mesh, image, wide, turned) == 0, cat.name


def test_mouth_takes_on_the_lip_colour(meshes):
    for _, mesh in meshes:
        assert mesh.lip_tint.shape == (3,) and abs(mesh.lip_tint.mean() - 1) < 1e-6
    reddest = max(meshes, key=lambda cm: cm[1].lip_tint[0])[1]
    palest = min(meshes, key=lambda cm: cm[1].lip_tint[0])[1]
    assert reddest.lip_tint[0] > palest.lip_tint[0] + 0.1  # the cats' lips really differ
