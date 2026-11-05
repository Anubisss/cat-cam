import cv2
import numpy as np
import pytest

from catcam import cats, head3d
from catcam.cat_mesh import CHIN_LINE, TOP_MARGIN, CatMesh, Renderer
from catcam.head_pose import to_matrix

pytestmark = pytest.mark.gl

OUT = (320, 180)
BG = (200, 200, 200)


def _box_head(folder, name, points, depth=0.02):
    """A shallow box head, front face (+z) red and the rest green (BGR), laid over
    the landmarks' box in the portrait: model x, y in [-0.5, 0.5] span it."""
    (folder / name).mkdir()
    v = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-depth, depth)], "f4")
    quads = [(1, 3, 7, 5), (0, 4, 6, 2), (0, 1, 5, 4), (2, 6, 7, 3), (4, 5, 7, 6), (0, 2, 3, 1)]
    faces = np.array([f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))], "i4")
    uv = np.where(v[:, 2:] > 0, [0.25, 0.5], [0.75, 0.5]).astype("f4")
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    scale = (hi - lo).max()
    centre = (lo + hi) / 2
    photo_px = centre + scale * v[:, :2] * (1, -1)
    landmarks = np.column_stack([(points[:, :2] - centre) / scale * (1, -1), np.full(len(points), depth)])
    np.savez(folder / name / "head3d.npz", vertices=v, normals=v / np.linalg.norm(v, axis=1, keepdims=True),
             uv=uv, faces=faces, photo_px=photo_px.astype("f4"),
             landmarks=landmarks.astype("f4"), cam_rot=np.eye(3, dtype="f4"), cam_scale=np.float32(scale),
             eye_index=np.zeros(0, "i4"), eye_px=np.zeros((0, 2), "f4"), eye_weight=np.zeros(0, "f4"))
    texture = np.zeros((8, 8, 3), np.uint8)
    texture[:, :4] = (0, 0, 255)
    texture[:, 4:] = (0, 255, 0)
    cv2.imwrite(str(folder / name / "head3d.png"), texture)


@pytest.fixture(scope="module")
def woofgang():
    return cats.load("Woofgang")


@pytest.fixture
def box(tmp_path, monkeypatch, woofgang):
    monkeypatch.setattr(head3d, "ASSETS", tmp_path)
    _box_head(tmp_path, "Box", woofgang.points)
    renderer = Renderer(OUT)
    flat = CatMesh(renderer, woofgang.sprite, woofgang.points, BG)
    return head3d.Head3D(renderer, "Box", flat, woofgang.sprite, woofgang.points, BG)


def test_has_head(tmp_path, monkeypatch, woofgang):
    monkeypatch.setattr(head3d, "ASSETS", tmp_path)
    assert not head3d.has_head("Box")
    _box_head(tmp_path, "Box", woofgang.points)
    assert head3d.has_head("Box")


def test_framed_like_the_photo_cats(box, woofgang):
    frame = box.render(woofgang.sprite, np.zeros((478, 3)), np.eye(3))
    rows = np.nonzero((frame != BG).any(axis=(1, 2)))[0]
    assert rows[0] == pytest.approx(TOP_MARGIN * OUT[1], abs=3)
    assert rows[-1] == pytest.approx(CHIN_LINE * OUT[1], abs=3)


def test_front_faces_camera_in_bgr_and_turns(box, woofgang):
    left = OUT[1] // 3, OUT[0] // 3  # off the drawn mouth
    rest = box.render(woofgang.sprite, np.zeros((478, 3)), np.eye(3))
    b, g, r = rest[left].astype(int)
    assert r > 200 and g < 60  # red front, BGR like the rest of the app
    turned = box.render(woofgang.sprite, np.zeros((478, 3)), to_matrix(np.array([0.0, np.pi / 2, 0.0])))
    b, g, r = turned[OUT[1] // 3, OUT[0] // 2].astype(int)
    assert g > 2 * r and g > 2 * b  # side on: the (shaded) green side


def test_vertices_follow_the_landmarks(box, woofgang):
    moves = np.zeros((478, 3))
    moves[:, 1] = 50.0  # every landmark 50 px down the portrait
    moved = box._shape(moves, (0.0, 0.0)) - box.rest
    weight = np.asarray(box.follow.sum(axis=1)).ravel()
    scale = np.ptp(woofgang.points[:, :2], axis=0).max()
    # Down the portrait is down the model, at the portrait's scale, as much as each vertex follows.
    assert np.allclose(moved[:, 1], -50.0 / scale * weight, atol=1e-6)
    assert np.allclose(moved[:, [0, 2]], 0, atol=1e-6)
    assert weight.max() > 0.5
