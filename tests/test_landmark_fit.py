import numpy as np
import pytest

from catcam.landmark_fit import head_bbox, mirror_points, plausible, symmetrize
from catcam.paths import FACE_MIRROR

# (left, right) MediaPipe landmark pairs: eye corners, mouth corners, face sides.
PAIRS = [(33, 263), (133, 362), (61, 291), (78, 308), (234, 454)]
MIDLINE = [1, 4, 13, 14, 152]


@pytest.fixture(scope="module")
def mirror():
    return np.load(FACE_MIRROR)


def test_mirror_map_swaps_sides(mirror):
    assert sorted(mirror) == list(range(478))
    np.testing.assert_array_equal(mirror[mirror], np.arange(478))
    for left, right in PAIRS:
        assert mirror[left] == right
    for i in MIDLINE:
        assert mirror[i] == i


def test_mirroring_twice_is_identity(mirror, rng):
    points = rng.normal(size=(478, 3))
    np.testing.assert_allclose(mirror_points(mirror_points(points, mirror, 5.0), mirror, 5.0), points)


def test_symmetrize_makes_pairs_mirror_images(mirror, rng):
    sym = symmetrize(rng.normal(size=(478, 3)) * 100, mirror)
    axis = sym[:, 0].mean()
    for left, right in PAIRS:
        assert np.isclose(sym[left, 0] - axis, axis - sym[right, 0])
        assert np.isclose(sym[left, 1], sym[right, 1])


def test_head_bbox():
    alpha = np.zeros((100, 200), np.uint8)
    alpha[10:60, 30:150] = 255
    np.testing.assert_array_equal(head_bbox(alpha), [30, 10, 149, 59])


def test_plausible_rejects_small_or_offset_faces():
    bbox = np.array([0.0, 0.0, 1000.0, 1000.0])
    face = np.zeros((478, 3))
    face[:, 0] = np.linspace(200, 800, 478)
    face[:, 1] = 500
    face[1, :2] = (500, 500)
    assert plausible(face, bbox)
    small = face.copy()
    small[:, 0] = np.linspace(450, 550, 478)
    small[1, 0] = 500
    assert not plausible(small, bbox)
    off_centre = face.copy()
    off_centre[1, 0] = 800
    assert not plausible(off_centre, bbox)
