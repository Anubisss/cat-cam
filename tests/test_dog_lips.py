import cv2
import numpy as np
import pytest

from catcam import cats, dog_lips
from catcam.mouth_fit import UPPER_INNER_LIP, snap_mouth


@pytest.fixture(scope="module")
def goldie():
    raw = cv2.imread(str(cats.ASSETS / "Goldie" / "cutout.png"), cv2.IMREAD_UNCHANGED)
    return raw, cats.load("Goldie")


def _pink(bgr, mask):
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    pink = ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    return pink[mask > 0].mean()


def test_the_pink_lips_are_covered_with_fur(goldie):
    raw, dog = goldie
    width = np.ptp(dog.points[:, 0])
    lips = dog_lips.lip_mask(raw[..., :3].copy(), snap_mouth(raw, dog.points, width), width)
    assert _pink(raw[..., :3], lips) > 0.3  # the photo has human lips (the mask reaches past them)...
    assert _pink(dog.sprite[..., :3], lips) < 0.05  # ...the dog has none


def test_only_the_mouth_is_repainted(goldie):
    raw, dog = goldie
    changed = np.abs(dog.sprite.astype(int) - raw.astype(int)).max(axis=2) > 0
    ys, xs = np.nonzero(changed)
    nose, chin = dog.points[1], dog.points[152]
    assert ys.min() > nose[1] - 0.1 * np.ptp(dog.points[:, 0])  # the eyes and above are untouched
    assert ys.max() < chin[1] + 0.05 * np.ptp(dog.points[:, 0])
    np.testing.assert_array_equal(dog.sprite[..., 3], raw[..., 3])


def test_a_dark_lip_line_runs_along_the_seam(goldie):
    raw, dog = goldie
    width = np.ptp(dog.points[:, 0])
    seam = snap_mouth(raw, dog.points, width)[UPPER_INNER_LIP[2:-2], :2].astype(int)
    grey = cv2.cvtColor(np.ascontiguousarray(dog.sprite[..., :3]), cv2.COLOR_BGR2GRAY)
    assert grey[seam[:, 1], seam[:, 0]].max() < 60
