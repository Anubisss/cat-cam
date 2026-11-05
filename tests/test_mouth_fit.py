import cv2
import numpy as np

from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP, _creases, trace


def test_traces_a_dark_crease():
    sprite = np.full((400, 400, 4), 200, np.uint8)
    xs = np.arange(400)
    crease = (200 + 30 * np.cos((xs - 200) / 60)).astype(int)  # an arch, lowest at the sides
    for x, y in zip(xs, crease):
        cv2.circle(sprite, (int(x), int(y)), 2, (60, 60, 60, 255), -1)
    path_x, path_y = trace(_creases(sprite, 400), np.array([200.0, 220.0]), 400)
    assert np.abs(path_y - crease[path_x]).max() <= 3


def test_hand_placed_lips_lie_on_their_line(real_cats):
    from catcam.paths import PORTRAITS

    cat = next(c for c in real_cats if c.name == "Woofgang")
    assert cat.mouth_set
    line = np.loadtxt(PORTRAITS / "WoofgangMouth.txt", delimiter=",")
    scale = max(cat.sprite.shape[:2]) / max(cv2.imread(str(PORTRAITS / "Woofgang.png")).shape[:2])
    line = line * scale
    np.testing.assert_array_equal(cat.points[UPPER_INNER_LIP, :2], cat.points[LOWER_INNER_LIP, :2])
    lips = cat.points[UPPER_INNER_LIP, :2]
    np.testing.assert_allclose(lips[:, 1], np.interp(lips[:, 0], line[:, 0], line[:, 1]), atol=1)
    # What moves with your lower lip and chin stays below the line.
    for i in (17, 152):
        assert cat.points[i, 1] > np.interp(cat.points[i, 0], line[:, 0], line[:, 1]), i
