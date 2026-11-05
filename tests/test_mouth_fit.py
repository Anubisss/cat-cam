import cv2
import numpy as np

from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP, _creases, snap_mouth, trace


def test_traces_a_dark_crease():
    sprite = np.full((400, 400, 4), 200, np.uint8)
    xs = np.arange(400)
    crease = (200 + 30 * np.cos((xs - 200) / 60)).astype(int)  # an arch, lowest at the sides
    for x, y in zip(xs, crease):
        cv2.circle(sprite, (int(x), int(y)), 2, (60, 60, 60, 255), -1)
    path_x, path_y = trace(_creases(sprite, 400), np.array([200.0, 220.0]), 400)
    assert np.abs(path_y - crease[path_x]).max() <= 3


def test_lips_snap_to_the_photos_crease_only_when_clearly_off(real_cats):
    for cat in real_cats:
        width = np.ptp(cat.points[:, 0])
        snapped = snap_mouth(cat.sprite, cat.points, width)
        if cat.name == "Mate":  # fitted past his lips onto the whisker pads
            assert np.ptp(snapped[[78, 308], 0]) < np.ptp(cat.points[[78, 308], 0]) - 0.1 * width
            np.testing.assert_allclose(snapped[[10, 33, 263]], cat.points[[10, 33, 263]])
            continue
        if cat.name == "Goldie":  # fitted onto her lower lip; left of it a whisker leads the crease off
            assert (snapped[13, 1] - cat.points[13, 1]) / width < -0.01
            centre = snapped[[78, 308], 0].mean()
            assert abs(centre - cat.points[[78, 308], 0].mean()) < 0.01 * width
            np.testing.assert_allclose(snapped[[10, 33, 263]], cat.points[[10, 33, 263]])
            continue
        if cat.name != "Balint":
            np.testing.assert_array_equal(snapped, cat.points, err_msg=cat.name)
            continue
        # Fitted onto his pouty lower lip: the lips go up onto the crease
        # under his whisker pads, and out to where it turns.
        assert (snapped[13, 1] - cat.points[13, 1]) / width < -0.03
        assert np.ptp(snapped[[78, 308], 0]) - np.ptp(cat.points[[78, 308], 0]) > 0.1 * width
        np.testing.assert_array_equal(snapped[UPPER_INNER_LIP, :2], snapped[LOWER_INNER_LIP, :2])
        # The rest of the face stays (the nose tip barely moves).
        np.testing.assert_allclose(snapped[[10, 33, 263]], cat.points[[10, 33, 263]])
        np.testing.assert_allclose(snapped[1], cat.points[1], atol=0.005 * width)
