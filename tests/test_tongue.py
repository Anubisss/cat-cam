import cv2
import numpy as np

from catcam import tongue
from catcam.mouth_fit import LOWER_INNER_LIP, OUTER_LIPS, UPPER_INNER_LIP

FACE = 700.0
MOUTH = 0.36 * FACE
UPPER = np.array([500.0, 600.0, -100.0])


def _shape(gap, out):
    return tongue.shape(UPPER, UPPER + [0, gap * FACE, 0], MOUTH, FACE, out)


def _tip(gap, out):
    """How far (face widths) below the lower lip the tongue's tip is."""
    points, _ = _shape(gap, out)
    return (points[:, 1].max() - (UPPER[1] + gap * FACE)) / FACE


def test_no_tongue_in_a_closed_or_barely_open_mouth():
    assert _shape(0.0, 0.0) is None
    assert _shape(0.04, 0.0) is None


def test_the_tongue_lies_inside_an_open_mouth():
    points, _ = _shape(0.13, 0.0)
    assert np.isclose(_tip(0.13, 0.0), -tongue.INSIDE)
    assert points[:, 1].min() >= UPPER[1] + tongue.INCISORS * MOUTH - 1e-9  # behind the upper front teeth
    assert (points[:, 2] < UPPER[2]).all()  # in front of the lips


def test_the_tongue_slides_out_with_yours():
    tips = [_tip(0.13, out) for out in (0.3, 0.6, 1.0)]
    assert tips[0] < tips[1] < tips[2]
    assert np.isclose(tips[2], tongue.HANG)


def test_the_tongue_comes_out_of_a_nearly_closed_mouth():
    points, _ = _shape(0.01, 1.0)
    root = points[:, 1].min()
    assert UPPER[1] <= root <= UPPER[1] + 0.01 * FACE  # between the lips
    assert np.isclose(_tip(0.01, 1.0), tongue.HANG)


def test_the_tongue_is_wider_at_its_tip_and_centred():
    points, inputs = _shape(0.13, 1.0)
    rows = points.reshape(tongue.ROWS, 3, 3)
    widths = rows[:, 2, 0] - rows[:, 0, 0]
    assert widths[-1] > widths[0]
    np.testing.assert_allclose(rows[:, 1, 0], UPPER[0])
    assert inputs.shape == (3 * tongue.ROWS, 4)
    np.testing.assert_allclose(inputs.reshape(tongue.ROWS, 3, 4)[-1, :, 1], 0)  # the tip
    assert len(np.unique(tongue.triangles())) == 3 * tongue.ROWS


LIPS = (150, 120, 200)  # BGR, pinkish red
CENTRE = np.array([320.0, 240.0])


def _webcam(opening, inside):
    """A webcam image (BGR) of a 300 px wide face whose lips are `opening` px
    apart, the gap filled with `inside`, and its landmarks."""
    face = np.tile(CENTRE, (478, 1))
    face[234], face[454] = CENTRE - (150, 0), CENTRE + (150, 0)  # the cheeks
    x = np.linspace(-1, 1, 11)
    arc = np.sqrt(1 - x**2)
    face[UPPER_INNER_LIP] = CENTRE + np.c_[45 * x, -opening / 2 * arc]
    face[LOWER_INNER_LIP] = CENTRE + np.c_[45 * x, opening / 2 * arc]
    upper_outer = CENTRE + np.c_[60 * x, -(opening / 2 + 15) * arc]
    lower_outer = (CENTRE + np.c_[60 * x, (opening / 2 + 18) * arc])[-2:0:-1]
    face[OUTER_LIPS] = np.vstack([upper_outer, lower_outer])
    frame = np.full((480, 640, 3), (120, 150, 190), np.uint8)  # skin
    cv2.fillPoly(frame, [face[OUTER_LIPS].astype(np.int32)], LIPS)
    opening = np.vstack([face[UPPER_INNER_LIP], face[LOWER_INNER_LIP][::-1]])
    cv2.fillPoly(frame, [opening.astype(np.int32)], inside)
    return frame, face


def test_your_tongue_fills_your_open_mouth():
    assert tongue.your_tongue(*_webcam(40, (140, 110, 210))) == 1.0  # a tongue, a bit lighter than the lips


def test_your_dark_mouth_or_teeth_are_no_tongue():
    assert tongue.your_tongue(*_webcam(40, (25, 20, 30))) == 0.0  # dark inside
    assert tongue.your_tongue(*_webcam(40, (235, 235, 235))) == 0.0  # teeth


def test_closed_lips_show_no_tongue():
    assert tongue.your_tongue(*_webcam(2, (140, 110, 210))) == 0.0
