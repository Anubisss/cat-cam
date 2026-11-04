import numpy as np
import pytest

from catcam.app import BLINK_RANGE, Mouse, blinks, preview, ramp
from catcam.picker import SIZE, Picker


def test_ramp():
    assert ramp(0.0, 0.2, 0.6) == 0.0
    assert ramp(0.4, 0.2, 0.6) == pytest.approx(0.5)
    assert ramp(1.0, 0.2, 0.6) == 1.0


def test_blinks_map_eyes_to_image_sides():
    lo, hi = BLINK_RANGE
    # MediaPipe's "right" eye is the person's right, which is the image's left.
    left_image, right_image = blinks({"eyeBlinkRight": hi, "eyeBlinkLeft": lo})
    assert left_image == 1.0 and right_image == 0.0


def test_mouse_click_is_taken_once():
    sprite = np.full((20, 20, 4), 255, np.uint8)
    picker = Picker([("A", sprite, np.zeros(3, np.uint8)), ("B", sprite, np.zeros(3, np.uint8))])
    mouse = Mouse(picker)
    x, y, w, h = picker.cards[1]
    mouse(0, x + 5, y + 5, 0, None)  # move
    assert picker.hover == 1
    mouse(4, x + 5, y + 5, 0, None)  # left button up
    assert mouse.take_click() == 1
    assert mouse.take_click() is None


def test_preview_is_mirrored_and_sized(real_cats):
    from types import SimpleNamespace

    out = np.zeros((720, 1280, 3), np.uint8)
    out[:, :100] = 255  # a white band on the left...
    view = preview(out, SimpleNamespace(name="Test"), looking=False)
    assert view.shape == (SIZE[1], SIZE[0], 3)
    assert view[100, -10].min() > 200 and view[100, 10].max() < 50  # ...shows on the right
