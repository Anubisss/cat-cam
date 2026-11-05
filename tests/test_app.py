import cv2
import numpy as np
import pytest

from catcam.app import BLINK_RANGE, ENTER, ESC, Controls, Mouse, blinks, preview, ramp, window_closed
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


def _controls(n=3):
    sprite = np.full((20, 20, 4), 255, np.uint8)
    picker = Picker([(f"C{i}", sprite, np.zeros(3, np.uint8)) for i in range(n)])
    return Controls(picker, Mouse(picker))


class _State:
    resets = 0

    def reset_neutral(self):
        self.resets += 1


def test_controls_start_in_the_picker_and_pick_by_number():
    controls, state = _controls(), _State()
    assert controls.picking
    controls.handle(ESC, state)  # nothing picked yet: Esc does not leave
    assert controls.picking
    controls.handle(ord("2"), state)
    assert not controls.picking and controls.current == 1


def test_controls_change_cat_and_go_back():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    controls.handle(ord("c"), state)
    assert controls.picking
    controls.handle(ESC, state)
    assert not controls.picking and controls.current == 0
    controls.handle(ord("3"), state)  # number keys work in the call too
    assert controls.current == 2


def test_controls_click_and_enter():
    controls, state = _controls(), _State()
    x, y, w, h = controls.picker.cards[2]
    controls.mouse(cv2.EVENT_MOUSEMOVE, x + 1, y + 1, 0, None)
    controls.handle(ENTER, state)
    assert controls.current == 2
    controls.handle(ord("c"), state)
    x, y, w, h = controls.picker.cards[1]
    controls.mouse(cv2.EVENT_LBUTTONUP, x + 1, y + 1, 0, None)
    controls.handle(-1 & 0xFF, state)
    assert controls.current == 1 and not controls.picking


def test_controls_neutral_and_quit():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    controls.handle(ord("n"), state)
    assert state.resets == 1
    controls.handle(ord("q"), state)
    assert controls.quit


def test_window_closed_when_opencv_says_it_is_gone(monkeypatch):
    def gone(*args):
        raise cv2.error("NULL window")

    monkeypatch.setattr(cv2, "getWindowProperty", gone)
    assert window_closed()
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    assert not window_closed()


def test_a_click_during_the_call_does_not_pick_later():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    x, y, w, h = controls.picker.cards[2]
    controls.mouse(cv2.EVENT_LBUTTONUP, x + 1, y + 1, 0, None)  # e.g. focusing the window
    controls.handle(-1 & 0xFF, state)
    controls.handle(ord("c"), state)
    controls.handle(-1 & 0xFF, state)
    assert controls.picking and controls.current == 0


def test_closed_eyes_count_as_a_blink_even_with_a_weak_score():
    from catcam.cat_face import EYES

    rng = np.random.default_rng(1)
    neutral = rng.normal(size=(478, 3)) * 5 + 300
    for upper, lower in EYES:  # open eyes: lower lid 12 px below the upper
        neutral[upper, 0] = neutral[lower, 0] = np.linspace(0, 60, len(upper)) + neutral[upper[0], 0]
        neutral[lower, 1] = neutral[upper, 1] + 12
        neutral[[upper[0], upper[-1]], 1] = neutral[upper[0], 1]
    shut = neutral.copy()
    for upper, lower in EYES:
        shut[upper[1:-1], 1] = shut[lower[1:-1], 1] - 1
    weak = {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1}
    np.testing.assert_allclose(blinks(weak, neutral, neutral), 0)
    assert blinks(weak, shut, neutral).min() == 1.0


@pytest.mark.gl
def test_check_draws_every_cat_without_a_camera(capsys):
    from catcam.app import run

    run(["--check"])
    assert capsys.readouterr().out.strip().endswith("cats")
