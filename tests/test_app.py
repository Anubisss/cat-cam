
import cv2
import numpy as np
import pytest

from catcam import app
from catcam.app import (
    BLINK_RANGE,
    ENTER,
    ESC,
    Controls,
    Mouse,
    Tracker,
    blinks,
    preview,
    ramp,
    toggle_rects,
    window_closed,
)
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
    controls.handle(ord("3"), state)  # in the call, number keys are reactions
    assert controls.current == 0 and controls.take_reaction() == "pray"
    assert controls.take_reaction() is None  # taken once
    controls.handle(ord("c"), state)
    controls.handle(ord("3"), state)  # in the picker, they pick a cat
    assert controls.current == 2 and controls.take_reaction() is None


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
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("OK: ")


def _centre(name, head3d=True):
    x, y, w, h = toggle_rects(head3d)[name]
    return x + w // 2, y + h // 2


def test_tail_switch_toggles_by_click_or_key():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)  # leave the picker
    assert not controls.tail
    controls.mouse(4, *_centre("tail"), 0, None)  # click the switch
    controls.handle(-1, state)
    assert controls.tail
    controls.mouse(4, 500, 300, 0, None)  # a click elsewhere
    controls.handle(-1, state)
    assert controls.tail
    controls.handle(ord("t"), state)
    assert not controls.tail


def test_head3d_switch_toggles_by_click_or_key():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)  # leave the picker
    assert controls.head3d  # on from the start
    controls.mouse(4, *_centre("head3d"), 0, None)
    controls.handle(-1, state)
    assert not controls.head3d and not controls.tail
    controls.handle(ord("d"), state)
    assert controls.head3d


def test_background_switch_toggles_by_click_or_key():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    assert not controls.background
    controls.mouse(4, *_centre("background"), 0, None)
    controls.handle(-1, state)
    assert controls.background
    controls.handle(ord("b"), state)
    assert not controls.background


def test_voice_switch_toggles_by_click_or_key():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    assert controls.voice  # the animal's voice from the start
    controls.mouse(4, *_centre("voice"), 0, None)
    controls.handle(-1, state)
    assert not controls.voice and not controls.background
    controls.handle(ord("v"), state)
    assert controls.voice
    controls.handle(ord("c"), state)  # the switch stays as it is for the next cat
    controls.handle(ord("2"), state)
    assert controls.voice


def test_switch_keys_are_all_different():
    from catcam.app import KEY_HINTS, TOGGLES

    keys = [key for _, _, key in TOGGLES] + [key for key, _ in KEY_HINTS]
    assert len(keys) == len(set(keys))


def test_switch_keys_leave_the_shortcuts_alone():
    from catcam.app import TOGGLES
    from catcam.fx import load

    keys = {key for _, _, key in TOGGLES} | {"c", "n", "q"}
    assert not keys & set(load())


def test_preview_shows_the_voice_switch():
    from types import SimpleNamespace

    out = np.zeros((720, 1280, 3), np.uint8)
    cat = SimpleNamespace(name="Test")
    assert sum(w for _, _, w, _ in toggle_rects(True).values()) < SIZE[0] - 2 * app.MARGIN  # all fit
    x, y, w, h = toggle_rects(False)["voice"]
    area = np.s_[y + 8:y + h - 8, x + 10:x + w - 10]
    off, on = preview(out, cat, False, voice=False)[area], preview(out, cat, False, voice=True)[area]
    assert off.any()
    assert np.abs(on.astype(int) - off).sum() > 0


@pytest.mark.parametrize("head3d", [True, False])
def test_the_controls_keep_to_the_corners_and_never_overlap(monkeypatch, head3d):
    """With the longest name, the widest paw mode, every voice setup note and "Looking
    for your face", nothing covers anything else or leaves the window."""
    from catcam import blackhole
    from catcam.paths import ASSETS

    names = [p.name for p in ASSETS.iterdir() if p.is_dir()]
    longest = max(names, key=lambda name: app.glass.text_width(name, app.NAME, 600, "rounded"))
    states = {"tail": False, "paws": "hands", "background": False, "voice": True} | (
        {"head3d": True} if head3d else {})
    width, height = SIZE
    view = np.zeros((height, width, 3), np.uint8)
    pill, bar, looking = app._switches(view, states), app._capsule(view, longest), app._looking(view)
    assert pill[2] < width * 0.65 and bar[2] < width * 0.5  # small, and the animal has the middle
    for stage in (blackhole.MISSING, blackhole.DOWNLOADING, blackhole.CHECKING, blackhole.OPENED,
                  blackhole.FAILED, blackhole.WEBSITE, blackhole.RESTART):
        chip = blackhole.VoiceSetup(stage).draw(view, width - app.MARGIN, height - app.MARGIN)
        rects = [pill, bar, looking, chip]
        for x, y, w, h in rects:
            assert 0 < x and x + w < width and 0 < y and y + h < height, stage
        for i, (ax, ay, aw, ah) in enumerate(rects):
            for bx, by, bw, bh in rects[i + 1:]:
                apart = ax + aw + 8 <= bx or bx + bw + 8 <= ax or ay + ah + 8 <= by or by + bh + 8 <= ay
                assert apart, (stage, (ax, ay, aw, ah), (bx, by, bw, bh))


def test_no_head3d_switch_without_a_3d_head():
    assert set(toggle_rects(False)) == {"tail", "paws", "background", "voice"}
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    controls.has_head3d = False
    controls.mouse(4, *_centre("background", False), 0, None)
    controls.handle(-1, state)
    assert controls.head3d and controls.background  # the click went to the switch that is there


def test_preview_shows_the_head3d_switch_only_for_cats_with_a_3d_head():
    from types import SimpleNamespace

    out = np.zeros((720, 1280, 3), np.uint8)
    cat = SimpleNamespace(name="Test")
    x, y, w, h = toggle_rects(True)["head3d"]
    area = np.s_[y + 8:y + h - 8, x + 10:x + w - 10]  # inside the pill, off its rim
    off, on = preview(out, cat, False, head3d=False)[area], preview(out, cat, False, head3d=True)[area]
    assert off.any()
    assert np.abs(on.astype(int) - off).sum() > 0
    # Without a 3D head, the background switch takes its place.
    without = preview(out, cat, False)[area]
    assert np.abs(without.astype(int) - off).sum() > 0
    assert toggle_rects(False)["background"][0] == x


def test_clicks_in_the_picker_do_not_toggle_the_tail():
    controls, state = _controls(), _State()
    controls.mouse(4, *_centre("tail"), 0, None)
    controls.handle(-1, state)
    assert not controls.tail


def test_preview_shows_the_tail_switch():
    from types import SimpleNamespace

    out = np.zeros((720, 1280, 3), np.uint8)
    cat = SimpleNamespace(name="Test")
    x, y, w, h = toggle_rects(False)["tail"]
    area = np.s_[y + 8:y + h - 8, x + 10:x + w - 10]
    off, on = preview(out, cat, False, tail=False)[area], preview(out, cat, False, tail=True)[area]
    assert off.any()  # the switch is drawn either way
    assert np.abs(on.astype(int) - off).sum() > 0  # and shows when it's on


@pytest.mark.gl
def test_tail_replaces_one_paw_only_while_on(real_cats):
    from catcam.app import FaceState, load_cats, render
    from catcam.cat_mesh import Renderer
    from catcam.paws import Paws
    from catcam.tail import Tail

    cat = next(c for c in load_cats(Renderer((1280, 720))) if c.name == "Woofgang")
    paws = Paws(1280)
    # An upright right hand showing its back (see catcam.paws.Paws.update for the layout).
    params = np.array([1100, 500, 200, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0])
    paws.slots = {"Right": {"presence": 1.0, "params": params}}
    tail = Tail((1280, 720))
    plain = cat.mesh.render(cat.sprite, np.zeros((478, 3)), np.eye(3))
    corner = np.s_[400:600, 1000:1200]  # where the paw is
    with_paw = render(cat, FaceState(), paws, tail, tail_hand=None)
    without = render(cat, FaceState(), paws, tail, tail_hand="Right")
    assert np.abs(with_paw[corner].astype(int) - plain[corner]).mean() > 10
    assert np.abs(without[corner].astype(int) - plain[corner]).mean() < 1


@pytest.mark.gl
def test_a_background_shows_around_the_animal(real_cats):
    from catcam.app import FaceState, load_cats, render, shown
    from catcam.cat_mesh import Renderer
    from catcam.paws import Paws
    from catcam.tail import Tail

    renderer = Renderer((1280, 720))
    for cat in load_cats(renderer):
        for drawn in (cat, shown(cat, True)):
            green = np.zeros((720, 1280, 3), np.uint8)
            green[:] = (0, 255, 0)
            plain = render(drawn, FaceState(), Paws(1280), Tail((1280, 720)), None)
            out = render(drawn, FaceState(), Paws(1280), Tail((1280, 720)), None, green)
            assert (out[5, 5] == (0, 255, 0)).all(), cat.name  # the corner: background
            np.testing.assert_array_equal(out[360, 640], plain[360, 640])  # the face: the animal
            assert not renderer.keep_coverage or out is not None
    renderer.keep_coverage = False



class _Webcam:
    """Sends frames 0..n-1 (each filled with its number) at once, then nothing."""

    def __init__(self, n: int):
        self.frames = [np.full((4, 6, 3), i, np.uint8) for i in range(n)]

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)


class _Landmarker:
    """Finds the frame's number."""

    def detect_for_video(self, image, timestamp):
        return int(image.numpy_view()[0, 0, 0])


def test_tracker_hands_on_the_newest_frame_then_the_webcam_failing(monkeypatch):
    monkeypatch.setattr(app, "FIRST_FRAME_TIMEOUT", 0.2)
    monkeypatch.setattr(app, "WEBCAM_TIMEOUT", 0.2)
    tracker = Tracker(_Webcam(5), (_Landmarker(), _Landmarker())).start()
    tracker._thread.join(timeout=5)  # all five tracked, the webcam gave out; drawing fell behind
    tracked = tracker.next()
    assert (tracked.face, tracked.hands, tracked.size) == (4, 4, (6, 4))
    with pytest.raises(SystemExit):  # raised on the main thread, as the webcam stopped
        tracker.next()
    tracker.stop()


def test_paws_choice_by_click_or_key():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    assert controls.paw_mode() == "3d"  # 3D paws to begin with
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "photo"
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "hands"
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "3d"  # round again
    controls.mouse(4, *_centre("paws"), 0, None)  # a click goes on to the next, too
    controls.handle(-1, state)
    assert controls.paw_mode() == "photo" and not controls.tail
    controls.handle(ord("c"), state)  # the choice stays for the next cat
    controls.handle(ord("2"), state)
    assert controls.paw_mode() == "photo"


def test_paws_choice_without_3d_paws():
    controls, state = _controls(), _State()
    controls.handle(ord("1"), state)
    controls.has_paw3d = False
    assert controls.paw_mode() == "photo"  # the photo paws stand in
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "hands"
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "photo"
    controls.has_paw3d = True
    controls.handle(ord("h"), state)
    assert controls.paw_mode() == "hands"


def test_preview_shows_the_paws_choice():
    from types import SimpleNamespace

    out = np.zeros((720, 1280, 3), np.uint8)
    cat = SimpleNamespace(name="Test")
    x, y, w, h = toggle_rects(True)["paws"]
    area = np.s_[y + 8:y + h - 8, x + 10:x + w - 10]
    views = [preview(out, cat, False, paw_mode=mode)[area].astype(int) for mode in ("3d", "photo", "hands")]
    assert all(np.abs(a - b).sum() > 0 for a, b in zip(views, views[1:] + views[:1]))  # each looks different
    two = preview(out, cat, False, paw_mode="photo", paw3d=False)[area].astype(int)
    assert np.abs(two - views[1]).sum() > 0  # without 3D paws, two dots


@pytest.mark.gl
def test_real_hands_take_the_place_of_paws_and_leave_the_tail_hand_out(real_cats):
    from catcam.app import FaceState, load_cats, render
    from catcam.cat_mesh import Renderer
    from catcam.paws import Paws
    from catcam.real_hands import Cut, RealHands
    from catcam.tail import Tail

    cat = next(c for c in load_cats(Renderer((1280, 720))) if c.name == "Woofgang")
    paws = Paws(1280)
    params = np.array([1100, 500, 200, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0])
    paws.slots = {"Right": {"presence": 1.0, "params": params, "detection": 0}}
    hand = np.zeros((80, 60, 4), np.uint8)
    hand[...] = (0, 0, 255, 255)  # a red block of a hand
    real = RealHands()
    real.update([Cut(hand, (300, 200), np.array([330.0, 240.0]))], paws.slots)
    tail = Tail((1280, 720))
    plain = render(cat, FaceState(), paws, tail, None, paw_mode="hands")  # no cut-outs given: no hands
    shown = render(cat, FaceState(), paws, tail, None, paw_mode="hands", real_hands=real)
    lifting = render(cat, FaceState(), paws, tail, "Right", paw_mode="hands", real_hands=real)
    assert (shown[500, 1100] == (0, 0, 255)).all()  # the palm lands where the paw would
    assert not (plain[500, 1100] == (0, 0, 255)).all()
    np.testing.assert_array_equal(lifting[450:550, 1050:1150], plain[450:550, 1050:1150])
    photo = render(cat, FaceState(), paws, tail, None, paw_mode="photo")
    paw_3d = render(cat, FaceState(), paws, tail, None, paw_mode="3d")
    corner = np.s_[400:600, 1000:1200]
    assert np.abs(photo[corner].astype(int) - paw_3d[corner]).mean() > 5  # two different paws


class _Segmenter:
    """Finds skin everywhere, at once; counts the frames it was given."""

    def __init__(self):
        self.started, self.closed = 0, False

    def start(self, frame, timestamp):
        from concurrent.futures import Future

        self.started += 1
        done = Future()
        done.set_result(np.ones(frame.shape[:2], np.float32))
        return done

    def close(self):
        self.closed = True


def test_tracker_cuts_hands_out_only_while_asked(monkeypatch):
    monkeypatch.setattr(app, "FIRST_FRAME_TIMEOUT", 0.2)
    monkeypatch.setattr(app, "WEBCAM_TIMEOUT", 0.2)
    for colours in (None, "the animal's colours"):
        segmenter = _Segmenter()
        tracker = Tracker(_Webcam(3), (_Landmarker(), _Landmarker()), segmenter)
        tracker.cut = lambda frame, skin, hands, colours: [(int(frame[0, 0, 0]), skin.result().shape, hands,
                                                            colours)]
        tracker.cut_for = colours
        tracker.start()._thread.join(timeout=5)
        tracked = tracker.next()
        if colours is None:
            assert tracked.cuts is None and segmenter.started == 0
        else:
            assert tracked.cuts == [(2, (4, 6), 2, colours)] and segmenter.started == 3
        tracker.stop()
        assert segmenter.closed
