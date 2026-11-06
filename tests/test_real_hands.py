from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from catcam import real_hands as rh
from catcam.paw3d import PawColours
from catcam.paws import WRIST

# A blue-grey animal's (BGR), well off the skin's colour.
COLOURS = PawColours(arm=(160, 130, 110), sock=(200, 170, 150), stripes=(90, 70, 60), tabby=0.0,
                     pad=(120, 120, 200))


def _hand(x: float, y: float, size: float = 60.0) -> np.ndarray:
    """(21, 2) px: an open hand pointing up, its wrist at (x, y), palm about size px long."""
    bases = [(-0.45, -0.35), (-0.3, -1.0), (-0.05, -1.05), (0.2, -1.0), (0.42, -0.9)]
    points = [(0.0, 0.0)]
    for bx, by in bases:
        for k in range(4):
            points.append((bx * (1 + 0.08 * k), by - 0.3 * k))
    return np.array(points) * size + (x, y)


def _scene(w: int = 640, h: int = 480):
    """A grey frame with a skin-coloured hand (and forearm) and a face-like blob next to it,
    and the segmenter's skin for it at SEGMENT_WIDTH: the hand and arm only."""
    frame = np.full((h, w, 3), 90, np.uint8)
    points = _hand(200, 260)
    hand = np.zeros((h, w), bool)
    ys, xs = np.mgrid[0:h, 0:w]
    lo, hi = points.min(axis=0) - 8, points.max(axis=0) + 8
    hand |= (xs >= lo[0]) & (xs <= hi[0]) & (ys >= lo[1]) & (ys <= hi[1])
    hand |= (np.abs(xs - 200) < 30) & (ys > 260)  # the forearm, down to the frame's edge
    frame[hand] = (110, 150, 200)
    face = (xs - 330) ** 2 + (ys - 150) ** 2 < 60 ** 2
    frame[face] = (100, 140, 210)
    sw = rh.SEGMENT_WIDTH
    small = np.zeros((round(sw * h / w), sw), np.float32)
    sy, sx = np.mgrid[0:small.shape[0], 0:sw]
    small[hand[(sy * h / small.shape[0]).astype(int), (sx * w / sw).astype(int)]] = 0.95
    return frame, small, points, hand, face


def test_steady_follows_the_newest_and_starts_afresh_after_a_gap():
    before, new = np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)
    np.testing.assert_allclose(rh.steady(new, before, 0.033), rh.SKIN_FOLLOW)
    np.testing.assert_allclose(rh.steady(new, before, rh.RESTART + 0.1), 1.0)
    np.testing.assert_allclose(rh.steady(new, None, None), 1.0)


def test_skin_lines_up_with_the_frame():
    small = np.zeros((90, 160), np.float32)
    small[40:50, 80:90] = 1.0  # small px 80..90 = frame px 320..360 at 4x
    up = rh.skin_in(small, (640, 360), (300, 140, 380, 220))
    assert up[60 - 20 + 2, 340 - 300] == 1.0  # frame (340, 182): inside
    assert up[5, 5] == 0.0 and up[60 - 20 + 2, 300 - 300] == 0.0


def test_cut_out_takes_the_hand_and_forearm_but_not_the_face():
    frame, small, points, hand, face = _scene()
    cut = rh.cut_out(frame, small, points, COLOURS)
    assert cut is not None
    x0, y0 = cut.origin
    h, w = cut.image.shape[:2]
    alpha = np.zeros(frame.shape[:2])
    alpha[y0:y0 + h, x0:x0 + w] = cut.image[..., 3] / 255
    assert alpha[220, 200] > 0.99  # the palm
    assert alpha[275, 200] > 0.99  # the wrist, just below
    assert alpha[face].max() < 0.01  # the face, though within reach of the fingers
    near = cv2.dilate(hand.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0  # 5 px round it
    assert alpha[~near].max() < 0.05  # outside the hand only its feathered edge
    down = rh.hand_size(points)
    assert alpha[int(260 + rh.FOREARM[1] * down) + 5:, :].max() == 0  # the forearm fades out
    np.testing.assert_allclose(cut.palm, (0.6 * points[[5, 9, 13, 17]].mean(axis=0) + 0.4 * points[WRIST]))


def test_a_hand_found_where_there_is_no_skin_is_not_cut_out():
    frame, small, _, _, _ = _scene()
    assert rh.cut_out(frame, small, _hand(330, 200), COLOURS) is None  # a "hand" over the face
    assert rh.cut_out(frame, small, _hand(-500, -500), COLOURS) is None  # off the frame


def test_the_wash_keeps_the_hands_shading_and_takes_the_fur_colour():
    frame, small, points, _, _ = _scene()
    frame[200:240, 180:220] = (70, 105, 140)  # a shadow on the palm
    cut = rh.cut_out(frame, small, points, COLOURS)
    x0, y0 = cut.origin
    palm = cut.image[250 - y0, 200 - x0, :3].astype(int)
    shadow = cut.image[220 - y0, 200 - x0, :3].astype(int)
    assert shadow.sum() < palm.sum() - 60  # still darker where the hand is
    skin, fur = np.array((110, 150, 200)), np.array(COLOURS.sock)
    hue = lambda c: c / np.linalg.norm(c)
    # Moved toward the fur's colour, but not all the way.
    assert np.linalg.norm(hue(palm) - hue(fur)) < np.linalg.norm(hue(skin) - hue(fur))
    assert np.abs(palm - fur).sum() > 20


def test_the_grain_runs_along_the_hand():
    points = _hand(100, 150)
    grain = rh.grain(points, (0, 0, 200, 200))
    assert -1 <= grain.min() and grain.max() <= 1
    along = np.abs(np.diff(grain, axis=0)).mean()  # the hand points up: streaks run along y
    across = np.abs(np.diff(grain, axis=1)).mean()
    assert along < across


def _slot(x, y, presence=1.0, detection=0):
    return {"params": np.array([x, y, 100.0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0]), "presence": presence,
            "detection": detection}


def test_hands_are_drawn_where_the_paws_go_scaled_about_the_palm():
    image = np.zeros((40, 20, 4), np.uint8)
    image[...] = (0, 255, 0, 255)
    cut = rh.Cut(image, (100, 100), np.array([110.0, 120.0]))
    real = rh.RealHands()
    slots = {"Left": _slot(400, 300)}
    real.update([cut], slots)
    out = real.draw(np.zeros((600, 800, 3), np.uint8), slots, 2.0)
    ys, xs = np.nonzero(out[..., 1] > 128)
    # Its px span 100..119 and 100..139 (centres), the palm at (110, 120) going to (400, 300).
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == pytest.approx((380, 418, 260, 338), abs=1)
    hidden = real.draw(np.zeros((600, 800, 3), np.uint8), slots, 2.0, hidden=("Left",))
    assert not hidden.any()


def test_a_lost_hand_fades_with_its_last_cut_out():
    image = np.full((10, 10, 4), 255, np.uint8)
    real = rh.RealHands()
    slots = {"Right": _slot(50, 50)}
    real.update([rh.Cut(image, (0, 0), np.array([5.0, 5.0]))], slots)
    slots["Right"].update(presence=0.5, detection=None)
    real.update([], slots)  # not found this frame
    out = real.draw(np.zeros((100, 100, 3), np.uint8), slots, 1.0)
    assert 110 < out[50, 50, 0] < 145  # half there
    slots["Right"]["presence"] = 0.01
    real.update([], slots)
    assert "Right" not in real.cuts
    real.update([rh.Cut(image, (0, 0), np.array([5.0, 5.0]))], {"Right": _slot(50, 50)})
    real.update(None, slots)  # cutting stopped: nothing stale is kept for when it starts again
    assert not real.cuts


def test_segmenter_finds_skin_on_its_thread():
    segmenter = rh.Segmenter()
    try:
        frame = np.full((360, 640, 3), 120, np.uint8)
        skin = segmenter.start(frame, 0).result(timeout=10)
        assert skin.shape == (round(rh.SEGMENT_WIDTH * 360 / 640), rh.SEGMENT_WIDTH)
        assert skin.dtype == np.float32 and 0 <= skin.min() and skin.max() <= 1
        assert segmenter.start(frame, 33).result(timeout=10).shape == skin.shape
    finally:
        segmenter.close()


def test_cut_all_follows_the_trackers_order():
    frame, small, points, _, _ = _scene()
    h, w = frame.shape[:2]
    landmarks = lambda pts: [SimpleNamespace(x=x / w, y=y / h) for x, y in pts]
    hands = SimpleNamespace(hand_landmarks=[landmarks(_hand(330, 200)), landmarks(points)])
    cuts = rh.cut_all(frame, small, hands, COLOURS)
    assert cuts[0] is None and cuts[1] is not None
