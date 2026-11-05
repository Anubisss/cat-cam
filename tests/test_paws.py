from types import SimpleNamespace

import cv2
import numpy as np

from catcam.paws import (
    CHEEKS,
    CHIN,
    PHOTO_ARM_LENGTH,
    Paws,
    finger_curls,
    palm_facing,
    palm_toward,
    paw_set,
    paw_sprite,
    paw_style,
    photo_paw,
)


def _cat():
    """A black cat with a white chin, and landmarks pointing at both."""
    sprite = np.zeros((400, 400, 4), np.uint8)
    sprite[..., 3] = 255
    sprite[300:, :, :3] = 240
    points = np.full((478, 3), 200.0)
    points[:, 0] = np.linspace(50, 350, 478)
    points[CHEEKS, :2] = (100, 100)
    points[CHIN, :2] = (200, 350)
    return sprite, points


def test_paw_takes_the_cats_colours():
    style = paw_style(*_cat())
    assert max(style.arm) < 20
    assert min(style.sock) > 220


def test_paw_sprite_has_a_paw_and_an_arm():
    paw = paw_sprite(paw_style(*_cat()), width=100)
    alpha = paw.image[..., 3]
    x, y = paw.anchor.astype(int)
    assert alpha[y, x] == 255  # paw centre
    assert alpha[-1, x] == 255  # arm runs to the bottom
    assert alpha[0, 0] == 0


def _hand(cx, cy, size=0.05):
    """21 hand landmarks (normalised) of an upright open hand."""
    points = [SimpleNamespace(x=cx, y=cy + size) for _ in range(21)]
    for i, dx in zip((5, 9, 13, 17), (-1.5, -0.5, 0.5, 1.5)):
        points[i] = SimpleNamespace(x=cx + dx * size / 2, y=cy - size)
    return points


def test_paws_follow_hands_and_fade_out():
    paw = paw_sprite(paw_style(*_cat()), width=100)
    paws = Paws(out_width=1280)
    frame = np.zeros((720, 1280, 3), np.uint8)
    hands = SimpleNamespace(
        hand_landmarks=[_hand(0.8, 0.5)],
        handedness=[[SimpleNamespace(category_name="Right")]],
    )
    for _ in range(20):
        paws.update(hands, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)
    drawn = paws.draw(frame, paw)
    changed = np.argwhere(drawn.any(axis=2))
    assert len(changed)
    assert changed[:, 1].mean() > 640  # the paw is on the side of the hand

    nothing = SimpleNamespace(hand_landmarks=[], handedness=[])
    for _ in range(40):
        paws.update(nothing, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)
    assert not paws.draw(frame, paw).any()


def test_two_hands_with_the_same_label_get_their_own_paws():
    paw = paw_sprite(paw_style(*_cat()), width=100)
    paws = Paws(out_width=1280)
    label = [SimpleNamespace(category_name="Left")]
    hands = SimpleNamespace(hand_landmarks=[_hand(0.2, 0.5), _hand(0.8, 0.5)], handedness=[label, label])
    for _ in range(20):
        paws.update(hands, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)
    drawn = paws.draw(np.zeros((720, 1280, 3), np.uint8), paw)
    columns = np.argwhere(drawn.any(axis=2))[:, 1]
    assert (columns < 640).any() and (columns > 640).any()


def test_hidden_hands_get_no_paw():
    paw = paw_sprite(paw_style(*_cat()), width=100)
    paws = Paws(out_width=1280)
    paws.slots = {"Right": {"presence": 1.0, "params": np.array([640, 300, 120, 0, -1, -1, 0])}}
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert paws.draw(frame, paw).any()
    assert not paws.draw(frame, paw, hidden=("Right",)).any()


def _photo_paw(color, width=60):
    """A BGRA paw photo: a wide paw on top of a narrower arm leaving at the bottom."""
    img = np.zeros((200, width + 20, 4), np.uint8)
    cv2.ellipse(img, (img.shape[1] // 2, 40), (width // 2, 30), 0, 0, 360, (*color, 255), -1)
    cv2.rectangle(img, (img.shape[1] // 2 - 15, 40), (img.shape[1] // 2 + 15, 199), (*color, 255), -1)
    return img


def test_photo_paw_arm_runs_long():
    paw = photo_paw(_photo_paw((10, 20, 30)))
    assert abs(paw.width - 60) <= 2
    assert len(paw.image) >= PHOTO_ARM_LENGTH * paw.width
    assert paw.image[-1, :, 3].max() == 255  # still arm at the very bottom
    x, y = paw.anchor.astype(int)
    assert paw.image[y, x, 3] == 255


def test_paw_set_falls_back():
    style = paw_style(*_cat())
    back = {("back", h): _photo_paw((0, 0, 255)) for h in ("left", "right")}
    paws = paw_set(back, style)
    assert set(paws) == {(v, h) for v in ("front", "back") for h in ("left", "right")}
    assert paws["front", "left"].image[int(paws["front", "left"].anchor[1]), 40, 2] == 255  # the sole
    drawn = paw_set({}, style)
    assert drawn["front", "left"] is drawn["back", "right"]


def _upright_hand(handedness, palm):
    """Hand points (px) with fingers up, index knuckle on the side that makes
    it show its palm or its back."""
    # A right hand showing its palm has its index knuckle on the image right.
    index_right = (handedness == "Right") == palm
    pts = np.zeros((21, 2))
    pts[0] = (0, 100)
    pts[9] = (0, 0)
    pts[5], pts[17] = ((20, 0), (-20, 0)) if index_right else ((-20, 0), (20, 0))
    return pts


def test_palm_or_back_of_the_hand():
    for handedness in ("Left", "Right"):
        assert palm_facing(palm_toward(handedness, _upright_hand(handedness, palm=True))) == 1
        assert palm_facing(palm_toward(handedness, _upright_hand(handedness, palm=False))) == 0
    edge_on = _upright_hand("Right", palm=True)
    edge_on[[5, 17], 0] = (1, -1)
    edge_on[[5, 17], 1] = (-20, 20)  # knuckles in line with the hand
    assert 0 < palm_facing(palm_toward("Right", edge_on)) < 1


def test_palms_show_the_sole_and_backs_the_top():
    paws_set = {
        (v, h): photo_paw(_photo_paw((0, 0, 255) if v == "back" else (255, 0, 0)))
        for v in ("front", "back") for h in ("left", "right")
    }
    frame = np.zeros((720, 1280, 3), np.uint8)
    for palm, channel in ((True, 2), (False, 0)):
        # A right hand: _hand has the index knuckle (5) on the image left, the
        # back of the hand; mirrored, it shows its palm.
        hand = _hand(0.8, 0.5)
        if palm:
            for i, dx in zip((5, 9, 13, 17), (1.5, 0.5, -0.5, -1.5)):
                hand[i] = SimpleNamespace(x=0.8 + dx * 0.025, y=0.45)
        hands = SimpleNamespace(hand_landmarks=[hand], handedness=[[SimpleNamespace(category_name="Right")]])
        paws = Paws(out_width=1280)
        for _ in range(20):
            paws.update(hands, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)
        drawn = paws.draw(frame, paws_set)
        painted = drawn[drawn.any(axis=2)]
        assert painted[:, channel].mean() > 200, palm


def _fingers(reach):
    """Hand points (px), fingers up (index on the image left), tips at `reach`
    (one per finger, index to pinky, or one for all) times the knuckles' distance from the wrist."""
    pts = np.zeros((21, 2))
    pts[0] = (0, 100)
    knuckles = ((8, 5, -30), (12, 9, -10), (16, 13, 10), (20, 17, 30))
    for (tip, mcp, x), r in zip(knuckles, np.broadcast_to(reach, 4)):
        pts[mcp] = (x, 0)
        pts[tip] = pts[0] + r * (pts[mcp] - pts[0])
    return pts


def test_fist_or_open_hand():
    np.testing.assert_array_equal(finger_curls(_fingers(2.0)), 0)
    np.testing.assert_array_equal(finger_curls(_fingers(0.8)), 1)
    assert (0 < finger_curls(_fingers(1.25))).all() and (finger_curls(_fingers(1.25)) < 1).all()


def test_each_finger_bends_on_its_own():
    np.testing.assert_array_equal(finger_curls(_fingers([2.0, 0.8, 2.0, 2.0])), [0, 1, 0, 0])


def _toe_tops(curls):
    """Top row of the paw drawn upright, per column, with the toes curled by curls (from the left)."""
    paw = photo_paw(_photo_paw((0, 0, 255)))
    paws = Paws(out_width=400)
    paws.slots = {"Right": {"presence": 1.0, "params": np.array([200, 150, 120, 0, -1, -1, *curls])}}
    drawn = paws.draw(np.zeros((400, 400, 3), np.uint8), paw).any(axis=2)
    assert drawn[-1].any()  # the arm is unchanged
    return np.where(drawn.any(axis=0), drawn.argmax(axis=0), 400)


def test_closed_paws_curl_their_toes_in():
    open_paw, fist = _toe_tops([0, 0, 0, 0]), _toe_tops([1, 1, 1, 1])
    assert fist.min() > open_paw.min() + 10  # the toes are shorter
    across_a_toe = slice(200 - int(0.2 * 120), 200 - int(0.04 * 120))
    assert np.ptp(fist[across_a_toe] - open_paw[across_a_toe]) <= 2  # slid down, not flattened


def test_one_bent_finger_curls_only_its_toe():
    open_paw, one = _toe_tops([0, 0, 0, 0]), _toe_tops([1, 0, 0, 0])
    left, right = 200 - int(0.32 * 120), 200 + int(0.32 * 120)  # the outer toes' middles
    assert one[left] > open_paw[left] + 5
    assert one[right] == open_paw[right]


def test_a_finger_raised_from_a_fist_stands_up():
    open_paw, middle = _toe_tops([0, 0, 0, 0]), _toe_tops([1, 0, 1, 1])
    finger, outer = 200 - int(0.12 * 120), 200 + int(0.32 * 120)
    assert middle[finger] < open_paw[finger] - 0.5 * 120  # half a paw width taller
    assert middle[outer] > open_paw[outer] + 5  # the others are curled in
    assert (middle < 400).sum() <= (open_paw < 400).sum()  # no wider for it


def test_fingers_bend_the_toes_on_their_side_of_the_paw():
    """A hand showing its back, index finger bent: the toe on the index's side
    of the screen curls, whichever way up the hand is."""
    for flip in (False, True):
        pts = _fingers([0.8, 2.0, 2.0, 2.0])
        if flip:  # pointing down: the index is now on the image right... of the upside-down paw
            pts = pts * (1, -1)
        hand = [SimpleNamespace(x=0.5 + x / 1280, y=0.5 + y / 720) for x, y in pts]
        hands = SimpleNamespace(hand_landmarks=[hand], handedness=[[SimpleNamespace(category_name="Right")]])
        paws = Paws(out_width=1280)
        paws.update(hands, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)
        curls = paws.slots["Right"]["params"][6:]
        np.testing.assert_allclose(curls, [0, 0, 0, 1] if flip else [1, 0, 0, 0], atol=1e-9)


def _detect(*hands):
    """A HandLandmarkerResult with (x, y, label) hands."""
    return SimpleNamespace(
        hand_landmarks=[_hand(x, y) for x, y, _ in hands],
        handedness=[[SimpleNamespace(category_name=label)] for _, _, label in hands],
    )


def _update(paws, result, frames=1):
    for _ in range(frames):
        paws.update(result, None, (1280, 720), 1 / 30, np.array([640.0, 300.0]), 300.0)


def test_a_hand_keeps_its_slot_when_its_left_right_guess_flips():
    paws = Paws(out_width=1280)
    _update(paws, _detect((0.8, 0.5, "Right")), 20)
    _update(paws, _detect((0.79, 0.5, "Left")), 3)  # MediaPipe briefly calls it a left hand
    assert paws.slots["Right"]["presence"] > 0.9
    assert paws.slots.get("Left", {}).get("presence", 0) < 0.02


def test_two_hands_keep_their_slots_when_both_guesses_flip():
    paws = Paws(out_width=1280)
    _update(paws, _detect((0.2, 0.5, "Left"), (0.8, 0.5, "Right")), 20)
    _update(paws, _detect((0.21, 0.5, "Right"), (0.79, 0.5, "Left")), 3)
    assert paws.slots["Left"]["params"][0] < 640 < paws.slots["Right"]["params"][0]


def test_a_new_hand_far_away_goes_by_its_guess():
    paws = Paws(out_width=1280)
    _update(paws, _detect((0.8, 0.5, "Right")), 20)
    _update(paws, _detect((0.8, 0.5, "Right"), (0.2, 0.5, "Left")), 3)
    assert paws.slots["Left"]["params"][0] < 640 < paws.slots["Right"]["params"][0]
