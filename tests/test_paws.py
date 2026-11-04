from types import SimpleNamespace

import cv2
import numpy as np

from catcam.paws import (
    CHEEKS,
    CHIN,
    PHOTO_ARM_LENGTH,
    Paws,
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
