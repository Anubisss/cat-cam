from types import SimpleNamespace

import numpy as np

from catcam.paws import CHEEKS, CHIN, Paws, paw_sprite, paw_style


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
