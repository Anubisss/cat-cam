from pathlib import Path

import numpy as np
import pytest

from catcam import reactions
from catcam.reactions import (
    DURATION,
    EMOJI_FONT,
    FADE_OUT,
    POP_IN,
    REACTIONS,
    Reactions,
    composite,
    pose,
    reaction_for,
)

NAMES = [name for name, _ in REACTIONS]
has_font = pytest.mark.skipif(not Path(EMOJI_FONT).exists(), reason="needs the Mac's emoji font")


def _frame(w=320, h=180):
    return np.full((h, w, 3), 90, np.uint8)


def _poses(name, step=0.01):
    return [pose(name, t) for t in np.arange(POP_IN, DURATION - FADE_OUT, step)]


def test_number_keys_one_to_eight_are_the_reactions_in_order():
    assert [reaction_for(ord(str(i))) for i in range(1, 9)] == NAMES
    assert NAMES[0] == "point" and NAMES[-1] == "wave"


@pytest.mark.parametrize("key", [ord("9"), ord("0"), ord("c"), ord("t"), -1, 255])
def test_other_keys_are_no_reaction(key):
    assert reaction_for(key) is None


@pytest.mark.parametrize("name", NAMES)
def test_every_reaction_grows_in_stays_and_fades_away(name):
    assert pose(name, 0.0).scale == pytest.approx(0, abs=1e-9) and pose(name, 0.0).alpha == 0
    middle = pose(name, DURATION / 2)
    assert middle.alpha == 1 and 0.7 < middle.scale < 1.3
    assert pose(name, DURATION - 0.01).alpha < 0.05
    assert pose(name, DURATION).alpha == 0 and pose(name, -0.1).alpha == 0


@pytest.mark.parametrize("name", NAMES)
def test_every_reaction_moves(name):
    poses = _poses(name)
    looks = {(round(p.dx, 3), round(p.dy, 3), round(p.scale * p.sx, 3), round(p.sy, 3), round(p.angle, 1))
             for p in poses}
    assert len(looks) > 5


def test_growing_in_overshoots_a_little():
    peak = max(pose("palm", t).scale for t in np.arange(0, POP_IN, 0.005))
    assert 1.03 < peak < 1.25


def test_the_gestures_fit_their_emoji():
    assert max(p.scale for p in _poses("point")) > 1.1  # pokes toward the viewer
    assert min(p.dy for p in _poses("raise")) < -0.05 and max(p.dy for p in _poses("raise")) <= 0  # jumps up
    assert max(p.dy for p in _poses("pray")) > 0.03  # bows
    assert pose("down", DURATION - FADE_OUT).dy > 0.15  # ends up lower
    assert min(p.sx for p in _poses("clap")) < 0.9  # squeezes
    shake = [p.dy for p in _poses("pinch")]
    assert min(shake) < -0.03 and max(shake) > 0.03  # up and down
    for name in ("palm", "wave"):  # sways either way
        angles = [p.angle for p in _poses(name)]
        assert min(angles) < -5 and max(angles) > 5
    assert max(abs(p.angle) for p in _poses("wave")) > max(abs(p.angle) for p in _poses("palm"))


def test_unknown_reaction_is_an_error():
    with pytest.raises(ValueError):
        pose("shrug", 1.0)


def _fake_picture(monkeypatch):
    """A plain square in place of the emoji, so these tests need no font."""
    picture = np.zeros((reactions.GLYPH_SIZE, reactions.GLYPH_SIZE, 4), np.float32)
    picture[20:-20, 20:-20] = (0.0, 0.0, 1.0, 1.0)  # opaque red (BGR, premultiplied)
    monkeypatch.setattr(reactions, "sprite", lambda emoji, font_path=EMOJI_FONT: picture)


def test_a_reaction_shows_then_goes(monkeypatch):
    _fake_picture(monkeypatch)
    shown, frame = Reactions(), _frame()
    assert shown.showing(0.0) is None and shown.draw(frame, 0.0) is frame
    shown.play("wave", 10.0)
    assert shown.showing(10.0 + DURATION / 2) == "wave"
    out = shown.draw(frame, 10.0 + DURATION / 2)
    assert out is not frame and out.shape == frame.shape and (out != frame).any()
    assert (frame == 90).all()  # the frame passed in stays as it was
    assert shown.showing(10.0 + DURATION + 0.01) is None
    assert shown.draw(frame, 10.0 + DURATION + 0.01) is frame


def test_the_same_key_again_starts_it_over_and_another_swaps_it(monkeypatch):
    _fake_picture(monkeypatch)
    shown = Reactions()
    shown.play("clap", 0.0)
    shown.play("clap", 2.0)
    assert shown.showing(DURATION + 1.0) == "clap"  # past where the first would have ended
    shown.play("down", 3.0)
    assert shown.showing(3.1) == "down"


def test_it_sits_beside_the_animal_in_the_lower_right(monkeypatch):
    _fake_picture(monkeypatch)
    frame = _frame(1280, 720)
    out = composite(frame, reactions.sprite("x"), pose("palm", DURATION / 2))
    ys, xs = np.nonzero((out != frame).any(axis=2))
    assert xs.min() > 1280 * 0.7 and ys.min() > 720 * 0.3
    assert 0.2 * 720 < ys.max() - ys.min() < 0.35 * 720  # about SIZE frame heights


def test_drawing_past_the_frame_edge_is_cut_off(monkeypatch):
    _fake_picture(monkeypatch)
    small = _frame(60, 40)
    for name in NAMES:
        for t in (0.05, 0.5, 1.3, 2.3):
            assert composite(small, reactions.sprite("x"), pose(name, t)).shape == small.shape


def test_without_the_font_keys_do_nothing():
    shown, frame = Reactions(font_path="/no/such/font.ttc"), _frame()
    shown.play("wave", 0.0)
    assert shown.showing(1.0) is None and shown.draw(frame, 1.0) is frame
    assert reactions.sprite("\U0001F44B", "/no/such/font.ttc") is None


@has_font
@pytest.mark.parametrize("emoji", [e for _, e in REACTIONS])
def test_every_emoji_comes_from_the_font(emoji):
    picture = reactions.sprite(emoji)
    assert picture is not None and picture.shape == (reactions.GLYPH_SIZE, reactions.GLYPH_SIZE, 4)
    alpha = picture[..., 3]
    assert 0.2 < (alpha > 0.5).mean() < 0.95  # a shape, not empty and not a filled square
    assert (picture[..., :3] <= alpha[..., None] + 1e-6).all()  # premultiplied
