import numpy as np
import pytest

from catcam import paw3d
from catcam.paws import Placement, hand_axes


@pytest.mark.parametrize("animal", ["cat", "dog"])
def test_shipped_paw_models_are_rigged(animal):
    m = paw3d.load(animal)
    assert len(m.pos) == len(m.normal) == len(m.toes) == len(m.pad) == len(m.light)
    assert len(m.arm_pos) == len(m.arm_normal) == len(m.arm_pad) == len(m.arm_light)
    assert m.faces.max() < len(m.pos) and m.arm_faces.max() < len(m.arm_pos)
    assert (m.toes >= 0).all() and (m.toes.sum(axis=1) <= 1.001).all()
    assert (m.toes.max(axis=0) > 0.99).all()  # every toe has vertices that are all its own
    # The toes are at the top: nothing below the palm bends.
    assert not m.toes[m.pos[:, 1] < 0].any()
    assert m.knuckles.shape == (4, 2) and m.toe_length > 0.2
    # Pads on the sole (-z), around the palm at the origin; the arm piece hangs below the paw.
    pads = m.pos[m.pad > 0.5]
    assert len(pads) and np.median(pads[:, 2]) < 0
    top, length = -m.arm_cut, m.arm_piece
    assert m.arm_pos[:, 1].max() <= top + 0.2 and m.arm_pos[:, 1].min() >= top - length - 0.2


def test_paws_take_their_colours_from_the_portrait(real_cats):
    colours = {c.name: paw3d.paw_colours(c.sprite, c.points) for c in real_cats}
    luma = lambda bgr: bgr @ np.array([0.114, 0.587, 0.299])  # noqa: E731

    dog = colours["Woofgang"]
    assert luma(np.array(dog.pad)) < 60  # a black nose, black pads
    assert dog.arm[2] > dog.arm[0] + 60  # golden fur: much more red than blue (BGR)
    white = colours["Meringue"]
    assert min(white.sock) > 150 and white.pad[2] > white.pad[1]  # white fur, pink pads
    assert colours["Grumbledore"].tabby > 0.5 and colours["Meringue"].tabby == 0


def test_toes_curl_toward_the_sole_and_a_raised_one_stands():
    m = paw3d.load("cat")
    bend, lift = paw3d.toe_bends(m, [0, 0, 0, 0])
    assert (bend == 0).all() and (lift == 0).all()
    bend, lift = paw3d.toe_bends(m, [1, 1, 1, 1])
    assert (bend < -1).all() and (lift == 0).all()
    bend, lift = paw3d.toe_bends(m, [0, 1, 1, 1])  # the index raised out of a fist
    assert bend[0] == 0 and lift[0] > 0 and (lift[1:] == 0).all()


def _params(across, hand):
    """Paws' params of an upright hand at (400, 300), 100 px, across toward the thumb."""
    return np.array([400, 300, 100, 0, -1, 0, *across, hand, 0, 0, 0, 0], float)


def test_hand_frames():
    # A right hand with its index on the image right shows its palm: its back faces away.
    right_palm = hand_axes(_params((1, 0, 0), 1))
    np.testing.assert_allclose(right_palm[:, 2], (0, 0, 1), atol=1e-9)
    assert np.linalg.det(right_palm) < 0  # right hands get the paw mirrored
    left_palm = hand_axes(_params((-1, 0, 0), -1))
    np.testing.assert_allclose(left_palm[:, 2], (0, 0, 1), atol=1e-9)
    assert np.linalg.det(left_palm) > 0
    np.testing.assert_allclose(hand_axes(_params((-1, 0, 0), 1))[:, 2], (0, 0, -1), atol=1e-9)  # its back


@pytest.fixture(scope="module")
def renderer():
    from catcam.cat_mesh import Renderer

    return Renderer((640, 400))


def _paw(renderer, animal="cat", pad=(255, 0, 255)):
    colours = paw3d.PawColours(arm=(90, 90, 90), sock=(90, 90, 90), stripes=(90, 90, 90), tabby=0.0, pad=pad)
    return paw3d.Paw3D(renderer, paw3d.load(animal), colours)


def _draw(paw, palm=True, curls=(0, 0, 0, 0), x=320, opacity=1.0):
    """A left paw (unmirrored), upright at (x, 200), 120 px wide, on black."""
    across = (-1, 0, 0) if palm else (1, 0, 0)
    axes = hand_axes(_params(across, -1))
    frame = np.zeros((400, 640, 3), np.uint8)
    return paw.render(frame, [Placement(x, 200, 120, axes, np.array(curls, float), opacity)])


def _pad_pixels(img):
    """Pixels in the pad colour (magenta)."""
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return (b > g + 60) & (r > g + 60)


@pytest.mark.gl
@pytest.mark.parametrize("animal", ["cat", "dog"])
def test_palms_show_the_pads_and_backs_the_fur(renderer, animal):
    paw = _paw(renderer, animal)
    palm, back = _draw(paw, palm=True), _draw(paw, palm=False)
    assert _pad_pixels(palm).sum() > 300
    assert _pad_pixels(back).sum() < 0.1 * _pad_pixels(palm).sum()
    drawn = np.argwhere(palm.any(axis=2))
    assert drawn[:, 0].max() == 399  # the arm runs off the bottom
    assert abs(np.median(drawn[drawn[:, 0] < 200, 1]) - 320) < 60  # around where the paw was put


@pytest.mark.gl
def test_a_fist_curls_the_toes_down(renderer):
    paw = _paw(renderer)

    def top(img):
        return np.argwhere(img.any(axis=2))[:, 0].min()

    assert top(_draw(paw, curls=(1, 1, 1, 1))) > top(_draw(paw)) + 10


@pytest.mark.gl
def test_each_finger_curls_its_own_toe(renderer):
    """Seen from the back, a left hand's index is on the image right: bending it curls
    the toe on that side only."""
    paw = _paw(renderer)

    def height(img, columns):
        return np.argwhere(img[:, columns].any(axis=2))[:, 0].min()

    open_paw, index = _draw(paw, palm=False), _draw(paw, palm=False, curls=(1, 0, 0, 0))
    right, left = slice(350, 385), slice(255, 295)  # the outer toes, a paw width being 120 px
    assert height(index, right) > height(open_paw, right) + 5
    assert abs(height(index, left) - height(open_paw, left)) <= 2


@pytest.mark.gl
def test_paws_fade_and_mirror(renderer):
    paw = _paw(renderer)
    assert not _draw(paw, opacity=0.0).any()
    half = _draw(paw, opacity=0.5).astype(int)
    full = _draw(paw).astype(int)
    assert 0.3 * full.sum() < half.sum() < 0.7 * full.sum()
    # A right paw is the left one mirrored: the same placement mirrored about the frame's
    # middle draws the mirror picture.
    left = _draw(paw, x=200)
    frame = np.zeros((400, 640, 3), np.uint8)
    axes = hand_axes(_params((1, 0, 0), 1))  # a right hand showing its palm
    right = paw.render(frame, [Placement(440, 200, 120, axes, np.zeros(4), 1.0)])
    assert np.abs(right[:, ::-1].astype(int) - left).mean() < 3


@pytest.mark.gl
def test_animals_get_their_own_paws(real_cats):
    from catcam.app import load_cats
    from catcam.cat_mesh import Renderer

    cats = {c.name: c for c in load_cats(Renderer((320, 180)))}
    assert cats["Woofgang"].paw.model is paw3d.load("dog")
    assert all(c.paw.model is paw3d.load("cat") for name, c in cats.items() if name != "Woofgang")


@pytest.mark.gl
def test_without_a_paw_model_the_photo_paws_stand_in(real_cats, monkeypatch, tmp_path):
    from catcam.app import load_cats
    from catcam.cat_mesh import Renderer

    monkeypatch.setattr(paw3d, "PAWS", tmp_path)
    cat = load_cats(Renderer((320, 180)))[0]
    assert isinstance(cat.paw, dict) and ("front", "left") in cat.paw
