import cv2
import numpy as np

from catcam import paw_photos
from catcam.paw_photos import PAW_WIDTH, divider, halves, paw_width, prepare_paws, upright

BACKDROP = (170, 180, 185)
PURPLE = (200, 40, 140)  # BGR


def _photo(toes_down=False):
    """Two paws (a wide paw on a narrower arm), split by a purple line."""
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = BACKDROP
    img[:, 197:203] = PURPLE
    for cx in (100, 300):
        cv2.rectangle(img, (cx - 30, 80), (cx + 30, 300), (40, 90, 160), -1)  # arm
        cv2.ellipse(img, (cx, 80), (50, 40), 0, 0, 360, (60, 120, 200), -1)  # paw
    return img[::-1].copy() if toes_down else img


def _cut_out(bgr):
    alpha = (np.abs(bgr.astype(int) - BACKDROP).sum(axis=2) > 30).astype(np.uint8) * 255
    return np.dstack([bgr, alpha])


def test_finds_the_purple_line():
    assert 196 <= divider(_photo()) <= 203
    plain = np.full((100, 200, 3), 128, np.uint8)
    assert divider(plain) == 100
    left, right = halves(_photo()).values()
    for half in (left, right):
        assert not (np.abs(half.astype(int) - PURPLE).sum(axis=2) < 30).any()


def test_paws_end_up_toes_up_at_a_fixed_width():
    for toes_down in (False, True):
        half = halves(_photo(toes_down))["left"]
        paw = upright(_cut_out(half), toes_down)
        assert abs(paw_width(paw[..., 3]) - PAW_WIDTH) <= 2
        alpha = paw[..., 3] > 127
        assert alpha[-1].sum() < alpha[: len(alpha) // 3].sum(axis=1).max()  # the arm leaves at the bottom


def test_prepare_paws_writes_each_view_and_hand(tmp_path):
    portraits, assets = tmp_path / "portraits", tmp_path / "assets"
    portraits.mkdir()
    cv2.imwrite(str(portraits / "AmyPawsBack.png"), _photo())
    assert paw_photos.has_photos("Amy", portraits) == ["back"]
    assert prepare_paws("Amy", portraits, assets, _cut_out) == ["back"]
    files = paw_photos.paw_files("Amy", assets)
    assert files["back", "left"].is_file() and files["back", "right"].is_file()
    assert not files["front", "left"].exists()


def test_paw_photos_are_told_apart_from_portraits():
    assert paw_photos.is_paw_photo("AmyPawsFront") and paw_photos.is_paw_photo("AmyPawsBack")
    assert not paw_photos.is_paw_photo("Amy")
