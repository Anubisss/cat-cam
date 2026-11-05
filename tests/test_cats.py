import cv2
import numpy as np
import pytest

from catcam import cats


def _prepare(folder, name, points_shape=(478, 3)):
    (folder / name).mkdir(parents=True)
    cv2.imwrite(str(folder / name / "cutout.png"), np.zeros((8, 8, 4), np.uint8))
    np.savez(folder / name / "face.npz", points=np.zeros(points_shape), background=np.zeros(3))


def test_available_lists_complete_cats_only(tmp_path):
    _prepare(tmp_path, "Zed")
    _prepare(tmp_path, "Amy")
    (tmp_path / "Half").mkdir()
    cv2.imwrite(str(tmp_path / "Half" / "cutout.png"), np.zeros((8, 8, 4), np.uint8))
    assert cats.available(tmp_path) == ["Amy", "Zed"]


def test_available_without_assets_folder(tmp_path):
    assert cats.available(tmp_path / "missing") == []


def test_unprepared(tmp_path):
    portraits, assets = tmp_path / "portraits", tmp_path / "assets"
    portraits.mkdir()
    for name in ("Amy", "Bob", "AmyPawsFront", "BobPawsBack"):
        cv2.imwrite(str(portraits / f"{name}.png"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(portraits / "Cid.PNG"), np.zeros((8, 8, 3), np.uint8))
    _prepare(assets, "Amy")
    assert cats.unprepared(portraits, assets) == ["Bob", "Cid"]  # paw photos aren't cats
    assert list(cats.portraits(portraits)) == ["Amy", "Bob", "Cid"]


def test_load_reads_paw_photos(tmp_path):
    _prepare(tmp_path, "Amy")
    cv2.imwrite(str(tmp_path / "Amy" / "paw_back_left.png"), np.zeros((8, 4, 4), np.uint8))
    assert list(cats.load("Amy", tmp_path).paws) == [("back", "left")]


def test_load_rejects_bad_landmarks(tmp_path):
    _prepare(tmp_path, "Odd", points_shape=(10, 3))
    with pytest.raises(ValueError, match="landmarks"):
        cats.load("Odd", tmp_path)


def test_load_rejects_missing_cutout(tmp_path):
    _prepare(tmp_path, "Gone")
    (tmp_path / "Gone" / "cutout.png").unlink()
    with pytest.raises(ValueError, match="cutout"):
        cats.load("Gone", tmp_path)


def test_shipped_cats(real_cats):
    names = ["Balint", "David", "Dori", "Laci", "Mate", "Milan", "TMarci", "VMarci"]
    assert [c.name for c in real_cats] == names
    for cat in real_cats:
        assert cat.sprite.shape[2] == 4
        assert cat.points.shape == (478, 3)
        h, w = cat.sprite.shape[:2]
        assert (cat.points[:, 0] >= 0).all() and (cat.points[:, 0] < w).all()
        assert (cat.points[:, 1] >= 0).all() and (cat.points[:, 1] < h).all()
        # The landmarks lie on the cat (the face outline can touch its edge).
        alpha = cat.sprite[cat.points[:, 1].astype(int), cat.points[:, 0].astype(int), 3]
        assert (alpha > 128).mean() > 0.95, cat.name
