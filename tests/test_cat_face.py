import numpy as np

from catcam.cat_face import EYES, CatFace


def test_open_eyes_leave_the_sprite_alone(real_cats):
    cat = real_cats[0]
    face = CatFace(cat.sprite, cat.points[:, :2])
    assert face.render((0.0, 0.0)) is cat.sprite


def test_blink_only_touches_the_eyes(real_cats):
    for cat in real_cats:
        face = CatFace(cat.sprite, cat.points[:, :2])
        closed = face.render((1.0, 1.0))
        changed = np.argwhere((closed != cat.sprite).any(axis=2))
        assert len(changed), cat.name
        eyes = cat.points[[i for pair in EYES for contour in pair for i in contour], :2]
        height = np.ptp(eyes[:, 1])
        assert changed[:, 0].min() > eyes[:, 1].min() - 4 * height, cat.name
        assert changed[:, 0].max() < eyes[:, 1].max() + 2 * height, cat.name
        # Closing the eye covers the eye: its middle now shows lid, not eye.
        for upper, lower in EYES:
            mid = ((cat.points[upper[4], :2] + cat.points[lower[4], :2]) / 2).astype(int)
            assert (closed[mid[1], mid[0]] != cat.sprite[mid[1], mid[0]]).any(), cat.name
