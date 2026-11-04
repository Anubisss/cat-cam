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


def test_eye_outside_the_sprite_is_ignored(real_cats):
    cat = real_cats[0]
    points = cat.points[:, :2].copy()
    points[[i for contour in EYES[0] for i in contour], 0] = -50
    face = CatFace(cat.sprite, points)
    assert face.render((1.0, 1.0)).shape == cat.sprite.shape


def test_snapped_eyes_stay_near_the_landmarks(real_cats):
    from catcam.cat_face import MAX_CORNER_SHIFT, MAX_LID_SHIFT, snap_eyes

    for cat in real_cats:
        snapped = snap_eyes(cat.sprite[..., :3], cat.points[:, :2])
        for (upper, lower), (up_idx, lo_idx) in zip(snapped, EYES):
            width = np.ptp(cat.points[up_idx, 0])
            limit = (max(MAX_LID_SHIFT, MAX_CORNER_SHIFT) + 0.01) * width
            assert np.abs(upper - cat.points[up_idx, :2]).max() <= limit, cat.name
            assert np.abs(lower - cat.points[lo_idx, :2]).max() <= limit, cat.name
            assert (lower[1:-1, 1] >= upper[1:-1, 1]).all(), cat.name  # still an open eye
