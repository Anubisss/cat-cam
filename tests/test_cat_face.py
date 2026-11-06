import numpy as np

from catcam.cat_face import EYES, CatFace, _sample


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
        # Measured on the eyes as snapped onto the photo: the raw landmarks can
        # squash a narrowed eye into a slit.
        eyes = np.vstack([contour for pair in face.eyes for contour in pair])
        height = np.ptp(eyes[:, 1])
        assert changed[:, 0].min() > eyes[:, 1].min() - 4 * height, cat.name
        assert changed[:, 0].max() < eyes[:, 1].max() + 2 * height, cat.name
        # Closing the eye covers the eye: round its middle it now shows lid, not eye
        # (a black cat's lid can match the pixel right in the middle of a squint).
        for upper, lower in face.eyes:
            x = upper[:, 0].mean()
            y = (_sample(upper, [x])[0] + _sample(lower, [x])[0]) / 2
            x, y = int(x), int(y)
            window = np.s_[y - 2:y + 3, x - 2:x + 3]
            assert (closed[window] != cat.sprite[window]).any(), cat.name


def test_eye_outside_the_sprite_is_ignored(real_cats):
    cat = real_cats[0]
    points = cat.points[:, :2].copy()
    points[[i for contour in EYES[0] for i in contour], 0] = -50
    face = CatFace(cat.sprite, points)
    assert face.render((1.0, 1.0)).shape == cat.sprite.shape


def test_snapped_eyes_stay_near_the_landmarks(real_cats):
    from catcam.cat_face import MAX_CORNER_SHIFT, MAX_LID_SHIFT, _sample, snap_eyes

    for cat in real_cats:
        snapped = snap_eyes(cat.sprite[..., :3], cat.points[:, :2])
        for (upper, lower), (up_idx, lo_idx) in zip(snapped, EYES):
            width = np.ptp(cat.points[up_idx, 0])
            limit = (max(MAX_LID_SHIFT, MAX_CORNER_SHIFT) + 0.01) * width
            assert np.abs(upper - cat.points[up_idx, :2]).max() <= limit, cat.name
            assert np.abs(lower - cat.points[lo_idx, :2]).max() <= limit, cat.name
            # Still an open eye (or a squint, at least not shut past itself).
            assert (lower[1:-1, 1] >= _sample(upper, lower[1:-1, 0]) - 1e-6).all(), cat.name


def test_blinks_stay_in_the_lid_box(real_cats):
    for cat in real_cats:
        face = CatFace(cat.sprite, cat.points[:, :2])
        x0, y0, x1, y1 = face.lid_box()
        for blink in [(0.1, 0.1), (0.5, 1.0), (1.0, 0.3), (1.0, 1.0)]:
            changed = (face.render(blink) != cat.sprite).any(axis=2)
            assert changed.any(), cat.name
            changed[y0:y1, x0:x1] = False
            assert not changed.any(), cat.name
