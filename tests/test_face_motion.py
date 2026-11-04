import numpy as np

from catcam.face_motion import STABLE, ExpressionTransfer, similarity
from catcam.head_pose import to_matrix

CHIN = 152


def _face(rng):
    return rng.normal(size=(478, 3)) * [100, 120, 40] + [600, 400, 0]


def _move(points, scale, rotation, shift):
    return scale * points @ rotation.T + shift


def test_similarity_recovers_transform(rng):
    src = rng.normal(size=(50, 3))
    rotation = to_matrix(np.array([0.3, -0.2, 0.5]))
    s, r, t = similarity(src, _move(src, 2.5, rotation, [1.0, 2.0, 3.0]))
    assert np.isclose(s, 2.5)
    np.testing.assert_allclose(r, rotation, atol=1e-9)
    np.testing.assert_allclose(t, [1.0, 2.0, 3.0], atol=1e-9)


def test_neutral_face_has_no_expression(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    np.testing.assert_allclose(transfer(face, relaxed=False), 0, atol=1e-9)


def test_head_movement_is_not_expression(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    transfer(face, relaxed=False)
    moved = _move(face, 0.7, to_matrix(np.array([0.1, 0.4, -0.2])), [50.0, -20.0, 5.0])
    np.testing.assert_allclose(transfer(moved, relaxed=False), 0, atol=1e-9)


def test_jaw_drop_is_transferred_in_face_widths(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    transfer(face, relaxed=False)
    open_mouth = face.copy()
    open_mouth[CHIN, 1] += 30
    offsets = transfer(open_mouth, relaxed=False)
    np.testing.assert_allclose(offsets[CHIN, 1], 30 / np.ptp(face[:, 0]), rtol=1e-6)
    assert np.abs(np.delete(offsets, CHIN, axis=0)).max() < 1e-9


def test_to_cat_scales_into_the_cat(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    transfer(face, relaxed=False)
    cat = face * 2  # same shape, twice the size
    open_mouth = face.copy()
    open_mouth[CHIN, 1] += 30
    on_cat = transfer.to_cat(transfer(open_mouth, relaxed=False), cat)
    assert np.isclose(on_cat[CHIN, 1], 60)


def test_relaxed_face_slowly_becomes_neutral(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    transfer(face, relaxed=True)
    smile = face.copy()
    smile[61, 1] -= 10
    first = transfer(smile, relaxed=True)[61, 1]
    for _ in range(200):
        last = transfer(smile, relaxed=True)[61, 1]
    assert first < last and abs(last) < abs(first) / 100


def test_reset_takes_next_face_as_neutral(rng):
    face = _face(rng)
    transfer = ExpressionTransfer()
    transfer(face, relaxed=False)
    other = face + rng.normal(size=face.shape)
    transfer.reset()
    np.testing.assert_allclose(transfer(other, relaxed=False), 0, atol=1e-9)
    assert len(STABLE) >= 4
