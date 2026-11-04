import math

import numpy as np

from catcam.head_pose import (
    MAX_ANGLE,
    OneEuroFilter,
    relative_rotation,
    rotation_of,
    to_matrix,
    to_vector,
)


def test_rotation_vector_round_trip():
    vec = np.array([0.1, -0.3, 0.2])
    np.testing.assert_allclose(to_vector(to_matrix(vec)), vec, atol=1e-9)


def test_rotation_is_clamped():
    vec = to_vector(to_matrix(np.array([0.0, 2.0, 0.0])))
    assert math.isclose(np.linalg.norm(vec), MAX_ANGLE)


def test_relative_rotation_of_same_pose_is_identity():
    pose = to_matrix(np.array([0.2, 0.1, -0.4]))
    np.testing.assert_allclose(relative_rotation(pose, pose), np.eye(3), atol=1e-12)


def test_relative_rotation_recovers_turn():
    reference = to_matrix(np.array([0.2, 0.0, 0.0]))
    turn = to_matrix(np.array([0.0, 0.3, 0.0]))
    np.testing.assert_allclose(relative_rotation(turn @ reference, reference), turn, atol=1e-12)


def test_rotation_of_flips_into_image_coordinates():
    # A yaw in MediaPipe's y-up/z-toward-camera space is the opposite yaw in
    # image space (y down, z away); scale in the matrix is ignored.
    matrix = np.eye(4)
    matrix[:3, :3] = 1.7 * to_matrix(np.array([0.0, 0.3, 0.0]))
    np.testing.assert_allclose(to_vector(rotation_of(matrix)), [0.0, -0.3, 0.0], atol=1e-9)


def test_one_euro_keeps_a_constant_signal():
    f = OneEuroFilter(1.0, 0.5)
    for _ in range(10):
        out = f(np.array([3.0, -1.0]), 1 / 30)
    np.testing.assert_allclose(out, [3.0, -1.0])


def test_one_euro_smooths_then_converges_on_a_step():
    f = OneEuroFilter(1.0, 0.0)
    f(np.zeros(1), 1 / 30)
    first = f(np.ones(1), 1 / 30)[0]
    assert 0 < first < 0.5
    for _ in range(300):
        out = f(np.ones(1), 1 / 30)[0]
    assert out > 0.999
