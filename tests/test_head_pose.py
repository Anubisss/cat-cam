import math

import numpy as np

from catcam.head_pose import (
    MAX_ANGLE,
    OneEuroFilter,
    facing_camera,
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


def test_facing_camera_straight_ahead_is_no_rotation():
    matrix = np.eye(4)
    matrix[:3, 3] = [0, 0, -50]
    np.testing.assert_allclose(facing_camera(matrix), np.eye(3), atol=1e-12)


def test_facing_camera_looks_at_the_camera_upright():
    matrix = np.eye(4)
    matrix[:3, 3] = [10, 6, -40]  # MediaPipe space: right of and above the camera
    rotation = facing_camera(matrix)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) > 0
    position = np.diag([1.0, -1.0, -1.0]) @ matrix[:3, 3]  # in image coords
    # The face's front (toward the viewer, -z at rest) points at the camera...
    np.testing.assert_allclose(rotation @ [0, 0, -1], -position / np.linalg.norm(position), atol=1e-12)
    assert abs(rotation[1, 0]) < 1e-12  # ...and the head stays upright: its side axis is level


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
