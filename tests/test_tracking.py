"""How FaceState follows a face: blinks, resting scores, losing the face."""

from types import SimpleNamespace

import numpy as np

from catcam.app import BLINK_RANGE, EASE_BACK, KEEP_POSE, RELAX, FaceState, blinks
from catcam.cat_face import EYES
from catcam.face_motion import REST_FALL, REST_RISE, RestingScores
from catcam.head_pose import to_matrix

DT = 1 / 15  # a webcam in a dim room


def _face(rng):
    """A face's landmarks (normalized image coords) with open eyes, and a
    matrix 50 cm in front of the camera."""
    points = rng.normal(size=(478, 3)) * [0.05, 0.06, 0.02] + [0.5, 0.5, 0.0]
    for upper, lower in EYES:  # each eye 0.05 wide, its lower lid 0.01 below the upper
        start = points[upper[0], :2]
        points[upper, 0] = points[lower, 0] = np.linspace(0, 0.05, len(upper)) + start[0]
        points[upper, 1] = start[1]
        points[lower, 1] = start[1] + 0.01
        points[[lower[0], lower[-1]], 1] = start[1]
    matrix = np.eye(4)
    matrix[:3, 3] = [0, 0, -50]
    return points, matrix


def _result(points, matrix, **scores):
    """A face tracker's result for one face; blendshapes not given score 0.1."""
    names = ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
             "mouthFunnel", "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
             "browOuterUpRight", "browDownLeft", "browDownRight", "cheekPuff"]
    return SimpleNamespace(
        face_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]],
        facial_transformation_matrixes=[matrix],
        face_blendshapes=[[SimpleNamespace(category_name=n, score=scores.get(n, 0.1)) for n in names]],
    )


def test_blinks_count_from_the_resting_score():
    # Someone whose eye scores 0.5 at rest: that is open, and a blink is the rise above it.
    lo, hi = BLINK_RANGE
    rest = {"eyeBlinkRight": 0.5, "eyeBlinkLeft": 0.1}
    open_eyes = {"eyeBlinkRight": 0.5, "eyeBlinkLeft": 0.1}
    np.testing.assert_allclose(blinks(open_eyes, rest=rest), 0)
    shut = {"eyeBlinkRight": 0.5 + hi, "eyeBlinkLeft": 0.1 + hi}
    np.testing.assert_allclose(blinks(shut, rest=rest), 1)


def test_resting_score_falls_fast_and_rises_slowly():
    resting = RestingScores()
    resting.update({"mouthPucker": 0.4}, DT)
    for _ in range(round(REST_FALL / DT * 3)):
        level = resting.update({"mouthPucker": 0.1}, DT)["mouthPucker"]
    assert level < 0.13
    for _ in range(round(3 / DT)):  # three seconds of a pucker barely count
        level = resting.update({"mouthPucker": 0.8}, DT)["mouthPucker"]
    assert level < 0.1 + 0.7 * 3 / REST_RISE * 1.1


def test_a_one_frame_blink_closes_the_lids_fully_then_they_open(rng):
    points, matrix = _face(rng)
    state = FaceState()
    for k in range(20):
        state.update(_result(points, matrix), (1280, 720), DT, k * DT)
    state.update(_result(points, matrix, eyeBlinkLeft=0.9, eyeBlinkRight=0.9), (1280, 720), DT, 20 * DT)
    np.testing.assert_allclose(state.blink, 1)
    for k in range(21, 26):
        state.update(_result(points, matrix), (1280, 720), DT, k * DT)
    assert state.blink.max() < 0.05


def test_a_pucker_at_rest_still_lets_the_neutral_face_adapt(rng):
    # Before, any blendshape over 0.25 (here a resting pucker of 0.4) kept the
    # neutral face from ever adapting, so an old expression stayed on the cat.
    points, matrix = _face(rng)
    state = FaceState()
    state.update(_result(points, matrix, mouthPucker=0.4), (1280, 720), DT, 0.0)
    moved = points.copy()
    moved[61, 1] += 0.01  # a mouth corner sits a little lower from now on
    for k in range(1, 200):
        state.update(_result(moved, matrix, mouthPucker=0.4), (1280, 720), DT, k * DT)
    assert abs(state.offsets[61, 1]) < 0.002


def test_talking_never_becomes_the_neutral_face(rng):
    # Between syllables your mouth is often still part open, and jawOpen then
    # rises only a little above its resting score. Taken for your neutral face,
    # that made the animal's mouth open less and less as you talked.
    points, matrix = _face(rng)
    state = FaceState()
    state.update(_result(points, matrix, jawOpen=0.03), (1280, 720), DT, 0.0)
    lower = [14, 87, 178, 317, 402, 17, 84, 314]
    width = np.ptp(points[:, 0])
    for k in range(1, round(20 / DT)):  # 20 s of syllables, 4 a second
        opening = 0.05 * width * abs(np.sin(np.pi * 4 * k * DT))
        talking = points.copy()
        talking[lower, 1] += opening
        state.update(_result(talking, matrix, jawOpen=0.03 + 4 * opening / width), (1280, 720), DT,
                     k * DT, talking=True)
    shut = _result(points, matrix, jawOpen=0.03)
    state.update(shut, (1280, 720), DT, 20.0, talking=True)
    # A closed mouth is still closed, not pulled up above where it rests.
    assert state.measured[1][14, 1] > -0.004


def test_still_quiet_face_becomes_neutral_after_talking(rng):
    points, matrix = _face(rng)
    state = FaceState()
    state.update(_result(points, matrix), (1280, 720), DT, 0.0)
    moved = points.copy()
    moved[61, 1] += 0.01  # a mouth corner sits a little lower from now on
    for k in range(1, 200):
        state.update(_result(moved, matrix), (1280, 720), DT, k * DT)
    assert abs(state.offsets[61, 1]) < 0.002


def _turned(matrix, yaw):
    """The matrix with the head turned by yaw radians (MediaPipe's axes)."""
    turned = matrix.copy()
    turned[:3, :3] = to_matrix(np.array([0.0, yaw, 0.0]))
    return turned


def _no_face():
    return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[], face_blendshapes=[])


def test_a_lost_face_keeps_its_pose_then_relaxes(rng):
    points, matrix = _face(rng)
    state = FaceState()
    for k in range(30):
        state.update(_result(points, _turned(matrix, 0.4)), (1280, 720), DT, k * DT)
    turned = state.head.copy()
    assert np.linalg.norm(turned) > 0.35
    t = 30 * DT
    while t < 30 * DT + KEEP_POSE - DT:
        state.update(_no_face(), (1280, 720), DT, t)
        t += DT
    np.testing.assert_allclose(state.head, turned)
    for _ in range(round(4 * RELAX / DT)):
        state.update(_no_face(), (1280, 720), DT, t)
        t += DT
    assert np.linalg.norm(state.head) < 0.05 * np.linalg.norm(turned)


def test_a_face_coming_back_elsewhere_eases_over_instead_of_jumping(rng):
    points, matrix = _face(rng)
    state = FaceState()
    t = 0.0
    for _ in range(30):
        state.update(_result(points, matrix), (1280, 720), DT, t)
        t += DT
    for _ in range(3):  # gone for a moment...
        state.update(_no_face(), (1280, 720), DT, t)
        t += DT
    steps, last = [], state.head.copy()
    for _ in range(round(2 * EASE_BACK / DT)):  # ...and back turned 0.5 rad
        state.update(_result(points, _turned(matrix, 0.5)), (1280, 720), DT, t)
        steps.append(np.linalg.norm(state.head - last))
        last, t = state.head.copy(), t + DT
    assert max(steps) < 0.5 * 0.6  # no single frame takes most of the turn
    assert abs(np.linalg.norm(state.head) - 0.5) < 0.02  # and it gets there
