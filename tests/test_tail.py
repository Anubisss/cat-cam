import numpy as np

from catcam.tail import LENGTH, SEGMENTS, Tail, TailSkin, tail_skin

W, H = 1280, 720
FACE_CENTER, FACE_WIDTH = np.array([640.0, 360.0]), 450.0


def _skin():
    image = np.zeros((100, 64, 3), np.uint8)
    image[..., 2] = 255
    return TailSkin(image, 60.0)


def _run(tail, hand, seconds):
    for _ in range(int(seconds * 30)):
        tail.update(hand, 0.0 if hand is None else 1.0, 1 / 30, FACE_CENTER, FACE_WIDTH)


def test_tail_skin_is_davids_arm(real_cats):
    david = next(c for c in real_cats if c.name == "David")
    skin = tail_skin(david.paws)
    assert skin.image.ndim == 3 and skin.image.shape[1] == 64 and len(skin.image) > 50
    assert skin.image.mean() > 80  # skin, not the transparent background
    assert tail_skin({}) is None


def test_tail_rises_to_the_hand_and_sinks_without_it():
    tail, frame = Tail((W, H)), np.zeros((H, W, 3), np.uint8)
    assert tail.draw(frame, _skin()) is frame
    hand = np.array([1000.0, 250.0])
    _run(tail, hand, 2.0)
    assert np.linalg.norm(tail.points[-1] - hand) < 25
    segments = np.linalg.norm(np.diff(tail.points, axis=0), axis=1)
    np.testing.assert_allclose(segments, LENGTH * H / SEGMENTS, rtol=0.02)  # it doesn't stretch
    drawn = tail.draw(frame, _skin())
    rows, columns = np.nonzero(drawn[..., 2])
    assert rows.max() == H - 1  # it comes up from under the screen
    assert columns.mean() > W / 2  # on the hand's side
    assert np.hypot(columns - hand[0], rows - hand[1]).min() < 25

    _run(tail, None, 3.0)
    assert not tail.draw(frame, _skin()).any()


def test_tail_bends_and_trails_behind_a_moving_hand():
    tail = Tail((W, H))
    _run(tail, np.array([1050.0, 300.0]), 2.0)
    tail.update(np.array([500.0, 150.0]), 1.0, 1 / 30, FACE_CENTER, FACE_WIDTH)
    tail.update(np.array([500.0, 150.0]), 1.0, 1 / 30, FACE_CENTER, FACE_WIDTH)
    p = tail.points
    ux, uy = (p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0])
    off_line = np.abs((p[:, 0] - p[0, 0]) * uy - (p[:, 1] - p[0, 1]) * ux)
    assert off_line.max() > 30  # curved, not a straight stick
    assert p[-1, 0] > 600  # the tip is still on its way


def test_tail_rides_out_a_few_frames_without_the_hand():
    from catcam.paws import FADE_RATE

    tail = Tail((W, H))
    hand = np.array([1000.0, 250.0])
    _run(tail, hand, 2.0)
    tip = tail.points[-1].copy()
    presence = 1.0
    for _ in range(4):  # the tracker misses the hand for 4 frames
        presence += FADE_RATE * (0 - presence)
        tail.update(hand, presence, 1 / 30, FACE_CENTER, FACE_WIDTH)
    assert np.linalg.norm(tail.points[-1] - tip) < 10  # the tail stays up
