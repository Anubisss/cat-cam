import cv2
import numpy as np
import pytest

from catcam.app import Controls, Mouse, _follow_switch
from catcam.background import Background, behind, cover
from catcam.picker import Picker

SIZE = (64, 36)


def test_cover_fills_and_crops_the_middle():
    tall = np.zeros((200, 100, 3), np.uint8)
    tall[95:105] = 255  # a stripe across the middle
    out = cover(tall, SIZE)
    assert out.shape == (36, 64, 3)
    assert out[18].mean() > 200 and out[0].mean() < 10


def test_a_picture_stays_put(tmp_path):
    path = tmp_path / "beach.png"
    cv2.imwrite(str(path), np.full((90, 160, 3), 120, np.uint8))
    background = Background(path, SIZE)
    assert background.frame(0.0).shape == (36, 64, 3)
    assert background.frame(5.0) is background.frame(0.0)


def test_a_video_plays_and_loops(tmp_path):
    path = tmp_path / "waves.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
    for shade in (0, 100, 200):
        writer.write(np.full((36, 64, 3), shade, np.uint8))
    writer.release()
    background = Background(path, SIZE)
    shades = [int(background.frame(t).mean()) for t in (0.0, 0.11, 0.21, 0.31)]
    assert shades[0] < 30 and 70 < shades[1] < 130 and shades[2] > 170
    assert shades[3] < 30  # back to the start
    background.close()


def test_unreadable_files_are_refused(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a picture")
    with pytest.raises(ValueError, match="notes.txt"):
        Background(path, SIZE)


def test_behind_mixes_by_coverage():
    frame = np.full((2, 2, 3), 200, np.uint8)
    background = np.zeros((2, 2, 3), np.uint8)
    coverage = np.array([[255, 0], [128, 255]], np.uint8)
    out = behind(frame, coverage, background)
    assert out[0, 0, 0] == 200 and out[0, 1, 0] == 0 and 95 < out[1, 0, 0] < 105


class _Chooser:
    def __init__(self, answers):
        self.answers, self.started = list(answers), 0

    @property
    def idle(self):
        return self.started == 0

    def start(self):
        self.started += 1

    def answer(self):
        return self.answers.pop(0)


def _controls():
    sprite = np.full((20, 20, 4), 255, np.uint8)
    picker = Picker([("A", sprite, np.zeros(3, np.uint8))])
    return Controls(picker, Mouse(picker))


def test_the_switch_asks_for_a_file_and_goes_off_if_none_is_chosen():
    controls = _controls()
    controls.background = True
    chooser = _Chooser([None, ""])  # still choosing, then cancelled
    assert _follow_switch(controls, chooser, None) is None and chooser.started == 1
    assert _follow_switch(controls, chooser, None) is None and controls.background
    assert _follow_switch(controls, chooser, None) is None and not controls.background


def test_the_switch_loads_the_chosen_file_and_drops_it_when_off(tmp_path):
    path = tmp_path / "beach.png"
    cv2.imwrite(str(path), np.full((90, 160, 3), 120, np.uint8))
    controls = _controls()
    controls.background = True
    chooser = _Chooser([str(path)])
    chooser.started = 1  # already asked
    background = _follow_switch(controls, chooser, None)
    assert isinstance(background, Background)
    assert _follow_switch(controls, chooser, background) is background
    controls.background = False
    assert _follow_switch(controls, chooser, background) is None
