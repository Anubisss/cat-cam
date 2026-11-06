import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "cat_voices.py"
spec = importlib.util.spec_from_file_location("cat_voices", TOOL)
cat_voices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cat_voices)
RATE = cat_voices.RATE


def test_every_cat_has_a_voice_from_credited_sources(real_cats):
    assert set(cat_voices.VOICES) == {cat.name for cat in real_cats} - {"Woofgang"}
    for name, takes in cat_voices.VOICES.items():
        for take in takes:
            assert take.source in cat_voices.SOURCES, name  # so its provenance is written down
            assert 0 <= take.start < take.end, name


def _source(seconds=1.0, pitch=500.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return {1: 0.3 * np.sin(2 * np.pi * pitch * t)}


def test_a_take_is_normalised_and_fades_in_and_out():
    sound = cat_voices.make(cat_voices.Take(1, 0.1, 0.9), _source())
    assert sound.dtype == np.float32
    assert np.abs(sound).max() == pytest.approx(cat_voices.PEAK, abs=1e-3)
    assert abs(sound[0]) < 1e-3 and abs(sound[-1]) < 1e-3  # no clicks


def test_higher_is_shorter_and_lower_is_longer():
    def length(**character):
        return len(cat_voices.make(cat_voices.Take(1, 0.0, 1.0, **character), _source()))

    assert length(semitones=12) == pytest.approx(RATE / 2, rel=0.01)  # an octave up: twice as fast
    assert length(semitones=-7) > 1.4 * RATE
    assert length(longest=0.3) == int(0.3 * RATE)


def test_a_bend_glides_the_pitch():
    rising = cat_voices.make(cat_voices.Take(1, 0.0, 1.0, bend=12), _source())
    half = len(rising) // 2

    def crossings(x):
        return np.count_nonzero(np.diff(np.sign(x)) > 0) / (len(x) / RATE)

    # The second half is pitched higher than the first.
    assert crossings(rising[half:]) > 1.3 * crossings(rising[:half])
