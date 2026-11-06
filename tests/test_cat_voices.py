import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "cat_voices.py"
spec = importlib.util.spec_from_file_location("cat_voices", TOOL)
cat_voices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cat_voices)
RATE = cat_voices.RATE


def test_every_animal_has_a_voice_from_credited_sources(real_cats):
    assert set(cat_voices.VOICES) == {cat.name for cat in real_cats}
    for name, takes in cat_voices.VOICES.items():
        for take in takes:
            assert take.source in cat_voices.SOURCES, name  # so its provenance is written down
            assert 0 <= take.start < take.end, name


def _meow(pitch=500.0):
    """A recording: 0.5 s of quiet, a 0.6 s meow that swells and fades, 0.5 s of quiet."""
    t = np.arange(int(0.6 * RATE)) / RATE
    meow = 0.3 * np.sin(np.pi * t / t[-1]) * np.sin(2 * np.pi * pitch * t)
    quiet = np.zeros(RATE // 2)
    noise = np.random.default_rng(0).normal(0, 1e-4, RATE + len(meow))  # a quiet room
    return {1: np.concatenate([quiet, meow, quiet]) + noise}


def _crossings(x):
    return np.count_nonzero(np.diff(np.sign(x)) > 0) / (len(x) / RATE)


def test_a_take_is_the_sound_as_recorded():
    sound = cat_voices.make(cat_voices.Take(1, 0.0, 1.6), _meow())
    assert sound.dtype == np.float32
    assert np.abs(sound).max() == pytest.approx(cat_voices.PEAK, abs=1e-3)
    # The quiet around it is left out, its natural tail is kept...
    assert 0.6 < len(sound) / RATE < 0.6 + cat_voices.LEAD + cat_voices.TAIL + 0.01
    # ...its pitch and speed are untouched...
    assert _crossings(sound[len(sound) // 4:-len(sound) // 4]) == pytest.approx(500, rel=0.02)
    # ...and it fades in and out, so it doesn't click.
    assert abs(sound[0]) < 1e-3 and abs(sound[-1]) < 1e-3


def test_a_take_cut_into_a_sound_is_refused():
    sources = _meow()
    assert cat_voices.check(cat_voices.Take(1, 0.3, 1.4), sources) is None
    assert "starts" in cat_voices.check(cat_voices.Take(1, 0.8, 1.4), sources)
    assert "ends" in cat_voices.check(cat_voices.Take(1, 0.3, 0.9), sources)


def test_digital_silence_is_trimmed_before_the_fade():
    sources = _meow()
    sources[1][int(0.95 * RATE):] = 0.0  # the recording was edited to silence mid-meow
    sound = cat_voices.make(cat_voices.Take(1, 0.3, 1.6), sources)
    assert len(sound) / RATE < 0.5  # ends where the meow was cut...
    assert abs(sound[-1]) < 1e-3  # ...but fades there instead of stopping dead
