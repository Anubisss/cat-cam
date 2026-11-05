import numpy as np
import pytest

from catcam.voice import (
    BLOCK,
    MIN_BARK,
    MIN_GAP,
    PITCH_WINDOW,
    RATE,
    Syllables,
    Voice,
    estimate_pitch,
    find_device,
    pick_mic,
)

DT = BLOCK / RATE


def _starts(levels):
    """Times (s) at which syllables start, for one loudness level (dB) per block."""
    syllables = Syllables()
    return [i * DT for i, level in enumerate(levels) if syllables(level, DT) is not None]


def _blocks(seconds, level):
    return [level] * round(seconds / DT)


def test_each_syllable_starts_once():
    quiet = _blocks(0.3, -65)
    for _ in range(5):  # syllables of 120 ms with 100 ms pauses
        quiet += _blocks(0.12, -20) + _blocks(0.1, -65)
    starts = _starts(quiet)
    assert len(starts) == 5
    assert np.diff(starts).min() >= MIN_GAP


def test_connected_syllables_count_by_their_dips():
    levels = _blocks(0.3, -65)
    for _ in range(4):  # no pause, only a 10 dB dip between them
        levels += _blocks(0.15, -20) + _blocks(0.08, -30)
    assert len(_starts(levels)) == 4


def test_silence_hum_and_clicks_stay_quiet():
    rng = np.random.default_rng(0)
    assert _starts(-65 + rng.normal(0, 2, 300)) == []  # noisy silence
    assert _starts(_blocks(3.0, -30)) == []  # steady hum from the start
    clicks = _blocks(0.3, -65)
    for _ in range(5):
        clicks += [-15] + _blocks(0.2, -65)
    assert _starts(clicks) == []


def test_a_long_vowel_is_one_syllable():
    assert len(_starts(_blocks(0.3, -65) + _blocks(1.5, -20))) == 1


def test_louder_syllables_count_as_louder():
    found = []
    for level in (-40, -20):
        syllables = Syllables()
        found += [v for v in (syllables(x, DT) for x in _blocks(0.3, -65) + _blocks(0.1, level))
                  if v is not None]
    assert len(found) == 2 and found[1] > found[0]


def _speak(voice, amplitudes):
    rng = np.random.default_rng(1)
    return np.concatenate([voice.process((a * rng.normal(size=BLOCK)).astype(np.float32))
                           for a in amplitudes])


def _voice(seconds, pitch, amplitude=0.1):
    """A buzzy voice: a sawtooth at pitch (Hz)."""
    t = np.arange(round(seconds * RATE)) / RATE
    return (amplitude * (2 * (t * pitch % 1) - 1)).astype(np.float32)


def _say(voice, signal):
    """Feeds signal to voice; returns the output and the barks started (samples arrays)."""
    out, started = [], []
    for i in range(0, len(signal), BLOCK):
        before = voice.barks[-1] if voice.barks else None
        out.append(voice.process(signal[i:i + BLOCK]))
        if voice.barks and voice.barks[-1] is not before:
            started.append(voice.barks[-1].samples)
    return np.concatenate(out), started


def _dog(rng_seed=0):
    voice = Voice(np.random.default_rng(rng_seed))
    voice.sounds = [np.sin(np.linspace(0, 400, int(0.35 * RATE))).astype(np.float32)]
    return voice


def _sounding(out):
    """s from the first to the last audible sample."""
    loud = np.flatnonzero(np.abs(out) > 1e-3)
    return (loud[-1] - loud[0]) / RATE if len(loud) else 0.0


def test_finds_the_pitch_of_a_voice():
    for pitch in (90, 140, 220, 330):
        assert estimate_pitch(_voice(PITCH_WINDOW, pitch)) == pytest.approx(pitch, rel=0.03)
    hiss = np.random.default_rng(0).normal(0, 0.1, int(PITCH_WINDOW * RATE))
    assert estimate_pitch(hiss) is None


def test_barks_follow_your_pitch():
    pause = np.zeros(int(0.2 * RATE), np.float32)
    usual = np.concatenate([np.concatenate([_voice(0.2, 120), pause]) for _ in range(4)])
    voice = _dog()
    _, normal = _say(voice, usual)
    _, high = _say(voice, np.concatenate([_voice(0.2, 180), pause]))
    _, low = _say(voice, np.concatenate([_voice(0.2, 90), pause]))
    # A higher bark is played faster, so it is shorter.
    assert len(high[0]) < 0.85 * np.median([len(b) for b in normal[1:]])
    assert len(low[0]) > 1.1 * np.median([len(b) for b in normal[1:]])


def test_barks_last_as_long_as_your_syllables():
    pause = np.zeros(int(0.5 * RATE), np.float32)
    short, _ = _say(_dog(), np.concatenate([pause, _voice(0.06, 120), pause]))
    held, _ = _say(_dog(), np.concatenate([pause, _voice(0.5, 120), pause]))
    assert MIN_BARK <= _sounding(short) < 0.16
    assert _sounding(held) > 0.3  # the whole bark


def test_the_next_syllable_cuts_the_bark_short():
    voice = _dog()
    # Quick connected syllables: 10 dB dips between them, no pauses.
    quick = np.concatenate([np.zeros(int(0.3 * RATE), np.float32)]
                           + [_voice(0.07, 120) * (1 if i % 2 == 0 else 0.3) for i in range(12)])
    _, started = _say(voice, quick)
    assert len(started) >= 4
    assert len(voice.barks) <= 2  # the earlier ones were faded out, not stacked up


def test_without_sounds_your_voice_passes_through():
    voice = Voice()
    block = np.random.default_rng(0).normal(0, 0.1, BLOCK).astype(np.float32)
    np.testing.assert_array_equal(voice.process(block), block)


def test_syllables_become_barks_as_loud_as_you_spoke():
    bark = np.sin(np.linspace(0, 200, RATE // 4)).astype(np.float32)
    outputs = []
    for amplitude in (0.01, 0.2):
        voice = Voice(np.random.default_rng(0))
        voice.sounds = [bark]
        out = _speak(voice, [0.0005] * 30 + [amplitude] * 12 + [0.0005] * 30)
        assert np.abs(out[:30 * BLOCK]).max() == 0  # nothing before you speak
        outputs.append(np.abs(out).max())
    assert 0 < outputs[0] < outputs[1] <= 1


def test_finds_devices_by_name_or_number():
    devices = [
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "AirPods", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Brio 500", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
    ]
    assert find_device(devices, "blackhole", "output") == 3
    assert find_device(devices[:3], "BlackHole", "output") is None
    assert find_device(devices, "brio", "input") == 2
    assert find_device(devices, "2", "input") == 2  # a number is never part of a name ("2ch")
    assert find_device(devices, "1", "input") is None  # the AirPods have no microphone here


def test_never_listens_to_the_virtual_microphone():
    devices = [
        {"name": "iPhone Microphone", "max_input_channels": 1},
        {"name": "BlackHole 2ch", "max_input_channels": 2},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        {"name": "MacBook Pro Speakers", "max_input_channels": 0},
    ]
    assert pick_mic(devices, 0) == 0  # the default, if it is a real one
    assert pick_mic(devices, 1) == 2  # the built-in one instead of BlackHole
    assert pick_mic(devices[:2], 1) == 0
    assert pick_mic(devices[1:2], 0) is None


def test_only_the_dog_barks(real_cats):
    for cat in real_cats:
        assert bool(cat.sounds) == (cat.name == "Goldie"), cat.name
    goldie = next(cat for cat in real_cats if cat.name == "Goldie")
    assert len(goldie.sounds) == 1  # one dog
    for sound in goldie.sounds:
        assert sound.dtype == np.float32 and sound.ndim == 1
        assert 0.1 * RATE < len(sound) < RATE
