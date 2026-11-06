import numpy as np
import pytest

from catcam.voice import (
    BLOCK,
    GAP_SPREAD,
    LEVEL,
    MIN_GAP,
    PITCH_LIMIT,
    PITCH_WINDOW,
    RATE,
    STYLE_FILE,
    Sounds,
    Style,
    Syllables,
    Voice,
    estimate_pitch,
    find_device,
    load_sounds,
    load_style,
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


def _say(voice, signal, when=None):
    """Feeds signal to voice; returns the output and the sounds started (samples
    arrays), and adds the times (s) they started at to `when`."""
    out, started = [], []
    for i in range(0, len(signal), BLOCK):
        before = voice.bark
        out.append(voice.process(signal[i:i + BLOCK]))
        if voice.bark is not None and voice.bark is not before:
            started.append(voice.bark.samples)
            if when is not None:
                when.append(i / RATE)
    return np.concatenate(out), started


def _tone(seconds):
    return np.sin(np.linspace(0, 2 * np.pi * 400 * seconds, int(seconds * RATE))).astype(np.float32)


def _dog(rng_seed=0, style=None):
    voice = Voice(np.random.default_rng(rng_seed))
    voice.sounds = Sounds((_tone(0.35),), style or Style())
    return voice


def _syllables(count, seconds=0.1, pause=0.1, pitch=120):
    """count syllables of your voice with pauses between them, after a pause."""
    gap = np.zeros(int(pause * RATE), np.float32)
    return np.concatenate([np.zeros(int(0.3 * RATE), np.float32)]
                          + [np.concatenate([_voice(seconds, pitch), gap]) for _ in range(count)])


def _sounding(out):
    """s from the first to the last audible sample."""
    loud = np.flatnonzero(np.abs(out) > 1e-3)
    return (loud[-1] - loud[0]) / RATE if len(loud) else 0.0


def test_finds_the_pitch_of_a_voice():
    for pitch in (90, 140, 220, 330):
        assert estimate_pitch(_voice(PITCH_WINDOW, pitch)) == pytest.approx(pitch, rel=0.03)
    hiss = np.random.default_rng(0).normal(0, 0.1, int(PITCH_WINDOW * RATE))
    assert estimate_pitch(hiss) is None


def test_sounds_follow_your_pitch_a_little():
    voice = Voice(np.random.default_rng(0))
    voice.sounds = Sounds((_tone(0.1),), Style(jitter=0, gap=0))
    _, normal = _say(voice, _syllables(4, pause=0.2, pitch=120))
    _, high = _say(voice, _syllables(1, pause=0.2, pitch=180))
    _, low = _say(voice, _syllables(1, pause=0.2, pitch=90))
    _, squeal = _say(voice, _syllables(1, pause=0.2, pitch=390))
    usual = np.median([len(s) for s in normal[1:]])
    # A higher sound is played faster, so it is shorter...
    assert len(high[0]) < 0.97 * usual and len(low[0]) > 1.03 * usual
    # ...but never by more than PITCH_LIMIT, however high you go, or it sounds like a robot.
    shortest = len(_tone(0.1)) * 2 ** (-PITCH_LIMIT / 12)
    assert all(len(s) >= shortest - 2 for s in normal + high + low + squeal)


def test_a_sound_always_plays_to_its_end():
    pause = np.zeros(int(0.5 * RATE), np.float32)
    voice = Voice(np.random.default_rng(0))
    voice.sounds = Sounds((_tone(1.0),), Style(jitter=0, pitch_follow=0))
    out, started = _say(voice, np.concatenate([pause, _voice(0.06, 120), pause, pause]))
    assert len(started) == 1
    assert _sounding(out) == pytest.approx(len(started[0]) / RATE, abs=0.01)  # not cut when you stop


def test_quick_syllables_never_cut_or_stack_sounds():
    voice = _dog(style=Style(gap=0))
    talk = np.concatenate([_syllables(20, seconds=0.07, pause=0.06), np.zeros(RATE // 2, np.float32)])
    out, started = _say(voice, talk)
    assert 3 <= len(started) < 20  # syllables come quicker than barks can, so some pass
    # One at a time and each whole: the output is just the sounds, one after another.
    assert np.sum(out ** 2) == pytest.approx(sum(np.sum(s ** 2) for s in started), rel=1e-4)
    assert np.abs(out).max() <= np.abs(np.concatenate(started)).max() + 1e-6


def test_the_animal_waits_its_gap_between_sounds():
    talk = _syllables(25, pause=0.08)  # 4.5 s of steady talk
    counts = []
    for gap in (0.0, 1.0):
        voice, when = _dog(style=Style(gap=gap)), []
        _, started = _say(voice, talk, when)
        counts.append(len(started))
        spacing = np.diff(when)
        assert spacing.min() >= 0.35 + GAP_SPREAD[0] * gap - 0.01  # the sound, then the gap
    assert counts[0] >= 8 and 2 <= counts[1] <= 4  # a chatty dog, a cat that takes its time


def test_without_sounds_your_voice_passes_through():
    voice = Voice()
    block = np.random.default_rng(0).normal(0, 0.1, BLOCK).astype(np.float32)
    np.testing.assert_array_equal(voice.process(block), block)


def test_syllables_become_barks_as_loud_as_you_spoke():
    bark = np.sin(np.linspace(0, 200, RATE // 4)).astype(np.float32)
    outputs = []
    for amplitude in (0.01, 0.2):
        voice = Voice(np.random.default_rng(0))
        voice.sounds = Sounds((bark,))
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


def test_every_animal_has_its_own_voice(real_cats):
    styles = set()
    for cat in real_cats:
        clips = cat.sounds.clips
        assert len(clips) >= 5, cat.name  # several, so it doesn't repeat itself
        if cat.name == "Woofgang":
            assert cat.sounds.style == Style()  # barks, the default style
        else:
            styles.add(cat.sounds.style)
        for sound in clips:
            assert sound.dtype == np.float32 and sound.ndim == 1
            assert 0.15 * RATE < len(sound) < 2 * RATE, cat.name
    assert len(styles) == len(real_cats) - 1  # every cat follows your voice in its own way


def test_voices_are_small(real_cats):
    from catcam.paths import ASSETS

    for cat in real_cats:
        files = list((ASSETS / cat.name / "voice").iterdir())
        assert {f.suffix for f in files} <= {".ogg", ".toml"}, cat.name
        assert sum(f.stat().st_size for f in files) < 150_000, cat.name


def test_a_voice_toml_sets_the_style(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    soundfile.write(str(tmp_path / "meow.wav"), np.sin(np.linspace(0, 300, RATE // 2)), RATE)
    assert load_sounds(tmp_path).style == Style()  # no file: the defaults
    (tmp_path / STYLE_FILE).write_text("gap = 1.5  # s\njitter = 0\n")
    sounds = load_sounds(tmp_path)
    assert len(sounds.clips) == 1  # the toml isn't taken for a sound
    assert sounds.style == Style(gap=1.5, jitter=0)  # the rest stays as it was
    assert not load_sounds(tmp_path / "missing")


def test_a_bad_voice_toml_says_what_is_wrong(tmp_path):
    path = tmp_path / STYLE_FILE
    path.write_text("lenght = 0.4\n")
    with pytest.raises(ValueError, match="unknown setting lenght"):
        load_style(path)
    path.write_text('gap = "long"\n')
    with pytest.raises(ValueError, match="gap must be a number"):
        load_style(path)
    path.write_text("jitter = -1\n")
    with pytest.raises(ValueError, match="jitter"):
        load_style(path)


def test_sounds_are_made_as_loud_as_each_other(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    for name, amplitude in (("quiet.wav", 0.05), ("loud.wav", 0.8)):
        soundfile.write(str(tmp_path / name), amplitude * _tone(0.5), RATE)
    for clip in load_sounds(tmp_path).clips:
        assert np.sqrt(np.mean(clip[RATE // 10:-RATE // 10] ** 2)) == pytest.approx(LEVEL, rel=0.02)


def test_the_same_sound_never_plays_twice_in_a_row():
    voice = Voice(np.random.default_rng(0))
    picks = [voice._pick(3) for _ in range(200)]
    assert set(picks) == {0, 1, 2}
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert voice._pick(1) == 0  # unless there is only one
