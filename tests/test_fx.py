from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import soundfile

from catcam import fx


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    for cat, colour in (("Amy", (0, 0, 255)), ("Bob", (0, 255, 0))):
        picture = np.full((900, 900, 3), colour, np.uint8)
        cv2.imwrite(str(folder / f"{cat}Laugh.png"), picture)
    cv2.imwrite(str(folder / "AmyWow.PNG"), np.full((900, 600, 3), 255, np.uint8))
    soundfile.write(str(folder / "Laugh.wav"), np.zeros((800, 2), np.float32), 8000)
    return folder


def test_pack_and_load(source, tmp_path):
    out = tmp_path / "fx.bin"
    manifest = fx.pack(source, {"x": "Laugh", "y": "wow"}, ["Amy", "Bob", "Cid"], out)
    assert sorted(manifest["x"]["pictures"]) == ["Amy", "Bob"]
    assert manifest["y"]["sound"] is None
    effects = fx.load(out)
    assert effects["x"].pictures["Bob"].shape == (fx.PICTURE_HEIGHT, fx.PICTURE_HEIGHT, 3)
    samples, rate = effects["x"].sound
    assert samples.shape == (800, 2) and rate == 8000
    assert effects["y"].sound is None and list(effects["y"].pictures) == ["Amy"]


def test_packed_file_does_not_look_like_media(source, tmp_path):
    out = tmp_path / "fx.bin"
    fx.pack(source, {"x": "Laugh"}, ["Amy", "Bob"], out)
    data = out.read_bytes()
    for signature in (b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"manifest", b"Laugh"):
        assert signature not in data


def test_missing_name_is_an_error(source, tmp_path):
    with pytest.raises(ValueError, match="nothing found"):
        fx.pack(source, {"z": "Nope"}, ["Amy"], tmp_path / "fx.bin")


def test_no_data_file_means_no_effects(tmp_path):
    assert fx.load(tmp_path / "missing.bin") == {}


def test_full_frame_fills_the_sides_with_the_backdrop():
    picture = np.full((100, 100, 3), 255, np.uint8)
    picture[40:60, 40:60] = 0
    frame = fx.full_frame(picture, (320, 180))
    assert frame.shape == (180, 320, 3)
    assert (frame[:, :50] == 255).all() and (frame[90, 160] == 0).all()


@pytest.fixture
def effects(monkeypatch):
    import sounddevice

    calls = []
    monkeypatch.setattr(sounddevice, "play", lambda samples, rate: calls.append("play"))
    monkeypatch.setattr(sounddevice, "stop", lambda: calls.append("stop"))
    sound = (np.zeros((10, 2), np.float32), 8000)
    picture = np.zeros((90, 90, 3), np.uint8)
    loaded = {"k": fx.Effect({"Amy": picture}, sound), "p": fx.Effect({"Amy": picture + 9}, None)}
    e = fx.Effects(loaded, (160, 90))
    e.calls = calls
    return e


def key(char):
    return SimpleNamespace(char=char)


def test_short_taps_do_nothing(effects):
    effects.press(key("k"))
    assert effects.update(effects.down["k"] + fx.DELAY / 2, "Amy") is None
    effects.release(key("k"))
    assert effects.calls == []


def test_pressing_shows_the_picture_and_plays_from_the_start(effects):
    effects.press(key("K"))  # with shift: still the same key
    t0 = effects.down["k"]
    frame = effects.update(t0 + fx.DELAY, "Amy")
    assert frame is effects.frames[("k", "Amy")]
    effects.press(key("k"))  # key repeat while down
    assert effects.update(t0 + 2, "Amy") is frame
    assert effects.calls == ["stop", "play"]
    effects.release(key("k"))
    assert effects.update(t0 + 3, "Amy") is None
    assert effects.calls[-1] == "stop"
    effects.press(key("k"))  # again: the sound starts over
    effects.update(effects.down["k"] + fx.DELAY, "Amy")
    assert effects.calls[-2:] == ["stop", "play"]


def test_silent_effect_and_cat_without_a_picture(effects):
    effects.press(key("p"))
    t = effects.down["p"] + fx.DELAY
    assert effects.update(t, "Amy") is effects.frames[("p", "Amy")]
    assert "play" not in effects.calls
    assert effects.update(t, "Bob") is None  # no picture for this cat: the cat stays


def test_other_keys_are_ignored(effects):
    effects.press(key("a"))
    effects.press(SimpleNamespace())  # special keys have no char
    assert effects.down == {}
