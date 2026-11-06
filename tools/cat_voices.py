"""Builds the cats' voices, assets/cats/<Name>/voice/*.ogg, from CC0 recordings.

Every recording used is on Freesound under CC0 (public domain); SOURCES says
which, so where each sound came from stays written down. The script downloads
Freesound's high-quality preview of each once (into build/cat-voices/), cuts
single meows out of it and gives each cat its own character from them: higher
or lower (which also makes it faster or slower, like a smaller or bigger cat),
a pitch that rises or falls, a rolled "rrr", a rasp, a duller or brighter tone.

    uv run python tools/cat_voices.py                # all cats
    uv run python tools/cat_voices.py Grumpkin       # only some

It replaces the cat's .ogg files but leaves its voice.toml (how the sounds
follow your voice, see catcam/voice.py) alone.
"""

import argparse
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catcam.paths import ASSETS  # noqa: E402

CACHE = ROOT / "build" / "cat-voices"
RATE = 48000  # Hz the sounds are made and written at
HIGHPASS = 150.0  # Hz, below this is handling noise and room rumble, never the cat
FADE_IN, FADE_OUT = 0.005, 0.04  # s, so no sound starts or ends with a click
PEAK = 0.89  # -1 dB of full scale

# Freesound id: (user, title). All CC0, checked on each sound's page.
SOURCES = {
    669532: ("HECKFRICKER", "Cat Politely Meowing"),
    686775: ("nathan-osman", "meow-1.wav"),
    732521: ("Lukey1028", "Begging Meow"),
    735643: ("XxTamisxx", "Kitten mewing / young baby cat meowing"),
    741166: ("Kinoton", "Cat, Meows 3x"),
    412017: ("skymary", "cat meow short"),
    381783: ("ghostlung", "Cat Meow"),
    541264: ("sweet_niche", "AnnoyedCatMeowsGruntsVowels.wav"),
    338819: ("cabled_mess", "Old cat meowing"),
    724090: ("FreqWincy", "Coco meow 17 year old cat"),
    826683: ("tcurless", "Siamese Cat Yowls"),
    757185: ("HenKonen", "Cat meowing raspy"),
    844502: ("Sadiquecat", "Pearl mmrrovvv sound"),
    455493: ("Wrenasmir", "Cat Meow.aif"),
    451250: ("dreamstobecome", "cat chirp.wav"),
}


@dataclass(frozen=True)
class Take:
    """One sound: a stretch of a source recording and what is done to it."""
    source: int  # Freesound id, see SOURCES
    start: float  # s into the recording
    end: float
    semitones: float = 0.0  # higher (and faster) or lower (and slower)
    bend: float = 0.0  # semitones the pitch rises (or falls, if negative) over the sound
    roll: float = 0.0  # Hz of a rolled "rrr" (0 = none)
    roll_depth: float = 0.0  # 0..1, how much the roll dips the sound
    rasp: float = 0.0  # 0..1, how much the sound is overdriven into a rasp
    lowpass: float = 0.0  # Hz above which the tone is dulled (0 = not)
    longest: float = 0.0  # s the sound is cut to at most (0 = whole)


def _takes(source: int, spans, **character) -> list[Take]:
    return [Take(source, start, end, **character) for start, end in spans]


# What each cat sounds like, and from which meows.
VOICES = {
    # Cheerful, chirpy trills: short high meows, rolled and rising.
    "Grinbert": (
        _takes(455493, [(3.95, 4.24), (4.59, 5.0), (5.37, 5.74)],
               semitones=2, bend=4, roll=24, roll_depth=0.55)
        + _takes(669532, [(5.35, 5.56), (14.82, 15.24), (18.24, 18.70)],
                 semitones=1, bend=4, roll=24, roll_depth=0.55)
        + _takes(451250, [(0.55, 0.76)], semitones=0, bend=3, roll=24, roll_depth=0.5)
    ),
    # Soft, pleading, high mews that droop at the end.
    "Pleadwell": (
        _takes(669532, [(0.05, 0.66), (1.15, 1.80), (2.63, 3.08)], semitones=2, bend=-1.5, lowpass=6000)
        + _takes(686775, [(0.09, 0.9)], semitones=1, bend=-1.5, lowpass=6000)
        + _takes(732521, [(0.07, 0.95)], semitones=1, bend=-1, lowpass=6000)
        + _takes(735643, [(2.31, 2.78), (3.81, 4.38)], semitones=0, bend=-1, lowpass=6000)
    ),
    # Posh and bored: long, slow meows that sink.
    "Meringue": (
        _takes(741166, [(0.11, 1.74), (3.33, 4.88), (6.51, 8.06)], semitones=-1, bend=-1.5, lowpass=7000)
        + _takes(412017, [(0.27, 1.0)], semitones=-2, bend=-2, lowpass=7000)
        + _takes(381783, [(0.0, 1.10), (1.27, 2.66)], semitones=-2, bend=-2, lowpass=7000)
    ),
    # Low, grumbly "mrrr"s.
    "Grumpkin": _takes(541264, [(7.65, 8.16), (9.07, 9.88), (11.55, 12.38), (12.67, 13.24),
                                (15.49, 16.28), (19.27, 20.06)],
                       semitones=-4, roll=26, roll_depth=0.35, lowpass=5000),
    # Raspy, sassy yowls with a rise of attitude at the end.
    "Wrinklestiltskin": (
        _takes(826683, [(0.29, 1.34), (1.91, 3.02), (3.65, 4.68), (7.21, 8.02)], bend=2, rasp=0.5)
        + _takes(757185, [(0.0, 0.78), (1.49, 2.46), (3.29, 4.04)], semitones=1, bend=2, rasp=0.35)
    ),
    # Short, snappy, suspicious "mrrp?"s.
    "Gingersnap": (
        _takes(541264, [(0.81, 1.22), (3.27, 3.52), (5.19, 5.56), (5.81, 6.14), (7.07, 7.40), (21.13, 21.44)],
               semitones=2, bend=3, roll=30, roll_depth=0.3, longest=0.35)
        + _takes(844502, [(0.09, 0.80)], semitones=2, bend=3, roll=30, roll_depth=0.3, longest=0.35)
    ),
    # Deep, slow, old grumbles.
    "Grumbledore": (
        _takes(338819, [(4.43, 5.30), (6.19, 7.22), (8.13, 9.24), (14.71, 15.68), (20.85, 21.76)],
               semitones=-7, roll=18, roll_depth=0.25, lowpass=3000)
        + _takes(724090, [(10.55, 11.22), (12.99, 13.58)], semitones=-7, roll=18, roll_depth=0.25,
                 lowpass=3000)
    ),
}


def _download(sound_id: int) -> Path:
    """The recording's high-quality preview, downloaded once."""
    path = CACHE / f"{sound_id}.ogg"
    if path.is_file():
        return path
    headers = {"User-Agent": "cat-cam voice builder"}
    page = urllib.request.urlopen(urllib.request.Request(f"https://freesound.org/s/{sound_id}/",
                                                         headers=headers)).read().decode()
    licenses = set(re.findall(r'href="(https?://creativecommons\.org/[^"]+)"', page))
    if not licenses or any("publicdomain/zero" not in url for url in licenses):
        sys.exit(f"Freesound {sound_id} is not CC0 ({sorted(licenses)}); not using it.")
    preview = re.search(r'data-ogg="([^"]+)"', page)
    if preview is None:
        sys.exit(f"No preview found for Freesound {sound_id}.")
    url = preview[1].replace("-lq.ogg", "-hq.ogg")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read())
    return path


def _load(sound_id: int) -> np.ndarray:
    samples, rate = soundfile.read(str(_download(sound_id)), dtype="float64", always_2d=True)
    mono = resample_poly(samples.mean(axis=1), RATE, rate)
    return sosfiltfilt(butter(2, HIGHPASS, "highpass", fs=RATE, output="sos"), mono)


def _respeed(x: np.ndarray, semitones: float, bend: float) -> np.ndarray:
    """Played faster (higher) or slower (lower), gliding by bend semitones from start to end."""
    def speed(progress):
        return 2 ** ((semitones + bend * progress) / 12)

    n = int(len(x) / speed(0.5))
    positions = np.concatenate([[0.0], np.cumsum(speed(np.linspace(0, 1, n)))])
    positions = positions[positions < len(x) - 1]
    return np.interp(positions, np.arange(len(x)), x)


def make(take: Take, sources: dict[int, np.ndarray]) -> np.ndarray:
    """The finished sound of one take, at RATE."""
    x = sources[take.source][int(take.start * RATE):int(take.end * RATE)]
    x = _respeed(x, take.semitones, take.bend)
    if take.lowpass:
        x = sosfiltfilt(butter(4, take.lowpass, "lowpass", fs=RATE, output="sos"), x)
    if take.roll:
        t = np.arange(len(x)) / RATE
        x = x * (1 - take.roll_depth * (0.5 - 0.5 * np.cos(2 * np.pi * take.roll * t)))
    if take.rasp:
        drive = 1 + 9 * take.rasp
        level = np.abs(x).max()
        x = level * np.tanh(drive * x / level) / np.tanh(drive)
    if take.longest:
        x = x[:int(take.longest * RATE)]
    fade_in, fade_out = int(FADE_IN * RATE), int(FADE_OUT * RATE)
    x = x.copy()
    x[:fade_in] *= np.linspace(0, 1, fade_in)
    x[-fade_out:] *= np.linspace(1, 0, fade_out)
    return (PEAK * x / np.abs(x).max()).astype(np.float32)


def build(name: str, takes: list[Take], assets: Path = ASSETS) -> list[Path]:
    folder = assets / name / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.ogg"):
        old.unlink()
    sources = {sid: _load(sid) for sid in sorted({take.source for take in takes})}
    paths = []
    for i, take in enumerate(takes, 1):
        path = folder / f"{name.lower()}{i}.ogg"
        soundfile.write(str(path), make(take, sources), RATE, format="OGG", subtype="VORBIS")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build the cats' voices from CC0 recordings.")
    parser.add_argument("names", nargs="*", help=f"cats to build (default all: {', '.join(VOICES)})")
    names = parser.parse_args().names or list(VOICES)
    for name in names:
        if name not in VOICES:
            sys.exit(f"No voice for '{name}'. Voices: {', '.join(VOICES)}")
        paths = build(name, VOICES[name])
        size = sum(p.stat().st_size for p in paths) / 1024
        print(f"{name}: {len(paths)} sounds, {size:.0f} KB")


if __name__ == "__main__":
    main()
