"""Builds the animals' voices, assets/cats/<Name>/voice/*.ogg, from CC0 recordings.

Every recording used is on Freesound under CC0 (public domain); SOURCES says
which, so where each sound came from stays written down. The script downloads
Freesound's high-quality preview of each once (into build/cat-voices/),
checking on the sound's page that it is CC0, and cuts single meows (or barks)
out of it.

The sounds are left as the animal made them: no pitch or speed change, no
effects. Each animal's character comes from choosing recordings that already
sound like it (a kitten for Pleadwell, an old cat for Grumbledore). Earlier
versions made the character with effects instead (resampled pitch, glides, a
synthetic rolled "rrr", overdriven rasp), and that is what made the cats sound
like robots. All that is done now is to take out the rumble below the animal's
voice, find where the sound starts and ends inside the given stretch, keep its
natural tail and fade the edges so nothing clicks.

    uv run python tools/cat_voices.py                # all animals
    uv run python tools/cat_voices.py Grumpkin       # only some

It replaces the animal's .ogg files but leaves its voice.toml (how the sounds
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
HIGHPASS = 120.0  # Hz, below this is handling noise and room rumble, never the animal's voice
FRAME = 0.005  # s, the loudness of a sound is measured in steps this long
KEEP = 40.0  # dB: a sound lasts from where it first comes within this of its peak to where it last does,
ABOVE_NOISE = 8.0  # dB, and is louder than the recording's background noise by at least this
LEAD = 0.015  # s kept before the sound starts, so its onset isn't cut
TAIL = 0.12  # s kept after it, so it dies away naturally instead of stopping
FADE_IN, FADE_OUT = 0.008, 0.06  # s of smooth fade at the edges, so no sound starts or ends with a click
QUIET_EDGE = 20.0  # dB: the stretch must start and end at least this much below the sound's peak
PEAK = 0.89  # -1 dB of full scale
SILENT = 1e-4  # amplitude below which a recording is digital silence (-80 dB)

# Freesound id: (user, title). All CC0, checked on each sound's page.
SOURCES = {
    200336: ("jsbarrett", "Cat trilling"),
    196251: ("lolamadeus", "Kitten Meowing.wav"),
    149191: ("videog", "kittens meowing"),
    214759: ("peridactyloptrix", "Cat meowing x5"),
    120160: ("Oneirophile", "Cat Wants Food"),
    757185: ("HenKonen", "Cat meowing raspy"),
    541264: ("sweet_niche", "AnnoyedCatMeowsGruntsVowels.wav"),
    338819: ("cabled_mess", "Old cat meowing"),
    146343: ("colinpoh", "Animal - Dog, Retriever (Teen)"),
}


@dataclass(frozen=True)
class Take:
    """One sound: the stretch of a source recording it is in."""
    source: int  # Freesound id, see SOURCES
    start: float  # s into the recording, a little before the sound starts
    end: float  # s, a little after it has died away


def _takes(source: int, spans) -> list[Take]:
    return [Take(source, start, end) for start, end in spans]


# What each animal sounds like, and from which recordings. Each is one real
# animal where possible, so it keeps one voice from sound to sound.
VOICES = {
    # Cheerful, chatty trills: a cat going "brrrp!".
    "Grinbert": _takes(200336, [(0.0, 0.85), (3.35, 4.3), (5.05, 5.95), (5.98, 6.8), (7.6, 8.5),
                                (9.78, 10.4), (12.3, 13.1)]),
    # Soft, pleading, high mews: kittens, which droop at the end by themselves.
    "Pleadwell": (
        _takes(196251, [(0.35, 1.15), (1.35, 2.25), (2.65, 3.55), (4.65, 5.65), (7.0, 7.9)])
        + _takes(149191, [(3.3, 3.95), (6.98, 7.7), (8.25, 9.0)])
    ),
    # Posh and bored: long, level meows.
    "Meringue": _takes(214759, [(1.3, 2.5), (5.5, 6.4), (8.3, 9.5), (10.6, 11.75), (13.9, 15.0)]),
    # Low, grumbly, demanding "mrraow"s from a cat that wants its food.
    "Grumpkin": _takes(120160, [(0.0, 1.25), (1.3, 2.75), (2.8, 3.95), (4.0, 5.35), (5.47, 6.85),
                                (6.95, 8.4), (8.52, 9.6)]),
    # Raspy, sassy yowls that rise with attitude at the end.
    "Wrinklestiltskin": _takes(757185, [(0.0, 0.95), (1.4, 2.6), (3.2, 4.25), (4.55, 5.6), (6.4, 7.45),
                                        (8.0, 9.1), (9.8, 10.9)]),
    # Short, snappy, questioning "mrrp?"s from an annoyed cat.
    "Gingersnap": _takes(541264, [(0.0, 0.7), (0.7, 1.4), (3.2, 3.68), (3.7, 4.35), (5.1, 5.7),
                                  (5.75, 6.3), (17.8, 18.4), (21.05, 21.6)]),
    # Deep, slow grumbles from an old cat.
    "Grumbledore": _takes(338819, [(0.0, 1.2), (4.3, 5.5), (6.05, 7.5), (8.0, 9.5), (11.6, 12.9),
                                   (14.6, 15.9), (18.8, 19.9), (20.75, 22.0)]),
    # A friendly young golden retriever's barks.
    "Woofgang": _takes(146343, [(0.0, 0.33), (0.33, 0.66), (0.66, 0.97), (0.97, 1.4), (1.65, 2.1),
                                (2.2, 2.57), (2.57, 3.0)]),
}


def _download(sound_id: int) -> Path:
    """The recording's high-quality preview, downloaded once, if its page says it is CC0."""
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


def _levels(x: np.ndarray) -> np.ndarray:
    """dB of each FRAME of x."""
    n = int(FRAME * RATE)
    frames = x[:len(x) // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def _fade(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def _noise(x: np.ndarray) -> float:
    """dB of the background noise of a recording: its quieter moments, leaving
    out digital silence (where it was edited)."""
    db = _levels(x)
    return float(np.percentile(db[db > -100], 10)) if np.any(db > -100) else -100.0


def make(take: Take, sources: dict[int, np.ndarray]) -> np.ndarray:
    """The finished sound of one take, at RATE: the sound in the stretch with a
    little lead and its natural tail, faded at the edges and normalised."""
    x = sources[take.source][int(take.start * RATE):int(take.end * RATE)]
    db = _levels(x)
    loud = np.flatnonzero(db > max(db.max() - KEEP, _noise(sources[take.source]) + ABOVE_NOISE))
    n = int(FRAME * RATE)
    start = max(loud[0] * n - int(LEAD * RATE), 0)
    end = min((loud[-1] + 1) * n + int(TAIL * RATE), len(x))
    x = x[start:end].copy()
    # Some recordings were edited to digital silence between sounds, which can
    # cut a sound off abruptly; then it ends (or starts) where the silence does.
    sounding = np.flatnonzero(np.abs(x) > SILENT)
    x = x[sounding[0]:sounding[-1] + 1]
    fade_in, fade_out = int(FADE_IN * RATE), int(FADE_OUT * RATE)
    x[:fade_in] *= _fade(fade_in)
    x[-fade_out:] *= _fade(fade_out)[::-1]
    return (PEAK * x / np.abs(x).max()).astype(np.float32)


def check(take: Take, sources: dict[int, np.ndarray]) -> str | None:
    """What is wrong with a take's stretch, if it doesn't start and end in quiet
    (then it cuts into a sound, its own or a neighbour's)."""
    db = _levels(sources[take.source][int(take.start * RATE):int(take.end * RATE)])
    edge = int(0.02 / FRAME)
    for where, level in (("starts", db[:edge].max()), ("ends", db[-edge:].max())):
        if level > db.max() - QUIET_EDGE:
            return f"{take} {where} only {db.max() - level:.0f} dB below its peak"
    return None


def build(name: str, takes: list[Take], assets: Path = ASSETS) -> list[Path]:
    folder = assets / name / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    sources = {sid: _load(sid) for sid in sorted({take.source for take in takes})}
    problems = [p for p in (check(take, sources) for take in takes) if p]
    if problems:
        sys.exit(f"{name}:\n  " + "\n  ".join(problems))
    for old in folder.glob("*.ogg"):
        old.unlink()
    paths = []
    for i, take in enumerate(takes, 1):
        path = folder / f"{name.lower()}{i}.ogg"
        soundfile.write(str(path), make(take, sources), RATE, format="OGG", subtype="VORBIS")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build the animals' voices from CC0 recordings.")
    parser.add_argument("names", nargs="*", help=f"animals to build (default all: {', '.join(VOICES)})")
    names = parser.parse_args().names or list(VOICES)
    for name in names:
        if name not in VOICES:
            sys.exit(f"No voice for '{name}'. Voices: {', '.join(VOICES)}")
        paths = build(name, VOICES[name])
        size = sum(p.stat().st_size for p in paths) / 1024
        print(f"{name}: {len(paths)} sounds, {size:.0f} KB")


if __name__ == "__main__":
    main()
