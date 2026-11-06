"""Your voice, as your animal's.

While you talk, your animal talks along in its own sounds: the dog barks, the
cats meow, trill or grumble. A sound starts with one of your syllables, as
loud as you said it and a little higher or lower as your voice goes above or
below its usual pitch, and then plays to its natural end, the way an animal
finishes its meow. The animal only starts its next sound with a later
syllable of yours, once it has been quiet for a while (its gap): a dog barks
along with almost every syllable, a cat meows a few times a sentence. The
sounds go to a virtual microphone (BlackHole), which the call uses instead of
your real one, so your own voice is never heard. With an animal that has no
sounds (or the app's Voice switch off), your voice passes through unchanged.

The sounds are real recordings, played as they are; they are only made a
little higher or lower (at most PITCH_LIMIT), as more than that makes an
animal sound like a sped-up tape or a robot.

An animal's sounds are in assets/cats/<Name>/voice/. A voice.toml next to them
can change how they follow your voice (see Style): a cat waits longer between
its meows than a dog between its barks.

Syllables are found in your loudness: one starts where it rises clearly above
both the background noise and the dip after the previous syllable.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

RATE = 48000
BLOCK = 480  # samples per audio block, 10 ms
VIRTUAL_MIC = "BlackHole"  # the output device whose name contains this
BUILT_IN_MIC = ("MacBook", "Built-in")  # names of the microphone used when the default is the virtual one
SOUND_TYPES = (".ogg", ".wav", ".flac", ".mp3")
STYLE_FILE = "voice.toml"  # optional, next to the sounds, see Style

# Loudness in dB of full scale.
SILENCE = -55.0  # quieter than this is never speech
ABOVE_NOISE = 12.0  # speech is at least this much louder than the background noise
RISE = 6.0  # a syllable is at least this much louder than the dip before it
SMOOTHING = 0.015  # s, time constant of the loudness
NOISE_CREEP = 3.0  # dB/s the noise level rises while it's louder (it drops at once)
MIN_VOICED = 0.03  # s the voice must last before it counts, so clicks don't bark
MIN_GAP = 0.12  # s between two syllables at least
LOUDNESS = (12.0, 36.0)  # dB above the noise mapped to the quietest..loudest sound
GAIN = (0.15, 0.5)  # volume of the quietest..loudest sound (1 = as loud as LEVEL)
LEVEL = 0.32  # RMS every sound is made as loud as at its loudest, so none stands out
LOUDEST = 0.05  # s, the stretch of a sound its loudness (for LEVEL) is measured over
LOUD_PITCH = 1.0  # semitones higher for the loudest sounds

# Your pitch, measured at the start of each syllable.
PITCH_RANGE = (70.0, 400.0)  # Hz of a speaking voice
PITCH_WINDOW = 0.04  # s of your voice it is measured on
VOICING = 0.5  # autocorrelation peak needed to count as a pitch, not a hiss
PITCH_FOLLOW = 0.5  # sound semitones per semitone you are above or below your usual pitch
PITCH_LIMIT = 2.0  # semitones a sound goes up or down at most; more sounds like a robot
USUAL_PITCH_ADAPT = 0.1  # how quickly your usual pitch adapts, per syllable
JITTER = 0.25  # semitones of random pitch, so a repeated sound differs a little

# Between sounds.
GAP = 0.2  # s the animal is quiet after a sound before the next can start
GAP_SPREAD = (0.7, 1.5)  # times the gap, at random, so the animal doesn't keep time like a machine


@dataclass(frozen=True)
class Style:
    """How an animal's sounds follow your voice. The defaults suit barks; an
    animal's voice.toml can set any of these, by the same names and units."""
    pitch_follow: float = PITCH_FOLLOW  # sound semitones per semitone you go above or below your usual pitch
    jitter: float = JITTER  # semitones of random pitch per sound
    gap: float = GAP  # s the animal is quiet after a sound before the next can start


@dataclass(frozen=True)
class Sounds:
    """An animal's sounds and how they follow your voice; false if there are none."""
    clips: tuple = ()  # mono float32 arrays at RATE
    style: Style = Style()

    def __bool__(self) -> bool:
        return bool(self.clips)


def load_style(path: Path) -> Style:
    """The Style a voice.toml asks for, with the defaults for what it leaves out."""
    with path.open("rb") as f:
        values = tomllib.load(f)
    known = {f.name for f in fields(Style)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{path}: unknown setting {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError(f"{path}: {name} must be a number of at least 0, not {value!r}")
    return Style(**{name: float(value) for name, value in values.items()})


def _level(sound: np.ndarray) -> np.ndarray:
    """sound made as loud as LEVEL in its loudest moment, without clipping."""
    n = min(max(int(LOUDEST * RATE), 1), len(sound))
    energy = np.concatenate([[0.0], np.cumsum(np.square(sound, dtype=np.float64))])
    loudest = np.sqrt((energy[n:] - energy[:-n]).max() / n) if n else 0.0
    scale = min(LEVEL / max(loudest, 1e-6), 0.99 / max(np.abs(sound).max(), 1e-6))
    return (scale * sound).astype(np.float32)


def load_sounds(folder: Path) -> Sounds:
    """The sounds in folder as mono float32 at RATE, all as loud as each other,
    with the style its voice.toml asks for; empty if there are none."""
    if not folder.is_dir():
        return Sounds()
    import soundfile
    from scipy.signal import resample_poly

    clips = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SOUND_TYPES):
        samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        mono = resample_poly(samples.mean(axis=1), RATE, rate).astype(np.float32)
        clips.append(_level(mono))
    style = folder / STYLE_FILE
    return Sounds(tuple(clips), load_style(style) if style.is_file() else Style())


def _ramp(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def estimate_pitch(samples: np.ndarray) -> float | None:
    """Fundamental frequency (Hz) of a stretch of voice, or None if it has none."""
    x = samples.astype(np.float64) - samples.mean()
    n = len(x)
    corr = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
    if corr[0] <= 0:
        return None
    lo, hi = int(RATE / PITCH_RANGE[1]), int(RATE / PITCH_RANGE[0])
    # Not normalised by the overlap, so the shortest matching period wins over its multiples.
    lag = lo + int(np.argmax(corr[lo:hi]))
    return RATE / lag if corr[lag] / corr[0] >= VOICING else None


class Syllables:
    """Finds the starts of syllables in a stream of loudness levels."""

    def __init__(self):
        self.level = None  # smoothed loudness, dB
        self.noise = None  # background noise level, dB
        self.voiced = 0.0  # s the voice has lasted
        self.speaking = False
        self.since = np.inf  # s since the last syllable started
        self.climbing, self.peak, self.dip = False, -np.inf, np.inf

    def __call__(self, level: float, dt: float) -> float | None:
        """level: dB of the newest block. A syllable starting now: how much
        louder than the noise it is (dB); otherwise None."""
        if self.level is None:
            self.level = self.noise = level
        self.level += (1 - np.exp(-dt / SMOOTHING)) * (level - self.level)
        self.noise = min(self.level, self.noise + NOISE_CREEP * dt)
        self.since += dt
        above = self.level - self.noise
        self.speaking = self.level > SILENCE and above > ABOVE_NOISE
        self.voiced = self.voiced + dt if self.speaking else 0.0
        # After a syllable starts, its dip is only looked for once it has peaked.
        if self.climbing and self.level >= self.peak:
            self.peak = self.level
        elif self.climbing:
            self.climbing, self.dip = False, self.level
        else:
            self.dip = min(self.dip, self.level)
        if (not self.climbing and self.voiced >= MIN_VOICED and self.since >= MIN_GAP
                and self.level - self.dip >= RISE):
            self.climbing, self.peak, self.since = True, self.level, 0.0
            return above
        return None


class Bark:
    """One playing sound. It always plays to its end: cut short, a meow
    sounds like a blip from a machine."""

    def __init__(self, samples: np.ndarray):
        self.samples, self.position = samples, 0

    def take(self, n: int) -> np.ndarray:
        part = self.samples[self.position:self.position + n]
        self.position += n
        return part

    @property
    def done(self) -> bool:
        return self.position >= len(self.samples)


class Voice:
    """Turns microphone blocks into what the virtual microphone sends."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.sounds = Sounds()  # the animal's; empty = your own voice
        self.syllables = Syllables()
        self.rng = rng or np.random.default_rng()
        self.recent = np.zeros(int(PITCH_WINDOW * RATE), np.float32)  # your latest voice
        self.usual_pitch = None  # log2 Hz, adapts to you
        self.bark: Bark | None = None  # the sound playing
        self.quiet = 0.0  # s the animal still stays quiet for before its next sound
        self.last = None  # index of the clip played last, so the next is another one

    def _semitones(self, pitch: float | None, follow: float) -> float:
        """How far your syllable is above your usual pitch, as sound semitones."""
        if pitch is None:
            return 0.0
        octave = np.log2(pitch)
        if self.usual_pitch is None:
            self.usual_pitch = octave
        above = 12 * (octave - self.usual_pitch)
        self.usual_pitch += USUAL_PITCH_ADAPT * (octave - self.usual_pitch)
        return follow * above

    def _pick(self, count: int) -> int:
        """A random clip, never the one just played (if there is another)."""
        options = [i for i in range(count) if i != self.last] or [0]
        self.last = options[int(self.rng.integers(len(options)))]
        return self.last

    def _bark(self, sounds: Sounds, above: float, tone: float) -> Bark:
        """A sound for a syllable `above` dB louder than the noise and `tone`
        sound semitones above your usual pitch."""
        loud = _ramp(above, *LOUDNESS)
        semitones = float(np.clip(tone + LOUD_PITCH * loud + self.rng.normal(0, sounds.style.jitter),
                                  -PITCH_LIMIT, PITCH_LIMIT))
        sound = sounds.clips[self._pick(len(sounds.clips))]
        speed = 2 ** (semitones / 12)
        resampled = np.interp(np.arange(0, len(sound) - 1, speed), np.arange(len(sound)), sound)
        return Bark((GAIN[0] + loud * (GAIN[1] - GAIN[0])) * resampled.astype(np.float32))

    def process(self, mic: np.ndarray) -> np.ndarray:
        """mic: one mono float32 block. Returns the block to send."""
        self.recent = np.concatenate([self.recent, mic])[-len(self.recent):]
        level = 10 * np.log10(np.mean(np.square(mic, dtype=np.float64)) + 1e-12)
        above = self.syllables(level, len(mic) / RATE)
        sounds = self.sounds  # read once: the app swaps them from another thread
        if not sounds:
            self.bark = None
            return mic.copy()
        self.quiet -= len(mic) / RATE
        if above is not None:
            # Every syllable teaches the usual pitch, even those the animal lets pass.
            tone = self._semitones(estimate_pitch(self.recent), sounds.style.pitch_follow)
            if self.bark is None and self.quiet <= 0:
                self.bark = self._bark(sounds, above, tone)
        out = np.zeros(len(mic), np.float32)
        if self.bark is not None:
            part = self.bark.take(len(mic))
            out[:len(part)] = part
            if self.bark.done:
                self.bark = None
                self.quiet = sounds.style.gap * self.rng.uniform(*GAP_SPREAD)
        return np.clip(out, -1, 1)


def find_device(devices, choice: str, kind: str) -> int | None:
    """Index of the first device with kind ("input" or "output") channels whose
    number is choice, or, if choice isn't a number, whose name contains it."""
    for index, device in enumerate(devices):
        if device[f"max_{kind}_channels"] == 0:
            continue
        if choice == str(index) if choice.isdigit() else choice.lower() in device["name"].lower():
            return index
    return None


def _names(devices, kind: str) -> str:
    return ", ".join(f"{i} {d['name']}" for i, d in enumerate(devices) if d[f"max_{kind}_channels"] > 0)


def pick_mic(devices, default: int) -> int | None:
    """The microphone to listen to: the default one, unless that is the virtual
    microphone, which would only hear its own barks; then the built-in one."""
    def real(index):
        device = devices[index]
        return device["max_input_channels"] > 0 and VIRTUAL_MIC.lower() not in device["name"].lower()

    if 0 <= default < len(devices) and real(default):
        return default
    inputs = [i for i in range(len(devices)) if real(i)]
    built_in = [i for i in inputs if any(n in devices[i]["name"] for n in BUILT_IN_MIC)]
    return (built_in or inputs or [None])[0]


SETUP_HINT = (
    "No virtual microphone found, so the call hears your real microphone.\n"
    "  One-time setup: brew install blackhole-2ch   (then restart Cat Cam)\n"
    "  and pick 'BlackHole 2ch' as the microphone in Zoom."
)
DRIVER = Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver")
RELOAD_HINT = (
    "BlackHole is installed but macOS hasn't loaded it yet, so the call hears your real microphone.\n"
    "  Run: sudo killall coreaudiod   (or restart the Mac), then restart Cat Cam."
)


class VirtualMic:
    """Your microphone, through a Voice, out to the virtual microphone (or
    another output device)."""

    def __init__(self, voice: Voice, mic: str | None = None, out: str | None = None):
        """mic, out: device number or (part of its) name; None = pick one."""
        self.voice, self.mic, self.out = voice, mic, out
        self.stream = None
        self.heard = self.device = None  # names of the microphone and output device, once running

    def start(self) -> bool:
        import sounddevice

        devices = sounddevice.query_devices()
        out = find_device(devices, self.out or VIRTUAL_MIC, "output")
        if out is None and self.out is None:
            print(RELOAD_HINT if DRIVER.exists() else SETUP_HINT)
            return False
        if out is None:
            print(f"No output device '{self.out}'. Outputs: {_names(devices, 'output')}")
            return False
        if self.mic is None:
            mic = pick_mic(devices, sounddevice.default.device[0])
        else:
            mic = find_device(devices, self.mic, "input")
        if mic is None:
            print(f"No microphone '{self.mic or ''}', so the virtual microphone stays silent. "
                  f"Microphones: {_names(devices, 'input')}")
            return False
        try:
            self.stream = sounddevice.Stream(
                samplerate=RATE, blocksize=BLOCK, device=(mic, out), channels=(1, 2),
                dtype="float32", callback=self._callback,
            )
            self.stream.start()
        except sounddevice.PortAudioError as e:
            print(f"Virtual microphone disabled: {e}")
            self.stream = None
            return False
        self.heard, self.device = devices[mic]["name"], devices[out]["name"]
        return True

    def _callback(self, indata, outdata, frames, time, status):
        outdata[:] = self.voice.process(indata[:, 0])[:, None]

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def main():
    """Try the voice live: your microphone, as the animal, in your headphones."""
    import argparse

    import sounddevice

    from catcam import cats

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--cat", default="Woofgang", help="whose voice (default Woofgang)")
    parser.add_argument("--mic", metavar="NAME", help="microphone, by name or number (default: the system's)")
    parser.add_argument("--out", metavar="NAME", help="output, by name or number (default: the system's)")
    parser.add_argument("--list", action="store_true", help="list the audio devices and their numbers")
    args = parser.parse_args()
    if args.list:
        print(sounddevice.query_devices())
        return
    voice = Voice()
    voice.sounds = cats.load(args.cat).sounds
    mic = VirtualMic(voice, args.mic, args.out or str(sounddevice.default.device[1]))
    if not mic.start():
        return
    print(f"Listening to '{mic.heard}', playing on '{mic.device}'.\n"
          "Use headphones: from speakers the microphone hears the barks and barks back.")
    try:
        input("Talk! Press Enter to stop.\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
