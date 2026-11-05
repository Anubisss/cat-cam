"""Your voice, as your animal's.

While you talk, every syllable you say comes out as the animal's sound (the
dog barks), shaped like you said it: as loud, higher or lower as your voice
goes above or below its usual pitch, and as long: a bark fades out when your
syllable ends and is cut off when the next one starts, so quick talk gives
quick, clipped barks. It goes to a virtual microphone (BlackHole), which the
call uses instead of your real one, so your own voice is never heard. With an
animal that has no sounds, your voice passes through unchanged.

Syllables are found in your loudness: one starts where it rises clearly above
both the background noise and the dip after the previous syllable.
"""

from pathlib import Path

import numpy as np

RATE = 48000
BLOCK = 480  # samples per audio block, 10 ms
VIRTUAL_MIC = "BlackHole"  # the output device whose name contains this
BUILT_IN_MIC = ("MacBook", "Built-in")  # names of the microphone used when the default is the virtual one
SOUND_TYPES = (".ogg", ".wav", ".flac", ".mp3")

# Loudness in dB of full scale.
SILENCE = -55.0  # quieter than this is never speech
ABOVE_NOISE = 12.0  # speech is at least this much louder than the background noise
RISE = 6.0  # a syllable is at least this much louder than the dip before it
SMOOTHING = 0.015  # s, time constant of the loudness
NOISE_CREEP = 3.0  # dB/s the noise level rises while it's louder (it drops at once)
MIN_VOICED = 0.03  # s the voice must last before it counts, so clicks don't bark
MIN_GAP = 0.12  # s between two barks at least
LOUDNESS = (12.0, 36.0)  # dB above the noise mapped to the quietest..loudest bark
GAIN = (0.3, 1.0)
LOUD_PITCH = 2.0  # semitones higher for the loudest barks

# Your pitch, measured at the start of each syllable.
PITCH_RANGE = (70.0, 400.0)  # Hz of a speaking voice
PITCH_WINDOW = 0.04  # s of your voice it is measured on
VOICING = 0.5  # autocorrelation peak needed to count as a pitch, not a hiss
PITCH_FOLLOW = 0.7  # bark semitones per semitone you are above or below your usual pitch
PITCH_LIMIT = 7.0  # semitones the bark goes up or down at most
USUAL_PITCH_ADAPT = 0.1  # how quickly your usual pitch adapts, per syllable
JITTER = 0.25  # semitones of random pitch, so repeated barks differ a little

# Bark length, in s.
MIN_BARK = 0.08  # played at least this long, even for the shortest syllable
RELEASE = 0.04  # fade-out when your syllable ends
CUT = 0.012  # fade-out when the next syllable starts


def load_sounds(folder: Path) -> list[np.ndarray]:
    """The sounds in folder as mono float32 at RATE, peaking at 0.9; empty if none."""
    if not folder.is_dir():
        return []
    import soundfile
    from scipy.signal import resample_poly

    sounds = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SOUND_TYPES):
        samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        mono = resample_poly(samples.mean(axis=1), RATE, rate).astype(np.float32)
        sounds.append(0.9 * mono / max(np.abs(mono).max(), 1e-6))
    return sounds


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
    """One playing sound, which can be faded out early."""

    def __init__(self, samples: np.ndarray):
        self.samples, self.position = samples, 0
        self.end, self.fade = len(samples), 1  # sample where it has faded out, over how many

    def stop(self, seconds: float):
        """Fade out over the next `seconds`, unless it ends sooner anyway."""
        fade = max(int(seconds * RATE), 1)
        if self.position + fade < self.end:
            self.end, self.fade = self.position + fade, fade

    def take(self, n: int) -> np.ndarray:
        index = np.arange(self.position, min(self.position + n, self.end))
        self.position += n
        return self.samples[index] * np.clip((self.end - index) / self.fade, 0, 1)

    @property
    def done(self) -> bool:
        return self.position >= self.end


class Voice:
    """Turns microphone blocks into what the virtual microphone sends."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.sounds: list[np.ndarray] = []  # the animal's; empty = your own voice
        self.syllables = Syllables()
        self.rng = rng or np.random.default_rng()
        self.recent = np.zeros(int(PITCH_WINDOW * RATE), np.float32)  # your latest voice
        self.usual_pitch = None  # log2 Hz, adapts to you
        self.barks: list[Bark] = []

    def _semitones(self, pitch: float | None) -> float:
        """How far your syllable is above your usual pitch, as bark semitones."""
        if pitch is None:
            return 0.0
        octave = np.log2(pitch)
        if self.usual_pitch is None:
            self.usual_pitch = octave
        above = 12 * (octave - self.usual_pitch)
        self.usual_pitch += USUAL_PITCH_ADAPT * (octave - self.usual_pitch)
        return float(np.clip(PITCH_FOLLOW * above, -PITCH_LIMIT, PITCH_LIMIT))

    def _bark(self, sounds: list[np.ndarray], above: float, pitch: float | None) -> Bark:
        loud = _ramp(above, *LOUDNESS)
        semitones = self._semitones(pitch) + LOUD_PITCH * loud + self.rng.normal(0, JITTER)
        sound = sounds[self.rng.integers(len(sounds))]
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
            self.barks.clear()
            return mic.copy()
        if above is not None:
            for bark in self.barks:
                bark.stop(CUT)
            self.barks.append(self._bark(sounds, above, estimate_pitch(self.recent)))
        elif not self.syllables.speaking:
            for bark in self.barks:
                if bark.position >= MIN_BARK * RATE:
                    bark.stop(RELEASE)
        out = np.zeros(len(mic), np.float32)
        for bark in self.barks:
            part = bark.take(len(mic))
            out[:len(part)] += part
        self.barks = [bark for bark in self.barks if not bark.done]
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
    parser.add_argument("--cat", default="Goldie", help="whose voice (default Goldie)")
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
