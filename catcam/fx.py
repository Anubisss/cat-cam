"""Key-triggered effects.

An optional data file (data/fx.bin) maps keys to a full-frame picture per cat
and an optional sound. While such a key is down, even with another app such
as Zoom in front, the picture replaces the cat and the sound plays from its
start; letting go brings the cat back. Short taps (typing) don't count.

Build the data file from a folder with <Cat><Name>.png pictures and an
optional <Name>.mp3 (or .wav) sound:

    uv run python -m catcam.fx pack <folder> --key <key>=<Name> [--key ...]
"""

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from catcam.paths import FX_FILE

DELAY = 0.25  # s a key must stay down before its effect starts, so typing doesn't trigger it
_SEED = 0x6361_7463_616D  # the data file is scrambled so it isn't recognisable as media
SOUND_TYPES = (".mp3", ".wav", ".ogg", ".flac")
PICTURE_HEIGHT = 720  # pictures are stored at the output height


class Effect(NamedTuple):
    pictures: dict[str, np.ndarray]  # cat name -> BGR picture
    sound: tuple[np.ndarray, int] | None  # samples, sample rate


def _scramble(data: bytes) -> bytes:
    """Reversible: scrambling twice gives the original back."""
    pad = np.random.default_rng(_SEED).bytes(len(data))
    return (np.frombuffer(data, np.uint8) ^ np.frombuffer(pad, np.uint8)).tobytes()


def _shrink(path: Path) -> bytes:
    picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if picture is None:
        raise ValueError(f"cannot read {path}")
    scale = min(PICTURE_HEIGHT / picture.shape[0], 1.0)
    size = (round(picture.shape[1] * scale), round(picture.shape[0] * scale))
    small = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def pack(folder: Path, keys: dict[str, str], cats: list[str], out: Path = FX_FILE) -> dict:
    """Bundle the pictures and sounds for `keys` ({key: name}) into `out`."""
    files = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    manifest, buffer = {}, io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for key, name in keys.items():
            entry = {"pictures": {}, "sound": None}
            for cat in cats:
                path = files.get(f"{cat}{name}.png".lower())
                if path:
                    entry["pictures"][cat] = f"{key}/{cat}.jpg"
                    archive.writestr(entry["pictures"][cat], _shrink(path))
            sound = next((files[f"{name}{ext}".lower()] for ext in SOUND_TYPES
                          if f"{name}{ext}".lower() in files), None)
            if sound:
                entry["sound"] = f"{key}/sound{sound.suffix.lower()}"
                archive.write(sound, entry["sound"])
            if not entry["pictures"] and not entry["sound"]:
                raise ValueError(f"nothing found for {name!r} in {folder}")
            manifest[key] = entry
        archive.writestr("manifest.json", json.dumps(manifest))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_scramble(buffer.getvalue()))
    return manifest


def load(path: Path = FX_FILE) -> dict[str, Effect]:
    """{key: Effect}; empty if there is no data file."""
    if not path.is_file():
        return {}
    import soundfile  # only needed when there are effects

    effects = {}
    with zipfile.ZipFile(io.BytesIO(_scramble(path.read_bytes()))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for key, entry in manifest.items():
            pictures = {
                cat: cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                for cat, name in entry["pictures"].items()
            }
            sound = None
            if entry["sound"]:
                samples, rate = soundfile.read(io.BytesIO(archive.read(entry["sound"])), dtype="float32")
                sound = (samples, rate)
            effects[key] = Effect(pictures, sound)
    return effects


def full_frame(picture: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The picture fitted into size (w, h), the sides filled with its own backdrop."""
    w, h = size
    scale = min(w / picture.shape[1], h / picture.shape[0])
    fitted = cv2.resize(picture, (round(picture.shape[1] * scale), round(picture.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)
    edges = np.concatenate([fitted[:, :4].reshape(-1, 3), fitted[:, -4:].reshape(-1, 3)])
    frame = np.empty((h, w, 3), np.uint8)
    frame[:] = np.median(edges, axis=0)
    y, x = (h - fitted.shape[0]) // 2, (w - fitted.shape[1]) // 2
    frame[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
    return frame


# macOS only lets an app see keys typed into other apps with Input Monitoring,
# granted to the terminal app Cat Cam runs in.
INPUT_MONITORING = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
TERMINALS = {"WarpTerminal": "Warp", "Apple_Terminal": "Terminal", "iTerm.app": "iTerm",
             "vscode": "Visual Studio Code", "ghostty": "Ghostty"}
SETTINGS_EVERY = 10.0  # s between opening System Settings again


def terminal_name() -> str:
    return TERMINALS.get(os.environ.get("TERM_PROGRAM", ""), "your terminal app")


def permission_hint() -> str:
    app = terminal_name()
    return (f"Shortcuts need keyboard access while another app is in front:\n"
            f"  System Settings > Privacy & Security > Input Monitoring > turn on {app},\n"
            f"  then quit and reopen {app} and run Cat Cam again.")


class MacKeys:
    """Key presses from every app on macOS, via a listen-only event tap.

    Needs the Input Monitoring permission for the terminal app; without it the
    tap can't be created, and macOS is asked to show its permission prompt.
    """

    def __init__(self, press, release, lost):
        self.press, self.release, self.lost = press, release, lost
        self.ready = threading.Event()
        self.tap = None

    def start(self) -> bool:
        import Quartz

        if not Quartz.CGPreflightListenEventAccess():
            Quartz.CGRequestListenEventAccess()  # the system prompt (macOS shows it once per app)
            return False
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.wait(5)
        return self.tap is not None

    def _run(self):
        import Quartz

        mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
        self.tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly, mask, self._event, None,
        )
        if self.tap is None:
            self.ready.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(self.tap, True)
        self.ready.set()
        Quartz.CFRunLoopRun()

    def _event(self, proxy, kind, event, refcon):
        import Quartz

        if kind in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self.tap, True)
            self.lost()  # releases may have been missed
        elif kind in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
            _, chars = Quartz.CGEventKeyboardGetUnicodeString(event, 4, None, None)
            if chars:
                (self.press if kind == Quartz.kCGEventKeyDown else self.release)(chars)
        return event


class Effects:
    """Tracks pressed keys (from any app) and runs the effect of the one pressed."""

    def __init__(self, effects: dict[str, Effect], size: tuple[int, int]):
        self.effects = effects
        self.frames = {
            (key, cat): full_frame(picture, size)
            for key, effect in effects.items() for cat, picture in effect.pictures.items()
        }
        self.down = {}  # key -> time pressed
        self.lock = threading.Lock()
        self.playing = None  # key whose sound is playing
        self.sound_ok = True
        self.listening = False  # hears keys typed into other apps
        self.settings_opened = -SETTINGS_EVERY

    def start(self):
        """Listen to the keyboard globally (only if there are effects)."""
        if not self.effects:
            return
        if sys.platform == "darwin":
            self.listening = MacKeys(self.press, self.release, self.release_all).start()
            if not self.listening:
                print(permission_hint())
            return
        from pynput import keyboard

        def char(key):
            return getattr(key, "char", None)

        keyboard.Listener(on_press=lambda k: self.press(char(k)),
                          on_release=lambda k: self.release(char(k)), daemon=True).start()
        self.listening = True

    def needs_permission(self, window_key: int, now: float) -> bool:
        """A shortcut key pressed in the Cat Cam window while keys can't be heard:
        opens the Input Monitoring settings (now and then) and returns True."""
        if self.listening or not 0 < window_key < 255 or chr(window_key).lower() not in self.effects:
            return False
        if sys.platform == "darwin" and now - self.settings_opened > SETTINGS_EVERY:
            self.settings_opened = now
            subprocess.Popen(["open", INPUT_MONITORING])
        return True

    def press(self, char: str | None):
        char = char.lower() if char else None
        if char in self.effects:
            with self.lock:
                self.down.setdefault(char, time.monotonic())  # key repeat keeps the first time

    def release(self, char: str | None):
        with self.lock:
            self.down.pop(char.lower() if char else None, None)

    def release_all(self):
        with self.lock:
            self.down.clear()

    def active(self, now: float) -> str | None:
        """The effect to show: the most recently pressed key down long enough."""
        with self.lock:
            ready = [(t, k) for k, t in self.down.items() if now - t >= DELAY]
        return max(ready)[1] if ready else None

    def update(self, now: float, cat: str) -> np.ndarray | None:
        """Starts/stops sounds; returns the frame to send instead of the cat, if any."""
        key = self.active(now)
        if key != self.playing:
            self._sound(key)
        return self.frames.get((key, cat)) if key else None

    def stop(self):
        if self.playing is not None:
            self._sound(None)

    def _sound(self, key: str | None):
        self.playing = key
        if not self.sound_ok:
            return
        import sounddevice

        try:
            sounddevice.stop()
            sound = self.effects[key].sound if key else None
            if sound is not None:
                sounddevice.play(*sound)  # always from the start
        except sounddevice.PortAudioError as e:
            print(f"Sound disabled: {e}")
            self.sound_ok = False


def main():
    from catcam import cats

    parser = argparse.ArgumentParser(description="Build the effects data file.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("pack", help="bundle pictures and sounds from a folder")
    build.add_argument("folder", type=Path)
    build.add_argument("--key", action="append", required=True, metavar="KEY=NAME",
                       help="e.g. x=Laugh uses <Cat>Laugh.png and Laugh.mp3")
    args = parser.parse_args()
    keys = {}
    for spec in args.key:
        key, _, name = spec.partition("=")
        if len(key) != 1 or not name:
            sys.exit(f"Bad --key {spec!r}: expected a single character, '=', and a name.")
        keys[key.lower()] = name
    manifest = pack(args.folder, keys, cats.available())
    for key, entry in manifest.items():
        sound = "with sound" if entry["sound"] else "no sound"
        print(f"{key}: {len(entry['pictures'])} pictures, {sound}")
    print(f"Wrote {FX_FILE}")


if __name__ == "__main__":
    main()
