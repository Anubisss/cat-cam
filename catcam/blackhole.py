"""The "Get the voice" button: setting up BlackHole without the Terminal.

The animal's voice goes to BlackHole 2ch, a free virtual microphone made by
Existential Audio (https://existential.audio/blackhole/). We may not ship
its installer, so when the Mac has no BlackHole the preview window offers a
button instead. Only when it is clicked, the official installer is fetched
from Existential Audio's own site, checked (its SHA-256, and that Apple says
Existential Audio signed it), and handed to Apple's Installer app, where you
click through and type your own password. Nothing is installed behind your
back: no sudo, no silent install, and a file that fails a check is deleted,
never opened. If anything goes wrong, the button opens BlackHole's web page
instead.

Like the picker, this is only an image plus hit-testing; the download runs
on a thread of its own so the preview keeps streaming.
"""

import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np

from catcam import glass
from catcam.voice import DRIVER, VIRTUAL_MIC, find_device

# The installer Homebrew's blackhole-2ch cask uses, its SHA-256 and who must
# have signed it (as `pkgutil --check-signature` names the certificate).
# To move to a new BlackHole version: take the new url and sha256 from the
# cask (`brew info --cask blackhole-2ch`, or Casks/b/blackhole-2ch.rb in
# Homebrew/homebrew-cask), download it once and check that
# `shasum -a 256 <pkg>` gives that sha256 and `pkgutil --check-signature <pkg>`
# shows SIGNER as certificate 1, then update URL and SHA256 here and the
# version in the docs.
URL = "https://existential.audio/downloads/BlackHole2ch-0.7.1.pkg"
SHA256 = "57b540f27a3e29c37e310e01bee0fdfab76733087e47f997ef9dccf851400dcf"
SIGNER = "Developer ID Installer: Existential Audio Inc. (Q5C99V536K)"
PAGE = "https://existential.audio/blackhole/"  # opened in the browser when the download fails
TIMEOUT = 20.0  # s to wait for the site to answer
MAX_BYTES = 20_000_000  # bigger than this is not BlackHole's installer (0.7.1 is 103 KB)
CHUNK = 16384  # bytes read at a time, for the progress

# Drawing: a small chip in the window's bottom right corner, in preview px.
CHIP_H = 26  # height of the chip (as the bar with the keys)
TEXT = 12  # its text size
BUTTON_H = 20  # height of the honey button inside it
INK = (32, 26, 20)  # BGR, the button's text on honey
SHRINK_AFTER = 20.0  # s after the installer or web page opened, the chip only says what is left

# Stages, and what the chip says at each.
MISSING = "missing"  # no BlackHole: the button
DOWNLOADING = "downloading"
CHECKING = "checking"
OPENED = "opened"  # Apple's Installer has it
FAILED = "failed"  # the button opens the web page instead
WEBSITE = "website"  # the web page is open
RESTART = "restart"  # installed, but macOS only loads it after a restart
READY = "ready"  # BlackHole is there: nothing to show

BUTTON = "Get the voice"
PROBLEM, WEBSITE_BUTTON = "Couldn't get BlackHole", "Website"  # what went wrong goes to the log
LEFT_TO_DO = "Restart your Mac for the voice"
NOTES = {
    CHECKING: "Checking…",
    OPENED: "Finish in the installer, then restart your Mac",
    WEBSITE: "Install “BlackHole 2ch”, then restart your Mac",
    RESTART: LEFT_TO_DO,
}


class Status(NamedTuple):
    stage: str
    progress: float = 0.0  # 0..1 of the download
    problem: str = ""  # what went wrong, when FAILED


class Failed(Exception):
    """A step failed; the message says why, in plain words."""


def stage_for(devices, driver: Path = DRIVER) -> str:
    """READY with a BlackHole output among devices (sounddevice's list),
    RESTART if its driver is installed but not loaded yet, else MISSING."""
    if find_device(devices, VIRTUAL_MIC, "output") is not None:
        return READY
    return RESTART if driver.exists() else MISSING


def download(url: str, path: Path, progress=lambda share: None) -> Path:
    """Saves url to path, telling progress (0..1) as it goes."""
    request = urllib.request.Request(url, headers={"User-Agent": "Cat Cam"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, path.open("wb") as file:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(CHUNK):
                done += len(chunk)
                if done > MAX_BYTES:
                    raise Failed("The download was far too big to be BlackHole, so it wasn't opened.")
                file.write(chunk)
                if total:
                    progress(min(done / total, 1.0))
    except OSError as e:  # no internet, the site down, a timeout (urllib's errors are OSErrors)
        raise Failed("Couldn't download BlackHole. Are you online?") from e
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_by(path: Path) -> str | None:
    """Who signed the installer at path, as pkgutil names the certificate,
    if macOS trusts the signature; else None."""
    try:
        result = subprocess.run(["pkgutil", "--check-signature", str(path)],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.strip().startswith("1. "):  # the first certificate in the chain: the signer's
            return line.strip()[3:]
    return None


def fetch_installer(folder: Path, progress=lambda share: None, checking=lambda: None) -> Path:
    """Downloads BlackHole's installer into folder and checks it; returns its
    path. A file that fails a check is deleted."""
    path = download(URL, folder / URL.rsplit("/", 1)[1], progress)
    checking()
    if sha256(path) != SHA256:
        path.unlink(missing_ok=True)
        raise Failed("The download isn't the BlackHole installer it should be, so it wasn't opened.")
    if signed_by(path) != SIGNER:
        path.unlink(missing_ok=True)
        raise Failed("The download isn't signed by BlackHole's makers, so it wasn't opened.")
    return path


def open_with_mac(target: str):
    """Opens a file in its app (an installer in Apple's Installer) or a web page in the browser."""
    subprocess.run(["open", target], check=True, capture_output=True, timeout=30)


class VoiceSetup:
    """The button's state, and the download behind it."""

    def __init__(self, stage: str):
        self.status = Status(stage)  # replaced whole, so the download's thread can set it
        self.button = None  # (x, y, w, h) of the button as last drawn, for clicks
        self._thread = None
        self._shown = (None, 0.0)  # the stage last drawn, and since when (time.monotonic())

    @classmethod
    def for_this_mac(cls) -> "VoiceSetup":
        import sounddevice

        return cls(stage_for(sounddevice.query_devices()))

    def click(self, x: int, y: int) -> bool:
        """Handles a click at preview px (x, y); True if it was on the button."""
        if self.button is None:
            return False
        bx, by, bw, bh = self.button
        if not (bx <= x < bx + bw and by <= y < by + bh):
            return False
        if self.status.stage == MISSING and self._thread is None:
            self.status = Status(DOWNLOADING)
            self._thread = threading.Thread(target=self._install, name="blackhole", daemon=True)
            self._thread.start()
        elif self.status.stage == FAILED:
            try:
                open_with_mac(PAGE)
                self.status = Status(WEBSITE)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"Couldn't open {PAGE}: {e}")
        return True

    def _install(self):
        try:
            folder = Path(tempfile.mkdtemp(prefix="Cat Cam "))
            try:
                path = fetch_installer(folder, lambda share: self._set(Status(DOWNLOADING, share)),
                                       lambda: self._set(Status(CHECKING)))
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            open_with_mac(str(path))
            self._set(Status(OPENED))
        except Failed as e:
            print(f"Setting up the voice: {e}")
            self._set(Status(FAILED, problem=str(e)))
        except Exception as e:  # e.g. `open` failing: say so rather than leave "Downloading" up
            print(f"Setting up the voice failed: {e!r}")
            self._set(Status(FAILED, problem="Couldn't open BlackHole's installer."))

    def _set(self, status: Status):
        self.status = status

    def draw(self, view: np.ndarray, right: int, bottom: int,
             now: float | None = None) -> tuple[int, int, int, int] | None:
        """Draws the chip with its bottom right corner at (right, bottom); returns its
        (x, y, w, h), or None with nothing to show. now: time.monotonic() by default."""
        status = self.status  # read once: the download's thread replaces it
        now = time.monotonic() if now is None else now
        if self._shown[0] != status.stage:
            self._shown = (status.stage, now)
        self.button = None
        top = bottom - CHIP_H
        if status.stage == READY:
            return None
        if status.stage == MISSING:
            return self._button(view, "", BUTTON, right, top)
        if status.stage == FAILED:
            return self._button(view, PROBLEM, WEBSITE_BUTTON, right, top)
        if status.stage == DOWNLOADING:
            note = f"Downloading… {round(status.progress * 100)}%"
            return glass.chip(view, note, right, top, CHIP_H, TEXT)
        if status.stage in (OPENED, WEBSITE) and now - self._shown[1] > SHRINK_AFTER:
            return glass.chip(view, LEFT_TO_DO, right, top, CHIP_H, TEXT, glass.MUTED)
        return glass.chip(view, NOTES[status.stage], right, top, CHIP_H, TEXT)

    def _button(self, view, words: str, label: str, right: int, top: int) -> tuple[int, int, int, int]:
        """Muted words, if any, then a honey button, on a glass chip."""
        inset = (CHIP_H - BUTTON_H) // 2
        button_w = glass.text_width(label, TEXT, 600) + BUTTON_H
        lead = CHIP_H // 2 + glass.text_width(words, TEXT) + 8 if words else inset  # before the button
        width = lead + button_w + inset
        x = right - width
        glass.panel(view, x, top, width, CHIP_H, CHIP_H // 2)
        if words:
            glass.text(view, words, x + CHIP_H // 2, top + CHIP_H / 2 + TEXT * 0.36, TEXT, glass.MUTED)
        bx, by = x + lead, top + inset
        glass.paint(view, glass.rounded_mask(button_w, BUTTON_H, BUTTON_H // 2), bx, by, glass.HONEY)
        glass.text(view, label, bx + button_w / 2, by + BUTTON_H / 2 + TEXT * 0.36, TEXT, INK, 600,
                   align="center")
        self.button = (bx, by, button_w, BUTTON_H)
        return x, top, width, CHIP_H
