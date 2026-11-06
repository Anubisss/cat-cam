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

# Drawing, in preview px.
GAP_BELOW = 10  # between the switches' pill and this panel
EDGE = 16  # the panel keeps at least this far from the window's left edge
ROW = 44  # height of a one-row panel (as the switches' pill)
LINE = 21  # line spacing of the text
BUTTON_H = 30
INK = (32, 26, 20)  # BGR, the button's text on honey

# Stages, and what the panel says at each.
MISSING = "missing"  # no BlackHole: the button
DOWNLOADING = "downloading"
CHECKING = "checking"
OPENED = "opened"  # Apple's Installer has it
FAILED = "failed"  # the button opens the web page instead
WEBSITE = "website"  # the web page is open
RESTART = "restart"  # installed, but macOS only loads it after a restart
READY = "ready"  # BlackHole is there: nothing to show

PROMPT = "For barks and meows in your calls:"
BUTTON = "Get the voice"
WEBSITE_BUTTON = "Open BlackHole's website"
LINES = {
    CHECKING: ["Checking the download…"],
    OPENED: ["BlackHole's installer is open: click through it and enter your Mac's password.",
             "Then restart your Mac and pick “BlackHole 2ch” as the microphone in Zoom."],
    WEBSITE: ["Get “BlackHole 2ch” from the page that just opened and install it.",
              "Then restart your Mac and pick it as the microphone in Zoom."],
    RESTART: ["Restart your Mac to finish setting up the voice"],
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

    def draw(self, view: np.ndarray, right: int, top: int):
        """Draws the panel with its right edge at x = right, its top at y = top."""
        status = self.status  # read once: the download's thread replaces it
        self.button = None
        if status.stage == READY:
            return
        if status.stage in (MISSING, FAILED):
            label = BUTTON if status.stage == MISSING else WEBSITE_BUTTON
            self._button_row(view, status.problem or PROMPT, label, right, top)
        elif status.stage == DOWNLOADING:
            self._progress(view, status.progress, right, top)
        else:
            _lines(view, LINES[status.stage], right, top)

    def _button_row(self, view, prompt: str, label: str, right: int, top: int):
        """Muted words, then a honey button."""
        button_w = glass.text_width(label, 15, 600) + 32
        width = 18 + glass.text_width(prompt, 14) + 12 + button_w + 7
        x = max(right - width, EDGE)
        glass.panel(view, x, top, width, ROW, ROW // 2)
        glass.text(view, prompt, x + 18, top + ROW / 2 + 5, 14, glass.MUTED)
        bx, by = x + width - 7 - button_w, top + (ROW - BUTTON_H) // 2
        glass.paint(view, glass.rounded_mask(button_w, BUTTON_H, BUTTON_H // 2), bx, by, glass.HONEY)
        glass.text(view, label, bx + button_w / 2, by + BUTTON_H / 2 + 5, 15, INK, 600, align="center")
        self.button = (bx, by, button_w, BUTTON_H)

    def _progress(self, view, share: float, right: int, top: int):
        """Downloading, with a bar filling up."""
        words = "Downloading BlackHole…"
        bar_w = 120
        width = 18 + glass.text_width(words, 14, 500) + 14 + bar_w + 18
        x = max(right - width, EDGE)
        glass.panel(view, x, top, width, ROW, ROW // 2)
        glass.text(view, words, x + 18, top + ROW / 2 + 5, 14, weight=500)
        bx, by = x + width - 18 - bar_w, top + ROW // 2 - 3
        glass.paint(view, glass.rounded_mask(bar_w, 6, 3), bx, by, glass.WHITE, 0.22)
        filled = int(round(bar_w * share))
        if filled >= 6:
            glass.paint(view, glass.rounded_mask(filled, 6, 3), bx, by, glass.HONEY)


def _lines(view, lines: list[str], right: int, top: int):
    """A glass panel with lines of text, its right edge at x = right."""
    width = max(glass.text_width(line, 14, 500) for line in lines) + 36
    height = ROW if len(lines) == 1 else 22 + LINE * len(lines)
    x = max(right - width, EDGE)
    glass.panel(view, x, top, width, height, min(height // 2, ROW // 2))
    first = top + height / 2 + 5 - LINE * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        glass.text(view, line, x + 18, first + LINE * i, 14, glass.WHITE if i == 0 else glass.MUTED, 500)
