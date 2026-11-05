"""A picture or video behind the animal, in place of its portrait's backdrop.

The animal is drawn as usual; where it isn't (the renderer's coverage), the
background shows. A video plays at its own speed and loops. The picture is
cropped to fill the frame, like a photo set as a desktop background.
"""

import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}  # anything else is read as a picture
MAX_LAG = 1.0  # s a video may fall behind before it skips ahead instead of catching up


def cover(image: np.ndarray, size) -> np.ndarray:
    """image scaled to fill size (w, h), its middle cropped out."""
    w, h = size
    scale = max(w / image.shape[1], h / image.shape[0])
    fit = (max(round(image.shape[1] * scale), w), max(round(image.shape[0] * scale), h))
    scaled = cv2.resize(image, fit, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y, x = (scaled.shape[0] - h) // 2, (scaled.shape[1] - w) // 2
    return np.ascontiguousarray(scaled[y:y + h, x:x + w])


class Background:
    """An image or a looping video, as frames of size (w, h)."""

    def __init__(self, path: str | Path, size):
        self.path, self.size = Path(path), size
        self.video = None
        if self.path.suffix.lower() in VIDEO_TYPES:
            self.video = cv2.VideoCapture(str(self.path))
            if not self.video.isOpened():
                raise ValueError(f"Can't play {self.path.name}.")
            self.step = 1 / (self.video.get(cv2.CAP_PROP_FPS) or 30)
            self.next_at = None
            self.current = self._read()
        else:
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Can't open {self.path.name} as a picture.")
            self.current = cover(image, size)

    def _read(self) -> np.ndarray:
        ok, frame = self.video.read()
        if not ok:  # the end: loop
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.video.read()
            if not ok:
                raise ValueError(f"Can't play {self.path.name}.")
        return cover(frame, self.size)

    def frame(self, now: float) -> np.ndarray:
        """The picture to show at time now (s)."""
        if self.video is None:
            return self.current
        if self.next_at is None or now - self.next_at > MAX_LAG:
            self.next_at = now + self.step  # show the current frame for its time first
        while now >= self.next_at:
            self.current = self._read()
            self.next_at += self.step
        return self.current

    def close(self):
        if self.video is not None:
            self.video.release()


def behind(frame: np.ndarray, coverage: np.ndarray, background: np.ndarray) -> np.ndarray:
    """frame with background where coverage (uint8, 255 = the animal) is low."""
    a = coverage.astype(np.float32)[..., None] / 255
    return (frame * a + background * (1 - a)).astype(np.uint8)


class Chooser:
    """Asks for a picture or video with the macOS file dialog, without stalling
    the stream: start() opens it; answer() is None while it is open, then the
    chosen path, or "" if it was cancelled."""

    PROMPT = "Choose a picture or video to put behind your animal"

    def __init__(self):
        self._thread, self._result = None, None

    @property
    def idle(self) -> bool:
        return self._thread is None

    def start(self):
        self._result = None
        self._thread = threading.Thread(target=self._ask, daemon=True)
        self._thread.start()

    def answer(self) -> str | None:
        if self._thread is None or self._thread.is_alive():
            return None
        answer, self._thread = self._result or "", None
        return answer

    def _ask(self):
        script = (f'POSIX path of (choose file with prompt "{self.PROMPT}" '
                  f'of type {{"public.image", "public.movie"}})')
        try:
            done = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
            self._result = done.stdout.strip() if done.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            self._result = None
