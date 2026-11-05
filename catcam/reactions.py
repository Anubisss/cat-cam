"""Emoji reactions: a number key puts an emoji beside the animal for a moment.

The emoji are drawn from the Mac's own emoji font at runtime, so they look
like Apple's without their pictures being part of this repo. Without that
font (or an emoji it lacks) the key simply does nothing. The emoji pops in,
moves the way its gesture does, then fades away; the same key again starts
it over, another key swaps it for its emoji.
"""

import functools
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
GLYPH_SIZE = 160  # px; the font's largest built-in picture size, so the emoji are sharpest

# (name, emoji) in key order: key 1 is the first.
REACTIONS = (
    ("point", "\U0001FAF5"),  # index pointing at the viewer
    ("raise", "\U0001F64C"),  # raising hands
    ("pray", "\U0001F64F"),  # folded hands
    ("down", "\U0001F44E"),  # thumbs down
    ("clap", "\U0001F44F"),  # clapping hands
    ("pinch", "\U0001F90C"),  # pinched fingers
    ("palm", "\U0001F91A"),  # raised back of hand
    ("wave", "\U0001F44B"),  # waving hand
)

SIZE = 0.3  # emoji height, in frame heights
CENTRE = (0.86, 0.6)  # where the emoji rests: beside the animal's cheek, in frame widths and heights
DURATION = 2.4  # s an emoji stays, from pressing its key until it is gone
POP_IN = 0.3  # s it takes to grow in, overshooting a little
FADE_OUT = 0.45  # s it takes to fade away at the end
OVERSHOOT = 1.7  # how far growing in overshoots (0 = not at all; 1.7 is about 10 % too big)
SHRINK_OUT = 0.15  # how much smaller it gets while fading, as a fraction of its size

# Where each emoji turns and grows about (fractions of its picture, from the
# top left); the centre unless listed.
PIVOTS = {
    "pray": (0.5, 1.0),  # the wrists, so it bows forward
    "palm": (0.5, 1.0),  # the wrist, so it sways
    "wave": (0.8, 0.95),  # the wrist, bottom right in Apple's picture
}

# The gestures. Distances are in emoji heights, angles in degrees, rates in
# times per second.
POKE = (0.16, 2.2)  # point: grows toward the viewer by this much, this often
BOUNCE = (0.12, 2.2)  # raise: jumps up this high, this often
BOW = (0.05, 4.0, 1.0)  # pray: dips this far, tilts this much, this often
DROP = (0.2, 0.2)  # down: sinks this far, starting this many s after its key
CLAP = (0.18, 3.0)  # clap: squeezes this much narrower, this often
SHAKE = (0.06, 4.5)  # pinch: shakes up and down this far, this often
SWAY = (8.0, 0.9)  # palm: tips this far either way, this often
WAVE = (18.0, 2.0)  # wave: swings this far either way, this often


@dataclass(frozen=True)
class Pose:
    """How an emoji is drawn at one moment: moved by (dx, dy) emoji heights
    (down is positive), sized by scale times (sx, sy), turned by angle degrees
    counterclockwise, with alpha 0..1 opacity."""
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    sx: float = 1.0
    sy: float = 1.0
    angle: float = 0.0
    alpha: float = 1.0


def reaction_for(key: int) -> str | None:
    """The reaction a key code from cv2.waitKey asks for, if any."""
    index = key - ord("1")
    return REACTIONS[index][0] if 0 <= index < len(REACTIONS) else None


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _ease_out_back(u: float) -> float:
    """0 -> 1, going a little past 1 before settling."""
    u -= 1
    return 1 + u * u * ((OVERSHOOT + 1) * u + OVERSHOOT)


def _gesture(name: str, t: float) -> Pose:
    """The emoji's own movement, t s after its key."""
    if name == "point":
        size, rate = POKE
        return Pose(scale=1 + size * max(math.sin(2 * math.pi * rate * t), 0.0) ** 2)
    if name == "raise":
        height, rate = BOUNCE
        return Pose(dy=-height * abs(math.sin(math.pi * rate * t)))
    if name == "pray":
        depth, tilt, rate = BOW
        bow = (1 - math.cos(2 * math.pi * rate * t)) / 2
        return Pose(dy=depth * bow, sy=1 - depth * bow, angle=-tilt * bow)
    if name == "down":
        depth, delay = DROP
        u = max(t - delay, 0.0)
        return Pose(dy=depth * (1 - math.exp(-6 * u) * math.cos(9 * u)))
    if name == "clap":
        size, rate = CLAP
        squeeze = max(math.sin(2 * math.pi * rate * t), 0.0) ** 2
        return Pose(sx=1 - size * squeeze, sy=1 + size * squeeze / 3)
    if name == "pinch":
        distance, rate = SHAKE
        return Pose(dy=distance * math.sin(2 * math.pi * rate * t))
    if name == "palm":
        angle, rate = SWAY
        return Pose(angle=angle * math.sin(2 * math.pi * rate * t))
    if name == "wave":
        angle, rate = WAVE
        return Pose(angle=angle * math.sin(2 * math.pi * rate * t))
    raise ValueError(f"unknown reaction {name!r}")


def pose(name: str, t: float) -> Pose:
    """The emoji's pose t s after its key: its gesture, grown in at the start
    and faded out at the end. Alpha is 0 once it is over."""
    if not 0 <= t < DURATION:
        return Pose(scale=0.0, alpha=0.0)
    gesture = _gesture(name, t)
    grow = _ease_out_back(_clamp(t / POP_IN))
    fade = _clamp((DURATION - t) / FADE_OUT)
    return Pose(
        dx=gesture.dx, dy=gesture.dy,
        scale=gesture.scale * grow * (1 - SHRINK_OUT * (1 - fade)),
        sx=gesture.sx, sy=gesture.sy, angle=gesture.angle,
        alpha=_clamp(t / (POP_IN / 2)) * fade,
    )


@functools.cache
def _font(path: str) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, GLYPH_SIZE)
    except OSError:  # no such font (not a Mac, or an older one)
        return None


@functools.cache
def sprite(emoji: str, font_path: str = EMOJI_FONT) -> np.ndarray | None:
    """The emoji as a square float32 picture, BGR premultiplied by alpha plus
    alpha, all 0..1; None if the font or this emoji in it is missing."""
    font = _font(font_path)
    if font is None:
        return None
    picture = Image.new("RGBA", (GLYPH_SIZE, GLYPH_SIZE), (0, 0, 0, 0))
    try:
        ImageDraw.Draw(picture).text((0, 0), emoji, font=font, embedded_color=True)
    except OSError:  # a font without colour pictures
        return None
    rgba = np.asarray(picture, np.float32) / 255
    if rgba[..., 3].max() == 0:  # the font has no picture for it
        return None
    alpha = rgba[..., 3:]
    return np.concatenate([rgba[..., 2::-1] * alpha, alpha], axis=2)


def _transform(p: Pose, pivot: tuple[float, float], frame_size: tuple[int, int]) -> np.ndarray:
    """2x3 matrix from the emoji's picture to frame px."""
    w, h = frame_size
    height = SIZE * h
    k = height / GLYPH_SIZE
    q = np.array(pivot) * GLYPH_SIZE  # the pivot in the picture
    # Where the pivot lands in the frame: the picture's centre at CENTRE, plus the move.
    target = np.array([CENTRE[0] * w, CENTRE[1] * h]) + np.array([p.dx, p.dy]) * height \
        + k * (q - GLYPH_SIZE / 2)
    a = math.radians(p.angle)
    rotation = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    linear = rotation @ np.diag([k * p.scale * p.sx, k * p.scale * p.sy])
    return np.hstack([linear, (target - linear @ q)[:, None]])


def composite(frame: np.ndarray, picture: np.ndarray, p: Pose, pivot=(0.5, 0.5)) -> np.ndarray:
    """A copy of frame (BGR uint8) with picture (see sprite) drawn on it in pose p."""
    if p.alpha <= 0 or p.scale <= 0:
        return frame
    h, w = frame.shape[:2]
    matrix = _transform(p, pivot, (w, h))
    corners = np.array([[0, 0, 1], [GLYPH_SIZE, 0, 1], [0, GLYPH_SIZE, 1], [GLYPH_SIZE, GLYPH_SIZE, 1]])
    spots = corners @ matrix.T
    x0, y0 = np.maximum(np.floor(spots.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(spots.max(axis=0)).astype(int) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:  # wholly outside the frame
        return frame
    matrix[:, 2] -= (x0, y0)
    drawn = cv2.warpAffine(picture, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = frame.copy()
    patch = out[y0:y1, x0:x1].astype(np.float32)
    cover = p.alpha * drawn[..., 3:]
    out[y0:y1, x0:x1] = np.clip(patch * (1 - cover) + p.alpha * 255 * drawn[..., :3], 0, 255).astype(np.uint8)
    return out


class Reactions:
    """The reaction showing right now, if any."""

    def __init__(self, font_path: str = EMOJI_FONT):
        self.font_path = font_path
        self.name: str | None = None
        self.since = 0.0  # when its key was pressed (s)

    def play(self, name: str, now: float):
        """Shows the named reaction from now on, from its start (also if it was showing)."""
        if self._picture(name) is not None:
            self.name, self.since = name, now

    def showing(self, now: float) -> str | None:
        """The name of the reaction showing at now, if any."""
        return self.name if self.name is not None and 0 <= now - self.since < DURATION else None

    def draw(self, frame: np.ndarray, now: float) -> np.ndarray:
        """The frame with the reaction on it: a new array while one shows, else frame itself."""
        name = self.showing(now)
        if name is None:
            self.name = None
            return frame
        p = pose(name, now - self.since)
        return composite(frame, self._picture(name), p, PIVOTS.get(name, (0.5, 0.5)))

    def _picture(self, name: str) -> np.ndarray | None:
        return sprite(dict(REACTIONS)[name], self.font_path)
