"""macOS-style "liquid glass" drawn into OpenCV images: the preview's controls
and the cat picker float over the live picture as smoky glass that blurs what
is behind it, bends it at the edges like a lens and catches the light on its
rim. Text is set in the system's SF Pro (and SF Pro Rounded for names).

Everything draws in place into a BGR uint8 image; shapes, masks and text are
cached by size, so drawing a few panels a frame stays cheap.
"""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = {
    "text": "/System/Library/Fonts/SFNS.ttf",
    "rounded": "/System/Library/Fonts/SFNSRounded.ttf",
}
BLUR = 7.0  # px, how much the glass blurs what is behind it
VIBRANCY = 1.4  # saturation of what shows through
SMOKE = (0.74, 8)  # the glass darkens what is behind it: scale, then adds this grey
LENS = 13  # px band along the edge where the glass bends the picture...
LENS_PULL = 1.5  # ...sampling this share of the band further in, so the edge magnifies
RIM = 1.0  # brightness of the light caught on the rim (0..1)
SHADOW = (0.3, 7, 10)  # drop shadow: strength, offset down (px), blur (px)

WHITE = (255, 255, 255)
HONEY = (72, 182, 255)  # BGR #FFB648, retriever gold: what is on, hovered or live
MUTED = (215, 215, 215)


@lru_cache(maxsize=64)
def _font(family: str, size: int, weight: int) -> ImageFont.FreeTypeFont:
    try:
        font = ImageFont.truetype(FONTS[family], size)
        axes = {a["name"]: a for a in font.get_variation_axes()}
        values = []
        for name, axis in axes.items():
            if name == b"Weight":
                values.append(weight)
            elif name == b"Optical Size":
                values.append(min(max(size, axis["minimum"]), axis["maximum"]))
            else:
                values.append(axis["default"])
        font.set_variation_by_axes(values)
        return font
    except (OSError, KeyError):  # not on a Mac: the app still runs, just plainer
        return ImageFont.load_default(size)


@lru_cache(maxsize=256)
def _text_patch(text: str, size: int, weight: int, family: str):
    """The text as an alpha mask (float 0..1) and the baseline's offset from its top."""
    font = _font(family, size, weight)
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    w, h = max(right - left, 1) + 2, max(bottom - top, 1) + 2
    image = Image.new("L", (w, h), 0)
    ImageDraw.Draw(image).text((1 - left, 1 - top), text, font=font, fill=255, anchor="ls")
    return np.asarray(image, np.float32) / 255, 1 - top, left - 1


def text_width(text: str, size: int, weight: int = 400, family: str = "text") -> int:
    return int(round(_font(family, size, weight).getlength(text)))


def text(view, s: str, x: float, baseline: float, size: int, color=WHITE, weight: int = 400,
         family: str = "text", align: str = "left"):
    """Draws s with its baseline at y = baseline, starting at x (align "left"),
    centred on it ("center") or ending there ("right")."""
    alpha, ascent, left = _text_patch(s, size, weight, family)
    if align != "left":
        x -= text_width(s, size, weight, family) / (2 if align == "center" else 1)
    paint(view, alpha, int(round(x + left)), int(round(baseline - ascent)), color)


def paint(view, alpha, x, y, color, opacity=1.0):
    """Paints color into view at (x, y) through alpha (float 0..1)."""
    h, w = alpha.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, view.shape[1]), min(y + h, view.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    roi = view[y0:y1, x0:x1].astype(np.float32)
    view[y0:y1, x0:x1] = (roi * (1 - a) + np.asarray(color, np.float32) * a).astype(np.uint8)


@lru_cache(maxsize=64)
def rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Anti-aliased rounded rectangle, float 0..1."""
    ss = 4
    mask = np.zeros((h * ss, w * ss), np.uint8)
    r = min(radius, w // 2, h // 2) * ss
    cv2.rectangle(mask, (r, 0), (w * ss - r, h * ss), 255, -1)
    cv2.rectangle(mask, (0, r), (w * ss, h * ss - r), 255, -1)
    for cx in (r, w * ss - r - 1):
        for cy in (r, h * ss - r - 1):
            cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


@lru_cache(maxsize=32)
def _optics(w: int, h: int, radius: int):
    """Per panel size: where each pixel samples the blurred picture from (the lens
    at the edge), and how much rim light it catches."""
    inside = (rounded_mask(w, h, radius) > 0.5).astype(np.uint8)
    edge = cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    gx = cv2.Sobel(edge, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.maximum(np.hypot(gx, gy), 1e-6)
    inward_x, inward_y = gx / norm, gy / norm
    bend = LENS * LENS_PULL * np.clip(1 - edge / LENS, 0, 1) ** 2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xs + inward_x * bend).astype(np.float32)
    map_y = (ys + inward_y * bend).astype(np.float32)
    # Light from the top left: the rim shines where it faces it, and where it
    # faces the opposite way (light passing through), less elsewhere.
    facing = -(inward_x * -0.6 + inward_y * -0.8)
    glint = 0.25 + 0.75 * np.maximum(facing, -facing * 0.6) ** 2
    rim = np.clip(2.2 - edge, 0, 1) * glint * RIM
    rim += np.clip(1 - edge / 5, 0, 1) * 0.1  # the glass's thickness glows faintly all round
    sheen = np.clip(1 - ys / (0.6 * h), 0, 1) * 0.08  # a faint brighter top half
    return map_x, map_y, rim.astype(np.float32), sheen.astype(np.float32)


@lru_cache(maxsize=32)
def _shadow(w: int, h: int, radius: int):
    strength, _, blur = SHADOW
    pad = int(blur * 3)
    mask = np.pad(rounded_mask(w, h, radius), pad)
    return cv2.GaussianBlur(mask, (0, 0), blur) * strength, pad


def panel(view, x: int, y: int, w: int, h: int, radius: int, glow: float = 0.0):
    """A glass panel over view at (x, y), w x h; glow brightens the rim (hover)."""
    shade, pad = _shadow(w, h, radius)
    paint(view, shade, x - pad, y - pad + SHADOW[1], (0, 0, 0))

    m = LENS + 2
    vh, vw = view.shape[:2]
    x0, y0, x1, y1 = x - m, y - m, x + w + m, y + h + m
    behind = cv2.copyMakeBorder(
        view[max(y0, 0):min(y1, vh), max(x0, 0):min(x1, vw)],
        max(-y0, 0), max(y1 - vh, 0), max(-x0, 0), max(x1 - vw, 0), cv2.BORDER_REFLECT)
    blurred = _soften(behind, BLUR)
    map_x, map_y, rim, sheen = _optics(w, h, radius)
    seen = cv2.remap(blurred, map_x + m, map_y + m, cv2.INTER_LINEAR).astype(np.float32)
    grey = seen.mean(axis=2, keepdims=True)
    seen = grey + (seen - grey) * VIBRANCY
    scale, add = SMOKE
    seen = seen * scale + add
    seen += (255 - seen) * sheen[..., None]
    light = np.minimum(rim * (1 + glow), 1)[..., None]
    seen = seen * (1 - light) + 255 * light
    _blend_rgb(view, np.clip(seen, 0, 255), rounded_mask(w, h, radius), x, y)


def _blend_rgb(view, image, alpha, x, y):
    h, w = alpha.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, view.shape[1]), min(y + h, view.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None]
    roi = view[y0:y1, x0:x1].astype(np.float32)
    view[y0:y1, x0:x1] = (image[y0 - y:y1 - y, x0 - x:x1 - x] * a + roi * (1 - a)).astype(np.uint8)


@lru_cache(maxsize=64)
def _outline(w: int, h: int, radius: int) -> np.ndarray:
    """A 1 px outline of a rounded rectangle."""
    inner = np.pad(rounded_mask(w - 2, h - 2, max(radius - 1, 0)), 1)
    return np.clip(rounded_mask(w, h, radius) - inner, 0, 1)


def ring(view, x: int, y: int, w: int, h: int, radius: int, color, width: int = 2):
    """An outline around a rounded rectangle, width px outside it."""
    outer = rounded_mask(w + 2 * width, h + 2 * width, radius + width)
    paint(view, np.clip(outer - np.pad(rounded_mask(w, h, radius), width), 0, 1), x - width, y - width, color)


SWITCH = (38, 22)  # w, h of a toggle switch


def switch(view, x: int, y: int, on: bool):
    """A macOS toggle switch with its top left at (x, y): honey when on."""
    w, h = SWITCH
    track = rounded_mask(w, h, h // 2)
    if on:
        paint(view, track, x, y, HONEY)
    else:
        paint(view, track, x, y, WHITE, 0.22)
    knob = h - 4
    kx = x + (w - knob - 2 if on else 2)
    disc = rounded_mask(knob, knob, knob // 2)
    paint(view, cv2.GaussianBlur(np.pad(disc, 3), (0, 0), 1.5), kx - 3, y + 3, (0, 0, 0), 0.35)
    paint(view, disc, kx, y + 2, WHITE)


def keycap(view, x: int, y: int, key: str, size: int = 20) -> int:
    """A small key with key on it, top left at (x, y); returns its width."""
    w = max(size, text_width(key, 12, 600) + 10)
    paint(view, rounded_mask(w, size, 5), x, y, WHITE, 0.16)
    paint(view, _outline(w, size, 5), x, y, WHITE, 0.35)
    text(view, key, x + w / 2, y + size * 0.72, 12, WHITE, 600, align="center")
    return w


def dot(view, cx: int, cy: int, r: int, color):
    disc = rounded_mask(2 * r, 2 * r, r)
    paint(view, disc, cx - r, cy - r, color)


def _soften(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, done at half size: just as soft, a quarter of the work."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 2), (w, h), interpolation=cv2.INTER_LINEAR)


def backdrop(frame: np.ndarray, size) -> np.ndarray:
    """A full-window picture to put glass on: the frame softened and dimmed."""
    w, h = size
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.5)
    return cv2.resize(cv2.convertScaleAbs(small, alpha=0.72), size, interpolation=cv2.INTER_LINEAR)
