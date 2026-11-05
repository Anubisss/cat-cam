"""Draws the app icon (a cat's head on a rounded tile) as a macOS .iconset.

    uv run python packaging/make_icon.py <out.iconset> [cat name]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from catcam import cats

SIZE = 1024
TILE_MARGIN = 0.1  # macOS icons leave this much space around the tile
CORNER = 0.22  # tile corner radius, of the tile size
TOP, BOTTOM = (60, 160, 255), (40, 70, 225)  # BGR gradient: orange to red
ICON_SIZES = [16, 32, 128, 256, 512]


def tile() -> np.ndarray:
    """The rounded gradient tile, BGRA."""
    inset = int(TILE_MARGIN * SIZE)
    side = SIZE - 2 * inset
    radius = int(CORNER * side)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.rectangle(mask, (inset + radius, inset), (SIZE - inset - radius, SIZE - inset), 255, -1)
    cv2.rectangle(mask, (inset, inset + radius), (SIZE - inset, SIZE - inset - radius), 255, -1)
    for x in (inset + radius, SIZE - inset - radius):
        for y in (inset + radius, SIZE - inset - radius):
            cv2.circle(mask, (x, y), radius, 255, -1, cv2.LINE_AA)
    t = np.linspace(0, 1, SIZE)[:, None, None]
    color = ((1 - t) * np.array(TOP) + t * np.array(BOTTOM)) * np.ones((1, SIZE, 1))
    return np.dstack([color.astype(np.uint8), mask])


def head(name: str) -> np.ndarray:
    """The cat's head cut out, BGRA, cropped to its outline."""
    sprite = cats.load(name).sprite
    ys, xs = np.nonzero(sprite[..., 3] > 20)
    return sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def over(base: np.ndarray, top: np.ndarray, x: int, y: int):
    """Draws BGRA top onto BGRA base (in place) at x, y."""
    h, w = top.shape[:2]
    a = top[..., 3:4] / 255
    region = base[y:y + h, x:x + w]
    region[..., :3] = (top[..., :3] * a + region[..., :3] * (1 - a)).astype(np.uint8)
    region[..., 3] = np.maximum(region[..., 3], top[..., 3])


def icon(name: str) -> np.ndarray:
    out = tile()
    cat = head(name)
    side = int((1 - 2 * TILE_MARGIN) * SIZE * 0.84)
    scale = side / max(cat.shape[:2])
    cat = cv2.resize(cat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shadow = np.zeros_like(cat)
    shadow[..., 3] = cv2.GaussianBlur(cat[..., 3], (0, 0), 12) // 2
    x, y = (SIZE - cat.shape[1]) // 2, SIZE - int(TILE_MARGIN * SIZE) - cat.shape[0] - 30
    over(out, shadow, x, y + 14)
    over(out, cat, x, y)
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    image = icon(sys.argv[2] if len(sys.argv) > 2 else "Woofgang")
    for size in ICON_SIZES:
        for factor, suffix in ((1, ""), (2, "@2x")):
            px = size * factor
            cv2.imwrite(str(out / f"icon_{size}x{size}{suffix}.png"),
                        cv2.resize(image, (px, px), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
