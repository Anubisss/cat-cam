"""Turns paw photos into the paw sprites a cat uses.

all_cats/<Name>PawsFront.png shows the cat's front legs from the front (the top
of the paws, toes down) and <Name>PawsBack.png the soles (toes up). Each photo
shows the left paw left of a purple dividing line and the right paw right of it.
They become assets/cats/<Name>/paw_<view>_<hand>.png: cut out, toes up, the
arm leaving at the bottom, scaled to a fixed paw width.
"""

from pathlib import Path

import cv2
import numpy as np

VIEWS = {"front": "PawsFront", "back": "PawsBack"}  # view -> portrait file suffix
HANDS = ("left", "right")
PAW_WIDTH = 240  # px, stored paw width
DIVIDER_MARGIN = 6  # px cut away on each side of the purple line
PAW_ROWS = 0.4  # the paw is the widest part of the top of the cutout (fraction of its height)


def paw_files(name: str, assets: Path) -> dict[tuple[str, str], Path]:
    """{(view, hand): path} of a cat's prepared paw sprites."""
    return {(v, h): assets / name / f"paw_{v}_{h}.png" for v in VIEWS for h in HANDS}


def is_paw_photo(stem: str) -> bool:
    return stem.endswith(tuple(VIEWS.values()))


def has_photos(name: str, portraits: Path) -> list[str]:
    """The views a cat has paw photos for."""
    return [v for v, suffix in VIEWS.items() if (portraits / f"{name}{suffix}.png").is_file()]


def divider(bgr: np.ndarray) -> int:
    """Column of the purple dividing line (the image middle if there is none)."""
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    purple = (b > 90) & (r > 60) & (g < 0.7 * np.minimum(b, r)) & (b - g > 50)
    counts = purple.sum(axis=0)
    w = bgr.shape[1]
    middle = slice(w // 4, 3 * w // 4)
    column = w // 4 + int(np.argmax(counts[middle]))
    return column if counts[column] > 0.3 * bgr.shape[0] else w // 2


def halves(bgr: np.ndarray) -> dict[str, np.ndarray]:
    x = divider(bgr)
    return {"left": bgr[:, :x - DIVIDER_MARGIN], "right": bgr[:, x + DIVIDER_MARGIN + 1:]}


def paw_width(alpha: np.ndarray) -> int:
    """Width of the paw: the widest row in the top part of an upright cutout."""
    rows = alpha[: max(int(PAW_ROWS * alpha.shape[0]), 1)] > 127
    return int(rows.sum(axis=1).max())


def upright(cutout: np.ndarray, toes_down: bool) -> np.ndarray:
    """Toes up, cropped to the paw, scaled to PAW_WIDTH."""
    if toes_down:
        cutout = cutout[::-1]
    ys, xs = np.nonzero(cutout[..., 3] > 20)
    cutout = cutout[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = PAW_WIDTH / max(paw_width(cutout[..., 3]), 1)
    size = (round(cutout.shape[1] * scale), round(cutout.shape[0] * scale))
    return cv2.resize(cutout, size, interpolation=cv2.INTER_AREA)


def prepare_paws(name: str, portraits: Path, assets: Path, cut_out) -> list[str]:
    """Writes the paw sprites from the cat's paw photos; returns the views done.
    cut_out: BGR -> BGRA background removal."""
    done = []
    (assets / name).mkdir(parents=True, exist_ok=True)
    for view, suffix in VIEWS.items():
        photo = portraits / f"{name}{suffix}.png"
        if not photo.is_file():
            continue
        bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read {photo}")
        for hand, half in halves(bgr).items():
            sprite = upright(cut_out(np.ascontiguousarray(half)), toes_down=view == "front")
            cv2.imwrite(str(paw_files(name, assets)[view, hand]), sprite)
        done.append(view)
    return done
