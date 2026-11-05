"""The prepared cats: discovery and loading of their assets."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from catcam import dog_lips
from catcam.depth import depth_file
from catcam.paths import ASSETS, PORTRAITS
from catcam.paw_photos import is_paw_photo, paw_files
from catcam.voice import load_sounds


@dataclass(frozen=True)
class CatAssets:
    name: str
    sprite: np.ndarray  # BGRA cutout (a dog's with dog lips, see catcam.dog_lips)
    points: np.ndarray  # (478, 3) face landmarks in sprite px (z uses the x scale)
    background: np.ndarray  # (3,) BGR colour of the portrait's backdrop
    paws: dict  # {(view, hand): BGRA paw photo}, see catcam.paw_photos; empty if none
    sounds: list  # mono sounds from voice/, see catcam.voice; empty if none
    depth: np.ndarray | None = None  # (h, w) 0..1, nearer = larger, see catcam.depth; None if not prepared
    kind: str = "cat"  # "dog" if kind.txt says so: dog lips, teeth and tongue


def available(assets: Path = ASSETS) -> list[str]:
    """Names of the cats that are prepared, sorted."""
    if not assets.is_dir():
        return []
    return sorted(
        d.name for d in assets.iterdir()
        if (d / "cutout.png").is_file() and (d / "face.npz").is_file()
    )


def portraits(folder: Path = PORTRAITS) -> dict[str, Path]:
    """{name: path} of the cat portraits (not the paw photos next to them)."""
    return {p.stem: p for p in sorted(folder.glob("*.png")) if not is_paw_photo(p.stem)}


def unprepared(folder: Path = PORTRAITS, assets: Path = ASSETS) -> list[str]:
    """Portraits that have no prepared assets yet."""
    ready = set(available(assets))
    return sorted(name for name in portraits(folder) if name not in ready)


def load(name: str, assets: Path = ASSETS) -> CatAssets:
    folder = assets / name
    sprite = cv2.imread(str(folder / "cutout.png"), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise ValueError(f"{folder / 'cutout.png'} is missing or has no transparency")
    with np.load(folder / "face.npz") as face:
        points = face["points"].astype(np.float64)
        background = face["background"].astype(np.uint8)
    if points.shape != (478, 3):
        raise ValueError(f"{folder / 'face.npz'} has {points.shape} landmarks, expected (478, 3)")
    paws = {}
    for key, path in paw_files(name, assets).items():
        if path.is_file():
            paws[key] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = None
    if depth_file(name, assets).is_file():
        depth = cv2.imread(str(depth_file(name, assets)), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        if depth.shape != sprite.shape[:2]:
            raise ValueError(f"{depth_file(name, assets)} is not the size of the cutout")
    kind = (folder / "kind.txt").read_text().strip() if (folder / "kind.txt").is_file() else "cat"
    if kind == "dog":
        sprite = dog_lips.repaint(sprite, points)
    return CatAssets(name, sprite, points, background, paws, load_sounds(folder / "voice"), depth, kind)
