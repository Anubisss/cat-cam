"""Turns the portraits in all_cats/ into the assets the filter runs on.

For every all_cats/<Name>.png this cuts the cat out of its background, finds
its face landmarks and writes assets/cats/<Name>/cutout.png and face.npz.
Already prepared cats are skipped unless --force is given.

Run after adding or replacing a portrait:

    uv run --group prepare python -m catcam.prepare
"""

import argparse
import sys

import cv2
import numpy as np

from catcam.landmark_fit import (
    FitError,
    Portrait,
    detects_directly,
    fit_face,
    head_bbox,
)
from catcam.paths import ASSETS, FACE_MIRROR, PORTRAITS

TEXTURE_SIZE = 1024  # longest side of the stored cutout
BORDER = 20  # px of portrait border sampled for the background colour


def cut_out(bgr: np.ndarray) -> np.ndarray:
    """BGRA cat with a clean, slightly feathered alpha."""
    from rembg import new_session, remove  # only needed here, not to run the filter

    session = new_session("isnet-general-use")
    rgba = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), session=session)
    # Keep only the cat itself (drops faint background leftovers).
    mask = (rgba[..., 3] > 128).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    alpha = cv2.GaussianBlur((labels == biggest).astype(np.float32), (5, 5), 0)
    alpha = np.minimum(alpha, rgba[..., 3] / 255)
    return np.dstack([bgr, (alpha * 255).astype(np.uint8)])


def background_color(bgr: np.ndarray) -> np.ndarray:
    border = np.concatenate([
        bgr[:BORDER].reshape(-1, 3), bgr[-BORDER:].reshape(-1, 3),
        bgr[:, :BORDER].reshape(-1, 3), bgr[:, -BORDER:].reshape(-1, 3),
    ])
    return np.median(border, axis=0).astype(np.uint8)


def save(name: str, cutout: np.ndarray, points: np.ndarray, background: np.ndarray):
    scale = TEXTURE_SIZE / max(cutout.shape[:2])
    size = (round(cutout.shape[1] * scale), round(cutout.shape[0] * scale))
    out = ASSETS / name
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "cutout.png"), cv2.resize(cutout, size, interpolation=cv2.INTER_AREA))
    np.savez(out / "face.npz", points=(points * scale).astype(np.float32), background=background)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="cats to prepare (default: all portraits)")
    parser.add_argument("--force", action="store_true", help="redo cats that are already prepared")
    args = parser.parse_args()

    portraits = {p.stem: p for p in sorted(PORTRAITS.glob("*.png"))}
    if not portraits:
        sys.exit(f"No portraits found in {PORTRAITS}.")
    unknown = set(args.names) - set(portraits)
    if unknown:
        sys.exit(f"No portrait for: {', '.join(sorted(unknown))} (looked in {PORTRAITS}).")
    todo = [n for n in (args.names or portraits) if args.force or not (ASSETS / n / "face.npz").exists()]
    if not todo:
        print("All cats are prepared already (use --force to redo them).")
        return

    print("Cutting out cats...")
    images, cutouts = {}, {}
    for name, path in portraits.items():
        images[name] = cv2.imread(str(path), cv2.IMREAD_COLOR)
        cached = ASSETS / name / "cutout.png"
        if name in todo or not cached.exists():
            cutouts[name] = cut_out(images[name])
        else:  # only its outline is needed, to use it as a primer
            small = cv2.imread(str(cached), cv2.IMREAD_UNCHANGED)
            h, w = images[name].shape[:2]
            cutouts[name] = np.dstack([images[name], cv2.resize(small[..., 3], (w, h))])
    subjects = {n: Portrait(images[n], head_bbox(cutouts[n][..., 3])) for n in portraits}
    primers = [n for n in portraits if detects_directly(subjects[n])]
    if not primers:
        sys.exit("The face detector finds none of the portraits, so there is nothing to start from.")
    mirror = np.load(FACE_MIRROR)

    failed = []
    for name in todo:
        print(f"Finding {name}'s face...")
        # Other cats make better primers (they test the tracking for real);
        # a cat only primes itself if it is the only one detected.
        own = [subjects[n] for n in primers if n != name] or [subjects[name]]
        try:
            points = fit_face(subjects[name], own, mirror)
        except FitError as e:
            print(f"  {name}: {e}, skipped.")
            failed.append(name)
            continue
        save(name, cutouts[name], points, background_color(images[name]))
        print(f"  saved {ASSETS / name}")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
