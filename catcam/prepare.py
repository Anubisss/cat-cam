"""Turns the portraits in all_cats/ into the assets the filter runs on.

For every all_cats/<Name>.png this cuts the cat out of its background, finds
its face landmarks and writes assets/cats/<Name>/cutout.png and face.npz.
Paw photos next to it (<Name>PawsFront.png, <Name>PawsBack.png) become its
paw sprites (see catcam.paw_photos). A <Name>Mouth.txt next to it puts the
lips where the face fit can't find them: one "x, y" per line, portrait px,
along the line where the lips meet from one mouth corner to the other. A
<Name>Eyes.txt does the same for eyes the fit misses: four "x, y" lines, the
image-left eye's outer and inner corner, then the image-right eye's inner and
outer corner, optionally followed by four more: the middle of the upper and the
lower lid, image-left eye first (see catcam.cat_face.place_eyes). Already
prepared cats and paws are skipped unless --force is given.

Run after adding or replacing a portrait:

    uv run --group prepare python -m catcam.prepare
"""

import argparse
import sys

import cv2
import numpy as np

from catcam import paw_photos
from catcam.cat_face import place_eyes
from catcam.cats import portraits as find_portraits
from catcam.landmark_fit import (
    FitError,
    Portrait,
    detects_directly,
    fit_face,
    head_bbox,
)
from catcam.mouth_fit import place_mouth
from catcam.paths import ASSETS, FACE_MIRROR, PORTRAITS

TEXTURE_SIZE = 1024  # longest side of the stored cutout
BORDER = 20  # px of portrait border sampled for the background colour
SOLID_INSIDE = 0.015  # of the image size: deeper inside the cat than this is fully opaque


class CutoutError(RuntimeError):
    pass


def cut_out(bgr: np.ndarray, session) -> np.ndarray:
    """BGRA cat with a clean, slightly feathered alpha."""
    from rembg import remove  # only needed here, not to run the filter

    rgba = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), session=session)
    soft = rgba[..., 3].astype(np.float32) / 255
    # Keep only the cat itself (drops faint background leftovers), with its
    # silhouette filled in.
    mask = (rgba[..., 3] > 20).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise CutoutError("found no cat to cut out")
    biggest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(biggest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = cv2.drawContours(np.zeros_like(biggest), contours, -1, 1, -1)
    # The matting model can leave dark fur on a dark backdrop half transparent;
    # only the fringe keeps its soft alpha, the inside is solid.
    edge = max(int(SOLID_INSIDE * max(bgr.shape[:2])), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    solid = cv2.GaussianBlur(cv2.erode(filled, kernel).astype(np.float32), (0, 0), edge / 2)
    outline = cv2.GaussianBlur(filled.astype(np.float32), (5, 5), 0)
    alpha = np.minimum(np.maximum(soft * filled, solid), outline)
    return np.dstack([bgr, (alpha * 255).astype(np.uint8)])


def background_color(bgr: np.ndarray) -> np.ndarray:
    border = np.concatenate([
        bgr[:BORDER].reshape(-1, 3), bgr[-BORDER:].reshape(-1, 3),
        bgr[:, :BORDER].reshape(-1, 3), bgr[:, -BORDER:].reshape(-1, 3),
    ])
    return np.median(border, axis=0).astype(np.uint8)


def save(name: str, cutout: np.ndarray, points: np.ndarray, background: np.ndarray, mouth_set: bool):
    scale = TEXTURE_SIZE / max(cutout.shape[:2])
    size = (round(cutout.shape[1] * scale), round(cutout.shape[0] * scale))
    out = ASSETS / name
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "cutout.png"), cv2.resize(cutout, size, interpolation=cv2.INTER_AREA))
    np.savez(out / "face.npz", points=(points * scale).astype(np.float32), background=background,
             mouth_set=mouth_set)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="cats to prepare (default: all portraits)")
    parser.add_argument("--force", action="store_true", help="redo cats that are already prepared")
    args = parser.parse_args()

    portraits = find_portraits()
    if not portraits:
        sys.exit(f"No portraits found in {PORTRAITS}.")
    unknown = set(args.names) - set(portraits)
    if unknown:
        sys.exit(f"No portrait for: {', '.join(sorted(unknown))} (looked in {PORTRAITS}).")
    names = args.names or list(portraits)
    todo = [n for n in names if args.force or not (ASSETS / n / "face.npz").exists()]
    paws_todo = [n for n in names if paw_photos.has_photos(n, PORTRAITS) and (args.force or not all(
        paw_photos.paw_files(n, ASSETS)[v, h].exists()
        for v in paw_photos.has_photos(n, PORTRAITS) for h in paw_photos.HANDS))]
    if not todo and not paws_todo:
        print("All cats are prepared already (use --force to redo them).")
        return

    from rembg import new_session

    session = new_session("isnet-general-use")
    failed = []
    for name in paws_todo:
        print(f"Cutting out {name}'s paws...")
        try:
            views = paw_photos.prepare_paws(name, PORTRAITS, ASSETS, lambda bgr: cut_out(bgr, session))
        except (CutoutError, ValueError) as e:
            print(f"  {name}: {e}, skipped.")
            failed.append(name)
            continue
        print(f"  saved {', '.join(views)} views")
    if not todo:
        if failed:
            sys.exit(f"Failed: {', '.join(failed)}")
        return

    print("Cutting out cats...")
    images, cutouts = {}, {}
    for name, path in portraits.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  {name}: cannot read {path}, skipped.")
            failed.append(name)
            continue
        cached = ASSETS / name / "cutout.png"
        try:
            if name in todo or not cached.exists():
                cutouts[name] = cut_out(image, session)
            else:  # only its outline is needed, to use it as a primer
                small = cv2.imread(str(cached), cv2.IMREAD_UNCHANGED)
                h, w = image.shape[:2]
                cutouts[name] = np.dstack([image, cv2.resize(small[..., 3], (w, h))])
        except CutoutError as e:
            print(f"  {name}: {e}, skipped.")
            failed.append(name)
            continue
        images[name] = image
    todo = [n for n in todo if n in images]
    subjects = {n: Portrait(images[n], head_bbox(cutouts[n][..., 3])) for n in images}
    primers = [n for n in subjects if detects_directly(subjects[n])]
    if not primers:
        sys.exit("The face detector finds none of the portraits, so there is nothing to start from.")
    mirror = np.load(FACE_MIRROR)

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
        mouth = PORTRAITS / f"{name}Mouth.txt"
        if mouth.is_file():
            points[:, :2] = place_mouth(points, np.loadtxt(mouth, delimiter=",", ndmin=2),
                                        np.ptp(points[:, 0]))[:, :2]
            print(f"  lips placed along {mouth.name}")
        eyes = PORTRAITS / f"{name}Eyes.txt"
        if eyes.is_file():
            points[:, :2] = place_eyes(points, np.loadtxt(eyes, delimiter=",", ndmin=2),
                                       np.ptp(points[:, 0]))[:, :2]
            print(f"  eyes placed at {eyes.name}")
        # The fit can put a chin past a head cut off at the picture's edge.
        h, w = images[name].shape[:2]
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        save(name, cutouts[name], points, background_color(images[name]), mouth.is_file())
        print(f"  saved {ASSETS / name}")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
