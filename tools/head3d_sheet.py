"""Renders cats' photo and 3D heads side by side in the poses worth checking
(rest, mouth half and wide open, eyes closed, ears flat, turned with the mouth
open), one row per head, so a new 3D head can be looked over without a camera.

    uv run python tools/head3d_sheet.py Woofgang [more cats...] -o sheet.jpg
"""

import argparse

import cv2
import numpy as np

from catcam.app import OUT_H, OUT_W, load_cats, shown
from catcam.cat_mesh import Renderer
from catcam.head_pose import to_matrix
from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP

TILE = (360, 324)
CROP = np.s_[0:720, 240:1040]  # the head in the 1280x720 frame


def jaw_drop(points: np.ndarray, amount: float) -> np.ndarray:
    """Landmark moves (portrait px) for a mouth opened by amount face widths:
    everything below the lips drops, most in the middle of the mouth."""
    width = np.ptp(points[:, 0])
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    order = np.argsort(seam[:, 0])
    line = np.interp(points[:, 0], seam[order, 0], seam[order, 1])
    below = points[:, 1] > line
    below[UPPER_INNER_LIP], below[LOWER_INNER_LIP] = False, True
    depth = np.clip((points[:, 1] - line) / (0.4 * width), 0, 1)
    middle, half = points[LOWER_INNER_LIP, 0].mean(), np.ptp(points[LOWER_INNER_LIP, 0]) / 2
    rounded = np.clip(1 - ((points[:, 0] - middle) / (1.3 * half)) ** 2, 0.15, 1)
    moves = np.zeros((len(points), 3))
    moves[below, 1] = amount * width * np.maximum(depth[below], 0.6) * rounded[below]
    return moves


def row(cat) -> np.ndarray:
    still = np.zeros((len(cat.points), 3))
    poses = [  # label, landmark moves, blink, head rotation, ears
        ("rest", still, (0, 0), (0, 0, 0), (0, 0)),
        ("mouth half open", jaw_drop(cat.points, 0.15), (0, 0), (0, 0, 0), (0, 0)),
        ("mouth wide open", jaw_drop(cat.points, 0.4), (0, 0), (0, 0, 0), (0, 0)),
        ("eyes closed", still, (1, 1), (0, 0, 0), (0, 0)),
        ("ears flat", still, (0, 0), (0, 0, 0), (0.4, 0.4)),
        ("turned, mouth open", jaw_drop(cat.points, 0.3), (0, 0), (0, 0.4, 0), (0, 0)),
    ]
    tiles = []
    for label, moves, blink, rotation, ears in poses:
        sprite = cat.eyelids.render(blink=blink)
        frame = cat.mesh.render(sprite, moves, to_matrix(np.array(rotation, float)), ears)
        tile = cv2.resize(np.ascontiguousarray(frame[CROP]), TILE, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", help="cats to render")
    parser.add_argument("-o", "--output", default="head3d_sheet.jpg")
    args = parser.parse_args()
    cats = {c.name: c for c in load_cats(Renderer((OUT_W, OUT_H)))}
    rows = []
    for name in args.names:
        cat = cats[name]
        rows.append(row(cat))
        if cat.head3d is not None:
            rows.append(row(shown(cat, True)))
        else:
            print(f"{name} has no 3D head yet")
    cv2.imwrite(args.output, np.vstack(rows))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
