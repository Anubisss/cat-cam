"""Renders animals with their paws up in the hand poses worth checking (open
palms, backs of the hands, fists, a finger raised, fingers spread, a hand turned
edge-on and one tilted toward the camera), one row per animal, so paws can be
looked over without a camera.

    uv run python tools/paw_sheet.py [Woofgang Grinbert ...] -o paws.jpg
"""

import argparse
from types import SimpleNamespace

import cv2
import numpy as np

from catcam.app import OUT_H, OUT_W, load_cats, shown
from catcam.cat_mesh import Renderer
from catcam.paws import Paws

TILE = (320, 180)
# A hand in camera px around its wrist: knuckles (index to little finger) across, the
# fingers' tips reaching out by how straight each is (2 = straight, 0.8 = a fist).
KNUCKLES_X = (-1.5, -0.5, 0.5, 1.5)
HAND = 40  # px, half a knuckle's spacing; the wrist is 2.5 of them below the knuckles


def hand(x, y, palm: bool, right: bool, reach=2.0, spread=0.0, turn=0.0, tilt=0.0):
    """21 normalised hand landmarks, fingers up. palm: the palm faces the camera; turn:
    radians the hand turns about its own up axis (toward edge-on); tilt: radians its
    fingers lean toward the camera."""
    # A right hand showing its palm has its index on the image right.
    side = -1 if right == palm else 1
    pts = np.zeros((21, 3))
    pts[0] = (0, 2.5 * HAND, 0)
    for f, (tip, mcp) in enumerate(((8, 5), (12, 9), (16, 13), (20, 17))):
        pts[mcp] = (side * KNUCKLES_X[f] * HAND, 0, 0)
        out = pts[mcp] - pts[0]
        angle = spread * (KNUCKLES_X[f] / 1.5) * side
        out = np.array([out[0] * np.cos(angle) - out[1] * np.sin(angle),
                        out[0] * np.sin(angle) + out[1] * np.cos(angle), 0])
        pts[tip] = pts[0] + np.broadcast_to(reach, 4)[f] * out
    c, s = np.cos(turn), np.sin(turn)
    pts = pts @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T  # about the vertical
    c, s = np.cos(tilt), np.sin(tilt)
    pts = pts @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]]).T  # fingers toward the camera
    return [SimpleNamespace(x=(x + px) / OUT_W, y=(y + py) / OUT_H, z=pz / OUT_W) for px, py, pz in pts]


POSES = [  # label, then (palm, reach, spread, turn, tilt) for your right and left hand
    ("open palms", (True, 2.0, 0, 0, 0), (True, 2.0, 0, 0, 0)),
    ("backs", (False, 2.0, 0, 0, 0), (False, 2.0, 0, 0, 0)),
    ("fists", (True, 0.8, 0, 0, 0), (False, 0.8, 0, 0, 0)),
    ("index up / half curled", (True, [2.0, 0.8, 0.8, 0.8], 0, 0, 0), (True, 1.3, 0, 0, 0)),
    ("spread", (True, 2.0, 0.25, 0, 0), (False, 2.0, 0.25, 0, 0)),
    ("turned / tilted", (True, 2.0, 0, 1.0, 0), (True, 2.0, 0, 0, 0.6)),
]


def row(cat) -> np.ndarray:
    tiles = []
    for label, right, left in POSES:
        paws = Paws(OUT_W)
        # Your right hand is on the image's left (the camera isn't mirrored).
        hands = SimpleNamespace(
            hand_landmarks=[hand(330, 520, *right[:1], True, *right[1:]),
                            hand(950, 520, *left[:1], False, *left[1:])],
            handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]])
        for _ in range(15):
            paws.update(hands, None, (OUT_W, OUT_H), 1 / 30, *cat.mesh.face_on_screen())
        sprite = cat.eyelids.render(blink=(0, 0))
        frame = cat.mesh.render(sprite, np.zeros((len(cat.points), 3)), np.eye(3))
        tile = cv2.resize(paws.draw(frame, cat.paw), TILE, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{cat.name}: {label}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="animals to show (default: all)")
    parser.add_argument("-o", "--output", default="paws.jpg")
    args = parser.parse_args()
    cats = [shown(c, True) for c in load_cats(Renderer((OUT_W, OUT_H)))]
    cats = [c for c in cats if not args.names or c.name in args.names]
    cv2.imwrite(args.output, np.vstack([row(c) for c in cats]))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
