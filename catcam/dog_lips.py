"""Dog lips, painted over a portrait's human-like ones.

A dog has no full, pink lips: furry flews reach down to a thin black lip line,
which meets a dark groove (the philtrum) coming down from the nose. So the
upper lip is covered with whisker-pad fur from beside the mouth, the lower lip
with chin fur from below it (blended in so the colours match), and the line
and the groove are drawn on top, along the seam where the mouth opens.
Sizes are in face widths.
"""

import cv2
import numpy as np

from catcam.mouth_fit import UPPER_INNER_LIP, snap_mouth

UPPER_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
REACH = 0.04  # pink pixels this close to the fitted lips belong to them too
GROW = 0.018  # the fur covers this much more than the lips
PAD = 0.04  # the upper lip's fur comes from the whisker pad this far beside the mouth
CHIN = 0.03  # the lower lip's from this far below it
LINE = 0.014  # lip line thickness in the middle; it thins toward the corners
PHILTRUM = 0.016
SHADOW = (0.025, 0.012, 0.35)  # under the upper lip and around the groove: width, blur, darkness
NOSE_GREY = 70  # the groove runs up to where the photo gets this dark (the nose)
SUPERSAMPLE = 4  # for smooth line edges


def _disc(radius: float) -> np.ndarray:
    k = 2 * max(int(radius), 1) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _between(shape, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.vstack([top, bottom[::-1]]).astype(np.int32)], 255)
    return mask


def _clone(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """target with source's texture blended in where mask is set."""
    ys, xs = np.nonzero(mask)
    centre = ((xs.min() + xs.max() + 1) // 2, (ys.min() + ys.max() + 1) // 2)
    return cv2.seamlessClone(source, target, mask, centre, cv2.NORMAL_CLONE)


def lip_mask(bgr: np.ndarray, points: np.ndarray, face_width: float) -> np.ndarray:
    """uint8 mask of the lips: between the fitted outer lips, plus the pink next to them."""
    lips = _between(bgr.shape[:2], points[UPPER_OUTER_LIP, :2], points[LOWER_OUTER_LIP, :2])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pink = ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    near = cv2.dilate(lips, _disc(REACH * face_width)) > 0
    lips = cv2.dilate(np.where(pink & near, 255, lips).astype(np.uint8), _disc(GROW * face_width))
    return cv2.morphologyEx(lips, cv2.MORPH_CLOSE, _disc(GROW * face_width))


def repaint(sprite: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The BGRA sprite with dog lips instead of the photo's."""
    face_width = np.ptp(landmarks[:, 0])
    points = snap_mouth(sprite, landmarks, face_width)
    bgr = np.ascontiguousarray(sprite[..., :3])
    h, w = bgr.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    seam = points[UPPER_INNER_LIP, :2]
    below_seam = ys >= np.interp(xs, seam[:, 0], seam[:, 1])
    lips = lip_mask(bgr, points, face_width)

    # Above the seam whisker pad fur from beside the mouth (mirrored, so both
    # halves meet), below it chin fur; blended in at once, so only the fur
    # around the lips sets the colours. The lip line covers the join.
    centre = points[[78, 308], 0].mean()
    beside = centre - (np.ptp(points[[61, 291], 0]) / 2 + PAD * face_width) - np.abs(xs - centre)
    pad = cv2.remap(bgr, beside.astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    drop = points[LOWER_OUTER_LIP, 1].max() - seam[:, 1].min() + CHIN * face_width
    chin = cv2.remap(bgr, xs, (ys + drop).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fur = np.where(below_seam[..., None], chin, pad)
    out = _clone(fur, bgr.copy(), lips).astype(np.float32)

    # The groove runs from the middle of the lip line up to the nose.
    middle = seam[len(seam) // 2]
    grey = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    column = grey[:, int(middle[0]) - 3:int(middle[0]) + 4].mean(axis=1)
    top = int(middle[1] - LINE * face_width)
    while top > 0 and column[top] > NOSE_GREY:
        top -= 1
    groove = np.array([middle[0], top])

    width, blur, darkness = SHADOW
    shadow = np.zeros((h, w), np.float32)
    cv2.polylines(shadow, [(seam + [0, width / 2 * face_width]).astype(np.int32)], False, 1.0,
                  int(width * face_width))
    cv2.line(shadow, tuple(groove.astype(int)), tuple(middle.astype(int)), 1.0, int(2 * width * face_width))
    out *= 1 - darkness * cv2.GaussianBlur(shadow, (0, 0), blur * face_width)[..., None]

    s = SUPERSAMPLE
    ink = np.zeros((h * s, w * s), np.uint8)
    taper = np.abs(np.linspace(-1, 1, len(seam)))
    for i in range(len(seam) - 1):
        thickness = LINE * face_width * (1 - 0.6 * max(taper[i], taper[i + 1]) ** 2)
        cv2.line(ink, tuple((seam[i] * s).astype(int)), tuple((seam[i + 1] * s).astype(int)), 255,
                 max(int(thickness * s), 1), cv2.LINE_AA)
    cv2.line(ink, tuple((groove * s).astype(int)), tuple((middle * s).astype(int)), 255,
             int(PHILTRUM * face_width * s), cv2.LINE_AA)
    ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    nose = bgr[max(top - 40, 0):top, int(middle[0]) - 40:int(middle[0]) + 40].reshape(-1, 3)
    black = np.percentile(nose, 10, axis=0)
    out = out * (1 - ink) + black * ink
    return np.dstack([np.clip(out, 0, 255).astype(np.uint8), sprite[..., 3]])
