"""Where a cat's lips meet, checked against its photo.

The face model's fitted line between the lips can miss the photo's, e.g. land
on a pouty lower lip, or reach past the lips onto the whisker pads. The
photo's is a crease: a line darker than the fur just above and below it,
traced across the mouth, with the mouth corners where it turns (its lowest
points on either side). Where the whole mouth agrees that the fit is clearly
too high or too low, or both traced corners lie clearly inside the fitted
ones, the lips go onto the crease and the face around the mouth follows with
a falloff.
"""

import cv2
import numpy as np

# Inner lip contours, both running corner (78) to corner (308).
UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
              375, 321, 405, 314, 17, 84, 181, 91, 146]  # around the mouth

# Sizes in face widths.
BLUR = 0.004  # smoothing of the photo before looking for the crease
CREASE = 0.012  # a crease is darker than the photo this far above and below it
SEARCH = (0.05, 0.03)  # checking the fit: how far (up, down) to look for the crease
BAND = 0.012  # ...averaging columns this wide
TOLERANCE = 0.015  # a fit this close to the crease is kept
TOO_WIDE = 0.04  # a fit whose corners reach this much past the crease's on both sides is too wide
TRACE = (0.3, 0.08)  # tracing: half the width and height of the area searched
FOLLOW = (0.03, 0.06)  # the face follows the lips: blending radius, falloff


def _creases(sprite: np.ndarray, face_width: float) -> np.ndarray:
    """How much darker each pixel is than the photo just above and below it."""
    gray = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), BLUR * face_width)
    d = max(int(CREASE * face_width), 1)
    return (np.roll(gray, d, axis=0) + np.roll(gray, -d, axis=0)) / 2 - gray


def _misses(crease: np.ndarray, landmarks: np.ndarray, face_width: float) -> np.ndarray:
    """Per column along the middle of the mouth: how far (px, + = down) the
    strongest crease is from the fitted line between the lips."""
    upper, lower = UPPER_INNER_LIP[2:-2], LOWER_INNER_LIP[2:-2]  # away from the corners
    misses = []
    for x, y in (landmarks[upper, :2] + landmarks[lower, :2]) / 2:
        ys = np.arange(int(y - SEARCH[0] * face_width), int(y + SEARCH[1] * face_width))
        xs = np.arange(int(x - BAND * face_width), int(x + BAND * face_width) + 1)
        ys, xs = np.clip(ys, 0, len(crease) - 1), np.clip(xs, 0, crease.shape[1] - 1)
        misses.append(ys[np.argmax(crease[np.ix_(ys, xs)].mean(axis=1))] - y)
    return np.array(misses)


def trace(crease: np.ndarray, centre: np.ndarray, face_width: float):
    """The strongest crease running left to right through the area around
    centre, as (xs, ys) px: the path of steps of at most 1 px up or down
    per column that is darkest in total."""
    w, h = (int(r * face_width) for r in TRACE)
    x0, y0 = max(int(centre[0]) - w, 0), max(int(centre[1]) - h, 0)
    area = crease[y0:int(centre[1]) + h, x0:int(centre[0]) + w]
    rows, cols = area.shape
    cost = -area.copy()
    step = np.zeros((rows, cols), int)
    for j in range(1, cols):
        before = np.stack([np.r_[np.inf, cost[:-1, j - 1]], cost[:, j - 1], np.r_[cost[1:, j - 1], np.inf]])
        step[:, j] = before.argmin(axis=0) - 1
        cost[:, j] += before.min(axis=0)
    path = np.zeros(cols, int)
    path[-1] = cost[:, -1].argmin()
    for j in range(cols - 1, 0, -1):
        path[j - 1] = path[j] + step[path[j], j]
    return np.arange(cols) + x0, path + y0


def snap_mouth(sprite: np.ndarray, landmarks: np.ndarray, face_width: float) -> np.ndarray:
    """landmarks (478, 2+) px with the lips moved onto the photo's crease
    between them, if the fit clearly missed it."""
    crease = _creases(sprite, face_width)
    misses = _misses(crease, landmarks, face_width)
    miss = float(np.median(misses))
    off = abs(miss) >= TOLERANCE * face_width and (np.sign(misses) == np.sign(miss)).all()
    centre = (landmarks[UPPER_INNER_LIP, :2] + landmarks[LOWER_INNER_LIP, :2]).mean(axis=0) / 2
    xs, ys = trace(crease, centre, face_width)
    middle = np.searchsorted(xs, centre[0])
    left = int(np.argmax(ys[:middle]))  # the corners: where the crease turns, at its lowest
    right = middle + int(np.argmax(ys[middle:]))
    fitted = landmarks[[UPPER_INNER_LIP[0], UPPER_INNER_LIP[-1]], 0]
    too_wide = xs[left] - fitted[0] > TOO_WIDE * face_width and fitted[1] - xs[right] > TOO_WIDE * face_width
    if not (off or too_wide):
        return landmarks
    seam_x = np.linspace(xs[left], xs[right], len(UPPER_INNER_LIP))
    seam = np.column_stack([seam_x, np.interp(seam_x, xs, ys)])

    lips = UPPER_INNER_LIP + LOWER_INNER_LIP
    moves = np.vstack([seam, seam]) - landmarks[lips, :2]
    # Every other point moves like the lip points near it, less the further it is.
    dist = np.linalg.norm(landmarks[:, None, :2] - landmarks[None, lips, :2], axis=2)
    weights = np.exp(-((dist / (FOLLOW[0] * face_width)) ** 2)) + 1e-12
    falloff = np.exp(-((dist.min(axis=1) / (FOLLOW[1] * face_width)) ** 2))
    out = landmarks.copy()
    out[:, :2] += (weights @ moves) / weights.sum(axis=1, keepdims=True) * falloff[:, None]
    out[lips, :2] = np.vstack([seam, seam])
    return out
