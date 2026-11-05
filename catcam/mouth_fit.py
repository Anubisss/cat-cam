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
CENTER_TOLERANCE = 0.01  # the middle matters most: a lower lip can hide a central miss
TOO_WIDE = 0.04  # a fit whose corners reach this much past the crease's on both sides is too wide
TRACE = (0.3, 0.08)  # tracing: half the width and height of the area searched
FOLLOW = (0.03, 0.06)  # the face follows the lips: blending radius, falloff
PLACE_FADE = 0.1  # a hand-placed mouth line's pull fades out this far past its corners
NOSE_BOTTOM, CHIN = 2, 152  # landmarks: the stretch to a hand-placed line ends at these
EYE_CORNERS = (33, 263)  # outer eye corners: a lower face fitted too low is lifted from their height


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
    middle_misses = misses[len(misses) // 2 - 1:len(misses) // 2 + 2]
    middle_miss = float(np.median(middle_misses))
    central_miss = (
        abs(middle_miss) >= CENTER_TOLERANCE * face_width
        and np.count_nonzero(np.sign(middle_misses) == np.sign(middle_miss)) >= 2
    )
    miss = float(np.median(misses))
    off = central_miss or (
        abs(miss) >= TOLERANCE * face_width and (np.sign(misses) == np.sign(miss)).all()
    )
    centre = (landmarks[UPPER_INNER_LIP, :2] + landmarks[LOWER_INNER_LIP, :2]).mean(axis=0) / 2
    xs, ys = trace(crease, centre, face_width)
    middle = np.searchsorted(xs, centre[0])
    left = int(np.argmax(ys[:middle]))  # the corners: where the crease turns, at its lowest
    right = middle + int(np.argmax(ys[middle:]))
    fitted = landmarks[[UPPER_INNER_LIP[0], UPPER_INNER_LIP[-1]], 0]
    too_wide = xs[left] - fitted[0] > TOO_WIDE * face_width and fitted[1] - xs[right] > TOO_WIDE * face_width
    if not (off or too_wide):
        return landmarks
    return _onto_line(landmarks, np.column_stack([xs[left:right + 1], ys[left:right + 1]]), face_width)


def place_mouth(landmarks: np.ndarray, line: np.ndarray, face_width: float) -> np.ndarray:
    """landmarks with the lips closed along line ((n, 2) px from one mouth corner to
    the other), for faces the fit and the crease search can't place, like a dog's
    smile, whose corners are its highest points rather than its lowest.

    The line can be far from the fitted lips, so the face between the nose and the
    chin is stretched to it column by column, keeping every point on its side of
    the lips: what moves with your lower lip stays below the new line."""
    line = line[np.argsort(line[:, 0])]
    lips = UPPER_INNER_LIP + LOWER_INNER_LIP
    landmarks = _lift_lower_face(landmarks, line)
    old = (landmarks[UPPER_INNER_LIP, :2] + landmarks[LOWER_INNER_LIP, :2]) / 2
    old = old[np.argsort(old[:, 0])]
    top, bottom = landmarks[NOSE_BOTTOM, 1], landmarks[CHIN, 1]
    out = landmarks.copy()
    x, y = landmarks[:, 0], landmarks[:, 1]
    was, now = np.interp(x, old[:, 0], old[:, 1]), np.interp(x, line[:, 0], line[:, 1])
    # Fade the stretch out past the mouth corners.
    beyond = np.maximum(line[0, 0] - x, x - line[-1, 0]).clip(0)
    now = was + (now - was) * np.exp(-((beyond / (PLACE_FADE * face_width)) ** 2))
    above = (y > top) & (y <= was)
    below = (y > was) & (y < bottom)
    up, down = np.maximum(was - top, 1e-6), np.maximum(bottom - was, 1e-6)
    out[above, 1] = top + (y - top)[above] * ((now - top) / up)[above]
    out[below, 1] = bottom - (bottom - y)[below] * ((bottom - now) / down)[below]
    seam_x = np.linspace(line[0, 0], line[-1, 0], len(UPPER_INNER_LIP))
    seam = np.column_stack([seam_x, np.interp(seam_x, line[:, 0], line[:, 1])])
    out[lips, :2] = np.vstack([seam, seam])
    return out


def _lift_lower_face(landmarks: np.ndarray, line: np.ndarray) -> np.ndarray:
    """landmarks with the face below the eyes squeezed up, if the fit put it so low
    that the nose sits below the hand-placed lips (line): the fitted mouth moves
    onto the line, everything between the eyes and the chin in proportion."""
    target = np.interp(np.median(line[:, 0]), line[:, 0], line[:, 1])
    if target > landmarks[NOSE_BOTTOM, 1]:
        return landmarks
    eyes, chin = landmarks[list(EYE_CORNERS), 1].mean(), landmarks[CHIN, 1]
    mouth = landmarks[list(UPPER_INNER_LIP + LOWER_INNER_LIP), 1].mean()
    out = landmarks.copy()
    between = (landmarks[:, 1] > eyes) & (landmarks[:, 1] < chin)
    out[between, 1] = np.interp(landmarks[between, 1], [eyes, mouth, chin], [eyes, target, chin])
    return out


def _onto_line(landmarks: np.ndarray, line: np.ndarray, face_width: float) -> np.ndarray:
    """landmarks with the lips closed along line ((n, 2) px, left to right) and the
    face around the mouth following them."""
    seam_x = np.linspace(line[0, 0], line[-1, 0], len(UPPER_INNER_LIP))
    seam = np.column_stack([seam_x, np.interp(seam_x, line[:, 0], line[:, 1])])

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
