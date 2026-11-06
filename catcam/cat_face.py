"""Closes the cat's eyelids in its texture.

The cats squint: their real lids sit outside where the face landmarker puts
them, so moving the mesh would leave yellow slivers (and the mesh keeps the
eyes rigid anyway, see cat_mesh). Instead the skin above each eye is stretched
down over the eye in the texture itself.
"""

import cv2
import numpy as np

# MediaPipe face mesh indices; each contour runs corner to corner.
EYES = (
    # (upper lid, lower lid) of the eye on the image's left, then image's right.
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]),
    ([263, 466, 388, 387, 386, 385, 384, 398, 362], [263, 249, 390, 373, 374, 380, 381, 382, 362]),
)
LID_SKIN = 0.9  # how much skin above the eye (in eye heights) slides down
# The cat's real upper lid sits above the landmarks by this fraction of the
# local eye height...
LID_OFFSET = 0.3
# ...and its lower lid below them, by this fraction of the local eye height;
# a closed lid reaches down that far.
LID_BELOW = 0.65
# The eye also reaches past the landmark corners: the lid extends this far
# beyond them (in eye widths), covering at least CORNER_PAD eye heights.
CORNER_REACH, CORNER_PAD = 0.2, 0.5


# Snapping the lids to the eye in the image: search window and seed ellipse
# (in eye widths), and the most a lid may move.
EYE_WINDOW = (1.1, 0.6)
EYE_SEED = (0.42, 0.2)
EYE_SEARCH = (0.95, 0.4)
MAX_LID_SHIFT = 0.25
MAX_CORNER_SHIFT = 0.35


def _eye_opening(bgr: np.ndarray, upper: np.ndarray, lower: np.ndarray):
    """Top and bottom edge of the visible eye, per column, from a GrabCut
    segmentation seeded around the landmarks. Returns (xs, tops, bottoms) in
    sprite px, or None."""
    points = np.vstack([upper, lower])
    centre, width = points.mean(axis=0), np.ptp(points[:, 0])
    if width < 4:
        return None
    x0, y0 = np.maximum((centre - np.array(EYE_WINDOW) * width).astype(int), 0)
    x1, y1 = (centre + np.array(EYE_WINDOW) * width).astype(int)
    crop = cv2.bilateralFilter(np.ascontiguousarray(bgr[y0:y1, x0:x1]), 9, 40, 9)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    c = tuple((centre - (x0, y0)).astype(int))
    for axes, label in ((EYE_SEARCH, cv2.GC_PR_BGD), (EYE_SEED, cv2.GC_PR_FGD)):
        cv2.ellipse(mask, c, tuple(int(a * width) for a in axes), 0, 0, 360, label, -1)
    models = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(crop, mask, None, *models, 6, cv2.GC_INIT_WITH_MASK)
    eye = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    eye = cv2.morphologyEx(eye, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(eye)
    if count < 2:
        return None
    eye = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    columns = np.nonzero(eye.any(axis=0))[0]
    tops = np.array([np.argmax(eye[:, x]) for x in columns])
    bottoms = np.array([eye.shape[0] - 1 - np.argmax(eye[::-1, x]) for x in columns])
    return columns + x0, tops + y0, bottoms + y0


def snap_eyes(bgr: np.ndarray, landmarks: np.ndarray):
    """Eye contours (upper, lower) per eye, moved onto the eyes in the image.

    The landmark fit often sits a little above or below a cat's eye, and can
    stop short of its corners. Each lid is shifted by how far it is off on
    average over the middle of the eye (robust to the segmentation leaking into
    the fur), the corners are moved out to where the eye ends, and both eyes
    get the same correction, as the portraits are symmetric.
    """
    eyes = [(landmarks[upper].copy(), landmarks[lower].copy()) for upper, lower in EYES]
    shifts = []
    for upper, lower in eyes:
        found = _eye_opening(bgr, upper, lower)
        if found is None:
            return eyes
        xs, tops, bottoms = found
        middle = np.abs(xs - upper[:, 0].mean()) < 0.25 * np.ptp(upper[:, 0])
        if middle.sum() < 3:
            return eyes
        # Leaks go outward (up into the brow, down into the cheek): use the
        # inner percentiles.
        top = np.percentile(tops[middle] - _sample(upper, xs[middle]), 70)
        bottom = np.percentile(bottoms[middle] - _sample(lower, xs[middle]), 30)
        left, right = upper[:, 0].min(), upper[:, 0].max()
        # Corners only ever move outward; "outer" is away from the nose.
        beyond_left, beyond_right = max(left - xs.min(), 0), max(xs.max() - right, 0)
        outer_on_left = upper[0, 0] < upper[-1, 0]
        outer, inner = (beyond_left, beyond_right) if outer_on_left else (beyond_right, beyond_left)
        shifts.append((top, bottom, outer, inner, right - left))
    top, bottom, _, _, width = np.mean(shifts, axis=0)
    # The segmentation often misses part of an eye's length, so the corners
    # take the furthest either eye reaches.
    _, _, outer, inner, _ = np.max(shifts, axis=0)
    top, bottom = np.clip([top, bottom], -MAX_LID_SHIFT * width, MAX_LID_SHIFT * width)
    outer, inner = np.minimum([outer, inner], MAX_CORNER_SHIFT * width)
    for eye, (upper, lower) in enumerate(eyes):
        upper[1:-1, 1] += top
        lower[1:-1, 1] += bottom
        left, right = upper[:, 0].min(), upper[:, 0].max()
        # Image-left eye: outer corner on the left; image-right eye: on the right.
        grow_left, grow_right = (outer, inner) if eye == 0 else (inner, outer)
        for contour in (upper, lower):
            contour[[0, -1], 1] += (top + bottom) / 2  # corners are shared by both lids
            t = (contour[:, 0] - left) / max(right - left, 1e-6)
            contour[:, 0] += (1 - t) * -grow_left + t * grow_right
        # A squint is barely open: shifting its lids must not cross them.
        lower[:, 1] = np.maximum(lower[:, 1], _sample(upper, lower[:, 0]))
    return eyes


def _sample(points, xs):
    """Sample a contour's y at xs (contour sorted by x first)."""
    order = np.argsort(points[:, 0])
    return np.interp(xs, points[order, 0], points[order, 1])


class CatFace:
    def __init__(self, sprite: np.ndarray, landmarks: np.ndarray):
        """sprite: BGRA cat, landmarks: (478, 2) pixel coords on the sprite."""
        self.sprite = sprite
        self.eyes = snap_eyes(sprite[..., :3], landmarks)

    def render(self, blink: tuple[float, float]) -> np.ndarray:
        """blink: closedness 0..1 per eye (image-left, image-right)."""
        if max(blink) <= 0.05:
            return self.sprite
        out = self.sprite.copy()
        for (upper, lower), amount in zip(self.eyes, blink):
            if amount > 0.05:
                self._close_lid(out, upper, lower, amount)
        return out

    def lid_box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1), the ends exclusive: the part of the sprite that closing
        the eyes may redraw (a lid lowered further only reaches further down)."""
        lids = [self._lid(upper, lower, 1.0) for upper, lower in self.eyes]
        boxes = np.array([lid[:4] for lid in lids if lid is not None]).reshape(-1, 4)
        if not len(boxes):
            return 0, 0, 0, 0
        return (int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                int(boxes[:, 2].max()), int(boxes[:, 3].max()))

    def _lid(self, upper, lower, amount: float):
        """Where one eye's lid is drawn: its box in the sprite (x0, y0, x1, y1, the
        ends exclusive), the box's columns (xs) and, per column, the top of the skin
        that slides down, the lid's edge and the skin's height. None if off the sprite."""
        h, w = self.sprite.shape[:2]
        left, right = upper[:, 0].min(), upper[:, 0].max()
        reach = CORNER_REACH * (right - left)
        x0 = max(int(np.floor(left - reach)), 0)
        x1 = min(int(np.ceil(right + reach)) + 1, w)
        if x1 <= x0:  # eye outside the sprite
            return None
        xs = np.arange(x0, x1, dtype=np.float32)
        lo = _sample(lower, xs)
        up = _sample(upper, xs)
        opening = np.maximum(lo - up, 0)
        up = up - LID_OFFSET * opening
        height = (lo - up).max()
        # 1 inside the corners, fading to 0 at `reach` beyond them.
        taper = np.clip(1 - np.maximum(left - xs, xs - right) / max(reach, 1e-3), 0, 1)
        cover = np.maximum((1 + LID_BELOW) * opening, CORNER_PAD * height * taper)
        edge = up + amount * cover
        skin = LID_SKIN * height
        top = up - skin
        y0 = max(int(np.floor(top.min())), 0)
        y1 = min(int(np.ceil(edge.max())) + 1, h)
        if y1 <= y0:
            return None
        return x0, y0, x1, y1, xs, top, edge, skin

    def _close_lid(self, out: np.ndarray, upper, lower, amount: float):
        """Redraw the region around one eye with its upper lid lowered."""
        lid = self._lid(upper, lower, amount)
        if lid is None:
            return
        x0, y0, x1, y1, xs, top, edge, skin = lid
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        inside = (y >= top) & (y <= edge)
        # Stretch the skin band [top, up] down so it covers [top, edge].
        src = top + (y - top) * skin / np.maximum(edge - top, 1e-3)
        map_y = np.where(inside, src, y).astype(np.float32)
        map_x = np.ascontiguousarray(np.broadcast_to(xs, map_y.shape))
        out[y0:y1, x0:x1] = cv2.remap(
            self.sprite, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
