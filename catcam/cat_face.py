"""Closes the cat's eyelids in its texture.

The cat squints: its real upper lids sit above where the face landmarker puts
them, so moving the mesh alone would leave a yellow sliver. Instead the skin
above each eye is stretched down over the eye in the texture itself.
"""

import cv2
import numpy as np

# MediaPipe face mesh indices; each contour runs corner to corner.
EYES = (
    # (upper lid, lower lid) of the eye on the image's left, then image's right.
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]),
    ([263, 466, 388, 387, 386, 385, 384, 398, 362], [263, 249, 390, 373, 374, 380, 381, 382, 362]),
)
LID_SKIN = 1.2  # how much skin above the eye (in eye heights) slides down
# The cat's real upper lid sits above the landmarks by this fraction of the
# local eye height...
LID_OFFSET = 0.3
# ...and its lower lid below them, by this fraction of the local eye height;
# a closed lid reaches down that far.
LID_BELOW = 0.6


def _sample(points, xs):
    """Sample a contour's y at xs (contour sorted by x first)."""
    order = np.argsort(points[:, 0])
    return np.interp(xs, points[order, 0], points[order, 1])


class CatFace:
    def __init__(self, sprite: np.ndarray, landmarks: np.ndarray):
        """sprite: BGRA cat, landmarks: (478, 2) pixel coords on the sprite."""
        self.sprite = sprite
        self.eyes = [(landmarks[upper], landmarks[lower]) for upper, lower in EYES]

    def render(self, blink: tuple[float, float]) -> np.ndarray:
        """blink: closedness 0..1 per eye (image-left, image-right)."""
        if max(blink) <= 0.05:
            return self.sprite
        out = self.sprite.copy()
        for (upper, lower), amount in zip(self.eyes, blink):
            if amount > 0.05:
                self._close_lid(out, upper, lower, amount)
        return out

    def _close_lid(self, out: np.ndarray, upper, lower, amount: float):
        """Redraw the region around one eye with its upper lid lowered."""
        h, w = self.sprite.shape[:2]
        x0 = max(int(np.floor(upper[:, 0].min())), 0)
        x1 = min(int(np.ceil(upper[:, 0].max())) + 1, w)
        if x1 <= x0:  # eye outside the sprite
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        lo = _sample(lower, xs)
        up = _sample(upper, xs)
        up = up - LID_OFFSET * np.maximum(lo - up, 0)
        height = (lo - up).max()
        edge = up + amount * (1 + LID_BELOW) * np.maximum(lo - up, 0)
        skin = LID_SKIN * height
        top = up - skin
        y0 = max(int(np.floor(top.min())), 0)
        y1 = min(int(np.ceil(edge.max())) + 1, h)
        if y1 <= y0:
            return
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        inside = (y >= top) & (y <= edge)
        # Stretch the skin band [top, up] down so it covers [top, edge].
        src = top + (y - top) * skin / np.maximum(edge - top, 1e-3)
        map_y = np.where(inside, src, y).astype(np.float32)
        map_x = np.ascontiguousarray(np.broadcast_to(xs, map_y.shape))
        out[y0:y1, x0:x1] = cv2.remap(
            self.sprite, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
