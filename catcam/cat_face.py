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
# local eye height.
LID_OFFSET = 0.8


def _sample(points, xs):
    """Sample a contour's y at xs (contour sorted by x first)."""
    order = np.argsort(points[:, 0])
    return np.interp(xs, points[order, 0], points[order, 1])


class CatFace:
    def __init__(self, sprite: np.ndarray, landmarks: np.ndarray):
        """sprite: BGRA cat, landmarks: (478, 2) pixel coords on the sprite."""
        self.sprite = sprite
        h, w = sprite.shape[:2]
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        self.eyes = [(landmarks[u], landmarks[l]) for u, l in EYES]

    def render(self, blink: tuple[float, float]) -> np.ndarray:
        """blink: closedness 0..1 per eye (image-left, image-right)."""
        if max(blink) <= 0.05:
            return self.sprite
        map_y = self.grid_y
        for (upper, lower), amount in zip(self.eyes, blink):
            if amount > 0.05:
                map_y = self._lid_map(map_y, upper, lower, amount)
        return cv2.remap(
            self.sprite, self.grid_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _lid_map(self, map_y, upper, lower, amount) -> np.ndarray:
        xs = self.grid_x[0]
        lo = _sample(lower, xs)
        up = _sample(upper, xs)
        up = up - LID_OFFSET * np.maximum(lo - up, 0)
        edge = up + amount * np.maximum(lo - up, 0)
        skin = LID_SKIN * (lo - up).max()
        top = up - skin
        y = self.grid_y
        inside = (
            (xs >= upper[:, 0].min()) & (xs <= upper[:, 0].max())
            & (y >= top) & (y <= edge)
        )
        # Stretch the skin band [top, up] down so it covers [top, edge].
        src = top + (y - top) * skin / np.maximum(edge - top, 1e-3)
        return np.where(inside, src, map_y).astype(np.float32)
