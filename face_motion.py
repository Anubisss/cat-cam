"""Transfers your facial expression onto the cat's face points.

Your landmarks are aligned to your own neutral face (removing head position,
rotation and distance), so what is left is pure expression: how far each point
moved. Those movements are mapped into the cat's frame and added to the cat's
matching landmarks.
"""

import numpy as np

# Points that barely move with expressions (forehead, nose bridge, temples,
# eye corners). Used to align faces without cancelling out the expression.
STABLE = [
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1,
    109, 67, 103, 54, 21, 162, 127, 338, 297, 332, 284, 251, 389, 356,
    33, 133, 263, 362, 234, 454,
]
NEUTRAL_RATE = 0.03  # how fast the neutral face adapts while you look relaxed


def similarity(src: np.ndarray, dst: np.ndarray):
    """Scale, rotation, translation with s * R @ src + t ~= dst (Umeyama)."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_src, dst - mu_dst
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / a.var(axis=0).sum()
    return scale, rotation, mu_dst - scale * rotation @ mu_src


class ExpressionTransfer:
    def __init__(self, cat_points: np.ndarray):
        self.cat = cat_points
        self.neutral = None
        self.to_cat = None

    def reset(self):
        """Take the next face as the new neutral face."""
        self.neutral = None

    def __call__(self, user: np.ndarray, relaxed: bool) -> np.ndarray:
        """user: (478, 3) landmarks in px. Returns (478, 3) offsets in cat px."""
        if self.neutral is None:
            self._set_neutral(user)
        s, r, t = similarity(user[STABLE], self.neutral[STABLE])
        aligned = s * user @ r.T + t
        if relaxed:
            self._set_neutral(self.neutral + NEUTRAL_RATE * (aligned - self.neutral))
        scale, rotation = self.to_cat
        return scale * (aligned - self.neutral) @ rotation.T

    def _set_neutral(self, points: np.ndarray):
        self.neutral = points
        s, r, _ = similarity(points[STABLE], self.cat[STABLE])
        self.to_cat = (s, r)
