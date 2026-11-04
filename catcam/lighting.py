"""Light and shadow that follow the head.

The photo already has its lighting, so at rest nothing changes. When the
head turns, nods or tilts, each part of it is brightened or darkened by how
much more or less it now faces the light than in the photo: the side turning
away falls into shadow, the side turning toward the light brightens a little.
"""

import numpy as np

# Direction toward the light: from the front, a bit above and to the left
# (z < 0 is toward the camera).
LIGHT = np.array([-0.3, -0.45, -1.0]) / np.linalg.norm([-0.3, -0.45, -1.0])
STRENGTH = 0.8  # how strongly a change in facing changes the brightness
SHADE_RANGE = (0.4, 1.2)  # darkest, brightest
NORMAL_RADIUS = 0.08  # face widths: surface patch each normal is fitted to


def vertex_normals(points: np.ndarray, face_width: float) -> np.ndarray:
    """(n, 3) unit normals, pointing toward the camera, of a smooth surface
    through points (n, 3): a plane fitted to each point's neighbourhood."""
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    weights = np.exp(-((dist / (NORMAL_RADIUS * face_width)) ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    mean = weights @ points
    second = (weights @ (points[:, :, None] * points[:, None, :]).reshape(len(points), 9)).reshape(-1, 3, 3)
    cov = second - mean[:, :, None] * mean[:, None, :]
    normals = np.linalg.eigh(cov)[1][:, :, 0]  # the direction the patch is flattest in
    return normals * -np.sign(normals[:, 2:3] + 1e-12)


def shading(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """(n,) brightness factor per vertex for the head rotated by rotation."""
    before = np.clip(normals @ LIGHT, 0, 1)
    after = np.clip(normals @ rotation.T @ LIGHT, 0, 1)
    return np.clip(1 + STRENGTH * (after - before), *SHADE_RANGE)
