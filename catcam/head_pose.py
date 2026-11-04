"""Head rotation from MediaPipe's facial transformation matrix, plus smoothing."""

import math

import cv2
import numpy as np

# MediaPipe's metric face space has y up and z toward the camera; image
# coordinates (used by the landmarks and the mesh) have y down and z away.
_FLIP = np.diag([1.0, -1.0, -1.0])
MAX_ANGLE = math.radians(55)  # beyond this the 2D texture falls apart


def rotation_of(matrix: np.ndarray) -> np.ndarray:
    """Pure rotation (image coords) from a 4x4 facial transformation matrix."""
    u, _, vt = np.linalg.svd(np.asarray(matrix)[:3, :3])
    return _FLIP @ (u @ vt) @ _FLIP


def relative_rotation(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotation that turns a head from the reference pose to the current one."""
    return current @ reference.T


def to_vector(rotation: np.ndarray) -> np.ndarray:
    vec = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(vec)
    return vec * (MAX_ANGLE / angle) if angle > MAX_ANGLE else vec


def to_matrix(vec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(vec, np.float64))[0]


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when still, responsive when moving.

    See https://gery.casiez.net/1euro/
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value = None
        self.speed = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.value is None:
            self.value = np.asarray(x, np.float64)
            return self.value
        # RMS speed per component, so the filter behaves the same for any size.
        speed = np.sqrt(np.mean(np.square(x - self.value))) / dt
        self.speed += self._alpha(self.d_cutoff, dt) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * self.speed
        self.value = self.value + self._alpha(cutoff, dt) * (x - self.value)
        return self.value
