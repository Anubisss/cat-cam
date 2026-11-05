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


def facing_camera(matrix: np.ndarray) -> np.ndarray:
    """Rotation (image coords) of an upright head where the matrix puts it,
    looking straight into the camera."""
    position = np.asarray(matrix)[:3, 3]
    forward = -position / np.linalg.norm(position)  # the camera sits at the origin
    side = np.cross([0.0, 1.0, 0.0], forward)
    side /= np.linalg.norm(side)
    return _FLIP @ np.column_stack([side, np.cross(forward, side), forward]) @ _FLIP


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

    See https://gery.casiez.net/1euro/. Speeds up to `floor` are what tracking
    noise alone makes; they don't loosen the filter, so it can be smooth at rest
    and still quick once you really move. With per_point, x is (points, dims)
    and each point gets its own speed and cutoff: an opening mouth then follows
    at once without the still rest of the face shaking more.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0, floor: float = 0.0,
                 per_point: bool = False):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.floor, self.per_point = floor, per_point
        self.reset()

    def reset(self):
        """Start afresh: the next value is taken as it is."""
        self.value = None
        self.speed = 0.0

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.value is None:
            self.value = x
            return self.value
        step = x - self.value
        if self.per_point:
            speed = np.linalg.norm(step, axis=-1) / dt
        else:  # RMS speed per component, so the filter behaves the same for any size
            speed = np.sqrt(np.mean(np.square(step))) / dt
        self.speed = self.speed + self._alpha(self.d_cutoff, dt) * (speed - self.speed)
        alpha = self._alpha(self.min_cutoff + self.beta * np.maximum(self.speed - self.floor, 0), dt)
        self.value = self.value + (alpha[..., None] if self.per_point else alpha) * step
        return self.value
