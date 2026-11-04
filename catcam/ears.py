"""Ears that move: they perk up when you raise your brows, flatten when you
frown, and now and then one twitches on its own.

Each ear bends around a hinge below its tip: mesh points rotate about the
hinge by an angle that grows from nothing at the hinge to the full angle near
the tip, so the ear swings while its base stays attached to the head.
Sizes are in face widths.
"""

from typing import NamedTuple

import numpy as np

EAR_SIDE = 0.15  # ear tips lie at least this far left/right of the face centre
HEAD_TOP = 0.1  # the top of the head between the ears is found within this of the centre
HINGE_BELOW = 1.2  # hinge height: tip-to-head-top distance times this, below the tip
HINGE_INWARD = 0.5  # hinge sits this much of the ear height toward the middle of the head
BEND = (0.0, 0.8)  # along the ear (hinge = 0, tip = 1): where the bend ramps up
EAR_WIDTH = (0.6, 1.0)  # across the ear, in ear heights: where it fades out

# Angles (radians; positive = tip outward, i.e. flattened).
FLATTEN, PERK = 0.4, -0.15
FROWN_RANGE = (0.15, 0.6)  # browDown score mapped to 0..1 flattening
RAISE_RANGE = (0.2, 0.7)  # brow-raise score mapped to 0..1 perking
EAR_SPEED = 12.0  # 1/s, how fast the ears follow your brows
TWITCH_EVERY = (3.0, 9.0)  # s between twitches, random in this range
TWITCH = (0.22, 7.0, 9.0, 0.6)  # amplitude (rad), frequency (Hz), damping (1/s), duration (s)


class Ear(NamedTuple):
    side: int  # -1 image-left, +1 image-right
    hinge: np.ndarray  # (2,) px
    weights: np.ndarray  # (n,) how much of the ear's angle each mesh point takes


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def find_ears(points: np.ndarray, outline: np.ndarray, face_count: int, face_width: float) -> list[Ear]:
    """The ears of a head mesh: points (n, 2+) px, of which the first face_count
    are face landmarks (they never move with the ears), outline (m, 2) px."""
    centre = points[:face_count, 0].mean()
    top = np.abs(outline[:, 0] - centre) < HEAD_TOP * face_width
    head_top = outline[top, 1].min() if top.any() else points[:face_count, 1].min()
    ears = []
    for side in (-1, 1):
        beside = (outline[:, 0] - centre) * side > EAR_SIDE * face_width
        if not beside.any():
            continue
        tip = outline[beside][outline[beside, 1].argmin()]
        height = head_top - tip[1]
        if height <= 0:  # no ear sticks up on this side
            continue
        hinge = tip + (-side * HINGE_INWARD * height, HINGE_BELOW * height)
        axis = tip - hinge
        length = np.linalg.norm(axis)
        axis /= length
        rel = points[:, :2] - hinge
        along = rel @ axis / length
        across = np.abs(rel[:, 0] * axis[1] - rel[:, 1] * axis[0]) / height
        weights = _smoothstep(along, *BEND) * (1 - _smoothstep(across, *EAR_WIDTH))
        weights[:face_count] = 0
        ears.append(Ear(side, hinge, weights))
    return ears


def bend(points: np.ndarray, ears: list[Ear], angles) -> np.ndarray:
    """points (n, 2+) with each ear rotated outward by its angle (image-left, image-right)."""
    out = points.copy()
    for ear in ears:
        angle = angles[0 if ear.side < 0 else 1]
        moving = ear.weights > 0
        if not angle or not moving.any():
            continue
        # In image coordinates (y down) a positive rotation moves an upright tip to the right.
        phi = ear.side * angle * ear.weights[moving]
        c, s = np.cos(phi), np.sin(phi)
        rel = out[moving, :2] - ear.hinge
        out[moving, 0] = ear.hinge[0] + c * rel[:, 0] - s * rel[:, 1]
        out[moving, 1] = ear.hinge[1] + s * rel[:, 0] + c * rel[:, 1]
    return out


def _ramp(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def brow_angles(scores: dict) -> np.ndarray:
    """Ear angles your brows ask for (image-left, image-right)."""
    # MediaPipe names sides from the person's view: their right is the image's left.
    angles = []
    for side in ("Right", "Left"):
        frown = _ramp(scores.get(f"browDown{side}", 0), *FROWN_RANGE)
        raised = _ramp(max(scores.get("browInnerUp", 0), scores.get(f"browOuterUp{side}", 0)), *RAISE_RANGE)
        angles.append(FLATTEN * frown + PERK * raised * (1 - frown))
    return np.array(angles)


class EarMotion:
    """Smoothed ear angles from your brows, plus random twitches."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.angles = np.zeros(2)
        self.twitch_at = None  # time of the next twitch
        self.twitch_ear, self.twitch_start = 0, -np.inf

    def update(self, scores: dict | None, dt: float, now: float) -> np.ndarray:
        """Ear angles (image-left, image-right); scores is None while no face is seen."""
        if scores is not None:
            self.angles += min(EAR_SPEED * dt, 1.0) * (brow_angles(scores) - self.angles)
        if self.twitch_at is None:
            self.twitch_at = now + self.rng.uniform(*TWITCH_EVERY)
        if now >= self.twitch_at:
            self.twitch_ear, self.twitch_start = int(self.rng.integers(2)), now
            self.twitch_at = now + self.rng.uniform(*TWITCH_EVERY)
        out = self.angles.copy()
        t = now - self.twitch_start
        amplitude, frequency, damping, duration = TWITCH
        if 0 <= t < duration:
            out[self.twitch_ear] += amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t)
        return out
