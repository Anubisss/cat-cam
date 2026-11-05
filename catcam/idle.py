"""Idle animation: while you sit still and quiet, the animal acts like a real
one on its own. It breathes (a slow nod), sways its head, and often looks
around or tilts its head. As soon as you turn your head, make a face or talk,
it stops and copies you again.

Angles are in radians, sizes in face widths.
"""

import numpy as np

# You are still while your head turns slower than HEAD_SPEED (rad/s), none of
# FIDGETS (blendshapes; blinks don't count) changes faster than FACE_SPEED
# (score per s) and you don't talk. Speeds are smoothed over MOTION_SMOOTHING s.
HEAD_SPEED, FACE_SPEED, MOTION_SMOOTHING = 0.3, 1.5, 0.15
FIDGETS = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "browDownLeft", "browDownRight", "cheekPuff",
]
STILL_AFTER = 2.5  # s you must be still before the animal acts on its own
FADE_IN, FADE_OUT = 1.5, 0.3  # s

# Breaths per s and the nod that goes with one. A breath is drawn in over
# INHALE of its time, let out over EXHALE, then a pause.
BREATHING = (0.3, 0.03)
INHALE, EXHALE = 0.4, 0.45
# A slow sway that never stops: nod, turn, tilt, each a sine at its own (Hz)
# so together they never quite repeat.
SWAY, SWAY_HZ = np.array([0.03, 0.08, 0.05]), np.array([0.13, 0.071, 0.097])
LOOK_EVERY, LOOK_TIME = (1.5, 4.5), (1.2, 3.0)  # s between glances, s one lasts
LOOK = np.array([0.12, 0.32, 0.17])  # largest glance: nod, turn, tilt


def _ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _bump(t: float, duration: float) -> float:
    """0 .. 1 .. 0 over a bout of duration s, t s after its start: eases in, stays, eases out."""
    s = t / duration
    if not 0 <= s <= 1:
        return 0.0
    return float(_ease(s / 0.25) * (1 - _ease((s - 0.75) / 0.25)))


class Idle:
    """What the animal does on its own while you are still: head, a rotation
    vector to add to yours; breath, how full its lungs are (0..1). Both are 0
    while you move."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.motion = 0.0  # your smoothed movement, over 1 = moving
        self.still_for = 0.0
        self.amount = 0.0  # 0 = copying you .. 1 = acting on its own
        self.last_head, self.last_scores = None, None
        self.look = None  # (start, duration) of the current or next glance
        self.look_at = np.zeros(3)
        self.head, self.breath = np.zeros(3), 0.0

    def update(self, head: np.ndarray, scores: dict | None, talking: bool, dt: float, now: float):
        """head: your rotation vector; scores: your blendshapes, None while no face is seen."""
        moving = 0.0
        if self.last_head is not None:
            moving = np.linalg.norm(head - self.last_head) / dt / HEAD_SPEED
        if scores is not None and self.last_scores is not None:
            changed = max(abs(scores.get(n, 0) - self.last_scores.get(n, 0)) for n in FIDGETS)
            moving = max(moving, changed / dt / FACE_SPEED)
        self.last_head, self.last_scores = np.array(head, float), scores
        self.motion += min(dt / MOTION_SMOOTHING, 1.0) * (moving - self.motion)
        self.still_for = 0.0 if talking or self.motion > 1 else self.still_for + dt
        target = 1.0 if self.still_for >= STILL_AFTER else 0.0
        step = dt / (FADE_IN if target > self.amount else FADE_OUT)
        self.amount = float(np.clip(self.amount + np.clip(target - self.amount, -step, step), 0, 1))
        if self.amount == 0:  # next time, start afresh
            self.look = None
        self._act(now)

    def _bout(self, bout, now, every, lasting):
        if bout is None or now > bout[0] + bout[1]:
            return now + self.rng.uniform(*every), self.rng.uniform(*lasting)
        return bout

    def _act(self, now: float):
        look = self._bout(self.look, now, LOOK_EVERY, LOOK_TIME)
        if look is not self.look:
            self.look_at = self.rng.uniform(-1, 1, 3) * LOOK
        self.look = look
        glance = _bump(now - self.look[0], self.look[1])
        phase = (BREATHING[0] * now) % 1
        breath = float(_ease(phase / INHALE) * (1 - _ease((phase - INHALE) / EXHALE)))
        sway = SWAY * np.sin(2 * np.pi * SWAY_HZ * now)
        ease = float(_ease(self.amount))
        self.head = ease * (glance * self.look_at + sway + [BREATHING[1] * breath, 0.0, 0.0])
        self.breath = ease * breath
