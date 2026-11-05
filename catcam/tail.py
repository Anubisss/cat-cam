"""A tail lifted by one of your hands.

The tail is a chain of points rooted below the screen, beside the cat's body.
Your hand only draws its tip along; the rest follows like a real tail: it
lags behind, swings past and settles, bends evenly along its length (stiff at
the root, loose toward the tip) and keeps swishing slowly by itself. When the
hand leaves, the tail sinks out of view. Its skin is the arm of the cat's paw
photo, wrapped around a body that tapers to a point.
"""

import math
from typing import NamedTuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from catcam.paws import photo_paw

TAIL_HAND = "Right"  # this hand moves the tail; the other one stays a paw
SEGMENTS = 24
LENGTH = 1.2  # root to tip, in screen heights
# The root is under the screen, beside the cat's face centre, and as deep as
# it takes for the tail to reach your hand with some length to spare for a curve.
ROOT_BELOW = 0.1  # at least this far under the screen, in screen heights
ROOT_SIDE = 0.85  # in cat face widths
SLACK = 1.15  # tail length over the straight distance from root to hand
ROOT_FOLLOW = 0.15  # per frame, how fast the root slides to its new depth
ROOT_LEAN = 0.3  # how far the root turns from straight up toward your hand
BASE_RADIUS, TIP_RADIUS = 0.12, 0.018  # in cat face widths
TAPER = 0.8  # >1 thins the tail early, <1 keeps it thick longer

STEP = 1 / 120  # physics time step, s
ITERATIONS = 8  # passes per step that keep the segments' lengths
DRAG = 0.92  # velocity kept per step
TIP_PULL = 0.3  # per step, how far the tip moves toward your hand
STIFFNESS = (0.3, 0.1)  # per step, how strongly a joint straightens: at the root, at the tip
MAX_BEND = (12, 45)  # degrees a joint can bend: at the root, at the tip
SWISH = 3.0  # side-to-side sway, in screen heights / s^2
SWISH_RATE = 0.4  # sways per second
SWISH_WAVES = 0.6  # waves along the tail: the sway travels from root to tip
FADE_RATE = 0.12  # how fast the tail rises / sinks, per frame
HAND_GONE = 0.15  # the tail hand's presence (see Paws) below which it is gone: the
                  # tracker loses a fast hand for a few frames now and then

SKIN_WIDTH = 64  # texture columns across the tail
SKIN_INSET = 0.08  # of the arm width, left off each side (the photo's soft edge)


class TailSkin(NamedTuple):
    image: np.ndarray  # BGR, rows along the tail, columns across it
    width: float  # the arm's width in the photo, in photo px


def tail_skin(paw_photos: dict) -> TailSkin | None:
    """The arm of a paw photo ({(view, hand): BGRA}, see catcam.paws.paw_set),
    straightened into a strip of skin; None without photos."""
    photo = paw_photos.get(("front", "left"), paw_photos.get(("back", "left")))
    if photo is None:
        return None
    paw = photo_paw(photo)
    alpha = photo[..., 3] > 127
    rows = [r for r in range(min(int(paw.anchor[1]) + paw.width, len(photo) - 2), len(photo))
            if alpha[r].sum() > 2]
    spans = np.array([np.flatnonzero(alpha[r])[[0, -1]] for r in rows], float)
    inset = SKIN_INSET * (spans[:, 1] - spans[:, 0])
    across = np.linspace(0, 1, SKIN_WIDTH, dtype=np.float32)
    map_x = (spans[:, :1] + inset[:, None] + across * (spans[:, 1:] - spans[:, :1] - 2 * inset[:, None]))
    map_y = np.repeat(np.array(rows, np.float32)[:, None], SKIN_WIDTH, axis=1)
    image = cv2.remap(photo[..., :3], map_x.astype(np.float32), map_y, cv2.INTER_LINEAR)
    return TailSkin(image, float(np.median(spans[:, 1] - spans[:, 0])))


class Tail:
    def __init__(self, out_size):
        self.out_w, self.out_h = out_size
        self.points = None  # (SEGMENTS + 1, 2) from root to tip, px
        self.previous = None
        self.presence = 0.0
        self.side = 1.0  # -1: root left of the face, 1: right
        self.target = None
        self.root = None
        self.face_width = 1.0
        self.time = 0.0
        joints = np.linspace(0, 1, SEGMENTS + 1)[1:-1]
        self.stiffness = STIFFNESS[0] + (STIFFNESS[1] - STIFFNESS[0]) * joints
        max_bend = np.radians(MAX_BEND[0] + (MAX_BEND[1] - MAX_BEND[0]) * joints)
        self.min_span = 2 * np.cos(max_bend / 2)  # closest a joint's neighbours get, in segments
        sway = np.linspace(0, 1, SEGMENTS + 1)
        self.sway_weight = sway * (1 - sway)  # the root stays put, the tip follows your hand

    def update(self, hand, presence: float, dt: float, cat_face_center: np.ndarray, cat_face_width: float):
        """hand: (x, y, ...) where your tail hand is on screen (see Paws), or None;
        presence: 0..1 how surely it is there."""
        present = hand is not None and presence > HAND_GONE
        if present and self.presence < 0.02:  # rising: from the side the hand is on
            self.side = 1.0 if hand[0] >= cat_face_center[0] else -1.0
        self.presence += FADE_RATE * (float(present) - self.presence)
        if self.presence < 0.02 and not present:
            self.points = None
            return
        self.face_width = cat_face_width
        length = LENGTH * self.out_h
        if present:
            self.target = np.array(hand[:2], float)
        root_x = cat_face_center[0] + self.side * ROOT_SIDE * cat_face_width
        if self.target is None:
            self.target = np.array([root_x, 0.0])
        reach = length / SLACK
        depth = max(self.out_h * (1 + ROOT_BELOW),
                    self.target[1] + np.sqrt(max(reach ** 2 - (self.target[0] - root_x) ** 2, 0)))
        depth += (1 - self.presence) * length  # sinking out of view
        if self.points is None:
            self.root = np.array([root_x, depth])
            self.points = self.root - np.outer(np.arange(SEGMENTS + 1), (0, length / SEGMENTS))
            self.previous = self.points.copy()
        self.root = self.root + ROOT_FOLLOW * (np.array([root_x, depth]) - self.root)
        target = self.target + (0, (1 - self.presence) * length)
        for _ in range(int(np.clip(round(dt / STEP), 1, 8))):
            self.time += STEP
            self._step(target, length / SEGMENTS)

    def _step(self, target: np.ndarray, segment: float):
        p = self.points
        velocity = DRAG * (p - self.previous)
        self.previous = p.copy()
        p += velocity
        tangent = np.gradient(p, axis=0)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6
        phase = 2 * np.pi * (SWISH_RATE * self.time - SWISH_WAVES * np.linspace(0, 1, len(p)))
        p += (SWISH * self.out_h * STEP ** 2) * (self.sway_weight * np.sin(phase))[:, None] * normal
        p[-1] += TIP_PULL * (target - p[-1])

        up = np.array([0.0, -1.0])
        toward = (target - self.root) / (np.linalg.norm(target - self.root) + 1e-6)
        root_direction = up + ROOT_LEAN * (toward - up)
        root_direction /= np.linalg.norm(root_direction) + 1e-6
        p[1:-1] += self.stiffness[:, None] * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        p[0], p[1] = self.root, self.root + root_direction * segment
        _constrain(p, segment, (self.min_span * segment).tolist())

    def draw(self, frame: np.ndarray, skin: TailSkin) -> np.ndarray:
        if self.points is None:
            return frame
        out = frame.copy()
        _paint(out, self.points, self.face_width, skin)
        return out


def _constrain(p: np.ndarray, segment: float, min_span: list[float]):
    """Moves the points of p (in place, the first two fixed) until every
    segment is `segment` long and points two apart are at least min_span[i]
    apart, so no joint bends further than the tail can."""
    xs, ys = p[:, 0].tolist(), p[:, 1].tolist()
    n = len(xs)
    weight = [0.0, 0.0] + [1.0] * (n - 2)

    def project(a, b, length, at_least):
        dx, dy = xs[b] - xs[a], ys[b] - ys[a]
        d = math.hypot(dx, dy) + 1e-9
        if at_least and d >= length:
            return
        share = (d - length) / (d * (weight[a] + weight[b]))
        xs[a] += weight[a] * share * dx
        ys[a] += weight[a] * share * dy
        xs[b] -= weight[b] * share * dx
        ys[b] -= weight[b] * share * dy

    for k in range(ITERATIONS):
        order = range(1, n - 1) if k % 2 else range(n - 2, 0, -1)  # alternate directions
        for i in order:
            project(i, i + 1, segment, False)
        for i in range(n - 2):
            if weight[i] or weight[i + 2]:
                project(i, i + 2, min_span[i], True)
    p[:, 0], p[:, 1] = xs, ys


def _paint(out: np.ndarray, points: np.ndarray, face_width: float, skin: TailSkin):
    """Draws the tail along points (root to tip) into out, in place."""
    spline = CubicSpline(np.arange(len(points)), points)
    rough = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    centre = spline(np.linspace(0, len(points) - 1, int(2 * rough) + 2))  # about 0.5 px apart
    steps = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    along = np.concatenate([[0], np.cumsum(steps)])
    length = along[-1]
    radius = TIP_RADIUS * face_width + (BASE_RADIUS - TIP_RADIUS) * face_width * (1 - along / length) ** TAPER
    # Skin rows: the photo's proportions are kept as the tail thins.
    rows = np.concatenate([[0], np.cumsum(steps * skin.width / (radius[1:] + radius[:-1]))])
    tangent = np.gradient(centre, axis=0)

    # Only the part of the screen near the tail, padded so a pixel's nearest
    # point on the tail is in the box even where the tail leaves the screen.
    frame_h, frame_w = out.shape[:2]
    pad = int(np.ceil(radius.max())) + 2
    x0, y0 = np.maximum(np.floor(centre.min(axis=0)).astype(int) - pad, (-pad, -pad))
    x1, y1 = np.minimum(np.ceil(centre.max(axis=0)).astype(int) + pad, (frame_w + pad, frame_h + pad))
    if x1 <= max(x0, 0) or y1 <= max(y0, 0) or x0 >= frame_w or y0 >= frame_h:
        return
    pixel = np.round(centre - (x0, y0)).astype(int)
    inside = np.all((pixel >= 0) & (pixel < (x1 - x0, y1 - y0)), axis=1)
    pixel, samples = pixel[inside], np.flatnonzero(inside)
    if not len(samples):
        return
    line = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    line[pixel[:, 1], pixel[:, 0]] = 0
    distance, labels = cv2.distanceTransformWithLabels(line, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    nearest = np.zeros(labels.max() + 1, int)
    nearest[labels[pixel[:, 1], pixel[:, 0]]] = samples

    # Each screen pixel close to the tail takes the skin across from its
    # nearest point on the centre line.
    vx0, vy0 = max(-x0, 0), max(-y0, 0)
    vx1, vy1 = min(frame_w - x0, x1 - x0), min(frame_h - y0, y1 - y0)
    ys, xs = np.nonzero(distance[vy0:vy1, vx0:vx1] <= radius.max() + 1)
    ys, xs = ys + vy0, xs + vx0
    k = nearest[labels[ys, xs]]
    offset = np.stack([xs + x0, ys + y0], axis=1) - centre[k]
    dist = np.linalg.norm(offset, axis=1)
    coverage = np.clip(radius[k] - dist + 0.5, 0, 1)
    keep = coverage > 0
    if not keep.any():
        return
    ys, xs, k, offset, dist, coverage = ys[keep], xs[keep], k[keep], offset[keep], dist[keep], coverage[keep]
    # Split the offset into along and across the tail, so the skin doesn't
    # step from one centre point to the next.
    t = tangent[k] / (np.linalg.norm(tangent[k], axis=1, keepdims=True) + 1e-9)
    side = t[:, 0] * offset[:, 1] - t[:, 1] * offset[:, 0]
    across = (0.5 + 0.5 * np.clip(side / radius[k], -1, 1)) * (skin.image.shape[1] - 1)
    span = len(skin.image) - 1
    row = rows[k] + np.sum(offset * t, axis=1) * skin.width / (2 * radius[k])
    down = span - np.abs(np.mod(row, 2 * span) - span)  # the photo, mirrored back and forth
    color = _sample(skin.image, across, down)
    sy, sx = ys + y0, xs + x0
    a = coverage[:, None]
    out[sy, sx] = np.clip(color * a + out[sy, sx] * (1 - a), 0, 255).astype(np.uint8)


def _sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """image at the points (x, y), interpolated: (n, channels)."""
    n, columns = len(x), 1024  # remap's maps must stay under 32767 px a side
    rows = -(-n // columns)
    maps = [np.resize(v.astype(np.float32), rows * columns).reshape(rows, columns) for v in (x, y)]
    return cv2.remap(image, *maps, cv2.INTER_LINEAR).reshape(rows * columns, -1)[:n]
