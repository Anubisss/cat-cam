"""The inside of a 3D head's mouth: real geometry, not a picture.

Teeth (incisors, fangs, premolars), gums, a tongue and the mouth cavity, built
in a local frame in mouth widths: origin where the lips meet in the middle,
x toward the image-right corner, y up, z out of the mouth. The lower half
(lower teeth, gums, tongue, the cavity's floor) carries a jaw weight, so it
swings open about a hinge behind the mouth while the upper half stays with
the skull. Head3D fits the frame to the head's lip landmarks.
"""

from typing import NamedTuple

import numpy as np

# Where things sit, in mouth widths.
HINGE = (0.0, 0.1, -0.9)  # the jaw's pivot, behind and a little above the lips
ARCH_FRONT, ARCH_BACK, ARCH_HALF = -0.06, -0.36, 0.38  # dental arch: z at the front and back, half width
UPPER_GUM_Y, LOWER_GUM_Y = 0.02, -0.03  # where the teeth come out of the gums
# Teeth from the middle outward: (x along the arch, base radius, length, tip radius as a
# share of the base: blunt incisors, pointed fangs); mirrored left/right.
UPPER_TEETH = [(0.035, 0.026, 0.06, 0.6), (0.085, 0.026, 0.06, 0.6), (0.135, 0.028, 0.065, 0.55),
               (0.24, 0.06, 0.26, 0.05), (0.33, 0.045, 0.08, 0.25)]
LOWER_TEETH = [(0.03, 0.022, 0.045, 0.6), (0.075, 0.022, 0.045, 0.6), (0.12, 0.024, 0.05, 0.55),
               (0.2, 0.05, 0.19, 0.05), (0.29, 0.04, 0.06, 0.25)]
FANG_CURVE = 0.25  # fang tips bend back by this share of their length
# Centre, radii: long, flat, lying on the jaw's floor behind the lower front teeth.
TONGUE = ((0.0, -0.15, -0.4), (0.23, 0.045, 0.3))
TONGUE_GROOVE = (0.012, 0.045)  # the dip down its middle: depth, half width
# Centre, radii; only its back half is built. Wider than the mouth so it fills the corners:
# Head3D only shows the mouth through the lips, so it can't poke out through the cheeks.
CAVITY = ((0.0, -0.08, -0.35), (0.9, 0.5, 0.5))
CAVITY_FRONT = -0.03  # the cavity's open rim, just behind the lips

# Colours (RGB 0..1). Gums, tongue and cavity are tinted by the cat's lip colour.
ENAMEL = (0.93, 0.9, 0.82)
ENAMEL_ROOT = (0.78, 0.72, 0.6)
GUM = (0.55, 0.28, 0.3)
TONGUE_COLOUR = (0.64, 0.32, 0.35)
CAVITY_COLOUR = (0.32, 0.1, 0.11)
THROAT = 0.15  # the cavity darkens to this at its back
ENAMEL_LIGHT = 1.3  # teeth catch more light than the soft tissue around them


class Parts(NamedTuple):
    """Mouth geometry in the local frame: per vertex position, normal, colour (RGB),
    jaw weight (0 = skull, 1 = jaw) and light (ambient occlusion, 0..1)."""
    pos: np.ndarray
    normal: np.ndarray
    colour: np.ndarray
    jaw: np.ndarray
    light: np.ndarray
    faces: np.ndarray


def _merge(parts: list[Parts]) -> Parts:
    offsets = np.cumsum([0] + [len(p.pos) for p in parts[:-1]])
    return Parts(*(np.concatenate([getattr(p, f) for p in parts]) for f in Parts._fields[:-1]),
                 np.concatenate([p.faces + o for p, o in zip(parts, offsets)]))


def _grid_faces(rows: int, cols: int, wrap: bool = False) -> np.ndarray:
    faces = []
    for r in range(rows - 1):
        for c in range(cols if wrap else cols - 1):
            a, b = r * cols + c, r * cols + (c + 1) % cols
            faces += [(a, a + cols, b), (b, a + cols, b + cols)]
    return np.array(faces)


def arch(x: np.ndarray) -> np.ndarray:
    """z of the dental arch at x: a parabola from the front teeth back to the molars."""
    return ARCH_FRONT + (ARCH_BACK - ARCH_FRONT) * (np.asarray(x) / ARCH_HALF) ** 2


def _tooth(x: float, radius: float, length: float, tip: float, upper: bool) -> Parts:
    """A tapered cone growing out of the gum at x on the arch, tip down (upper) or up;
    tip: its end's radius as a share of the base's."""
    sides, rings = 8, 5
    z0 = float(arch(x))
    # Out of the arch (toward the lips) and along it, to flatten the tooth a little.
    out = np.array([2 * (ARCH_BACK - ARCH_FRONT) * x / ARCH_HALF ** 2, 0, -1.0])
    out = -out / np.linalg.norm(out)
    along = np.cross([0, 1, 0], out)
    sign = -1 if upper else 1
    base = np.array([x, UPPER_GUM_Y if upper else LOWER_GUM_Y, z0])
    fang = length > 0.1
    pos, normal, colour = [], [], []
    for i in range(rings):
        t = i / (rings - 1)
        r = radius * (tip + (1 - tip) * (1 - t) ** 0.8)
        bent = FANG_CURVE * length * t ** 2 if fang else 0.0
        centre = base + np.array([0, sign * length * t, 0]) - out * bent
        for k in range(sides):
            a = 2 * np.pi * k / sides
            d = np.cos(a) * out * 0.75 + np.sin(a) * along
            pos.append(centre + r * d)
            normal.append(d + np.array([0, -sign * 0.3, 0]))
            colour.append(np.add(np.multiply(ENAMEL_ROOT, 1 - t), np.multiply(ENAMEL, t)))
    pos, normal = np.array(pos), np.array(normal)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    faces = _grid_faces(rings, sides, wrap=True)
    if not upper:
        faces = faces[:, [0, 2, 1]]
    n = len(pos)
    jaw = np.full(n, 0.0 if upper else 1.0)
    return Parts(pos, normal, np.array(colour), jaw, np.full(n, ENAMEL_LIGHT), faces)


def _gum(upper: bool, tint: np.ndarray) -> Parts:
    """A band along the arch the teeth grow out of."""
    xs = np.linspace(-ARCH_HALF, ARCH_HALF, 25)
    y = UPPER_GUM_Y if upper else LOWER_GUM_Y
    sign = 1 if upper else -1
    rows = [(0.0, 0.0), (0.035, 0.015), (0.06, -0.02)]  # (height into the gum, step back)
    pos = np.array([[x, y + sign * h, arch(x) - back] for h, back in rows for x in xs])
    normal = np.tile([0, -sign * 0.5, 1.0], (len(pos), 1))
    faces = _grid_faces(len(rows), len(xs))
    if upper:
        faces = faces[:, [0, 2, 1]]
    n = len(pos)
    colour = np.tile(np.multiply(GUM, (1 + tint) / 2), (n, 1))  # only half the lip tint: it's strong
    return Parts(pos, normal / np.linalg.norm(normal, axis=1, keepdims=True), colour,
                 np.full(n, 0.0 if upper else 1.0), np.full(n, 0.9), faces)


def _ellipsoid(centre, radii, keep, colour, jaw, light, inward=False, u=24, v=16) -> Parts:
    """Points of an ellipsoid where keep(unit sphere point) is true, as a grid."""
    theta = np.linspace(0, np.pi, v)
    phi = np.linspace(0, 2 * np.pi, u, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)], axis=-1).reshape(-1, 3)
    pos = np.asarray(centre) + unit * radii
    normal = unit / np.asarray(radii)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    faces = _grid_faces(v, u, wrap=True)
    faces = faces[keep(unit)[faces].all(axis=1)]
    if inward:
        normal, faces = -normal, faces[:, [0, 2, 1]]
    return Parts(pos, normal, colour(pos), jaw(pos), light(pos), faces)


def build(tint) -> Parts:
    """All the mouth parts; tint: the cat's lip colour (RGB, mean 1)."""
    tint = np.asarray(tint, float)
    teeth = [_tooth(side * x, r, length, tip, upper)
             for upper, row in ((True, UPPER_TEETH), (False, LOWER_TEETH))
             for x, r, length, tip in row for side in (-1, 1)]
    (cx, cy, cz), radii = CAVITY
    depth = lambda pos: np.clip((CAVITY_FRONT - pos[:, 2]) / (2 * radii[2]), 0, 1)  # noqa: E731
    cavity = _ellipsoid(
        CAVITY[0], radii, keep=lambda unit: cz + unit[:, 2] * radii[2] <= CAVITY_FRONT,
        colour=lambda pos: np.tile(np.multiply(CAVITY_COLOUR, tint), (len(pos), 1)),
        jaw=lambda pos: np.clip(0.5 - (pos[:, 1] - cy) / radii[1], 0, 1),
        light=lambda pos: 1 - (1 - THROAT) * depth(pos), inward=True)
    tongue = _ellipsoid(
        TONGUE[0], TONGUE[1], keep=lambda unit: unit[:, 1] > -0.3,
        colour=lambda pos: np.tile(np.multiply(TONGUE_COLOUR, (1 + tint) / 2), (len(pos), 1)),
        jaw=lambda pos: np.ones(len(pos)),
        light=lambda pos: 1 - 0.65 * np.clip((ARCH_FRONT - pos[:, 2]) / 0.6, 0, 1))
    top = tongue.pos[:, 1] > TONGUE[0][1]
    depth, half = TONGUE_GROOVE
    tongue.pos[top, 1] -= depth * np.exp(-((tongue.pos[top, 0] / half) ** 2))
    return _merge(teeth + [_gum(True, tint), _gum(False, tint), tongue, cavity])
