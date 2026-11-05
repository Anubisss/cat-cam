"""The inside of a 3D head's mouth: real geometry, not a picture.

Teeth, gums, a palate and a tongue, modelled in Blender by tools/mouth_model.py
(one set for cats, one for dogs, in assets/mouth/), plus a mouth cavity behind
them for heads without their own inside. All of it lives in a local frame in
mouth widths: origin where the lips meet in the middle, x toward the
image-right corner, y up, z out of the mouth. The lower half (lower teeth and
gum, tongue, the cavity's floor) carries a jaw weight, so it swings open about
a hinge behind the mouth while the upper half stays with the skull. Head3D fits
the frame to the head's lips.
"""

from typing import NamedTuple

import numpy as np

from catcam.paths import MOUTHS

ANIMALS = ("cat", "dog")  # the modelled sets, assets/mouth/<animal>.npz
HINGE = (0.0, 0.1, -0.9)  # the jaw's pivot, behind and a little above the lips, mouth widths
# Centre, radii; only its back half is built. Wider than the mouth so it fills the corners:
# Head3D only shows the mouth through the lips, so it can't poke out through the cheeks.
CAVITY = ((0.0, -0.08, -0.35), (0.9, 0.5, 0.5))
CAVITY_FRONT = -0.03  # the cavity's open rim, just behind the lips
CAVITY_COLOUR = (0.32, 0.1, 0.11)  # RGB 0..1, tinted by the cat's lip colour
THROAT = 0.15  # the cavity darkens to this at its back


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


def _cavity(tint: np.ndarray) -> Parts:
    """The back half of an ellipsoid seen from inside, its floor in the jaw."""
    (_, cy, cz), radii = CAVITY
    u, v = 24, 16
    theta = np.linspace(0, np.pi, v)
    phi = np.linspace(0, 2 * np.pi, u, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)], axis=-1).reshape(-1, 3)
    pos = np.asarray(CAVITY[0]) + unit * radii
    normal = unit / np.asarray(radii)
    normal /= -np.linalg.norm(normal, axis=1, keepdims=True)  # facing in
    faces = _grid_faces(v, u, wrap=True)[:, [0, 2, 1]]
    faces = faces[(cz + unit[:, 2] * radii[2] <= CAVITY_FRONT)[faces].all(axis=1)]
    depth = np.clip((CAVITY_FRONT - pos[:, 2]) / (2 * radii[2]), 0, 1)
    return Parts(pos, normal, np.tile(np.multiply(CAVITY_COLOUR, tint), (len(pos), 1)),
                 np.clip(0.5 - (pos[:, 1] - cy) / radii[1], 0, 1), 1 - (1 - THROAT) * depth, faces)


def load(animal: str, tint) -> Parts:
    """The modelled teeth, gums, palate and tongue of animal (one of ANIMALS), the soft
    parts taking on the cat's lip colour as far as the model says."""
    data = np.load(MOUTHS / f"{animal}.npz")
    tint = np.asarray(tint, float)
    shift = data["tint"].astype(float)[:, None]
    colour = data["colour"] / 255 * (1 + shift * (tint - 1))
    return Parts(data["pos"].astype(float), data["normal"].astype(float), colour,
                 data["jaw"].astype(float), data["light"].astype(float), data["faces"].astype(np.int64))


def build(tint, cavity: bool = True, animal: str = "cat") -> Parts:
    """All the mouth parts; tint: the cat's lip colour (RGB, mean 1). cavity: whether
    to add the mouth cavity (a head with its own mouth has its own inside)."""
    tint = np.asarray(tint, float)
    return _merge([load(animal, tint)] + ([_cavity(tint)] if cavity else []))
