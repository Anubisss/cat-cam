"""Models the inside of a mouth in Blender, once for a cat and once for a dog:
teeth with real crowns, the gums they sit in, a palate with its ridges and a
tongue. No lips or skin: the heads have their own. Run it headless:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python tools/mouth_model.py -- [--preview DIR]

It writes assets/mouth/cat.npz and assets/mouth/dog.npz for catcam.mouth3d: per
vertex position, normal, colour, jaw weight, how much the cat's lip colour tints
it, and light (ambient occlusion from ray casts, darker deeper in the mouth).
--preview also renders each set from the front, the side and from below with
the jaw open, to judge the shapes.

Everything is built in catcam.mouth3d's mouth frame, in mouth widths: origin
where the lips meet in the middle, x toward the image-right corner, y up, z out
of the mouth. Blender's axes are used as they are, so "up" is y here.
"""

import argparse
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "mouth"

# The dental arch, in mouth widths: z of the front teeth, and z at x = ±ARCH_HALF on the
# way back (a parabola, continued past there for the cheek teeth).
ARCH_FRONT, ARCH_BACK, ARCH_HALF = -0.06, -0.36, 0.38
LOWER_ARCH = (0.9, -0.03)  # the lower arch is this much narrower and sits this far back, inside the upper
UPPER_GUM_Y, LOWER_GUM_Y = 0.02, -0.03  # where the teeth come out of the gums
ROOT_DEPTH = 0.02  # teeth start this far inside the gum, so nothing shows where they come out
FIRST_GAP = 0.004  # between the two middle incisors

# Teeth on each side, from the middle back: (kind, width along the arch, thickness across
# it, crown length, gap before it), mouth widths. Cats: tiny incisors, short rounded fangs,
# a gap, then blade-like premolars ending in the carnassial. Dogs: bigger incisors, stout
# rounded fangs and a row of premolars. Friendly, not fierce: the fangs are kept short.
TEETH = {
    "cat": {
        "upper": [("incisor", 0.022, 0.016, 0.03, 0), ("incisor", 0.022, 0.016, 0.032, 0.003),
                  ("incisor", 0.026, 0.02, 0.04, 0.003), ("canine", 0.05, 0.044, 0.1, 0.02),
                  ("premolar", 0.03, 0.018, 0.025, 0.05), ("premolar", 0.055, 0.024, 0.05, 0.006),
                  ("carnassial", 0.095, 0.03, 0.06, 0.006)],
        "lower": [("incisor", 0.018, 0.014, 0.025, 0), ("incisor", 0.018, 0.014, 0.027, 0.003),
                  ("incisor", 0.02, 0.016, 0.03, 0.003), ("canine", 0.044, 0.04, 0.08, 0.01),
                  ("premolar", 0.05, 0.022, 0.045, 0.06), ("premolar", 0.055, 0.024, 0.05, 0.006),
                  ("molar", 0.08, 0.028, 0.055, 0.006)],
    },
    "dog": {
        "upper": [("incisor", 0.032, 0.022, 0.045, 0), ("incisor", 0.032, 0.022, 0.047, 0.003),
                  ("incisor", 0.04, 0.03, 0.06, 0.004), ("canine", 0.07, 0.06, 0.11, 0.016),
                  ("premolar", 0.05, 0.024, 0.045, 0.03), ("premolar", 0.06, 0.026, 0.05, 0.008),
                  ("carnassial", 0.1, 0.032, 0.07, 0.008)],
        "lower": [("incisor", 0.028, 0.02, 0.04, 0), ("incisor", 0.028, 0.02, 0.042, 0.003),
                  ("incisor", 0.032, 0.024, 0.048, 0.003), ("canine", 0.064, 0.056, 0.09, 0.012),
                  ("premolar", 0.045, 0.022, 0.04, 0.035), ("premolar", 0.05, 0.024, 0.045, 0.008),
                  ("premolar", 0.055, 0.026, 0.05, 0.008)],
    },
}
CANINE_CURVE = {"cat": 0.08, "dog": 0.06}  # fang tips bend back into the mouth by this share of their length

# Gums: the crest wraps each tooth this much beyond its thickness, and dips between teeth
# (and rises over each tooth's middle) by SCALLOP; heights of the band into the lip on the
# outside and toward the palate or floor inside.
GUM_COVER, SCALLOP = 0.012, 0.007
GUM_OUTSIDE = {False: 0.05, True: 0.03}  # upper, lower: the lower lip covers less of its gum
GUM_INSIDE = 0.035
# Across the band, from the lip side over the crest to the inside: (out, into, on the crest)
# as shares of (half its thickness, its height on that side, the crest's scallop).
GUM_SECTION = [(1.1, 1, 0), (1.15, 0.5, 0), (1.0, 0.12, 0.5), (0.7, 0, 1), (-0.7, 0, 1),
               (-1.05, 0.45, 0), (-1.25, 1, 0)]
GUM_PAST = 0.03  # the band runs on this far past the last tooth

# Palate: a vault inside the upper arch, from behind the incisors to PALATE_BACK, with
# transverse ridges (rugae) bowed back toward the sides, split by the midline.
PALATE_BACK = -0.85
PALATE_VAULT = 0.03  # how much higher its middle is than its edges
RUGAE = {"cat": 6, "dog": 7}  # how many
RUGA_SPACING, RUGA_RADIUS, RUGA_SWEEP = 0.055, 0.007, 0.9  # z step, thickness, how far back they bow

# Tongue: (tip z, back z, top y, half width at the tip and at the back, thickness, groove
# depth and half width). It lies in the lower arch, its tip behind the lower incisors.
TONGUE = {
    "cat": (-0.13, -0.85, -0.045, 0.12, 0.2, 0.09, 0.008, 0.03),
    "dog": (-0.12, -0.9, -0.04, 0.15, 0.26, 0.1, 0.012, 0.045),
}

# Colours (RGB 0..1), and how much the cat's lip colour tints each (0 = not at all, 1 = fully).
ENAMEL, ENAMEL_ROOT = (0.95, 0.93, 0.86), (0.82, 0.76, 0.62)
GUM, GUM_MARGIN = (0.62, 0.3, 0.33), (0.78, 0.47, 0.48)  # deep, and the paler crest at the teeth
PALATE, RUGA = (0.55, 0.27, 0.3), (0.72, 0.44, 0.45)
TONGUE_COLOUR, TONGUE_TIP = {"cat": (0.78, 0.45, 0.5), "dog": (0.74, 0.34, 0.4)}, (0.85, 0.5, 0.54)
PAPILLAE = {"cat": 0.1, "dog": 0.04}  # speckle on the tongue: cats' is rough
SOFT_TINT = 0.5  # gums, palate and tongue take half the lip colour: it's strong

# Light: ambient occlusion from AO_RAYS rays over each vertex's hemisphere, blocked by
# anything within AO_REACH (mouth widths) in the same half of the mouth; the throat end
# darkens by up to DEEP over DEEP_SPAN behind the front teeth; enamel catches more light.
AO_RAYS, AO_REACH, AO_FLOOR = 64, 0.15, 0.3
DEEP, DEEP_SPAN = 0.6, 0.8
ENAMEL_LIGHT = 1.25


def _gauss(u, centre, width):
    return np.exp(-(((u - centre) / width) ** 2))


class Arch:
    """The dental arch: points, tangent and outward direction at a signed distance s
    along it from the middle (negative to the image-left)."""

    def __init__(self, lower: bool):
        scale, back = LOWER_ARCH if lower else (1.0, 0.0)
        x = np.linspace(0, 0.7, 2000)
        z = ARCH_FRONT + (ARCH_BACK - ARCH_FRONT) * (x / ARCH_HALF) ** 2
        self.x, self.z = x * scale, z + back
        self.s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.z)))])

    def point(self, s):
        s = np.asarray(s, float)
        a = np.abs(s)
        return np.stack([np.sign(s) * np.interp(a, self.s, self.x), np.zeros_like(s),
                         np.interp(a, self.s, self.z)], axis=-1)

    def frame(self, s):
        """(point, along (toward larger s), out (away from the mouth's inside))."""
        p = self.point(s)
        along = self.point(np.asarray(s) + 1e-4) - self.point(np.asarray(s) - 1e-4)
        along /= np.linalg.norm(along, axis=-1, keepdims=True)
        out = np.stack([-along[..., 2], np.zeros_like(along[..., 0]), along[..., 0]], axis=-1)
        return p, along, out


def place_teeth(animal: str, lower: bool):
    """Each tooth of one jaw as (kind, s of its middle, width, thickness, length), both sides."""
    teeth, s = [], FIRST_GAP / 2
    for kind, width, thick, length, gap in TEETH[animal]["lower" if lower else "upper"]:
        s += gap + width / 2
        teeth += [(kind, side * s, width, thick, length) for side in (1, -1)]
        s += width / 2
    return teeth


def _grid_faces(rows: int, cols: int, wrap: bool = False):
    faces = []
    for r in range(rows - 1):
        for c in range(cols if wrap else cols - 1):
            a, b = r * cols + c, r * cols + (c + 1) % cols
            faces += [(a, b, a + cols), (b, b + cols, a + cols)]
    return faces


def _ring_tooth(c, along, out, up, width, thick, length, kind, curve):
    """An incisor (a chisel: flat across the arch, thinning to its edge) or a canine (a
    curved cone) from its rings of cross-sections, root in the gum."""
    sides = 8
    if kind == "canine":
        ts = [-ROOT_DEPTH / length, 0, 0.15, 0.35, 0.55, 0.75, 0.9]
        # A rounded cone: it tapers, then closes in a dome rather than a point.
        size = lambda t: (1 - 0.5 * t) * np.sqrt(max(1 - t ** 3, 0.0))  # noqa: E731
        across, power, bend = lambda t: 1.0, 2.0, curve  # noqa: E731
    else:
        ts = [-ROOT_DEPTH / length, 0, 0.4, 0.75, 0.92]
        size = lambda t: (1 + 0.08 * t) * np.sqrt(max(1 - t ** 6, 0.0)) * (1 - 0.15 * t)  # noqa: E731
        across, power, bend = lambda t: 1 - 0.6 * t, 3.0, 0.12  # noqa: E731
    angle = 2 * np.pi * np.arange(sides) / sides
    ca, sa = np.cos(angle), np.sin(angle)
    ring_a = np.sign(ca) * np.abs(ca) ** (2 / power)
    ring_b = np.sign(sa) * np.abs(sa) ** (2 / power)
    verts = []
    for t in ts:
        tc = max(t, 0)
        centre = c + up * length * t - out * bend * length * tc ** 2
        a, b = width / 2 * size(tc), thick / 2 * size(tc) * across(tc)
        verts += list(centre + np.outer(ring_a * a, along) + np.outer(ring_b * b, out))
    verts.append(c + up * length - out * bend * length)
    faces = _grid_faces(len(ts), sides, wrap=True)
    tip, last = len(verts) - 1, (len(ts) - 1) * sides
    faces += [(last + k, last + (k + 1) % sides, tip) for k in range(sides)]
    return np.array(verts), np.array(faces)


def _blade_profile(kind):
    """Crown height (share of its length) along a cheek tooth, u from -1 (front) to 1."""
    taper = lambda u: (1 - u ** 2) ** 0.4  # noqa: E731
    if kind == "carnassial":  # the upper shearing tooth: a small front cusp, a tall one, a blade
        return lambda u: taper(u) * (0.15 + np.maximum.reduce(
            [0.4 * _gauss(u, -0.62, 0.2), 0.85 * _gauss(u, -0.12, 0.3), 0.55 * _gauss(u, 0.5, 0.3)]))
    if kind == "molar":  # the lower one: two cusps with a notch between
        return lambda u: taper(u) * (0.15 + np.maximum(0.8 * _gauss(u, -0.35, 0.3),
                                                       0.85 * _gauss(u, 0.32, 0.3)))
    return lambda u: taper(u) * np.minimum(0.2 + 0.8 * _gauss(u, -0.1, 0.42) + 0.3 * _gauss(u, 0.6, 0.18), 1)


def _blade_tooth(c, along, out, up, width, thick, length, kind):
    """A cheek tooth: a blade along the arch, thin at its cusps, from slices across it."""
    us, phis = np.linspace(-1, 1, 9), np.linspace(0, np.pi, 7)
    height = length * _blade_profile(kind)(us)
    envelope = (1 - us ** 2) ** 0.35
    verts = [c + along * width / 2 * u + out * thick / 2 * e * np.cos(p)
             + up * (-ROOT_DEPTH + (h + ROOT_DEPTH) * np.sin(p) ** 0.7)
             for u, h, e in zip(us, height, envelope) for p in phis]
    return np.array(verts), np.array(_grid_faces(len(us), len(phis)))


def teeth(animal: str, lower: bool):
    """(name, verts, faces, kinds) of every tooth in one jaw."""
    arch = Arch(lower)
    up = np.array([0, 1.0 if lower else -1.0, 0])
    gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y
    out = []
    for kind, s, width, thick, length in place_teeth(animal, lower):
        point, along, outward = arch.frame(s)
        along = along * np.sign(s)  # from the middle toward the back, on either side
        base = point + [0, gum_y, 0]
        if kind in ("incisor", "canine"):
            v, f = _ring_tooth(base, along, outward, up, width, thick, length, kind, CANINE_CURVE[animal])
        else:
            v, f = _blade_tooth(base, along, outward, up, width, thick, length, kind)
        out.append((v, f, base, up, length))
    return out


def gum(animal: str, lower: bool):
    """The band of gum along one arch the teeth come out of: a U-shaped section from the
    lip side over the crest to the inside, scalloped around each tooth."""
    arch = Arch(lower)
    placed = place_teeth(animal, lower)
    end = max(abs(s) + w / 2 for _, s, w, _, _ in placed) + GUM_PAST
    stations = {0.0, end, -end}  # each tooth's middle and edges, so the scallops fall on teeth
    for _, s, w, _, _ in placed:
        stations |= {s - w / 2, s, s + w / 2}
    ss = np.array(sorted(stations))
    near = [min(placed, key=lambda t: abs(t[1] - s)) for s in ss]
    half = np.array([t[3] / 2 for t in near]) + GUM_COVER
    crest = np.array([SCALLOP if abs(s - t[1]) < 1e-6 else -SCALLOP * 0.5 for s, t in zip(ss, near)])
    crest[[0, -1]] = 0
    into = np.array([0, -1.0 if lower else 1.0, 0])  # into the gum, away from the teeth's tips
    gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y
    verts, margin = [], []
    for s, h, m in zip(ss, half, crest):
        p, _, out = arch.frame(s)
        for o, height, on_crest in GUM_SECTION:
            side = GUM_OUTSIDE[lower] if o > 0 else GUM_INSIDE
            verts.append(p + [0, gum_y, 0] + out * o * h + into * (height * side + on_crest * m))
            margin.append(on_crest)
    faces = np.array(_grid_faces(len(ss), len(GUM_SECTION)))
    return np.array(verts), faces, np.array(margin, float), -into


def _inner_edge(gum_verts):
    """The upper gum's inner top edge on the right, from the back to the front."""
    edge = gum_verts[len(GUM_SECTION) - 1::len(GUM_SECTION)]
    edge = edge[edge[:, 0] >= 0]
    return edge[np.argsort(edge[:, 2])]


def palate_y(x, z, front, edge):
    """Height of the palate at (x, z): its edge on the gum, vaulting up in the middle
    once past the front teeth. edge: the gum's inner edge, from _inner_edge."""
    half = np.interp(z, edge[:, 2], edge[:, 0]) * 1.04
    across = np.clip(x / max(half, 1e-6), -1, 1)
    return UPPER_GUM_Y + GUM_INSIDE + 0.003 + PALATE_VAULT * (1 - across ** 2) * min((front - z) / 0.15, 1)


def palate(gum_verts):
    """A vault filling the upper arch from its inner gum edge back to the throat."""
    edge = _inner_edge(gum_verts)
    front = edge[-1, 2]
    rows = front + (PALATE_BACK - front) * np.linspace(0, 1, 13) ** 1.4
    verts = []
    for z in rows:
        half = np.interp(z, edge[:, 2], edge[:, 0]) * 1.04
        verts += [(x, palate_y(x, z, front, edge), z) for x in np.linspace(-half, half, 11)]
    return np.array(verts), np.array(_grid_faces(len(rows), 11))


def rugae(animal: str, gum_verts):
    """Ridges across the palate, as Blender curves with a round bevel, two per row (they
    part at the midline) and bowing back toward the sides."""
    edge = _inner_edge(gum_verts)
    front = edge[-1, 2]
    objects = []
    for k in range(RUGAE[animal]):
        z0 = front - 0.05 - k * RUGA_SPACING
        half = np.interp(z0, edge[:, 2], edge[:, 0])
        for side in (1, -1):
            xs = side * np.linspace(0.025, 0.85 * half, 6)
            zs = z0 - RUGA_SWEEP * xs ** 2
            ys = [palate_y(x, z, front, edge) - 0.3 * RUGA_RADIUS for x, z in zip(xs, zs)]
            curve = bpy.data.curves.new(f"ruga{k}{side}", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth, curve.bevel_resolution, curve.use_fill_caps = RUGA_RADIUS, 0, True
            spline = curve.splines.new("POLY")
            spline.points.add(len(xs) - 1)
            for point, x, y, z in zip(spline.points, xs, ys, zs):
                point.co = (x, y, z, 1)
            # Thinner toward the sides, where they fade into the gum.
            for i, point in enumerate(spline.points):
                point.radius = 1.0 - 0.5 * i / (len(xs) - 1)
            ob = bpy.data.objects.new(curve.name, curve)
            bpy.context.collection.objects.link(ob)
            objects.append(ob)
    return objects


def tongue(animal: str):
    """The tongue: a rounded cube, smoothed by subdivision, then shaped: a round tip
    widening to the back, a flat top with a groove down its middle, the tip dipping."""
    tip, back, top, tip_half, back_half, thick, groove, groove_half = TONGUE[animal]
    bpy.ops.mesh.primitive_cube_add(size=2)
    ob = bpy.context.object
    mod = ob.modifiers.new("round", "SUBSURF")
    mod.levels = 3
    bpy.ops.object.modifier_apply(modifier=mod.name)
    p = np.array([v.co[:] for v in ob.data.vertices])
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    # Unit sphere -> tongue: zn runs from the back (-1) to the tip (1).
    zn = p[:, 2]
    ring = np.sqrt(np.maximum(1 - zn ** 2, 1e-6))
    side, height = p[:, 0] / ring, p[:, 1] / ring  # where around the cross-section, -1..1
    roundness = np.sqrt(np.maximum(1 - np.clip((zn - 0.3) / 0.7, 0, 1) ** 2, 0))  # the round tip
    half = (back_half + (tip_half - back_half) * np.clip((zn + 0.6) / 1.6, 0, 1) ** 0.8) * roundness
    x = side * half
    flat = np.sign(height) * np.abs(height) ** 0.45  # a flat top and bottom, rounded edges
    y = top - thick / 2 + flat * thick / 2 * np.maximum(roundness, 0.35)
    fade = np.clip((1 - zn) / 0.25, 0, 1) * (height > 0)
    y -= groove * np.exp(-((x / groove_half) ** 2)) * fade * np.clip(height, 0, 1)
    z = back + (zn + 1) / 2 * (tip - back)
    y -= 0.02 * np.clip((zn - 0.55) / 0.45, 0, 1) ** 2  # the tip dips behind the teeth
    for v, co in zip(ob.data.vertices, np.column_stack([x, y, z])):
        v.co = co
    ob.data.update()
    return ob


def _orient(verts, faces, want):
    """Flip all faces if on the whole their normals point away from want(face centres)."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    return faces if (n * want((a + b + c) / 3)).sum() >= 0 else faces[:, ::-1]


def _object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-7)
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    return ob


def _arrays(ob):
    """Final vertices, smooth vertex normals and triangles of an object, modifiers and
    curve bevels applied."""
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = ob.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    n = len(mesh.vertices)
    verts, normals = np.zeros(n * 3), np.zeros(n * 3)
    mesh.vertices.foreach_get("co", verts)
    mesh.vertices.foreach_get("normal", normals)
    tris = np.zeros(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    evaluated.to_mesh_clear()
    return verts.reshape(-1, 3), normals.reshape(-1, 3), tris.reshape(-1, 3)


def _noise(pos, scale):
    """Cheap fixed speckle in -1..1 from positions."""
    h = np.sin(pos @ np.array([127.1, 311.7, 74.7]) / scale) * 43758.5453
    return 2 * (h - np.floor(h)) - 1


def build(animal: str):
    """All the parts of one set, as lists of per vertex arrays (pos, normal, colour, jaw,
    tint) and triangles, each part tagged with its half of the mouth."""
    parts = []

    def add(ob, colour, jaw, tint, enamel=False):
        v, n, f = _arrays(ob)
        parts.append(dict(pos=v, normal=n, faces=f, colour=colour(v, n), jaw=np.full(len(v), float(jaw)),
                          tint=np.full(len(v), tint), enamel=np.full(len(v), enamel)))

    for lower in (False, True):
        for i, (v, f, base, up, length) in enumerate(teeth(animal, lower)):
            axis = base
            f = _orient(v, f, lambda c, axis=axis, up=up: (c - axis) - np.outer((c - axis) @ up, up))
            ob = _object(f"tooth{int(lower)}{i}", v, f)
            ob.data.shade_smooth()

            def enamel(pos, n, base=base, up=up, length=length):
                t = np.clip((pos - base) @ up / (0.5 * length), 0, 1) ** 0.7
                return np.outer(1 - t, ENAMEL_ROOT) + np.outer(t, ENAMEL)
            add(ob, enamel, lower, 0.0, enamel=True)

        v, f, margin, away = gum(animal, lower)
        f = _orient(v, f, lambda c, away=away: np.tile(away, (len(c), 1)))
        ob = _object(f"gum{int(lower)}", v, f)
        ob.data.shade_smooth()
        gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y

        def gum_colour(pos, n, gum_y=gum_y):
            t = np.clip(np.abs(pos[:, 1] - gum_y) / 0.025, 0, 1)
            return np.outer(1 - t, GUM_MARGIN) + np.outer(t, GUM)
        add(ob, gum_colour, lower, SOFT_TINT)
        if not lower:
            gum_verts = v
            pv, pf = palate(v)
            pf = _orient(pv, pf, lambda c: np.tile([0, -1.0, 0], (len(c), 1)))
            ob = _object("palate", pv, pf)
            ob.data.shade_smooth()
            add(ob, lambda pos, n: np.tile(PALATE, (len(pos), 1)), 0, SOFT_TINT)
            for ob in rugae(animal, gum_verts):
                add(ob, lambda pos, n: np.tile(RUGA, (len(pos), 1)), 0, SOFT_TINT)

    ob = tongue(animal)
    ob.data.shade_smooth()
    tip_z = TONGUE[animal][0]

    def tongue_colour(pos, n):
        t = np.clip((pos[:, 2] - (tip_z - 0.15)) / 0.15, 0, 1)
        colour = np.outer(1 - t, TONGUE_COLOUR[animal]) + np.outer(t, TONGUE_TIP)
        middle = np.exp(-((pos[:, 0] / TONGUE[animal][7]) ** 2)) * (n[:, 1] > 0.5)
        speckle = 1 + PAPILLAE[animal] * _noise(pos, 0.004)
        return colour * ((1 - 0.15 * middle) * speckle)[:, None]
    add(ob, tongue_colour, 1, SOFT_TINT)
    return parts


def bake_light(parts):
    """Ambient occlusion per vertex by ray casts against its own half of the mouth (the
    jaw is open whenever any of it shows), darker toward the throat."""
    k = np.arange(AO_RAYS) + 0.5
    r, a = np.sqrt(k / AO_RAYS), k * np.pi * (3 - np.sqrt(5))  # cosine-weighted, spread evenly
    local = np.column_stack([r * np.cos(a), r * np.sin(a), np.sqrt(1 - r ** 2)])
    for jaw in (0.0, 1.0):
        group = [p for p in parts if p["jaw"][0] == jaw]
        offsets = np.cumsum([0] + [len(p["pos"]) for p in group])
        tree = BVHTree.FromPolygons(
            [tuple(v) for p in group for v in p["pos"]],
            [tuple(int(i) for i in f + o) for p, o in zip(group, offsets) for f in p["faces"]])
        for p in group:
            light = np.empty(len(p["pos"]))
            for i, (pos, n) in enumerate(zip(p["pos"], p["normal"])):
                t1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
                t1 /= np.linalg.norm(t1)
                dirs = local @ np.array([t1, np.cross(n, t1), n])
                start = pos + n * 1e-4
                hits = sum(tree.ray_cast(start, d, AO_REACH)[0] is not None for d in dirs)
                light[i] = AO_FLOOR + (1 - AO_FLOOR) * (1 - hits / AO_RAYS)
            deep = np.clip((ARCH_FRONT - p["pos"][:, 2]) / DEEP_SPAN, 0, 1)
            light *= 1 - DEEP * deep ** 0.8
            p["light"] = np.where(p["enamel"], light * ENAMEL_LIGHT, light)


def save(parts, path: Path):
    keys = ("pos", "normal", "colour", "jaw", "tint", "light")
    merged = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    offsets = np.cumsum([0] + [len(p["pos"]) for p in parts[:-1]])
    faces = np.concatenate([p["faces"] + o for p, o in zip(parts, offsets)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pos=merged["pos"].astype("f4"), normal=merged["normal"].astype("f2"),
                        colour=np.round(np.clip(merged["colour"], 0, 1) * 255).astype("u1"),
                        jaw=merged["jaw"].astype("u1"), tint=merged["tint"].astype("f2"),
                        light=merged["light"].astype("f2"), faces=faces.astype("u2"))
    print(f"{path.name}: {len(merged['pos'])} vertices, {len(faces)} triangles, "
          f"{path.stat().st_size // 1024} KB")
    return merged, faces


def _clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for curve in list(bpy.data.curves):
        bpy.data.curves.remove(curve)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def preview(merged, faces, folder: Path, animal: str):
    """Renders the set as built (its colour times its light) from the front, the side,
    and from below in front with the jaw open, with Blender's workbench."""
    _clear()
    views = {  # name: (jaw angle, camera position, rotation (x, y), scale)
        "front": (0.3, (0, -0.1, 3), (0, 0), 0.9),
        "side": (0.3, (3, -0.15, -0.4), (0, np.pi / 2), 1.2),
        "below": (0.3, (0.9, -1.6, 2.4), (np.radians(30), np.radians(20)), 1.0),
    }
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = 900, 700
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "VERTEX"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.12, 0.12, 0.14)
    hinge = np.array([0.0, 0.1, -1.35])
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    for name, (angle, position, (rx, ry), scale) in views.items():
        c, s = np.cos(angle), np.sin(angle)
        turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pos = merged["pos"].copy()
        jaw = merged["jaw"] > 0.5
        pos[jaw] = (pos[jaw] - hinge) @ turn.T + hinge
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([tuple(v) for v in pos], [], [tuple(int(i) for i in f) for f in faces])
        # Lit roughly like the app: its colour, its light, facing a light near the camera.
        normal = merged["normal"].astype(float).copy()
        normal[jaw] = normal[jaw] @ turn.T
        facing = np.clip(normal @ (np.array(position) / np.linalg.norm(position)), 0, 1)
        shade = merged["colour"] * (merged["light"] * (0.55 + 0.45 * facing))[:, None]
        colours = mesh.color_attributes.new("colour", "FLOAT_COLOR", "POINT")
        colours.data.foreach_set("color", np.column_stack([np.clip(shade, 0, 1), np.ones(len(pos))]).ravel())
        mesh.color_attributes.active_color = colours
        ob = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(ob)
        camera.location, camera.rotation_euler = position, (rx, ry, 0)
        camera.data.ortho_scale = scale
        scene.render.filepath = str(folder / f"{animal}_{name}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(ob)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preview", type=Path, help="also render the sets into this folder")
    args = parser.parse_args(argv)
    for animal in TEETH:
        _clear()
        parts = build(animal)
        bake_light(parts)
        merged, faces = save(parts, OUT / f"{animal}.npz")
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            preview(merged, faces, args.preview, animal)


if __name__ == "__main__":
    main()
