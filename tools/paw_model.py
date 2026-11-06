"""Turns a generated 3D paw (a .glb without a texture, e.g. from Meshy) into the
paw catcam.paw3d draws on your hands:

    uv run --with trimesh --with fast-simplification python tools/paw_model.py \\
        ~/Downloads/catpaws.glb cat --pair
    uv run --with trimesh --with fast-simplification python tools/paw_model.py \\
        ~/Downloads/dogpaw.glb dog

It writes assets/paws/<animal>.npz and never changes the .glb. --pair: the model
has a left and a right paw side by side along x, mirror twins, so only one is
kept (the other hand gets it mirrored, like a single paw would).

What it works out, so nothing has to be placed by hand:
- the pads (and claws): where the mesh is coarse among the fine tufts of fur, found
  at full detail before the mesh is thinned out to a size the app draws quickly (an
  inside surface, which the dog has, is dropped first);
- the paw's frame from the pads and the arm: y up the paw toward the toes, z out of
  the back of the paw (away from the pads), x toward the thumb side; in paw widths,
  with the origin in the middle of the palm (the main pad). A paw bent at the wrist
  is turned upright on its arm, as your hand steers the paw, not your forearm;
- the four toes, from the four toe pads: per vertex, how much it bends with each
  toe, and the knuckles they bend about;
- a piece of the forearm that catcam.paw3d repeats, mirrored back and forth, so
  the arm runs off-screen whatever the model's length;
- light: skin sunk below its surroundings (between tufts, between toes) darker.
"""

import argparse
from pathlib import Path

import fast_simplification
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "paws"

TARGET_FACES = 60000  # the thinned out paw, arm piece included
INSIDE_SHARE = 0.3  # a model with more faces than this facing in has an inside surface too
# Finding the pads at full detail. The modeller spends its triangles on the fur's tufts,
# so the smooth pads and claws are where the mesh is coarse: its edge lengths, averaged
# over SPARSE_RADIUS (model lengths, its longest side), as percentiles from not a pad
# at [0] to surely one at [1]; the mask is then evened out over MASK_RADIUS.
SPARSE_RADIUS, SPARSE_BAND = 0.006, (80, 97)
MASK_RADIUS = 0.008
PAD_SURE = 0.6  # pad mask above which skin is taken as pad when finding the pads' shapes
SOLE_FACING = 0.4  # pads face the sole's way at least this much (cosine); the fur over toe tips less
FACING_EDGE = 0.15  # pad fades out from facing the sole's way this much (cosine) to facing as much back
SOLE_EDGE = 0.15  # share of the paw's thickness over which skin turns from the sole's side to the back's
ARM_PART = 0.45  # the forearm's direction is taken from this share of the model, at the arm's end
PAW_PART = 0.4  # the pads are within this share of the model, at the toes' end
CAP_FACING = 0.8  # a patch facing along the arm more than this (cosine) is a cut-off arm's cap
# In paw widths: where the paw's own arm ends (below the palm) and the length of the
# piece above that the arm is made of beyond it.
ARM_CUT, ARM_PIECE = 0.8, 0.7
# The paw turns upright on the arm about the wrist, this far below the main pad, over a
# band this long (both paw widths).
WRIST_DROP, WRIST_BAND = 0.2, 0.5
UPRIGHT_PASSES = 3  # times the pads are found again on the paw turned upright, to turn it more truly
TOE_CORE = 0.3  # share of the gap to the next toe that bends with this toe alone
KNUCKLE_BAND = 0.6  # toe lengths over which a toe goes from not bending to bending fully
HINGE = 0.5  # where the knuckles sit, as a share of the paw's thickness from the sole to the back
# Light: how much darker skin gets per paw width it is sunk below its surroundings,
# finely (between tufts) and broadly (between toes, the arm's crease), and the darkest.
FINE_SINK, BROAD_SINK, DARKEST = 30.0, 6.0, 0.35
FINE_SMOOTHING, BROAD_SMOOTHING = 4, 40


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)  # the UV seams split it into pieces
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    # Faces wound so their normals point out (a negative volume means they point in).
    volume = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()
    return v, (f if volume > 0 else f[:, ::-1])


def submesh(v: np.ndarray, f: np.ndarray, keep_faces: np.ndarray):
    """The faces kept and only the vertices they use, renumbered; also the old vertex numbers."""
    f = f[keep_faces]
    used = np.unique(f)
    index = np.full(len(v), -1)
    index[used] = np.arange(len(used))
    return v[used], index[f], used


def adjacency(n: int, f: np.ndarray) -> sp.csr_matrix:
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(float).tocsr()


def normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    face = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], face)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def smoothed(v: np.ndarray, f: np.ndarray, times: int) -> np.ndarray:
    """v with the surface evened out (each vertex halfway to its neighbours' mean, times times)."""
    a = adjacency(len(v), f)
    mean = sp.diags(1 / np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1)) @ a
    s = v.copy()
    for _ in range(times):
        s = 0.5 * s + 0.5 * (mean @ s)
    return s


def sunk(v: np.ndarray, f: np.ndarray, times: int) -> np.ndarray:
    """How far each vertex stands out of (+) or is sunk into (-) the surface smoothed times times."""
    s = smoothed(v, f, times)
    return np.einsum("ij,ij->i", v - s, normals(s, f))


def around(tree: cKDTree, values: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """Mean of values over the vertices within radius of each point (at most 128 of them)."""
    nearest = tree.query(points, k=128, distance_upper_bound=radius)[1]
    valid = nearest < len(values)
    picked = values[np.where(valid, nearest, 0)]
    shape = (-1,) + (1,) * (values.ndim - 1)
    total = (picked * valid.reshape(valid.shape + (1,) * (values.ndim - 1))).sum(axis=1)
    return total / np.maximum(valid.sum(axis=1), 1).reshape(shape)


def ramp(x: np.ndarray, band) -> np.ndarray:
    lo, hi = np.percentile(x, band)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def pad_mask(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """0..1 per vertex: how surely it is pad (or claw) rather than fur, from how coarse
    the mesh is around it."""
    size = np.ptp(v, axis=0).max()
    tree = cKDTree(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    ends = e.ravel()
    count = np.maximum(np.bincount(ends, minlength=len(v)), 1)
    mean = np.bincount(ends, np.repeat(length, 2), len(v)) / count
    score = ramp(around(tree, mean, v, SPARSE_RADIUS * size), SPARSE_BAND)
    return around(tree, score, v, MASK_RADIUS * size)


def components(f: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    """Vertices of each connected patch of the kept vertices."""
    _, label = connected_components(adjacency(len(keep), f[keep[f].all(axis=1)]), directed=False)
    label = np.where(keep, label, -1)
    order = np.argsort(label)
    starts = np.searchsorted(label[order], np.arange(label.max() + 1))
    return [g for g in np.split(order, starts[1:]) if len(g) > 2 and keep[g[0]]]


def axis_of(points: np.ndarray) -> np.ndarray:
    return np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]


def frame(v: np.ndarray, f: np.ndarray, pads: np.ndarray, toward=None):
    """(origin, rows x, y, z) of the paw's frame in model units: y along the forearm toward
    the toes, z out of the back of the paw, x = y cross z. pads: which vertices are pad;
    the biggest patch of them that doesn't face along the arm (a cap where the arm was
    cut off is coarse too) is the main pad, at the toes' end on the sole. toward: a point
    the thumb side faces (the other paw of a pair), which makes x point there by
    mirroring if need be."""
    centre = v.mean(axis=0)
    along = axis_of(v)
    area, n = vertex_areas(v, f), normals(v, f)

    def facing_along(g):
        mean = np.average(n[g], axis=0, weights=area[g])
        return abs(mean @ along) / (np.linalg.norm(mean) + 1e-12)

    patches = [g for g in components(f, pads) if facing_along(g) < CAP_FACING]
    main = max(patches, key=lambda g: area[g].sum())
    sole = np.average(v[main], axis=0, weights=area[main]) - centre
    if sole @ along < 0:
        along = -along
    t = (v - centre) @ along
    arm = v[t < np.percentile(t, 100 * ARM_PART)]
    y = axis_of(arm)
    y = y if y @ along > 0 else -y
    z = -(sole - (sole @ y) * y)
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    mirrored = toward is not None and (np.asarray(toward) - centre) @ x < 0
    return centre, np.stack([-x if mirrored else x, y, z]), mirrored


def middles(p: np.ndarray) -> np.ndarray:
    """(n, 3) for each vertex of a framed paw p, the middle (x, z) of the paw at its height."""
    bins = np.linspace(p[:, 1].min(), p[:, 1].max(), 41)
    which = np.clip(np.digitize(p[:, 1], bins) - 1, 0, 39)
    middle = np.array([np.median(p[which == k], axis=0) if (which == k).any() else np.zeros(3)
                       for k in range(40)])[which]
    middle[:, 1] = p[:, 1]
    return middle


def on_sole(p: np.ndarray) -> np.ndarray:
    """0..1 per vertex of a framed paw p: whether it is on the sole's side (z below the
    middle of the paw at its height). Smooth fur on the back of the toes is no pad."""
    return np.clip((middles(p)[:, 2] - p[:, 2]) / (SOLE_EDGE * np.ptp(p[:, 2])) + 0.5, 0, 1)


def outward(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Vertex normals of a framed paw p turned to point away from its middle: a model with
    an inside surface too (the dog's) has that one's normals pointing in."""
    n = normals(p, f)
    return n * np.where(np.einsum("ij,ij->i", n, p - middles(p)) < 0, -1.0, 1.0)[:, None]


def vertex_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    face = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 6
    area = np.zeros(len(v))
    for k in range(3):
        np.add.at(area, f[:, k], face)
    return area


def palm_and_toes(p: np.ndarray, f: np.ndarray, sure: np.ndarray):
    """The main pad's vertices and the four toe pads' (from the thumb side, x falling) of
    a framed paw p, among the sure pad facing the way the sole does: the biggest patch
    is the main pad, the rest above it falls into four clusters, seeded spread across
    the paw. Clustering copes with toe pads the mask splits in two or runs together."""
    paw = p[:, 1] > p[:, 1].max() - PAW_PART * np.ptp(p[:, 1])
    facing = -outward(p, f)[:, 2] > SOLE_FACING
    area = vertex_areas(p, f)
    main = max(components(f, sure & facing & paw), key=lambda g: area[g].sum())
    rest = sure & facing & paw & (p[:, 1] > np.median(p[main, 1]))
    rest[main] = False
    index = np.flatnonzero(rest)
    xy, weight = p[index, :2], area[index]
    centres = np.array([(np.quantile(xy[:, 0], q), xy[:, 1].mean()) for q in (7 / 8, 5 / 8, 3 / 8, 1 / 8)])
    for _ in range(30):
        label = np.argmin(np.linalg.norm(xy[:, None] - centres[None], axis=2), axis=1)
        centres = np.array([np.average(xy[label == k], axis=0, weights=weight[label == k])
                            if (label == k).any() else centres[k] for k in range(4)])
    toes = [index[label == k] for k in range(4)]
    if min(len(g) for g in toes) == 0:
        raise SystemExit("could not tell the four toe pads apart: look at the pad mask")
    return main, sorted(toes, key=lambda g: -np.median(p[g, 0]))


def toe_weights(p: np.ndarray, toe_x: list, knuckles: np.ndarray, toe_length: float) -> np.ndarray:
    """(n, 4) how much each vertex bends with each toe: by which toe it is across the
    paw (toes from the thumb side, x falling) and how far up that toe it is (from its
    knuckle, so the main pad stays put, to KNUCKLE_BAND toe lengths above it)."""
    up = np.clip((p[:, 1:2] - knuckles[None]) / (KNUCKLE_BAND * toe_length), 0, 1)
    up = up * up * (3 - 2 * up)
    centres = np.array(toe_x)
    edges, values = [], []
    for i, c in enumerate(centres):  # each toe's core, then a blend over the gap to the next
        gap_before = centres[i - 1] - c if i else np.inf
        gap_after = c - centres[i + 1] if i < 3 else np.inf
        edges += [c + min(TOE_CORE * gap_before, 1e3), c - min(TOE_CORE * gap_after, 1e3)]
        values += [i, i]
    order = np.argsort(edges)
    position = np.interp(p[:, 0], np.array(edges)[order], np.array(values, float)[order])
    weights = np.clip(1 - np.abs(position[:, None] - np.arange(4)[None]), 0, 1)
    return weights * up


def upright(p: np.ndarray, f: np.ndarray, main: np.ndarray, toes: list, width: float):
    """How to turn the paw of a framed paw p (along its forearm) at the wrist so that it
    stands upright on the arm: its toes straight up from the main pad, its sole facing
    -z. Your hand, not your forearm, steers the paw; a paw bent at the wrist (the dog's,
    its pads facing along the arm) would otherwise show its sole edge-on. Returns
    (pivot, rotation vector, wrist height) for bent()."""
    area, n = vertex_areas(p, f), normals(p, f)
    z = -np.average(n[main], axis=0, weights=area[main])
    z /= np.linalg.norm(z)
    up = p[np.concatenate(toes)].mean(axis=0) - p[main].mean(axis=0)
    y = up - (up @ z) * z
    y /= np.linalg.norm(y)
    turn = Rotation.from_matrix(np.stack([np.cross(y, z), y, z])).as_rotvec()
    wrist = np.percentile(p[main, 1], 10) - WRIST_DROP * width
    near = np.abs(p[:, 1] - wrist) < 0.1 * width
    pivot = np.array([np.median(p[near, 0]), wrist, np.median(p[near, 2])])
    return pivot, turn, WRIST_BAND * width


def bent(p: np.ndarray, bend) -> np.ndarray:
    """p turned at the wrist as upright() says: fully above it, not at all below, smoothly
    in between."""
    pivot, turn, band = bend
    w = np.clip((p[:, 1] - pivot[1]) / (2 * band) + 0.5, 0, 1)
    w = w * w * (3 - 2 * w)
    return pivot + Rotation.from_rotvec(w[:, None] * turn).apply(p - pivot)


def straighten(p: np.ndarray, below: float) -> np.ndarray:
    """p with the arm below y = below sheared upright, so a piece of it repeated
    straight down lines up with itself."""
    arm = p[:, 1] < below
    ys = p[arm, 1]
    slope = np.polyfit(ys, p[arm][:, [0, 2]], 1)[0]  # x and z per y
    out = p.copy()
    out[arm, 0] -= slope[0] * (ys - below)
    out[arm, 2] -= slope[1] * (ys - below)
    return out


def light(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """0..1 per vertex of a paw in paw widths, darker where it is sunk below its surroundings."""
    fine = np.minimum(sunk(p, f, FINE_SMOOTHING), 0)
    broad = np.minimum(sunk(p, f, BROAD_SMOOTHING), 0)
    return np.clip(1 + FINE_SINK * fine + BROAD_SINK * broad, DARKEST, 1)


def without_inside(v: np.ndarray, f: np.ndarray):
    """The mesh without its inside surface, if it has one (the dog's is a shell with
    thickness): the faces facing in, toward the middle of the arm, when there are a lot
    of them (a single surface has only a few, under tufts of fur)."""
    centre = v.mean(axis=0)
    along = axis_of(v)
    corners = v[f]
    middle = corners.mean(axis=1)
    t = (middle - centre) @ along
    bins = np.linspace(t.min(), t.max(), 41)
    which = np.clip(np.digitize(t, bins) - 1, 0, 39)
    axis = np.array([np.median(middle[which == k], axis=0) if (which == k).any() else centre
                     for k in range(40)])[which]
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.einsum("ij,ij->i", face_normal, middle - axis) < 0
    if inward.mean() < INSIDE_SHARE:
        return v, f
    print(f"dropping the inside surface ({inward.mean():.0%} of the faces)")
    kept_v, kept_f, _ = submesh(v, f, ~inward)
    return kept_v, kept_f


def one_paw(v: np.ndarray, f: np.ndarray, pair: bool):
    """The paw to work on (without any inside surface) and, for a pair, a point its thumb
    side faces: the other paw. A pair keeps the paw at -x."""
    toward = None
    if pair:
        # The gap between them: the emptiest place in the middle half along x. Each
        # connected piece goes to the side its middle is on, so neither paw is cut.
        lo, hi = np.percentile(v[:, 0], [25, 75])
        counts, edges = np.histogram(v[:, 0], bins=50, range=(lo, hi))
        gap = (edges[np.argmin(counts)] + edges[np.argmin(counts) + 1]) / 2
        _, piece = connected_components(adjacency(len(v), f), directed=False)
        middle_x = np.bincount(piece, v[:, 0]) / np.bincount(piece)
        left = middle_x[piece] < gap
        toward = v[~left].mean(axis=0)
        v, f, _ = submesh(v, f, left[f[:, 0]])
    return *without_inside(v, f), toward


def analyse(v: np.ndarray, f: np.ndarray, toward):
    """At full detail, where the pads have plenty of vertices: the paw framed along its
    forearm (p, and its faces, turned back if the frame mirrors it), its pad mask, the
    main pad's and four toe pads' vertices, and the frame (origin, axes)."""
    pads = pad_mask(v, f)
    # Framed once to tell the sole's side from the back's, then again from the sole's pads
    # only, which are all that is kept of the pads (with their soft edges).
    origin, axes, _ = frame(v, f, pads > PAD_SURE, toward)
    kept = (pads > PAD_SURE) & (on_sole((v - origin) @ axes.T) > 0.5)
    origin, axes, mirrored = frame(v, f, kept, toward)
    p = (v - origin) @ axes.T
    if mirrored:  # mirrored coordinates turn the faces inside out; turn them back
        f = f[:, ::-1]
    # Pads sit on the paw (not up the arm), on the sole's side, and don't face the back
    # (as the fur over curled toe tips, on the sole's side by then, does).
    tree, radius = cKDTree(v), 2 * MASK_RADIUS * np.ptp(v, axis=0).max()
    paw = p[:, 1] > p[:, 1].max() - PAW_PART * np.ptp(p[:, 1])
    facing = -around(tree, outward(p, f), v, radius)[:, 2]
    kept = (pads > PAD_SURE) & (on_sole(p) > 0.5) & paw & (facing > 0)
    pads = pads * np.clip(2 * around(tree, kept.astype(float), v, radius), 0, 1)
    pads *= np.clip((facing + FACING_EDGE) / (2 * FACING_EDGE), 0, 1)
    main, toes = palm_and_toes(p, f, pads > PAD_SURE)
    return p, f, pads, main, toes, origin, axes


def build(path: Path, pair: bool) -> dict:
    v, f, toward = one_paw(*load(path), pair)
    print(f"{len(v)} vertices, {len(f)} faces: finding the pads")
    p, f, pads, main, toes, origin, axes = analyse(v, f, toward)
    palm = np.median(p[main], axis=0)
    level = np.abs(p[:, 1] - palm[1]) < 0.1 * np.ptp(p[:, 1])
    width = np.percentile(p[level, 0], 98) - np.percentile(p[level, 0], 2)
    # The pads are told apart better with the sole facing the camera: found again on the
    # paw turned upright, they turn it more truly.
    bend = upright(p, f, main, toes, width)
    for _ in range(UPRIGHT_PASSES):
        main, toes = palm_and_toes(bent(p, bend), f, pads > PAD_SURE)
        bend = upright(p, f, main, toes, width)
    p = bent(p, bend)
    main, toes = palm_and_toes(p, f, pads > PAD_SURE)
    palm = np.median(p[main], axis=0)
    level = np.abs(p[:, 1] - palm[1]) < 0.1 * np.ptp(p[:, 1])
    centre = np.array([palm[0], palm[1], np.median(p[level, 2])])
    p = (p - centre) / width
    toe_x = [float(np.median(p[g, 0])) for g in toes]
    # Each toe bends about its knuckle, halfway from the top of the main pad to its own pad.
    top = np.percentile(p[main, 1], 90)
    knuckles = np.array([(top + np.percentile(p[g, 1], 10)) / 2 for g in toes])
    tip = float(np.percentile(p[:, 1], 99.5))
    hinges = np.array([np.percentile(p[(np.abs(p[:, 1] - k) < 0.05) & (np.abs(p[:, 0] - x) < 0.15), 2],
                                     [5, 95]) @ (1 - HINGE, HINGE) for k, x in zip(knuckles, toe_x)])
    toe_length = float(tip - knuckles.mean())
    print(f"width {width:.3f} model units; toes at x {np.round(toe_x, 2)}, "
          f"knuckles at y {knuckles.round(2)}, tips at {tip:.2f}")

    print("thinning out")
    thin_v, thin_f = fast_simplification.simplify(v, f, target_reduction=1 - TARGET_FACES / len(f))
    thin_v, thin_f = np.asarray(thin_v, float), np.asarray(thin_f, np.int64)
    pads = pads[cKDTree(v).query(thin_v)[1]]
    p = (bent((thin_v - origin) @ axes.T, bend) - centre) / width
    p = straighten(p, -ARM_CUT)
    weights = toe_weights(p, toe_x, knuckles, toe_length)
    shade = light(p, thin_f)
    n = normals(p, thin_f)
    centre_y = p[thin_f].mean(axis=1)[:, 1]
    paw_v, paw_f, paw_used = submesh(p, thin_f, centre_y >= -ARM_CUT)
    arm_v, arm_f, arm_used = submesh(p, thin_f, (centre_y < -ARM_CUT) & (centre_y >= -ARM_CUT - ARM_PIECE))
    if p[:, 1].min() > -ARM_CUT - ARM_PIECE:
        raise SystemExit("the model's arm is too short for its piece: lower ARM_CUT or ARM_PIECE")
    return {
        "pos": paw_v.astype("f4"), "normal": n[paw_used].astype("f2"), "faces": paw_f.astype("i4"),
        "toes": weights[paw_used].astype("f2"), "pad": pads[paw_used].astype("f2"),
        "light": shade[paw_used].astype("f2"),
        "arm_pos": arm_v.astype("f4"), "arm_normal": n[arm_used].astype("f2"),
        "arm_faces": arm_f.astype("i4"),
        "arm_pad": pads[arm_used].astype("f2"), "arm_light": shade[arm_used].astype("f2"),
        "arm_piece": np.float32(ARM_PIECE), "arm_cut": np.float32(ARM_CUT),
        "knuckles": np.column_stack([knuckles, hinges]).astype("f4"), "toe_length": np.float32(toe_length),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path, help="the .glb (only read)")
    parser.add_argument("animal", help="what it is: assets/paws/<animal>.npz, e.g. cat or dog")
    parser.add_argument("--pair", action="store_true", help="the model has two mirror-twin paws along x")
    args = parser.parse_args()
    data = build(args.model.expanduser(), args.pair)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.animal}.npz"
    np.savez_compressed(out, **data)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): "
          f"{len(data['pos'])} + {len(data['arm_pos'])} vertices")


if __name__ == "__main__":
    main()
