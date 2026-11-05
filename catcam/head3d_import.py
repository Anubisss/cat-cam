"""Turns a generated 3D head (.glb, e.g. from TRELLIS.2) into a cat's head3d
assets: the mesh, its texture with the cat's real eyes painted in from the
portrait, and where every vertex and face landmark sits in the portrait, which
is what lets the head take on the 2D cat's expressions.

    uv run --with trimesh python -m catcam.head3d_import <head.glb> <Name> [--own-mouth] [--animal dog]

--own-mouth keeps the head's own mouth: no lips or mouth cavity are added. The
import finds the head's own lower jaw instead and cuts it free along the slit
between its lips, so it can swing open there, with teeth and a tongue inside
(see catcam.mouth3d). --animal dog gives it a dog's teeth and tongue instead of
a cat's.
"""

import argparse

import cv2
import moderngl
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from catcam import cats
from catcam.cat_face import EYES
from catcam.head3d import files
from catcam.mouth3d import ANIMALS
from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP

YAW_SEARCH = np.arange(-15, 16, 3)  # degrees around face-on tried when fitting the camera
PITCH_SEARCH = np.arange(-12, 13, 3)
DEPTH_TOL = 0.01  # model units: a point this close to the front surface counts as visible
EYE_GROW = 0.35  # painted eye area reaches this share of the eye's width past its contour
EYE_FEATHER = 0.15  # soft edge, eye widths
EYE_SEARCH = 0.4  # how far (eye widths) to look for the model's eye around the photo's
NOSE = [1, 2, 4, 5, 98, 327, 195, 197, 48, 278]  # landmarks around the nose
GUTTER = 8  # texels past the uv charts' edges that get painted too (filtering reads them)
SOLID = 250  # portrait alpha from which a pixel counts as the cat when matching colours
# Specks: the texture baker leaves near-black blobs (thin ears especially). A blob is
# darker than SPECK_DARK (grey 0..255), SPECK_GAP darker than its surroundings (the
# median over SPECK_WINDOW texels) and under SPECK_AREA texels, so dark fur stays.
SPECK_DARK, SPECK_GAP, SPECK_WINDOW, SPECK_AREA = 15, 60, 41, 600
# A head's own jaw (--own-mouth) grows over its surface from the lower lip and chin
# (JAW_SEEDS), the rest of the face from the upper lip and nose (FACE_SEEDS), and it
# is cut free where they meet. Going into a fold costs as much as JAW_FOLD face widths
# of surface per radian it bends, so they meet in the slit between the lips. Bends
# are measured on the surface smoothed over JAW_SMOOTH face widths, so fur and pores
# don't count as folds, and the cut is smoothed as much.
JAW_SEEDS = [17, 18, 83, 84, 152, 175, 199, 200, 313, 314]
FACE_SEEDS = [0, 1, 2, 37, 39, 40, 98, 164, 185, 267, 269, 270, 327, 409]
# Each side also starts right by the lips: this far (face widths) below and above the
# line where they meet, so the two meet in the slit and not in a deeper fold nearby
# (a cat's chin, a grin's lower lip pad).
LIP_SEED_GAP = 0.04
# The jaw is cut free only along the lips: within CUT_REACH face widths of the line
# where they meet, fading out over CUT_FADE. Further along its edge (down the sides
# of the chin, under it) the skin stretches instead, the jaw's pull fading over
# JAW_SOFT face widths across the edge.
CUT_REACH, CUT_FADE, JAW_SOFT = 0.06, 0.04, 0.03
# At the corners of the mouth the lips stay together, so it opens in a lens, not a slot,
# and the corners keep following your face (a smile): from CORNER_START of the way out
# from its middle, the jaw's pull on the lips fades to nothing at the corners and past
# them, coming back over CORNER_DEPTH face widths below the line where the lips meet.
CORNER_START, CORNER_DEPTH = 0.4, 0.12
LIPS_APART = 0.3  # share of the jaw's pull by which a cut's two sides must differ to count as the lips' cut
JAW_FOLD = 0.1
JAW_SMOOTH = 0.01
SLIT_FRONT = 0.3  # face widths behind the cut's front: how deep the slit seen between the lips goes
JAW_REACH = (0.3, 0.5)  # face widths from the lower lip: within the first it all moves, past the second none

DEPTH_VS = """
#version 330
uniform mat4 mvp;
in vec3 pos; out float z;
void main() { vec4 q = mvp * vec4(pos, 1.0); gl_Position = q; z = -q.z * 3.0; }
"""
DEPTH_FS = """
#version 330
in float z; out vec4 color;
void main() { color = vec4(z, 1.0, 0.0, 1.0); }
"""
UV_VS = """
#version 330
in vec2 uv; in vec3 pos;
out vec3 p;
void main() { gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0); p = pos; }
"""
UV_FS = """
#version 330
in vec3 p; out vec4 out_p;
void main() { out_p = vec4(p, 1.0); }
"""


def rotation(yaw: float, pitch: float) -> np.ndarray:
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


class FrontCamera:
    """Orthographic camera looking at the model like the portrait does:
    photo px = centre + scale * (rot @ p).xy, y flipped."""

    def __init__(self, rot, scale, centre):
        self.rot, self.scale, self.centre = rot, scale, np.asarray(centre, float)

    def project(self, points: np.ndarray):
        q = points @ self.rot.T
        return np.column_stack([self.centre[0] + self.scale * q[:, 0],
                                self.centre[1] - self.scale * q[:, 1]]), q[:, 2]

    def unproject(self, px: np.ndarray, z: np.ndarray) -> np.ndarray:
        q = np.column_stack([(px[:, 0] - self.centre[0]) / self.scale,
                             -(px[:, 1] - self.centre[1]) / self.scale, z])
        return q @ self.rot


class Raster:
    def __init__(self, vertices, faces, uv):
        self.ctx = moderngl.create_standalone_context()
        self.vertices, self.faces, self.uv = vertices, faces, uv
        self.depth_prog = self.ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        self.depth_vao = self.ctx.vertex_array(
            self.depth_prog, [(self.ctx.buffer(vertices.astype("f4")), "3f", "pos")],
            self.ctx.buffer(faces.astype("i4")))

    def depth(self, cam: FrontCamera, size) -> np.ndarray:
        """Camera-space depth (bigger = nearer) per photo pixel; -inf off the model."""
        w, h = size
        mvp = np.eye(4)
        mvp[:3, :3] = cam.rot
        ortho = np.diag([2 * cam.scale / w, 2 * cam.scale / h, -1 / 3, 1.0])
        ortho[0, 3] = 2 * cam.centre[0] / w - 1
        ortho[1, 3] = 1 - 2 * cam.centre[1] / h
        self.depth_prog["mvp"].write((ortho @ mvp).astype("f4").T.tobytes())
        fbo = self.ctx.framebuffer([self.ctx.renderbuffer((w, h), 4, dtype="f4")],
                                   self.ctx.depth_renderbuffer((w, h)))
        fbo.use()
        fbo.clear(-1e9, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.depth_vao.render()
        out = np.frombuffer(fbo.read(components=4, dtype="f4"), "f4").reshape(h, w, 4)[::-1, :, 0].copy()
        fbo.release()
        return out

    def texels(self, size: int):
        """3D position of every texel (rows top-down) and whether the mesh covers it."""
        prog = self.ctx.program(vertex_shader=UV_VS, fragment_shader=UV_FS)
        data = np.hstack([self.uv, self.vertices]).astype("f4")
        vao = self.ctx.vertex_array(prog, [(self.ctx.buffer(data), "2f 3f", "uv", "pos")],
                                    self.ctx.buffer(self.faces.astype("i4")))
        tex = self.ctx.texture((size, size), 4, dtype="f4")
        fbo = self.ctx.framebuffer([tex])
        fbo.use()
        fbo.clear(0, 0, 0, 0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        vao.render()
        p = np.frombuffer(tex.read(), "f4").reshape(size, size, 4)[::-1]
        return p[..., :3], p[..., 3] > 0


def fit_camera(raster: Raster, alpha: np.ndarray) -> tuple[float, FrontCamera]:
    """The face-on camera whose view of the model best overlaps the portrait's outline."""
    h, w = alpha.shape
    mask = alpha > 128
    ys, xs = np.nonzero(mask)
    p_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    best = (-1.0, None)
    for yaw in YAW_SEARCH:
        for pitch in PITCH_SEARCH:
            rot = rotation(yaw, pitch)
            c0, s0 = np.array([w / 2, h / 2]), w / 2.4
            ys2, xs2 = np.nonzero(raster.depth(FrontCamera(rot, s0, c0), (w, h)) > -1e8)
            r_box = np.array([xs2.min(), ys2.min(), xs2.max(), ys2.max()], float)
            k = (p_box[3] - p_box[1]) / (r_box[3] - r_box[1])  # match heights, then centres
            centre = (p_box[:2] + p_box[2:]) / 2 - k * ((r_box[:2] + r_box[2:]) / 2 - c0)
            cam = FrontCamera(rot, s0 * k, centre)
            sil = raster.depth(cam, (w, h)) > -1e8
            iou = (sil & mask).sum() / (sil | mask).sum()
            if iou > best[0]:
                best = (iou, cam)
    return best


def sample(image: np.ndarray, px: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear colours of image at px (n, 2); remap wants 2D maps under 32k a side."""
    n = len(px)
    pad = -n % width
    maps = [np.pad(c.astype(np.float32), (0, pad)).reshape(-1, width) for c in px.T]
    channels = image.shape[2] if image.ndim == 3 else 1
    return cv2.remap(image, *maps, cv2.INTER_LINEAR).reshape(-1, channels)[:n]


def model_view(raster, cam, depth, texture, size):
    """The model as the portrait's camera sees it (grey), plus each covered texel's
    index, portrait px and whether that camera sees it."""
    w, h = size
    pos, covered = raster.texels(texture.shape[0])
    idx = np.nonzero(covered.ravel())[0]
    px, z = cam.project(pos.reshape(-1, 3)[idx])
    ui, vi = np.clip(px[:, 0].round().astype(int), 0, w - 1), np.clip(px[:, 1].round().astype(int), 0, h - 1)
    front = z >= depth[vi, ui] - DEPTH_TOL
    view = np.zeros((h, w, 3), np.float32)
    count = np.zeros((h, w), np.float32)
    np.add.at(view, (vi[front], ui[front]), texture.reshape(-1, 3)[idx[front]])
    np.add.at(count, (vi[front], ui[front]), 1)
    view = cv2.cvtColor((view / np.maximum(count, 1)[..., None]).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return view, idx, px, front


def feature_shift(view, photo, around: np.ndarray) -> np.ndarray:
    """Where the feature around these portrait px is in the model's view, relative
    to the portrait (px): the model's = the portrait's + shift."""
    width = np.ptp(around[:, 0])
    x0, y0 = (around.min(0) - EYE_GROW * width).astype(int)
    x1, y1 = (around.max(0) + EYE_GROW * width).astype(int)
    r = int(EYE_SEARCH * width)
    score = cv2.matchTemplate(view[y0 - r:y1 + r, x0 - r:x1 + r].astype(np.float32),
                              photo[y0:y1, x0:x1].astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, _, _, (bx, by) = cv2.minMaxLoc(score)
    return np.array([bx - r, by - r], float)


def align_features(raster, cam, texture, sprite, points) -> FrontCamera:
    """The camera moved so the model's eyes and nose land on the portrait's: the
    outline fit alone can be tens of px off at the features."""
    h, w = sprite.shape[:2]
    view, *_ = model_view(raster, cam, raster.depth(cam, (w, h)), texture, (w, h))
    photo = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2GRAY)
    features = [points[upper + lower, :2] for upper, lower in EYES] + [points[NOSE, :2]]
    shift = np.mean([feature_shift(view, photo, f) for f in features], axis=0)
    print(f"  features were {shift.round()} px off; moved the camera")
    return FrontCamera(cam.rot, cam.scale, cam.centre - shift)


def eye_paint(raster, cam, depth, texture, sprite, points):
    """Where the portrait's eyes go in the texture: (texel index, portrait px, weight).
    Each eye is matched to the model's own eye first, so the two line up."""
    h, w = sprite.shape[:2]
    view, idx, px, front = model_view(raster, cam, depth, texture, (w, h))
    photo = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2GRAY)
    weight = np.zeros(len(idx), np.float32)
    source = np.zeros((len(idx), 2), np.float32)
    for upper, lower in EYES:
        eye = points[upper + lower, :2]
        width = np.ptp(eye[:, 0])
        shift = feature_shift(view, photo, eye)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(eye.astype(np.float32)).astype(np.int32), 255)
        grow = int(EYE_GROW * width)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        mask = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), EYE_FEATHER * width)
        a = sample(mask, px - shift)[:, 0] * front
        mine = a > weight
        weight[mine], source[mine] = a[mine], (px - shift)[mine]
    idx, source, weight = into_gutters(idx, texture.shape[0], source, weight)
    keep = weight > 1e-3
    return idx[keep].astype("i4"), source[keep], np.clip(weight[keep], 0, 1)


def into_gutters(idx, size, source, weight):
    """Texels just outside the uv charts take the paint of the nearest one inside:
    bilinear filtering and mipmaps read them at a chart's edge, and left as they were
    (the model's own eye) they show as dark cracks over the painted eye."""
    covered = np.zeros((size, size), bool)
    covered.flat[idx] = True
    dist, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
    gutter = np.nonzero((dist.ravel() > 0) & (dist.ravel() <= GUTTER))[0]
    lookup = np.full(size * size, -1)
    lookup[idx] = np.arange(len(idx))
    nearest = lookup[(iy * size + ix).ravel()[gutter]]
    return (np.concatenate([idx, gutter]), np.concatenate([source, source[nearest]]),
            np.concatenate([weight, weight[nearest]]))


def despeckle(texture: np.ndarray) -> np.ndarray:
    """The texture (BGR uint8) with the baker's black specks filled in from around them."""
    grey = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    around = cv2.medianBlur(grey, SPECK_WINDOW)
    dark = ((grey < SPECK_DARK) & (around.astype(int) - grey > SPECK_GAP)).astype(np.uint8)
    dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))  # their soft edges too
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < SPECK_AREA
    return cv2.inpaint(texture, small[labels].astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


def match_colours(raster, cam, depth, texture, sprite) -> np.ndarray:
    """The texture with its colours matched to the portrait's: the model's come out
    duller and darker. Each Lab channel's distribution over the face seen from the
    front is mapped onto the portrait's there, and that mapping applied everywhere."""
    h, w = sprite.shape[:2]
    _, idx, px, front = model_view(raster, cam, depth, texture, (w, h))
    alpha = sample(sprite[..., 3:].astype(np.float32), px)[:, 0]
    seen = front & (alpha >= SOLID)
    lab = lambda bgr: cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.float32) / 255,  # noqa: E731
                                   cv2.COLOR_BGR2LAB).reshape(-1, 3)
    model = lab(texture.reshape(-1, 3))
    ours, theirs = model[idx[seen]], lab(sample(sprite[..., :3].astype(np.float32), px[seen]))
    q = np.linspace(0, 1, 257)
    for c in range(3):
        model[:, c] = np.interp(model[:, c], np.quantile(ours[:, c], q), np.quantile(theirs[:, c], q))
    bgr = cv2.cvtColor(model.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(texture.shape)
    return np.clip(bgr * 255, 0, 255)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def find_jaw(vertices, faces, landmarks, by_lips, face_model):
    """The head's own lower jaw (see JAW_SEEDS), per vertex: how far it lies on the
    jaw's side of the cut (surface, face widths; negative: the face's side) and how
    much it would move with the jaw, 0..1. landmarks: the face's on the model;
    by_lips: points on the model just below and just above the lips (see
    LIP_SEED_GAP); face_model: a face width in model units."""
    # Over the surface: vertices split along the texture's seams are one point.
    points, weld = np.unique(vertices, axis=0, return_inverse=True)
    weld = weld.ravel()
    tri = weld[faces]
    corners = points[tri]
    area_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, tri[:, corner], area_normals)
    tree = cKDTree(points)
    around = tree.query_ball_point(points, JAW_SMOOTH * face_model)
    smooth = np.array([normals[near].sum(axis=0) for near in around])
    smooth /= np.linalg.norm(smooth, axis=1, keepdims=True) + 1e-12
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    a, b = np.unique(np.sort(edges, axis=1), axis=0).T
    bend = np.arccos(np.clip((smooth[a] * smooth[b]).sum(axis=1), -1, 1))
    # A valley, not a ridge: the normals turn toward each other.
    fold = ((points[b] - points[a]) * (smooth[a] - smooth[b])).sum(axis=1) < 0
    cost = np.linalg.norm(points[a] - points[b], axis=1) + JAW_FOLD * face_model * bend * fold
    n = len(points)
    ends = (np.concatenate([a, b]), np.concatenate([b, a]))
    graph = coo_matrix((np.tile(cost, 2), ends), shape=(n, n)).tocsr()
    below, above = by_lips
    jaw_seeds = np.unique(tree.query(np.vstack([landmarks[JAW_SEEDS], below]))[1])
    to_jaw = dijkstra(graph, indices=jaw_seeds, min_only=True)
    face_seeds = np.unique(tree.query(np.vstack([landmarks[FACE_SEEDS], above]))[1])
    to_face = dijkstra(graph, indices=face_seeds, min_only=True)
    # Loose bits of the model (tufts of fur) that neither side reaches go with the
    # surface nearest them, so a chin's tufts drop with the chin.
    reached = np.isfinite(to_jaw) & np.isfinite(to_face)
    margin = np.zeros(len(points))
    margin[reached] = (to_face[reached] - to_jaw[reached]) / face_model
    nearest = cKDTree(points[reached]).query(points[~reached])[1]
    margin[~reached] = margin[reached][nearest]
    margin = np.array([margin[near].mean() for near in around])  # a smooth line to cut along
    # Only the jaw's own patch: not specks of skin it happens to reach first elsewhere.
    patch = margin > 0
    _, label = connected_components(graph[patch][:, patch], directed=False)
    labels = np.full(n, -1)
    labels[patch] = label
    stray = patch & ~np.isin(labels, labels[jaw_seeds])
    margin[stray] = -margin[stray]
    near, far = JAW_REACH
    reach = np.linalg.norm(points - landmarks[[14, 17]].mean(axis=0), axis=1) / face_model
    return margin[weld], (1 - smoothstep((reach - near) / (far - near)))[weld]


def cut_jaw(attributes: dict, faces, margin, reach, cut=None):
    """Cuts the jaw free along the line where margin is 0 (see find_jaw), so it opens
    without stretching skin: the faces across that line are split along it, and its
    points are made twice, once for each side. cut: per vertex, 0..1, how much the
    edge there is cut rather than stretched (default: all of it). attributes:
    per-vertex arrays, carried over onto the new points; returns them, the faces and
    each vertex's jaw weight (reach on the jaw's side, 0 on the face's where cut;
    blending across the edge where stretched)."""
    n = len(margin)
    cut = np.ones(n) if cut is None else cut
    on_jaw = margin > 0
    sides = on_jaw[faces]
    across = sides.any(axis=1) & ~sides.all(axis=1) & (cut[faces].mean(axis=1) > 0.5)
    hard = cut
    cut, kept = faces[across], faces[~across]
    # Turn each face so its first corner is the one alone on its side (winding kept).
    alone = np.where(sides[across].sum(axis=1) == 1, np.argmax(sides[across], axis=1),
                     np.argmin(sides[across], axis=1))
    cut = np.take_along_axis(cut, (alone[:, None] + np.arange(3)) % 3, axis=1)
    # Where the line crosses the edges from that corner; neighbouring faces share them.
    edges, slot = np.unique(np.sort(np.concatenate([cut[:, [0, 1]], cut[:, [0, 2]]]), axis=1),
                            axis=0, return_inverse=True)
    m = len(edges)
    a, b = edges.T
    t = margin[a] / (margin[a] - margin[b])
    out = {}
    for name, values in attributes.items():
        on_line = values[a] + t.reshape(-1, *[1] * (values.ndim - 1)) * (values[b] - values[a])
        out[name] = np.concatenate([values, on_line, on_line])  # the face's copy, then the jaw's
    along = lambda values: values[a] + t * (values[b] - values[a])  # noqa: E731
    stretch = smoothstep(margin / JAW_SOFT + 0.5)
    pull = reach * (hard * on_jaw + (1 - hard) * stretch)
    line_reach, line_hard = along(reach), along(hard)
    jaw = np.concatenate([pull, line_reach * (1 - line_hard) / 2, line_reach * (1 + line_hard) / 2])
    first, second = slot.ravel()[:len(cut)], slot.ravel()[len(cut):]
    own = m * on_jaw[cut[:, 0]]  # the lone corner's side
    other = m - own
    faces = np.concatenate([
        kept,
        np.column_stack([cut[:, 0], n + first + own, n + second + own]),
        np.column_stack([n + first + other, cut[:, 1], cut[:, 2]]),
        np.column_stack([n + first + other, cut[:, 2], n + second + other])])
    return out, faces, jaw.astype(np.float32)


def corners_together(vertices, lips, cam: "FrontCamera", face_model) -> np.ndarray:
    """How much of the jaw's pull each vertex keeps so the lips stay together at the
    corners of the mouth (see CORNER_START); lips: the line where they meet (model)."""
    q, line = vertices @ cam.rot.T, lips @ cam.rot.T  # seen from the front: x right, y up
    line = line[np.argsort(line[:, 0])]
    middle, half = line[:, 0].mean(), np.ptp(line[:, 0]) / 2
    out = smoothstep((np.abs(q[:, 0] - middle) / half - CORNER_START) / (1 - CORNER_START))
    below = (np.interp(q[:, 0], line[:, 0], line[:, 1]) - q[:, 1]) / face_model
    return 1 - out * (1 - smoothstep(below / CORNER_DEPTH))


def find_slit(points, opens, cam: "FrontCamera", face_model):
    """Where the lips meet: the middle of the slit seen from the front (model) and its
    width (model units), from the points on the cut (points) where the jaw opens."""
    q = points[opens] @ cam.rot.T
    front = q[:, 2] >= q[:, 2].max() - SLIT_FRONT * face_model
    q, slit = q[front], points[opens][front]
    width = np.ptp(q[:, 0])
    middle = np.abs(q[:, 0] - np.median(q[:, 0])) < 0.05 * width
    return slit[middle][np.argmax(q[middle, 2])], width


def outward(vertices, faces) -> np.ndarray:
    """faces turned so every piece of the model faces outward, all the same way.
    Meshy's remeshes mix them up, which an open mouth shows as a sawtooth of skin
    where its dark inside should be."""
    import trimesh

    welded = trimesh.Trimesh(vertices, faces, process=True)  # the same faces, on shared points
    before = welded.face_normals.copy()
    trimesh.repair.fix_normals(welded, multibody=True)
    flip = (welded.face_normals * before).sum(axis=1) < 0
    return np.where(flip[:, None], faces[:, ::-1], faces)


def import_glb(glb: str, name: str, own_mouth: bool = False, animal: str = "cat"):
    import trimesh  # only needed here

    mesh = trimesh.load(glb, force="mesh")
    vertices = mesh.vertices.astype(np.float64)
    faces = outward(vertices, mesh.faces)
    texture = despeckle(np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))[..., ::-1].copy())
    cat = cats.load(name)
    sprite, points = cat.sprite, cat.points
    h, w = sprite.shape[:2]

    raster = Raster(vertices, faces, mesh.visual.uv)
    iou, cam = fit_camera(raster, sprite[..., 3])
    print(f"{name}: portrait outline overlap {iou:.3f}")
    cam = align_features(raster, cam, texture.astype(np.float32), sprite, points)
    depth = raster.depth(cam, (w, h))
    texture = match_colours(raster, cam, depth, texture.astype(np.float32), sprite)

    # Every vertex's spot in the portrait.
    photo_px, _ = cam.project(vertices)
    # The face landmarks on the model's surface.
    # A point the face fit put past the model's edge (a chin below a head cut off at
    # the neck) takes the depth of the nearest spot on the model.
    _, (ny, nx) = distance_transform_edt(depth < -1e8, return_indices=True)

    def on_model(px):  # portrait px -> the model's front surface there
        x = np.clip(px[:, 0].round().astype(int), 0, w - 1)
        y = np.clip(px[:, 1].round().astype(int), 0, h - 1)
        return cam.unproject(px, depth[ny[y, x], nx[y, x]])

    landmarks = on_model(points[:, :2])
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    uv, jaw = mesh.visual.uv, np.zeros(0, np.float32)
    slit = {}
    if own_mouth:
        face_model = np.ptp(points[:, 0]) / cam.scale
        gap = (0, LIP_SEED_GAP * np.ptp(points[:, 0]))
        by_lips = [on_model(points[lip[1:-1], :2] + side * np.array(gap))
                   for lip, side in ((LOWER_INNER_LIP, 1), (UPPER_INNER_LIP, -1))]
        margin, reach = find_jaw(vertices, faces, landmarks, by_lips, face_model)
        n = len(vertices)
        lips = cKDTree(landmarks[UPPER_INNER_LIP + LOWER_INNER_LIP]).query(vertices)[0] / face_model
        cut = 1 - smoothstep((lips - CUT_REACH) / CUT_FADE)
        attributes = {"vertices": vertices, "normals": normals, "uv": uv, "photo_px": photo_px}
        parts, faces, jaw = cut_jaw(attributes, faces, margin, reach, cut)
        vertices, normals, uv, photo_px = (parts[k] for k in ("vertices", "normals", "uv", "photo_px"))
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        jaw = jaw * corners_together(vertices, landmarks[UPPER_INNER_LIP], cam, face_model).astype(np.float32)
        middle, width = find_slit(vertices[n:], jaw[n:] > 0.5, cam, face_model)
        slit = {"slit_middle": middle.astype("f4"), "slit_width": np.float32(width)}
        # Where the lips were cut apart: each point twice, on the face's side and on the
        # jaw's (see cut_jaw). The app shows the mouth's inside between the two.
        m = (len(vertices) - n) // 2
        upper = n + np.arange(m)
        lips_apart = jaw[upper + m] - jaw[upper] > LIPS_APART
        slit["lip_cut"] = np.column_stack([upper, upper + m])[lips_apart].astype("i4")
        print(f"  {np.count_nonzero(jaw > 0.5)} vertices move with the jaw")

    eye_index, eye_px, eye_weight = eye_paint(raster, cam, depth, texture.astype(np.float32), sprite, points)
    flat = texture.reshape(-1, 3).astype(np.float32)
    colours = sample(sprite[..., :3].astype(np.float32), eye_px)
    flat[eye_index] = colours * eye_weight[:, None] + flat[eye_index] * (1 - eye_weight[:, None])
    texture = np.clip(flat, 0, 255).astype(np.uint8).reshape(texture.shape)

    mesh_file, texture_file = files(name)
    np.savez_compressed(
        mesh_file, vertices=vertices.astype("f4"), normals=normals.astype("f4"),
        uv=uv.astype("f4"), faces=faces.astype("i4"),
        photo_px=photo_px.astype("f4"), landmarks=landmarks.astype("f4"),
        cam_rot=cam.rot.astype("f4"), cam_scale=np.float32(cam.scale),
        eye_index=eye_index, eye_px=eye_px.astype("f4"), eye_weight=eye_weight.astype("f4"),
        own_mouth=own_mouth, jaw_weight=jaw, animal=animal, **slit)
    cv2.imwrite(str(texture_file), texture)
    print(f"  {len(vertices)} vertices, {len(eye_index)} eye texels -> {mesh_file.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a generated 3D head for a cat.")
    parser.add_argument("glb", help="the head, a .glb with its texture")
    parser.add_argument("name", help="the cat, as in all_cats/<Name>.png")
    parser.add_argument("--own-mouth", action="store_true",
                        help="keep the head's own lips and inside, and cut its own jaw free")
    parser.add_argument("--animal", choices=ANIMALS, default="cat", help="whose teeth and tongue it gets")
    args = parser.parse_args()
    import_glb(args.glb, args.name, args.own_mouth, args.animal)
