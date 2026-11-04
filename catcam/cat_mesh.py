"""3D cat head mesh rendered with OpenGL.

Vertices are the cat's 478 face landmarks (with depth), points along the cat's
outline and a grid of "skull" points filling the fur in between. Face
vertices are moved by your expression each frame, the whole head rotates with
yours, and the gap between the lips is filled with a drawn mouth (teeth, dark
inside, tongue). The texture is the cat sprite, so at rest the mesh reproduces
the photo.

Sizes are in face widths, so any sprite resolution behaves the same.
"""

import cv2
import moderngl
import numpy as np
from scipy.spatial import Delaunay

OUTLINE_POINTS = 90
SKULL_DEPTH = 0.15  # head outline sits this far (in face widths) behind the nearest face points
OUTLINE_DEPTH_SMOOTHING = 9  # outline points averaged for a smooth silhouette depth
CAMERA_DISTANCE = 4.0  # perspective strength, in face widths (bigger = flatter)
DEPTH_RANGE = 2.0  # face widths in front of / behind the pivot that fit in the GL depth range
MSAA_SAMPLES = 4
TEETH_SHOW = (0.01, 0.035)  # lip gap over which the teeth fade in
PROBE_OPENING = 0.18  # used once to find the triangles spanning the mouth
PROBE_TOLERANCE = 0.005
# Fur-follows-face: neighbourhood size and falloff.
FUR_FOLLOW = (0.1, 0.3)
SKULL_GRID = 0.07  # spacing of the skull points
SKULL_MARGIN = 0.03  # skull points keep this far from the face and the outline
LIP_SEAM = 0.0015  # half the gap between the lips of the closed mouth
# Framing: ear tips and chin as fractions of the screen height.
TOP_MARGIN, CHIN_LINE = 0.02, 0.93

# Inner lip contours, both running corner (78) to corner (308).
UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

FACE_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    uv = in_uv;
}
"""
FACE_FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(tex, uv);
    if (c.a < 0.02) discard;
    color = c;
}
"""
# Mouth interior: t runs 0 at the upper lip to 1 at the lower lip, s runs 0 at
# one mouth corner to 1 at the other. Colors are BGR because the whole pipeline is BGR.
MOUTH_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec2 in_ts;
out vec2 ts;
void main() {
    // Slightly in front of the lips, so no skin stretched across the opening
    // can show through; the nose etc. are far enough forward to still cover it.
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.01, 1.0);
    ts = in_ts;
}
"""
MOUTH_FRAGMENT_SHADER = """
#version 330
uniform float teeth_amount;  // 0 while the lips touch, so a closed mouth stays a dark line
in vec2 ts;
out vec4 color;
const float TEETH = 9.0;  // upper teeth across the mouth
void main() {
    float t = ts.x, s = ts.y;
    vec3 inside = vec3(0.10, 0.06, 0.16);
    vec3 tongue = vec3(0.38, 0.36, 0.70);
    vec3 c = mix(inside, tongue, smoothstep(0.62, 0.85, t));

    // Upper teeth: rounded tips, dark gaps between them, shaded toward the corners.
    float tooth = fract(s * TEETH);
    float edge = min(tooth, 1.0 - tooth);
    float length = 0.2 + 0.12 * smoothstep(0.0, 0.35, edge);
    float corner = smoothstep(0.0, 0.3, min(s, 1.0 - s));
    vec3 enamel = vec3(0.82, 0.88, 0.90) * mix(0.55, 1.0, smoothstep(0.0, 0.1, edge));
    enamel *= mix(0.35, 1.0, corner) * mix(1.0, 0.85, t / length);
    float on_tooth = (1.0 - smoothstep(length - 0.03, length, t)) * teeth_amount;
    color = vec4(mix(c, enamel, on_tooth), 1.0);
}
"""


def _outline(alpha: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (alpha > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=len)[:, 0].astype(np.float64) + 0.5
    step = len(contour) / OUTLINE_POINTS
    return contour[(np.arange(OUTLINE_POINTS) * step).astype(int)]


def _closed_mouth(face: np.ndarray, face_width: float) -> np.ndarray:
    """The face with its lips together, so at rest no mouth opening shows.

    (The fitted inner lips of a cat rarely touch exactly.) They stay a hair
    apart, as the mesh needs distinct vertices.
    """
    face = face.copy()
    upper, lower = UPPER_INNER_LIP[1:-1], LOWER_INNER_LIP[1:-1]
    middle = (face[upper] + face[lower]) / 2
    face[upper] = middle - (0, LIP_SEAM * face_width, 0)
    face[lower] = middle + (0, LIP_SEAM * face_width, 0)
    return face


def _outline_depth(outline_xy: np.ndarray, face: np.ndarray, face_width: float) -> np.ndarray:
    """Depth of the head outline: behind the face points next to it.

    Beside the cheeks that is behind the face's sides (in front of them the mesh
    would fold over when turning); above the forehead it is much less deep,
    or the forehead would slide over the top of the head when nodding.
    """
    nearest = np.linalg.norm(outline_xy[:, None] - face[None, :, :2], axis=2).argmin(axis=1)
    z = face[nearest, 2] + SKULL_DEPTH * face_width
    k = OUTLINE_DEPTH_SMOOTHING
    padded = np.concatenate([z[-k:], z, z[:k]])  # the outline is closed
    return np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def _skull_points(alpha: np.ndarray, face: np.ndarray, face_width: float, outline: np.ndarray):
    """Grid points in the fur around the face, with depth easing from the face
    edge back to the outline, so the head bends smoothly when it turns."""
    mask = (alpha > 20).astype(np.uint8)
    to_outline = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    hull = cv2.convexHull(face[:, :2].astype(np.float32))
    step = SKULL_GRID * face_width
    margin = SKULL_MARGIN * face_width
    ys, xs = np.mgrid[step / 2:alpha.shape[0]:step, step / 2:alpha.shape[1]:step]
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    d_outline = to_outline[grid[:, 1].astype(int), grid[:, 0].astype(int)]
    d_face = -np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in grid])
    keep = (d_outline > margin) & (d_face > margin)
    grid, d_outline, d_face = grid[keep], d_outline[keep], d_face[keep]
    near_face = np.linalg.norm(grid[:, None] - face[None, :, :2], axis=2).argmin(axis=1)
    near_outline = np.linalg.norm(grid[:, None] - outline[None, :, :2], axis=2).argmin(axis=1)
    t = d_face / (d_face + d_outline)
    z = (1 - t) * face[near_face, 2] + t * outline[near_outline, 2]
    return np.column_stack([grid, z])


def _face_triangles(points_2d: np.ndarray, face_width: float) -> np.ndarray:
    """Delaunay triangles, minus the ones bridging the lips (the mouth opening).

    The mouth is opened artificially (everything below the upper lip moves
    down) and triangulated open, so the triangles follow the lips; the ones
    inside the opening are then dropped.
    """
    upper = points_2d[UPPER_INNER_LIP]
    order = np.argsort(upper[:, 0])
    lip_line = np.interp(points_2d[:, 0], upper[order, 0], upper[order, 1])
    opened = points_2d.copy()
    below = points_2d[:, 1] > lip_line + 0.5 * PROBE_TOLERANCE * face_width
    below[LOWER_INNER_LIP[1:-1]] = True
    below[UPPER_INNER_LIP] = False
    opened[below, 1] += PROBE_OPENING * face_width

    triangles = Delaunay(opened).simplices
    mouth = np.vstack([opened[UPPER_INNER_LIP], opened[LOWER_INNER_LIP][::-1]]).astype(np.float32)
    corners = opened[triangles]  # (n, 3, 2)
    probes = np.concatenate([
        corners.mean(axis=1, keepdims=True),
        (corners + np.roll(corners, 1, axis=1)) / 2,  # edge midpoints
    ], axis=1)
    tolerance = PROBE_TOLERANCE * face_width
    inside = np.array([
        any(cv2.pointPolygonTest(mouth, (float(x), float(y)), True) > tolerance for x, y in tri)
        for tri in probes
    ])
    return triangles[~inside]


def _mouth_triangles() -> np.ndarray:
    """Strip of triangles between the inner lip contours."""
    u, lo = UPPER_INNER_LIP, LOWER_INNER_LIP
    quads = [((u[i], u[i + 1], lo[i + 1]), (u[i], lo[i + 1], lo[i])) for i in range(len(u) - 1)]
    return np.array(quads).reshape(-1, 3)


class Renderer:
    """OpenGL context, shaders and frame buffers, shared by all cats."""

    def __init__(self, out_size):
        self.out_w, self.out_h = out_size
        self.ctx = moderngl.create_standalone_context(require=330)
        self.face_prog = self.ctx.program(
            vertex_shader=FACE_VERTEX_SHADER, fragment_shader=FACE_FRAGMENT_SHADER
        )
        self.mouth_prog = self.ctx.program(
            vertex_shader=MOUTH_VERTEX_SHADER, fragment_shader=MOUTH_FRAGMENT_SHADER
        )
        size = (self.out_w, self.out_h)
        self.msaa = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size, 4, samples=MSAA_SAMPLES)],
            depth_attachment=self.ctx.depth_renderbuffer(size, samples=MSAA_SAMPLES),
        )
        self.resolved = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4)])


class CatMesh:
    def __init__(self, renderer: Renderer, sprite: np.ndarray, landmarks: np.ndarray, bg_color):
        """sprite: BGRA, landmarks: (478, 3) sprite px with depth, bg_color: BGR."""
        self.renderer = renderer
        self.out_w, self.out_h = renderer.out_w, renderer.out_h
        self.bg = tuple(np.asarray(bg_color) / 255) + (1.0,)
        tex_h, tex_w = sprite.shape[:2]

        self.face_count = len(landmarks)
        self.face_width = np.ptp(landmarks[:, 0])
        face = _closed_mouth(landmarks, self.face_width)
        outline_xy = _outline(sprite[..., 3])
        outline = np.column_stack([outline_xy, _outline_depth(outline_xy, face, self.face_width)])
        skull = _skull_points(sprite[..., 3], face, self.face_width, outline)

        self.rest = np.vstack([face, skull, outline])
        # The fur follows the nearby face points (e.g. cheeks widening in a
        # smile) with a falloff, so the face edge can't fold over the fur.
        fur = self.rest[self.face_count:, :2]
        dist = np.linalg.norm(fur[:, None] - face[None, :, :2], axis=2)
        weights = np.exp(-((dist / (FUR_FOLLOW[0] * self.face_width)) ** 2))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9
        falloff = np.exp(-((dist.min(axis=1) / (FUR_FOLLOW[1] * self.face_width)) ** 2))
        self.fur_follow = weights * falloff[:, None]  # (fur points, face points)
        # Rotate around the middle of the face so the head turns in place.
        self.pivot = np.array([face[:, 0].mean(), face[1, 1], np.median(face[:, 2])])
        self.focal = CAMERA_DISTANCE * self.face_width

        top, chin = outline_xy[:, 1].min(), face[:, 1].max()
        self.scale = (CHIN_LINE - TOP_MARGIN) * self.out_h / (chin - top)
        self.offset = np.array([
            self.out_w / 2 - self.pivot[0] * self.scale,
            TOP_MARGIN * self.out_h - top * self.scale,
        ])

        ctx = renderer.ctx
        self.vbo = ctx.buffer(reserve=len(self.rest) * 3 * 4, dynamic=True)
        uv = (self.rest[:, :2] / (tex_w, tex_h)).astype("f4")
        triangles = _face_triangles(self.rest[:, :2], self.face_width)
        self.face_vao = ctx.vertex_array(
            renderer.face_prog,
            [(self.vbo, "3f", "in_pos"), (ctx.buffer(uv.tobytes()), "2f", "in_uv")],
            ctx.buffer(triangles.astype("i4").tobytes()),
        )
        lip_ts = np.zeros((len(self.rest), 2), "f4")
        lip_ts[LOWER_INNER_LIP[1:-1], 0] = 1.0
        steps = np.linspace(0, 1, len(UPPER_INNER_LIP))
        lip_ts[UPPER_INNER_LIP, 1] = steps
        lip_ts[LOWER_INNER_LIP, 1] = steps
        self.mouth_vao = ctx.vertex_array(
            renderer.mouth_prog,
            [(self.vbo, "3f", "in_pos"), (ctx.buffer(lip_ts.tobytes()), "2f", "in_ts")],
            ctx.buffer(_mouth_triangles().astype("i4").tobytes()),
        )

        self.texture = ctx.texture((tex_w, tex_h), 4)
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._uploaded = None  # sprite currently in the texture

    def face_on_screen(self):
        """Screen position of the cat's face center and its width in screen px."""
        return self.offset + self.pivot[:2] * self.scale, self.face_width * self.scale

    def _project(self, offsets: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Deform the face, rotate the head, and project to NDC."""
        shaped = self.rest.copy()
        shaped[: self.face_count] += offsets
        shaped[self.face_count:] += self.fur_follow @ offsets
        pos = (shaped - self.pivot) @ rotation.T + self.pivot
        # Perspective relative to the unrotated pose, so at rest the mesh matches the photo.
        depth = lambda z: self.focal / (self.focal + z - self.pivot[2])
        persp = (depth(pos[:, 2]) / depth(shaped[:, 2]))[:, None]
        xy = self.pivot[:2] + (pos[:, :2] - self.pivot[:2]) * persp
        screen = self.offset + xy * self.scale
        return np.column_stack([
            2 * screen[:, 0] / self.out_w - 1,
            1 - 2 * screen[:, 1] / self.out_h,
            (pos[:, 2] - self.pivot[2]) / (DEPTH_RANGE * self.face_width),
        ]).astype("f4")

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Draw the cat with face offsets (478, 3) and head rotation; returns BGR."""
        if sprite is not self._uploaded:  # unchanged while the eyes are open
            self.texture.write(np.ascontiguousarray(sprite).tobytes())
            self.texture.build_mipmaps()
            self._uploaded = sprite
        self.vbo.write(self._project(offsets, rotation).tobytes())
        gap = np.mean(offsets[LOWER_INNER_LIP, 1] - offsets[UPPER_INNER_LIP, 1]) / self.face_width
        teeth = np.clip((gap - TEETH_SHOW[0]) / (TEETH_SHOW[1] - TEETH_SHOW[0]), 0, 1)
        r = self.renderer
        r.mouth_prog["teeth_amount"].value = float(teeth)

        r.msaa.use()
        r.ctx.clear(*self.bg, depth=1.0)
        r.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        r.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.mouth_vao.render()
        self.texture.use(0)
        self.face_vao.render()
        r.ctx.copy_framebuffer(r.resolved, r.msaa)

        data = r.resolved.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.out_h, self.out_w, 3)[::-1]
