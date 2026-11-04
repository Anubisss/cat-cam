"""3D cat head mesh rendered with OpenGL.

Vertices are the cat's 478 face landmarks (with depth) plus points along the
cat's outline. Face vertices are moved by your expression each frame, the whole
head rotates with yours, and the gap between the lips is filled with a drawn
mouth (teeth, dark inside, tongue). The texture is the cat sprite, so at rest
the mesh reproduces the photo.
"""

import cv2
import moderngl
import numpy as np
from scipy.spatial import Delaunay

OUTLINE_POINTS = 90
SKULL_DEPTH = 0.15  # head outline sits this far (in face widths) behind the face sides
CAMERA_DISTANCE = 4.0  # perspective strength, in face widths (bigger = flatter)
DEPTH_RANGE = 400.0  # sprite px mapped to the GL depth range
MSAA_SAMPLES = 4
TEETH_SHOW = (2.0, 8.0)  # lip gap (sprite px) over which the teeth fade in
PROBE_OPENING = 40.0
# Outline-follows-face: neighbourhood size and falloff, in face widths.
OUTLINE_FOLLOW = (0.1, 0.3)  # sprite px; used once to find the triangles spanning the mouth
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


def _face_triangles(points_2d: np.ndarray) -> np.ndarray:
    """Delaunay triangles, minus the ones bridging the lips (the mouth opening).

    The mouth is opened artificially (everything below the upper lip moves
    down); triangles that then overlap the opening are the bridging ones.
    """
    triangles = Delaunay(points_2d).simplices
    upper = points_2d[UPPER_INNER_LIP]
    order = np.argsort(upper[:, 0])
    lip_line = np.interp(points_2d[:, 0], upper[order, 0], upper[order, 1])
    opened = points_2d.copy()
    below = points_2d[:, 1] > lip_line + 0.5
    below[LOWER_INNER_LIP[1:-1]] = True
    below[UPPER_INNER_LIP] = False
    opened[below, 1] += PROBE_OPENING

    mouth = np.vstack([opened[UPPER_INNER_LIP], opened[LOWER_INNER_LIP][::-1]]).astype(np.float32)
    corners = opened[triangles]  # (n, 3, 2)
    probes = np.concatenate([
        corners.mean(axis=1, keepdims=True),
        (corners + np.roll(corners, 1, axis=1)) / 2,  # edge midpoints
    ], axis=1)
    inside = np.array([
        any(cv2.pointPolygonTest(mouth, (float(x), float(y)), True) > 1 for x, y in tri)
        for tri in probes
    ])
    return triangles[~inside]


def _mouth_triangles() -> np.ndarray:
    """Strip of triangles between the inner lip contours."""
    u, lo = UPPER_INNER_LIP, LOWER_INNER_LIP
    quads = [((u[i], u[i + 1], lo[i + 1]), (u[i], lo[i + 1], lo[i])) for i in range(len(u) - 1)]
    return np.array(quads).reshape(-1, 3)


class CatMesh:
    def __init__(self, sprite: np.ndarray, landmarks: np.ndarray, out_size, bg_color):
        """sprite: BGRA, landmarks: (478, 3) sprite px with depth, bg_color: BGR."""
        self.out_w, self.out_h = out_size
        self.bg = tuple(np.asarray(bg_color) / 255) + (1.0,)
        tex_h, tex_w = sprite.shape[:2]

        face = landmarks
        self.face_count = len(face)
        self.face_width = np.ptp(face[:, 0])
        # The head outline (widest point of the skull) lies behind the face's
        # sides; in front of them the mesh would fold over when turning.
        silhouette_z = face[:, 2].max() + SKULL_DEPTH * self.face_width
        outline_xy = _outline(sprite[..., 3])
        outline = np.column_stack([outline_xy, np.full(len(outline_xy), silhouette_z)])

        self.rest = np.vstack([face, outline])
        # The fur outline follows the nearby face points (e.g. cheeks widening
        # in a smile) with a falloff, so the face edge can't fold over the fur.
        dist = np.linalg.norm(outline_xy[:, None] - face[None, :, :2], axis=2)
        weights = np.exp(-((dist / (OUTLINE_FOLLOW[0] * self.face_width)) ** 2))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9
        falloff = np.exp(-((dist.min(axis=1) / (OUTLINE_FOLLOW[1] * self.face_width)) ** 2))
        self.outline_follow = weights * falloff[:, None]  # (outline, face)
        # Rotate around the middle of the face so the head turns in place.
        self.pivot = np.array([face[:, 0].mean(), face[1, 1], np.median(face[:, 2])])
        self.focal = CAMERA_DISTANCE * self.face_width

        top, chin = outline_xy[:, 1].min(), face[:, 1].max()
        self.scale = (CHIN_LINE - TOP_MARGIN) * self.out_h / (chin - top)
        self.offset = np.array([
            self.out_w / 2 - self.pivot[0] * self.scale,
            TOP_MARGIN * self.out_h - top * self.scale,
        ])

        self.ctx = moderngl.create_standalone_context(require=330)
        self.vbo = self.ctx.buffer(reserve=len(self.rest) * 3 * 4, dynamic=True)
        uv = (self.rest[:, :2] / (tex_w, tex_h)).astype("f4")
        face_prog = self.ctx.program(
            vertex_shader=FACE_VERTEX_SHADER, fragment_shader=FACE_FRAGMENT_SHADER
        )
        self.face_vao = self.ctx.vertex_array(
            face_prog,
            [(self.vbo, "3f", "in_pos"), (self.ctx.buffer(uv.tobytes()), "2f", "in_uv")],
            self.ctx.buffer(_face_triangles(self.rest[:, :2]).astype("i4").tobytes()),
        )
        lip_ts = np.zeros((len(self.rest), 2), "f4")
        lip_ts[LOWER_INNER_LIP[1:-1], 0] = 1.0
        steps = np.linspace(0, 1, len(UPPER_INNER_LIP))
        lip_ts[UPPER_INNER_LIP, 1] = steps
        lip_ts[LOWER_INNER_LIP, 1] = steps
        mouth_prog = self.ctx.program(
            vertex_shader=MOUTH_VERTEX_SHADER, fragment_shader=MOUTH_FRAGMENT_SHADER
        )
        self.teeth_amount = mouth_prog["teeth_amount"]
        self.mouth_vao = self.ctx.vertex_array(
            mouth_prog,
            [(self.vbo, "3f", "in_pos"), (self.ctx.buffer(lip_ts.tobytes()), "2f", "in_ts")],
            self.ctx.buffer(_mouth_triangles().astype("i4").tobytes()),
        )

        self.texture = self.ctx.texture((tex_w, tex_h), 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        size = (self.out_w, self.out_h)
        self.msaa = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size, 4, samples=MSAA_SAMPLES)],
            depth_attachment=self.ctx.depth_renderbuffer(size, samples=MSAA_SAMPLES),
        )
        self.resolved = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4)])

    def face_on_screen(self):
        """Screen position of the cat's face center and its width in screen px."""
        return self.offset + self.pivot[:2] * self.scale, self.face_width * self.scale

    def _project(self, offsets: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Deform the face, rotate the head, and project to NDC."""
        shaped = self.rest.copy()
        shaped[: self.face_count] += offsets
        shaped[self.face_count:] += self.outline_follow @ offsets
        pos = (shaped - self.pivot) @ rotation.T + self.pivot
        # Perspective relative to the unrotated pose, so at rest the mesh matches the photo.
        depth = lambda z: self.focal / (self.focal + z - self.pivot[2])
        persp = (depth(pos[:, 2]) / depth(shaped[:, 2]))[:, None]
        xy = self.pivot[:2] + (pos[:, :2] - self.pivot[:2]) * persp
        screen = self.offset + xy * self.scale
        return np.column_stack([
            2 * screen[:, 0] / self.out_w - 1,
            1 - 2 * screen[:, 1] / self.out_h,
            pos[:, 2] / DEPTH_RANGE,
        ]).astype("f4")

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Draw the cat with face offsets (478, 3) and head rotation; returns BGR."""
        self.texture.write(np.ascontiguousarray(sprite).tobytes())
        self.vbo.write(self._project(offsets, rotation).tobytes())
        gap = np.mean(offsets[LOWER_INNER_LIP, 1] - offsets[UPPER_INNER_LIP, 1])
        self.teeth_amount.value = float(np.clip((gap - TEETH_SHOW[0]) / (TEETH_SHOW[1] - TEETH_SHOW[0]), 0, 1))

        self.msaa.use()
        self.ctx.clear(*self.bg, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.mouth_vao.render()
        self.texture.use(0)
        self.face_vao.render()
        self.ctx.copy_framebuffer(self.resolved, self.msaa)

        data = self.resolved.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.out_h, self.out_w, 3)[::-1]
