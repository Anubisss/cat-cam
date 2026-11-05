"""A cat drawn as a real 3D head model instead of the warped photo.

The head is a textured mesh (assets/cats/<Name>/head3d.npz + head3d.png),
made from the cat's portrait by an image-to-3D model and imported with
catcam.head3d_import, which also records where every vertex sits in the
portrait. That is what lets it borrow everything the 2D cat does: each vertex
moves with the face landmarks around it (expressions, the jaw), the lips part
over the same drawn mouth, the eye areas of the texture blink with the
portrait's eyelids, and the ears bend about the same hinges. Head3D stands in
for CatMesh: same render() and face_on_screen().
"""

import json

import cv2
import moderngl
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from catcam.cat_mesh import CHIN_LINE, TOP_MARGIN, CatMesh, Renderer, _mouth_triangles, _outline
from catcam.ears import bend, find_ears
from catcam.lighting import LIGHT, SHADE_RANGE, STRENGTH
from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP
from catcam.paths import ASSETS

FOV = 25  # vertical field of view of the virtual camera, degrees
FACE_BAND = (0.35, 0.75)  # the face's width is measured between these heights (0 = chin, 1 = ear tips)
# How vertices follow the face landmarks (face widths): each takes a Gaussian-weighted
# mix of its nearest landmarks' movement, fading out away from the face.
NEAREST = 8
FOLLOW = 0.06
FALLOFF = 0.25  # wide, so the whole lower jaw and chin move together instead of tearing
# Near the mouth, vertices only follow landmarks on their own side of the lips, so the
# lips part instead of stretching: this far past the mouth corners and above/below the lip line.
MOUTH_ZONE = (0.06, 0.15)
DEPTH_SPAN = 1.5  # model units in front of / behind the head kept in the depth range
# Image axes (x right, y down, z away from the camera) <-> model axes (y up, z toward it).
FLIP = np.diag([1.0, -1.0, -1.0])
LIPS = UPPER_INNER_LIP + LOWER_INNER_LIP

VERTEX_SHADER = """
#version 330
uniform mat4 proj;
uniform mat3 rot;
uniform vec3 shift;
uniform vec3 light;
uniform float strength;
uniform vec2 shade_range;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec2 uv;
out float shade;
void main() {
    gl_Position = proj * vec4(rot * in_pos + shift, 1.0);
    uv = in_uv;
    // The texture already has the light of the photo it came from baked in; only the
    // change in facing since then brightens or darkens it (see catcam.lighting).
    float before = clamp(dot(in_normal, light), 0.0, 1.0);
    float after = clamp(dot(rot * in_normal, light), 0.0, 1.0);
    shade = clamp(1.0 + strength * (after - before), shade_range.x, shade_range.y);
}
"""
FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
in vec2 uv;
in float shade;
out vec4 color;
void main() { color = vec4(texture(tex, uv).rgb * shade, 1.0); }
"""
# The inside of the mouth, from a photo of the cat with its mouth wide open
# (head3d_mouth.png + .json: the opening traced in that photo). Each tooth row
# stays with its own lip: the part of the gap below the upper lip shows the
# photo from its upper edge down, the part above the lower lip shows it from
# its lower edge up, unsquashed, so opening wider slides more teeth into view.
MOUTH_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_mouth;
out vec3 m;
void main() {
    // Slightly in front of the lips, so the skin stretched across the opening can't show.
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.01, 1.0);
    m = in_mouth;
}
"""
MOUTH_FRAGMENT_SHADER = """
#version 330
uniform sampler2D photo;
uniform vec2 photo_size;
uniform vec2 upper[11];
uniform vec2 lower[11];
uniform float face_px;  // a face width in the photo
const float FULL_OPEN = 0.5;  // gap, as a share of the photo's, at which all of the photo's mouth shows
in vec3 m;  // along the mouth 0..1, face widths below the upper lip, gap here (face widths)
out vec4 color;

vec2 along(vec2 points[11], float s) {
    float i = clamp(s, 0.0, 1.0) * 10.0;
    int j = min(int(i), 9);
    return mix(points[j], points[j + 1], i - float(j));
}

void main() {
    vec2 top = along(upper, m.x), bottom = along(lower, m.x);
    vec2 span = bottom - top;
    float photo_gap = length(span);
    vec2 down = photo_gap > 1e-3 ? span / photo_gap : vec2(0.0, 1.0);
    float gap = max(m.z, 1e-4), t = clamp(m.y / gap, 0.0, 1.0);
    // How much of the photo's opening shows: just the gap's own size (teeth edges,
    // unsquashed) when barely open, all of it (both fang rows) from FULL_OPEN on.
    float photo_fw = photo_gap / face_px;
    float shown = mix(gap, photo_fw, smoothstep(0.0, FULL_OPEN * photo_fw, gap)) * face_px;
    shown = min(shown, photo_gap);
    vec2 p = t <= 0.5 ? top + down * t * shown : bottom - down * (1.0 - t) * shown;
    color = vec4(texture(photo, p / photo_size).rgb, 1.0);
}
"""


def files(name: str):
    return ASSETS / name / "head3d.npz", ASSETS / name / "head3d.png"


def mouth_files(name: str):
    """Optional open-mouth photo and its traced opening (see MOUTH_FRAGMENT_SHADER)."""
    return ASSETS / name / "head3d_mouth.png", ASSETS / name / "head3d_mouth.json"


def has_head(name: str) -> bool:
    return all(f.exists() for f in files(name))


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])


def sample(image: np.ndarray, px: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear colours of image at px (n, 2); remap wants 2D maps under 32k a side."""
    n = len(px)
    if n == 0:
        return np.zeros((0, image.shape[2]), np.float32)
    pad = -n % width
    maps = [np.pad(c.astype(np.float32), (0, pad)).reshape(-1, width) for c in px.T]
    return cv2.remap(image, *maps, cv2.INTER_LINEAR).reshape(-1, image.shape[2])[:n]


def follow_weights(vertices, photo_px, landmarks, points, face_model, face_width) -> csr_matrix:
    """(vertices, landmarks) weights: how each vertex moves with the face landmarks.
    Closeness is measured on the model (landmarks: their spots on it; face_model: a
    face width in model units), so the back of the head stays put; which side of the
    lips a vertex is on is decided in the portrait (photo_px, points, face_width px)."""
    tree = cKDTree(landmarks)
    dist, nearest = tree.query(vertices, k=NEAREST)
    weights = np.exp(-((dist / (FOLLOW * face_model)) ** 2))
    # Lips: keep each vertex to the landmarks on its side of the line between them.
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    order = np.argsort(seam[:, 0])
    line_y = lambda x: np.interp(x, seam[order, 0], seam[order, 1])  # noqa: E731
    left, right = seam[:, 0].min(), seam[:, 0].max()
    reach_x, reach_y = (r * face_width for r in MOUTH_ZONE)
    near_mouth = ((photo_px[:, 0] > left - reach_x) & (photo_px[:, 0] < right + reach_x)
                  & (np.abs(photo_px[:, 1] - line_y(photo_px[:, 0])) < reach_y))
    below = points[:, 1] > line_y(points[:, 0])
    below[UPPER_INNER_LIP], below[LOWER_INNER_LIP] = False, True
    vertex_below = photo_px[:, 1] > line_y(photo_px[:, 0])
    wrong_side = near_mouth[:, None] & (below[nearest] != vertex_below[:, None])
    weights[wrong_side] = 0
    total = weights.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0)
    falloff = np.exp(-((dist[:, 0] / (FALLOFF * face_model)) ** 2))
    weights *= falloff[:, None]
    rows = np.repeat(np.arange(len(photo_px)), NEAREST)
    return csr_matrix((weights.ravel(), (rows, nearest.ravel())), shape=(len(photo_px), len(points)))


class Head3D:
    def __init__(self, renderer: Renderer, name: str, flat: CatMesh, sprite: np.ndarray,
                 points: np.ndarray, bg_color):
        """flat: the same cat's CatMesh, whose expression smoothing and mouth this
        head shares; sprite: its open-eyed portrait (BGRA); points: its landmarks;
        bg_color: BGR."""
        mesh_file, texture_file = files(name)
        mesh = np.load(mesh_file)
        self.renderer, self.flat, self.open_sprite = renderer, flat, sprite
        self.out_w, self.out_h = renderer.out_w, renderer.out_h
        self.bg = tuple(np.asarray(bg_color) / 255) + (1.0,)

        centre = (mesh["vertices"].min(0) + mesh["vertices"].max(0)) / 2
        self.rest = mesh["vertices"] - centre
        self.landmarks = mesh["landmarks"] - centre
        self.photo_px = mesh["photo_px"]
        # A movement in portrait px (x right, y down, z away) as a movement of the model.
        self.to_model = (mesh["cam_rot"].T @ np.diag([1, -1, -1]) / mesh["cam_scale"]).astype(np.float64)
        face_width = np.ptp(points[:, 0])
        self.lip_width = face_width
        self.lip_rest = points[LIPS, :2]
        self.follow = follow_weights(self.rest, self.photo_px, self.landmarks, points,
                                     face_width / float(mesh["cam_scale"]), face_width)
        outline = _outline(sprite[..., 3])
        self.ears = find_ears(np.vstack([points[:, :2], self.photo_px]), outline, len(points), face_width)
        self.face_count = len(points)

        # Frame it like CatMesh: top of the ears at TOP_MARGIN, chin at CHIN_LINE.
        height = np.ptp(self.rest[:, 1])
        half = np.tan(np.radians(FOV) / 2)
        distance = height / ((CHIN_LINE - TOP_MARGIN) * 2 * half)
        middle_ndc = 1 - (TOP_MARGIN + CHIN_LINE)
        self.shift = np.array([0.0, middle_ndc * distance * half, -distance])
        self.proj = perspective(FOV, self.out_w / self.out_h, distance - DEPTH_SPAN, distance + DEPTH_SPAN)
        px_per_unit = self.out_h / (2 * half * distance)
        lo, hi = self.rest[:, 1].min() + np.array(FACE_BAND) * height
        band = self.rest[(self.rest[:, 1] > lo) & (self.rest[:, 1] < hi)]
        self.face_width = np.ptp((band if len(band) else self.rest)[:, 0]) * px_per_unit
        self.face_centre = np.array([self.out_w / 2, (1 - middle_ndc) / 2 * self.out_h])

        ctx = renderer.ctx
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.prog["proj"].write(self.proj.astype("f4").T.tobytes())
        self.prog["shift"].value = tuple(self.shift)
        self.prog["light"].value = tuple(LIGHT @ FLIP)
        self.prog["strength"].value = STRENGTH
        self.prog["shade_range"].value = SHADE_RANGE
        self.vbo = ctx.buffer(self.rest.astype("f4").tobytes(), dynamic=True)
        self.vao = ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_pos"),
                        (ctx.buffer(mesh["normals"].astype("f4").tobytes()), "3f", "in_normal"),
                        (ctx.buffer(mesh["uv"].astype("f4").tobytes()), "2f", "in_uv")],
            ctx.buffer(mesh["faces"].astype("i4").tobytes()))

        # The mouth between the lips: from the open-mouth photo if the cat has one,
        # else CatMesh's drawn mouth.
        index = {v: i for i, v in enumerate(LIPS)}
        self.mouth_pos = ctx.buffer(reserve=len(LIPS) * 3 * 4, dynamic=True)
        self.mouth_attr = ctx.buffer(reserve=len(LIPS) * 3 * 4, dynamic=True)
        triangles = ctx.buffer(np.vectorize(index.get)(_mouth_triangles()).astype("i4").tobytes())
        mouth_image, mouth_json = mouth_files(name)
        self.photo_mouth = mouth_image.exists() and mouth_json.exists()
        if self.photo_mouth:
            trace = json.loads(mouth_json.read_text())
            photo = cv2.imread(str(mouth_image))
            self.mouth_prog = ctx.program(vertex_shader=MOUTH_VERTEX_SHADER,
                                          fragment_shader=MOUTH_FRAGMENT_SHADER)
            self.mouth_prog["photo_size"].value = (photo.shape[1], photo.shape[0])
            self.mouth_prog["upper"].value = [tuple(p) for p in trace["upper"]]
            self.mouth_prog["lower"].value = [tuple(p) for p in trace["lower"]]
            self.mouth_prog["face_px"].value = float(trace["face_px"])
            # Rows top-down, as the shader looks up photo px directly.
            self.mouth_photo = ctx.texture(photo.shape[1::-1], 3, np.ascontiguousarray(photo).tobytes())
            self.mouth_vao = ctx.vertex_array(
                self.mouth_prog, [(self.mouth_pos, "3f", "in_pos"), (self.mouth_attr, "3f", "in_mouth")],
                triangles)
        else:
            self.mouth_vao = ctx.vertex_array(
                renderer.mouth_prog,
                [(self.mouth_pos, "3f", "in_pos"), (self.mouth_attr, "3f", "in_mouth"),
                 (ctx.buffer(np.ones(len(LIPS), "f4").tobytes()), "1f", "in_shade")],
                triangles)

        # Blinks: the eye areas were painted from the portrait, so they repaint from its
        # eyelid-drawn versions.
        self.base = cv2.imread(str(texture_file))
        self.eye_index, self.eye_px = mesh["eye_index"], mesh["eye_px"]
        self.eye_weight = mesh["eye_weight"][:, None]
        self.eye_open = sample(sprite[..., :3].astype(np.float32), self.eye_px)
        self.eye_base = self.base.reshape(-1, 3)[self.eye_index].astype(np.float32)
        self.texture = ctx.texture(self.base.shape[1::-1], 3)
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.anisotropy = 8.0
        self._uploaded = None

    def face_on_screen(self):
        """Screen position of the cat's face center and its width in screen px."""
        return self.face_centre, self.face_width

    def _upload(self, sprite: np.ndarray):
        texture = self.base
        if sprite is not self.open_sprite:
            texture = self.base.copy()
            flat = texture.reshape(-1, 3)
            closed = sample(np.ascontiguousarray(sprite[..., :3]), self.eye_px).astype(np.float32)
            eyes = self.eye_base + self.eye_weight * (closed - self.eye_open)
            flat[self.eye_index] = np.clip(eyes, 0, 255).astype(np.uint8)
        # BGR like the rest of the app; OpenGL's first row is the bottom one.
        self.texture.write(np.ascontiguousarray(texture[::-1]).tobytes())
        self.texture.build_mipmaps()
        self._uploaded = sprite

    def _shape(self, landmark_moves: np.ndarray, ears) -> np.ndarray:
        """Rest vertices moved by the landmarks' movement (portrait px) and the ear angles."""
        moves = self.follow @ landmark_moves
        if any(ears):
            here = np.vstack([np.zeros((self.face_count, 2)), self.photo_px + moves[:, :2]])
            moves[:, :2] += (bend(here, self.ears, ears) - here)[self.face_count:]
        return self.rest + moves @ self.to_model.T

    def _draw_mouth(self, landmark_moves: np.ndarray, offsets: np.ndarray, rotation: np.ndarray,
                    model_rot: np.ndarray):
        lips = self.landmarks[LIPS] + landmark_moves[LIPS] @ self.to_model.T
        clip = np.column_stack([lips @ model_rot.T + self.shift, np.ones(len(LIPS))]) @ self.proj.T
        self.mouth_pos.write((clip[:, :3] / clip[:, 3:]).astype("f4").tobytes())
        if self.photo_mouth:
            # Per lip point: along the mouth, face widths below the upper lip, the gap there.
            lip_px = self.lip_rest + landmark_moves[LIPS, :2]
            n = len(UPPER_INNER_LIP)
            gaps = np.maximum(lip_px[n:, 1] - lip_px[:n, 1], 0) / self.lip_width
            across = np.linspace(0, 1, n)
            attr = np.vstack([np.column_stack([across, np.zeros(n), gaps]),
                              np.column_stack([across, gaps, gaps])])
            self.mouth_photo.use(0)
        else:
            flat = self.flat
            flat._update_mouth(flat._shape(offsets), rotation)  # its uniforms and per-lip values
            attr = flat.mouth[LIPS]
        self.mouth_attr.write(attr.astype("f4").tobytes())
        self.mouth_vao.render()

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray,
               ears=(0.0, 0.0)) -> np.ndarray:
        """Draw the head with face offsets (478, 3) portrait px, head rotation (image
        axes, as for CatMesh) and ear angles; returns BGR. sprite: the portrait with
        its eyelids drawn as blinking (or the open-eyed one)."""
        if sprite is not self._uploaded:
            self._upload(sprite)
        landmark_moves = self.flat._steady_eyes(offsets)
        self.vbo.write(self._shape(landmark_moves, ears).astype("f4").tobytes())
        model_rot = FLIP @ rotation @ FLIP
        self.prog["rot"].write(model_rot.astype("f4").T.tobytes())
        r = self.renderer
        r.msaa.use()
        r.ctx.clear(*self.bg, depth=1.0)
        r.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        r.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._draw_mouth(landmark_moves, offsets, rotation, model_rot)
        self.texture.use(0)
        self.vao.render()
        r.ctx.copy_framebuffer(r.resolved, r.msaa)
        data = r.resolved.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.out_h, self.out_w, 3)[::-1]
