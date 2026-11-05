"""A cat drawn as a real 3D head model instead of the warped photo.

The head is a textured mesh (assets/cats/<Name>/head3d.npz + head3d.png),
made from the cat's portrait by an image-to-3D model and imported with
catcam.head3d_import, which also records where every vertex sits in the
portrait. That is what lets it borrow what the 2D cat does: each vertex
moves with the face landmarks around it (expressions, the jaw), the eye areas
of the texture blink with the portrait's eyelids, and the ears bend about the
same hinges. The lips part over a real 3D mouth
(catcam.mouth3d) whose jaw swings open with your lower lip. Head3D stands in
for CatMesh: same render() and face_on_screen().
"""

import cv2
import moderngl
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from catcam import mouth3d
from catcam.cat_mesh import CHIN_LINE, TOP_MARGIN, CatMesh, Renderer, _outline
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
# Face widths: only skin this close to a face landmark (in 3D) counts as the lips; the back
# of the head lines up with the mouth in the portrait too.
ON_FACE = 0.1
DEPTH_SPAN = 1.5  # model units in front of / behind the head kept in the depth range
# Image axes (x right, y down, z away from the camera) <-> model axes (y up, z toward it).
FLIP = np.diag([1.0, -1.0, -1.0])
MOUTH_CORNERS = (UPPER_INNER_LIP[0], UPPER_INNER_LIP[-1])
LIP_MIDDLE = (13, 14)  # where the upper and lower inner lips meet in the middle
JAW_MAX = 1.0  # radians, the most the jaw tilts; beyond that it just drops with the lip
BREATH_RISE = 0.008  # head heights a full breath lifts the head by
# A head with its own jaw (see catcam.head3d_import) opens it about a hinge this
# far from the middle of its lips (mouth widths: right, up, out of the mouth), at
# most OWN_JAW_MAX radians.
JAW_HINGE = (0.0, 0.1, -1.35)
OWN_JAW_MAX = 0.5
OWN_MOUTH_INSIDE = (0.16, 0.06, 0.06)  # RGB of a head's own inside, seen between its parted lips

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
out vec3 normal;
void main() {
    gl_Position = proj * vec4(rot * in_pos + shift, 1.0);
    uv = in_uv;
    normal = rot * in_normal;
    // The texture already has the light of the photo it came from baked in; only the
    // change in facing since then brightens or darkens it (see catcam.lighting).
    float before = clamp(dot(in_normal, light), 0.0, 1.0);
    float after = clamp(dot(rot * in_normal, light), 0.0, 1.0);
    shade = clamp(1.0 + strength * (after - before), shade_range.x, shade_range.y);
}
"""
# Whether screen px p is inside the opening between the lips (a polygon, crossing test).
IN_OPENING = """
uniform vec2 opening[22];  // the gap between the lips on screen (px), around its edge
bool in_opening(vec2 p) {
    bool inside = false;
    for (int i = 0, j = 21; i < 22; j = i++) {
        vec2 a = opening[i], b = opening[j];
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}
"""
FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
uniform bool seam;       // drawing the skin across the lips, which gives way where they part
uniform bool own_mouth;  // nothing added inside: the head's own inside shows between the lips...
uniform vec3 inside;     // ...in this colour (BGR)
in vec2 uv;
in float shade;
in vec3 normal;
out vec4 color;
""" + IN_OPENING + """
void main() {
    if (own_mouth && !gl_FrontFacing) {
        // Darker where the inside turns away from the camera, so it reads as a hollow.
        color = vec4(inside * (0.35 + 0.65 * abs(normalize(normal).z)), 1.0);
        return;
    }
    if (seam && in_opening(gl_FragCoord.xy)) discard;
    color = vec4(texture(tex, uv).rgb * shade, 1.0);
}
"""
# The mouth parts (catcam.mouth3d): the jaw half swings about the hinge, the rest
# moves with the upper lip; lit by a light from the camera's side, darker deep inside.
MOUTH_VERTEX_SHADER = """
#version 330
uniform mat4 proj;
uniform mat3 rot;
uniform vec3 shift;
uniform mat4 skull;  // mouth frame -> model, moving with the upper lip
uniform mat4 jaw;    // the same with the jaw opened
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
in float in_jaw;
in float in_light;
out vec3 colour;
out vec3 normal;
out float light;
void main() {
    vec3 p = mix((skull * vec4(in_pos, 1.0)).xyz, (jaw * vec4(in_pos, 1.0)).xyz, in_jaw);
    vec3 n = mix(mat3(skull) * in_normal, mat3(jaw) * in_normal, in_jaw);
    gl_Position = proj * vec4(rot * p + shift, 1.0);
    normal = rot * n;
    colour = in_colour;
    light = in_light;
}
"""
MOUTH_FRAGMENT_SHADER = """
#version 330
in vec3 colour;
in vec3 normal;
in float light;
uniform bool clip;        // only seen through the opening between the lips...
uniform sampler2D skin;   // ...or else never in front of the head's outside (its depth)
out vec4 color;
""" + IN_OPENING + """
void main() {
    // Only seen through the mouth: parts that would poke out through the chin or
    // cheeks (which slope back faster than the mouth inside) stay hidden. A head with
    // its own mouth really parts its lips: there, what pokes out in front of its skin
    // goes, and its skin hides the rest.
    if (clip) {
        if (!in_opening(gl_FragCoord.xy)) discard;
    } else {
        float outside = texelFetch(skin, ivec2(gl_FragCoord.xy), 0).r;
        if (outside < 1.0 && gl_FragCoord.z < outside) discard;
    }
    float facing = max(dot(normalize(normal), normalize(vec3(0.2, 0.4, 1.0))), 0.0);
    color = vec4((colour * light * (0.55 + 0.45 * facing)).bgr, 1.0);  // BGR like the app
}
"""


def files(name: str):
    return ASSETS / name / "head3d.npz", ASSETS / name / "head3d.png"


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


def lip_line(points):
    """y of the line between the lips (portrait px) at given x."""
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    order = np.argsort(seam[:, 0])
    return lambda x: np.interp(x, seam[order, 0], seam[order, 1])


def lip_sides(photo_px, points, face_width):
    """For each vertex: whether it is near the mouth, below the line between the lips,
    and between the mouth corners (all in the portrait), and for each landmark whether
    it belongs below the line."""
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    line_y = lip_line(points)
    left, right = seam[:, 0].min(), seam[:, 0].max()
    reach_x, reach_y = (r * face_width for r in MOUTH_ZONE)
    x, y = photo_px[:, 0], photo_px[:, 1]
    close = np.abs(y - line_y(x)) < reach_y
    near_mouth = close & (x > left - reach_x) & (x < right + reach_x)
    inside = close & (x > left) & (x < right)
    below = points[:, 1] > line_y(points[:, 0])
    below[UPPER_INNER_LIP], below[LOWER_INNER_LIP] = False, True
    return near_mouth, y > line_y(x), inside, below


def follow_weights(vertices, photo_px, landmarks, points, face_model, face_width,
                   on_jaw=None) -> csr_matrix:
    """(vertices, landmarks) weights: how each vertex moves with the face landmarks.
    Closeness is measured on the model (landmarks: their spots on it; face_model: a
    face width in model units), so the back of the head stays put; which side of the
    lips a vertex is on is decided in the portrait (photo_px, points, face_width px),
    or for a head with its own jaw by whether it is on the jaw (on_jaw)."""
    tree = cKDTree(landmarks)
    dist, nearest = tree.query(vertices, k=NEAREST)
    weights = np.exp(-((dist / (FOLLOW * face_model)) ** 2))
    # Lips: keep each vertex to the landmarks on its side of the line between them.
    near_mouth, vertex_below, _, below = lip_sides(photo_px, points, face_width)
    if on_jaw is None:
        weights[near_mouth[:, None] & (below[nearest] != vertex_below[:, None])] = 0
    else:  # below the lips only the jaw moves, about its hinge; the skin around it goes along with it
        weights[below[nearest] != on_jaw[:, None]] = 0
    total = weights.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0)
    falloff = np.exp(-((dist[:, 0] / (FALLOFF * face_model)) ** 2))
    weights *= falloff[:, None]
    rows = np.repeat(np.arange(len(photo_px)), NEAREST)
    return csr_matrix((weights.ravel(), (rows, nearest.ravel())), shape=(len(photo_px), len(points)))


def lip_seam(faces, photo_px, points, face_width, on_face) -> np.ndarray:
    """Which faces bridge the lips between the mouth corners: they give way where the
    mouth opens. on_face: which vertices are on the face (not behind it)."""
    _, vertex_below, inside, _ = lip_sides(photo_px, points, face_width)
    sides = vertex_below[faces]
    lips = (inside & on_face)[faces].all(axis=1)
    return lips & (sides.any(axis=1) != sides.all(axis=1))


def mouth_frame(landmarks, facing):
    """Origin (where the lips meet in the middle), axes (columns: toward the image-right
    corner, up, out of the mouth) and width of the mouth, in model units; facing: the
    direction the face looks in, in the model."""
    left, right = landmarks[MOUTH_CORNERS[0]], landmarks[MOUTH_CORNERS[1]]
    origin = (landmarks[LIP_MIDDLE[0]] + landmarks[LIP_MIDDLE[1]]) / 2
    x = (right - left) / np.linalg.norm(right - left)
    z = facing - (facing @ x) * x
    z /= np.linalg.norm(z)
    return origin, np.column_stack([x, np.cross(z, x), z]), np.linalg.norm(right - left)


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
        self.bg = tuple(np.asarray(bg_color) / 255) + (0.0,)  # alpha 0: not the animal

        centre = (mesh["vertices"].min(0) + mesh["vertices"].max(0)) / 2
        self.rest = mesh["vertices"] - centre
        self.landmarks = mesh["landmarks"] - centre
        self.photo_px = mesh["photo_px"]
        # A movement in portrait px (x right, y down, z away) as a movement of the model.
        self.to_model = (mesh["cam_rot"].T @ np.diag([1, -1, -1]) / mesh["cam_scale"]).astype(np.float64)
        face_width = np.ptp(points[:, 0])
        jaw = mesh["jaw_weight"] if "jaw_weight" in mesh.files else np.zeros(0)
        self.jaw_weight = jaw.astype(np.float64)[:, None] if len(jaw) else None
        self.follow = follow_weights(self.rest, self.photo_px, self.landmarks, points,
                                     face_width / float(mesh["cam_scale"]), face_width,
                                     None if self.jaw_weight is None else jaw > 0.5)
        outline = _outline(sprite[..., 3])
        self.ears = find_ears(np.vstack([points[:, :2], self.photo_px]), outline, len(points), face_width)
        self.face_count = len(points)

        # Frame it like CatMesh: top of the ears at TOP_MARGIN, chin at CHIN_LINE.
        height = np.ptp(self.rest[:, 1])
        half = np.tan(np.radians(FOV) / 2)
        distance = height / ((CHIN_LINE - TOP_MARGIN) * 2 * half)
        middle_ndc = 1 - (TOP_MARGIN + CHIN_LINE)
        self.shift = np.array([0.0, middle_ndc * distance * half, -distance])
        self.lift = BREATH_RISE * height  # model units, at a full breath
        self.moved = self.shift  # where the head is this frame (it rises as it breathes)
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
        face_model = face_width / float(mesh["cam_scale"])
        on_face = cKDTree(self.landmarks).query(self.rest)[0] < ON_FACE * face_model
        attributes = [(self.vbo, "3f", "in_pos"),
                      (ctx.buffer(mesh["normals"].astype("f4").tobytes()), "3f", "in_normal"),
                      (ctx.buffer(mesh["uv"].astype("f4").tobytes()), "2f", "in_uv")]
        if self.jaw_weight is None:
            seam = lip_seam(mesh["faces"], self.photo_px, points, face_width, on_face)
        else:  # the jaw is cut free already (see catcam.head3d_import.cut_jaw)
            seam = np.zeros(len(mesh["faces"]), bool)
        self.vao, self.seam_vao = (
            ctx.vertex_array(self.prog, attributes, ctx.buffer(mesh["faces"][part].astype("i4").tobytes()))
            if part.any() else None for part in (~seam, seam))

        # The mouth: real teeth, tongue and cavity behind lips that part, unless the
        # head keeps its own (imported with --own-mouth): then only its inside shows.
        self.own_mouth = bool(mesh["own_mouth"]) if "own_mouth" in mesh.files else False
        self.prog["own_mouth"].value = self.own_mouth
        self.prog["inside"].value = OWN_MOUTH_INSIDE[::-1]
        # The portrait's camera looks straight at the face: its axis is where the face looks.
        origin, axes, width = mouth_frame(self.landmarks, mesh["cam_rot"][2].astype(np.float64))
        if "slit_middle" in mesh.files:  # its own lips: the mouth sits where they really meet
            origin, width = mesh["slit_middle"] - centre, float(mesh["slit_width"])
        self.mouth_origin, self.mouth_axes, self.mouth_width = origin, axes, width
        # The portrait's lips rarely meet exactly; the mouth opens as far as they part
        # from there, so a closed mouth shows no gap.
        half_gap = (self.landmarks[LOWER_INNER_LIP] - self.landmarks[UPPER_INNER_LIP]) / 2
        self.lips_meet = np.vstack([half_gap, -half_gap[::-1]])
        parts = mouth3d.build(flat.lip_tint, cavity=not self.own_mouth)
        self.mouth_prog = ctx.program(vertex_shader=MOUTH_VERTEX_SHADER,
                                      fragment_shader=MOUTH_FRAGMENT_SHADER)
        self.mouth_prog["clip"].value = not self.own_mouth
        if self.own_mouth:  # the depth of the head's outside, for the mouth to stay within
            self.skin = ctx.depth_texture((self.out_w, self.out_h))
            self.skin.compare_func = ""
            self.skin_fbo = ctx.framebuffer(depth_attachment=self.skin)
            self.mouth_prog["skin"].value = 1
        self.mouth_prog["proj"].write(self.proj.astype("f4").T.tobytes())
        self.mouth_prog["shift"].value = tuple(self.shift)
        data = np.column_stack([parts.pos, parts.normal, parts.colour, parts.jaw, parts.light]).astype("f4")
        self.mouth_vao = ctx.vertex_array(
            self.mouth_prog,
            [(ctx.buffer(data.tobytes()), "3f 3f 3f 1f 1f", "in_pos", "in_normal", "in_colour", "in_jaw",
              "in_light")],
            ctx.buffer(parts.faces.astype("i4").tobytes()))

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
        """Rest vertices moved by the landmarks' movement (portrait px) and the ear
        angles, and swung open with the jaw if it is marked."""
        moves = self.follow @ landmark_moves
        if any(ears):
            here = np.vstack([np.zeros((self.face_count, 2)), self.photo_px + moves[:, :2]])
            moves[:, :2] += (bend(here, self.ears, ears) - here)[self.face_count:]
        shaped = self.rest + moves @ self.to_model.T
        if self.jaw_weight is None:
            return shaped
        swung = self._swing(self.rest, landmark_moves)
        return shaped + self.jaw_weight * (swung - shaped)

    def _own_jaw(self, landmark_moves: np.ndarray) -> np.ndarray:
        """Model -> model (4x4) for the head's own jaw: turned about its hinge as far
        as your lower lip has dropped, and moved with the upper lip."""
        moved = landmark_moves[list(LIP_MIDDLE)] @ self.to_model.T
        drop = (moved[1] - moved[0]) @ self.mouth_axes[:, 1]  # the lower lip's, down is negative
        hinge = np.array(JAW_HINGE) * self.mouth_width
        angle = min(np.arcsin(np.clip(-drop / np.linalg.norm(hinge[1:]), 0, 1)), OWN_JAW_MAX)
        c, s = np.cos(angle), np.sin(angle)
        turn = self.mouth_axes @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ self.mouth_axes.T
        pivot = self.mouth_origin + self.mouth_axes @ hinge
        jaw = np.eye(4)
        jaw[:3, :3] = turn
        jaw[:3, 3] = pivot - turn @ pivot + moved[0]
        return jaw

    def _swing(self, points: np.ndarray, landmark_moves: np.ndarray) -> np.ndarray:
        """points (model, at rest) where the head's own jaw puts them."""
        jaw = self._own_jaw(landmark_moves)
        return points @ jaw[:3, :3].T + jaw[:3, 3]

    def _mouth_matrices(self, landmark_moves: np.ndarray):
        """Mouth frame -> model, for the parts moving with the upper lip and with the jaw."""
        moved = landmark_moves[list(LIP_MIDDLE)] @ self.to_model.T
        follow = moved[0]  # everything moves with the upper lip
        drop = (moved[1] - moved[0]) @ self.mouth_axes[:, 1]  # the lower lip's, down is negative
        hinge = np.array(mouth3d.HINGE)
        reach = np.linalg.norm(hinge[1:]) * self.mouth_width  # hinge to the front teeth
        angle = min(np.arcsin(np.clip(-drop / reach, 0, 1)), JAW_MAX)
        skull = np.eye(4)
        skull[:3, :3] = self.mouth_axes * self.mouth_width
        skull[:3, 3] = self.mouth_origin + follow
        if self.jaw_weight is not None:  # the parts in the jaw go with the head's own
            at_rest = skull.copy()
            at_rest[:3, 3] = self.mouth_origin
            return skull, self._own_jaw(landmark_moves) @ at_rest
        c, s = np.cos(angle), np.sin(angle)
        swing = np.eye(4)
        swing[1:3, 1:3] = [[c, -s], [s, c]]
        to_hinge, back = np.eye(4), np.eye(4)
        to_hinge[:3, 3], back[:3, 3] = -hinge, hinge
        jaw = skull @ back @ swing @ to_hinge
        # Pinned to the lower lip: its front teeth stay right behind it however wide it opens.
        lower_lip = self.landmarks[LIP_MIDDLE[1]] + moved[1]
        jaw[:3, 3] += lower_lip - jaw[:3, 3]
        return skull, jaw

    def _draw_mouth(self, landmark_moves: np.ndarray, model_rot: np.ndarray):
        if not self.own_mouth:
            # The gap between the lips on screen: upper lip left to right, lower lip back.
            ring = UPPER_INNER_LIP + LOWER_INNER_LIP[::-1]
            lips = self.landmarks[ring] + self.lips_meet + landmark_moves[ring] @ self.to_model.T
            clip = np.column_stack([lips @ model_rot.T + self.moved, np.ones(len(ring))]) @ self.proj.T
            ndc = clip[:, :2] / clip[:, 3:]
            screen = (ndc + 1) / 2 * (self.out_w, self.out_h)  # gl_FragCoord: px from the bottom left
            opening = [tuple(p) for p in screen]
            self.prog["opening"].value = opening
            self.mouth_prog["opening"].value = opening
        skull, jaw = self._mouth_matrices(landmark_moves)
        self.mouth_prog["rot"].write(model_rot.astype("f4").T.tobytes())
        self.mouth_prog["skull"].write(skull.astype("f4").T.tobytes())
        self.mouth_prog["jaw"].write(jaw.astype("f4").T.tobytes())
        self.mouth_vao.render()

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray,
               ears=(0.0, 0.0), breath: float = 0.0) -> np.ndarray:
        """Draw the head with face offsets (478, 3) portrait px, head rotation (image
        axes, as for CatMesh), ear angles and breath (0..1, how full its lungs are,
        see catcam.idle); returns BGR. sprite: the portrait with its eyelids drawn
        as blinking (or the open-eyed one)."""
        if sprite is not self._uploaded:
            self._upload(sprite)
        landmark_moves = self.flat._steady_eyes(offsets)
        self.vbo.write(self._shape(landmark_moves, ears).astype("f4").tobytes())
        model_rot = FLIP @ rotation @ FLIP
        self.prog["rot"].write(model_rot.astype("f4").T.tobytes())
        self.moved = self.shift + (0.0, self.lift * breath, 0.0)
        self.prog["shift"].value = tuple(self.moved)
        self.mouth_prog["shift"].value = tuple(self.moved)
        r = self.renderer
        r.ctx.enable(moderngl.DEPTH_TEST)
        r.ctx.disable(moderngl.BLEND)
        self.texture.use(0)
        if self.own_mouth:
            self.skin_fbo.use()
            self.skin_fbo.clear(depth=1.0)
            r.ctx.enable(moderngl.CULL_FACE)
            self.prog["seam"].value = False
            self.vao.render()
            r.ctx.disable(moderngl.CULL_FACE)
            self.skin.use(1)
        r.msaa.use()
        r.ctx.clear(*self.bg, depth=1.0)
        self._draw_mouth(landmark_moves, model_rot)
        self.prog["seam"].value = False
        self.vao.render()
        if self.seam_vao is not None:
            self.prog["seam"].value = True
            self.seam_vao.render()
        r.ctx.copy_framebuffer(r.resolved, r.msaa)
        return r.read()
