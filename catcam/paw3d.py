"""Real 3D paws on your hands, drawn with OpenGL.

The paws are generated models without a texture (assets/paws/cat.npz and
dog.npz, made by tools/paw_model.py): fur, pads and a piece of forearm, in paw
widths, its toes rigged to bend about their knuckles. Each animal colours them
from its own portrait: the arm takes the fur of its cheeks, the paw the
lighter fur of its chin, tabbies get stripes when their fur is finely
patterned, and the pads take the colour of its nose (pink, brick, black). One
model serves both hands: your left hand gets it as it is, your right hand
mirrored. The paw turns in 3D with your hand (palm or back to the camera, and
everything in between), and each toe bends with its finger, a finger raised
out of a fist standing a little taller, as catcam.paws does for the photo
paws. The arm is its piece repeated down, mirrored back and forth, so it
always runs off-screen. Light comes from the same side as on the heads (see
catcam.lighting).
"""

from functools import cache
from typing import NamedTuple

import cv2
import moderngl
import numpy as np

from catcam.cat_mesh import MSAA_SAMPLES, Renderer
from catcam.lighting import LIGHT
from catcam.paths import PAWS
from catcam.paws import STAND, Placement, _fur_color, paw_style

CURL_ANGLE = np.radians(65)  # a fully bent finger turns its toe this far about the knuckle, toward the sole
RAISE = 0.2  # toe lengths a straight toe stands taller while the other fingers make a fist (see STAND)
ARM_PIECES = 24  # copies of the model's arm piece down from the paw: enough to always leave the screen
DEPTH = 40.0  # paw widths in front of and behind the palm that fit in the depth range
AMBIENT = 0.5  # share of the light that reaches everywhere; the rest falls by facing the light
PAD_SHINE = 0.35  # how much the pads (smooth skin) glint where they face the light
PAD_EDGE = (0.3, 0.7)  # pad mask (see tools/paw_model.py) from fur at [0] to pad at [1]
# The paw (down to SOCK[0] paw widths below the palm) has the sock colour, the arm (from
# SOCK[1] down) the arm colour, blended in between. The sock is the arm's fur gone
# SOCK_SHARE of the way to the chin's: lighter, but still the animal's colour.
SOCK = (-0.3, -1.0)
SOCK_SHARE = 0.35
GRAIN = 0.12  # brightness wobble of the fur, so tufts read apart from each other
# Stripes: tabbies have finely patterned fur. That pattern's strength (the fur's
# brightness off its blurred self over its brightness, a median over the forehead and
# cheeks) gives no stripes at TABBY[0], full ones at [1]. Stripes ring the arm every
# STRIPE_SPACING paw widths, wobbling by STRIPE_WOBBLE of that, in the fur's darkest tones.
TABBY = (0.065, 0.09)
TABBY_BLUR = 0.02  # face widths: the pattern is what blurring this much takes away
STRIPE_SPACING, STRIPE_WOBBLE = 0.32, 0.35
# Where to look on the portrait (MediaPipe face landmarks), and how far round (face widths).
FOREHEAD_CHEEKS = [10, 151, 9, 108, 337, 67, 297, 109, 338, 234, 454, 93, 323, 132, 361]
FUR_REACH = 0.06
# The nose: the line from the tip of the nose (4) down to the upper lip (0) and this far
# round it, in face widths; the pads take the colour of the part of it least like the fur
# (that percentile and up), the nose leather rather than the fur around it.
NOSE_LINE, NOSE_REACH, NOSE_PICK = (4, 0), 0.06, 80

VERTEX_SHADER = """
#version 330
uniform mat3 place;   // paw frame -> screen px, columns: thumb side, up, back of the paw (z away)
uniform vec3 at;      // the middle of the palm on screen, px
uniform vec2 screen;  // output size, px
uniform float depth;  // px in front of and behind the palm that fit in the depth range
uniform vec4 bend;    // each toe's turn about its knuckle (radians, about x, toward the sole)...
uniform vec4 lift;    // ...how far it stands taller (paw widths, along y)...
uniform vec2 knuckle[4];  // ...and its knuckle (y, z), in the paw frame
uniform bool arm;     // drawing the arm: one piece per instance, mirrored back and forth
uniform vec2 piece;   // the arm piece's top and length, paw widths
in vec3 in_pos;
in vec3 in_normal;
in vec4 in_toes;
in vec3 in_colour;
in float in_light;
in float in_shine;
out vec3 colour;
out vec3 normal;
out float light;
out float shine;
void main() {
    vec3 p = in_pos;
    vec3 n = in_normal;
    if (arm) {
        if (gl_InstanceID % 2 == 1) {
            p.y = 2.0 * piece.x - piece.y - p.y;
            n.y = -n.y;
        }
        p.y -= float(gl_InstanceID) * piece.y;
    } else {
        // The toes all bend about x, so a vertex shared by two of them turns by the mix
        // of their angles about the mix of their knuckles: it bends, it doesn't collapse.
        float share = dot(in_toes, vec4(1.0));
        if (share > 0.0) {
            vec2 hinge = vec2(0.0);
            for (int i = 0; i < 4; i++) hinge += in_toes[i] * knuckle[i];
            hinge /= share;
            float a = dot(in_toes, bend);
            mat2 turn = mat2(cos(a), sin(a), -sin(a), cos(a));
            p.yz = hinge + turn * (p.yz - hinge);
            p.y += dot(in_toes, lift);
            n.yz = turn * n.yz;
        }
    }
    vec3 s = at + place * p;
    gl_Position = vec4(2.0 * s.x / screen.x - 1.0, 1.0 - 2.0 * s.y / screen.y, s.z / depth, 1.0);
    normal = place * n;
    colour = in_colour;
    light = in_light;
    shine = in_shine;
}
"""
FRAGMENT_SHADER = """
#version 330
uniform vec3 light_dir;  // toward the light, screen axes (z away from the camera)
uniform float ambient;
uniform float pad_shine;
uniform float opacity;
in vec3 colour;
in vec3 normal;
in float light;
in float shine;
out vec4 color;
void main() {
    vec3 n = normalize(normal);
    if (n.z > 0.0) n = -n;  // whichever side of a tuft faces the camera
    float facing = 0.5 + 0.5 * dot(n, light_dir);  // wrapped round, soft as fur is
    vec3 lit = colour * light * (ambient + (1.0 - ambient) * facing);
    vec3 glint = reflect(-light_dir, n);
    lit += pad_shine * shine * pow(max(-glint.z, 0.0), 12.0);
    color = vec4(lit * opacity, opacity);  // premultiplied, for laying over the frame
}
"""


class PawModel(NamedTuple):
    """A rigged paw (see tools/paw_model.py), paw widths, origin in the middle of the palm."""
    pos: np.ndarray
    normal: np.ndarray
    faces: np.ndarray
    toes: np.ndarray  # (n, 4) how much each vertex bends with each toe, thumb side first
    pad: np.ndarray  # 0 (fur) .. 1 (pad)
    light: np.ndarray  # 0..1, darker where sunk between tufts and toes
    arm_pos: np.ndarray
    arm_normal: np.ndarray
    arm_faces: np.ndarray
    arm_pad: np.ndarray
    arm_light: np.ndarray
    arm_cut: float  # the arm piece hangs from this far below the palm...
    arm_piece: float  # ...and is this long
    knuckles: np.ndarray  # (4, 2) each toe's knuckle: height and depth (y, z)
    toe_length: float


class PawColours(NamedTuple):
    """BGR 0..255."""
    arm: tuple
    sock: tuple
    stripes: tuple  # the fur's darkest tones
    tabby: float  # 0..1 how strongly the stripes show
    pad: tuple


def model_file(animal: str):
    return PAWS / f"{animal}.npz"


def has_model(animal: str) -> bool:
    return model_file(animal).is_file()


@cache
def load(animal: str) -> PawModel:
    with np.load(model_file(animal)) as data:
        fields = {field: data[field] for field in PawModel._fields}
    return PawModel(**{field: value.item() if value.ndim == 0
                       else value.astype(np.float32) if value.dtype.kind == "f" else value
                       for field, value in fields.items()})


def _near_line(shape, points: np.ndarray, line, reach: float) -> np.ndarray:
    """Mask (uint8) of the pixels within reach of the segment between two landmarks."""
    mask = np.zeros(shape, np.uint8)
    for t in np.linspace(0, 1, 8):
        x, y = (1 - t) * points[line[0], :2] + t * points[line[1], :2]
        cv2.circle(mask, (int(x), int(y)), max(int(reach), 1), 255, -1)
    return mask


def paw_colours(sprite: np.ndarray, points: np.ndarray) -> PawColours:
    """An animal's paw colours, sampled from its portrait (BGRA) with its landmarks
    ((478, 2+) sprite px)."""
    style = paw_style(sprite, points)
    face_width = np.ptp(points[:, 0])
    solid = np.where(sprite[..., 3] > 200, 255, 0).astype(np.uint8)
    bgr = sprite[..., :3]

    nose = _near_line(sprite.shape[:2], points, NOSE_LINE, NOSE_REACH * face_width) & solid
    px = bgr[nose > 0]
    if len(px):
        lab = cv2.cvtColor(px[None], cv2.COLOR_BGR2LAB)[0].astype(float)
        fur = cv2.cvtColor(np.uint8([[style.sock]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        unlike = np.linalg.norm(lab - fur, axis=1)
        pad = tuple(int(c) for c in np.median(px[unlike >= np.percentile(unlike, NOSE_PICK)], axis=0))
    else:
        pad = style.sock

    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(luma, (0, 0), TABBY_BLUR * face_width)
    fur = np.zeros(sprite.shape[:2], np.uint8)
    for x, y in points[FOREHEAD_CHEEKS, :2]:
        cv2.circle(fur, (int(x), int(y)), max(int(FUR_REACH * face_width), 1), 255, -1)
    fur = (fur & solid) > 0
    if fur.any():
        pattern = float(np.median(np.abs(luma - blurred)[fur] / (blurred[fur] + 20)))
        tabby = float(np.clip((pattern - TABBY[0]) / (TABBY[1] - TABBY[0]), 0, 1))
    else:
        tabby = 0.0
    stripes = _fur_color(sprite, points[FOREHEAD_CHEEKS, :2], FUR_REACH * face_width, 10)
    sock = tuple(int(a + SOCK_SHARE * (c - a)) for a, c in zip(style.arm, style.sock))
    return PawColours(style.arm, sock, stripes, tabby, pad)


def _noise(pos: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Smooth-ish value noise in -1..1 at positions (n, 3): a few random waves."""
    rng = np.random.default_rng(seed)
    waves = rng.normal(size=(6, 3)) / scale
    phases = rng.uniform(0, 2 * np.pi, 6)
    return np.sin(pos @ waves.T + phases).mean(axis=1) * np.sqrt(2)


def vertex_colours(pos: np.ndarray, pad: np.ndarray, colours: PawColours) -> np.ndarray:
    """(n, 3) BGR 0..1 per vertex: fur (sock on the paw, arm colour up the arm, tabby
    stripes ringing it, a grain of light and dark tufts) with the pads in their colour."""
    y = pos[:, 1]
    sock = np.clip((y - SOCK[1]) / (SOCK[0] - SOCK[1]), 0, 1)[:, None]
    fur = (1 - sock) * np.array(colours.arm, float) + sock * np.array(colours.sock, float)
    ring = y + STRIPE_WOBBLE * STRIPE_SPACING * _noise(pos, 0.25, 1)
    band = np.clip(1.5 * np.sin(2 * np.pi * ring / STRIPE_SPACING), 0, 1)[:, None]
    fur += colours.tabby * band * (np.array(colours.stripes, float) - fur)
    fur *= 1 + GRAIN * _noise(pos, 0.03, 2)[:, None]
    t = np.clip((pad - PAD_EDGE[0]) / (PAD_EDGE[1] - PAD_EDGE[0]), 0, 1)[:, None]
    t = t * t * (3 - 2 * t)
    return np.clip(((1 - t) * fur + t * np.array(colours.pad, float)) / 255, 0, 1)


def toe_bends(model: PawModel, curls: np.ndarray):
    """Each toe's (4,) turn about its knuckle (radians about x; negative turns the tip
    toward the sole) by its curl, and how far it stands taller (paw widths) while
    straight with the other fingers curled: a finger raised out of a fist."""
    curls = np.clip(np.asarray(curls, float), 0, 1)
    others = (curls.sum() - curls) / (len(curls) - 1)
    stands = np.clip((others - STAND[0]) / (STAND[1] - STAND[0]), 0, 1)
    return -CURL_ANGLE * curls, RAISE * model.toe_length * (1 - curls) * stands


class _Target(NamedTuple):
    """What every animal's paws draw with: the shaders and an off-screen frame (drawn
    with multisampling, then resolved) to lay over the picture."""
    prog: moderngl.Program
    fbo: moderngl.Framebuffer
    resolved: moderngl.Framebuffer


@cache
def _target(renderer: Renderer) -> _Target:
    ctx = renderer.ctx
    size = (renderer.out_w, renderer.out_h)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    prog["screen"].value = size
    prog["light_dir"].value = tuple(LIGHT)
    prog["ambient"].value = AMBIENT
    prog["pad_shine"].value = PAD_SHINE
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, 4, samples=MSAA_SAMPLES)],
                          depth_attachment=ctx.depth_renderbuffer(size, samples=MSAA_SAMPLES))
    return _Target(prog, fbo, ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, 4)]))


class Paw3D:
    """One animal's 3D paws, drawn over frames by Paws.draw (see catcam.paws)."""

    def __init__(self, renderer: Renderer, model: PawModel, colours: PawColours):
        self.renderer, self.model = renderer, model
        self.target = _target(renderer)
        self.paw = self._vao(model.pos, model.normal, model.toes, model.pad, model.light, model.faces,
                             colours)
        self.arm = self._vao(model.arm_pos, model.arm_normal, np.zeros((len(model.arm_pos), 4)),
                             model.arm_pad, model.arm_light, model.arm_faces, colours)

    def _vao(self, pos, normal, toes, pad, light, faces, colours):
        data = np.column_stack([pos, normal, toes, vertex_colours(pos, pad, colours), light,
                                np.clip(pad, 0, 1)]).astype("f4")
        ctx = self.renderer.ctx
        return ctx.vertex_array(
            self.target.prog,
            [(ctx.buffer(data.tobytes()), "3f 3f 4f 3f 1f 1f",
              "in_pos", "in_normal", "in_toes", "in_colour", "in_light", "in_shine")],
            ctx.buffer(faces.astype("i4").tobytes()))

    def render(self, frame: np.ndarray, placements: list[Placement]) -> np.ndarray:
        """frame with the paws drawn over it (BGR, the renderer's size)."""
        if not placements:
            return frame
        ctx, (prog, fbo, resolved) = self.renderer.ctx, self.target
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        prog["piece"].value = (-self.model.arm_cut, self.model.arm_piece)
        prog["knuckle"].write(self.model.knuckles.astype("f4").tobytes())
        for place in placements:
            prog["place"].write((place.size * place.axes).astype("f4").T.tobytes())
            prog["at"].value = (place.x, place.y, 0.0)
            prog["depth"].value = DEPTH * place.size
            prog["opacity"].value = place.opacity
            bend, lift = toe_bends(self.model, place.curls)
            prog["bend"].value, prog["lift"].value = tuple(bend), tuple(lift)
            prog["arm"].value = False
            self.paw.render()
            prog["arm"].value = True
            self.arm.render(instances=ARM_PIECES)
        ctx.copy_framebuffer(resolved, fbo)
        h, w = frame.shape[:2]
        rgba = np.frombuffer(resolved.read(components=4), np.uint8).reshape(h, w, 4)[::-1]
        rows, cols = np.flatnonzero(rgba[..., 3].any(axis=1)), np.flatnonzero(rgba[..., 3].any(axis=0))
        if not len(rows):
            return frame
        box = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        out = frame.copy()
        paw = rgba[box].astype(np.float32)
        alpha = paw[..., 3:] / 255
        out[box] = np.clip(paw[..., :3] + out[box] * (1 - alpha), 0, 255).astype(np.uint8)
        return out


def for_animal(renderer: Renderer, animal: str, sprite: np.ndarray, points: np.ndarray) -> Paw3D | None:
    """The 3D paws for an animal ("cat", "dog") coloured from its portrait, or None
    without a model for it (the photo paws stand in)."""
    if not has_model(animal):
        return None
    return Paw3D(renderer, load(animal), paw_colours(sprite, points))
