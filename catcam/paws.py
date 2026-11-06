"""Cat paws that follow your hands.

Each tracked hand becomes one of the cat's paws, your left hand as the cat's
left paw. Animals with a 3D paw model get real 3D paws (catcam.paw3d) that turn
with your hand in 3D. Otherwise cats with paw photos (see catcam.paw_photos)
show the top of the paw when you show the back of your hand and the sole when
you show your palm, and cats without photos get a drawn socked paw with pink
toe beans, coloured like the cat: the arm takes the fur of its cheeks, the
sock the fur of its chin. Bending a finger curls the toe on its side of the
paw in, so a fist curls them all. A paw's position is taken relative to your
face, so a hand next to your chin puts the paw next to the cat's chin.
"""

from typing import NamedTuple

import cv2
import numpy as np

from catcam.head_pose import OneEuroFilter

PAW_SIZE = 1.1  # paw width relative to your palm width
PAW_MIN_CUTOFF, PAW_BETA = 1.5, 0.02
FADE_RATE = 0.35  # how fast paws appear / disappear
# A detected hand within this many paw sizes of where a still visible paw was
# is that same hand, whatever MediaPipe guesses (its left/right guess flips
# now and then, which would swap a paw for the tail or the other paw).
SAME_HAND = 2.5
TRACKED = 0.1  # a paw counts as still visible above this presence
ARM_LENGTH = 14  # in paw widths
DEFAULT_FACE_WIDTH = 0.25  # of the camera width, used until a face is seen
PHOTO_ARM_LENGTH = 8  # paw photos' arms are extended to this many paw widths
PALM_RANGE = (-0.25, 0.25)  # how squarely the palm faces the camera: back view up to [0], sole from [1]
# How much of your hand's depth (its tilt toward or away from the camera, which the
# hand tracker only guesses) a 3D paw copies: 0 keeps paws flat to the screen.
HAND_DEPTH = 0.7
# Fingertip-to-wrist distance over knuckle-to-wrist distance: about 2 for a
# straight finger, under 1 for one bent into a fist.
CLOSED_RANGE = (1.5, 1.0)  # straight at [0], fully bent at [1]
# The toes are the paw from the top down to TOE_LENGTH paw widths. A curled
# toe slides down into the paw by CURL of its height, keeping its shape (the
# fur KNUCKLE toe lengths above its base is squeezed instead), and is drawn in
# toward the paw's middle by NARROW. A straight toe stands up, up to RAISE toe
# lengths taller, as the other toes' mean curl goes over STAND (a raised
# finger in a fist): its tip moves up, opening the toe at KNUCKLE, and the
# gap is filled by the fur about FUR toe lengths below that, mirrored back and forth.
TOE_LENGTH = 0.5
CURL, NARROW = 0.5, 0.12
RAISE, STAND = 1.2, (0.5, 1.0)
KNUCKLE, FUR = 0.35, 0.2
# The four toes' middles, in paw widths from the paw's middle; each keeps its
# own length TOE_HALF either side of it.
TOE_X = (-0.32, -0.12, 0.12, 0.32)
TOE_HALF = 0.08

BEAN = (178, 165, 240)
OUTLINE_SHADE = 0.75  # toe outlines: the sock colour, darkened
# Face landmarks whose fur colours the paw, and which brightness of that fur
# (as a percentile) to use: chins sit in shadow, so the sock takes a light tone.
CHEEKS, ARM_BRIGHTNESS = [234, 454, 93, 323], 50
CHIN, SOCK_BRIGHTNESS = [152, 175, 199, 200, 205, 425, 36, 266], 80
SAMPLE_RADIUS = 0.03  # in face widths
ROUNDNESS = 0.3  # how much darker the arm and paw get toward their sides

WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
FINGERS = ((8, INDEX_MCP), (12, MIDDLE_MCP), (16, RING_MCP), (20, PINKY_MCP))  # (tip, knuckle)
VIEWS = ("front", "back")  # top of the paw (back of your hand), sole (palm)


class PawStyle(NamedTuple):
    arm: tuple[int, int, int]  # BGR
    sock: tuple[int, int, int]


class PawSprite(NamedTuple):
    image: np.ndarray  # BGRA, arm pointing down
    anchor: np.ndarray  # paw centre in the image
    width: int  # paw width in the image
    toes: float  # image row where the toes start (they reach up from here)


class Placement(NamedTuple):
    """Where and how a 3D paw goes this frame (see catcam.paw3d)."""
    x: float  # the middle of the palm on screen, px
    y: float
    size: float  # paw width, px
    axes: np.ndarray  # (3, 3) columns: toward the thumb, up the hand, out of its back (image axes, z away)
    curls: np.ndarray  # (4,) index to little finger: 0 straight .. 1 bent
    opacity: float


def _fur_color(sprite: np.ndarray, points: np.ndarray, radius: float, brightness: float):
    """Colour of the fur around points, at a brightness percentile."""
    mask = np.zeros(sprite.shape[:2], np.uint8)
    for x, y in points:
        cv2.circle(mask, (int(x), int(y)), max(int(radius), 1), 255, -1)
    mask &= np.where(sprite[..., 3] > 200, 255, 0).astype(np.uint8)
    fur = sprite[..., :3][mask > 0].astype(np.float64)
    if not len(fur):
        return (128, 128, 128)
    luma = fur @ (0.114, 0.587, 0.299)  # BGR
    lo, hi = np.percentile(luma, [max(brightness - 10, 0), min(brightness + 10, 100)])
    band = fur[(luma >= lo) & (luma <= hi)]
    return tuple(int(c) for c in band.mean(axis=0))


def paw_style(sprite: np.ndarray, landmarks: np.ndarray) -> PawStyle:
    """Paw colours sampled from the cat. landmarks: (478, 2+) sprite px."""
    radius = SAMPLE_RADIUS * np.ptp(landmarks[:, 0])
    return PawStyle(
        arm=_fur_color(sprite, landmarks[CHEEKS, :2], radius, ARM_BRIGHTNESS),
        sock=_fur_color(sprite, landmarks[CHIN, :2], radius, SOCK_BRIGHTNESS),
    )


def paw_sprite(style: PawStyle, width: int = 220) -> PawSprite:
    """Draws a paw on an arm pointing down."""
    ss = 2  # draw at 2x and downscale for smooth edges
    w = width * ss
    # The arm is long enough to always run off-screen.
    canvas_w, canvas_h = int(w * 1.4), int(w * ARM_LENGTH)
    cx, cy = canvas_w // 2, int(w * 0.75)
    color = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    alpha = np.zeros((canvas_h, canvas_w), np.uint8)

    def blob(fill, draw):
        draw(color, fill)
        draw(alpha, 255)

    arm_w = int(w * 0.62)
    outline = tuple(int(c * OUTLINE_SHADE) for c in style.sock)
    blob(style.arm, lambda img, c: cv2.rectangle(
        img, (cx - arm_w // 2, cy), (cx + arm_w // 2, canvas_h), c, -1))

    toes = [
        (cx + int(w * 0.40 * np.sin(a)), cy - int(w * 0.36 * np.cos(a)))
        for a in np.radians([-52, -18, 18, 52])
    ]
    toe_r = int(w * 0.17)

    def sock(img, c, grow=0):
        cv2.ellipse(img, (cx, cy + int(w * 0.25)), (arm_w // 2 + grow, int(w * 0.55) + grow),
                    0, 0, 360, c, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (int(w * 0.42) + grow, int(w * 0.36) + grow),
                    0, 0, 360, c, -1, cv2.LINE_AA)
        for toe in toes:
            cv2.circle(img, toe, toe_r + grow, c, -1, cv2.LINE_AA)

    blob(outline, lambda img, c: sock(img, c, grow=ss * 3))
    sock(color, style.sock)
    # Toe separations.
    for toe in toes:
        cv2.circle(color, toe, toe_r, outline, ss * 2, cv2.LINE_AA)
    sock(color, style.sock, grow=-ss * 3)

    cv2.ellipse(color, (cx, cy + int(w * 0.06)), (int(w * 0.2), int(w * 0.15)),
                0, 0, 360, BEAN, -1, cv2.LINE_AA)
    for toe in toes:
        cv2.ellipse(color, toe, (int(toe_r * 0.5), int(toe_r * 0.6)), 0, 0, 360, BEAN, -1, cv2.LINE_AA)

    # A little fur grain so it doesn't look flat.
    rng = np.random.default_rng(0)
    grain = cv2.GaussianBlur(rng.normal(0, 1, alpha.shape).astype(np.float32), (0, 0), ss * 1.5)
    # Rounded look: darker toward the sides of the arm.
    side = np.abs(np.arange(canvas_w) - cx) / (arm_w / 2)
    shade = 1 - ROUNDNESS * np.clip(side, 0, 1.3) ** 2
    color = np.clip(color * (shade[None, :, None] + 0.15 * grain[..., None]), 0, 255).astype(np.uint8)

    sprite = cv2.resize(np.dstack([color, alpha]), (canvas_w // ss, canvas_h // ss),
                        interpolation=cv2.INTER_AREA)
    return PawSprite(sprite, np.array([cx / ss, cy / ss]), width, cy / ss)


def photo_paw(image: np.ndarray) -> PawSprite:
    """A paw photo (BGRA, toes up, arm leaving at the bottom) as a sprite, its
    arm extended (mirrored back and forth) so it runs off-screen."""
    alpha = image[..., 3] > 127
    rows = alpha[: max(int(0.4 * len(alpha)), 1)].sum(axis=1)
    paw_row = int(np.argmax(rows))
    width = int(rows[paw_row])
    xs = np.flatnonzero(alpha[paw_row])
    anchor = np.array([(xs.min() + xs.max()) / 2, paw_row], float)
    start = min(paw_row + width, len(image) - 2)  # the arm, below the paw
    arm = image[start:]
    pieces, height = [image], len(image)
    while height < paw_row + PHOTO_ARM_LENGTH * width:
        arm = arm[::-1]
        pieces.append(arm)
        height += len(arm)
    top = np.flatnonzero(alpha.any(axis=1))[0]
    return PawSprite(np.ascontiguousarray(np.concatenate(pieces)), anchor, width, top + TOE_LENGTH * width)


def paw_set(photos: dict, style: PawStyle) -> dict:
    """{(view, hand): PawSprite} from a cat's paw photos ({(view, hand): BGRA}).
    A missing top view uses the sole, and without photos the drawn paw is used."""
    drawn = None
    paws = {}
    for hand in ("left", "right"):
        for view in VIEWS:
            photo = photos.get((view, hand))
            if photo is None:
                photo = photos.get(("back", hand))
            if photo is not None:
                paws[view, hand] = photo_paw(photo)
            else:
                drawn = drawn or paw_sprite(style)
                paws[view, hand] = drawn
    return paws


def palm_toward(handedness: str, pts: np.ndarray) -> float:
    """-1 (back of the hand to the camera) .. 1 (palm to the camera).

    From the 2D cross product of the hand's up direction (wrist -> middle
    knuckle) and its knuckles (pinky -> index): on an unmirrored image a right
    hand shows its palm when it is positive, a left hand when it is negative."""
    up = pts[MIDDLE_MCP] - pts[WRIST]
    across = pts[INDEX_MCP] - pts[PINKY_MCP]
    return _turn(up[:2], across[:2], 1.0 if handedness == "Right" else -1.0)


def _turn(up: np.ndarray, across: np.ndarray, hand: float) -> float:
    """palm_toward from a hand's up and across directions on screen; hand: +1 for a
    right hand, -1 for a left one."""
    turn = (up[0] * across[1] - up[1] * across[0]) / (np.linalg.norm(up) * np.linalg.norm(across) + 1e-6)
    return float(turn if hand >= 0 else -turn)


def hand_axes(params: np.ndarray) -> np.ndarray:
    """(3, 3) a tracked hand's frame (columns: toward the thumb, up the hand, out of its
    back; image axes, z away from the camera) from Paws' params. A right hand's frame is
    mirrored (its determinant is -1), so a paw modelled as a left one comes out right."""
    up, across, hand = params[3:6], params[6:9], params[9]
    up = up / (np.linalg.norm(up) + 1e-9)
    across = across - (across @ up) * up
    across /= np.linalg.norm(across) + 1e-9
    back = np.cross(up, across) * (1.0 if hand >= 0 else -1.0)
    return np.column_stack([across, up, back])


def sprite_curls(params: np.ndarray) -> np.ndarray:
    """A tracked hand's finger curls (Paws' params) ordered as the toes lie across a paw
    sprite placed on it, from its left: the sprite's x axis is (-up_y, up_x) on screen."""
    up, across, curls = params[3:5], params[6:8], np.asarray(params[10:14])
    return curls[::-1] if across @ (-up[1], up[0]) > 0 else curls


def finger_curls(pts: np.ndarray) -> np.ndarray:
    """(4,) index to pinky: 0 (straight) .. 1 (bent in), from how far each
    fingertip reaches past its knuckle."""
    reach = np.array([np.linalg.norm(pts[tip] - pts[WRIST]) / (np.linalg.norm(pts[mcp] - pts[WRIST]) + 1e-6)
                      for tip, mcp in FINGERS])
    return np.clip((reach - CLOSED_RANGE[0]) / (CLOSED_RANGE[1] - CLOSED_RANGE[0]), 0, 1)


def palm_facing(toward: float) -> float:
    """palm_toward as 0 (show the top of the paw) .. 1 (show the sole)."""
    t = np.clip((toward - PALM_RANGE[0]) / (PALM_RANGE[1] - PALM_RANGE[0]), 0, 1)
    return float(t * t * (3 - 2 * t))


class Paws:
    def __init__(self, out_width: int):
        self.out_w = out_width
        self.slots = {}  # handedness -> {"filter", "params", "presence"}

    def update(self, hands, user_face: np.ndarray | None, cam_size, dt: float,
               cat_face_center: np.ndarray, cat_face_width: float):
        """hands: HandLandmarkerResult; user_face: (478, 3) face landmarks in px,
        or None; cat_face_*: where the cat's face is on screen."""
        cam_w, cam_h = cam_size
        if user_face is not None:
            face_center = user_face[:, :2].mean(axis=0)
            face_width = np.ptp(user_face[:, 0])
        else:  # face never seen yet: assume a typical webcam framing
            face_center = np.array([cam_w / 2, cam_h * 0.4])
            face_width = cam_w * DEFAULT_FACE_WIDTH
        detections = []
        for landmarks, handedness in zip(hands.hand_landmarks, hands.handedness):
            # z (depth, the tracker's guess) uses the x scale, as MediaPipe defines it.
            pts = np.array([(p.x * cam_w, p.y * cam_h, getattr(p, "z", 0.0) * cam_w * HAND_DEPTH)
                            for p in landmarks])
            up = pts[MIDDLE_MCP] - pts[WRIST]
            up /= np.linalg.norm(up) + 1e-6
            across = pts[INDEX_MCP] - pts[PINKY_MCP]
            across /= np.linalg.norm(across) + 1e-6
            knuckles = pts[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)
            palm = 0.6 * knuckles + 0.4 * pts[WRIST, :2]
            palm_width = np.linalg.norm(pts[INDEX_MCP, :2] - pts[PINKY_MCP, :2])
            label = handedness[0].category_name
            # Placed by where the hand is relative to your face in the camera
            # frame (so hands stay on screen even though the cat fills it);
            # sized relative to your face, like a real paw next to the cat's.
            target = np.array([
                *(cat_face_center + (palm - face_center) * (self.out_w / cam_w)),
                palm_width * (cat_face_width / face_width) * PAW_SIZE,
                *up,
                *across,
                1.0 if label == "Right" else -1.0,  # the detected handedness, before any relabelling
                *finger_curls(pts[:, :2]),  # index to little finger
            ])
            detections.append((label, target))
        seen = set()
        for name, target in self._label(detections):
            slot = self.slots.setdefault(
                name, {"filter": OneEuroFilter(PAW_MIN_CUTOFF, PAW_BETA), "presence": 0.0}
            )
            slot["params"] = slot["filter"](target, dt)
            seen.add(name)
        for name, slot in self.slots.items():
            present = 1.0 if name in seen else 0.0
            slot["presence"] += FADE_RATE * (present - slot["presence"])
            if name not in seen and slot["presence"] < 0.02:
                slot["filter"] = OneEuroFilter(PAW_MIN_CUTOFF, PAW_BETA)  # restart fresh

    def _label(self, detections):
        """Gives each detected hand its slot ("Left" or "Right"): a hand close
        to where a still visible paw was continues that paw; other hands go by
        MediaPipe's guess, or by which side they are on if both got the same guess."""
        tracked = {
            name: slot["params"] for name, slot in self.slots.items()
            if "params" in slot and slot["presence"] > TRACKED
        }

        def distance(name, target):
            """How far target is from the paw `name`, in paw sizes (inf if none)."""
            last = tracked.get(name)
            if last is None:
                return np.inf
            return float(np.linalg.norm(last[:2] - target[:2]) / max(last[2], 1.0))

        names = ("Left", "Right")
        if len(detections) == 2:
            (guess_a, a), (guess_b, b) = detections
            near = lambda name, target: min(distance(name, target), SAME_HAND)
            straight = near("Left", a) + near("Right", b)
            swapped = near("Right", a) + near("Left", b)
            if straight != swapped:  # at least one hand continues a paw
                return [("Left", a), ("Right", b)] if straight < swapped else [("Right", a), ("Left", b)]
            if guess_a != guess_b:
                return detections
            # Both guessed the same: your right hand is on the image's left.
            return [("Right", a), ("Left", b)] if a[0] < b[0] else [("Left", a), ("Right", b)]
        labelled = []
        for guess, target in detections:
            near = min(names, key=lambda name: distance(name, target))
            labelled.append((near if distance(near, target) < SAME_HAND else guess, target))
        return labelled

    def draw(self, frame: np.ndarray, paws, hidden=()) -> np.ndarray:
        """paws: a PawSprite for every hand, {(view, hand): PawSprite} (see paw_set),
        or 3D paws (catcam.paw3d.Paw3D); hidden: hands ("Left", "Right") that get no paw."""
        shown = [(name, slot["params"], slot["presence"]) for name, slot in self.slots.items()
                 if slot["presence"] >= 0.02 and "params" in slot and name not in hidden]
        if not isinstance(paws, (PawSprite, dict)):
            return paws.render(frame, [
                Placement(p[0], p[1], p[2], hand_axes(p), np.clip(p[10:14], 0, 1), presence)
                for _, p, presence in shown])
        out = frame.copy()
        for name, params, presence in shown:
            x, y, size, ux, uy = params[:5]
            norm = np.hypot(ux, uy) + 1e-6
            placement = (x, y, size, ux / norm, uy / norm, np.clip(sprite_curls(params), 0, 1))
            if isinstance(paws, PawSprite):
                layers = [(paws, 1.0)]
            else:
                palm = palm_facing(_turn(params[3:5], params[6:8], params[9]))
                hand = name.lower()
                layers = [(paws["front", hand], 1 - palm), (paws["back", hand], palm)]
            _blend(out, [(paw, w * presence) for paw, w in layers if w > 0.01], placement)
        return out


def _blend(out: np.ndarray, layers, placement):
    """Draws paws [(PawSprite, opacity)] crossfaded into out (in place), at
    placement (x, y, size, up_x, up_y, curls), curls (4,) of the toes from the sprite's left."""
    x, y, size, ux, uy, curls = placement
    frame_h, frame_w = out.shape[:2]
    color = alpha = box = None
    for paw, opacity in layers:
        s = size / paw.width
        # Rotate the sprite's "up" (0, -1) onto the hand's up direction.
        rot = s * np.array([[-uy, -ux], [ux, -uy]])
        m = np.hstack([rot, (np.array([x, y]) - rot @ paw.anchor)[:, None]])
        if box is None:  # only warp and blend the screen area the first paw covers, raised toes included
            sprite_h, sprite_w = paw.image.shape[:2]
            top = -RAISE * paw.toes
            corners = np.array([[0, top], [sprite_w, top], [0, sprite_h], [sprite_w, sprite_h]], float)
            corners = corners @ m[:, :2].T + m[:, 2]
            x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int) - 2, 0, (frame_w, frame_h))
            x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 2, 0, (frame_w, frame_h))
            if x1 <= x0 or y1 <= y0:
                return
            box = (x0, y0, x1, y1)
            color = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            alpha = np.zeros((y1 - y0, x1 - x0, 1), np.float32)
        x0, y0, x1, y1 = box
        m[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(
            paw.image, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        if max(curls) > 0.01:
            _curl_toes(warped, paw, m, curls)
        a = warped[..., 3:4].astype(np.float32) / 255 * opacity
        color += warped[..., :3] * a
        alpha += a
    if box is None:
        return
    x0, y0, x1, y1 = box
    roi = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = np.clip(color + roi * (1 - np.minimum(alpha, 1)), 0, 255).astype(np.uint8)


def _curl_toes(warped: np.ndarray, paw: PawSprite, m: np.ndarray, curls: np.ndarray):
    """Redraws the toes of a paw placed in warped by m (sprite -> warped) curled in.
    Below the toes the curl changes nothing, so only the toes' area (and above,
    where raised toes reach) is redrawn."""
    w, top = paw.image.shape[1], -RAISE * paw.toes
    corners = np.array([[0, top], [w, top], [0, paw.toes], [w, paw.toes]], float) @ m[:, :2].T + m[:, 2]
    h_out, w_out = warped.shape[:2]
    x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int) - 1, 0, (w_out, h_out))
    x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 1, 0, (w_out, h_out))
    if x1 <= x0 or y1 <= y0:
        return
    u, v = _curled(paw, m, (x0, y0, x1, y1), curls)
    warped[y0:y1, x0:x1] = cv2.remap(paw.image, u, v, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _toe_heights(paw: PawSprite) -> np.ndarray:
    """(4,) how far (sprite px) each toe's middle reaches above the paw's toe line."""
    columns = np.round(paw.anchor[0] + np.array(TOE_X) * paw.width).astype(int)
    columns = np.clip(columns, 0, paw.image.shape[1] - 1)
    inside = paw.image[:int(paw.toes), columns, 3] > 128
    return int(paw.toes) - np.where(inside.any(axis=0), inside.argmax(axis=0), int(paw.toes))


def _curled(paw: PawSprite, m: np.ndarray, box, curls: np.ndarray):
    """Sprite coordinates (u, v) to sample for each pixel of box (x0, y0, x1,
    y1) of the area the paw is placed in by m, with each toe curled in by its
    curl (curls (4,), from the sprite's left)."""
    inverse = cv2.invertAffineTransform(m)
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    v = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
    toe = TOE_LENGTH * paw.width
    others = (np.sum(curls) - curls) / (len(curls) - 1)
    stands = np.clip((others - STAND[0]) / (STAND[1] - STAND[0]), 0, 1)
    column = (u - paw.anchor[0]) / paw.width
    toe_edges = np.ravel([(x - TOE_HALF, x + TOE_HALF) for x in TOE_X])

    def per_toe(values):  # the value of the toe in each column
        return np.interp(column, toe_edges, np.repeat(values, 2))

    closed = per_toe(curls)
    dropped = per_toe(CURL * curls * _toe_heights(paw))
    raised = per_toe(RAISE * toe * (1 - curls) * stands)
    knuckle = KNUCKLE * toe
    above = np.maximum(paw.toes - v, 0)  # how far up the toes, as drawn
    up_finger = above - knuckle
    back_and_forths = np.maximum(np.round(raised / (2 * FUR * toe)), 1)  # whole ones, so the tip joins on
    band = np.maximum(raised / (2 * back_and_forths), 1e-6)
    fur = knuckle - (band - np.abs(np.mod(up_finger, 2 * band) - band))
    above = np.where(above < knuckle, above, np.where(up_finger < raised, fur, above - raised))
    above += dropped * np.minimum(above / knuckle, 1)
    v = np.where(v < paw.toes, paw.toes - above, v)
    narrow = 1 - NARROW * closed * np.clip(above / (TOE_LENGTH * paw.width), 0, 1)
    u = paw.anchor[0] + (u - paw.anchor[0]) / narrow
    return u.astype(np.float32), v.astype(np.float32)
