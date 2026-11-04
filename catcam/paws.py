"""Cat paws that follow your hands.

Each tracked hand becomes a socked paw with pink toe beans on a furry arm,
coloured like the cat: the arm takes the fur of its cheeks, the sock the fur of
its chin. Its position is taken relative to your face, so a hand next to your
chin puts the paw next to the cat's chin.
"""

from typing import NamedTuple

import cv2
import numpy as np

from catcam.head_pose import OneEuroFilter

PAW_SIZE = 1.1  # paw width relative to your palm width
PAW_MIN_CUTOFF, PAW_BETA = 1.5, 0.02
FADE_RATE = 0.35  # how fast paws appear / disappear
ARM_LENGTH = 14  # in paw widths
DEFAULT_FACE_WIDTH = 0.25  # of the camera width, used until a face is seen

BEAN = (178, 165, 240)
OUTLINE_SHADE = 0.75  # toe outlines: the sock colour, darkened
# Face landmarks whose fur colours the paw, and which brightness of that fur
# (as a percentile) to use: chins sit in shadow, so the sock takes a light tone.
CHEEKS, ARM_BRIGHTNESS = [234, 454, 93, 323], 50
CHIN, SOCK_BRIGHTNESS = [152, 175, 199, 200, 205, 425, 36, 266], 80
SAMPLE_RADIUS = 0.03  # in face widths
ROUNDNESS = 0.3  # how much darker the arm and paw get toward their sides

WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17


class PawStyle(NamedTuple):
    arm: tuple[int, int, int]  # BGR
    sock: tuple[int, int, int]


class PawSprite(NamedTuple):
    image: np.ndarray  # BGRA, arm pointing down
    anchor: np.ndarray  # paw centre in the image
    width: int  # paw width in the image


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
    return PawSprite(sprite, np.array([cx / ss, cy / ss]), width)


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
        seen = set()
        for landmarks, handedness in zip(hands.hand_landmarks, hands.handedness):
            pts = np.array([(p.x * cam_w, p.y * cam_h) for p in landmarks])
            up = pts[MIDDLE_MCP] - pts[WRIST]
            up /= np.linalg.norm(up) + 1e-6
            palm = 0.6 * pts[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0) + 0.4 * pts[WRIST]
            palm_width = np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])
            # Placed by where the hand is relative to your face in the camera
            # frame (so hands stay on screen even though the cat fills it);
            # sized relative to your face, like a real paw next to the cat's.
            target = np.array([
                *(cat_face_center + (palm - face_center) * (self.out_w / cam_w)),
                palm_width * (cat_face_width / face_width) * PAW_SIZE,
                *up,
            ])
            name = handedness[0].category_name
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

    def draw(self, frame: np.ndarray, paw: PawSprite) -> np.ndarray:
        out = frame.copy()
        frame_h, frame_w = out.shape[:2]
        sprite_h, sprite_w = paw.image.shape[:2]
        sprite_corners = np.array([[0, 0], [sprite_w, 0], [0, sprite_h], [sprite_w, sprite_h]], float)
        for slot in self.slots.values():
            if slot["presence"] < 0.02 or "params" not in slot:
                continue
            x, y, size, ux, uy = slot["params"]
            norm = np.hypot(ux, uy) + 1e-6
            ux, uy = ux / norm, uy / norm
            s = size / paw.width
            # Rotate the sprite's "up" (0, -1) onto the hand's up direction.
            rot = s * np.array([[-uy, -ux], [ux, -uy]])
            m = np.hstack([rot, (np.array([x, y]) - rot @ paw.anchor)[:, None]])
            # Only warp and blend the screen area the paw covers.
            corners = sprite_corners @ m[:, :2].T + m[:, 2]
            x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, (frame_w, frame_h))
            x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, (frame_w, frame_h))
            if x1 <= x0 or y1 <= y0:
                continue
            m[:, 2] -= (x0, y0)
            warped = cv2.warpAffine(
                paw.image, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )
            a = warped[..., 3:4].astype(np.float32) / 255 * slot["presence"]
            roi = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = (warped[..., :3] * a + roi * (1 - a)).astype(np.uint8)
        return out
