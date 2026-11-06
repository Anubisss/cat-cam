"""Your own hands in place of paws: cut out of the webcam picture and washed in
the animal's fur colours.

MediaPipe's multiclass selfie segmenter (models/selfie_multiclass_256x256.tflite)
tells the skin of your hands and arms apart from your face, hair, clothes and
the background. Only that skin is cut out, and only within reach of a tracked
hand: around its landmarks and a short way down its forearm, fading out there.
So your face and the rest of you never are, even with a hand in front of the
face. The segmenter runs on a small copy of the frame on a thread of its own,
alongside the face and hand trackers (see catcam.app.Tracker), and the cutting
out and colouring follow on the tracker's thread too: the main loop only gets
the finished cut-outs, and only while the hands are shown.

Each hand is drawn where its paw would be (catcam.paws: placed by where it is
relative to your face, as big next to the animal's face as it is next to
yours), in front of the animal. It keeps its own shading and detail, so it is
still clearly your hand, under a gentle wash of the animal's fur (the paw's
colour on the hand, the arm's further down, as on the 3D paws: see
catcam.paw3d.paw_colours) and a fine grain of fur running along the hand.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from catcam.paths import SEGMENT_MODEL
from catcam.paw3d import PawColours
from catcam.paws import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST, palm_of

SEGMENT_WIDTH = 256  # px the frame is shrunk to for the segmenter (its model looks at 256 x 256)
BODY_SKIN = 2  # the segmenter's class for skin other than the face's (hands, arms, neck)
# A pixel is skin from SKIN[0] to SKIN[1] of the segmenter's confidence, blended in between.
SKIN = (0.35, 0.65)
# The segmenter's confidence follows the newest frame by this share each frame (1 = no
# smoothing), steadying flicker at the edges; after a gap of RESTART s it starts afresh.
SKIN_FOLLOW, RESTART = 0.6, 0.25
# Sizes below are in hand sizes: the larger of the palm's length (wrist to middle knuckle)
# and its width (index to little knuckle), at least MIN_HAND px.
MIN_HAND = 12.0
REACH = 0.45  # how far past the hand's landmarks the cut-out may reach (fingers are wide)
FOREARM = (0.6, 1.6)  # the forearm shows fully to [0] below the wrist and fades out by [1]...
FOREARM_WIDTH = 1.4  # ...within a band this wide
FEATHER = 0.03  # how soft the cut-out's edge is
# The hand tracker now and then takes something else for a hand (a face, say): a "hand"
# whose landmarks the segmenter mostly doesn't take for skin (their median under this
# confidence) isn't cut out.
HAND_SKIN = 0.3
# The colouring: WASH of the way to the fur colour (lit by your hand's own shading),
# GRAIN the strength of the fur's light and dark streaks, STREAK their length and width,
# repeating every GRAIN_TILE.
WASH = 0.4
GRAIN, STREAK, GRAIN_TILE = 0.08, (0.06, 0.006), 0.8
SHADE_RANGE = (0.35, 1.6)  # your hand's shading (its brightness over its median), clamped
# Paw colour on the hand, the arm colour down the forearm: blended between these far below the wrist.
SOCK = (0.1, 0.6)
TILE = 256  # px of the grain's texture
SHARPEN, SHARPEN_BLUR = 0.6, 1.2  # unsharp mask (strength, px): a webcam's hand is soft, and drawn bigger
# A hand is drawn as much bigger than in the webcam picture as a paw would be (the animal's
# face over yours), but at most MAX_SCALE times: past that it looks blurred.
MAX_SCALE = 2.0
STALE = 10  # frames a hand found but not cut out (see HAND_SKIN) keeps showing its last cut-out


class Cut(NamedTuple):
    """One hand cut out of a webcam frame, coloured."""
    image: np.ndarray  # BGRA, straight alpha
    origin: tuple[int, int]  # its top left in the webcam frame, px
    palm: np.ndarray  # the middle of the palm in the webcam frame, px (as catcam.paws places paws)


class Segmenter:
    """The multiclass selfie segmenter on a thread of its own: start() hands it a
    frame and returns at once; the future's result is the frame's skin, 0..1 at
    SEGMENT_WIDTH, smoothed over the frames before."""

    def __init__(self, model=SEGMENT_MODEL):
        self._segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
        ))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmenter")
        self._skin, self._last = None, None

    def start(self, frame: np.ndarray, timestamp: int) -> Future:
        """frame: BGR; timestamp: ms, increasing."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (SEGMENT_WIDTH, max(round(SEGMENT_WIDTH * h / w), 1)),
                           interpolation=cv2.INTER_AREA)
        return self._pool.submit(self._segment, small, timestamp)

    def _segment(self, small: np.ndarray, timestamp: int) -> np.ndarray:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        result = self._segmenter.segment_for_video(image, timestamp)
        skin = np.squeeze(result.confidence_masks[BODY_SKIN].numpy_view()).astype(np.float32)
        self._skin = steady(skin, self._skin, (timestamp - self._last) / 1000 if self._last is not None
                            else None)
        self._last = timestamp
        return self._skin

    def close(self):
        self._pool.shutdown(wait=True)
        self._segmenter.close()


def steady(skin: np.ndarray, before: np.ndarray | None, gap: float | None) -> np.ndarray:
    """skin smoothed over time: following the newest by SKIN_FOLLOW, afresh after a gap (s)."""
    if before is None or gap is None or gap > RESTART or before.shape != skin.shape:
        return skin
    return before + SKIN_FOLLOW * (skin - before)


def hand_size(points: np.ndarray) -> float:
    """A hand's size in px (see MIN_HAND) from its landmarks ((21, 2+) px)."""
    length = np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2])
    width = np.linalg.norm(points[INDEX_MCP, :2] - points[PINKY_MCP, :2])
    return float(max(length, width, MIN_HAND))


def _down(points: np.ndarray) -> np.ndarray:
    """Unit vector from the middle knuckle to the wrist: down the forearm."""
    down = points[WRIST, :2] - points[MIDDLE_MCP, :2]
    return down / (np.linalg.norm(down) + 1e-6)


def reach(points: np.ndarray, frame_size) -> tuple[int, int, int, int] | None:
    """The area (x0, y0, x1, y1, px) of the frame a hand's cut-out can cover, or None
    if it is off the frame."""
    size = hand_size(points)
    wrist = points[WRIST, :2]
    arm_end = wrist + _down(points) * FOREARM[1] * size
    corners = np.vstack([points[:, :2], arm_end])
    margin = max(REACH, FOREARM_WIDTH / 2) * size + 2
    w, h = frame_size
    x0, y0 = np.clip(np.floor(corners.min(axis=0) - margin).astype(int), 0, (w, h))
    x1, y1 = np.clip(np.ceil(corners.max(axis=0) + margin).astype(int), 0, (w, h))
    return (int(x0), int(y0), int(x1), int(y1)) if x1 > x0 and y1 > y0 else None


def below_wrist(points: np.ndarray, box) -> np.ndarray:
    """(h, w) float over box: how far each px is down the forearm from the wrist, in hand sizes."""
    x0, y0, x1, y1 = box
    down = _down(points) / hand_size(points)
    wrist = points[WRIST, :2]
    xs = (np.arange(x0, x1, dtype=np.float32) - wrist[0]) * down[0]
    ys = (np.arange(y0, y1, dtype=np.float32) - wrist[1]) * down[1]
    return ys[:, None] + xs[None, :]


def region(points: np.ndarray, box, below: np.ndarray) -> np.ndarray:
    """(h, w) float 0..1 over box: where a hand's cut-out may be, the hand and its
    forearm, the forearm fading out further down (below: see below_wrist)."""
    x0, y0, x1, y1 = box
    size = hand_size(points)
    pts = points[:, :2] - (x0, y0)
    shape = (y1 - y0, x1 - x0)
    hand = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(hand, hull, 255, cv2.LINE_AA)
    cv2.polylines(hand, [hull], True, 255, max(int(2 * REACH * size), 1), cv2.LINE_AA)
    arm = np.zeros(shape, np.uint8)
    end = pts[WRIST] + _down(points) * FOREARM[1] * size
    cv2.line(arm, tuple(np.round(pts[WRIST]).astype(int)), tuple(np.round(end).astype(int)), 255,
             max(int(FOREARM_WIDTH * size), 1), cv2.LINE_AA)
    fade = np.clip((FOREARM[1] - below) / (FOREARM[1] - FOREARM[0]), 0, 1)
    return np.maximum(hand.astype(np.float32), arm * fade) / 255


def skin_in(skin: np.ndarray, frame_size, box) -> np.ndarray:
    """The segmenter's skin (small) scaled up to box (x0, y0, x1, y1) of the frame, 0..1
    by SKIN."""
    x0, y0, x1, y1 = box
    w, _ = frame_size
    s = skin.shape[1] / w  # small px per frame px
    # Pixel centres: frame px x samples the small mask at (x + 0.5) * s - 0.5.
    m = np.array([[s, 0, (x0 + 0.5) * s - 0.5], [0, s, (y0 + 0.5) * s - 0.5]], np.float32)
    up = cv2.warpAffine(skin, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    return np.clip((up - SKIN[0]) / (SKIN[1] - SKIN[0]), 0, 1)


def skin_at(skin: np.ndarray, frame_size, points: np.ndarray) -> np.ndarray:
    """The segmenter's confidence (small) that each of points ((n, 2+) frame px) is skin."""
    s = skin.shape[1] / frame_size[0]
    xy = np.round((points[:, :2] + 0.5) * s - 0.5).astype(int)
    xy = np.clip(xy, 0, (skin.shape[1] - 1, skin.shape[0] - 1))
    return skin[xy[:, 1], xy[:, 0]]


@cache
def grain_tile() -> np.ndarray:
    """(TILE, TILE) float -1..1: streaks of fur running along y, seamless when repeated."""
    noise = np.random.default_rng(7).normal(size=(TILE, TILE)).astype(np.float32)
    length, width = (s / GRAIN_TILE * TILE for s in STREAK)
    pad = int(3 * max(length, width)) + 1  # blurred with the tile repeated round it, so its edges meet
    streaks = cv2.GaussianBlur(np.pad(noise, pad, mode="wrap"), (0, 0), sigmaX=max(width, 0.5), sigmaY=length)
    streaks = streaks[pad:pad + TILE, pad:pad + TILE]
    return streaks / (np.abs(streaks).max() + 1e-6)


def grain(points: np.ndarray, box) -> np.ndarray:
    """(h, w) float -1..1 over box: the fur's streaks laid along the hand (from its
    wrist toward its fingers), moving and turning with it."""
    x0, y0, x1, y1 = box
    size = hand_size(points)
    down = _down(points)
    up, across = -down, np.array([down[1], -down[0]])
    k = TILE / (GRAIN_TILE * size)  # tile px per frame px
    origin = points[WRIST, :2] - (x0, y0)
    # Tile coordinates of a box px p: (across . (p - origin), up . (p - origin)) * k.
    m = np.array([[across[0] * k, across[1] * k, -k * (across @ origin)],
                  [up[0] * k, up[1] * k, -k * (up @ origin)]], np.float32)
    return cv2.warpAffine(grain_tile(), m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


def colour(bgr: np.ndarray, alpha: np.ndarray, points: np.ndarray, box, below: np.ndarray,
           colours: PawColours) -> np.ndarray:
    """A hand's pixels (BGR uint8 over box) washed in the animal's fur: the fur colour
    lit by the hand's own shading, WASH of the way, and the fur's grain over it."""
    image = bgr.astype(np.float32)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    middle = float((luma * alpha).sum() / max(alpha.sum(), 1.0))  # the hand's mean brightness
    shade = np.clip(luma / max(middle, 1.0), *SHADE_RANGE)[..., None]
    arm = np.clip((below - SOCK[0]) / (SOCK[1] - SOCK[0]), 0, 1)[..., None]
    fur = (1 - arm) * np.array(colours.sock, np.float32) + arm * np.array(colours.arm, np.float32)
    washed = image + WASH * (fur * shade - image)
    washed *= 1 + GRAIN * grain(points, box)[..., None]
    return np.clip(washed, 0, 255).astype(np.uint8)


def cut_out(frame: np.ndarray, skin: np.ndarray, points: np.ndarray, colours: PawColours) -> Cut | None:
    """One hand (landmarks (21, 2+) in frame px) cut out of frame (BGR) by the
    segmenter's skin (small, 0..1) and coloured; None if it is off the frame."""
    size = (frame.shape[1], frame.shape[0])
    box = reach(points, size)
    if box is None:
        return None
    if np.median(skin_at(skin, size, points)) < HAND_SKIN:
        return None
    below = below_wrist(points, box)
    alpha = skin_in(skin, size, box) * region(points, box, below)
    # Only the part with anything in it, and room for the feathered edge, goes on.
    feather = FEATHER * hand_size(points)
    rows, cols = np.flatnonzero(alpha.max(axis=1) > 0.01), np.flatnonzero(alpha.max(axis=0) > 0.01)
    if not len(rows):
        return None
    pad = int(np.ceil(3 * feather))
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, alpha.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, alpha.shape[1])
    alpha, below = alpha[r0:r1, c0:c1], below[r0:r1, c0:c1]
    box = (box[0] + c0, box[1] + r0, box[0] + c1, box[1] + r1)
    x0, y0, x1, y1 = box
    if feather > 0.3:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    crop = frame[y0:y1, x0:x1]
    crop = cv2.addWeighted(crop, 1 + SHARPEN, cv2.GaussianBlur(crop, (0, 0), SHARPEN_BLUR), -SHARPEN, 0)
    bgr = colour(crop, alpha, points, box, below, colours)
    image = np.dstack([bgr, np.round(alpha * 255).astype(np.uint8)])
    return Cut(image, (x0, y0), palm_of(points))


def cut_all(frame: np.ndarray, skin: np.ndarray, hands, colours: PawColours) -> list[Cut | None]:
    """Each tracked hand (a HandLandmarkerResult's, in its order) cut out of frame."""
    h, w = frame.shape[:2]
    return [cut_out(frame, skin, np.array([(p.x * w, p.y * h) for p in landmarks]), colours)
            for landmarks in hands.hand_landmarks]


class RealHands:
    """Draws the cut-out hands where the paws would be (see catcam.paws.Paws), each
    hand fading in and out with its paw's presence. A hand the tracker lost for a
    moment keeps its last cut-out while it fades."""

    def __init__(self):
        self.cuts: dict[str, Cut] = {}  # by Paws' slot ("Left", "Right")
        self.stale: dict[str, int] = {}  # frames since each one's cut-out was new

    def update(self, cuts: list[Cut | None] | None, slots: dict):
        """cuts: this frame's, in the order of the hand tracker's hands (see cut_all),
        or None when nothing was cut; slots: Paws.slots, updated with the same frame."""
        if cuts is None:  # not cutting (yet): nothing old to show when it starts again
            self.cuts.clear()
            return
        for name, slot in slots.items():
            index = slot.get("detection")
            if index is not None and index < len(cuts) and cuts[index] is not None:
                self.cuts[name], self.stale[name] = cuts[index], 0
                continue
            self.stale[name] = self.stale.get(name, 0) + (index is not None)
            if slot["presence"] < 0.02 or self.stale[name] > STALE:
                self.cuts.pop(name, None)

    def draw(self, frame: np.ndarray, slots: dict, scale: float, hidden=()) -> np.ndarray:
        """frame with the hands over it; scale: output px per webcam px for a hand's size
        (the animal's face width over yours, see Paws.scale; at most MAX_SCALE); hidden:
        hands that get none."""
        out = None
        scale = min(scale, MAX_SCALE)
        for name, slot in slots.items():
            cut = self.cuts.get(name)
            if cut is None or name in hidden or slot["presence"] < 0.02 or "params" not in slot:
                continue
            out = frame.copy() if out is None else out
            x, y = slot["params"][:2]
            # Webcam px -> output px: the palm lands where the paw would be, scaled about it.
            m = np.array([[scale, 0, x - scale * (cut.palm[0] - cut.origin[0])],
                          [0, scale, y - scale * (cut.palm[1] - cut.origin[1])]])
            _lay(out, cut.image, m, min(slot["presence"], 1.0))
        return frame if out is None else out


def _lay(out: np.ndarray, image: np.ndarray, m: np.ndarray, opacity: float):
    """Lays image (BGRA) over out (in place), placed by the affine m (image -> out)."""
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ m[:, :2].T + m[:, 2]
    frame_h, frame_w = out.shape[:2]
    x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, (frame_w, frame_h))
    x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, (frame_w, frame_h))
    if x1 <= x0 or y1 <= y0:
        return
    m = m.copy()
    m[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(image, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[..., 3].astype(np.float32) * (opacity / 255)
    roi = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = cv2.blendLinear(warped[..., :3], roi, alpha, 1 - alpha)
