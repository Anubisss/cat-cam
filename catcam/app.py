"""Cat face filter for Zoom.

Pick a cat, then it copies you live: it turns, nods and tilts with your head,
copies your whole face (brows, eyes, cheeks, lips, jaw) and your hands become
paws. Only the rendered cat goes out to the virtual camera; no webcam pixels
are ever sent, so your face is never visible.

Run:  uv run cat-cam      then pick "OBS Virtual Camera" in Zoom.
      uv run cat-cam --camera 1   to use another webcam.
Keys in the preview window: c = change cat, n = "this is my neutral face"
(recalibrate), q = quit.
"""

import argparse
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

from catcam import cats as cat_assets
from catcam import glass
from catcam.background import Background, Chooser, behind
from catcam.cat_face import EYES, CatFace
from catcam.cat_mesh import CatMesh, Renderer
from catcam.ears import EarMotion
from catcam.face_motion import ExpressionTransfer
from catcam.fx import Effects, terminal_name
from catcam.fx import load as load_effects
from catcam.head3d import Head3D, has_head
from catcam.head_pose import (
    OneEuroFilter,
    facing_camera,
    relative_rotation,
    rotation_of,
    to_matrix,
    to_vector,
)
from catcam.idle import Idle
from catcam.paths import FACE_MODEL, HAND_MODEL, LOG_FILE
from catcam.paws import Paws, paw_set, paw_style
from catcam.picker import SIZE as PREVIEW_SIZE
from catcam.picker import Picker
from catcam.reactions import Reactions, reaction_for
from catcam.tail import Tail, TailHand, TailSkin, tail_skin
from catcam.voice import VirtualMic, Voice

OUT_W, OUT_H, FPS = 1280, 720, 30
WINDOW = "Cat Cam"
# Switches in a glass pill in the preview's top left corner: (name, label, key).
# While "Tail" is on, a hand alone in view lifts a tail instead of being a paw
# (two hands are two paws); "3D head" shows the cat's 3D head, if it has one;
# "Background" asks for a picture or video to put behind the animal.
TOGGLES = (("tail", "Tail", "t"), ("head3d", "3D head", "d"), ("background", "Background", "b"))
PILL = (16, 16, 44)  # the pill's x, y and height (preview px)
CAPSULE = (44, 16)  # the bottom capsule's height and its gap to the window's edge
# Keys shown in the bottom capsule (the switches show their own).
KEY_HINTS = (("c", "Change cat"), ("1-8", "Reactions"), ("n", "Neutral face"), ("q", "Quit"))

EXPRESSION_GAIN = 1.3  # exaggerate your expressions on the cat (1 = 1:1)
BLINK_SMOOTHING = 0.5  # 0..1, higher = snappier; soften small tracking changes
# One Euro filters (see catcam.head_pose): lower MIN_CUTOFF (Hz) = less jitter
# when still, higher BETA = less lag when moving fast; moving slower than FLOOR
# is taken for tracking noise. Head speeds are in radians per second; expression
# speeds in face widths per second, for each face point on its own. Tuned with
# tools/tracking_eval.py.
ROTATION_MIN_CUTOFF, ROTATION_BETA, ROTATION_FLOOR = 0.1, 12.0, 0.15
EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA, EXPRESSION_FLOOR = 0.7, 20.0, 0.06
SPEED_CUTOFF = (3.0, 2.0)  # Hz, how quickly the head's and the face's speed estimates follow

# Blendshape score range mapped to 0..1 eye closedness...
BLINK_RANGE = (0.45, 0.75)
# ...and, as some people's blink score stays low, also how much the eye
# opening shrank compared to your neutral face.
SHUT_RANGE = (0.5, 0.9)
# The neutral face only adapts while all of these are low.
EXPRESSIVE = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff",
]
RELAXED_BELOW = 0.25
NO_FACE_HINT_AFTER = 1.0  # s without a face before the preview says so
WEBCAM_TIMEOUT = 5.0  # s of failed webcam reads before giving up
FIRST_FRAME_TIMEOUT = 60.0
ENTER, ESC = 13, 27


@dataclass(frozen=True)
class Cat:
    """Everything needed to draw one cat."""
    name: str
    points: np.ndarray
    eyelids: CatFace
    mesh: CatMesh | Head3D
    head3d: Head3D | None  # the real 3D head, if this cat has one
    paw: dict  # {(view, hand): PawSprite}, see catcam.paws.paw_set
    tail: TailSkin | None
    sounds: list  # what your syllables become, see catcam.voice; empty = your own voice
    sprite: np.ndarray
    background: np.ndarray


def load_cats(renderer: Renderer) -> list[Cat]:
    names = cat_assets.available()
    missing = cat_assets.unprepared()
    if missing:
        print(f"Not prepared yet (skipped): {', '.join(missing)}. "
              "Run: uv run --group prepare python -m catcam.prepare")
    if not names:
        sys.exit("No cats found. Put portraits in all_cats/ and run: "
                 "uv run --group prepare python -m catcam.prepare")
    cats = []
    for name in names:
        a = cat_assets.load(name)
        mesh = CatMesh(renderer, a.sprite, a.points, a.background, snap=not a.mouth_set)
        cats.append(Cat(
            name=name,
            points=a.points,
            eyelids=CatFace(a.sprite, a.points[:, :2]),
            mesh=mesh,
            head3d=Head3D(renderer, name, mesh, a.sprite, a.points, a.background) if has_head(name) else None,
            paw=paw_set(a.paws, paw_style(a.sprite, a.points)),
            tail=tail_skin(a.paws),
            sounds=a.sounds,
            sprite=a.sprite,
            background=a.background,
        ))
    return cats


def shown(cat: Cat, head3d: bool) -> Cat:
    """The cat as drawn: with its 3D head while that switch is on, if it has one."""
    return replace(cat, mesh=cat.head3d) if head3d and cat.head3d is not None else cat


def landmarks_px(landmarks, width, height):
    """(478, 3) pixel coords; z uses the x scale, as MediaPipe defines it."""
    return np.array([(p.x * width, p.y * height, p.z * width) for p in landmarks])


def ramp(value, lo, hi):
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def eye_openness(face: np.ndarray) -> np.ndarray:
    """Height of each eye's opening relative to its width (image-left, image-right)."""
    openness = []
    for upper, lower in EYES:
        middle = slice(2, -2)  # skip the corners
        height = np.mean(face[lower[middle], 1] - face[upper[middle], 1])
        openness.append(height / (np.linalg.norm(face[upper[0], :2] - face[upper[-1], :2]) + 1e-6))
    return np.array(openness)


def blinks(scores, face=None, neutral=None):
    """Eye closedness (image-left, image-right), each 0..1."""
    # MediaPipe names eyes from the person's view: their right eye is on the
    # image's left.
    closed = np.array([
        ramp(scores["eyeBlinkRight"], *BLINK_RANGE),
        ramp(scores["eyeBlinkLeft"], *BLINK_RANGE),
    ])
    if face is not None and neutral is not None:
        shrunk = 1 - eye_openness(face) / np.maximum(eye_openness(neutral), 1e-6)
        closed = np.maximum(closed, [ramp(v, *SHUT_RANGE) for v in shrunk])
    return closed


class FaceState:
    """Your smoothed head pose and expression, independent of the cat."""

    def __init__(self):
        self.rotation_filter = OneEuroFilter(ROTATION_MIN_CUTOFF, ROTATION_BETA, SPEED_CUTOFF[0],
                                             ROTATION_FLOOR)
        self.expression_filter = OneEuroFilter(EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA, SPEED_CUTOFF[1],
                                               EXPRESSION_FLOOR, per_point=True)
        self.transfer = ExpressionTransfer()
        self.neutral_pose = None  # the pose that faces the viewer; None = looking into the camera
        self.take_pose = False
        self.head = np.zeros(3)  # rotation vector; 0 = facing the viewer
        self.blink = np.zeros(2)
        self.ear_motion = EarMotion()
        self.ears = np.zeros(2)  # ear angles, see catcam.ears
        self.idle = Idle()  # what the animal does on its own while you are still
        self.offsets = np.zeros((478, 3))  # in face widths
        self.face = None  # last seen face landmarks in px
        self.last_seen = None
        # The last frame's head, offsets and blink before smoothing (for tools/tracking_eval.py).
        self.measured = (self.head, self.offsets, self.blink)

    def reset_neutral(self):
        """Your next face and pose become the neutral ones."""
        self.transfer.reset()
        self.take_pose = True

    def update(self, result, cam_size, dt: float, now: float, talking: bool = False):
        """result: the face tracker's; talking: whether you are speaking (see catcam.voice)."""
        if not result.face_landmarks:  # no face -> keep the last pose and expression
            self.ears = self.ear_motion.update(None, dt, now)
            self.idle.update(self.head, None, talking, dt, now)
            return
        self.face = landmarks_px(result.face_landmarks[0], *cam_size)
        self.last_seen = now
        matrix = result.facial_transformation_matrixes[0]
        pose = rotation_of(matrix)
        if self.take_pose:
            # Your pose when you asked for it (e.g. looking down at the screen
            # past the camera) is "facing the viewer" for the cat from now on.
            self.neutral_pose, self.take_pose = pose, False
        reference = facing_camera(matrix) if self.neutral_pose is None else self.neutral_pose
        head = to_vector(relative_rotation(pose, reference))
        self.head = self.rotation_filter(head, dt)
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        self.ears = self.ear_motion.update(scores, dt, now)
        closed = blinks(scores, self.face, self.transfer.neutral)
        self.blink = self.blink + BLINK_SMOOTHING * (closed - self.blink)
        relaxed = max(scores[name] for name in EXPRESSIVE) < RELAXED_BELOW
        offsets = EXPRESSION_GAIN * self.transfer(self.face, relaxed)
        self.offsets = self.expression_filter(offsets, dt)
        self.measured = (head, offsets, closed)
        self.idle.update(self.head, scores, talking, dt, now)


def render(cat: Cat, state: FaceState, paws: Paws, tail: Tail, tail_hand: str | None,
           background: np.ndarray | None = None) -> np.ndarray:
    """tail_hand: the hand ("Left", "Right") that lifts the tail instead of
    being a paw, if any. The tail is drawn while it sinks away, too.
    background: a picture (BGR, output size) to put behind the animal."""
    sprite = cat.eyelids.render(blink=tuple(state.blink))
    offsets = state.transfer.to_cat(state.offsets, cat.points)
    rotation = to_matrix(state.head + state.idle.head)
    renderer = cat.mesh.renderer
    renderer.keep_coverage = background is not None
    if isinstance(cat.mesh, Head3D):
        frame = cat.mesh.render(sprite, offsets, rotation, state.ears, state.idle.breath)
    else:
        frame = cat.mesh.render(sprite, offsets, rotation, state.ears)
    if background is not None:
        frame = behind(frame, renderer.coverage, background)
    if cat.tail is None:
        return paws.draw(frame, cat.paw)
    hidden = (tail_hand,) if tail_hand else ()
    return paws.draw(tail.draw(frame, cat.tail), cat.paw, hidden=hidden)


def toggle_rects(head3d: bool = True) -> dict[str, tuple[int, int, int, int]]:
    """Each switch's area in the pill (x, y, w, h, preview px); head3d: whether the
    cat has a 3D head, else its switch is left out."""
    x, y, h = PILL
    rects, x = {}, x + 6
    for name, label, _ in TOGGLES:
        if name == "head3d" and not head3d:
            continue
        w = 10 + glass.SWITCH[0] + 10 + glass.text_width(label, 15, 500) + 10 + 20 + 10
        rects[name] = (x, y, w, h)
        x += w
    return rects


def toggle_at(x: int, y: int, head3d: bool) -> str | None:
    """The switch a click at preview px (x, y) lands on, if any."""
    for name, (rx, ry, w, h) in toggle_rects(head3d).items():
        if rx <= x < rx + w and ry <= y < ry + h:
            return name
    return None


def _switches(view: np.ndarray, states: dict[str, bool]):
    rects = toggle_rects("head3d" in states)
    x, y, h = PILL
    right = max(rx + w for rx, _, w, _ in rects.values()) + 6
    glass.panel(view, x, y, right - x, h, h // 2)
    for name, label, key in TOGGLES:
        if name not in rects:
            continue
        rx, ry, _, _ = rects[name]
        glass.switch(view, rx + 10, ry + (h - glass.SWITCH[1]) // 2, states[name])
        tx = rx + 10 + glass.SWITCH[0] + 10
        glass.text(view, label, tx, ry + h / 2 + 5, 15, weight=500)
        glass.keycap(view, tx + glass.text_width(label, 15, 500) + 10, ry + (h - 20) // 2, key)


def _capsule(view: np.ndarray, name: str):
    """The bottom capsule: who you are, and the keys."""
    w, h = PREVIEW_SIZE
    height, gap = CAPSULE
    name_w = glass.text_width(name, 17, 600, "rounded")
    hint_w = [max(20, glass.text_width(key, 12, 600) + 10) + 8 + glass.text_width(label, 14)
              for key, label in KEY_HINTS]  # keycap (as wide as glass.keycap makes it), gap, label
    width = 22 + name_w + 18 + 1 + 18 + sum(hint_w) + 18 * (len(KEY_HINTS) - 1) + 22
    x, y = (w - width) // 2, h - gap - height
    glass.panel(view, x, y, width, height, height // 2)
    glass.text(view, name, x + 22, y + height / 2 + 6, 17, weight=600, family="rounded")
    divider = x + 22 + name_w + 18
    glass.paint(view, np.ones((height - 20, 1), np.float32), divider, y + 10, glass.WHITE, 0.3)
    hx = divider + 1 + 18
    for (key, label), hw in zip(KEY_HINTS, hint_w):
        cap_w = glass.keycap(view, hx, y + (height - 20) // 2, key)
        glass.text(view, label, hx + cap_w + 8, y + height / 2 + 5, 14, glass.MUTED)
        hx += hw + 18


def _message(view: np.ndarray, lines: list[str], y: int):
    """A glass panel with lines of text, centred across the window at y."""
    w, _ = PREVIEW_SIZE
    width = max(glass.text_width(line, 15, 500) for line in lines) + 48
    height = 24 + 22 * len(lines)
    x = (w - width) // 2
    glass.panel(view, x, y, width, height, min(height // 2, 22))
    for i, line in enumerate(lines):
        glass.text(view, line, w / 2, y + 12 + 16 + 22 * i, 15, weight=500, align="center")


def preview(out: np.ndarray, cat: Cat, looking: bool, notice: list[str] | None = None,
            tail: bool = False, head3d: bool | None = None, background: bool = False) -> np.ndarray:
    """Mirrored like a mirror, with the switches and hints on glass, none of which
    reach the call. head3d: whether the 3D head is on; None leaves its switch out
    (the cat has none)."""
    view = cv2.flip(cv2.resize(out, PREVIEW_SIZE, interpolation=cv2.INTER_AREA), 1)
    states = {"tail": tail} | ({} if head3d is None else {"head3d": head3d}) | {"background": background}
    _switches(view, states)
    _capsule(view, cat.name)
    if looking:
        _message(view, ["Looking for your face"], PILL[1])
    if notice:
        _message(view, notice, PREVIEW_SIZE[1] // 2 - 40)
    return view


def create_trackers():
    video = vision.RunningMode.VIDEO
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=video,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=video,
        num_hands=2,
    ))
    return face, hands


def open_webcam(index: int):
    # macOS asks for camera permission asynchronously; keep retrying while the
    # permission dialog is open.
    for _ in range(30):
        webcam = cv2.VideoCapture(index)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
            return webcam
        webcam.release()
        time.sleep(1)
    sys.exit(f"Cannot open webcam {index}. Grant camera access to {terminal_name()} in "
             "System Settings > Privacy & Security > Camera, or pick another one with --camera.")


def open_virtual_camera():
    try:
        return pyvirtualcam.Camera(OUT_W, OUT_H, FPS, fmt=pyvirtualcam.PixelFormat.BGR)
    except RuntimeError as e:
        sys.exit(f"{e}\n\nOne-time setup: open OBS, click 'Start Virtual Camera', "
                 "approve the system extension, click 'Stop Virtual Camera', quit OBS. "
                 "Keep OBS closed while this runs.")


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # macOS raises once the window is gone
        return True


class Mouse:
    """Collects mouse events from OpenCV's callback: clicks on the picker's
    cards, and where the last click was (for the preview's switches)."""

    def __init__(self, picker: Picker):
        self.picker = picker
        self.clicked = None
        self.position = None

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.picker.hover = self.picker.hit(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.clicked = self.picker.hit(x, y)
            self.position = (x, y)

    def take_click(self) -> int | None:
        """The card clicked since the last take, if any."""
        clicked, self.clicked, self.position = self.clicked, None, None
        return clicked

    def take_position(self) -> tuple[int, int] | None:
        """Where the last click since the last take was, if any."""
        position, self.clicked, self.position = self.position, None, None
        return position


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cat face filter for Zoom.")
    parser.add_argument("--camera", type=int, default=0, metavar="N",
                        help="webcam to use (default 0; try 1 if the wrong camera opens)")
    parser.add_argument("--mic", default=None, metavar="NAME",
                        help="microphone, by name or number (default: the system's; "
                             "list them: python -m catcam.voice --list)")
    parser.add_argument("--background", default=None, metavar="FILE",
                        help="a picture or video to put behind the animal")
    parser.add_argument("--check", action="store_true",
                        help="load and draw every cat once, without camera, then quit")
    return parser.parse_args(argv)


class Controls:
    """Which cat is streamed and whether the picker is open, driven by keys and clicks."""

    def __init__(self, picker: Picker, mouse: Mouse):
        self.picker, self.mouse = picker, mouse
        self.current, self.picking, self.picked_once = 0, True, False
        self.tail = False  # the tail switch
        self.head3d = True  # the 3D head switch
        self.background = False  # the background switch
        self.has_head3d = True  # whether the current cat has a 3D head (its switch shows)
        self.reaction = None  # a reaction asked for by its number key, see take_reaction
        self.quit = False

    def take_reaction(self) -> str | None:
        """The reaction asked for since the last take, if any."""
        reaction, self.reaction = self.reaction, None
        return reaction

    def handle(self, key: int, state: FaceState):
        """Number keys pick a cat in the picker, and are reactions in the call."""
        if key == ord("q"):
            self.quit = True
            return
        choice = None
        if self.picking:
            choice = self.picker.key_choice(key)
            if key == ENTER and self.picker.hover is not None:
                choice = self.picker.hover
            clicked = self.mouse.take_click()
            if clicked is not None:
                choice = clicked
            if key == ESC and self.picked_once:
                self.picking = False
        else:
            click = self.mouse.take_position()
            hit = toggle_at(*click, self.has_head3d) if click is not None else None
            if key == ord("t") or hit == "tail":
                self.tail = not self.tail
            if key == ord("d") or hit == "head3d":
                self.head3d = not self.head3d
            if key == ord("b") or hit == "background":
                self.background = not self.background
            if key == ord("c"):
                self.picking = True
                self.picker.hover = None
            elif key == ord("n"):
                state.reset_neutral()
            self.reaction = reaction_for(key) or self.reaction
        if choice is not None:
            self.current, self.picking, self.picked_once = choice, False, True


def frames(webcam):
    """Yields (frame, time) from the webcam; exits if it stops sending."""
    # The first frame can take long: macOS may still be asking for permission.
    last_frame, timeout = time.monotonic(), FIRST_FRAME_TIMEOUT
    while True:
        ok, frame = webcam.read()
        now = time.monotonic()
        if ok:
            last_frame, timeout = now, WEBCAM_TIMEOUT
            yield frame, now
        elif now - last_frame > timeout:
            sys.exit("The webcam stopped sending images (is another app using it?).")
        else:
            time.sleep(0.01)


def stream(cats: list[Cat], controls: Controls, effects: Effects, voice: Voice, trackers, vcam, webcam,
           background: Background | None = None):
    face_tracker, hand_tracker = trackers
    state, paws, tail, tail_hands = FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H)), TailHand()
    chooser, reactions = Chooser(), Reactions()
    controls.background = background is not None
    start = last = time.monotonic()
    notice_until = 0.0
    app = terminal_name()
    restart = "quit Cat Cam and open it again" if app == "Cat Cam" else f"quit and reopen {app}"
    notice = [f"Allow {app} under Input Monitoring (opened in System Settings),",
              f"then {restart}" + ("." if app == "Cat Cam" else " and start Cat Cam again.")]
    for frame, now in frames(webcam):
        dt, last = max(now - last, 1e-3), now
        cam_size = (frame.shape[1], frame.shape[0])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int((now - start) * 1000)
        state.update(face_tracker.detect_for_video(image, timestamp), cam_size, dt, now,
                     voice.syllables.speaking)
        hands = hand_tracker.detect_for_video(image, timestamp)
        cat = shown(cats[controls.current], controls.head3d)
        controls.has_head3d = cat.head3d is not None
        voice.sounds = cat.sounds
        background = _follow_switch(controls, chooser, background)
        paws.update(hands, state.face, cam_size, dt, *cat.mesh.face_on_screen())
        tail_hand = tail_hands.update(paws.slots) if controls.tail and cat.tail is not None else None
        hand = paws.slots.get(tail_hand, {}) if tail_hand else {}
        tail.update(hand.get("params"), hand.get("presence", 0.0), dt, *cat.mesh.face_on_screen())

        special = effects.update(now, cat.name)
        picture = background.frame(now) if background is not None else None
        out = special if special is not None else render(cat, state, paws, tail, tail_hand, picture)
        out = reactions.draw(out, now)
        vcam.send(out)
        if controls.picking:
            cv2.imshow(WINDOW, controls.picker.draw(controls.current if controls.picked_once else None, out))
        else:
            looking = state.last_seen is None or now - state.last_seen > NO_FACE_HINT_AFTER
            cv2.imshow(WINDOW, preview(out, cat, looking, notice if now < notice_until else None,
                                       tail=controls.tail,
                                       head3d=controls.head3d if cat.head3d is not None else None,
                                       background=controls.background))
        key = cv2.waitKey(1) & 0xFF
        if effects.needs_permission(key, now):
            notice_until = now + 10
        controls.handle(key, state)
        if (reaction := controls.take_reaction()) is not None:
            reactions.play(reaction, now)
        if controls.quit or window_closed():
            return


def _follow_switch(controls: Controls, chooser: Chooser, background: Background | None) -> Background | None:
    """The background as the switch wants it: switched on, it asks for a picture or
    video (the switch goes back off if none is chosen); switched off, it goes."""
    if not controls.background:
        if background is not None:
            background.close()
        return None
    if background is not None:
        return background
    if chooser.idle:
        chooser.start()
        return None
    path = chooser.answer()
    if path is None:  # still choosing
        return None
    if path:
        try:
            return Background(path, (OUT_W, OUT_H))
        except ValueError as e:
            print(e)
    controls.background = False
    return None


def check(cats: list[Cat]):
    """Draws every cat once at rest, with paws, as a quick test of the install."""
    for cat in cats + [shown(c, True) for c in cats if c.head3d is not None]:
        out = render(cat, FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H)), tail_hand="Right")
        if out.shape != (OUT_H, OUT_W, 3) or not out.any():
            sys.exit(f"Drawing {cat.name} failed.")
    load_effects()  # its data file must load too
    import sounddevice  # noqa: F401  (imported only when needed, so check here they're there)
    if sys.platform == "darwin":
        import Quartz  # noqa: F401
    print(f"OK: {len(cats)} cat{'' if len(cats) == 1 else 's'}")


def alert(message: str):
    """A dialog for the standalone app, which has no terminal to print to."""
    quoted = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display alert "Cat Cam" message "{quoted}"'], check=False)


def main(argv=None):
    """Runs Cat Cam; the standalone app shows errors in a dialog and a log file."""
    if not getattr(sys, "frozen", False):
        return run(argv)
    try:
        return run(argv)
    except SystemExit as e:
        if isinstance(e.code, str):
            alert(e.code)
        raise
    except Exception:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(traceback.format_exc())
        alert(f"Cat Cam stopped because of an error. Details are in {LOG_FILE}.")
        raise


def run(argv=None):
    args = parse_args(argv)
    print("Loading cats...")
    renderer = Renderer((OUT_W, OUT_H))
    cats = load_cats(renderer)
    if args.check:
        return check(cats)
    picker = Picker([(c.name, c.sprite, c.background) for c in cats])
    trackers = create_trackers()
    vcam = open_virtual_camera()
    webcam = open_webcam(args.camera)

    background = None
    if args.background:
        try:
            background = Background(args.background, (OUT_W, OUT_H))
        except ValueError as e:
            sys.exit(str(e))
    effects = Effects(load_effects(), (OUT_W, OUT_H))
    effects.start()
    voice = Voice()
    mic = VirtualMic(voice, args.mic)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    mouse = Mouse(picker)
    cv2.setMouseCallback(WINDOW, mouse)
    print(f"Streaming to '{vcam.device}'. Select it as the camera in Zoom.")
    if mic.start():
        print(f"Listening to '{mic.heard}', sending your voice to '{mic.device}'. "
              "Select it as the microphone in Zoom.")
    print("Pick a cat in the window. Keep a relaxed face for a second at the start.")
    try:
        with vcam:
            stream(cats, Controls(picker, mouse), effects, voice, trackers, vcam, webcam, background)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        effects.stop()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
