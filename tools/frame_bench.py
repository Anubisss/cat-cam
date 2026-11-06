"""Frame-time benchmark: Cat Cam's per-frame work without anyone in front of the camera.

Runs what the app does for every webcam frame (tracking, your face's state,
paws and tail, drawing the animal, the preview) for each animal, on a recorded
video or on blank frames that arrive like a webcam's (FPS a second, the latest
one wins). A scripted face and hand drive the animal, so every run is the
same: the head turns and nods, the mouth talks, the eyes blink and half close
as when you look down at the screen, the brows move the ears, and a hand comes
and goes. The trackers still run on the video's frames, for their cost.

Prints, per animal, each stage's time and the frame-time percentiles: "work"
is how long a frame keeps the main loop busy, "interval" the time between two
frames going out (what looks choppy when it jumps) and "missed" the webcam
frames never drawn. Also counts the garbage collector's pauses.

    uv run python tools/frame_bench.py                         # every 3D head, blank frames
    uv run python tools/frame_bench.py --video clip.mp4        # trackers on a recording
    uv run python tools/frame_bench.py --record clip.mp4       # record one from the webcam (10 s)
    uv run python tools/frame_bench.py --cats Woofgang --flat  # the photo cat instead of the 3D head
    uv run python tools/frame_bench.py --video clip.mp4 --paws hands  # your real hands, cut out of the video
    uv run python tools/frame_bench.py --save frames/          # also write a few rendered frames
"""

import argparse
import gc
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from catcam import app
from catcam.app import OUT_H, OUT_W, FaceState, load_cats, shown
from catcam.cat_face import CatFace
from catcam.cat_mesh import CatMesh, Renderer
from catcam.fx import Effects
from catcam.head3d import Head3D
from catcam.head_pose import to_matrix
from catcam.paws import Paws
from catcam.picker import Picker
from catcam.real_hands import RealHands, Segmenter
from catcam.tail import Tail
from catcam.voice import BLOCK, RATE, Voice

FPS = 30  # how often the pretend webcam sends a frame
PRELOAD = 120  # video frames kept in memory and looped (decoding is not what is measured)
SAVE_EVERY = 45  # with --save, every this many frames is written
SLOWEST = 3  # the slowest frames of each animal are listed with their stages
# The scripted face: periods (s) of its head turn, nod and tilt and how far (rad)...
TURN, NOD, TILT = (4.0, 0.35), (3.1, 0.15), (5.3, 0.1)
TALK = (0.9, 0.12)  # ...how often the mouth opens and how far (face widths)...
BLINK_EVERY, BLINK_TIME = 2.7, 0.15  # ...blinks (s)...
LOOK_DOWN = (7.0, 9.0)  # ...and from when to when in each LOOP the eyes are half shut
LOOP = 12.0  # s after which the script starts over
HAND_IN = (3.0, 8.0)  # s in each LOOP the hand is in view
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
_FLIP = np.diag([1.0, -1.0, -1.0])


class Webcam:
    """Stands in for cv2.VideoCapture: frames of a video (looped) or blank ones,
    coming FPS a second like a webcam's. read() gives the newest frame, waiting
    for one if the last was already taken, and counts the ones nobody read."""

    def __init__(self, video: str | None, fps: float):
        self.frames = []
        if video:
            capture = cv2.VideoCapture(video)
            while len(self.frames) < PRELOAD:
                ok, frame = capture.read()
                if not ok:
                    break
                self.frames.append(cv2.resize(frame, (OUT_W, OUT_H)) if frame.shape[:2] != (OUT_H, OUT_W)
                                   else frame)
            capture.release()
            if not self.frames:
                sys.exit(f"Cannot read {video}.")
        else:
            self.frames = [np.full((OUT_H, OUT_W, 3), 60, np.uint8)]
        self.fps, self.missed = fps, 0
        self.restart()

    def restart(self):
        """Starts sending afresh, as when the webcam is opened."""
        self.start, self.taken = time.monotonic(), -1

    def read(self):
        if self.fps <= 0:  # as fast as the reader can take them
            self.taken += 1
            return True, self.frames[self.taken % len(self.frames)].copy()
        newest = int((time.monotonic() - self.start) * self.fps)
        if newest <= self.taken:
            newest = self.taken + 1
            time.sleep(max(self.start + newest / self.fps - time.monotonic(), 0))
        self.missed += newest - self.taken - 1
        self.taken = newest
        return True, self.frames[newest % len(self.frames)].copy()

    def release(self):
        pass


def _smooth_pulse(t: float, start: float, end: float, edge: float = 0.08) -> float:
    """0 outside [start, end], 1 inside, with soft edges (s)."""
    return float(np.clip(min(t - start, end - t) / edge, 0, 1))


class Script:
    """A face and a hand as MediaPipe would report them, moving with time."""

    def __init__(self, face: np.ndarray):
        """face: (478, 3) landmarks of a neutral face (any px), e.g. a cat's."""
        width = np.ptp(face[:, 0])
        scale = 0.22 * OUT_W / width  # a face 22% of the frame wide, as at a laptop
        self.face = (face - face.mean(axis=0)) * scale
        self.face_width = width * scale
        self.lip = self.face[[13, 14], 1].mean()

    def face_result(self, t: float):
        t %= LOOP
        talk = max(np.sin(2 * np.pi * t / TALK[0]), 0) ** 2
        points = self.face.copy()
        below = np.clip((points[:, 1] - self.lip) / (0.15 * self.face_width), 0, 1)
        points[:, 1] += TALK[1] * talk * self.face_width * below
        points[:, :2] += (OUT_W / 2, OUT_H * 0.45)
        landmarks = [SimpleNamespace(x=x / OUT_W, y=y / OUT_H, z=z / OUT_W) for x, y, z in points]
        blink = 1.0 if (t % BLINK_EVERY) < BLINK_TIME else 0.0
        blink = max(blink, 0.6 * _smooth_pulse(t, *LOOK_DOWN, edge=0.5))
        scores = dict.fromkeys(BLENDSHAPES, 0.0) | {
            "jawOpen": talk, "eyeBlinkLeft": blink, "eyeBlinkRight": blink,
            "browInnerUp": 0.8 * _smooth_pulse(t, 1.0, 2.0, 0.3),
            "browDownLeft": 0.7 * _smooth_pulse(t, 9.5, 10.5, 0.3),
            "browDownRight": 0.7 * _smooth_pulse(t, 9.5, 10.5, 0.3),
        }
        angles = [NOD[1] * np.sin(2 * np.pi * t / NOD[0]), TURN[1] * np.sin(2 * np.pi * t / TURN[0]),
                  TILT[1] * np.sin(2 * np.pi * t / TILT[0])]
        matrix = np.eye(4)
        matrix[:3, :3] = _FLIP @ to_matrix(angles) @ _FLIP
        matrix[:3, 3] = (0.0, 0.0, -40.0)  # cm in front of the camera
        blendshapes = [SimpleNamespace(category_name=n, score=s) for n, s in scores.items()]
        return SimpleNamespace(face_landmarks=[landmarks], face_blendshapes=[blendshapes],
                               facial_transformation_matrixes=[matrix])

    def hand_result(self, t: float):
        t %= LOOP
        if not HAND_IN[0] <= t <= HAND_IN[1]:
            return SimpleNamespace(hand_landmarks=[], handedness=[])
        # An open right hand, palm to the camera, waving beside the face (px, wrist first).
        wave = 0.3 * np.sin(2 * np.pi * t / 1.5)
        size = 0.9 * self.face_width
        bases = np.array([(-0.4, -0.3), (-0.25, -0.9), (-0.05, -0.95), (0.15, -0.9), (0.32, -0.8)])
        points = [(0.0, 0.0)]
        for i, (x, y) in enumerate(bases):
            for k in range(1, 5):
                reach = 0.45 if i == 0 else 0.25
                points.append((x * (1 + 0.1 * k) + (0.12 * k if i == 0 else 0), y - (k - 1) * reach))
        c, s = np.cos(wave), np.sin(wave)
        pts = np.array(points) @ np.array([[c, s], [-s, c]]) * size
        pts += (OUT_W / 2 - 1.3 * self.face_width, OUT_H * 0.75)
        landmarks = [SimpleNamespace(x=x / OUT_W, y=y / OUT_H, z=0.0) for x, y in pts]
        return SimpleNamespace(hand_landmarks=[landmarks],
                               handedness=[[SimpleNamespace(category_name="Right", score=0.9)]])


class Timings:
    """Per-frame stage times (ms), plus the garbage collector's pauses."""

    def __init__(self):
        self.frames: list[dict] = []
        self.current = defaultdict(float)
        self.gc_pauses: list[float] = []
        self._gc_start = None
        gc.callbacks.append(self._gc)

    def _gc(self, phase, info):
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif self._gc_start is not None:
            pause = (time.perf_counter() - self._gc_start) * 1000
            self.gc_pauses.append(pause)
            self.current["gc"] += pause

    def add(self, stage: str, ms: float):
        self.current[stage] += ms

    def next_frame(self):
        self.frames.append(dict(self.current))
        self.current = defaultdict(float)

    def close(self):
        gc.callbacks.remove(self._gc)


def watch(timings: Timings, owner, name: str, stage: str):
    """Times every call of owner.name (a method or a module's function) as stage."""
    original = getattr(owner, name)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timings.add(stage, (time.perf_counter() - start) * 1000)

    setattr(owner, name, timed)
    return lambda: setattr(owner, name, original)


class VoiceLoad:
    """Runs a Voice on a talking-like signal every audio block on its own thread,
    as the microphone's callback does, and records how late each block is."""

    def __init__(self, sounds: list):
        self.voice = Voice(np.random.default_rng(0))
        self.voice.sounds = sounds
        self.stop_event, self.late = threading.Event(), []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        rng = np.random.default_rng(1)
        period, n = BLOCK / RATE, 0
        start = time.perf_counter()
        while not self.stop_event.is_set():
            t = n * period
            loud = 0.2 if (t % 0.25) < 0.15 else 0.002  # syllables four times a second
            block = (loud * np.sin(2 * np.pi * 150 * (t + np.arange(BLOCK) / RATE))
                     + 0.001 * rng.standard_normal(BLOCK)).astype(np.float32)
            self.voice.process(block)
            n += 1
            wait = start + n * period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                self.late.append(-wait * 1000)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()


def percentiles(values) -> str:
    v = np.asarray(values, float)
    if not len(v):
        return "-"
    p50, p95, p99 = np.percentile(v, [50, 95, 99])
    return f"{p50:6.1f} {p95:6.1f} {p99:6.1f} {v.max():6.1f}"


def report(name: str, timings: Timings, intervals: list[float], missed: int, skipped: int,
           tracking: list[float], voice_late=None):
    frames = timings.frames
    stages = sorted({k for f in frames for k in f}, key=lambda k: (k != "work", k))
    print(f"\n{name}: {len(frames)} frames, {missed} webcam frames missed, {skipped} tracked but "
          f"skipped, {len(timings.gc_pauses)} GC pauses (max {max(timings.gc_pauses, default=0):.1f} ms)")
    print(f"  {'ms':<14}{'p50':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    for stage in stages:
        print(f"  {stage:<14}{percentiles([f.get(stage, 0.0) for f in frames])}")
    print(f"  {'interval':<14}{percentiles(intervals)}")
    print(f"  {'tracking':<14}{percentiles(tracking)}   (on its own thread)")
    for frame in sorted(frames, key=lambda f: -f.get("work", 0))[:SLOWEST]:
        biggest = sorted(frame.items(), key=lambda kv: -kv[1])
        stages = ", ".join(f"{k} {v:.1f}" for k, v in biggest if v >= 0.5)
        print(f"  slow frame: {stages}")
    if voice_late is not None:
        print(f"  voice blocks late: {len(voice_late)} (max {max(voice_late, default=0):.1f} ms)")


class ScriptedTracker(app.Tracker):
    """The app's Tracker: it still tracks the webcam's frames, for the cost, but hands
    on the script's face and hand, so that every run moves the same. Also times the
    main loop: a frame's work runs from getting its tracking to asking for the next.
    Real hands are cut out where the video's hands really are (none on blank frames),
    and drawn for the script's hand."""

    def __init__(self, webcam, script: Script, timings: Timings):
        # Fresh trackers: their clocks restart.
        super().__init__(webcam, app.create_trackers(), Segmenter())
        self.script, self.timings = script, timings
        self.tracking: list[float] = []
        self.handed = 0
        self.began = None
        self.real_hands = None

    def track(self, frame, timestamp):
        start = time.perf_counter()
        _, self.real_hands = super().track(frame, timestamp)
        self.tracking.append((time.perf_counter() - start) * 1000)
        t = timestamp / 1000
        return self.script.face_result(t), self.script.hand_result(t)

    def cut(self, frame, segmenting, hands, colours):
        start = time.perf_counter()
        cuts = super().cut(frame, segmenting, self.real_hands, colours)
        self.tracking[-1] += (time.perf_counter() - start) * 1000  # waiting for the segmenter, then cutting
        return cuts

    def next(self):
        if self.began is not None:
            self.timings.add("work", (time.perf_counter() - self.began) * 1000)
            self.timings.next_frame()
        tracked = super().next()
        self.began, self.handed = time.perf_counter(), self.handed + 1
        return tracked


class Output:
    """Stands in for the virtual camera: notes when each frame goes out, and asks
    the loop to stop after so many."""

    def __init__(self, controls: app.Controls, frames: int, save: Path | None):
        self.controls, self.frames, self.save = controls, frames, save
        self.sent: list[float] = []
        self.kept: list[np.ndarray] = []

    def send(self, frame: np.ndarray):
        self.sent.append(time.perf_counter())
        if self.save is not None and len(self.sent) % SAVE_EVERY == 1:
            self.kept.append(frame.copy())
        if len(self.sent) >= self.frames:
            self.controls.quit = True


def run_cat(cats: list, index: int, flat: bool, script: Script, webcam: Webcam, picker: Picker, frames: int,
            save: Path | None, voice: bool, window: bool, paws: str = "3d", tail: bool = True) -> dict:
    """The app's loop (catcam.app.stream) for one animal, timed."""
    cat = shown(cats[index], not flat)
    timings = Timings()
    undo = [watch(timings, CatFace, "render", "eyelids"),
            watch(timings, Head3D, "_upload", "texture"),
            watch(timings, Head3D, "_shape", "shape"),
            watch(timings, CatMesh, "_shape", "shape"),
            watch(timings, Renderer, "read", "readback"),
            watch(timings, Paws, "draw", "paws_draw"),
            watch(timings, RealHands, "draw", "paws_draw"),
            watch(timings, RealHands, "update", "state"),
            watch(timings, Tail, "draw", "tail_draw"),
            watch(timings, FaceState, "update", "state"),
            watch(timings, Paws, "update", "state"),
            watch(timings, Tail, "update", "state"),
            watch(timings, app, "render", "render"),
            watch(timings, app, "preview", "preview"),
            watch(timings, cv2, "imshow", "show")]
    if not window:
        undo += [stub(app, "window_closed", lambda: False), stub(cv2, "imshow", lambda *a: None),
                 stub(cv2, "waitKey", lambda *a: -1)]
    controls = app.Controls(picker, app.Mouse(picker))
    controls.current, controls.picking, controls.picked_once = index, False, True
    controls.head3d, controls.tail, controls.paws = not flat, tail, paws
    output = Output(controls, frames, save)
    tracker = ScriptedTracker(webcam, script, timings)
    webcam.restart()
    missed_before = webcam.missed
    voice_load = VoiceLoad(cat.sounds) if voice else None
    real_state = FaceState.__init__

    def seeded(state):  # the same twitches and glances every run
        real_state(state)
        state.idle.rng, state.ear_motion.rng = np.random.default_rng(2), np.random.default_rng(3)

    undo.append(stub(FaceState, "__init__", seeded))
    if voice_load:
        voice_load.__enter__()
    try:
        tracker.start()
        app.stream(cats, controls, Effects({}, (OUT_W, OUT_H)), voice_load.voice if voice_load else Voice(),
                   tracker, output)
    finally:
        tracker.stop()
        if voice_load:
            voice_load.__exit__(None, None, None)
        for u in reversed(undo):
            u()
        timings.close()
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        for k, out in enumerate(output.kept):
            kind = ("flat" if flat else "3d") + ("" if paws == "3d" else f"_{paws}")
            cv2.imwrite(str(save / f"{cat.name}_{kind}_{k * SAVE_EVERY:04d}.png"), out)
    missed = webcam.missed - missed_before
    intervals = list(np.diff(output.sent) * 1000)[1:]
    skipped = len(tracker.tracking) - tracker.handed
    report(cat.name, timings, intervals, missed, skipped, tracker.tracking,
           voice_load.late if voice_load else None)
    work = [f["work"] for f in timings.frames[1:]]
    return {"work": work, "intervals": intervals, "missed": missed + skipped,
            "first": timings.frames[0]["work"]}


def stub(owner, name: str, replacement):
    """Puts replacement in place of owner.name; returns what undoes it."""
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    return lambda: setattr(owner, name, original)


def record(path: str, seconds: float, camera: int):
    webcam = app.open_webcam(camera)
    writer = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ok, frame = webcam.read()
        if not ok:
            continue
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, frame.shape[1::-1])
        writer.write(frame)
    writer.release()
    webcam.release()
    print(f"Recorded {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", help="video to track on (looped); default: blank frames")
    parser.add_argument("--record", metavar="FILE", help="record a clip from the webcam to FILE, then quit")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to record")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cats", nargs="*", help="animals to run (default: all with a 3D head)")
    parser.add_argument("--flat", action="store_true", help="draw the photo cats, not the 3D heads")
    parser.add_argument("--paws", choices=[mode for mode, _ in app.PAW_MODES], default="3d",
                        help="the paws: 3D, photo, or your real hands (cut out of --video)")
    parser.add_argument("--no-tail", action="store_true",
                        help="the tail off, so a hand alone is a paw (or your real hand), not the tail")
    parser.add_argument("--frames", type=int, default=300, help="frames per animal")
    parser.add_argument("--fps", type=float, default=FPS, help="webcam frame rate; 0 = as fast as possible")
    parser.add_argument("--voice", action="store_true", help="also run the voice on its thread")
    parser.add_argument("--save", type=Path, help="write some rendered frames to this folder")
    parser.add_argument("--window", action="store_true", help="show the preview window, as the app does")
    args = parser.parse_args(argv)
    if args.record:
        return record(args.record, args.seconds, args.camera)

    print("Loading cats...")
    renderer = Renderer((OUT_W, OUT_H))
    cats = load_cats(renderer)
    chosen = [i for i, c in enumerate(cats) if (args.cats is None or c.name in args.cats)
              and (args.flat or c.head3d is not None)]
    if not chosen:
        sys.exit("No such animals.")
    script = Script(cats[chosen[0]].points)
    picker = Picker([(c.name, c.sprite, c.background) for c in cats])
    webcam = Webcam(args.video, args.fps)
    app.settle()
    if args.window:
        cv2.namedWindow(app.WINDOW, cv2.WINDOW_AUTOSIZE)
    results = {}
    for index in chosen:
        results[cats[index].name] = run_cat(cats, index, args.flat, script, webcam, picker, args.frames,
                                            args.save, args.voice, args.window, args.paws, not args.no_tail)
    if args.window:
        cv2.destroyAllWindows()
    work = np.concatenate([r["work"] for r in results.values()])
    intervals = np.concatenate([r["intervals"] for r in results.values()])
    print(f"\nAll animals ({len(work)} frames, {sum(r['missed'] for r in results.values())} missed):")
    print(f"  {'ms':<14}{'p50':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    print(f"  {'work':<14}{percentiles(work)}")
    print(f"  {'interval':<14}{percentiles(intervals)}")
    print(f"  {'first frame':<14}{percentiles([r['first'] for r in results.values()])}")
    over = np.mean(intervals > 1000 / FPS * 1.5) * 100
    print(f"  frames out more than 1.5 webcam frames apart: {over:.1f}%")


if __name__ == "__main__":
    main()
