"""Measures how faithfully and steadily the animal follows a face, by replaying
recorded webcam clips through the tracking (no camera, no window needed).

    uv run python tools/tracking_eval.py record clip --seconds 30   # clip.avi + clip_t.npy
    uv run python tools/tracking_eval.py clip.avi [more.avi ...]     # the numbers
    uv run python tools/tracking_eval.py clip.avi --sheet sheet.jpg  # and a look at it

Keep clips out of git: they are your face. The face tracker's results are
cached next to each clip (<clip>.track.npz), so tuning the smoothing replays
in seconds.

What it measures, per signal (head turn, mouth opening, mouth corners, brows,
blinks), against a reference made by smoothing the unfiltered tracking forward
and backward in time (no lag, little noise):

- jitter: how much the output shakes from frame to frame while the reference
  is still, as an RMS per frame;
- lag: how far the output trails the reference while it moves, in ms (the
  shift that best lines them up), and the RMS error left while moving;
- blinks: closings the reference shows (both eyes, or one: a wink) that the
  output shows too, how late, and closings the output makes up;
- dropouts: tracking taken away for a while and given back, the largest jump
  the head then makes in one frame (a snap) and how long it takes to settle.
"""

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from catcam import app

# The reference: the unfiltered signal smoothed both ways with this Gaussian (s).
REFERENCE_SIGMA = 0.06
# A signal counts as still while it moves slower than this (per s, measured on
# the reference smoothed by another STILL_SIGMA s), as moving while the
# reference moves faster than MOVING_ABOVE times that.
STILL_SIGMA = 0.15
STILL_BELOW = {"head": math.radians(4), "mouth": 0.02, "corners": 0.04, "brows": 0.02}
MOVING_ABOVE = 3.0
LAGS = np.arange(0, 0.301, 0.005)  # s, the shifts tried when lining up output and reference
# A closing is where an eye's blink score rises BLINK_SPIKE above its median
# over the 2 s around; the output must close past BLINK_SHOWN (0..1) within
# BLINK_WINDOW s of it to count.
BLINK_SPIKE, BLINK_SHOWN, BLINK_WINDOW = 0.25, 0.6, 0.25
# Dropouts: tracking is taken away for these many s, every DROPOUT_EVERY s.
DROPOUTS, DROPOUT_EVERY = (0.3, 1.0, 2.0), 6.0
SETTLED = math.radians(1)  # the head has settled once within this of the undisturbed run

UPPER_LIP, LOWER_LIP = 13, 14
CORNERS = [61, 291]
BROWS = [105, 334]


# --- recording and tracking --------------------------------------------------------------------------

def record(path: Path, seconds: float, camera: int):
    """A clip from the webcam: path.avi (MJPEG) and path_t.npy (frame times, s)."""
    webcam = app.open_webcam(camera)
    for _ in range(10):  # let the exposure settle
        webcam.read()
    writer, times, start = None, [], time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = webcam.read()
        if not ok:
            continue
        times.append(time.monotonic() - start)
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(str(path.with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
        writer.write(frame)
    writer.release()
    webcam.release()
    np.save(path.parent / f"{path.stem}_t.npy", np.array(times))
    print(f"{len(times)} frames, {len(times) / times[-1]:.1f} fps")


def track(clip: Path, refresh: bool = False) -> dict:
    """The face tracker's results for every frame of a clip, cached next to it."""
    cache = clip.with_suffix(".track.npz")
    if cache.exists() and not refresh:
        return dict(np.load(cache))
    times = np.load(clip.parent / f"{clip.stem}_t.npy")
    tracker = app.create_trackers()[0]
    video = cv2.VideoCapture(str(clip))
    found, points, matrices, scores, names, size = [], [], [], [], None, None
    for t in times:
        ok, frame = video.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = tracker.detect_for_video(image, int(t * 1000))
        found.append(bool(result.face_landmarks))
        if result.face_landmarks:
            points.append([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
            matrices.append(np.asarray(result.facial_transformation_matrixes[0]))
            scores.append([c.score for c in result.face_blendshapes[0]])
            names = [c.category_name for c in result.face_blendshapes[0]]
        else:
            points.append(np.zeros((478, 3)))
            matrices.append(np.eye(4))
            scores.append(np.zeros(52))
    data = dict(t=times[:len(found)], found=np.array(found), points=np.array(points),
                matrices=np.array(matrices), scores=np.array(scores), names=np.array(names),
                size=np.array(size))
    np.savez_compressed(cache, **data)
    return data


def result_at(data: dict, i: int, present: bool = True):
    """Frame i of the cached tracking, shaped like the face tracker's result."""
    if not (present and data["found"][i]):
        return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[], face_blendshapes=[])
    points = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in data["points"][i]]
    shapes = [SimpleNamespace(category_name=str(n), score=float(s))
              for n, s in zip(data["names"], data["scores"][i])]
    return SimpleNamespace(face_landmarks=[points], facial_transformation_matrixes=[data["matrices"][i]],
                           face_blendshapes=[shapes])


# --- signals -------------------------------------------------------------------------------------------

def signals(head, offsets, blink) -> dict:
    """The signals compared: head rotation (rad), mouth opening, mouth corners and
    brows (face widths), eye closedness (0..1)."""
    offsets = np.asarray(offsets)
    return {
        "head": np.asarray(head, float),
        "mouth": np.array([offsets[LOWER_LIP, 1] - offsets[UPPER_LIP, 1]]),
        "corners": offsets[CORNERS, :2].ravel(),
        "brows": offsets[BROWS, 1],
        "blink": np.asarray(blink, float),
    }


def replay(data: dict, present: np.ndarray | None = None, at=None) -> tuple[dict, dict]:
    """Runs the app's face state over a clip: the signals it shows, and the same
    measured before any smoothing (frames without a face repeat the last).
    present: per frame, whether the face is given to it (default: whenever it
    was found); at(i, state) is called after each frame."""
    state, size = app.FaceState(), tuple(int(v) for v in data["size"])
    out, raw, last = {}, {}, None
    for i, now in enumerate(data["t"]):
        dt = max(now - last, 1e-3) if last is not None else 1 / 30
        last = now
        state.update(result_at(data, i, present is None or present[i]), size, dt, now)
        for signals_of, into in ((signals(state.head, state.offsets, state.blink), out),
                                 (signals(*state.measured), raw)):
            for name, value in signals_of.items():
                into.setdefault(name, []).append(value)
        if at is not None:
            at(i, state)
    return ({name: np.array(values) for name, values in out.items()},
            {name: np.array(values) for name, values in raw.items()})


def reference(raw: dict, t: np.ndarray) -> dict:
    """The unfiltered signals smoothed forward and backward: no lag, little noise."""
    sigma = REFERENCE_SIGMA / np.median(np.diff(t))
    return {name: gaussian_filter1d(values, sigma, axis=0, mode="nearest") for name, values in raw.items()}


# --- metrics -------------------------------------------------------------------------------------------

UNITS = {"head": ("deg", math.degrees(1)), "mouth": ("mfw", 1000), "corners": ("mfw", 1000),
         "brows": ("mfw", 1000)}  # mfw: thousandths of a face width


def _settled(found: np.ndarray, t: np.ndarray, after: float = 0.5) -> np.ndarray:
    """Frames with a face and no frame without one in the last `after` s."""
    lost_at = np.where(~found, t, -np.inf)
    return found & (t - np.maximum.accumulate(lost_at) > after)


def motion_metrics(out: np.ndarray, ref: np.ndarray, t: np.ndarray, ok: np.ndarray, still_below: float):
    """(jitter per frame while still, lag in s while moving, RMS error while moving, frames still, moving)."""
    calm = gaussian_filter1d(ref, STILL_SIGMA / np.median(np.diff(t)), axis=0, mode="nearest")
    still = ok & (np.linalg.norm(np.gradient(calm, t, axis=0), axis=1) < still_below)
    moving = ok & (np.linalg.norm(np.gradient(ref, t, axis=0), axis=1) > MOVING_ABOVE * still_below)
    # Shaking is what changes direction from frame to frame: the second difference,
    # which a steady drift doesn't have.
    triples = still[2:] & still[1:-1] & still[:-2]
    shake = np.linalg.norm(np.diff(out, 2, axis=0), axis=1)[triples]
    jitter = float(np.sqrt(np.mean(shake ** 2))) if len(shake) else math.nan
    if moving.sum() < 5:
        return jitter, math.nan, math.nan, int(still.sum()), int(moving.sum())
    errors = []
    for lag in LAGS:
        shifted = np.column_stack([np.interp(t - lag, t, ref[:, k]) for k in range(ref.shape[1])])
        errors.append(np.sqrt(np.mean(np.sum((out - shifted)[moving] ** 2, axis=1))))
    return jitter, float(LAGS[int(np.argmin(errors))]), float(errors[0]), int(still.sum()), int(moving.sum())


def blink_events(data: dict, ok: np.ndarray) -> list[list[tuple[int, int]]]:
    """Per eye (image-left, image-right), the frames (start, end) where its blink
    score jumps well above its own level of the seconds around: a closing."""
    names = [str(n) for n in data["names"]]
    size = max(int(2.0 / np.median(np.diff(data["t"]))), 3)
    events = []
    for name in ("eyeBlinkRight", "eyeBlinkLeft"):  # the person's right eye is on the image's left
        score = data["scores"][:, names.index(name)]
        spike = ok & (score - median_filter(score, size, mode="nearest") > BLINK_SPIKE)
        edges = np.flatnonzero(np.diff(np.r_[0, spike.astype(int), 0]))
        events.append(list(zip(edges[::2], edges[1::2])))
    return events


def blink_metrics(out: np.ndarray, data: dict, ok: np.ndarray) -> dict:
    """Closings shown (of those the scores show), how late and how deep, closings
    made up, and how closed the eyes are on average away from any closing."""
    t = data["t"]
    events = blink_events(data, ok)
    window = int(round(BLINK_WINDOW / np.median(np.diff(t))))
    total, delays, depths, made_up = 0, [], [], 0
    near = np.zeros(len(t), bool)
    for eye, eye_events in enumerate(events):
        shown = out[:, eye] > BLINK_SHOWN
        covered = np.zeros(len(t), bool)
        for start, end in eye_events:
            total += 1
            covered[max(start - window, 0):end + window] = True
            depths.append(out[start:end + window, eye].max())
            after = np.flatnonzero(shown[start:end + window])
            if len(after):
                delays.append(t[start + after[0]] - t[start])
        near |= covered
        rises = np.flatnonzero(np.diff(shown.astype(int)) == 1) + 1
        made_up += int(np.sum(~covered[rises] & ok[rises]))
    calm = ok & ~near
    return {"blinks": total, "shown": len(delays), "delay_ms": 1000 * np.mean(delays) if delays else math.nan,
            "depth": float(np.mean(depths)) if depths else math.nan, "made_up": made_up,
            "closed_at_rest": float(np.mean(out[calm])) if calm.any() else math.nan}


def dropout_mask(data: dict) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Faces taken away for each of DROPOUTS in turn, every DROPOUT_EVERY s, only
    where the face was found all along; returns the mask and the gaps (start, end)."""
    t, found = data["t"], data["found"]
    present, gaps, k, at = found.copy(), [], 0, 3.0
    while at + DROPOUT_EVERY < t[-1]:
        length = DROPOUTS[k % len(DROPOUTS)]
        start, end = np.searchsorted(t, at), np.searchsorted(t, at + length)
        after = np.searchsorted(t, at + length + 1.5)
        if found[start:after].all():
            present[start:end] = False
            gaps.append((start, end))
            k += 1
        at += DROPOUT_EVERY
    return present, gaps


def dropout_metrics(full: np.ndarray, gapped: np.ndarray, t: np.ndarray, gaps) -> dict:
    """Head (rad) without and with the dropouts: the largest jump in one frame in
    the 0.5 s after the face is back (deg; the undisturbed run's for comparison)
    and how long until the head is back where it would have been (s)."""
    snaps, normal, settle = [], [], []
    for _, end in gaps:
        stop = np.searchsorted(t, t[end] + 0.5)
        snaps.append(np.degrees(np.linalg.norm(np.diff(gapped[end - 1:stop], axis=0), axis=1)).max())
        normal.append(np.degrees(np.linalg.norm(np.diff(full[end - 1:stop], axis=0), axis=1)).max())
        horizon = np.searchsorted(t, t[end] + 1.5)
        off = np.flatnonzero(np.linalg.norm(gapped[end:horizon] - full[end:horizon], axis=1) > SETTLED)
        settle.append(t[end + off[-1] + 1] - t[end] if len(off) else 0.0)
    return {"gaps": len(gaps), "snap_deg": float(np.max(snaps)), "normal_deg": float(np.max(normal)),
            "settle_s": float(np.mean(settle))}


def evaluate(clip: Path) -> dict:
    data = track(clip)
    t, found = data["t"], data["found"]
    report = {"frames": len(t), "fps": float(1 / np.median(np.diff(t))), "found": float(found.mean())}
    if found.sum() < 10:  # nothing to measure (e.g. nobody in front of the camera)
        return report
    ok = _settled(found, t)
    out, raw = replay(data)
    ref = reference(raw, t)
    for name, (unit, scale) in UNITS.items():
        jitter, lag, error, still, moving = motion_metrics(out[name], ref[name], t, ok, STILL_BELOW[name])
        raw_jitter = motion_metrics(raw[name], ref[name], t, ok, STILL_BELOW[name])[0]
        report[name] = {"jitter": scale * jitter, "raw_jitter": scale * raw_jitter, "lag_ms": 1000 * lag,
                        "error": scale * error, "unit": unit, "still": still, "moving": moving}
    report["blink"] = blink_metrics(out["blink"], data, ok)
    present, gaps = dropout_mask(data)
    if gaps:
        report["dropout"] = dropout_metrics(out["head"], replay(data, present)[0]["head"], t, gaps)
    return report


# --- looking at it -------------------------------------------------------------------------------------

def sheet(clip: Path, path: Path, cat_name: str | None = None, frames=(), count: int = 12):
    """Webcam frame and the animal side by side, for the frames asked for, the
    frames where the mouth is widest, an eye most closed, the head turned
    furthest, and evenly between."""
    from catcam.cat_mesh import Renderer
    from catcam.paws import Paws
    from catcam.tail import Tail

    data = track(clip)
    raw = replay(data)[1]
    found = np.flatnonzero(data["found"])
    picks = set(frames) or {int(found[np.argmax(raw["mouth"][found, 0])]),
                            int(found[np.argmax(raw["blink"][found].max(1))]),
                            int(found[np.argmax(np.linalg.norm(raw["head"][found], axis=1))])}
    if not frames:
        picks |= {int(i) for i in np.linspace(found[0] + 10, len(data["t"]) - 1, count - len(picks))}
    cats = app.load_cats(Renderer((app.OUT_W, app.OUT_H)))
    cat = app.shown(next((c for c in cats if c.name == cat_name), cats[0]), True)
    video, tiles = cv2.VideoCapture(str(clip)), []

    def draw(i, state):
        ok, frame = video.read()
        if i not in picks or not ok:
            return
        drawn = app.render(cat, state, Paws(app.OUT_W), Tail((app.OUT_W, app.OUT_H)), None)
        face = cv2.convertScaleAbs(frame, alpha=2.0, beta=10)  # the clips are often dark
        face = cv2.resize(face, (face.shape[1] * drawn.shape[0] // face.shape[0], drawn.shape[0]))
        tile = cv2.resize(np.hstack([face, drawn]), (960, 270))
        cv2.putText(tile, f"{i}  {data['t'][i]:.2f} s" + ("" if data["found"][i] else "  no face"), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    replay(data, at=draw)
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
    print(f"  sheet: {path}")


def print_report(clip: Path, report: dict):
    print(f"\n{clip.name}: {report['frames']} frames at {report['fps']:.1f} fps, face found in "
          f"{100 * report['found']:.0f}%")
    if "head" not in report:
        return
    print(f"  {'signal':8} {'jitter':>8} {'(raw)':>8} {'lag ms':>7} {'error':>8}  unit  frames still/moving")
    for name in UNITS:
        r = report[name]
        print(f"  {name:8} {r['jitter']:8.3f} {r['raw_jitter']:8.3f} {r['lag_ms']:7.0f} {r['error']:8.3f}  "
              f"{r['unit']:4}  {r['still']}/{r['moving']}")
    b = report["blink"]
    print(f"  blinks: {b['shown']}/{b['blinks']} shown, {b['delay_ms']:.0f} ms late, as deep as "
          f"{b['depth']:.2f}; {b['made_up']} made up; eyes {b['closed_at_rest']:.3f} closed at rest")
    if "dropout" in report:
        d = report["dropout"]
        print(f"  dropouts: {d['gaps']}, largest jump after {d['snap_deg']:.2f} deg/frame "
              f"(undisturbed {d['normal_deg']:.2f}), back on track after {d['settle_s']:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", help="clips (.avi), or 'record' and a name")
    parser.add_argument("--seconds", type=float, default=30, help="how long to record")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--retrack", action="store_true", help="run the face tracker again")
    parser.add_argument("--sheet", type=Path, help="also render the animal for some frames into this image")
    parser.add_argument("--cat", default=None, help="the animal for --sheet (default: the first)")
    parser.add_argument("--frames", default="", help="frames for --sheet, e.g. 90,95,100 (default: picked)")
    args = parser.parse_args()
    if args.clips[0] == "record":
        return record(Path(args.clips[1]), args.seconds, args.camera)
    for clip in map(Path, args.clips):
        if args.retrack:
            track(clip, refresh=True)
        report = evaluate(clip)
        print_report(clip, report)
        if args.sheet and "head" in report:
            frames = [int(f) for f in args.frames.split(",") if f]
            sheet(clip, args.sheet.with_stem(f"{args.sheet.stem}_{clip.stem}"), args.cat, frames)


if __name__ == "__main__":
    main()
