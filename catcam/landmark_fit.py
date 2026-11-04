"""Finds the 478 MediaPipe face landmarks on a cat portrait.

The cats have human-like faces, so the human face landmark model fits them,
but its face *detector* misses heavily striped or furry cats. The landmarker's
video mode only runs the detector on the first frame and then tracks, so the
tracker is primed with a portrait the detector does find, and that portrait is
then cross-faded into the target over a few frames: the landmarks follow it
onto the target's face.

Several runs (different primers, smoothed, mirrored) are combined by their
median, and the result is made left/right symmetric, since the portraits are
frontal. That removes most of the tilt and drift single runs have on fur.
"""

from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from catcam.paths import FACE_MODEL

# A detection only counts if it looks like the cat's face: wide enough
# relative to the whole head, and centred on it.
MIN_FACE_WIDTH = 0.45  # of the head width
MAX_NOSE_OFFSET = 0.15  # of the head width, from the head's centre line
FADE_FRAMES, TRACK_FRAMES, KEEP_FRAMES = 15, 30, 10
MAX_PRIMERS = 3
NOSE_TIP = 1


class FitError(RuntimeError):
    pass


class Portrait(NamedTuple):
    bgr: np.ndarray  # (h, w, 3)
    bbox: np.ndarray  # head bounding box (x0, y0, x1, y1) from the cutout


def head_bbox(alpha: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(alpha > 128)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def mirror_points(points: np.ndarray, mirror: np.ndarray, axis_x: float) -> np.ndarray:
    """Landmarks of the mirror image (about x = axis_x), renumbered so each
    index is the same facial feature again (left eye stays the left eye)."""
    mirrored = points[mirror].copy()
    mirrored[:, 0] = 2 * axis_x - mirrored[:, 0]
    return mirrored


def symmetrize(points: np.ndarray, mirror: np.ndarray) -> np.ndarray:
    """Average a face with its own mirror image about its centre line."""
    return (points + mirror_points(points, mirror, points[:, 0].mean())) / 2


def plausible(points: np.ndarray, bbox: np.ndarray) -> bool:
    head_width = bbox[2] - bbox[0]
    centre = (bbox[0] + bbox[2]) / 2
    nose_x, nose_y = points[NOSE_TIP, :2]
    return (
        np.ptp(points[:, 0]) >= MIN_FACE_WIDTH * head_width
        and abs(nose_x - centre) <= MAX_NOSE_OFFSET * head_width
        and bbox[1] <= nose_y <= bbox[3]
    )


def _mp_image(bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _points(result, width: int, height: int) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    return np.array([(p.x * width, p.y * height, p.z * width) for p in result.face_landmarks[0]])


def detects_directly(portrait: Portrait) -> bool:
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)))
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        h, w = portrait.bgr.shape[:2]
        points = _points(landmarker.detect(_mp_image(portrait.bgr)), w, h)
    return points is not None and plausible(points, portrait.bbox)


def _flip(portrait: Portrait) -> Portrait:
    w = portrait.bgr.shape[1]
    x0, y0, x1, y1 = portrait.bbox
    return Portrait(np.ascontiguousarray(portrait.bgr[:, ::-1]), np.array([w - 1 - x1, y0, w - 1 - x0, y1]))


def _track(primer: Portrait, target: Portrait) -> np.ndarray | None:
    """Prime the tracker on `primer` (moved onto the target's head) and
    cross-fade to the target. Returns the median of the last frames."""
    h, w = target.bgr.shape[:2]
    pw, ph = primer.bbox[2:] - primer.bbox[:2]
    tw, th = target.bbox[2:] - target.bbox[:2]
    s = min(tw / pw, th / ph)
    move = np.array([
        [s, 0, target.bbox[0] - s * primer.bbox[0]],
        [0, s, target.bbox[1] - s * primer.bbox[1]],
    ])
    start = cv2.warpAffine(primer.bgr, move, (w, h), borderMode=cv2.BORDER_REPLICATE)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        # Cats score low as human faces; never drop the track because of that.
        min_face_presence_confidence=0.0,
        min_tracking_confidence=0.0,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        first = _points(landmarker.detect_for_video(_mp_image(start), 0), w, h)
        if first is None or not plausible(first, target.bbox):
            return None
        kept = []
        for frame in range(1, TRACK_FRAMES + 1):
            fade = min(frame / FADE_FRAMES, 1.0)
            image = cv2.addWeighted(target.bgr, fade, start, 1 - fade, 0)
            points = _points(landmarker.detect_for_video(_mp_image(image), frame * 33), w, h)
            if points is None:
                return None
            if frame > TRACK_FRAMES - KEEP_FRAMES:
                kept.append(points)
    return np.median(kept, axis=0)


def fit_face(target: Portrait, primers: list[Portrait], mirror: np.ndarray) -> np.ndarray:
    """(478, 3) landmarks of `target` in its pixel coords (z uses the x scale)."""
    w = target.bgr.shape[1]
    smoothed = Portrait(cv2.bilateralFilter(target.bgr, 15, 80, 15), target.bbox)
    runs = []
    for version in (target, smoothed):
        for flipped in (False, True):
            subject = _flip(version) if flipped else version
            for primer in primers[:MAX_PRIMERS]:
                points = _track(primer, subject)
                if points is None:
                    continue
                runs.append(mirror_points(points, mirror, (w - 1) / 2) if flipped else points)
    if not runs:
        raise FitError("could not find the cat's face")
    return symmetrize(np.median(runs, axis=0), mirror)
