"""Cat face filter for Zoom.

Tracks your face and hands on the webcam and renders the cat as a 3D head mesh
filling the screen: it turns, nods and tilts with your head, copies your whole
face (brows, eyes, cheeks, lips, jaw) and your hands become paws. Only the
rendered cat goes out to the virtual camera; no webcam pixels are ever sent, so
your face is never visible.

Run:  uv run cat_cam.py      then pick "OBS Virtual Camera" in Zoom.
Keys in the preview window: q = quit, n = "this is my neutral face" (recalibrate).
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

from cat_face import CatFace
from cat_mesh import CatMesh
from face_motion import ExpressionTransfer
from head_pose import OneEuroFilter, relative_rotation, rotation_of, to_matrix, to_vector
from paws import Paws

HERE = Path(__file__).parent
OUT_W, OUT_H, FPS = 1280, 720, 30
FACE_MODEL = str(HERE / "face_landmarker.task")
HAND_MODEL = str(HERE / "hand_landmarker.task")

EXPRESSION_GAIN = 1.3  # exaggerate your expressions on the cat (1 = 1:1)
BLINK_SMOOTHING = 0.7  # 0..1, higher = snappier; kept snappy so blinks show
# One Euro filters: lower MIN_CUTOFF = less jitter when still, higher BETA =
# less lag when moving fast.
ROTATION_MIN_CUTOFF, ROTATION_BETA = 0.8, 3.5
EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA = 3.0, 0.3

# Blendshape score range mapped to 0..1 eye closedness.
BLINK_RANGE = (0.35, 0.7)
# The neutral face only adapts while all of these are low.
EXPRESSIVE = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff",
]
RELAXED_BELOW = 0.25
# Below the chin the cat fades out (in face heights below the chin), so no neck.
NECK_FADE = (0.03, 0.25)


def load_sprite(chin_y: float, face_height: float):
    sprite = cv2.imread(str(HERE / "cat_cutout.png"), cv2.IMREAD_UNCHANGED)
    # Keep only the cat blob (drops faint background leftovers), soft edges.
    mask = (sprite[..., 3] > 128).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    alpha = cv2.GaussianBlur((labels == biggest).astype(np.float32), (5, 5), 0)
    start, end = (chin_y + f * face_height for f in NECK_FADE)
    fade = np.clip((end - np.arange(sprite.shape[0])) / (end - start), 0, 1)[:, None]
    sprite[..., 3] = (alpha * fade * 255).astype(np.uint8)
    return sprite


def background_color():
    photo = cv2.imread(str(HERE / "cat.jpg"))
    return photo[:20, :20].reshape(-1, 3).mean(axis=0)


def landmarks_px(landmarks, width, height):
    """(478, 3) pixel coords; z uses the x scale, as MediaPipe defines it."""
    return np.array([(p.x * width, p.y * height, p.z * width) for p in landmarks])


def cat_landmarks():
    """The cat has a human face, so the landmarker finds its features too.

    Returns (478, 3) landmarks in sprite px and the cat's head rotation in the photo.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        output_facial_transformation_matrixes=True,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(str(HERE / "cat.jpg"))
        result = landmarker.detect(image)
    if not result.face_landmarks:
        sys.exit("Could not find the face in cat.jpg.")
    points = landmarks_px(result.face_landmarks[0], image.width, image.height)
    return points, rotation_of(result.facial_transformation_matrixes[0])


def ramp(value, lo, hi):
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def blinks(scores):
    """Eye closedness (image-left, image-right), each 0..1."""
    # MediaPipe names eyes from the person's view: their right eye is on the
    # image's left.
    return np.array([
        ramp(scores["eyeBlinkRight"], *BLINK_RANGE),
        ramp(scores["eyeBlinkLeft"], *BLINK_RANGE),
    ])


def create_trackers():
    video = vision.RunningMode.VIDEO
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=video,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=video,
        num_hands=2,
    ))
    return face, hands


def open_webcam():
    # macOS asks for camera permission asynchronously; keep retrying while the
    # permission dialog is open.
    for _ in range(30):
        webcam = cv2.VideoCapture(0)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
            return webcam
        time.sleep(1)
    sys.exit("Cannot open webcam. Grant camera access to your terminal in "
             "System Settings > Privacy & Security > Camera.")


def open_virtual_camera():
    try:
        return pyvirtualcam.Camera(OUT_W, OUT_H, FPS, fmt=pyvirtualcam.PixelFormat.BGR)
    except RuntimeError as e:
        sys.exit(f"{e}\n\nOne-time setup: open OBS, click 'Start Virtual Camera', "
                 "approve the system extension, click 'Stop Virtual Camera', quit OBS. "
                 "Keep OBS closed while this runs.")


def main():
    cat_points, cat_rotation = cat_landmarks()
    sprite = load_sprite(cat_points[:, 1].max(), np.ptp(cat_points[:, 1]))
    cat_face = CatFace(sprite, cat_points[:, :2])
    cat_mesh = CatMesh(sprite, cat_points, (OUT_W, OUT_H), background_color())
    paws = Paws(*cat_mesh.face_on_screen(), OUT_W)
    transfer = ExpressionTransfer(cat_points)
    face_tracker, hand_tracker = create_trackers()
    webcam = open_webcam()
    vcam = open_virtual_camera()

    rotation_filter = OneEuroFilter(ROTATION_MIN_CUTOFF, ROTATION_BETA)
    expression_filter = OneEuroFilter(EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA)
    # Until a face is found: cat looks straight ahead with its own expression.
    # (The cat in the photo is turned slightly; its inverse rotation undoes that.)
    head = to_vector(cat_rotation.T)
    blink = np.zeros(2)
    offsets = np.zeros_like(cat_points)
    user_face = None
    print(f"Streaming to '{vcam.device}'. Select it as camera in Zoom.\n"
          "Keep a relaxed face for a second at start. Keys: q = quit, n = recalibrate neutral face.")
    with vcam:
        start = last = time.monotonic()
        while True:
            ok, frame = webcam.read()
            if not ok:
                continue
            now = time.monotonic()
            dt, last = max(now - last, 1e-3), now
            cam_h, cam_w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int((now - start) * 1000)
            face = face_tracker.detect_for_video(image, timestamp)
            hands = hand_tracker.detect_for_video(image, timestamp)

            if face.face_landmarks:  # no face -> keep the last pose and expression
                user_face = landmarks_px(face.face_landmarks[0], cam_w, cam_h)
                user_rotation = rotation_of(face.facial_transformation_matrixes[0])
                head = rotation_filter(to_vector(relative_rotation(user_rotation, cat_rotation)), dt)
                scores = {c.category_name: c.score for c in face.face_blendshapes[0]}
                blink = blink + BLINK_SMOOTHING * (blinks(scores) - blink)
                relaxed = max(scores[name] for name in EXPRESSIVE) < RELAXED_BELOW
                offsets = expression_filter(EXPRESSION_GAIN * transfer(user_face, relaxed), dt)
            paws.update(hands, user_face, (cam_w, cam_h), dt)

            out = cat_mesh.render(cat_face.render(blink=tuple(blink)), offsets, to_matrix(head))
            out = paws.draw(out)
            vcam.send(out)
            cv2.imshow("cat cam (q quit, n neutral)", cv2.flip(cv2.resize(out, (640, 360)), 1))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                transfer.reset()

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
