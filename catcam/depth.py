"""Depth of a cat portrait, from a monocular depth model (used by prepare).

The face landmarks' depth is a human face's: flat, with a small nose. A depth
model (Depth Anything V2) sees the animal's own shape: a dog's long muzzle,
the eye sockets, ears hanging beside the head. It runs once per cat, through
Core ML on Apple Silicon, and is stored as assets/cats/<Name>/depth.png:
16 bit, nearer = brighter, stretched over the cat.
"""

import functools
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
INPUT_SIZE = 518  # longest side fed to the model
PATCH = 14  # the model's input sides must be multiples of this
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def depth_file(name: str, assets: Path) -> Path:
    return assets / name / "depth.png"


@functools.cache
def _session():
    import onnxruntime as ort
    import pooch

    model = pooch.retrieve(MODEL_URL, f"sha256:{MODEL_SHA256}", path=pooch.os_cache("catcam"))
    available = ort.get_available_providers()
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
    return ort.InferenceSession(model, providers=providers)


def estimate(bgr: np.ndarray) -> np.ndarray:
    """(h, w) relative inverse depth of the image (nearer = larger)."""
    h, w = bgr.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    size = tuple(max(round(side * scale / PATCH), 1) * PATCH for side in (w, h))
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_CUBIC)
    pixels = ((rgb / 255 - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)[None]
    session = _session()
    disparity = session.run(None, {session.get_inputs()[0].name: pixels})[0][0]
    return cv2.resize(disparity, (w, h), interpolation=cv2.INTER_CUBIC)


def to_png(disparity: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """uint16 depth image: 0 at the farthest point of the cat, 65535 at the nearest, 0 around it."""
    cat = alpha > 128
    lo, hi = disparity[cat].min(), disparity[cat].max()
    stretched = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
    stretched[~cat] = 0
    return np.round(stretched * 65535).astype(np.uint16)


def prepare_depth(name: str, assets: Path):
    """Writes depth.png for a prepared cat, from its cutout on its backdrop."""
    folder = assets / name
    cutout = cv2.imread(str(folder / "cutout.png"), cv2.IMREAD_UNCHANGED)
    with np.load(folder / "face.npz") as face:
        background = face["background"].astype(np.float32)
    alpha = cutout[..., 3:] / 255
    portrait = (cutout[..., :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    cv2.imwrite(str(depth_file(name, assets)), to_png(estimate(portrait), cutout[..., 3]))
