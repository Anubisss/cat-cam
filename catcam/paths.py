"""Where the filter finds its models, portraits and prepared cat assets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORTRAITS = ROOT / "all_cats"  # source portraits, one <Name>.png per cat
ASSETS = ROOT / "assets" / "cats"  # prepared by `python -m catcam.prepare`
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
FACE_MIRROR = Path(__file__).resolve().parent / "data" / "face_mirror.npy"
