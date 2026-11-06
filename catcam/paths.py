"""Where the filter finds its models, portraits and prepared cat assets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORTRAITS = ROOT / "all_cats"  # source portraits, one <Name>.png per cat
ASSETS = ROOT / "assets" / "cats"  # prepared by `python -m catcam.prepare`
MOUTHS = ROOT / "assets" / "mouth"  # 3D teeth, gums and tongues, made by tools/mouth_model.py
PAWS = ROOT / "assets" / "paws"  # 3D paws, made by tools/paw_model.py
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
SEGMENT_MODEL = ROOT / "models" / "selfie_multiclass_256x256.tflite"  # for showing your real hands
FACE_MIRROR = Path(__file__).resolve().parent / "data" / "face_mirror.npy"
FX_FILE = Path(__file__).resolve().parent / "data" / "fx.bin"  # optional, see catcam.fx
LOG_FILE = Path.home() / "Library" / "Logs" / "Cat Cam.log"  # errors of the standalone app
