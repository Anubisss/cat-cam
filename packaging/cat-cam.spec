# PyInstaller spec for the standalone macOS app. Build with packaging/build_mac.sh.
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

datas = [
    (str(ROOT / "models"), "models"),
    (str(ROOT / "assets" / "cats"), "assets/cats"),
    (str(ROOT / "catcam" / "data"), "catcam/data"),
    *collect_data_files("mediapipe"),
    *collect_data_files("_sounddevice_data"),  # PortAudio
    *collect_data_files("_soundfile_data"),  # libsndfile
]
binaries = [*collect_dynamic_libs("mediapipe"), *collect_dynamic_libs("pyvirtualcam")]

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    datas=datas,
    binaries=binaries,
    hiddenimports=["glcontext", "Quartz"],
    excludes=["tkinter", "rembg", "onnxruntime", "pytest", "IPython"],  # mediapipe needs matplotlib
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Cat Cam", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Cat Cam")
app = BUNDLE(
    coll,
    name="Cat Cam.app",
    icon=str(ROOT / "build" / "AppIcon.icns"),
    bundle_identifier="com.tarrcimar.catcam",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "Cat Cam",
        "CFBundleShortVersionString": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSCameraUsageDescription":
            "Cat Cam turns your webcam picture into a cat. Your camera picture never leaves your computer.",
    },
)
