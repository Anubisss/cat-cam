"""Runs trellis-mac's generate.py with DINOv3 loaded from ./dinov3-vitl16
(made by convert_dinov3.py from timm's ungated copy) instead of the gated
facebook/ repo. Takes generate.py's arguments:

    TRELLIS_REMESH=1 .venv/bin/python run_local_dino.py <portrait.png> --output out/<name> --texture-size 2048
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "dinov3-vitl16")
sys.path.insert(0, HERE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch, as generate.py does

import transformers  # noqa: E402

_load = transformers.DINOv3ViTModel.from_pretrained.__func__


def _from_pretrained(cls, name, *args, **kwargs):
    if name == "facebook/dinov3-vitl16-pretrain-lvd1689m" and os.path.isdir(LOCAL):
        name = LOCAL
    return _load(cls, name, *args, **kwargs)


transformers.DINOv3ViTModel.from_pretrained = classmethod(_from_pretrained)
sys.argv[0] = os.path.join(HERE, "generate.py")
runpy.run_path(sys.argv[0], run_name="__main__")
