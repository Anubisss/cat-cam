# Cat Cam

Be a cat in your Zoom calls. Pick one of six cats; it turns, nods and tilts
with your head, copies your whole face (brows, blinks, cheeks, smile, open
mouth with teeth) and your hands become paws.

Your own face is never sent anywhere. Only the drawn cat goes out to the
virtual camera, and everything runs locally on your computer.

![The cat picker](docs/picker.jpg)

![Rest, smile, and a smile with the head turned](docs/expressions.jpg)

## What you need

- **macOS**. It is developed and tested on Apple Silicon. Windows should work
  too (pyvirtualcam and OpenGL both support it) but is untested.
- A webcam.
- [**uv**](https://docs.astral.sh/uv/), which installs Python and all
  libraries for you.
- [**OBS Studio**](https://obsproject.com). Only its virtual camera driver is
  used; OBS itself stays closed.

## One-time setup

1. **Install uv**:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install OBS and enable its virtual camera.** Run
   `brew install --cask obs` or download it from obsproject.com. Then:
   1. Open OBS and click **Start Virtual Camera**.
   2. When macOS asks, allow the system extension. You may need to approve it
      under *System Settings → General → Login Items & Extensions*.
   3. Click **Stop Virtual Camera** and **quit OBS**.

   You only do this once. Afterwards, keep OBS closed while Cat Cam runs, or
   it takes the virtual camera for itself.

3. **Get the code**:

   ```sh
   git clone <this repository> cat-cam
   cd cat-cam
   ```

## Run it

```sh
uv run cat-cam
```

The first run downloads the libraries, which takes a minute. macOS then asks
whether your terminal may use the camera; allow it. If you said no, enable it
under *System Settings → Privacy & Security → Camera* and run again.

A window opens with the cat picker. Click a cat, or press its number.

In Zoom, go to *Settings → Video → Camera* and choose **OBS Virtual Camera**.
Do the same in any other app (Meet, Teams, FaceTime, ...). If the camera is
not listed, restart Zoom while Cat Cam is running.

Keep a relaxed face for the first second. The cat compares your face against
how it looks at the start, and the way you sit at the start counts as "looking
at the viewer". If the cat looks grumpy or turned while you are relaxed, press
`n`.

Raise your hands into the camera's view for paws.

### Keys (in the Cat Cam window)

| Key        | What it does                                              |
|------------|-----------------------------------------------------------|
| `1` to `6` | switch cat, at any time                                   |
| `c`        | open the cat picker (the call keeps showing your cat)     |
| `Esc`      | close the picker                                          |
| `n`        | "this is my neutral face": recalibrate face and head pose |
| `q`        | quit (closing the window or Ctrl+C also works)            |

The preview in the window is mirrored, like a mirror. The call gets the
unmirrored picture, just as a normal camera would send.

### Options

```sh
uv run cat-cam --camera 1   # use another webcam (e.g. if your iPhone camera opens)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open webcam` | Allow camera access for your terminal (see above). If you have several cameras, try `--camera 1`. |
| Error about the virtual camera | Do the OBS setup above, and make sure OBS is **not** running. |
| Zoom shows a black picture | OBS is probably running; quit it and restart Cat Cam. |
| Zoom does not list "OBS Virtual Camera" | Restart Zoom while Cat Cam is running. |
| The cat looks off while you are relaxed | Press `n` with a neutral face. |
| "Looking for your face..." | Face the camera and make sure your face is lit. |

## Add your own cat

1. Put a portrait in `all_cats/<Name>.png`. It works best when the face looks
   straight at the camera with the mouth closed, against a plain background.
2. Prepare it. This cuts the cat out and finds its face, and takes about
   10 seconds per cat:

   ```sh
   uv run --group prepare python -m catcam.prepare
   ```

   The results go to `assets/cats/<Name>/`. Already prepared cats are skipped;
   add `--force` to redo them, or name cats to do only those.
3. Run `uv run cat-cam`. The new cat appears in the picker.

## Tuning

These constants sit at the top of their files:

| Constant | File | Effect |
|---|---|---|
| `EXPRESSION_GAIN` | `catcam/app.py` | how exaggerated the cat's expressions are (1 = same as you) |
| `EXPRESSION_MIN_CUTOFF`, `EXPRESSION_BETA` | `catcam/app.py` | face smoothing: lower cutoff means less jitter, higher beta means less lag |
| `ROTATION_MIN_CUTOFF`, `ROTATION_BETA` | `catcam/app.py` | the same for head movement |
| `BLINK_RANGE` | `catcam/app.py` | how far you close your eyes before the cat blinks |
| `PAW_SIZE` | `catcam/paws.py` | paw size relative to your hand |
| `TOP_MARGIN`, `CHIN_LINE` | `catcam/cat_mesh.py` | how much of the screen the cat fills |

## How it works

1. MediaPipe tracks 478 points on your face plus your hands, from the webcam.
2. Each cat has the same 478 points, found once by `catcam.prepare`. The
   cats have human-like faces, so the human face model fits them.
3. Your points are aligned to your neutral face, which removes head position,
   rotation and distance. What remains is your expression. That is mapped onto
   the cat's points (`catcam/face_motion.py`).
4. The cat photo is a textured 3D mesh. It is deformed by your expression,
   rotated by your head pose and drawn with OpenGL. The mouth's inside and
   teeth are drawn in the gap between the lips (`catcam/cat_mesh.py`).
5. Blinks lower the cat's eyelids in its texture (`catcam/cat_face.py`), and
   paws are drawn where your hands are (`catcam/paws.py`).
6. The frame goes to the OBS virtual camera through pyvirtualcam.

## Development

```sh
uv run pytest          # tests, including render checks on every cat
uv run ruff check .    # lint
```

```
catcam/
  app.py           main loop: webcam -> tracking -> render -> virtual camera
  picker.py        the cat picker
  cats.py          finding and loading prepared cats
  cat_mesh.py      the 3D head mesh and its OpenGL rendering
  cat_face.py      eyelids
  face_motion.py   expression transfer
  head_pose.py     head rotation and the One Euro smoothing filter
  paws.py          paws
  prepare.py       turns all_cats/*.png into assets/cats/*
  landmark_fit.py  finds a cat's face points (used by prepare)
all_cats/          source portraits
assets/cats/       prepared cats (cutout + face points), used at runtime
models/            MediaPipe face and hand models
```

The MediaPipe models in `models/` are by Google and licensed under Apache 2.0.
