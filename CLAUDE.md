# Cat Cam: notes for contributors (and Claude Code)

A webcam filter that turns you into one of several cats and streams it to
Zoom through the OBS virtual camera. Python 3.12, managed with
[uv](https://docs.astral.sh/uv/); developed on Apple Silicon macOS. The
README covers setup and use; this file covers working on the code.

## Everyday commands

```sh
uv run cat-cam                 # run it (needs a webcam and the OBS virtual camera, OBS itself closed)
uv run cat-cam --check         # load and draw every cat once, no camera needed
uv run pytest                  # all tests; they render every real cat with OpenGL
uv run pytest -m "not gl"      # only the tests that need no OpenGL context
uv run ruff check .            # lint (line length 110)
uv run python tools/frame_bench.py   # frame times of every 3D head, no camera needed
```

Run the tests and ruff before every commit. A change to how something looks
(mesh, mouth, paws, tail, ...) also needs a look: render the affected cats to
an image and check them, as the tests only catch holes and gross errors. A
change to what runs every frame needs `tools/frame_bench.py` before and after
(`--video clip.mp4` tracks a recording, `--record clip.mp4` makes one): the
main loop has about 33 ms a frame, and spikes show as choppy video. Keep work
off it: tracking runs on its own thread (`Tracker`), the 3D heads are shaped
in their vertex shader, and blinks repaint only the texels near the lids.

## Layout

- `catcam/app.py`: main loop (webcam, tracking, render, virtual camera), the
  preview window, keys, the switches; `glass.py` draws the window's glass
  panels, switches and text, `picker.py` the cat picker.
- `catcam/voice.py`: your microphone as the animal's sounds (a bark or a
  meow per syllable, from `assets/cats/<Name>/voice/`, made for the cats by
  `tools/cat_voices.py`), out to the BlackHole virtual microphone. `idle.py`: what the
  animal does by itself while you are still. `background.py`: a picture or
  video behind the animal. `reactions.py`: emoji reactions on the number
  keys, drawn from the Mac's emoji font at runtime (never commit Apple's
  emoji pictures).
- `catcam/cat_mesh.py`: the 3D head mesh, its shaders and the drawn mouth;
  `mouth_fit.py`, `ears.py` and `lighting.py` feed into it.
- `catcam/cat_face.py`: eyelids. `face_motion.py`: expression transfer and
  each blendshape's resting score. `head_pose.py`: head rotation and the One
  Euro filter. `tools/tracking_eval.py` replays recorded webcam clips through
  the tracking and measures jitter, lag, blinks and dropouts: run it before and
  after changing any smoothing (keep the clips out of git, they are faces).
- `catcam/paws.py`, `paw_photos.py`: paws. `tail.py`: the tail.
- `catcam/head3d.py`: real 3D heads, shown while the "3D head" switch is
  on, for cats with an `assets/cats/<Name>/head3d.npz`. It borrows the 2D cat's
  expressions, blinks and ears; `mouth3d.py` builds its teeth, tongue and
  mouth cavity, with a jaw that swings open. `head3d_import.py` turns a generated
  `.glb` head (e.g. from TRELLIS.2 or Meshy) into those assets:
  `uv run --with trimesh python -m catcam.head3d_import <head.glb> <Name>`;
  with `--own-mouth` it keeps the head's own mouth instead, finding its lower
  jaw and cutting it free along the slit between the lips.
  The whole recipe, from installing TRELLIS.2 on to checking the result, is in
  `docs/3d-heads.md`; `tools/trellis/` has its install script and fixes and
  `tools/head3d_sheet.py` renders a cat's heads in the poses worth checking.
- `catcam/prepare.py`, `landmark_fit.py`: turn portraits into cat assets
  (only needed when adding or changing a cat).
- `assets/cats/<Name>/`: prepared cats, used at runtime. `all_cats/`: the
  source pictures. `models/`: MediaPipe models.
- `packaging/`: the standalone macOS app and disk image.

## Conventions

- Sizes are in face widths (or other relative units) rather than pixels, so
  any picture resolution behaves the same. Tunable constants sit at the top
  of their module with a comment saying what they do and in which unit.
- Comments and docstrings say why, in plain words. Match the style of the
  file you are in.
- Conventional commit messages: `feat: ...`, `fix: ...`, `docs: ...`,
  `chore: ...`, `test: ...`, with a body explaining the why. One logical
  change per commit.
- Keep `README.md` in step with user-visible changes (keys, setup, tuning
  table, the "How it works" list, the file list) and `packaging/Read me
  first.txt` with anything the app's users need to know.

## Adding or changing a cat

1. Put the portrait in `all_cats/<Name>.png`: face looking straight at the
   camera, mouth closed, plain background. Optionally add paw photos:
   `<Name>PawsFront.png` (front legs seen from the front) and
   `<Name>PawsBack.png` (the soles), left paw left and right paw right of a
   vertical purple line.
2. `uv run --group prepare python -m catcam.prepare <Name>` (add `--force`
   to redo a cat).
3. Check the cat: `uv run cat-cam --check`, then render it at rest, with the
   mouth open, eyes closed and head turned, and look at the mouth closely.
   `catcam/mouth_fit.py` corrects lips that were fitted onto the wrong line;
   if the mouth still looks off, that is where to look, or place the lips by
   hand with an `all_cats/<Name>Mouth.txt` (see `catcam/prepare.py`).
4. Add the name to `test_shipped_cats` in `tests/test_cats.py`, regenerate
   `docs/picker.jpg` if the picker changed, and commit the pictures in
   `all_cats/` together with `assets/cats/<Name>/`.
5. Optionally make its 3D head (`docs/3d-heads.md`) and check it with
   `uv run python tools/head3d_sheet.py <Name> -o sheet.jpg`. A cat that only
   exists as a 3D head (a Meshy model) gets its portrait from it first:
   `tools/head_portrait.py`, run in Blender without its window.

## Making a release

Releases are GitHub releases on tarrcimar/cat-cam with the disk image
attached, which is what the README's "Easy install" points people to. The
build only runs on an Apple Silicon Mac (the app is arm64 only).

1. Start from an up-to-date, clean `main`: `git pull`, `git status`.
2. Bump `version` in `pyproject.toml` (e.g. `0.2.0`) and run `uv lock`.
3. Test: `uv run pytest` and `uv run ruff check .`.
4. Build: `packaging/build_mac.sh`. It draws the icon, bundles the app with
   PyInstaller into `dist/Cat Cam.app`, signs it ad hoc, runs the app's
   `--check`, and writes `dist/Cat-Cam-<version>.dmg`.
5. Try the app once for real: open `dist/Cat Cam.app`, pick a cat, make sure
   the camera works and Zoom shows "OBS Virtual Camera".
6. Commit and tag:
   ```sh
   git commit -am "chore: release v<version>"
   git tag v<version>
   git push origin main --tags
   ```
7. Publish:
   ```sh
   gh release create v<version> dist/Cat-Cam-<version>.dmg \
       --title "Cat Cam <version>" --notes "What changed, in a few lines."
   ```

The app is not notarized (that needs a paid Apple developer account), so on
first launch macOS asks people to approve it under System Settings > Privacy
& Security > Open Anyway. `packaging/Read me first.txt`, which ships in the
disk image, walks them through it; keep it accurate.

`build/` and `dist/` are build output and are not committed.
