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
```

Run the tests and ruff before every commit. A change to how something looks
(mesh, mouth, paws, tail, ...) also needs a look: render the affected cats to
an image and check them, as the tests only catch holes and gross errors.

## Layout

- `catcam/app.py`: main loop (webcam, tracking, render, virtual camera), the
  preview window, keys, the Tail checkbox.
- `catcam/cat_mesh.py`: the 3D head mesh, its shaders and the drawn mouth;
  `mouth_fit.py`, `ears.py` and `lighting.py` feed into it.
- `catcam/cat_face.py`: eyelids. `face_motion.py`: expression transfer.
  `head_pose.py`: head rotation and the One Euro filter.
- `catcam/paws.py`, `paw_photos.py`: paws. `tail.py`: the tail.
- `catcam/head3d.py`: experimental real 3D heads, shown as an extra "<Name> 3D"
  cat when `assets/cats/<Name>/head3d.npz` exists. It borrows the 2D cat's
  expressions, blinks and ears; `mouth3d.py` builds its teeth, tongue and
  mouth cavity, with a jaw that swings open. `head3d_import.py` turns a generated
  `.glb` head (e.g. from TRELLIS.2) into those assets:
  `uv run --with trimesh python -m catcam.head3d_import <head.glb> <Name>`.
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
   if the mouth still looks off, that is where to look.
4. Add the name to `test_shipped_cats` in `tests/test_cats.py`, regenerate
   `docs/picker.jpg` if the picker changed, and commit the pictures in
   `all_cats/` together with `assets/cats/<Name>/`.

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
