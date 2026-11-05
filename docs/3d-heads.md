# Making a 3D head for a cat

Each cat can have a real 3D head next to its photo head (the **3D head**
switch in the app, or `d`). It turns properly, and when the mouth opens there
are teeth, a tongue and a jaw inside. The head is generated from the cat's
portrait by [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), an
image-to-3D model, running locally on an Apple Silicon Mac. This page is the
recipe the first heads were made with, including what we tried that didn't work.

Everything after generation is automatic. The eyes, the mouth, expressions,
blinks, ears and shading all come from the import and the app; there is
nothing to draw or rig by hand.

TRELLIS.2 is free and local. [Meshy](https://www.meshy.ai) (paid, online)
makes more detailed heads, and the same import takes them; see
[Heads from Meshy](#heads-from-meshy).

## What you need

- An Apple Silicon Mac. The heads were made on an M5 MacBook Pro with 32 GB;
  generating one needs about 6 GB of memory.
- About 35 GB of free disk: about 30 GB of model weights (downloaded on the
  first run) plus a 1 GB Python environment.
- **Xcode** from the App Store (the full app, not only the command line
  tools), opened once. Then, in Terminal:

  ```sh
  sudo xcode-select -s /Applications/Xcode.app
  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain
  ```

  Without the Metal toolchain the texture is baked by a slower, worse
  fallback.
- A free [Hugging Face](https://huggingface.co) account:
  - Request access to [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0),
    the background remover TRELLIS uses. Approval is usually instant.
  - Then log in: `uvx --from huggingface_hub hf auth login`.
  - You don't need access to Meta's DINOv3; see step 3 of the install.

## Install TRELLIS.2 (once)

We use [trellis-mac](https://github.com/shivampkumar/trellis-mac), a port
of TRELLIS.2 to Apple's Metal. Out of the box it doesn't build with today's
Xcode and torch, and its textures come out with flaws. `tools/trellis/`
holds the exact versions and fixes the heads were made with.

1. Get trellis-mac and its parts at the tested versions, with the fixes
   applied. Run this from the Cat Cam folder; it is safe to run again:

   ```sh
   tools/trellis/install.sh ~/trellis-mac
   ```

   It does four things:
   - Builds the Metal parts as C++20.
   - Fetches the Eigen library that one of them needs.
   - Adds a `TRELLIS_REMESH` switch to `generate.py`.
   - Fixes the texture baker: black specks, dark seams along texture edges,
     and a texture that came out upside down.
2. Run trellis-mac's own setup, which installs and builds everything:

   ```sh
   cd ~/trellis-mac
   bash setup.sh
   ```

3. Make the DINOv3 image encoder TRELLIS needs. Meta's copy is gated behind
   an access request that can take days. This converts timm's public copy of
   the same weights instead, and checks the two give the same results:

   ```sh
   .venv/bin/python convert_dinov3.py dinov3-vitl16
   ```

## Make a head

1. **Generate** it from the cat's portrait, the same `all_cats/<Name>.png`
   the photo cat uses. TRELLIS removes the background itself.

   ```sh
   cd ~/trellis-mac
   TRELLIS_REMESH=1 .venv/bin/python run_local_dino.py \
       ~/path/to/cat-cam/all_cats/<Name>.png --output out/<name> --texture-size 2048
   ```

   The first run downloads the model weights. After that a head takes about
   6 minutes. The result is `out/<name>.glb`, which you can look at in
   Preview or any 3D viewer.
2. **Import** it into Cat Cam, from the Cat Cam folder:

   ```sh
   uv run --with trimesh python -m catcam.head3d_import ~/trellis-mac/out/<name>.glb <Name>
   ```

   This writes `assets/cats/<Name>/head3d.npz` and `head3d.png`. It prints
   two numbers worth reading:

   ```
   Woofgang: portrait outline overlap 0.961
     features were [ 7. 42.] px off; moved the camera
   ```

   - **Outline overlap:** how well the head's outline, seen face-on, matches
     the portrait. 0.95 or more is good. Well under 0.9 means the head came
     out a different shape from the photo; generate it again with another
     `--seed`.
   - **Features off:** how far the eyes and nose had to be nudged to land on
     the portrait's. Up to a few dozen pixels is normal; the import corrects it.
3. **Look at it** without a camera. This renders the photo head and the 3D
   head side by side: at rest, mouth half and wide open, eyes closed, ears
   flat, and turned with the mouth open:

   ```sh
   uv run python tools/head3d_sheet.py <Name> -o sheet.jpg
   ```

   Then try it for real: `uv run cat-cam`, pick the cat, and press `d`.
4. **Commit** `assets/cats/<Name>/head3d.npz` and `head3d.png`. They are
   about 15 MB per cat.

## Heads from Meshy

Woofgang's head was made with Meshy's Image to 3D instead of TRELLIS:

1. **Settings:**
   - Multi-view on, with the front portrait as the main view and the
     profiles and back as Left, Back and Right. "Left" faces the image's
     left.
   - Meshy 7.1 Flagship, High Detail, Ultra 2K.
   - Texture on, with the Meshy 7 texture model at 2K and no PBR maps.
   - Pose off, Image Enhancement off, licence Private.
2. **Download** it as GLB. Meshy's heads have millions of triangles, far
   too many to move every frame. Reduce a copy to 300,000 (your download
   stays as it is):

   ```sh
   uv run --no-project --with pymeshlab --with trimesh --with pillow python - <head.glb> <small.glb> <<'PY'
   import sys, tempfile, pymeshlab, trimesh
   head = next(iter(trimesh.load(sys.argv[1]).geometry.values()))
   ms = pymeshlab.MeshSet()
   ms.add_mesh(pymeshlab.Mesh(vertex_matrix=head.vertices, face_matrix=head.faces,
                              v_tex_coords_matrix=head.visual.uv))
   ms.compute_texcoord_transfer_vertex_to_wedge()
   ms.meshing_decimation_quadric_edge_collapse_with_texture(
       targetfacenum=300000, preserveboundary=True, preservenormal=True)
   with tempfile.TemporaryDirectory() as tmp:  # an .obj keeps the texture coordinates per corner
       ms.save_current_mesh(f"{tmp}/small.obj")
       small = trimesh.load(f"{tmp}/small.obj", process=False, force="mesh")
   small.visual = trimesh.visual.TextureVisuals(
       uv=small.visual.uv, material=trimesh.visual.material.PBRMaterial(
           baseColorTexture=head.visual.material.baseColorTexture))
   small.export(sys.argv[2])
   PY
   ```
3. **Import** it as above. Add `--own-mouth` to keep the head's own mouth
   instead of the added one (see below):

   ```sh
   uv run --with trimesh python -m catcam.head3d_import <small.glb> <Name> --own-mouth
   ```

## How the eyes and mouth work

**Eyes.** TRELLIS's eyes come out blurry and empty-looking. The import finds
the head's eyes and paints the portrait's real eyes over them in the texture,
lined up and with soft edges. Blinks reuse the photo cat's eyelids: when they
close, the app repaints those same texture spots from the eyelid drawing.
The knobs are at the top of `catcam/head3d_import.py`:
- `EYE_GROW`: how far past the eye the paint reaches.
- `EYE_FEATHER`: how soft its edge is.
- `EYE_SEARCH`: how far to look for the model's own eye.

**Mouth.** The generated head's mouth is closed and sealed, so nothing about
it is generated. When the app loads a 3D head it does three things:
- **Lip cut:** it splits the lips along the cat's mouth line, so they can
  part.
- **Inside of the mouth:** `catcam/mouth3d.py` builds a mouth in real
  geometry: incisors, fangs and premolars; gums; a grooved tongue; and a
  dark mouth cavity that fades toward the throat. The gums and tongue take
  on the cat's own lip colour.
- **Jaw:** the lower teeth and tongue sit on a jaw that swings open about a
  hinge behind the mouth and stays with the lower lip. The mouth is only
  drawn through the lip opening, so it can't show through the cheeks or chin.

The mouth is the same for every cat, sized to its lips. To change it, edit
the constants at the top of `mouth3d.py`, in mouth widths: tooth sizes and
positions, the tongue, the hinge, and the colours.

**Its own mouth instead** (`--own-mouth`): no teeth, tongue or cavity are
added. The lips still split along the mouth line and the jaw still drops
with yours, and the head's own inside shows between them as a dark hollow.
Woofgang's is like that.

**The mouth line** is where the lips meet in the portrait, found when the cat
is prepared. The fit is made for cats; on other faces it can land elsewhere.
A golden retriever's lands on its nose, for example, because its mouth is a
smile whose corners are its highest points. Then write the line yourself in
`all_cats/<Name>Mouth.txt`: one `x, y` per line, in portrait pixels, from
one mouth corner to the other. Then prepare the cat again with
`--force`, and import its head again. `all_cats/WoofgangMouth.txt` is an
example.

**Everything else** comes from the photo cat:
- **Expressions:** the head is tied to the photo cat's face points, so every
  expression moves both heads alike.
- **Ears:** they bend the same way on both heads.
- **Shading:** turning the head shades it as it does the photo cat.

## What didn't work

So you don't have to try it again:

- **Several photos of the cat** (profiles, three-quarter views, the back).
  TRELLIS.2 makes a head from a single image. We projected the other views
  onto the head as a texture, and the result showed ghosting and seams.
  TRELLIS's own texture, with the real eyes painted in, looked better. The
  front portrait is all you need.
- **No remesh.** Without `TRELLIS_REMESH=1` the head has holes, flaky
  patches of skin, and places you can see through. Remeshing rebuilds it as
  one closed surface.
- **Closing the holes with pymeshfix.** It wrecked the head.
- **A photo of the open mouth pasted inside.** It was too low-res and
  didn't match the 3D lips. The built mouth looks much better.
- **Hunyuan3D 2.** We passed on it without trying: its licence excludes the
  EU, the UK and South Korea. TRELLIS.2 is MIT licensed.

## Licences

- **TRELLIS.2:** MIT.
- **DINOv3 weights:** Meta's
  [DINOv3 licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md),
  whichever copy you download.
- **RMBG-2.0:** CC BY-NC 4.0, non-commercial. That's fine for a cat filter
  among friends; selling heads made with it is not.

## Troubleshooting

- **`setup.sh` fails compiling C++.** Run `tools/trellis/install.sh` on that
  folder first. It is safe to run on a folder where `setup.sh` already ran.
- **`xcrun: error` or something about a licence.** Accept Xcode's licence:
  `sudo xcodebuild -license accept`.
- **The texture has black specks, dark seams or is upside down.** The
  trellis-mac fixes weren't applied. Run `install.sh` again and look for
  errors.
- **Access errors for RMBG-2.0.** Request access on its Hugging Face page
  and log in again.
- **The eyes in the 3D head sit off the model's eyes.** The import prints how
  far it moved them. If that's tens of pixels, the portrait and the head
  disagree; try another `--seed`.
