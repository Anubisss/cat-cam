# Cat Cam

Be a cat in your Zoom calls. Pick one of six cats (or Goldie, a golden retriever); it turns, nods and tilts
with your head, copies your whole face (brows, blinks, cheeks, smile, open
mouth with fangs) and your hands become its paws: the top of the paw when
you show the back of your hand, the soft toe beans when you show your palm. Its ears perk up when you raise
your brows, flatten when you frown, and twitch now and then, and the side of
its head that turns away from the light falls into shadow. Goldie has a dog's
mouth: black lips, canines, and a tongue that hangs out when you stick yours
out. She also takes over your voice: every syllable you say comes
out as a bark, as loud as you said it. When you sit still and quiet for a
few seconds, the animal comes alive on its own: it breathes, looks around
and tilts its head now and then, and Goldie pants with her tongue out. It
stops the moment you move your head, make a face or talk.

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
- [**BlackHole 2ch**](https://existential.audio/blackhole/), a virtual
  microphone, for Goldie's barking. Without it everything else still works.

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

3. **Allow keyboard access for your terminal app.** A few keyboard
   shortcuts also work while Zoom is in front, which macOS only allows with
   *Input Monitoring*. macOS usually does **not** ask for it by itself:
   1. Open *System Settings → Privacy & Security → Input Monitoring*.
   2. If your terminal app (Terminal, iTerm, Warp, ...) isn't in the list,
      click **+** at the bottom, choose it from *Applications*, and add it.
   3. Switch it on.
   4. **Quit and reopen the terminal app** (⌘Q). The permission only takes
      effect after a restart.

4. **Install the virtual microphone** (for barking):

   ```sh
   brew install blackhole-2ch
   ```

   It asks for your password. macOS only loads it after its audio service
   restarts: run `sudo killall coreaudiod` (or restart the Mac).

5. **Get the code**:

   ```sh
   git clone https://github.com/tarrcimar/cat-cam.git
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

To hear the barking yourself, live and without Zoom or the camera, put on
headphones and run `uv run python -m catcam.voice` (`--list` shows the
audio devices; `--mic` and `--out` pick them by name or number, e.g.
`--mic Brio --out AirPods`).

For the barking, choose **BlackHole 2ch** under *Settings → Audio →
Microphone* too, and set *Suppress background noise* to **Low**: Zoom
otherwise filters barks out as noise. Use headphones, so the others' voices
don't get barked back at them. With a cat, your own voice goes through
unchanged, so you can leave Zoom's microphone on BlackHole. macOS asks the
first time whether your terminal may use the microphone; allow it.

Keep a relaxed face for the first second. The cat compares your face against
how it looks at the start; if it looks grumpy while you are relaxed, press
`n`. When you look into the camera, the cat looks straight at the viewer. If
you'd rather have it look at the viewer while you look at your screen, press
`n` while looking at the screen; your pose at that moment counts as straight
ahead from then on.

Raise your hands into the camera's view for paws. Turn a hand around to
show the other side of the paw. Bending a finger curls the toe on that side
of the paw in, so you can wiggle them one by one; a fist curls them all, and
a finger you keep straight out of a fist stands up tall (yes, that one too).

### Keys (in the Cat Cam window)

| Key        | What it does                                              |
|------------|-----------------------------------------------------------|
| `1` to `7` | switch cat, at any time                                   |
| `c`        | open the cat picker (the call keeps showing your cat)     |
| `Esc`      | close the picker                                          |
| `n`        | "this is my neutral face": recalibrate face and head pose |
| `q`        | quit (closing the window or Ctrl+C also works)            |

The preview in the window is mirrored, like a mirror. The call gets the
unmirrored picture, just as a normal camera would send.

### Options

```sh
uv run cat-cam --camera 1   # use another webcam (e.g. if your iPhone camera opens)
uv run cat-cam --mic Brio   # use another microphone, by (part of its) name or number
                            # (list them: uv run python -m catcam.voice --list)
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
| "No virtual microphone found" | Install BlackHole (see above) and restart Cat Cam. |
| Zoom hears no barks | Choose **BlackHole 2ch** as Zoom's microphone, set background noise suppression to Low, and allow microphone access for your terminal. |
| Barks without you talking | Use headphones; a noisy room or loud speakers also trigger barks. |

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
   Besides the face, it also finds the animal's depth with a depth model
   (downloaded once, about 100 MB; it runs on the Apple GPU through Core ML),
   so a dog's long muzzle sticks out when it turns its head. A cat prepared
   before this has no `depth.png` and keeps its flatter, face-shaped depth;
   name it (`python -m catcam.prepare <Name>`) to give it one.
   Optionally add paw photos next to it: `all_cats/<Name>PawsFront.png`
   (the front legs seen from the front, as if the cat stood facing you) and
   `all_cats/<Name>PawsBack.png` (the soles). In each, put the left paw on
   the left and the right paw on the right, separated by a vertical purple
   line. `prepare` cuts them out too; a cat without paw photos gets drawn
   paws in its colours.
   To give it a voice, put short sounds (`.ogg`, `.wav`, `.flac` or `.mp3`)
   in `assets/cats/<Name>/voice/`; each syllable you say plays one of them.
   For a dog, put the word `dog` in `assets/cats/<Name>/kind.txt`: its
   portrait's lips are painted over with fur and a black lip line, and it
   gets a dog's teeth and tongue.
3. Run `uv run cat-cam`. The new cat appears in the picker.

## Tuning

These constants sit at the top of their files:

| Constant | File | Effect |
|---|---|---|
| `EXPRESSION_GAIN` | `catcam/app.py` | how exaggerated the cat's expressions are (1 = same as you) |
| `EXPRESSION_MIN_CUTOFF`, `EXPRESSION_BETA` | `catcam/app.py` | face smoothing: lower cutoff means less jitter, higher beta means less lag |
| `ROTATION_MIN_CUTOFF`, `ROTATION_BETA` | `catcam/app.py` | the same for head movement |
| `BLINK_RANGE` | `catcam/app.py` | how far you close your eyes before the cat blinks |
| `FLATTEN`, `PERK`, `TWITCH_EVERY` | `catcam/ears.py` | how far the ears move, and how often they twitch |
| `STRENGTH`, `LIGHT` | `catcam/lighting.py` | how much turning the head changes its shading, and where the light comes from |
| `BREATH_RISE`, `BREATH_SWELL` | `catcam/cat_mesh.py` | how much a full breath lifts the head and widens the chest |
| `STILL_AFTER` | `catcam/idle.py` | how long you must be still before the animal acts on its own |
| `HEAD_SPEED`, `FACE_SPEED` | `catcam/idle.py` | how much you may move and still count as still. Raise them if it never starts, lower them if it keeps going while you move |
| `BREATHING`, `SWAY`, `LOOK`, `LOOK_EVERY`, `PANT_EVERY`, `PANTING` | `catcam/idle.py` | breathing speed and depth, its constant head sway, how far and how often it looks around, how often and how hard a dog pants |
| `PAW_SIZE` | `catcam/paws.py` | paw size relative to your hand |
| `CLOSED_RANGE`, `CURL` | `catcam/paws.py` | how far you bend a finger to curl its toe, and how far (share of its height) a curled toe sinks into the paw |
| `RAISE`, `STAND` | `catcam/paws.py` | how much taller a finger raised from a fist gets, and how closed the rest of the hand must be |
| `TOP_MARGIN`, `CHIN_LINE` | `catcam/cat_mesh.py` | how much of the screen the cat fills |
| `HEAD_DEPTH` | `catcam/cat_mesh.py` | for animals with a depth map: how far the nose sticks out in front of the head's outline, in face widths |
| `HANG`, `HALF_WIDTH` | `catcam/tongue.py` | a dog's tongue: how far it hangs out, how wide it is |
| `TONGUE_RED`, `TONGUE_LIGHT`, `TONGUE_FILL` | `catcam/tongue.py` | spotting your tongue: how red and light (compared with your lips) it must be, and how much of your open mouth it must fill. Lower them if it doesn't react to your tongue, raise them if it comes out on its own |
| `ABOVE_NOISE`, `RISE` | `catcam/voice.py` | how loud you must talk to bark, and how clearly syllables must stand apart |
| `MIN_GAP` | `catcam/voice.py` | shortest time between two barks |
| `GAIN`, `SPEED` | `catcam/voice.py` | bark volume and pitch, from your quietest to your loudest syllables |

## How it works

1. MediaPipe tracks 478 points on your face plus your hands, from the webcam.
2. Each cat has the same 478 points, found once by `catcam.prepare`. The
   cats have human-like faces, so the human face model fits them.
3. Your points are aligned to your neutral face, which removes head position,
   rotation and distance. What remains is your expression. That is mapped onto
   the cat's points (`catcam/face_motion.py`).
4. The cat photo is a textured 3D mesh. It is deformed by your expression,
   rotated by your head pose and drawn with OpenGL, shaded by how much each
   part turned toward or away from the light (`catcam/lighting.py`). The
   mouth's inside and cat teeth are drawn in the gap between the lips,
   coloured to match the cat's lips (`catcam/cat_mesh.py`). An animal with a
   depth map (`catcam/depth.py`, made by `prepare`) takes the mesh's depth
   from it instead of from the human face model, and is shaded per pixel on
   the GPU from the map's surface normals: Goldie's muzzle, brow, eye sockets
   and ears catch the light as she turns. A dog's lips are repainted when it
   is loaded (`catcam/dog_lips.py`), its mouth gets a dog's teeth and dark
   gums, and its tongue is a strip of the mesh lying on the lower teeth. The
   face tracker can't see tongues, so yours is spotted in the webcam image:
   the gap between your lips filled with something about as red as your lips
   (a mouth's inside is dark and teeth aren't red). The more it fills, the
   further the dog's tongue slides out and hangs from its lower lip
   (`catcam/tongue.py`).
5. Blinks lower the cat's eyelids in its texture (`catcam/cat_face.py`), the
   ears bend about a hinge below their tips (`catcam/ears.py`), and paws are
   drawn where your hands are (`catcam/paws.py`), showing their top or their
   sole depending on which way your hand faces, each toe curling as you bend
   its finger.
6. While your head, expression and voice stay still for a few seconds
   (blinks don't count), an idle animation fades in on top of your pose: a
   breath that swells the chest and lifts the head on it (drawn in, let out,
   a pause), a slow head sway, frequent random glances and head tilts, and for a dog, bouts of
   panting that open its jaw and hang its tongue out (`catcam/idle.py`).
   Any movement or talking fades it out in a third of a second.
7. The frame goes to the OBS virtual camera through pyvirtualcam.
8. Your microphone is followed in 10 ms blocks (`catcam/voice.py`). A
   syllable starts where your loudness rises clearly above the room's noise
   and above the dip after the previous syllable; each start plays one of the
   animal's sounds, louder and a little higher for louder syllables. The mix
   goes to BlackHole, which the call uses as its microphone.

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
  dog_lips.py      a dog's lips, painted over the portrait's
  tongue.py        a dog's tongue
  ears.py          ear movement
  idle.py          what the animal does on its own while you are still
  lighting.py      shading as the head turns
  face_motion.py   expression transfer
  head_pose.py     head rotation and the One Euro smoothing filter
  paws.py          paws
  voice.py         your syllables as the animal's sounds, out to the virtual microphone
  paw_photos.py    turns paw photos into paw sprites (used by prepare)
  prepare.py       turns all_cats/*.png into assets/cats/*
  landmark_fit.py  finds a cat's face points (used by prepare)
  depth.py         finds a cat's depth map (used by prepare)
all_cats/          source portraits
assets/cats/       prepared cats (cutout + face points), used at runtime
models/            MediaPipe face and hand models
```

The MediaPipe models in `models/` are by Google and licensed under Apache 2.0.
The depth model `prepare` downloads is
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small),
licensed under Apache 2.0.

Goldie's bark in `assets/cats/Goldie/voice/` is public domain (CC0), as
cleaned up by [Space Station 14](https://github.com/space-wizards/space-station-14/blob/master/Resources/Audio/Voice/Vulpkanin/attributions.yml)
from a Freesound recording by
[Geoff-Bremner-Audio](https://freesound.org/people/Geoff-Bremner-Audio/sounds/688201/).
