"""Renders a cat's portrait from its 3D head, for cats that only exist as a model
(a Meshy head made from pictures that aren't a usable portrait). Run it headless:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python tools/head_portrait.py -- <head.glb> all_cats/<Name>.png

The head is seen face-on from a little above, unlit (its texture already has the
light baked in), on the same plain backdrop as the other portraits.
"""

import math
import sys

import bpy
import mathutils

SIZE = 1254  # px, square
TILT = 6  # degrees the camera looks down at the face
MARGIN = 1.12  # the frame's width over the head's
BACKDROP = (188, 170, 156)  # RGB, sRGB 0..255


def linear(c: float) -> float:
    """An sRGB channel (0..1) as Blender's linear colour."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def main(src: str, dst: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]
    corners = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    lo = mathutils.Vector([min(p[i] for p in corners) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in corners) for i in range(3)])

    camera = bpy.data.cameras.new("Portrait")
    camera.type = "ORTHO"
    camera.ortho_scale = max(hi.x - lo.x, hi.z - lo.z) * MARGIN
    camera.clip_end = 20
    cam = bpy.data.objects.new("Portrait", camera)
    scene.collection.objects.link(cam)
    # glTF's front (+z) is Blender's -y.
    tilt = math.radians(TILT)
    cam.location = (lo + hi) / 2 + mathutils.Vector((0, -5 * math.cos(tilt), 5 * math.sin(tilt)))
    cam.rotation_euler = (math.radians(90) - tilt, 0, 0)
    scene.camera = cam

    scene.world = bpy.data.worlds.new("Backdrop")
    scene.world.color = [linear(c / 255) for c in BACKDROP]
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.background_type = "WORLD"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = dst
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main(*sys.argv[sys.argv.index("--") + 1:])
