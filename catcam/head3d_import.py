"""Turns a generated 3D head (.glb, e.g. from TRELLIS.2) into a cat's head3d
assets: the mesh, its texture with the cat's real eyes painted in from the
portrait, and where every vertex and face landmark sits in the portrait, which
is what lets the head take on the 2D cat's expressions.

    uv run --with trimesh python -m catcam.head3d_import <head.glb> <Name>
"""

import sys

import cv2
import moderngl
import numpy as np

from catcam import cats
from catcam.cat_face import EYES
from catcam.head3d import files

YAW_SEARCH = np.arange(-15, 16, 3)  # degrees around face-on tried when fitting the camera
PITCH_SEARCH = np.arange(-12, 13, 3)
DEPTH_TOL = 0.01  # model units: a point this close to the front surface counts as visible
EYE_GROW = 0.35  # painted eye area reaches this share of the eye's width past its contour
EYE_FEATHER = 0.15  # soft edge, eye widths
EYE_SEARCH = 0.4  # how far (eye widths) to look for the model's eye around the photo's

DEPTH_VS = """
#version 330
uniform mat4 mvp;
in vec3 pos; out float z;
void main() { vec4 q = mvp * vec4(pos, 1.0); gl_Position = q; z = -q.z * 3.0; }
"""
DEPTH_FS = """
#version 330
in float z; out vec4 color;
void main() { color = vec4(z, 1.0, 0.0, 1.0); }
"""
UV_VS = """
#version 330
in vec2 uv; in vec3 pos;
out vec3 p;
void main() { gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0); p = pos; }
"""
UV_FS = """
#version 330
in vec3 p; out vec4 out_p;
void main() { out_p = vec4(p, 1.0); }
"""


def rotation(yaw: float, pitch: float) -> np.ndarray:
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


class FrontCamera:
    """Orthographic camera looking at the model like the portrait does:
    photo px = centre + scale * (rot @ p).xy, y flipped."""

    def __init__(self, rot, scale, centre):
        self.rot, self.scale, self.centre = rot, scale, np.asarray(centre, float)

    def project(self, points: np.ndarray):
        q = points @ self.rot.T
        return np.column_stack([self.centre[0] + self.scale * q[:, 0],
                                self.centre[1] - self.scale * q[:, 1]]), q[:, 2]

    def unproject(self, px: np.ndarray, z: np.ndarray) -> np.ndarray:
        q = np.column_stack([(px[:, 0] - self.centre[0]) / self.scale,
                             -(px[:, 1] - self.centre[1]) / self.scale, z])
        return q @ self.rot


class Raster:
    def __init__(self, vertices, faces, uv):
        self.ctx = moderngl.create_standalone_context()
        self.vertices, self.faces, self.uv = vertices, faces, uv
        self.depth_prog = self.ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        self.depth_vao = self.ctx.vertex_array(
            self.depth_prog, [(self.ctx.buffer(vertices.astype("f4")), "3f", "pos")],
            self.ctx.buffer(faces.astype("i4")))

    def depth(self, cam: FrontCamera, size) -> np.ndarray:
        """Camera-space depth (bigger = nearer) per photo pixel; -inf off the model."""
        w, h = size
        mvp = np.eye(4)
        mvp[:3, :3] = cam.rot
        ortho = np.diag([2 * cam.scale / w, 2 * cam.scale / h, -1 / 3, 1.0])
        ortho[0, 3] = 2 * cam.centre[0] / w - 1
        ortho[1, 3] = 1 - 2 * cam.centre[1] / h
        self.depth_prog["mvp"].write((ortho @ mvp).astype("f4").T.tobytes())
        fbo = self.ctx.framebuffer([self.ctx.renderbuffer((w, h), 4, dtype="f4")],
                                   self.ctx.depth_renderbuffer((w, h)))
        fbo.use()
        fbo.clear(-1e9, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.depth_vao.render()
        out = np.frombuffer(fbo.read(components=4, dtype="f4"), "f4").reshape(h, w, 4)[::-1, :, 0].copy()
        fbo.release()
        return out

    def texels(self, size: int):
        """3D position of every texel (rows top-down) and whether the mesh covers it."""
        prog = self.ctx.program(vertex_shader=UV_VS, fragment_shader=UV_FS)
        data = np.hstack([self.uv, self.vertices]).astype("f4")
        vao = self.ctx.vertex_array(prog, [(self.ctx.buffer(data), "2f 3f", "uv", "pos")],
                                    self.ctx.buffer(self.faces.astype("i4")))
        tex = self.ctx.texture((size, size), 4, dtype="f4")
        fbo = self.ctx.framebuffer([tex])
        fbo.use()
        fbo.clear(0, 0, 0, 0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        vao.render()
        p = np.frombuffer(tex.read(), "f4").reshape(size, size, 4)[::-1]
        return p[..., :3], p[..., 3] > 0


def fit_camera(raster: Raster, alpha: np.ndarray) -> tuple[float, FrontCamera]:
    """The face-on camera whose view of the model best overlaps the portrait's outline."""
    h, w = alpha.shape
    mask = alpha > 128
    ys, xs = np.nonzero(mask)
    p_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    best = (-1.0, None)
    for yaw in YAW_SEARCH:
        for pitch in PITCH_SEARCH:
            rot = rotation(yaw, pitch)
            c0, s0 = np.array([w / 2, h / 2]), w / 2.4
            ys2, xs2 = np.nonzero(raster.depth(FrontCamera(rot, s0, c0), (w, h)) > -1e8)
            r_box = np.array([xs2.min(), ys2.min(), xs2.max(), ys2.max()], float)
            k = (p_box[3] - p_box[1]) / (r_box[3] - r_box[1])  # match heights, then centres
            centre = (p_box[:2] + p_box[2:]) / 2 - k * ((r_box[:2] + r_box[2:]) / 2 - c0)
            cam = FrontCamera(rot, s0 * k, centre)
            sil = raster.depth(cam, (w, h)) > -1e8
            iou = (sil & mask).sum() / (sil | mask).sum()
            if iou > best[0]:
                best = (iou, cam)
    return best


def sample(image: np.ndarray, px: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear colours of image at px (n, 2); remap wants 2D maps under 32k a side."""
    n = len(px)
    pad = -n % width
    maps = [np.pad(c.astype(np.float32), (0, pad)).reshape(-1, width) for c in px.T]
    channels = image.shape[2] if image.ndim == 3 else 1
    return cv2.remap(image, *maps, cv2.INTER_LINEAR).reshape(-1, channels)[:n]


def eye_paint(raster, cam, depth, texture, sprite, points):
    """Where the portrait's eyes go in the texture: (texel index, portrait px, weight).
    Each eye is matched to the model's own eye first, so the two line up."""
    size = texture.shape[0]
    h, w = sprite.shape[:2]
    pos, covered = raster.texels(size)
    idx = np.nonzero(covered.ravel())[0]
    px, z = cam.project(pos.reshape(-1, 3)[idx])
    ui, vi = np.clip(px[:, 0].round().astype(int), 0, w - 1), np.clip(px[:, 1].round().astype(int), 0, h - 1)
    front = z >= depth[vi, ui] - DEPTH_TOL
    # The model as the portrait's camera sees it, to find its eyes in.
    view = np.zeros((h, w, 3), np.float32)
    count = np.zeros((h, w), np.float32)
    np.add.at(view, (vi[front], ui[front]), texture.reshape(-1, 3)[idx[front]])
    np.add.at(count, (vi[front], ui[front]), 1)
    view = cv2.cvtColor((view / np.maximum(count, 1)[..., None]).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    photo = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2GRAY)

    weight = np.zeros(len(idx), np.float32)
    source = np.zeros((len(idx), 2), np.float32)
    for upper, lower in EYES:
        eye = points[upper + lower, :2]
        width = np.ptp(eye[:, 0])
        x0, y0 = (eye.min(0) - EYE_GROW * width).astype(int)
        x1, y1 = (eye.max(0) + EYE_GROW * width).astype(int)
        r = int(EYE_SEARCH * width)
        score = cv2.matchTemplate(view[y0 - r:y1 + r, x0 - r:x1 + r].astype(np.float32),
                                  photo[y0:y1, x0:x1].astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, _, _, (bx, by) = cv2.minMaxLoc(score)
        shift = np.array([bx - r, by - r], float)  # the model's eye = the portrait's + shift
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(eye.astype(np.float32)).astype(np.int32), 255)
        grow = int(EYE_GROW * width)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        mask = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), EYE_FEATHER * width)
        a = sample(mask, px - shift)[:, 0] * front
        mine = a > weight
        weight[mine], source[mine] = a[mine], (px - shift)[mine]
    keep = weight > 1e-3
    return idx[keep].astype("i4"), source[keep], np.clip(weight[keep], 0, 1)


def import_glb(glb: str, name: str):
    import trimesh  # only needed here

    mesh = trimesh.load(glb, force="mesh")
    vertices = mesh.vertices.astype(np.float64)
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))[..., ::-1].copy()
    cat = cats.load(name)
    sprite, points = cat.sprite, cat.points
    h, w = sprite.shape[:2]

    raster = Raster(vertices, mesh.faces, mesh.visual.uv)
    iou, cam = fit_camera(raster, sprite[..., 3])
    depth = raster.depth(cam, (w, h))
    print(f"{name}: portrait outline overlap {iou:.3f}")

    # Every vertex's spot in the portrait, and whether the portrait shows it.
    photo_px, z = cam.project(vertices)
    ui = np.clip(photo_px[:, 0].round().astype(int), 0, w - 1)
    vi = np.clip(photo_px[:, 1].round().astype(int), 0, h - 1)
    front = (z >= depth[vi, ui] - DEPTH_TOL) & (sprite[vi, ui, 3] > 128)
    # The face landmarks on the model's surface.
    lx = np.clip(points[:, 0].round().astype(int), 0, w - 1)
    ly = np.clip(points[:, 1].round().astype(int), 0, h - 1)
    landmarks = cam.unproject(points[:, :2], depth[ly, lx])

    eye_index, eye_px, eye_weight = eye_paint(raster, cam, depth, texture.astype(np.float32), sprite, points)
    flat = texture.reshape(-1, 3).astype(np.float32)
    colours = sample(sprite[..., :3].astype(np.float32), eye_px)
    flat[eye_index] = colours * eye_weight[:, None] + flat[eye_index] * (1 - eye_weight[:, None])
    texture = np.clip(flat, 0, 255).astype(np.uint8).reshape(texture.shape)

    mesh_file, texture_file = files(name)
    np.savez_compressed(
        mesh_file, vertices=vertices.astype("f4"), normals=mesh.vertex_normals.astype("f4"),
        uv=mesh.visual.uv.astype("f4"), faces=mesh.faces.astype("i4"),
        photo_px=photo_px.astype("f4"), front=front, landmarks=landmarks.astype("f4"),
        cam_rot=cam.rot.astype("f4"), cam_scale=np.float32(cam.scale),
        eye_index=eye_index, eye_px=eye_px.astype("f4"), eye_weight=eye_weight.astype("f4"))
    cv2.imwrite(str(texture_file), texture)
    print(f"  {len(vertices)} vertices, {front.sum()} seen in the portrait, "
          f"{len(eye_index)} eye texels -> {mesh_file.parent}")


if __name__ == "__main__":
    import_glb(*sys.argv[1:3])
