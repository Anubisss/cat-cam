"""A dog's tongue: it lies on the lower teeth when the mouth is open, and
when you stick your tongue out, it slides out over the lower lip and hangs down.

The face tracker has no idea of tongues, so yours is found in the webcam
image: the opening between your lips filled with something as red as your
lips (the inside of a mouth is dark, teeth are not red).

The dog's is a strip of the head mesh: it moves and turns with the lower lip,
a bit in front of it. Its outline (rounded tip), the groove down its middle,
the mouth's shadow and a wet shine are drawn by the shader.
Sizes are in face widths.
"""

import cv2
import numpy as np

from catcam.mouth_fit import LOWER_INNER_LIP, OUTER_LIPS, UPPER_INNER_LIP

ROOT = 0.45  # it starts this far down the gap between the lips...
INCISORS = 0.1  # ...but at least this far (in mouth widths) below the upper lip, behind the front teeth...
SLIT = 0.6  # ...and no further down the gap than this (it can come out of a nearly closed mouth)
INSIDE = 0.02  # in, its tip lies this far behind the lower lip...
HANG = 0.17  # ...and fully out, this far below it
HALF_WIDTH = (0.24, 0.4)  # of the mouth's width, at its root (between the canines) and its tip
LIFT = 0.03  # in front of the lower lip
ROWS = 12

# Your tongue: your lips must be at least OPENING apart. Of the opening, the
# share at least TONGUE_RED as red (Lab a*) and TONGUE_LIGHT as light as your
# lips is the tongue: not out at TONGUE_FILL[0], fully out at [1].
OPENING = 0.015
TONGUE_RED, TONGUE_LIGHT = 0.6, 0.55
TONGUE_FILL = (0.3, 0.65)

VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec4 in_tongue;
out vec4 t;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    t = in_tongue;
}
"""
# Per vertex: u runs -1..1 across, then its distance from the tip, how far
# below the lower lip it is, and its half width.
FRAGMENT_SHADER = """
#version 330
uniform float shade;
uniform vec2 turn;  // image shift per face width of depth (from the head rotation)
in vec4 t;
out vec4 color;

const vec3 TONGUE = vec3(0.88, 0.42, 0.48);

void main() {
    float u = t.x, tip = t.y, below = t.z, half_width = t.w;
    float rounded = half_width * (1.0 - sqrt(max(1.0 - u * u, 0.0)));
    float aa = fwidth(tip) + 1e-5, ua = fwidth(u) + 1e-5;
    float cover = smoothstep(rounded - aa, rounded + aa, tip) * (1.0 - smoothstep(1.0 - ua, 1.0, abs(u)));
    if (cover < 0.01) discard;
    vec3 c = TONGUE;
    c *= 1.0 - 0.32 * exp(-pow(u / 0.09, 2.0)) * smoothstep(0.0, 0.1, tip);  // the groove
    c *= mix(1.0, 0.72, pow(abs(u), 3.0));  // the sides curve away
    c *= mix(0.35, 1.0, smoothstep(-0.05, 0.01, below));  // in the mouth's shadow
    c *= 1.0 + 0.12 * exp(-pow((below - 0.012) / 0.012, 2.0));  // the fold over the lower lip
    c *= mix(0.72, 1.0, smoothstep(0.0, 0.025, tip - rounded));  // the curled rim at the tip
    float shine = exp(-pow((u + 0.35 + 0.8 * turn.x) / 0.2, 2.0))
                  * smoothstep(0.0, 0.05, tip) * smoothstep(-0.01, 0.03, below);
    c += vec3(0.2, 0.18, 0.18) * shine;
    color = vec4(c.bgr * shade, cover);  // the pipeline is BGR
}
"""


def your_tongue(frame: np.ndarray, face: np.ndarray) -> float:
    """0 (in) .. 1 (out): how much of your open mouth your tongue fills.
    frame: BGR webcam image, face: (478, 2+) your face landmarks in its px."""
    width = np.ptp(face[:, 0])
    upper, lower = face[UPPER_INNER_LIP, :2], face[LOWER_INNER_LIP, :2]
    if np.mean(lower[1:-1, 1] - upper[1:-1, 1]) < OPENING * width:
        return 0.0
    lips = face[OUTER_LIPS, :2]
    margin = 2
    x0, y0 = np.maximum(np.floor(lips.min(axis=0)).astype(int) - margin, 0)
    x1, y1 = np.ceil(lips.max(axis=0)).astype(int) + margin
    lab = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_BGR2LAB).astype(np.float32)
    if lab.size == 0:
        return 0.0

    def mask(polygon):
        out = np.zeros(lab.shape[:2], np.uint8)
        cv2.fillPoly(out, [(polygon - (x0, y0)).astype(np.int32)], 255)
        return out

    opening = mask(np.vstack([upper, lower[::-1]]))
    lip = (mask(lips) > 0) & (cv2.dilate(opening, np.ones((5, 5), np.uint8)) == 0)
    if not lip.any() or not opening.any():
        return 0.0
    light, red = lab[..., 0], lab[..., 1] - 128
    lip_light, lip_red = np.median(light[lip]), np.median(red[lip])
    if lip_red <= 0:
        return 0.0
    tongue = (red >= TONGUE_RED * lip_red) & (light >= TONGUE_LIGHT * lip_light)
    fill = tongue[opening > 0].mean()
    return float(np.clip((fill - TONGUE_FILL[0]) / (TONGUE_FILL[1] - TONGUE_FILL[0]), 0, 1))


def triangles() -> np.ndarray:
    """The strip's triangles: ROWS rows of 3 vertices (left, middle, right)."""
    quads = []
    for r in range(ROWS - 1):
        for c in range(2):
            a, b = 3 * r + c, 3 * (r + 1) + c
            quads += [(a, a + 1, b + 1), (a, b + 1, b)]
    return np.array(quads)


def shape(upper: np.ndarray, lower: np.ndarray, mouth_width: float, face_width: float, out: float):
    """The tongue for the middle points (3,) of the upper and lower inner lip
    (sprite px), the mouth's width (px) and how far out it is (0..1): None
    while it doesn't show, else vertex positions (ROWS * 3, 3) sprite px and
    their shader inputs (ROWS * 3, 4)."""
    gap = max(lower[1] - upper[1], 0.0)
    root = upper[1] + min(max(ROOT * gap, INCISORS * mouth_width), SLIT * gap)
    tip = lower[1] + (out * (HANG + INSIDE) - INSIDE) * face_width
    if tip <= root:
        return None
    along = np.linspace(0, 1, ROWS)  # root to tip
    y = root + along * (tip - root)
    half = mouth_width * (HALF_WIDTH[0] + along * (HALF_WIDTH[1] - HALF_WIDTH[0]))
    u = np.array([-1.0, 0.0, 1.0])
    x = lower[0] + half[:, None] * u
    z = np.full_like(x, lower[2] - LIFT * face_width)
    points = np.stack([x, np.repeat(y[:, None], 3, axis=1), z], axis=2).reshape(-1, 3)
    inputs = np.stack([
        np.broadcast_to(u, x.shape),
        np.repeat(((tip - y) / face_width)[:, None], 3, axis=1),
        np.repeat(((y - lower[1]) / face_width)[:, None], 3, axis=1),
        np.repeat((half / face_width)[:, None], 3, axis=1),
    ], axis=2).reshape(-1, 4)
    return points, inputs.astype("f4")
