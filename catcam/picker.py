"""The cat picker, drawn into the preview window.

Glass cards, one per cat, floating over the live picture (softened behind
them). Hovering lights a card's rim; clicking it or pressing its number
picks the cat. It is only an image plus hit-testing, so the app keeps
streaming while it is shown.
"""

import math

import cv2
import numpy as np

from catcam import glass

SIZE = (960, 540)  # also the size of the live preview, so the window keeps its size
COLUMNS = 3
MARGIN, GAP = 40, 22
HEADER, FOOTER = 120, 40
CARD_MAX = (300, 340)  # w, h: a lone cat doesn't fill the whole window
RADIUS = 26
NAME_ROW = 48  # the name's row at the bottom of a card
EMPTY = (64, 58, 52)  # BGR behind the glass before the first frame arrives


def _cutout(sprite: np.ndarray, w: int, h: int) -> np.ndarray:
    """The cat cropped to itself and fitted into w x h (BGRA)."""
    ys, xs = np.nonzero(sprite[..., 3] > 20)
    cat = sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(w / cat.shape[1], h / cat.shape[0])
    size = (max(int(cat.shape[1] * scale), 1), max(int(cat.shape[0] * scale), 1))
    return cv2.resize(cat, size, interpolation=cv2.INTER_AREA)


def _place(canvas: np.ndarray, cat: np.ndarray, x: int, y: int, dim: float = 1.0):
    """Puts a BGRA cutout on canvas at (x, y), standing on a soft contact shadow."""
    h, w = cat.shape[:2]
    alpha = cat[..., 3].astype(np.float32) / 255
    shadow = cv2.GaussianBlur(alpha, (0, 0), 6) * 0.45
    roi = canvas[y + 5:y + 5 + h, x:x + w].astype(np.float32)
    canvas[y + 5:y + 5 + h, x:x + w] = (roi * (1 - shadow[..., None])).astype(np.uint8)
    roi = canvas[y:y + h, x:x + w].astype(np.float32)
    a = alpha[..., None]
    canvas[y:y + h, x:x + w] = (cat[..., :3] * dim * a + roi * (1 - a)).astype(np.uint8)


class Picker:
    def __init__(self, cats: list[tuple[str, np.ndarray, np.ndarray]]):
        """cats: (name, BGRA sprite, BGR background) per cat."""
        self.names = [name for name, _, _ in cats]
        width, height = SIZE
        # More columns for many cats, so the cards keep room for the cat.
        columns = min(len(cats), max(COLUMNS, math.ceil(math.sqrt(len(cats) * 1.5))))
        rows = math.ceil(len(cats) / columns)
        card_w = min((width - 2 * MARGIN - (columns - 1) * GAP) // columns, CARD_MAX[0])
        card_h = min((height - HEADER - FOOTER - (rows - 1) * GAP) // rows, CARD_MAX[1])
        top = HEADER + (height - HEADER - FOOTER - rows * card_h - (rows - 1) * GAP) // 2
        self.cards = []  # (x, y, w, h) per cat
        for i in range(len(cats)):
            row, col = divmod(i, columns)
            in_row = min(columns, len(cats) - row * columns)  # centre a short last row
            x0 = (width - in_row * card_w - (in_row - 1) * GAP) // 2
            self.cards.append((x0 + col * (card_w + GAP), top + row * (card_h + GAP), card_w, card_h))
        pad = 18
        self.cutouts = [_cutout(s, card_w - 2 * pad, card_h - NAME_ROW - pad) for _, s, _ in cats]
        self.hover = None

    def hit(self, x: int, y: int) -> int | None:
        for i, (cx, cy, w, h) in enumerate(self.cards):
            if cx <= x < cx + w and cy <= y < cy + h:
                return i
        return None

    def key_choice(self, key: int) -> int | None:
        """Cat picked by a number key (1-9), if any."""
        index = key - ord("1")
        return index if 0 <= index < min(len(self.names), 9) else None

    def draw(self, current: int | None, frame: np.ndarray | None = None) -> np.ndarray:
        """current: index of the cat being streamed, marked on its card; frame: the
        picture going out right now (BGR, unmirrored), shown softened behind the glass."""
        width, height = SIZE
        if frame is None:
            canvas = np.empty((height, width, 3), np.uint8)
            canvas[:] = EMPTY
        else:
            canvas = glass.backdrop(cv2.flip(frame, 1), SIZE)
        title = "Who are you today?" if len(self.names) > 1 else "Ready when you are"
        glass.text(canvas, title, width / 2, 62, 34, weight=700, family="rounded", align="center")
        if current is None:
            hint = "Click a card or press its number" if len(self.names) > 1 else "Click the card to start"
        else:
            hint = "Pick someone else, or press Esc to go back" if len(self.names) > 1 \
                else "Click the card or press Esc to go back"
        glass.text(canvas, hint, width / 2, 92, 15, glass.MUTED, align="center")

        for i, ((x, y, w, h), cat, name) in enumerate(zip(self.cards, self.cutouts, self.names)):
            hovered, live = i == self.hover, i == current
            glass.panel(canvas, x, y, w, h, RADIUS, glow=1.0 if hovered else 0.0)
            if hovered:
                glass.ring(canvas, x - 2, y - 2, w + 4, h + 4, RADIUS + 2, glass.HONEY)
            cx = x + (w - cat.shape[1]) // 2
            _place(canvas, cat, cx, y + h - NAME_ROW - cat.shape[0], 1.0 if hovered else 0.92)
            baseline = y + h - 18
            name_w = glass.text_width(name, 19, 600, "rounded")
            if i < 9:
                cap_w = max(20, glass.text_width(str(i + 1), 12, 600) + 10)
                left = x + (w - cap_w - 10 - name_w) / 2
                glass.keycap(canvas, int(left), baseline - 15, str(i + 1))
                left += cap_w + 10
            else:
                left = x + (w - name_w) / 2
            glass.text(canvas, name, left, baseline, 19, weight=600, family="rounded")
            if live:
                glass.dot(canvas, x + w - 22, y + 22, 5, glass.HONEY)
                glass.text(canvas, "Live", x + w - 32, y + 27, 13, weight=500, align="right")
        return canvas
