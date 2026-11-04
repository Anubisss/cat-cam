import cv2
import numpy as np
import pytest

from catcam.picker import SIZE, Picker


def _cats(n):
    sprite = np.zeros((60, 40, 4), np.uint8)
    sprite[10:50, 5:35] = (40, 80, 120, 255)
    return [(f"Cat{i}", sprite, np.array([90, 90, 90], np.uint8)) for i in range(n)]


@pytest.mark.parametrize("n", [*range(1, 10), 16, 24, 40])
def test_cards_fit_and_do_not_overlap(n):
    picker = Picker(_cats(n))
    width, height = SIZE
    for x, y, w, h in picker.cards:
        assert 0 <= x and x + w <= width and 0 <= y and y + h <= height
    for i, a in enumerate(picker.cards):
        for b in picker.cards[i + 1:]:
            apart = a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
            assert apart


def test_hit_finds_the_card_under_the_mouse():
    picker = Picker(_cats(6))
    for i, (x, y, w, h) in enumerate(picker.cards):
        assert picker.hit(x + w // 2, y + h // 2) == i
    assert picker.hit(2, 2) is None


def test_number_keys():
    picker = Picker(_cats(3))
    assert picker.key_choice(ord("1")) == 0
    assert picker.key_choice(ord("3")) == 2
    assert picker.key_choice(ord("4")) is None
    assert picker.key_choice(-1 & 0xFF) is None


def test_draw():
    picker = Picker(_cats(6))
    picker.hover = 2
    image = picker.draw(current=4)
    assert image.shape == (SIZE[1], SIZE[0], 3)
    x, y, w, h = picker.cards[2]
    ring = image[y - 2, x + w // 2]
    assert ring[2] > 200  # hovered card has the orange ring (BGR)
    assert cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std() > 10
