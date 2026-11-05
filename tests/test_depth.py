import numpy as np

from catcam.depth import to_png


def test_to_png_stretches_the_depth_over_the_cat():
    disparity = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 99.0]], np.float32)
    alpha = np.array([[0, 255, 255], [255, 255, 0]], np.uint8)
    png = to_png(disparity, alpha)
    assert png.dtype == np.uint16
    assert png[0, 1] == 0 and png[1, 1] == 65535  # the cat's farthest and nearest points
    assert png[0, 2] == 21845  # a third of the way
    assert png[0, 0] == 0 and png[1, 2] == 0  # around the cat
