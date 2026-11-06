import numpy as np

from catcam.head3d_import import cut_jaw, lip_edge
from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP


def _grid(n=6):
    """A flat n x n grid of unit squares in z = 0, facing +z."""
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)]).astype(float)
    corner = (np.arange(n)[:, None] * (n + 1) + np.arange(n)).ravel()
    faces = np.concatenate([np.column_stack([corner, corner + 1, corner + n + 2]),
                            np.column_stack([corner, corner + n + 2, corner + n + 1])])
    return vertices, faces


def _area_normals(vertices, faces):
    p = vertices[faces]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def test_the_jaw_is_cut_free_along_its_edge():
    vertices, faces = _grid()
    line = 2.4  # the jaw is everything above y = 2.4
    margin = vertices[:, 1] - line
    parts, cut, jaw = cut_jaw({"vertices": vertices}, faces, margin, np.ones(len(vertices)))
    new = parts["vertices"]
    # Every face lies wholly on one side: on the jaw or on the face.
    assert np.ptp(jaw[cut], axis=1).max() == 0
    assert set(np.unique(jaw)) == {0.0, 1.0}
    # The new points lie on the edge, made twice; the jaw's own are above it, the face's below.
    added = new[len(vertices):]
    assert np.allclose(added[:, 1], line)
    assert np.all(new[jaw == 1][:, 1] >= line - 1e-9) and np.all(new[jaw == 0][:, 1] <= line + 1e-9)
    # Nothing is lost or turned over: the same area, all facing the same way.
    normals = _area_normals(new, cut)
    assert np.all(normals[:, 2] > 0)
    assert np.isclose(normals[:, 2].sum(), _area_normals(vertices, faces)[:, 2].sum())


def test_the_jaw_keeps_its_reach():
    vertices, faces = _grid()
    reach = np.clip(1 - vertices[:, 0] / 6, 0, 1)  # fading toward x = 6
    _, cut, jaw = cut_jaw({"vertices": vertices}, faces, vertices[:, 1] - 2.4, reach)
    assert np.allclose(jaw[:len(vertices)], np.where(vertices[:, 1] > 2.4, reach, 0))


def _mouth(depth_rows, fit_y=500.0):
    """Face points 1000 px wide with the lips meeting at fit_y from x 400 to 600, and a
    depth map (1000 px square, 1 px per model unit) given row by row."""
    points = np.zeros((478, 3))
    points[:, 0] = 500
    points[0, 0], points[1, 0] = 0, 1000
    for lip in (UPPER_INNER_LIP, LOWER_INNER_LIP):
        points[lip, 0] = np.linspace(400, 600, len(lip))
        points[lip, 1] = fit_y
    depth = np.repeat(np.asarray(depth_rows, float)[:, None], 1000, axis=1)
    return depth, points


def test_the_lips_meet_where_the_upper_lip_ends():
    # An upper lip hanging 30 px over the lower one, its edge 40 px above the face fit's line.
    rows = np.arange(1000)
    depth, points = _mouth(np.where(rows < 460, 30.0, 0.0) - 0.2 * rows)
    edge = lip_edge(depth, 1.0, points)
    assert np.allclose(edge[:, 0], np.arange(400, 601))
    assert np.all(np.abs(edge[20:-20, 1] - 460) <= 5)


def test_a_smooth_face_keeps_the_fitted_line():
    rows = np.arange(1000)
    # Curving away more and more gently going down: steepest at the top, but nowhere an edge.
    depth, points = _mouth(-0.2 * rows + 1e-3 * (rows - 400) ** 2)
    assert np.allclose(lip_edge(depth, 1.0, points)[:, 1], 500)
    # An edge below the fitted line is the chin's, not the lips'.
    depth, points = _mouth(np.where(rows < 540, 30.0, 0.0) - 0.2 * rows)
    assert np.allclose(lip_edge(depth, 1.0, points)[:, 1], 500)
