import numpy as np

from catcam.head3d_import import cut_jaw


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
