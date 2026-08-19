import numpy as np

from marrow.core.fiber import fiber_from_polyline, tet_centroids


def _line(n=5, length=4.0):
    """A straight polyline along +X from the origin."""
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)


def test_centroids_are_the_mean_of_the_four_nodes():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    assert np.allclose(tet_centroids(nodes, tets), [[0.25, 0.25, 0.25]])


def test_straight_line_gives_axis_direction_and_arclength_phase():
    centroids = np.array([[0.5, 0.2, 0.0], [2.5, -0.3, 0.1], [3.9, 0.0, 0.0]])
    out = fiber_from_polyline(_line(), centroids)

    assert out.shape == (3, 4)
    assert np.allclose(out[:, :3], np.array([[1.0, 0.0, 0.0]] * 3), atol=1e-9)
    # Phase is the arclength of the nearest point, which for a line along X
    # is the centroid's own x.
    assert np.allclose(out[:, 3], [0.5, 2.5, 3.9], atol=1e-9)


def test_directions_are_unit_length():
    centroids = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    out = fiber_from_polyline(_line(), centroids)
    assert np.allclose(np.linalg.norm(out[:, :3], axis=1), 1.0, atol=1e-9)


def test_tangent_follows_a_corner():
    """An L: along +X to (1,0,0), then along +Y to (1,2,0)."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    )
    out = fiber_from_polyline(points, np.array([[0.5, 0.0, 0.0], [1.0, 1.5, 0.0]]))
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(out[1, :3], [0.0, 1.0, 0.0], atol=1e-9)
    # Arclength keeps accumulating around the corner: 1.0 along X plus 1.5.
    assert abs(out[1, 3] - 2.5) < 1e-9


def test_duplicate_points_do_not_poison_the_result():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    out = fiber_from_polyline(points, np.array([[1.0, 0.5, 0.0]]))
    assert np.all(np.isfinite(out)), "a zero-length segment must not produce NaN"
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0], atol=1e-9)


def test_a_degenerate_polyline_yields_no_fiber():
    """One point, or every point identical, means there is no direction to
    give. Zero rows are the solver's 'skip this tet' signal."""
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(fiber_from_polyline(np.zeros((1, 3)), centroids), 0.0)
    assert np.allclose(fiber_from_polyline(np.zeros((4, 3)), centroids), 0.0)
