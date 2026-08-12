import numpy as np
import pytest

from marrow.core.bind import bind_points, deform
from marrow.core.lattice import build_lattice

NODES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
TETS = np.array([[0, 1, 2, 3]], dtype=np.int32)

# One healthy tet plus one whose four nodes are coplanar, so it has no volume.
MIXED_NODES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
    ]
)
MIXED_TETS = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)


def test_corner_binds_to_pure_weight():
    idx, w = bind_points(NODES, TETS, NODES[1][None, :])
    assert idx[0] == 0
    assert np.allclose(w[0], [0.0, 1.0, 0.0, 0.0], atol=1e-9)


def test_weights_sum_to_one():
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [0.0, 0.0, 0.0]])
    _, w = bind_points(NODES, TETS, pts)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_interior_weights_are_non_negative():
    pts = np.array([[0.2, 0.2, 0.2]])
    _, w = bind_points(NODES, TETS, pts)
    assert np.all(w >= -1e-12)


def test_deform_reproduces_original_points_when_cage_unmoved():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.05, 0.95, size=(50, 3))
    idx, w = bind_points(mesh.nodes, mesh.tets, pts)
    out = deform(mesh.nodes, mesh.tets, idx, w)
    assert np.allclose(out, pts, atol=1e-9)


def test_deform_follows_a_rigid_translation():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    pts = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.9]])
    idx, w = bind_points(mesh.nodes, mesh.tets, pts)
    shift = np.array([1.0, -2.0, 0.5])
    out = deform(mesh.nodes + shift, mesh.tets, idx, w)
    assert np.allclose(out, pts + shift, atol=1e-9)


def test_point_outside_all_tets_still_binds():
    pts = np.array([[5.0, 5.0, 5.0]])
    idx, w = bind_points(NODES, TETS, pts)
    assert idx[0] == 0
    assert np.isclose(w.sum(), 1.0)


def test_degenerate_tet_is_named_not_left_to_linalg():
    """The solve is batched, so one flat tet fails every point in the mesh.

    Without the guard this surfaces as a bare LinAlgError from inside numpy,
    with nothing pointing at which tet is at fault - and it fires even for a
    point sitting comfortably inside a healthy tet.
    """
    inside_the_good_tet = np.array([[0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="degenerate"):
        bind_points(MIXED_NODES, MIXED_TETS, inside_the_good_tet)


def test_degenerate_tet_error_names_the_offender():
    with pytest.raises(ValueError, match=r"\b1\b"):
        bind_points(MIXED_NODES, MIXED_TETS, np.array([[0.1, 0.1, 0.1]]))


def test_healthy_subset_of_the_same_nodes_still_binds():
    """Guard rejects the bad tet, not the mesh it happens to share nodes with."""
    idx, w = bind_points(MIXED_NODES, MIXED_TETS[:1], np.array([[0.1, 0.1, 0.1]]))
    assert idx[0] == 0
    assert np.isclose(w.sum(), 1.0)


def test_non_finite_nodes_are_rejected_loudly():
    """A NaN used to reach the uniform-0.25 fallback and bind silently."""
    nodes = NODES.copy()
    nodes[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        bind_points(nodes, TETS, np.array([[0.1, 0.1, 0.1]]))


def test_non_finite_points_are_rejected_loudly():
    with pytest.raises(ValueError, match="NaN or inf"):
        bind_points(NODES, TETS, np.array([[0.1, np.inf, 0.1]]))
