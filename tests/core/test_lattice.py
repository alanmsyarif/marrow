import numpy as np

from marrow.core.lattice import build_lattice, grid_dims
from marrow.core.tetmesh import signed_volumes


def test_grid_dims_rounds_up_to_cover_bounds():
    dims = grid_dims(np.zeros(3), np.array([1.0, 1.0, 1.0]), 0.4)
    assert dims == (3, 3, 3)


def test_grid_dims_is_at_least_one_cell():
    assert grid_dims(np.zeros(3), np.array([0.01, 0.01, 0.01]), 1.0) == (1, 1, 1)


def test_single_cell_makes_five_tets():
    mask = np.ones((1, 1, 1), dtype=bool)
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_tets == 5
    assert mesh.n_nodes == 8


def test_all_tets_have_positive_volume():
    mask = np.ones((3, 3, 3), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    assert np.all(signed_volumes(mesh.nodes, mesh.tets) > 0)
    mesh.validate()


def test_tet_volumes_sum_to_cell_volume():
    mask = np.ones((2, 2, 2), dtype=bool)
    spacing = 0.5
    mesh = build_lattice(np.zeros(3), spacing, mask)
    total = signed_volumes(mesh.nodes, mesh.tets).sum()
    assert np.isclose(total, 8 * spacing**3)


def test_empty_mask_yields_empty_mesh():
    mesh = build_lattice(np.zeros(3), 1.0, np.zeros((2, 2, 2), dtype=bool))
    assert mesh.n_tets == 0
    assert mesh.n_nodes == 0


def test_unused_nodes_are_dropped():
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_nodes == 8  # only the one kept cell's corners


def test_parity_split_is_conforming_across_neighbours():
    """Two adjacent cells must share exactly the 4 nodes of their common face."""
    mask = np.ones((2, 1, 1), dtype=bool)
    mesh = build_lattice(np.zeros(3), 1.0, mask)
    assert mesh.n_nodes == 12  # 8 + 4 new, shared face reused
    assert mesh.n_tets == 10


def test_node_coordinates_are_correct_on_a_non_cubic_grid():
    """Guards against axis transposition in the corner-id decode.

    A cubic mask cannot catch an x/z swap: its coordinate set is symmetric
    under one. Unequal dims plus an off-origin bounds_min plus an asymmetric
    occupied cell make any axis permutation change the result.
    """
    mask = np.zeros((3, 1, 2), dtype=bool)
    mask[2, 0, 1] = True
    bounds_min = np.array([1.0, -2.0, 0.5])
    mesh = build_lattice(bounds_min, 0.5, mask)

    assert mesh.n_nodes == 8
    expected = {
        (x, y, z)
        for x in (2.0, 2.5)
        for y in (-2.0, -1.5)
        for z in (1.0, 1.5)
    }
    actual = {tuple(round(float(c), 9) for c in p) for p in mesh.nodes}
    assert actual == expected
