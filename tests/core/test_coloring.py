import numpy as np

from marrow.core.coloring import color_groups, color_tets
from marrow.core.lattice import build_lattice


def test_disjoint_tets_share_one_color():
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    colors = color_tets(tets, 8)
    assert colors[0] == colors[1] == 0


def test_tets_sharing_a_node_get_different_colors():
    tets = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    colors = color_tets(tets, 7)
    assert colors[0] != colors[1]


def test_no_two_tets_in_a_color_share_a_node():
    mask = np.ones((3, 3, 3), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    colors = color_tets(mesh.tets, mesh.n_nodes)
    for group in color_groups(colors):
        seen = set()
        for tet in mesh.tets[group]:
            nodes = set(tet.tolist())
            assert not (seen & nodes), "colour group has a node collision"
            seen |= nodes


def test_every_tet_gets_a_color():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    colors = color_tets(mesh.tets, mesh.n_nodes)
    assert colors.shape == (mesh.n_tets,)
    assert np.all(colors >= 0)


def test_color_groups_partition_all_tets():
    mask = np.ones((2, 2, 2), dtype=bool)
    mesh = build_lattice(np.zeros(3), 0.5, mask)
    groups = color_groups(color_tets(mesh.tets, mesh.n_nodes))
    assert sum(len(g) for g in groups) == mesh.n_tets
    assert len(set(np.concatenate(groups).tolist())) == mesh.n_tets


def test_empty_input():
    colors = color_tets(np.zeros((0, 4), dtype=np.int32), 0)
    assert colors.shape == (0,)
    assert color_groups(colors) == []
