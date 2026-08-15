import numpy as np
import pytest

from marrow.core.attach import synth_weights, targets_from


def _grid_verts(n):
    """An n x n x n lattice of vertices on the unit cube."""
    ticks = np.linspace(0.0, 1.0, n)
    x, y, z = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


VERTS = _grid_verts(5)  # 125 vertices


def test_weights_are_a_partition_of_unity():
    nodes = np.random.default_rng(7).uniform(0.0, 1.0, (50, 3))
    idx, w = synth_weights(nodes, VERTS)
    assert idx.shape == (50, 4)
    assert w.shape == (50, 4)
    assert np.all(w >= 0.0)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert idx.min() >= 0 and idx.max() < VERTS.shape[0]


def test_weights_pick_the_nearest_vertices():
    node = np.array([[0.51, 0.51, 0.51]])
    idx, w = synth_weights(node, VERTS)
    picked = set(int(i) for i in idx[0])
    dists = np.linalg.norm(VERTS - node, axis=1)
    truth = set(int(i) for i in np.argsort(dists)[:4])
    assert picked == truth
    # Closer vertices carry more weight.
    order = np.argsort(-w[0])
    picked_d = dists[idx[0]]
    assert np.all(np.diff(picked_d[order]) >= -1e-12)


def test_coincident_vertex_takes_the_whole_weight():
    exact = VERTS[17][None, :]
    idx, w = synth_weights(exact, VERTS)
    assert 17 in idx[0]
    slot = int(np.flatnonzero(idx[0] == 17)[0])
    assert w[0, slot] == 1.0
    assert w[0].sum() == 1.0


def test_blend_is_linear_in_the_vertex_positions():
    """The per-frame blend must commute with any affine motion of the
    vertices: applying the motion to the vertices and blending gives the
    same positions as blending at rest and applying the motion. This is
    what lets targets ride the bones - rotation, scale and translation
    together - without reimplementing skinning.

    The blend anchors each node to its weighted handful of vertices, not
    to the node's own rest position, so the invariant is stated about
    that anchor.
    """
    rng = np.random.default_rng(3)
    nodes = rng.uniform(0.0, 1.0, (30, 3))
    idx, w = synth_weights(nodes, VERTS)
    anchor = targets_from(idx, w, VERTS)
    # The anchor approximates the node it belongs to.
    assert np.allclose(anchor, nodes, atol=0.26)  # grid spacing is 0.25

    theta = 0.7
    rot = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    moved = (VERTS @ rot.T) * 1.3 + np.array([2.0, -1.0, 0.5])
    targets = targets_from(idx, w, moved)
    expected = (anchor @ rot.T) * 1.3 + np.array([2.0, -1.0, 0.5])
    assert np.allclose(targets, expected, atol=1e-9)


def test_too_few_vertices_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        synth_weights(np.zeros((1, 3)), np.zeros((3, 3)), k=4)


def test_stale_weights_against_a_smaller_mesh_are_rejected():
    idx = np.array([[0, 1, 2, 3]], dtype=np.int32)
    w = np.full((1, 4), 0.25)
    with pytest.raises(ValueError, match="vertices"):
        targets_from(idx, w, np.zeros((2, 3)))


def test_empty_input_round_trips():
    idx, w = synth_weights(np.zeros((0, 3)), VERTS)
    assert idx.shape == (0, 4)
    assert targets_from(idx, w, VERTS).shape == (0, 3)
