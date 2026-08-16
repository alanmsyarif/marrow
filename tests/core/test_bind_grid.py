"""The accelerated bind must agree with the brute force it replaces, exactly.

bind_points used to solve a 3x3 system for every (point, tet) pair, which is
200ns per pair measured - 57 minutes on a 34k-vertex mesh against a 500k-tet
cage, with Blender's main thread blocked the whole time. The spatial grid
tests only the tets whose bounding box covers the point.

Same answer or it is not a speedup, it is a behaviour change: the bind index
and weights are written into the mesh and every later frame reads them, so a
different tie-break here is a different silhouette forever.
"""

import numpy as np

from marrow.core.bind import bind_points
from marrow.core.lattice import build_lattice
from marrow.core.tetmesh import signed_volumes


def _brute_force(nodes, tets, points):
    """The original algorithm, kept here as the oracle.

    Deliberately a transcription of what bind_points did before the grid,
    including its tie-breaks: first tet in index order among the most
    interior containments, and np.argmin's first-wins on the nearest
    centroid fallback.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    idx = np.zeros(points.shape[0], dtype=np.int32)
    weights = np.zeros((points.shape[0], 4), dtype=np.float64)
    centroids = nodes[tets].mean(axis=1)

    for i in range(points.shape[0]):
        p0 = nodes[tets[:, 0]]
        m = np.stack(
            [nodes[tets[:, 1]] - p0, nodes[tets[:, 2]] - p0, nodes[tets[:, 3]] - p0],
            axis=2,
        )
        solved = np.linalg.solve(m, (points[i][None, :] - p0)[..., None])[..., 0]
        bary = np.concatenate([(1.0 - solved.sum(axis=1))[:, None], solved], axis=1)

        inside = np.all(bary >= -1e-9, axis=1)
        if np.any(inside):
            candidates = np.flatnonzero(inside)
            best = candidates[np.argmax(bary[candidates].min(axis=1))]
            w = bary[best]
        else:
            best = int(np.argmin(np.linalg.norm(centroids - points[i], axis=1)))
            w = np.clip(bary[best], 0.0, None)
            w = w / w.sum()
        idx[i] = best
        weights[i] = w
    return idx, weights


def _assert_same(nodes, tets, points, what):
    want_idx, want_w = _brute_force(nodes, tets, points)
    got_idx, got_w = bind_points(nodes, tets, points)

    bad = np.flatnonzero(want_idx != got_idx)
    assert bad.size == 0, (
        f"{what}: {bad.size} of {len(points)} points bound to a different tet, "
        f"first at point {bad[0]} ({points[bad[0]]}): "
        f"brute force {want_idx[bad[0]]}, grid {got_idx[bad[0]]}"
    )
    assert np.allclose(got_w, want_w, atol=1e-12), (
        f"{what}: weights differ, max {np.abs(got_w - want_w).max():.3e}"
    )


UNIFORM = build_lattice(np.zeros(3), 0.25, np.ones((4, 4, 4), dtype=bool))


def test_points_inside_the_cage_bind_identically():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.02, 0.98, size=(200, 3))
    _assert_same(UNIFORM.nodes, UNIFORM.tets, pts, "interior points")


def test_points_outside_the_cage_bind_identically():
    """The nearest-centroid fallback. Hot on any mesh with thin features:
    a tentacle thinner than one cell has surface vertices in no tet at all."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.5, 2.5, size=(200, 3))
    outside = pts[(pts < 0.0).any(axis=1) | (pts > 1.0).any(axis=1)]
    _assert_same(UNIFORM.nodes, UNIFORM.tets, outside, "exterior points")


def test_points_on_faces_and_nodes_bind_identically():
    """Exact boundary hits are where a tie-break difference would show."""
    pts = np.vstack([
        UNIFORM.nodes,                                   # every cage node
        UNIFORM.nodes[UNIFORM.tets].mean(axis=1)[:80],   # tet centroids
        UNIFORM.nodes[UNIFORM.tets[:80, :2]].mean(axis=1),   # edge midpoints
    ])
    _assert_same(UNIFORM.nodes, UNIFORM.tets, pts, "boundary points")


def test_a_ragged_cage_binds_identically():
    """Occupancy with holes, so cells are missing and the grid is sparse."""
    rng = np.random.default_rng(2)
    mask = rng.random((6, 6, 6)) > 0.45
    mask[0, 0, 0] = True
    mesh = build_lattice(np.zeros(3), 0.25, mask)
    pts = rng.uniform(-0.3, 1.8, size=(200, 3))
    _assert_same(mesh.nodes, mesh.tets, pts, "ragged cage")


def test_wildly_uneven_tet_sizes_bind_identically():
    """An adaptive cage mixes cell sizes, so one grid pitch cannot suit all.

    Built by hand rather than through the octree: a coarse block next to a
    fine one is the property that matters, and this states it directly.
    """
    coarse = build_lattice(np.zeros(3), 1.0, np.ones((2, 2, 2), dtype=bool))
    fine = build_lattice(np.array([2.0, 0.0, 0.0]), 0.1,
                         np.ones((4, 4, 4), dtype=bool))
    nodes = np.vstack([coarse.nodes, fine.nodes])
    tets = np.vstack([coarse.tets, fine.tets + coarse.n_nodes]).astype(np.int32)
    assert np.all(signed_volumes(nodes, tets) > 0.0), "fixture built a bad cage"

    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.5, 3.0, size=(250, 3))
    _assert_same(nodes, tets, pts, "mixed tet sizes")


def test_a_single_tet_cage_still_binds_identically():
    """Degenerate grid: one cell, one tet, inside and outside both."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    pts = np.array([[0.1, 0.1, 0.1], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
    _assert_same(nodes, tets, pts, "single tet")


def test_the_grid_actually_narrows_the_search():
    """Guards the point of the change: same answer, far less work.

    Without this the whole thing could quietly fall back to brute force and
    every equivalence test above would still pass.
    """
    from marrow.core.bind import _TetGrid

    mesh = build_lattice(np.zeros(3), 0.1, np.ones((10, 10, 10), dtype=bool))
    grid = _TetGrid(mesh.nodes, mesh.tets)
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.05, 0.95, size=(50, 3))

    worst = max(grid.candidates(p).size for p in pts)
    assert worst < mesh.n_tets / 20, (
        f"grid examined {worst} of {mesh.n_tets} tets - no better than brute force"
    )
