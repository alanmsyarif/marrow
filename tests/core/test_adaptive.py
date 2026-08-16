"""Core tests for the adaptive octree lattice, no GPU and no bpy."""

import numpy as np

from marrow.core.adaptive import balance, build_adaptive_lattice, refine
from marrow.core.coloring import color_sets
from marrow.core.lattice import build_lattice, grid_dims
from marrow.core.tetmesh import signed_volumes


class BoxOracle:
    """Analytic oracle for a union of axis-aligned boxes.

    Distance is the minimum over the boxes' surface distances, which can only
    underestimate the true union distance near junctions - conservative, and
    conservative merely over-refines.
    """

    def __init__(self, boxes):
        self.boxes = [(np.asarray(lo, float), np.asarray(hi, float))
                      for lo, hi in boxes]
        self.bounds_min = np.min([lo for lo, _ in self.boxes], axis=0)
        self.bounds_max = np.max([hi for _, hi in self.boxes], axis=0)

    def distance(self, p):
        p = np.asarray(p, dtype=np.float64)
        best = np.inf
        for lo, hi in self.boxes:
            if np.all(p > lo) and np.all(p < hi):
                d = float(np.min(np.minimum(p - lo, hi - p)))
            else:
                d = float(np.linalg.norm(p - np.clip(p, lo, hi)))
            best = min(best, d)
        return best

    def inside(self, p):
        p = np.asarray(p, dtype=np.float64)
        return any(
            np.all(p > lo) and np.all(p < hi) for lo, hi in self.boxes
        )


def _stub_oracle():
    """Unit bulk plus a thin stub, the thick-plus-thin canonical shape."""
    return BoxOracle([
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([1.0, 0.4, 0.4], [1.2, 0.6, 0.6]),
    ])


def _centre(cell, max_size):
    lv, i, j, k = cell
    size = max_size / (1 << lv)
    return (np.array([i, j, k]) + 0.5) * size


def test_thin_stub_refines_to_min_while_bulk_stays_coarse():
    oracle = _stub_oracle()
    leaves = refine(0.25, 0.03, oracle)

    # Leaves actually inside the stub must resolve its 0.2 cross-section
    # with at least two cells across. (The domain shell around the stub is
    # empty space; it coarsens freely and says nothing about the feature.)
    stub = [c for c in leaves
            if c[0] > 0 and oracle.inside(_centre(c, 0.25))
            and _centre(c, 0.25)[0] > 1.0]
    assert stub, "the stub must produce leaves"
    for c in stub:
        assert 0.2 / (0.25 / (1 << c[0])) >= 2.0

    bulk = [c for c in leaves if np.all(_centre(c, 0.25) > 0.3)
            and np.all(_centre(c, 0.25) < 0.7)]
    assert bulk, "deep bulk must produce leaves"
    assert min(c[0] for c in bulk) < max(c[0] for c in stub), \
        "bulk must stay coarser than the stub"


def test_balance_keeps_adjacent_levels_within_one():
    oracle = _stub_oracle()
    max_size = 0.25
    leaves = balance(refine(max_size, 0.03, oracle),
                     grid_dims(oracle.bounds_min, oracle.bounds_max, max_size))
    leaf_set = set(map(tuple, leaves))

    def covering(level, ijk):
        """Level of the leaf containing (level, ijk).

        None means the region is split into finer leaves, which satisfies
        the invariant just as well. Walking up past the region would
        misread that case as a coarse covering leaf.
        """
        for lvv in range(level, -1, -1):
            shift = level - lvv
            c = (ijk[0] >> shift, ijk[1] >> shift, ijk[2] >> shift)
            if (lvv, *c) in leaf_set:
                return lvv
        return None

    dims = grid_dims(oracle.bounds_min, oracle.bounds_max, max_size)
    for lv, i, j, k in leaves:
        if lv == 0:
            continue
        pi, pj, pk = i >> 1, j >> 1, k >> 1
        span = 1 << (lv - 1)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
                  (0, 0, -1)):
            nb = (pi + d[0], pj + d[1], pk + d[2])
            if not (0 <= nb[0] < dims[0] * span
                    and 0 <= nb[1] < dims[1] * span
                    and 0 <= nb[2] < dims[2] * span):
                continue  # outside the domain: nothing to balance against
            cov = covering(lv - 1, nb)
            assert cov is None or cov >= lv - 1


def test_blend_rows_interpolate_their_masters_at_rest():
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    assert idx.shape[0] > 0, "a level boundary must produce glue rows"
    assert idx.shape[1] == 5 and w.shape[1] == 4
    assert idx.max() < mesh.n_nodes

    assert np.allclose(w.sum(axis=1), 1.0)
    nonzero = (w > 0).sum(axis=1)
    assert set(nonzero.tolist()) <= {2, 4}
    edge = nonzero == 2
    assert np.allclose(w[edge].max(axis=1), 0.5)

    blended = np.einsum("rw,rwd->rd", w, mesh.nodes[idx[:, 1:]])
    assert np.allclose(blended, mesh.nodes[idx[:, 0]], atol=1e-9)


def test_adaptive_mesh_is_valid_and_tiles_the_unit_cube():
    oracle = BoxOracle([([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
    mesh, idx, w = build_adaptive_lattice(0.25, 0.1, oracle)
    mesh.validate()
    total = signed_volumes(mesh.nodes, mesh.tets).sum()
    assert np.isclose(total, 1.0, atol=1e-6)


def test_single_level_matches_the_uniform_lattice():
    oracle = BoxOracle([([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
    spacing = 0.25
    mesh, idx, w = build_adaptive_lattice(spacing, spacing, oracle)
    assert idx.shape[0] == 0 and w.shape[0] == 0

    dims = grid_dims(oracle.bounds_min, oracle.bounds_max, spacing)
    mask = np.zeros(dims, dtype=bool)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                mask[i, j, k] = oracle.inside((np.array([i, j, k]) + 0.5)
                                              * spacing)
    uniform = build_lattice(np.asarray(oracle.bounds_min), spacing, mask)

    def signature(m):
        # Position-keyed: the two meshes number their nodes differently.
        return {
            tuple(sorted(tuple(round(c, 9) for c in m.nodes[t]) for t in tet))
            for tet in m.tets
        }

    assert signature(mesh) == signature(uniform)


def test_color_sets_keeps_colours_node_disjoint_and_skips_padding():
    rows = [
        [0, 1, 2, -1, -1],
        [1, 3, 4, -1, -1],   # shares node 1 with row 0
        [5, 6, 7, -1, -1],   # disjoint from both
    ]
    colors = color_sets(rows, 8)
    assert colors[0] != colors[1]
    assert colors[2] == colors[0]

    by_color = {}
    for r, c in enumerate(colors):
        by_color.setdefault(int(c), []).append(r)
    for rows_of_color in by_color.values():
        seen = set()
        for r in rows_of_color:
            nodes = {n for n in rows[r] if n >= 0}
            assert not (nodes & seen)
            seen |= nodes


def test_blend_weights_reproduce_translation_exactly():
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    tau = np.array([0.7, -1.3, 2.1])
    moved = mesh.nodes + tau
    blended = np.einsum("rw,rwd->rd", w, moved[idx[:, 1:]])
    assert np.allclose(blended, moved[idx[:, 0]], atol=1e-9)
