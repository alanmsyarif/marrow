"""Chunked variants of the slow passes, for the modal progress driver.

Tetrahedralize blocks Blender's main thread, so a long run is
indistinguishable from a hang - measured at 66s on a 34k-vertex mesh at
Resolution 0.08, with the window reporting Not Responding throughout. The
fix is to run each slow pass as a generator the operator can pull from a
slice at a time, yielding to Blender's event loop in between.

The generator is the implementation and the blocking function drains it, so
there is exactly one copy of each algorithm. These tests pin that: drained
output must equal what the blocking call produced before the split.
"""

import numpy as np

from marrow.core.bind import bind_points, bind_points_iter
from marrow.core.coloring import color_sets, color_sets_iter, color_tets
from marrow.core.lattice import build_lattice
from marrow.core.progress import drain

MESH = build_lattice(np.zeros(3), 0.25, np.ones((4, 4, 4), dtype=bool))


def _fractions(work):
    """Run a generator to exhaustion, collecting what it reported."""
    seen = []
    while True:
        try:
            seen.append(next(work))
        except StopIteration as done:
            return seen, done.value


def _assert_well_behaved(seen, what):
    assert seen, f"{what}: reported no progress at all"
    assert all(0.0 <= f <= 1.0 for f in seen), (
        f"{what}: fraction outside 0..1: min {min(seen)} max {max(seen)}"
    )
    assert seen == sorted(seen), f"{what}: progress went backwards"
    assert seen[-1] == 1.0, f"{what}: finished at {seen[-1]}, not 1.0"


# --- drain -----------------------------------------------------------------

def test_drain_returns_the_generators_return_value():
    def work():
        yield 0.5
        return "done"

    assert drain(work()) == "done"


def test_drain_of_a_generator_that_yields_nothing_still_returns():
    def work():
        return 7
        yield  # pragma: no cover - makes it a generator

    assert drain(work()) == 7


# --- colouring -------------------------------------------------------------

def test_chunked_colouring_matches_the_blocking_call():
    seen, colors = _fractions(color_sets_iter(MESH.tets, MESH.n_nodes, block=64))
    _assert_well_behaved(seen, "color_sets_iter")
    assert np.array_equal(colors, color_sets(MESH.tets, MESH.n_nodes))


def test_colouring_is_unaffected_by_where_the_chunks_fall():
    """Greedy colouring is order dependent, so chunking must not reorder it."""
    a = drain(color_sets_iter(MESH.tets, MESH.n_nodes, block=1))
    b = drain(color_sets_iter(MESH.tets, MESH.n_nodes, block=10_000))
    assert np.array_equal(a, b)
    assert np.array_equal(a, color_tets(MESH.tets, MESH.n_nodes))


def test_colouring_an_empty_set_reports_complete():
    seen, colors = _fractions(color_sets_iter(np.zeros((0, 4), dtype=np.int32), 0))
    assert colors.size == 0
    assert seen[-1] == 1.0


# --- binding ---------------------------------------------------------------

def test_chunked_bind_matches_the_blocking_call():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.02, 0.98, size=(300, 3))

    seen, (idx, w) = _fractions(
        bind_points_iter(MESH.nodes, MESH.tets, pts, block=32)
    )
    _assert_well_behaved(seen, "bind_points_iter")

    want_idx, want_w = bind_points(MESH.nodes, MESH.tets, pts)
    assert np.array_equal(idx, want_idx)
    assert np.allclose(w, want_w, atol=1e-12)


def test_chunked_bind_matches_for_points_outside_the_cage_too():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.5, 2.5, size=(200, 3))
    idx, w = drain(bind_points_iter(MESH.nodes, MESH.tets, pts, block=16))
    want_idx, want_w = bind_points(MESH.nodes, MESH.tets, pts)
    assert np.array_equal(idx, want_idx)
    assert np.allclose(w, want_w, atol=1e-12)


def test_a_block_larger_than_the_point_count_is_one_chunk():
    pts = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    seen, (idx, _) = _fractions(
        bind_points_iter(MESH.nodes, MESH.tets, pts, block=10_000)
    )
    assert seen == [1.0]
    assert idx.shape == (2,)


def test_binding_no_points_reports_complete():
    seen, (idx, w) = _fractions(
        bind_points_iter(MESH.nodes, MESH.tets, np.zeros((0, 3)))
    )
    assert idx.shape == (0,) and w.shape == (0, 4)
    assert seen[-1] == 1.0


def test_the_grid_is_built_once_for_the_whole_run_not_per_chunk():
    """Rebuilding per chunk would be O(tets) per block and undo the point.

    Counted rather than timed, so it cannot pass on a fast machine.
    """
    import marrow.core.bind as bind_mod

    built = 0
    real = bind_mod._TetGrid

    class Counted(real):
        def __init__(self, *a, **k):
            nonlocal built
            built += 1
            super().__init__(*a, **k)

    bind_mod._TetGrid = Counted
    try:
        rng = np.random.default_rng(2)
        pts = rng.uniform(0.02, 0.98, size=(200, 3))
        drain(bind_points_iter(MESH.nodes, MESH.tets, pts, block=8))
    finally:
        bind_mod._TetGrid = real

    assert built == 1, f"grid rebuilt {built} times, expected once"
