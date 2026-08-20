import numpy as np

from marrow.core.attach import tet_scalar
from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    make_state,
    precompute,
    solve_constraints,
)

BLOCK = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def test_tet_scalar_averages_the_four_corners():
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    values = np.array([0.0, 1.0, 1.0, 2.0])
    assert tet_scalar(values, tets)[0] == 1.0


def test_tet_scalar_is_uniform_on_a_uniform_paint():
    values = np.full(BLOCK.n_nodes, 0.35)
    out = tet_scalar(values, BLOCK.tets)
    assert out.shape == (BLOCK.n_tets,)
    assert np.allclose(out, 0.35)


def test_tet_scalar_rejects_out_of_range_indices():
    tets = np.array([[0, 1, 2, 7]], dtype=np.int64)
    try:
        tet_scalar(np.zeros(4), tets)
    except ValueError as exc:
        assert "7" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an out-of-range node")


def test_tet_scalar_on_no_tets_is_empty():
    assert tet_scalar(np.zeros(4), np.zeros((0, 4), dtype=np.int64)).shape == (0,)


def _stretched(region, params, h=1.0 / 240.0):
    """One projection of a cage pulled apart along X, with a given region."""
    state = make_state(BLOCK.nodes)
    state.predicted = BLOCK.nodes.copy()
    state.predicted[:, 0] *= 1.3
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    solve_constraints(
        state, BLOCK.tets, dm_inv, rest_vol, params, h, region=region
    )
    return state.predicted


def test_all_ones_region_matches_no_region_exactly():
    """Bit-identical, not merely close: an unpainted body must solve as it
    did before the multiplier existed, or every pre-existing parity test is
    silently measuring something new."""
    params = SolverParams(mu=1.0e4, lam=1.0e5)
    plain = _stretched(None, params)
    ones = _stretched(np.ones(BLOCK.n_tets), params)
    assert np.array_equal(plain, ones)


def test_a_soft_region_pulls_back_less_than_a_stiff_one():
    params = SolverParams(mu=1.0e4, lam=1.0e5)
    stiff = _stretched(np.ones(BLOCK.n_tets), params)
    soft = _stretched(np.full(BLOCK.n_tets, 0.05), params)
    start = BLOCK.nodes.copy()
    start[:, 0] *= 1.3
    # Both recover towards rest; the soft cage has to still be wider.
    assert _extent(soft) > _extent(stiff)
    assert _extent(stiff) < _extent(start)


def test_a_zero_region_projects_nothing():
    """Multiplier zero is the one value that has to short-circuit rather
    than divide: compliance is 1 / (mu * rest_vol)."""
    params = SolverParams(mu=1.0e4, lam=1.0e5)
    out = _stretched(np.zeros(BLOCK.n_tets), params)
    start = BLOCK.nodes.copy()
    start[:, 0] *= 1.3
    assert np.array_equal(out, start)


def test_a_half_soft_cage_deforms_unevenly():
    """The point of the feature: one body, two materials."""
    params = SolverParams(mu=1.0e4, lam=1.0e5)
    centroids = BLOCK.nodes[BLOCK.tets].mean(axis=1)
    region = np.where(centroids[:, 0] < 0.5, 0.02, 1.0)
    out = _stretched(region, params)
    moved = np.abs(out - BLOCK.nodes)[:, 0]
    soft_side = BLOCK.nodes[:, 0] < 0.5
    # The stiff half snaps back harder, so it is the side that moved most
    # relative to where the stretch put it.
    stretched = BLOCK.nodes[:, 0] * 1.3
    recovery = np.abs(out[:, 0] - stretched)
    assert recovery[~soft_side].max() > recovery[soft_side].max()
    assert moved.max() > 0.0


def _extent(nodes):
    return float(nodes[:, 0].max() - nodes[:, 0].min())
