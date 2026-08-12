import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import SolverParams, make_state, precompute, step

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))


def test_precompute_shapes_and_rest_volume():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    assert dm_inv.shape == (CUBE.n_tets, 3, 3)
    assert rest_vol.shape == (CUBE.n_tets,)
    assert np.isclose(rest_vol.sum(), 1.0)
    assert np.all(rest_vol > 0)


def test_free_fall_matches_analytic_curve():
    """With no constraints and no pins, the body is in free fall."""
    params = SolverParams(dt=1 / 24, substeps=10, damping=1.0, mu=0.0, lam=0.0)
    state = make_state(CUBE.nodes)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)

    start_z = state.nodes[:, 2].copy()
    n_frames = 12
    for _ in range(n_frames):
        step(state, CUBE.tets, dm_inv, rest_vol, params)

    t = n_frames * params.dt
    expected_drop = 0.5 * 9.81 * t * t
    actual_drop = (start_z - state.nodes[:, 2]).mean()
    # Symplectic Euler over-integrates slightly; 2% is the honest tolerance.
    assert np.isclose(actual_drop, expected_drop, rtol=0.02)


def test_pinned_nodes_never_move():
    params = SolverParams(mu=0.0, lam=0.0)
    state = make_state(CUBE.nodes, pinned=np.array([0, 1], dtype=np.int32))
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    before = state.nodes[[0, 1]].copy()
    for _ in range(5):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    assert np.allclose(state.nodes[[0, 1]], before)


def test_pinned_nodes_have_zero_inverse_mass():
    state = make_state(CUBE.nodes, pinned=np.array([3], dtype=np.int32))
    assert state.inv_mass[3] == 0.0
    assert np.all(state.inv_mass[[0, 1, 2]] > 0.0)


def test_damping_reduces_speed():
    fast = SolverParams(damping=1.0, mu=0.0, lam=0.0)
    slow = SolverParams(damping=0.5, mu=0.0, lam=0.0)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)

    a = make_state(CUBE.nodes)
    b = make_state(CUBE.nodes)
    for _ in range(5):
        step(a, CUBE.tets, dm_inv, rest_vol, fast)
        step(b, CUBE.tets, dm_inv, rest_vol, slow)

    assert np.linalg.norm(b.velocities) < np.linalg.norm(a.velocities)
