import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    attach_compliance,
    make_state,
    precompute,
    solve_attachment,
    solve_constraints,
    step,
)

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))
BLOCK = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def _total_volume(nodes, tets):
    from marrow.core.tetmesh import signed_volumes

    return signed_volumes(nodes, tets).sum()


def _rest_hold(substeps, frames=5):
    """Hold an undeformed cube under no gravity. Returns (max drift, volume)."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps)
    state = make_state(CUBE.nodes)
    before = state.nodes.copy()
    for _ in range(frames):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    return (
        float(np.abs(state.nodes - before).max()),
        _total_volume(state.nodes, CUBE.tets),
    )


def test_rest_configuration_converges_to_a_fixed_point():
    """An undeformed body under no gravity holds still as substeps refine.

    The rest state is stress-free in energy: gamma = 1 + mu/lam makes the
    deviatoric and hydrostatic gradients cancel exactly at F = I. But XPBD
    projects the two constraints sequentially, so each substep leaves a
    Gauss-Seidel residual. That residual is discretisation error, not bias,
    and must fall off with substep count. A formulation that zeroed the
    deviatoric term at rest instead would leave a drift that never converges
    and settle the body at det(F) = gamma.
    """
    coarse, _ = _rest_hold(4)
    fine, fine_volume = _rest_hold(40)
    assert fine < coarse / 100.0, f"rest drift did not converge: {coarse} -> {fine}"
    assert np.isclose(fine_volume, 1.0, rtol=1e-3), (
        f"cube did not hold its rest volume: {fine_volume}"
    )


def test_stretched_body_is_pulled_back():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=20)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    state.nodes[:, 0] *= 1.5  # stretch along x
    stretched_span = np.ptp(state.nodes[:, 0])
    for _ in range(20):
        step(state, CUBE.tets, dm_inv, rest_vol, params)
    assert np.ptp(state.nodes[:, 0]) < stretched_span


def test_volume_is_preserved_under_compression():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=20, lam=1.0e7)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    state = make_state(BLOCK.nodes)
    rest_total = rest_vol.sum()

    state.nodes[:, 2] *= 0.6  # squash in z
    for _ in range(40):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)

    assert np.isclose(_total_volume(state.nodes, BLOCK.tets), rest_total, rtol=0.10)


def test_solver_stays_finite_under_extreme_deformation():
    params = SolverParams(substeps=10)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    state = make_state(BLOCK.nodes)
    state.nodes *= 3.0
    for _ in range(30):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)
    assert np.all(np.isfinite(state.nodes)), "solver produced NaN or inf"


def test_pinned_nodes_stay_put_with_constraints_active():
    params = SolverParams(substeps=10)
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    pinned = np.array([0, 1, 2], dtype=np.int32)
    state = make_state(BLOCK.nodes, pinned=pinned)
    before = state.nodes[pinned].copy()
    for _ in range(20):
        step(state, BLOCK.tets, dm_inv, rest_vol, params)
    assert np.allclose(state.nodes[pinned], before)


def test_solve_constraints_is_a_noop_at_zero_stiffness():
    params = SolverParams(mu=0.0, lam=0.0)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = state.nodes + 0.1
    before = state.predicted.copy()
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, 1e-3)
    assert np.allclose(state.predicted, before)


def test_attachment_zero_compliance_snaps_to_the_targets():
    """k = 1 maps to zero compliance: one projection lands every node
    exactly on its target, whatever the substep length."""
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes + 0.3
    targets = CUBE.nodes + np.array([1.0, -0.5, 0.25])
    solve_attachment(state, targets, 0.0, 1.0 / 240.0)
    assert np.allclose(state.predicted, targets, atol=1e-12)


def test_zero_attach_stiffness_leaves_the_trajectory_untouched():
    """attach = 0 must be bit-exact free fall even when targets exist."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    params = SolverParams(substeps=5, attach=0.0)
    targets = CUBE.nodes + 1.0

    plain = make_state(CUBE.nodes)
    aimed = make_state(CUBE.nodes)
    for _ in range(3):
        step(plain, CUBE.tets, dm_inv, rest_vol, params)
        step(aimed, CUBE.tets, dm_inv, rest_vol, params, targets)
    assert np.array_equal(plain.nodes, aimed.nodes)
    assert np.array_equal(plain.velocities, aimed.velocities)


def test_attachment_lands_between_node_and_target():
    """A positive compliance moves each node along the line to its target,
    closer than before but not all the way, by one uniform fraction."""
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    targets = CUBE.nodes + 1.0
    solve_attachment(state, targets, attach_compliance(0.5, 1.0 / 24.0), 1.0 / 240.0)
    offset = state.predicted - CUBE.nodes
    assert np.all(offset > 0.0) and np.all(offset < 1.0)
    frac = offset / 1.0
    assert np.allclose(frac, frac.mean(), atol=1e-12)


def test_attachment_ignores_pinned_nodes():
    pinned = np.array([0, 1], dtype=np.int32)
    state = make_state(BLOCK.nodes, pinned=pinned)
    state.predicted[:] = BLOCK.nodes
    solve_attachment(state, BLOCK.nodes + 2.0, 0.0, 1.0 / 240.0)
    assert np.allclose(state.predicted[pinned], BLOCK.nodes[pinned])
    free = np.ones(BLOCK.n_nodes, dtype=bool)
    free[pinned] = False
    assert np.allclose(state.predicted[free], BLOCK.nodes[free] + 2.0)


def test_step_with_full_attachment_tracks_a_moving_target():
    """Full-stiffness attachment over whole frames: the cage rides the
    targets exactly, frame after frame - the animation enters the loop
    rather than being painted on afterwards."""
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=10, attach=1.0, damping=1.0
    )
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    shift = np.array([0.05, 0.0, 0.0])
    for f in range(1, 4):
        targets = CUBE.nodes + f * shift
        step(state, CUBE.tets, dm_inv, rest_vol, params, targets)
        assert np.allclose(state.nodes, targets, atol=1e-9)


def test_soft_attachment_imparts_velocity():
    """The point of feeding animation into the sim: with a soft pull the
    cage lags its targets but picks up real velocity from them, which a
    post-hoc armature deform could never produce."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.5)
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    state = make_state(CUBE.nodes)
    shift = np.array([1.0, 0.0, 0.0])
    targets = CUBE.nodes + shift
    step(state, CUBE.tets, dm_inv, rest_vol, params, targets)
    # Every node advanced towards its own target without passing it on the
    # first frame...
    moved = state.nodes - CUBE.nodes
    assert np.all(moved[:, 0] > 0.05)
    assert np.all(state.nodes[:, 0] < CUBE.nodes[:, 0] + shift[0])
    # ...and now carries physical velocity in the direction of motion.
    assert np.all(state.velocities[:, 0] > 0.0)
