import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    fiber_activation,
    make_state,
    precompute,
    solve_constraints,
    step,
)

CUBE = build_lattice(np.zeros(3), 1.0, np.ones((1, 1, 1), dtype=bool))


def _fibers_along_x(n_tets, phase=0.75):
    """Every tet pulls along +X, all at the same phase so they fire together."""
    fiber = np.zeros((n_tets, 4), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _extent(nodes, axis):
    return float(nodes[:, axis].max() - nodes[:, axis].min())


def test_activation_is_one_when_amplitude_is_zero():
    params = SolverParams(wave_amp=0.0)
    assert fiber_activation(0.5, 0.0, params) == 1.0


def test_smooth_waveform_peaks_at_half_a_cycle():
    params = SolverParams(wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=0)
    assert abs(fiber_activation(0.0, 0.0, params) - 1.0) < 1e-12
    assert abs(fiber_activation(0.5, 0.0, params) - 0.6) < 1e-12


def test_square_waveform_is_on_or_off():
    params = SolverParams(wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=1)
    assert fiber_activation(0.25, 0.0, params) == 1.0
    assert abs(fiber_activation(0.75, 0.0, params) - 0.6) < 1e-12


def test_phase_wraps_the_same_way_for_negative_time():
    """wave_time * wave_speed drives the phase negative almost immediately,
    and GLSL fract and numpy % must agree there or the GPU diverges."""
    params = SolverParams(wave_amp=1.0, wave_len=1.0, wave_speed=1.0, waveform=0)
    assert abs(fiber_activation(0.0, 0.25, params) - fiber_activation(0.75, 0.0, params)) < 1e-12


def test_the_wave_reaches_two_tets_at_different_times():
    early = SolverParams(wave_amp=1.0, wave_len=1.0, wave_speed=1.0, waveform=1)
    at_t0 = (fiber_activation(0.1, 0.0, early), fiber_activation(0.6, 0.0, early))
    assert at_t0[0] != at_t0[1], "tets at different arclength must not fire together"


def test_contraction_shortens_along_the_fiber_and_bulges_across():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0),
        fiber_k=1.0e4,
        wave_amp=0.4,
        wave_len=1.0,
        wave_speed=0.0,
        waveform=1,          # square, phase 0.75 -> fully on, held
    )
    state = make_state(CUBE.nodes)
    x0, y0 = _extent(CUBE.nodes, 0), _extent(CUBE.nodes, 1)
    for _ in range(20):
        step(state, CUBE.tets, dm_inv, rest_vol, params, fiber=fiber)

    assert _extent(state.nodes, 0) < x0 - 1e-3, "fiber must shorten along +X"
    assert _extent(state.nodes, 1) > y0 + 1e-4, "volume must go sideways"


def test_zero_fiber_stiffness_reproduces_the_current_solve():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(gravity=(0.0, 0.0, 0.0), fiber_k=0.0, wave_amp=0.9)
    h = params.dt / params.substeps

    with_fiber = make_state(CUBE.nodes)
    without = make_state(CUBE.nodes)
    for st in (with_fiber, without):
        st.predicted[:] = CUBE.nodes * 1.2

    solve_constraints(with_fiber, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    solve_constraints(without, CUBE.tets, dm_inv, rest_vol, params, h)
    assert np.array_equal(with_fiber.predicted, without.predicted)


def test_a_zero_direction_row_is_skipped():
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = np.zeros((CUBE.n_tets, 4), dtype=np.float64)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.9, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    assert np.array_equal(state.predicted, CUBE.nodes)


def test_fiber_alone_still_solves_with_both_stiffnesses_off():
    """The early bail in solve_constraints must know about fiber_k, or the
    one test that isolates this feature passes for the wrong reason."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.5, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    solve_constraints(state, CUBE.tets, dm_inv, rest_vol, params, h, fiber=fiber)
    assert not np.array_equal(state.predicted, CUBE.nodes)
