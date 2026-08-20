import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    fiber_activation,
    make_state,
    precompute,
    solve_constraints,
    step,
    wobble,
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


def test_step_threads_the_clock_across_frames():
    """step() must return the advanced clock so a caller can pass it back
    in as the next frame's t0 - otherwise the wave would restart at phase
    0 every frame and a travelling wave would jitter in place instead of
    travelling. wave_speed=3.0 moves the phase by 0.125 of a cycle over one
    default frame (dt=1/24), which is not a whole cycle, so the smooth
    waveform must read differently at the threaded clock than it would
    at a clock that had been reset to 0."""
    dm_inv, rest_vol = precompute(CUBE.nodes, CUBE.tets)
    fiber = _fibers_along_x(CUBE.n_tets, phase=0.75)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.5, wave_len=1.0, wave_speed=3.0, waveform=0,
    )
    state = make_state(CUBE.nodes)

    t1 = step(state, CUBE.tets, dm_inv, rest_vol, params, fiber=fiber)
    assert t1 > 0.0, "the clock must advance over a frame"

    # Threading t1 into the next frame continues the wave; a caller that
    # never threads it would see frame two start from phase 0 again, i.e.
    # exactly what frame one saw.
    threaded_activation = fiber_activation(0.75, t1, params)
    replayed_activation = fiber_activation(0.75, 0.0, params)
    assert abs(threaded_activation - replayed_activation) > 1e-6, (
        "a threaded second frame must see a different activation than a "
        "frame whose clock was reset to 0"
    )

    # The plumbing itself: step() must accept t0 and keep advancing from it.
    t2 = step(state, CUBE.tets, dm_inv, rest_vol, params, fiber=fiber, t0=t1)
    assert t2 > t1


def test_fiber_gradient_is_not_symmetric_under_a_diagonal_direction():
    """Every other fixture in this file points along +X, where
    outer(Fa/|Fa|, a) collapses to a diagonal matrix and a transposed dC/dF
    in the coming GLSL kernel would pass by coincidence. A single tet,
    stretched anisotropically and pulled along the (1,1,0) diagonal, breaks
    that: F a is no longer parallel to a, so the correct gradient pulls
    node 1 unevenly in x and y (tracking F's 1.5:1 stretch) while a
    transposed gradient would pull it evenly, since a's own x and y
    components are equal."""
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dm_inv, rest_vol = precompute(nodes, tets)

    fiber = np.array([[1.0, 1.0, 0.0, 0.75]], dtype=np.float64)
    fiber[:, :3] /= np.linalg.norm(fiber[:, :3])

    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.0,  # s == 1.0: an undisguised gradient probe
    )
    h = params.dt / params.substeps
    state = make_state(nodes)
    # Anisotropic pre-stretch: x by 1.5, y and z untouched, so F = diag(1.5,
    # 1, 1) and F a is skewed towards x - no longer parallel to a.
    stretched = nodes * np.array([1.5, 1.0, 1.0])
    state.predicted[:] = stretched

    solve_constraints(state, tets, dm_inv, rest_vol, params, h, fiber=fiber)

    disp = state.predicted[1] - stretched[1]
    assert abs(disp[0]) > 1e-9 and abs(disp[1]) > 1e-9, "fiber must move node 1"
    assert abs(disp[0]) > abs(disp[1]) * 1.1, (
        "node 1's pull must be stronger in x than y, tracking F's stretch; "
        "a transposed gradient would pull x and y equally since a's own "
        "components are equal"
    )


def test_noise_zero_is_the_exact_clockwork_wave():
    """Not close - equal. The noise block is skipped, not multiplied by
    zero, so every scene that predates the slider keeps its motion."""
    plain = SolverParams(wave_amp=0.4, wave_len=0.8, wave_speed=1.3, waveform=0)
    noisy = SolverParams(
        wave_amp=0.4, wave_len=0.8, wave_speed=1.3, waveform=0, wave_noise=0.0
    )
    for phase in (0.0, 0.31, 1.7, 4.2):
        for t in (0.0, 0.4, 2.6):
            assert fiber_activation(phase, t, plain) == fiber_activation(
                phase, t, noisy
            )


def test_noise_changes_the_activation():
    quiet = SolverParams(wave_amp=0.4, wave_len=0.8, wave_speed=1.3)
    loud = SolverParams(wave_amp=0.4, wave_len=0.8, wave_speed=1.3, wave_noise=0.8)
    differing = sum(
        1
        for phase in np.linspace(0.0, 4.0, 40)
        if abs(fiber_activation(phase, 0.5, quiet) - fiber_activation(phase, 0.5, loud))
        > 1e-6
    )
    assert differing > 30, f"only {differing}/40 samples moved"


def test_noise_never_drives_the_target_to_zero():
    """s <= 0 is a constraint no length can satisfy: the tet would be pulled
    inward forever. The clamp on amp is what prevents it, and wave_amp at
    its UI maximum with noise at its own maximum is the worst case."""
    params = SolverParams(wave_amp=0.9, wave_len=0.5, wave_speed=2.0, wave_noise=1.0)
    lowest = min(
        fiber_activation(phase, t, params)
        for phase in np.linspace(0.0, 10.0, 400)
        for t in np.linspace(0.0, 5.0, 40)
    )
    assert lowest >= 0.05 - 1e-12, lowest


def test_noise_stays_within_rest_length():
    """Activation must never exceed 1: the wave shortens material, it does
    not stretch it."""
    params = SolverParams(wave_amp=0.5, wave_len=0.7, wave_speed=1.1, wave_noise=1.0)
    highest = max(
        fiber_activation(phase, t, params)
        for phase in np.linspace(0.0, 10.0, 400)
        for t in np.linspace(0.0, 5.0, 40)
    )
    assert highest <= 1.0 + 1e-12, highest


def test_noise_varies_more_slowly_than_the_wave():
    """The jitter has to be smoother along the body than the wave it
    perturbs. If it were not, neighbouring tets would get unrelated phases
    and the body would shred instead of undulating."""
    params = SolverParams(wave_amp=0.4, wave_len=1.0, wave_speed=0.0, wave_noise=1.0)
    phases = np.linspace(0.0, 6.0, 600)
    values = np.array([fiber_activation(p, 0.0, params) for p in phases])
    # Sign changes in the derivative count the crests. A wavelength of 1.0
    # over 6 units is 6 crests; noise must not multiply that.
    slope = np.diff(values)
    turns = int(np.sum(slope[1:] * slope[:-1] < 0))
    assert turns <= 14, f"{turns} turning points - the jitter is too fast"


def test_wobble_is_bounded():
    u = np.linspace(-50.0, 50.0, 20001)
    w = wobble(u)
    assert w.max() <= 1.0 and w.min() >= -1.0, (w.min(), w.max())


def test_wobble_does_not_repeat_over_the_useful_range():
    """A visibly periodic 'noise' is just a second wave. Compare the field
    against itself shifted by each of its own component periods."""
    u = np.linspace(0.0, 40.0, 4000)
    base = wobble(u)
    for shift in (2 * np.pi, 2 * np.pi / 2.3941, 2 * np.pi / 5.1287):
        assert np.abs(wobble(u + shift) - base).max() > 0.3, shift
