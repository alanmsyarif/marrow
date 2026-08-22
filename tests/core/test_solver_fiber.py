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
    fiber = np.zeros((n_tets, 5), dtype=np.float64)
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
    fiber = np.zeros((CUBE.n_tets, 5), dtype=np.float64)
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

    fiber = np.array([[1.0, 1.0, 0.0, 0.75, 0.0]], dtype=np.float64)
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


def _phase_slope(params, t, span=40.0, n=40000):
    """d(wrapped phase)/dx along the body, unwrapped - the anti-shred test.

    Computed from the same pieces fiber_activation uses rather than from its
    output, because the output is wrapped by the fract and its derivative is
    discontinuous at every crest.
    """
    x = np.linspace(0.0, span, n) / params.wave_len
    jitter = params.wave_noise * 0.3 * wobble(
        x * 1.2 + t * (1.1 + 0.6 * params.wave_speed)
    )
    return np.diff(x + jitter) / np.diff(x)


def test_the_phase_stays_monotonic_along_the_body():
    """The jitter must never run the phase backwards. If it did, two
    neighbouring tets would be handed unrelated points in the cycle and the
    body would shred rather than undulate.

    This is a bound, not a taste: wobble' peaks at 0.5 + 0.3*2.3941 +
    0.2*5.1287 = 2.2444, so the 0.3 and 1.2 coefficients cap d(jitter)/dx at
    0.808 against a d(x)/dx of 1. Raising 1.2 to 1.5 breaks it, and this
    test is what says so.
    """
    for noise in (0.2, 0.5, 0.8, 1.0):
        params = SolverParams(
            wave_amp=0.4, wave_len=2.0, wave_speed=1.0, wave_noise=noise
        )
        for t in (0.0, 0.7, 3.1):
            slope = _phase_slope(params, t)
            assert slope.min() > 0.0, (
                f"noise {noise} at t={t}: phase runs backwards "
                f"(min slope {slope.min():.3f})"
            )


def test_noise_is_not_a_rigid_translation():
    """The failure that made 1.7.0's noise useless.

    Any activation that is a function of (x - v t) alone is a rigid
    travelling wave: one fixed profile sliding along the body, the same
    picture every frame, however wobbly that profile is. 1.7.0 drove the
    noise from the wave phase u, which is exactly that - the shape at t=0.5
    matched the shape at t=0 shifted by wave_len * wave_speed * 0.5 to
    within 0.001, and it read on screen as clockwork.
    """
    params = SolverParams(
        wave_amp=0.72, wave_len=2.0, wave_speed=1.0, wave_noise=0.7
    )
    x = np.linspace(0.0, 12.0, 1200)
    a = np.array([fiber_activation(xi, 0.0, params) for xi in x])
    b = np.array([fiber_activation(xi, 0.5, params) for xi in x])

    best = min(
        np.abs(a[s:] - b[: len(b) - s]).max() if s > 0
        else np.abs(a[: len(a) + s] - b[-s:]).max() if s < 0
        else np.abs(a - b).max()
        for s in range(-400, 401)
    )
    assert best > 0.1, (
        f"best residual over every shift is {best:.4f} - the body is still a "
        f"rigid profile sliding along, which is what looks mechanical"
    )


def test_a_noiseless_wave_is_a_rigid_translation():
    """The converse, so the test above is measuring what it claims to. With
    noise off the wave IS rigid, by design."""
    params = SolverParams(wave_amp=0.72, wave_len=2.0, wave_speed=1.0)
    x = np.linspace(0.0, 12.0, 1200)
    a = np.array([fiber_activation(xi, 0.0, params) for xi in x])
    b = np.array([fiber_activation(xi, 0.5, params) for xi in x])
    shift = 100  # wave_len * wave_speed * 0.5 in samples
    assert np.abs(a[: len(a) - shift] - b[shift:]).max() < 0.01


def test_noise_varies_the_beat_at_a_fixed_point():
    """What a viewer actually sees: one place on the body, contracting over
    and over. Without noise every beat is identical."""
    x = 3.0
    ts = np.linspace(0.0, 30.0, 30000)

    def beats(noise):
        params = SolverParams(
            wave_amp=0.5, wave_len=2.0, wave_speed=1.0, wave_noise=noise
        )
        v = np.array([fiber_activation(x, t, params) for t in ts])
        low = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
        return np.diff(ts[low]), v[low]

    gaps, depths = beats(0.0)
    # Neither is exactly zero: crest times land on the 1 ms sampling grid,
    # which is worth ~2e-4 s of interval spread and ~1e-6 of depth. Both are
    # orders below the noisy case asserted underneath.
    assert gaps.std() < 1e-3, "a noiseless wave must beat like a metronome"
    assert depths.max() - depths.min() < 1e-5

    gaps, depths = beats(0.7)
    assert gaps.std() > 0.05, f"beat interval spread only {gaps.std():.4f} s"
    assert depths.max() - depths.min() > 0.15, "every beat still bites the same"


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


def test_the_pulse_stays_single_peaked():
    """The skew warps the cycle before the cosine. Past 1/(2*pi) = 0.159 that
    warp folds and one contraction becomes two, which would read as the wave
    doubling its frequency in patches. The 0.12 coefficient is the margin,
    and this is what holds it."""
    ts = np.linspace(0.0, 20.0, 20000)
    for noise in (0.25, 0.5, 0.75, 1.0):
        params = SolverParams(
            wave_amp=0.6, wave_len=1.5, wave_speed=1.2, wave_noise=noise
        )
        v = np.array([fiber_activation(2.0, t, params) for t in ts])
        low = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
        high = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
        # One release between every pair of contractions, and no more.
        assert abs(len(low) - len(high)) <= 1, (
            f"noise {noise}: {len(low)} contractions against {len(high)} "
            f"releases - the skew has folded a crest in two"
        )


def test_the_pulse_is_symmetric_without_noise():
    """The converse of the skew test: with noise off, contraction and
    release take exactly as long as each other."""
    params = SolverParams(wave_amp=0.9, wave_len=1.5, wave_speed=1.2)
    ts = np.linspace(0.0, 40.0, 40000)
    v = np.array([fiber_activation(2.0, t, params) for t in ts])
    low = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
    high = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    ratios = [
        (ts[c] - ts[high[high < c][-1]]) / (ts[high[high > c][0]] - ts[c])
        for c in low
        if len(high[high < c]) and len(high[high > c])
    ]
    assert max(abs(r - 1.0) for r in ratios) < 0.01


def test_noise_skews_the_pulse():
    """Every crest the same shape is what 'too perfect' meant. Contraction
    and release must stop taking the same time as each other."""
    params = SolverParams(
        wave_amp=0.9, wave_len=1.5, wave_speed=1.2, wave_noise=0.35
    )
    ts = np.linspace(0.0, 40.0, 40000)
    v = np.array([fiber_activation(2.0, t, params) for t in ts])
    low = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
    high = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    ratios = np.array([
        (ts[c] - ts[high[high < c][-1]]) / (ts[high[high > c][0]] - ts[c])
        for c in low
        if len(high[high < c]) and len(high[high > c])
    ])
    assert ratios.std() > 0.05, f"pulse shape spread only {ratios.std():.4f}"
    assert ratios.min() < 0.85 and ratios.max() > 1.15, (
        f"asymmetry only reaches {ratios.min():.3f}..{ratios.max():.3f}"
    )


def test_bend_puts_the_two_flanks_half_a_cycle_apart():
    """The whole point of the side column. At bend 1 one flank contracts
    while the other releases, which is what bends a body; without it every
    tet at a station contracts together and the wave can only squeeze."""
    params = SolverParams(
        wave_amp=0.5, wave_len=1.0, wave_speed=0.0, fiber_bend=1.0
    )
    left = fiber_activation(0.0, 0.0, params, side=1.0)
    right = fiber_activation(0.0, 0.0, params, side=-1.0)
    # side +1 lands at phase +0.25, side -1 at -0.25: half a cycle apart, so
    # the smooth pulse reads the same height on opposite slopes.
    assert abs(left - right) < 1e-12
    centre = fiber_activation(0.0, 0.0, params, side=0.0)
    assert abs(left - centre) > 0.2, "the flanks must differ from the centre"


def test_bend_drives_the_flanks_in_opposition():
    """Sampled across a whole cycle, one flank shortening while the other
    lengthens is the signature. Correlation must be strongly negative."""
    params = SolverParams(
        wave_amp=0.5, wave_len=1.0, wave_speed=1.0, fiber_bend=1.0
    )
    ts = np.linspace(0.0, 4.0, 400)
    left = np.array([fiber_activation(0.0, t, params, side=1.0) for t in ts])
    right = np.array([fiber_activation(0.0, t, params, side=-1.0) for t in ts])
    r = np.corrcoef(left, right)[0, 1]
    assert r < -0.8, f"flanks correlate at {r:+.3f} - they are not opposed"


def test_bend_zero_contracts_a_station_together():
    """Inert by default, and inert exactly: with bend off, side must not
    reach the arithmetic at all."""
    params = SolverParams(wave_amp=0.5, wave_len=1.0, wave_speed=0.7)
    for side in (-1.0, -0.3, 0.0, 0.6, 1.0):
        for t in (0.0, 0.4, 1.9):
            assert fiber_activation(0.0, t, params, side=side) == (
                fiber_activation(0.0, t, params)
            )


def test_bend_scales_between_squeeze_and_undulation():
    params_half = SolverParams(
        wave_amp=0.5, wave_len=1.0, wave_speed=1.0, fiber_bend=0.5
    )
    params_full = SolverParams(
        wave_amp=0.5, wave_len=1.0, wave_speed=1.0, fiber_bend=1.0
    )
    ts = np.linspace(0.0, 4.0, 400)

    def spread(p):
        left = np.array([fiber_activation(0.0, t, p, side=1.0) for t in ts])
        right = np.array([fiber_activation(0.0, t, p, side=-1.0) for t in ts])
        return np.abs(left - right).mean()

    assert spread(params_half) < spread(params_full)
    assert spread(params_half) > 0.0


def _recurrence(noise, speed=1.2):
    """How strongly the whole-body pattern repeats itself later on.

    Autocorrelation of the activation field over time, ignoring lags under
    two seconds so the wave period itself is not what gets measured. 1.0
    means the body strikes the same pose again exactly.
    """
    xs = np.linspace(0.0, 9.0, 90)
    ts = np.linspace(0.0, 40.0, 400)
    params = SolverParams(
        wave_amp=0.5, wave_len=1.5, wave_speed=speed, wave_noise=noise
    )
    field = np.array([[fiber_activation(x, t, params) for t in ts] for x in xs])
    r = field - field.mean()
    ac = np.array([np.sum(r[:, : len(ts) - k] * r[:, k:]) for k in range(len(ts) // 2)])
    ac /= ac[0]
    lags = np.arange(len(ac)) * (ts[1] - ts[0])
    keep = lags > 2.0
    return float(ac[keep].max())


def test_noise_breaks_the_repetition():
    """A travelling wave at a fixed Speed strikes the same pose every
    1/Speed seconds however each crest is dressed, because the rhythm
    underneath never changes. Drifting the stroke rate is what breaks that,
    and this is the measurement that says whether it worked."""
    quiet = _recurrence(0.0)
    loud = _recurrence(0.7)
    assert quiet > 0.85, f"a noiseless wave has to repeat, got {quiet:.3f}"
    assert loud < 0.55, f"the pattern still repeats at {loud:.3f}"


def test_more_noise_repeats_less():
    a, b, c = _recurrence(0.35), _recurrence(0.7), _recurrence(1.0)
    assert a > b > c, f"not monotonic: {a:.3f}, {b:.3f}, {c:.3f}"


def test_the_stroke_rate_never_runs_backwards():
    """The drift is scaled by wave_speed so it stays a fraction of the rate.
    If it could exceed it, the wave would stop and reverse mid-stroke, which
    is not a rhythm change but a glitch. The 0.4 coefficient is picked
    against exactly this bound and this test is what holds it."""
    t = np.linspace(0.0, 200.0, 200000)
    for noise in (0.35, 0.7, 1.0):
        for speed in (0.4, 1.2, 3.0):
            phase = t * speed + noise * 0.4 * abs(speed) * wobble(t * 0.5)
            rate = np.diff(phase) / np.diff(t)
            assert rate.min() > 0.0, (
                f"noise {noise} at speed {speed}: the wave reverses "
                f"(slowest rate {rate.min():+.3f})"
            )


def test_a_standing_wave_has_no_rate_to_drift():
    """The drift is scaled by wave_speed, so at Speed 0 it vanishes rather
    than shaking a wave that is not travelling anywhere. The other noise
    terms keep working - only the rhythm term goes quiet, because there is
    no rhythm."""
    params = SolverParams(
        wave_amp=0.5, wave_len=1.5, wave_speed=0.0, wave_noise=1.0
    )
    ts = np.linspace(0.0, 20.0, 400)
    values = np.array([fiber_activation(2.0, t, params) for t in ts])
    # Still alive: the position-and-time jitter does not depend on speed.
    assert values.std() > 1e-3, "a standing wave should still breathe"
    # But the phase never sweeps, so it never completes a cycle.
    lows = np.nonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0]
    assert len(lows) < len(ts) // 8, (
        f"{len(lows)} contractions from a wave with no speed - the rate "
        f"drift is sweeping the phase on its own"
    )
