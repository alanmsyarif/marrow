"""numpy XPBD reference solver.

This exists to be *correct and readable*, not fast. It is the oracle the GPU
compute kernels are diffed against, because a sign error in a compute shader
is otherwise indistinguishable from a sign error in the constraint algebra.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverParams:
    dt: float = 1.0 / 24.0
    substeps: int = 10
    gravity: tuple = (0.0, 0.0, -9.81)
    mu: float = 1.0e4       # deviatoric stiffness
    lam: float = 1.0e5      # hydrostatic (volume) stiffness
    damping: float = 0.999
    attach: float = 0.0     # attachment stiffness, 0..1; 0 disables the pass
    # Pinned nodes ride the attachment targets rather than staying put.
    # Needs the attachment pass, which is where targets arrive.
    pin_kinematic: bool = False

    # --- fiber ---
    # Anisotropic muscle term. Inert at these defaults, so every body that
    # does not ask for fibers solves exactly as it did before.
    fiber_k: float = 0.0    # fiber stiffness; 0 disables the term
    wave_amp: float = 0.0   # peak contraction, 0.3 shortens to 70%
    wave_len: float = 1.0   # wave period in arclength units
    wave_speed: float = 0.0  # cycles per second; negative reverses travel
    waveform: int = 0       # 0 smooth cosine, 1 square
    # Irregularity, 0..1. Jitters both when a crest arrives and how hard it
    # bites. Zero is not merely the default but the bit-identical path: the
    # noise arithmetic is skipped entirely, not multiplied by zero.
    wave_noise: float = 0.0


@dataclass
class SolverState:
    nodes: np.ndarray        # (N, 3) float64
    velocities: np.ndarray   # (N, 3) float64
    inv_mass: np.ndarray     # (N,)   float64, 0 means pinned
    predicted: np.ndarray = field(default=None, repr=False)


def make_state(nodes: np.ndarray, density: float = 1.0, pinned=None) -> SolverState:
    nodes = np.array(nodes, dtype=np.float64, copy=True)
    inv_mass = np.full(nodes.shape[0], 1.0 / density, dtype=np.float64)
    if pinned is not None and len(pinned):
        inv_mass[np.asarray(pinned, dtype=np.int64)] = 0.0
    return SolverState(
        nodes=nodes,
        velocities=np.zeros_like(nodes),
        inv_mass=inv_mass,
        predicted=np.zeros_like(nodes),
    )


def precompute(nodes: np.ndarray, tets: np.ndarray):
    """Rest shape inverse and rest volume per tet."""
    p0 = nodes[tets[:, 0]]
    dm = np.stack(
        [nodes[tets[:, 1]] - p0, nodes[tets[:, 2]] - p0, nodes[tets[:, 3]] - p0],
        axis=2,
    )  # (T, 3, 3)
    dm_inv = np.linalg.inv(dm)
    rest_vol = np.linalg.det(dm) / 6.0
    return dm_inv, rest_vol


def step(state: SolverState, tets, dm_inv, rest_vol, params: SolverParams,
         targets=None, fiber=None, t0: float = 0.0, region=None) -> float:
    """Advance one frame of ``params.substeps`` XPBD substeps, in place.

    ``targets``, when given alongside ``params.attach > 0``, are the
    per-node positions the animation wants this frame; the attachment
    pass pulls nodes towards them after the elastic solve, mirroring the
    GPU pass order.

    Returns the advanced clock, so a caller stepping frame after frame can
    thread it back in as the next frame's ``t0``.
    """
    h = params.dt / params.substeps
    gravity = np.asarray(params.gravity, dtype=np.float64)
    movable = state.inv_mass > 0.0

    # Simulation clock for the fiber wave. Advances per substep, not per
    # frame: a per-frame clock steps the wave in visible stairs at low
    # substep counts. Starts from ``t0`` rather than always 0, because the
    # wave must keep travelling across frame boundaries - GPUSolver.sim_time
    # is the GPU's persistent counterpart, reset only on a live restart, and
    # a step() that always restarted at 0 would snap the wave back every
    # frame instead of letting it travel.
    t = t0

    for _ in range(params.substeps):
        # predict
        state.predicted[:] = state.nodes
        state.predicted[movable] += (
            state.velocities[movable] * h + gravity * (h * h)
        )

        solve_constraints(
            state, tets, dm_inv, rest_vol, params, h, fiber=fiber, t=t,
            region=region,
        )
        # Stiffness 0 still runs the pass when there are driven pins: they
        # need targets, and the free material needs to be left alone. See
        # solve_attachment's drive_free.
        drive_free = params.attach > 0.0
        if targets is not None and (drive_free or params.pin_kinematic):
            solve_attachment(
                state, targets,
                attach_compliance(params.attach, params.dt) if drive_free else 0.0,
                h,
                kinematic=params.pin_kinematic,
                drive_free=drive_free,
            )

        # integrate
        state.velocities[movable] = (
            (state.predicted[movable] - state.nodes[movable]) / h * params.damping
        )
        state.nodes[movable] = state.predicted[movable]
        # A kinematic pin's position advances too, or the target the
        # attachment pass stored would be discarded every substep. Its
        # velocity stays zero: it is driven, not simulated, and predict
        # reads no velocity for a pinned node. Guarded rather than
        # unconditional, mirroring the kernel - for a static pin this would
        # be a no-op, but the guard is what keeps "a pin does not move" a
        # property of the integrator itself and not merely of every pass
        # that runs before it.
        if params.pin_kinematic:
            state.nodes[~movable] = state.predicted[~movable]

        t += h

    return t


def wobble(u):
    """Smooth aperiodic field in roughly [-1, 1]. Mirrors the GLSL wobble.

    Three sines whose frequencies share no small common multiple, so the sum
    has no visible period. A hash would be the obvious choice for noise and
    is the wrong one here - see the comment on the GLSL twin.
    """
    return (
        0.5 * np.sin(u)
        + 0.3 * np.sin(u * 2.3941 + 1.7)
        + 0.2 * np.sin(u * 5.1287 + 4.3)
    )


def fiber_activation(phase: float, t: float, params: SolverParams) -> float:
    """Target stretch along the fiber for one tet at time ``t``.

    1.0 is rest. Below 1.0 the tet is being told to shorten. The phase
    argument is the tet's baked arclength, so two tets at different points
    along the body reach their peak at different times - which is the whole
    difference between a travelling wave and a body that pulses in unison.

    ``wave_noise`` above zero jitters both the arrival of each crest and its
    strength, which is what stops a pure sinusoid reading as clockwork.

    Mirrored in kernels.SOLVE_SRC. GLSL fract and numpy % agree on negative
    inputs, which matters because t * wave_speed drives this negative within
    the first second.
    """
    # The wave's own phase, before wrapping. Noise rides this coordinate so
    # it rescales with wave_len and wave_speed rather than needing a re-tune.
    u = phase / params.wave_len - t * params.wave_speed

    jitter = 0.0
    amp = params.wave_amp
    if params.wave_noise > 0.0:
        jitter = params.wave_noise * 0.25 * wobble(u * 0.6)
        gain = 1.0 + params.wave_noise * 0.5 * wobble(u * 0.37 + 11.0)
        # See the GLSL twin: amp past 1 is a constraint no length satisfies.
        amp = float(np.clip(params.wave_amp * gain, 0.0, 0.95))

    cycle = (u + jitter) % 1.0
    if params.waveform == 0:
        pulse = 0.5 * (1.0 - np.cos(2.0 * np.pi * cycle))
    else:
        pulse = 1.0 if cycle >= 0.5 else 0.0
    return 1.0 - amp * pulse


def attach_compliance(stiffness: float, dt: float) -> float:
    """XPBD compliance for an attachment stiffness in (0, 1].

    The raw map (1 - k) / k is rescaled by dt squared. Compliance enters
    the projection as alpha / h^2 with h = dt / substeps, so unscaled it
    would read tens of thousands at typical substep sizes and the slider
    would be dead everywhere but its last percent. Scaling by dt^2 makes
    the response a property of the frame, not of the substep count: at
    k = 0.5 roughly a third of the gap closes per frame, k near 1 rides
    the bones, and k = 1 is exactly zero compliance - a hard snap.
    """
    if stiffness <= 0.0:
        raise ValueError("attachment stiffness must be positive")
    return (1.0 - stiffness) / stiffness * float(dt) * float(dt)


def solve_attachment(state, targets, compliance: float, h: float,
                     kinematic: bool = False, drive_free: bool = True) -> None:
    """Pull every free node towards its animation target, in place.

    One position constraint per node, C = x - q, projected once per
    substep. ``compliance`` is the XPBD compliance (see
    attach_compliance); zero is a hard snap, larger values let the flesh
    lag and overshoot. The projection is diagonal - each node moves along
    its own constraint only - so no colouring is needed.

    ``kinematic`` drives the pinned nodes too. A pin normally outranks the
    animation and is skipped here; under a kinematic pin it takes its
    target outright and carries the surrounding material along. That is a
    store rather than a projection, because compliance has no meaning
    without an inverse mass to weigh the correction against.

    ``drive_free`` off leaves every unpinned node alone - targets for the
    pins, hands off everything else. This is what makes a driven pin able
    to carry a body: aiming free material at its evaluated position aims
    it at the REST pose wherever the animation does not reach, so the same
    pass that feeds the pin its target is otherwise nailing the body down.
    Measured on a 23,697-node cage, a pin travelling 1.473 dragged the body
    0.158 at stiffness 0.05 and 1.515 with the grip released.
    """
    if compliance < 0.0:
        return
    targets = np.asarray(targets, dtype=np.float64)
    movable = state.inv_mass > 0.0
    if drive_free:
        alpha_tilde = compliance / (h * h)
        w = state.inv_mass[movable]
        pull = w / (w + alpha_tilde)
        state.predicted[movable] += (
            (targets[movable] - state.predicted[movable]) * pull[:, None]
        )
    if kinematic:
        state.predicted[~movable] = targets[~movable]


def blend_project(state, rows, weights) -> None:
    """Project the hanging-node interpolation constraints, in place.

    ``rows`` is (R, 5) of [hanging, m0..m3] and ``weights`` the (R, 4) master
    weights, summing to one; zero-weight slots are padding. One Gauss-Seidel
    sweep in row order over C = x_h - sum w_i x_m with compliance zero - the
    CPU mirror of the GPU blend pass, so a sign error there is detectable.
    """
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    p = state.predicted
    for r in range(rows.shape[0]):
        h = int(rows[r, 0])
        masters = rows[r, 1:]
        w = weights[r]

        xm = np.zeros(3, dtype=np.float64)
        denom = float(state.inv_mass[h])
        for i in range(4):
            if w[i] > 0.0:
                xm += w[i] * p[masters[i]]
                denom += float(state.inv_mass[masters[i]]) * w[i] * w[i]
        if denom < 1e-20:
            continue
        dlambda = -(p[h] - xm) / denom
        if state.inv_mass[h] > 0.0:
            p[h] += state.inv_mass[h] * dlambda
        for i in range(4):
            if w[i] > 0.0 and state.inv_mass[masters[i]] > 0.0:
                p[masters[i]] -= state.inv_mass[masters[i]] * (w[i] * dlambda)


def _grads_from_dcdf(dcdf: np.ndarray, dm_inv: np.ndarray) -> np.ndarray:
    """Map a gradient in F to per-node gradients. Returns (4, 3)."""
    g123 = dcdf @ dm_inv.T          # (3, 3), columns are nodes 1, 2, 3
    g1, g2, g3 = g123[:, 0], g123[:, 1], g123[:, 2]
    g0 = -(g1 + g2 + g3)
    return np.stack([g0, g1, g2, g3], axis=0)


def _apply(state, nodes_idx, grads, c_value, compliance, h, lam_acc):
    """One XPBD constraint projection. Returns the updated multiplier."""
    w = state.inv_mass[nodes_idx]
    denom = float(np.sum(w * np.einsum("ij,ij->i", grads, grads)))
    alpha_tilde = compliance / (h * h)
    denom += alpha_tilde
    if denom < 1e-20:
        return lam_acc
    dlambda = (-c_value - alpha_tilde * lam_acc) / denom
    state.predicted[nodes_idx] += grads * (w[:, None] * dlambda)
    return lam_acc + dlambda


def solve_constraints(state, tets, dm_inv, rest_vol, params, h,
                      fiber=None, t=0.0, region=None) -> None:
    """Stable neo-Hookean, plus an optional anisotropic fiber term.

    ``region``, when given, is one stiffness multiplier per tet: mu and lam
    are both scaled by it, so a body can be stiff in one place and soft in
    another without a second material. None means a multiplier of 1
    everywhere, and is not merely the default but the bit-identical path -
    nothing is multiplied at all.
    """
    fiber_on = params.fiber_k > 0.0 and fiber is not None
    if params.mu <= 0.0 and params.lam <= 0.0 and not fiber_on:
        return

    gamma = 1.0 + (params.mu / params.lam if params.lam > 0.0 else 0.0)
    n_tets = int(tets.shape[0])

    # Multipliers reset every substep, which is standard XPBD.
    lam_dev = np.zeros(n_tets, dtype=np.float64)
    lam_hyd = np.zeros(n_tets, dtype=np.float64)

    for t_i in range(n_tets):
        idx = tets[t_i]
        if not np.any(state.inv_mass[idx] > 0.0):
            continue

        # gamma above deliberately stays on the unscaled ratio: the
        # multiplier cancels out of mu/lam, and it is the ratio the
        # rest-state correction depends on. Mirrors SOLVE_SRC.
        if region is None:
            mu_t, lam_t = params.mu, params.lam
        else:
            rw = float(region[t_i])
            mu_t, lam_t = params.mu * rw, params.lam * rw

        p0 = state.predicted[idx[0]]
        ds = np.stack(
            [
                state.predicted[idx[1]] - p0,
                state.predicted[idx[2]] - p0,
                state.predicted[idx[3]] - p0,
            ],
            axis=1,
        )
        f = ds @ dm_inv[t_i]

        # Deviatoric: resist distortion. The constraint is driven to zero with
        # no rest offset, which on its own would collapse the tet to a point.
        # gamma below is what holds it open: at F = I the two gradients are
        # mu*I and -mu*I, so the rest state is stress-free. Subtracting sqrt(3)
        # here as well would zero this term twice over and leave the volume
        # constraint inflating the body to det(F) = gamma forever.
        if mu_t > 0.0:
            c_dev = float(np.sqrt(np.sum(f * f)))
            if c_dev > 1e-12:
                grads = _grads_from_dcdf(f / c_dev, dm_inv[t_i])
                lam_dev[t_i] = _apply(
                    state, idx, grads, c_dev,
                    1.0 / (mu_t * abs(rest_vol[t_i])), h, lam_dev[t_i],
                )

        # Fiber: resist stretch along a, and drive it below rest length when
        # the wave says so. Sits between the two isotropic terms because the
        # hydrostatic pass below rebuilds F from the positions this moved,
        # and that is what turns shortening into a sideways bulge.
        if fiber_on:
            a = fiber[t_i, :3]
            if float(a @ a) > 0.5:
                # F is recomputed rather than reused, for the same reason the
                # hydrostatic block below recomputes it: the deviatoric
                # projection above has already moved state.predicted, so the
                # stale F would linearise this constraint about the wrong
                # configuration. Deliberate, not an oversight.
                p0 = state.predicted[idx[0]]
                ds = np.stack(
                    [
                        state.predicted[idx[1]] - p0,
                        state.predicted[idx[2]] - p0,
                        state.predicted[idx[3]] - p0,
                    ],
                    axis=1,
                )
                f = ds @ dm_inv[t_i]
                s = fiber_activation(float(fiber[t_i, 3]), t, params)
                fa = f @ a
                fiber_len = float(np.linalg.norm(fa))
                if fiber_len > 1e-12:
                    grads = _grads_from_dcdf(np.outer(fa / fiber_len, a), dm_inv[t_i])
                    _apply(
                        state, idx, grads, fiber_len - s,
                        1.0 / (params.fiber_k * abs(rest_vol[t_i])), h, 0.0,
                    )

        # Hydrostatic: resist volume change.
        if lam_t > 0.0:
            # F is recomputed rather than reused: the deviatoric projection
            # above has already moved state.predicted, so the stale F would
            # linearise the volume constraint about the wrong configuration.
            # This duplication is load-bearing, not an oversight.
            p0 = state.predicted[idx[0]]
            ds = np.stack(
                [
                    state.predicted[idx[1]] - p0,
                    state.predicted[idx[2]] - p0,
                    state.predicted[idx[3]] - p0,
                ],
                axis=1,
            )
            f = ds @ dm_inv[t_i]
            f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
            dcdf = np.stack(
                [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=1
            )
            grads = _grads_from_dcdf(dcdf, dm_inv[t_i])
            c_hyd = float(np.linalg.det(f) - gamma)
            lam_hyd[t_i] = _apply(
                state, idx, grads, c_hyd,
                1.0 / (lam_t * abs(rest_vol[t_i])), h, lam_hyd[t_i],
            )


def solve_plane_contact(state, ground_z: float, friction: float) -> None:
    """Depenetrate a ground plane, with Coulomb friction, in place.

    The oracle for the friction in the GLSL contact kernels. All three of
    them compute the same thing from the same two quantities - the
    correction the contact itself applied, which gives both the normal and
    the penetration depth, and how far the node has moved since the substep
    began. Only where those come from differs, so this plane case pins the
    algebra for all of them.

    Friction is a position correction, not a velocity one: the tangential
    motion of this substep is given back, up to ``friction`` times the
    penetration depth. Under that clamp the whole tangential step is
    cancelled and the node holds, which is static friction; over it the
    contact slips at a rate the coefficient sets. One coefficient covers
    both, so there is no second slider that has to stay consistent with the
    first. ``friction`` of 0 leaves the trajectory bit identical to plain
    non-penetration.
    """
    p = state.predicted
    hit = (state.inv_mass > 0.0) & (p[:, 2] < ground_z)
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return

    depth = ground_z - p[idx, 2]
    p[idx, 2] = ground_z
    if friction <= 0.0:
        return

    # The normal is +z, so the tangent plane is xy and dropping the z
    # component is the whole projection.
    slide = p[idx] - state.nodes[idx]
    slide[:, 2] = 0.0
    mag = np.linalg.norm(slide, axis=1)

    # A node that only fell has no tangential motion to resist, and dividing
    # by that zero would poison it with NaN.
    live = mag > 1e-9
    if not np.any(live):
        return
    moving = idx[live]
    scale = np.minimum(1.0, friction * depth[live] / mag[live])
    p[moving] -= slide[live] * scale[:, None]
