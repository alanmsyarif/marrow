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
         targets=None) -> None:
    """Advance one frame of ``params.substeps`` XPBD substeps, in place.

    ``targets``, when given alongside ``params.attach > 0``, are the
    per-node positions the animation wants this frame; the attachment
    pass pulls nodes towards them after the elastic solve, mirroring the
    GPU pass order.
    """
    h = params.dt / params.substeps
    gravity = np.asarray(params.gravity, dtype=np.float64)
    movable = state.inv_mass > 0.0

    for _ in range(params.substeps):
        # predict
        state.predicted[:] = state.nodes
        state.predicted[movable] += (
            state.velocities[movable] * h + gravity * (h * h)
        )

        solve_constraints(state, tets, dm_inv, rest_vol, params, h)
        if params.attach > 0.0 and targets is not None:
            solve_attachment(
                state, targets,
                attach_compliance(params.attach, params.dt), h,
            )

        # integrate
        state.velocities[movable] = (
            (state.predicted[movable] - state.nodes[movable]) / h * params.damping
        )
        state.nodes[movable] = state.predicted[movable]


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


def solve_attachment(state, targets, compliance: float, h: float) -> None:
    """Pull every free node towards its animation target, in place.

    One position constraint per node, C = x - q, projected once per
    substep. ``compliance`` is the XPBD compliance (see
    attach_compliance); zero is a hard snap, larger values let the flesh
    lag and overshoot. The projection is diagonal - each node moves along
    its own constraint only - so no colouring is needed.
    """
    if compliance < 0.0:
        return
    alpha_tilde = compliance / (h * h)
    movable = state.inv_mass > 0.0
    w = state.inv_mass[movable]
    pull = w / (w + alpha_tilde)
    state.predicted[movable] += (
        (np.asarray(targets, dtype=np.float64)[movable] - state.predicted[movable])
        * pull[:, None]
    )


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


def solve_constraints(state, tets, dm_inv, rest_vol, params, h) -> None:
    """Stable neo-Hookean: one deviatoric and one hydrostatic constraint per tet."""
    if params.mu <= 0.0 and params.lam <= 0.0:
        return

    gamma = 1.0 + (params.mu / params.lam if params.lam > 0.0 else 0.0)
    n_tets = int(tets.shape[0])

    # Multipliers reset every substep, which is standard XPBD.
    lam_dev = np.zeros(n_tets, dtype=np.float64)
    lam_hyd = np.zeros(n_tets, dtype=np.float64)

    for t in range(n_tets):
        idx = tets[t]
        if not np.any(state.inv_mass[idx] > 0.0):
            continue

        p0 = state.predicted[idx[0]]
        ds = np.stack(
            [
                state.predicted[idx[1]] - p0,
                state.predicted[idx[2]] - p0,
                state.predicted[idx[3]] - p0,
            ],
            axis=1,
        )
        f = ds @ dm_inv[t]

        # Deviatoric: resist distortion. The constraint is driven to zero with
        # no rest offset, which on its own would collapse the tet to a point.
        # gamma below is what holds it open: at F = I the two gradients are
        # mu*I and -mu*I, so the rest state is stress-free. Subtracting sqrt(3)
        # here as well would zero this term twice over and leave the volume
        # constraint inflating the body to det(F) = gamma forever.
        if params.mu > 0.0:
            c_dev = float(np.sqrt(np.sum(f * f)))
            if c_dev > 1e-12:
                grads = _grads_from_dcdf(f / c_dev, dm_inv[t])
                lam_dev[t] = _apply(
                    state, idx, grads, c_dev,
                    1.0 / (params.mu * abs(rest_vol[t])), h, lam_dev[t],
                )

        # Hydrostatic: resist volume change.
        if params.lam > 0.0:
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
            f = ds @ dm_inv[t]
            f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
            dcdf = np.stack(
                [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=1
            )
            grads = _grads_from_dcdf(dcdf, dm_inv[t])
            c_hyd = float(np.linalg.det(f) - gamma)
            lam_hyd[t] = _apply(
                state, idx, grads, c_hyd,
                1.0 / (params.lam * abs(rest_vol[t])), h, lam_hyd[t],
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
