"""Collision between two Marrow bodies, with both of them deforming.

See docs/superpowers/specs/2026-08-13-marrow-body-to-body-collision-design.md.

Like the self-collision tests these run with mu = lam = 0, so the elastic
solve contributes nothing and the only thing that can move a node is contact.
Unlike them there is no rest-distance gate to work around: two bodies share
no rest configuration, so a pair inside the thickness is always a contact.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.solver_ref import SolverParams, make_state
from marrow.core.tetmesh import TetMesh, surface_nodes
from marrow.gpu.solver import GPUSolver

THICK = 0.2

_UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _tet_at(x):
    """A single unit tet whose node 0 sits at (x, 0, 0)."""
    nodes = _UNIT + np.array([x, 0.0, 0.0])
    return TetMesh(nodes, np.array([[0, 1, 2, 3]], dtype=np.int32))


def _inert(mesh, inv_mass=None, distance=THICK, substeps=1):
    """A solver whose only active constraint is body collision."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps, mu=0.0, lam=0.0)
    if inv_mass is None:
        inv_mass = make_state(mesh.nodes).inv_mass
    return GPUSolver(mesh, inv_mass, params, body_distance=distance)


def _pair(gap, inv_a=None, inv_b=None, distance=THICK):
    """Two tets whose nearest nodes - a's node 1 and b's node 0 - are ``gap``
    apart. Every other cross pair is well outside the thickness."""
    a = _tet_at(0.0)
    b = _tet_at(1.0 + gap)
    return (
        _inert(a, inv_mass=inv_a, distance=distance),
        _inert(b, inv_mass=inv_b, distance=distance),
    )


def _substep(sa, sb, n=1):
    """Advance both bodies the way the group driver does.

    Constraints for both, then integration for both. Integrating sa before
    sb solved would let sb see sa's updated tex_x and take a smaller share of
    every contact - measured at a persistent two-to-one split.
    """
    h = sa.params.dt / sa.params.substeps
    for _ in range(n):
        sa.substep_constraints(h, [sb])
        sb.substep_constraints(h, [sa])
        sa.substep_integrate(h)
        sb.substep_integrate(h)


def _sep(sa, sb):
    return float(np.linalg.norm(sa.positions()[1] - sb.positions()[0]))


def test_both_bodies_deform():
    """The point of the feature. One body pushing and the other ignoring it
    would pass every separation test below, so check the movement itself."""
    a, b = _pair(0.5 * THICK)
    before_a, before_b = a.positions().copy(), b.positions().copy()
    _substep(a, b)

    moved_a = float(np.abs(a.positions() - before_a).max())
    moved_b = float(np.abs(b.positions() - before_b).max())
    assert moved_a > 1e-6, "body A did not move"
    assert moved_b > 1e-6, "body B did not move"


def test_equal_bodies_take_equal_and_opposite_corrections():
    a, b = _pair(0.5 * THICK)
    before_a, before_b = a.positions()[1].copy(), b.positions()[0].copy()
    _substep(a, b)

    da = a.positions()[1] - before_a
    db = b.positions()[0] - before_b
    assert np.allclose(da, -db, atol=1e-6), f"asymmetric: {da} vs {db}"
    assert _sep(a, b) > 0.99 * THICK, f"pair still overlapping: {_sep(a, b):.4f}"


def test_a_pinned_body_pushes_without_moving():
    a, b = _pair(0.5 * THICK, inv_b=np.zeros(4))
    before_b = b.positions().copy()
    _substep(a, b)

    assert np.allclose(b.positions(), before_b, atol=1e-6), "the pinned body moved"
    assert _sep(a, b) > 0.99 * THICK, (
        f"the free body did not take the whole correction: {_sep(a, b):.4f}"
    )


def test_bodies_outside_the_thickness_are_left_alone():
    a, b = _pair(3.0 * THICK)
    before_a, before_b = a.positions().copy(), b.positions().copy()
    _substep(a, b, n=4)

    assert np.allclose(a.positions(), before_a, atol=1e-9)
    assert np.allclose(b.positions(), before_b, atol=1e-9)


def test_interior_and_pinned_nodes_survive_the_ping_pong():
    """The cross kernel writes a second image and the two are swapped, so a
    thread that returns without storing leaves a stale texel behind."""
    inv_mass = make_state(BLOCK.nodes).inv_mass
    inv_mass[0] = 0.0
    interior = sorted(
        set(range(BLOCK.n_nodes)) - set(surface_nodes(BLOCK.tets).tolist())
    )
    assert interior, "BLOCK must have an interior node for this to mean anything"

    near = TetMesh(BLOCK.nodes + np.array([5.0, 0.0, 0.0]), BLOCK.tets)
    a = _inert(BLOCK, inv_mass=inv_mass, distance=0.1)
    b = _inert(near, distance=0.1)
    _substep(a, b)

    after = a.positions()
    for i in interior + [0]:
        assert np.allclose(after[i], BLOCK.nodes[i], atol=1e-6), (
            f"node {i} was not written through: {after[i]} was {BLOCK.nodes[i]}"
        )


def test_off_by_default_and_bit_identical_when_disabled():
    params = SolverParams(substeps=4)
    inv_mass = make_state(BLOCK.nodes).inv_mass

    default = GPUSolver(BLOCK, inv_mass, params)
    assert default.sh_body is None, "body collision must be off unless asked for"

    zero = GPUSolver(BLOCK, inv_mass, params, body_distance=0.0)
    for _ in range(5):
        default.step()
        zero.step()
    assert np.array_equal(default.positions(), zero.positions())


def test_substep_repeated_equals_step():
    """The refactor guard. Every solo body now goes through substep()."""
    params = SolverParams(substeps=4)
    inv_mass = make_state(BLOCK.nodes).inv_mass

    whole = GPUSolver(BLOCK, inv_mass, params)
    piece = GPUSolver(BLOCK, inv_mass, params)
    h = params.dt / params.substeps
    for _ in range(3):
        whole.step()
        for _ in range(params.substeps):
            piece.substep(h)

    assert np.array_equal(whole.positions(), piece.positions()), (
        "substep() x n drifted from step()"
    )


def test_a_body_with_no_partner_is_unaffected_by_the_pass():
    """A group of one still runs substep(others=()) - it must be a no-op."""
    params = SolverParams(substeps=4)
    inv_mass = make_state(BLOCK.nodes).inv_mass

    plain = GPUSolver(BLOCK, inv_mass, params)
    armed = GPUSolver(BLOCK, inv_mass, params, body_distance=0.3)
    for _ in range(5):
        plain.step()
        armed.step()

    assert np.array_equal(plain.positions(), armed.positions()), (
        "body collision changed a body that has nothing to collide with"
    )
