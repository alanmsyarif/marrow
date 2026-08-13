"""Self-collision between surface nodes of the cage.

All pairs, Jacobi, no spatial hash - see
docs/superpowers/specs/2026-08-13-marrow-self-collision-design.md.

Most of these run with mu = lam = 0 so the elastic solve contributes nothing
and the only thing that can move a node is the self-collision pass. Where the
rest configuration matters, tex_x is overwritten after construction: the
solver builds rest state from the mesh it is given, so rest-far/now-close is
only reachable by moving the current positions afterwards.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.layout import pack_nodes
from marrow.core.solver_ref import SolverParams, make_state
from marrow.core.tetmesh import TetMesh, surface_nodes
from marrow.gpu.solver import GPUSolver
from marrow.gpu.textures import upload

THICK = 0.2

_UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _two_tets(gap):
    """Two disjoint unit tets, the second offset ``gap`` along x.

    Node 1 sits at (1,0,0) and node 4 at (gap,0,0), so the closest pair
    between the two tets is exactly gap - 1 apart and every other pair is
    far outside any thickness used here.
    """
    nodes = np.vstack([_UNIT, _UNIT + np.array([gap, 0.0, 0.0])])
    return TetMesh(nodes, np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32))


def _inert(mesh, inv_mass=None, distance=THICK, substeps=1):
    """A solver whose only active constraint is self-collision."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps, mu=0.0, lam=0.0)
    if inv_mass is None:
        inv_mass = make_state(mesh.nodes).inv_mass
    return GPUSolver(mesh, inv_mass, params, self_distance=distance)


def _teleport(solver, positions, inv_mass):
    """Move the body without touching its rest state, as poison_for_test does."""
    solver.tex_x = upload(pack_nodes(positions, inv_mass))


def _sep(solver, i, j):
    p = solver.positions()
    return float(np.linalg.norm(p[i] - p[j]))


def test_two_nodes_inside_the_thickness_are_pushed_to_it():
    """The feature. Rest positions far apart, so the gate does not fire."""
    mesh = _two_tets(10.0)
    inv_mass = make_state(mesh.nodes).inv_mass
    solver = _inert(mesh)

    now = mesh.nodes.copy()
    now[4:] -= np.array([10.0 - (1.0 + 0.5 * THICK), 0.0, 0.0])
    _teleport(solver, now, inv_mass)
    assert np.isclose(_sep(solver, 1, 4), 0.5 * THICK), "test setup is wrong"

    solver.step()
    assert _sep(solver, 1, 4) > 0.99 * THICK, (
        f"pair still overlapping: {_sep(solver, 1, 4):.4f}, want {THICK}"
    )


def test_the_rest_distance_gate_leaves_close_rest_neighbours_alone():
    """Nodes that were always this close are not a collision.

    Without this the lattice's own neighbours - one Resolution apart at rest,
    which is the default thickness - would fight their tets every substep.
    """
    mesh = _two_tets(1.0 + 0.5 * THICK)
    before = mesh.nodes.copy()
    solver = _inert(mesh)
    assert np.isclose(_sep(solver, 1, 4), 0.5 * THICK), "test setup is wrong"

    solver.step()
    moved = float(np.abs(solver.positions() - before).max())
    assert moved < 1e-6, f"the gate did not hold: something moved {moved:.6f}"


def test_a_pinned_partner_takes_none_of_the_correction():
    """Mass weighted, where the reference splits a flat half."""
    mesh = _two_tets(10.0)
    inv_mass = make_state(mesh.nodes).inv_mass
    inv_mass[4] = 0.0
    solver = _inert(mesh, inv_mass=inv_mass)

    now = mesh.nodes.copy()
    now[4:] -= np.array([10.0 - (1.0 + 0.5 * THICK), 0.0, 0.0])
    _teleport(solver, now, inv_mass)
    solver.step()

    after = solver.positions()
    assert np.allclose(after[4], now[4], atol=1e-6), (
        f"pinned node moved to {after[4]}, was {now[4]}"
    )
    assert _sep(solver, 1, 4) > 0.99 * THICK, (
        f"free node did not take the whole correction: {_sep(solver, 1, 4):.4f}"
    )


def test_nodes_the_kernel_skips_still_come_out_of_the_ping_pong():
    """Interior and pinned nodes must be written through, not returned early.

    The pass writes a second image and the two are swapped, so a thread that
    returns without storing leaves whatever the other buffer last held - zero,
    on the first substep - and the node teleports to the origin.
    """
    inv_mass = make_state(BLOCK.nodes).inv_mass
    inv_mass[0] = 0.0
    interior = sorted(set(range(BLOCK.n_nodes)) - set(surface_nodes(BLOCK.tets).tolist()))
    assert interior, "BLOCK must have an interior node for this test to mean anything"

    solver = _inert(BLOCK, inv_mass=inv_mass, distance=0.1)
    solver.step()

    after = solver.positions()
    for i in interior + [0]:
        assert np.allclose(after[i], BLOCK.nodes[i], atol=1e-6), (
            f"node {i} was not written through: {after[i]} was {BLOCK.nodes[i]}"
        )


def test_off_by_default_and_bit_identical_when_disabled():
    params = SolverParams(substeps=4)
    inv_mass = make_state(BLOCK.nodes).inv_mass

    default = GPUSolver(BLOCK, inv_mass, params)
    assert default.sh_self is None, "self-collision must be off unless asked for"

    zero = GPUSolver(BLOCK, inv_mass, params, self_distance=0.0)
    assert zero.sh_self is None
    for _ in range(5):
        default.step()
        zero.step()
    assert np.array_equal(default.positions(), zero.positions()), (
        "self_distance=0 changed the trajectory"
    )


def test_a_flattened_body_pushes_itself_back_apart():
    """Integration: many simultaneous contacts, not one isolated pair.

    Squash the block to a twentieth of its height with the material switched
    off, so nothing but self-collision can un-flatten it. Its layers sit 0.5
    apart at rest, so the gate passes and each pair is driven to the
    thickness.
    """
    inv_mass = make_state(BLOCK.nodes).inv_mass
    flat = BLOCK.nodes.copy()
    flat[:, 2] *= 0.05

    def height(distance):
        solver = _inert(BLOCK, inv_mass=inv_mass, distance=distance, substeps=4)
        _teleport(solver, flat, inv_mass)
        for _ in range(10):
            solver.step()
        z = solver.positions()[:, 2]
        return float(z.max() - z.min())

    on = height(0.3)
    off = height(0.0)
    assert off < 0.06, f"control moved on its own: height {off:.4f}"
    assert on > 0.4, f"self-collision did not un-flatten the block: {on:.4f}"
