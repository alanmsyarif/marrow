"""Sticky colliders: hold material to the surface and drag it along.

Non-penetration can only push. A collider that lifts away therefore leaves the
body behind, which makes the whole squash-and-draw-out shape of a stretch shot
impossible. A sticky collider records the contact point in its own LOCAL space,
so the anchor rides the animated transform and the material follows it.
"""

import numpy as np
from mathutils import Matrix

from _oracle_harness import BLOCK
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver

# BLOCK spans 0..1 on every axis. A box collider is a unit cube in its own
# local space, so this one covers the top half of the block and nothing else.
def _plate(z, half=4.0):
    """Box whose underside sits at ``z``, wide enough to cover the block."""
    world = Matrix.Translation((0.5, 0.5, z + 1.0)) @ Matrix.Diagonal(
        (half, half, 1.0, 1.0)
    )
    return (2, world.inverted(), world, True)


def _solver(colliders, stick_break=0.0, substeps=4):
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps)
    state = make_state(BLOCK.nodes)
    return GPUSolver(
        BLOCK, state.inv_mass, params,
        colliders=colliders, stick_break=stick_break,
    )


def _top_z(solver):
    return float(solver.positions()[:, 2].max())


def test_a_sticky_collider_drags_the_body_when_it_moves_away():
    """The point of the feature. A push-only collider cannot do this."""
    solver = _solver([_plate(0.6)])
    solver.step()
    for lift in (0.8, 1.0, 1.2, 1.4):
        solver.colliders = [_plate(lift)]
        solver.step()
    assert _top_z(solver) > 1.15, (
        f"body did not follow the lifting plate: top z {_top_z(solver):.3f}, "
        f"plate underside at 1.4"
    )


def test_a_non_sticky_collider_leaves_the_body_behind():
    """Control: same motion, sticky off, so only push-out applies."""
    kind, to_local, to_world, _sticky = _plate(0.6)
    solver = _solver([(kind, to_local, to_world, False)])
    solver.step()
    for lift in (0.8, 1.0, 1.2, 1.4):
        k, tl, tw, _s = _plate(lift)
        solver.colliders = [(k, tl, tw, False)]
        solver.step()
    assert _top_z(solver) < 1.05, (
        f"a non-sticky collider dragged the body: top z {_top_z(solver):.3f}"
    )


def test_the_hold_releases_once_the_material_pulls_hard_enough():
    """Stick Break is the knob. A tiny one must let go almost at once."""
    held = _solver([_plate(0.6)], stick_break=0.0)
    breaks = _solver([_plate(0.6)], stick_break=1.0e-6)
    for s in (held, breaks):
        s.step()
        for lift in (0.8, 1.0, 1.2, 1.4):
            s.colliders = [_plate(lift)]
            s.step()
    assert _top_z(breaks) < _top_z(held), (
        f"break distance changed nothing: broke {_top_z(breaks):.3f} "
        f"vs held {_top_z(held):.3f}"
    )


def test_a_sticky_collider_still_keeps_material_out_of_itself():
    """Sticking must not cost non-penetration."""
    solver = _solver([_plate(0.6)])
    solver.step()
    out = solver.positions()
    assert out[:, 2].max() < 0.6 + 1e-4, (
        f"material is inside the plate: top z {out[:, 2].max():.4f}"
    )


def test_nothing_sticks_before_contact():
    """A plate held clear of the body must not grab it from a distance.

    Compared against a solver with no collider at all, not against the rest
    pose: a free cage does not sit perfectly still either. The hydrostatic
    constraint targets det(F) = 1 + mu/lam, so an untouched body breathes
    outwards by a fraction of a percent, and measuring against rest would
    read that as the plate having pulled it.
    """
    free = _solver([])
    plated = _solver([_plate(3.0)])
    for lift in (6.0, 9.0):
        free.step()
        plated.step()
        plated.colliders = [_plate(lift)]
    assert np.allclose(plated.positions(), free.positions(), atol=1e-6), (
        f"body moved for a plate it never touched: max delta from the "
        f"untouched control "
        f"{float(np.abs(plated.positions() - free.positions()).max()):.3e}"
    )
