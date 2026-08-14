"""Velocity clamp: no node keeps a speed above 0.2 thicknesses per substep.

The reference self-collision caps particle velocity at 0.2 * thickness / h so
fast material cannot tunnel through thin contact features and wad up instead
of folding. Marrow applies the same cap in integrate, active whenever a
contact thickness (self or body) is set, inert otherwise.

The cap limits the velocity carried into the next predict; the position
corrections of the substep that produced it stand. The tests therefore load
a huge velocity, take one step to let integrate clamp it, and measure the
step after.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.layout import pack_nodes
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver
from marrow.gpu.textures import upload

THICK = 0.2
FAST = 100.0


def _fast_block(distance, body=False):
    """The whole block translating rigidly at FAST m/s, no contacts.

    Every node gets the same velocity, so relative positions never change and
    the rest-distance gate keeps self-collision out of the measurement. The
    first step consumes the raw velocity and clamps it; the caller measures
    the second.
    """
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=1, mu=0.0, lam=0.0, damping=1.0
    )
    inv_mass = make_state(BLOCK.nodes).inv_mass
    kwargs = {"body_distance": distance} if body else {"self_distance": distance}
    solver = GPUSolver(BLOCK, inv_mass, params, **kwargs)

    vel = np.zeros_like(BLOCK.nodes)
    vel[:, 0] = FAST
    solver.tex_v = upload(pack_nodes(vel, np.zeros(BLOCK.n_nodes)))
    return solver


def _second_step_move(solver):
    solver.step()
    mid = solver.positions()
    solver.step()
    return float(np.abs(solver.positions() - mid)[:, 0].max())


def test_a_fast_body_is_capped_at_a_fifth_of_the_thickness_per_substep():
    solver = _fast_block(THICK)
    moved = _second_step_move(solver)
    cap = 0.2 * THICK
    assert moved < cap * 1.01, (
        f"clamp did not hold: moved {moved:.4f} per substep, cap {cap:.4f}"
    )
    # The body still translates, it is merely capped.
    assert moved > 0.9 * cap, f"clamp over-limited: moved {moved:.4f}"


def test_body_collision_thickness_engages_the_cap_too():
    solver = _fast_block(THICK, body=True)
    moved = _second_step_move(solver)
    assert moved < 0.2 * THICK * 1.01, (
        f"body thickness did not engage the clamp: moved {moved:.4f}"
    )


def test_no_contact_thickness_means_no_cap():
    solver = _fast_block(0.0)
    moved = _second_step_move(solver)
    want = FAST * solver.params.dt
    assert np.isclose(moved, want, rtol=1e-3), (
        f"no-contact trajectory changed by the clamp: moved {moved:.4f}, "
        f"want {want:.4f}"
    )
