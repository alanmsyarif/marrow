"""A cage that starts below the ground plane must not be launched by it.

The collide pass depenetrates by moving the predicted position and integrate
reads that move as velocity. Mid-simulation the depth is bounded by v * h, so
the velocity read back is the one that caused it. The starting state has no
such bound, and a half-buried body used to leave its first substep at
hundreds of metres per second and skin into spikes.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver

# BLOCK spans 0..1 in z, so this buries all but the top of it.
SUNK = BLOCK.nodes - np.array([0.0, 0.0, 0.8])
SUNK_MESH = type(BLOCK)(SUNK, BLOCK.tets)


def _solver(mesh, ground_on=True):
    state = make_state(mesh.nodes)
    return GPUSolver(
        mesh, state.inv_mass, SolverParams(),
        ground_z=0.0, ground_on=ground_on,
    )


def test_a_buried_cage_starts_on_the_ground_not_under_it():
    out = _solver(SUNK_MESH).positions()
    assert out[:, 2].min() > -1e-6, (
        f"cage still starts below the ground plane: min z {out[:, 2].min():.4f}"
    )


def test_the_lift_is_rigid_so_the_shape_survives():
    """A clamp would flatten the buried half; a translation must not."""
    out = _solver(SUNK_MESH).positions()
    shifted = SUNK + np.array([0.0, 0.0, 0.8])
    assert np.allclose(out, shifted, atol=1e-5), (
        f"lift was not a rigid translation: max delta "
        f"{float(np.abs(out - shifted).max()):.3e}"
    )


def test_a_buried_cage_stays_put_instead_of_being_launched():
    solver = _solver(SUNK_MESH)
    start = solver.positions()[:, 2].mean()
    for _ in range(10):
        solver.step()
    end = solver.positions()[:, 2].mean()
    assert abs(end - start) < 0.5, (
        f"body was launched: mean z moved {end - start:+.3f} in 10 frames"
    )


def test_a_cage_above_the_ground_is_left_alone():
    out = _solver(BLOCK).positions()
    assert np.allclose(out, BLOCK.nodes, atol=1e-6), (
        f"clear cage was moved: max delta "
        f"{float(np.abs(out - BLOCK.nodes).max()):.3e}"
    )


def test_the_lift_needs_the_ground_plane_switched_on():
    out = _solver(SUNK_MESH, ground_on=False).positions()
    assert np.allclose(out, SUNK, atol=1e-6), (
        "cage was lifted with the ground plane disabled"
    )
