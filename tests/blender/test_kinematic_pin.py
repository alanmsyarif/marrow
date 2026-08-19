"""GLSL side of pins that ride the animation, diffed against the oracle.

A pin holds material by zeroing inverse mass, and both the attachment and
integrate kernels read that as "skip". A kinematic pin flips exactly those
two guards: attach stores the animation target for a pinned node, and
integrate advances its position instead of returning. Everything else -
predict, solve, blend, all three contact passes - still skips it, which is
what keeps a driven pin rigid.

The integrate change is deliberately unguarded by the kinematic flag, on
the claim that p == x already holds for a static pin. test_a_static_pin_...
below is that claim, checked rather than asserted.
"""

import gpu
import numpy as np

from _oracle_harness import BLOCK
from marrow.core.coloring import color_tets
from marrow.core.layout import color_ordered
from marrow.core.solver_ref import SolverParams, make_state, precompute, step
from marrow.gpu.solver import GPUSolver

gpu.init()

TOL = 1e-4
PINNED = np.array([0, 1], dtype=np.int32)
SHIFT = np.array([0.02, 0.0, 0.0])


def _targets(nodes, scale):
    out = np.asarray(nodes, dtype=np.float64).copy()
    out[PINNED] += scale * SHIFT
    return out


def _solver(params, kinematic):
    state = make_state(BLOCK.nodes, pinned=PINNED)
    # Frame-zero targets are the rest nodes, so the solver's start state is
    # the rest shape - with attachment on it starts from the targets.
    return GPUSolver(
        BLOCK, state.inv_mass, params,
        attach_stiffness=params.attach,
        attach_targets=_targets(BLOCK.nodes, 0),
        pin_kinematic=kinematic,
    )


def _gpu_run(params, frames, kinematic):
    solver = _solver(params, kinematic)
    for f in range(1, frames + 1):
        solver.set_targets(_targets(BLOCK.nodes, f))
        solver.step()
    return solver.positions()


def _oracle_run(params, frames):
    colors = color_tets(BLOCK.tets, BLOCK.n_nodes)
    ordered, _ = color_ordered(BLOCK.tets, colors)
    state = make_state(BLOCK.nodes, pinned=PINNED)
    dm_inv, rest_vol = precompute(BLOCK.nodes, ordered)
    for f in range(1, frames + 1):
        step(state, ordered, dm_inv, rest_vol, params, _targets(BLOCK.nodes, f))
    return state.nodes


def test_a_kinematic_pin_lands_on_its_target():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1,
                          damping=1.0, pin_kinematic=True)
    out = _gpu_run(params, 5, kinematic=True)
    want = _targets(BLOCK.nodes, 5)[PINNED]
    assert np.allclose(out[PINNED], want, atol=1e-5), (
        f"driven pin missed its target by {np.abs(out[PINNED] - want).max():.2e}"
    )


def test_a_static_pin_ignores_a_moving_target():
    """Kinematic is opt-in. With the flag off the pin stays frozen even
    though the targets underneath it are moving."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1,
                          damping=1.0)
    out = _gpu_run(params, 5, kinematic=False)
    assert np.allclose(out[PINNED], BLOCK.nodes[PINNED], atol=1e-6), (
        f"static pin drifted {np.abs(out[PINNED] - BLOCK.nodes[PINNED]).max():.2e}"
    )


def test_a_kinematic_pin_carries_the_material_with_it():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1,
                          damping=1.0, pin_kinematic=True)
    driven = _gpu_run(params, 5, kinematic=True)
    frozen = _gpu_run(
        SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1,
                     damping=1.0),
        5, kinematic=False,
    )
    free = np.ones(BLOCK.n_nodes, dtype=bool)
    free[PINNED] = False
    assert driven[free][:, 0].mean() > frozen[free][:, 0].mean() + 1e-3, (
        "the free material did not follow the driven pin"
    )


def test_a_kinematic_pin_matches_the_oracle():
    """The repo's rule: a wrong sign in GLSL is otherwise indistinguishable
    from a wrong sign in the derivation."""
    params = SolverParams(gravity=(0.0, 0.0, -9.81), substeps=10, attach=0.3,
                          pin_kinematic=True)
    got = _gpu_run(params, 6, kinematic=True)
    want = _oracle_run(params, 6)
    gap = np.abs(got - want).max()
    assert gap < TOL, f"GPU and oracle disagree by {gap:.2e}"


def test_a_kinematic_pin_outranks_a_collider():
    """Driven means driven: a collider pressing on a kinematic pin does not
    push it off its target, the same way a static pin outranks one."""
    from mathutils import Matrix

    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1,
                          damping=1.0, pin_kinematic=True)
    solver = _solver(params, kinematic=True)
    # A box swallowing the whole block, so the pinned nodes are deep inside
    # it and a push-out would be unmissable.
    world = Matrix.Translation((0.5, 0.5, 0.5)) @ Matrix.Diagonal((4.0, 4.0, 4.0, 1.0))
    solver.colliders = [(2, world.inverted(), world, False, None, 0.0)]
    for f in range(1, 6):
        solver.set_targets(_targets(BLOCK.nodes, f))
        solver.step()
    out = solver.positions()
    want = _targets(BLOCK.nodes, 5)[PINNED]
    assert np.allclose(out[PINNED], want, atol=1e-5), (
        f"a collider moved a driven pin by {np.abs(out[PINNED] - want).max():.2e}"
    )


def _pins_only_params(**kw):
    return SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.0,
                        damping=1.0, pin_kinematic=True, **kw)


def test_pins_only_leaves_the_free_material_alone():
    """Attach Stiffness 0 with a driven pin: targets for the pins, hands
    off everything else. The pass still has to run - the pin needs its
    target - but no free node may be pulled towards one."""
    solver = _solver(_pins_only_params(), kinematic=True)
    assert solver.attach_enabled, "the pass must still run for the pins"
    for f in range(1, 6):
        solver.set_targets(_targets(BLOCK.nodes, f))
        solver.step()
    out = solver.positions()
    want = _targets(BLOCK.nodes, 5)[PINNED]
    assert np.allclose(out[PINNED], want, atol=1e-5)
    free = np.ones(BLOCK.n_nodes, dtype=bool)
    free[PINNED] = False
    # Free targets never move, so an attachment grip would hold them at
    # rest. Dragged by the pin through the material instead, they move.
    assert np.abs(out[free] - BLOCK.nodes[free]).max() > 1e-4, (
        "the free material was held at its rest pose"
    )


def test_pins_only_carries_the_body_further_than_a_stiff_attachment():
    free = np.ones(BLOCK.n_nodes, dtype=bool)
    free[PINNED] = False

    def travel(params):
        out = _gpu_run(params, 20, kinematic=True)
        return float((out[free] - BLOCK.nodes[free])[:, 0].mean())

    gripped = travel(SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10,
                                  attach=0.5, damping=1.0, pin_kinematic=True))
    released = travel(_pins_only_params())
    # BLOCK is a 27-node lattice with 2 of them pinned, so the separation
    # is far smaller than on a real cage, where the same comparison
    # measured 0.158 against 1.515 - roughly tenfold. Here it converges to
    # almost exactly 2x (measured 0.2559 against 0.1281) and stays there
    # however long the run, so the bar sits below that rather than on it.
    assert released > gripped * 1.5, (
        f"pins-only carried the body {released:.4f} against {gripped:.4f} "
        f"at stiffness 0.5"
    )


def test_pins_only_matches_the_oracle():
    params = _pins_only_params()
    params.gravity = (0.0, 0.0, -9.81)
    params.damping = 0.999
    got = _gpu_run(params, 6, kinematic=True)
    want = _oracle_run(params, 6)
    gap = np.abs(got - want).max()
    assert gap < TOL, f"GPU and oracle disagree by {gap:.2e}"


def test_pins_only_starts_the_free_cage_at_rest():
    """With a full attachment the whole cage starts from the targets, so the
    body begins posed. Pins-only must not do that: aiming free material at
    the pose is exactly the grip this mode exists to drop, and starting it
    there would apply the same displacement before the first substep."""
    solver = _solver(_pins_only_params(), kinematic=True)
    start = solver.positions()
    free = np.ones(BLOCK.n_nodes, dtype=bool)
    free[PINNED] = False
    assert np.allclose(start[free], BLOCK.nodes[free], atol=1e-6), (
        "free cage nodes were displaced to the pose before the first substep"
    )
