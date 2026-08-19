"""A pin that rides the animation instead of the origin.

Pinning holds material by zeroing inverse mass, and every pass reads that
as "does not move" - which is also why hooking a pinned region to an Empty
and dragging it does nothing. A kinematic pin keeps the rigidity and gives
up the frozen part: the node is driven to its animation target and takes
the surrounding material with it.

Two guards carry the whole change. The attachment pass stores the target
for a pinned node instead of skipping it, and the integrator advances a
pinned node's position instead of returning. This is the oracle side; the
GLSL is diffed against it in tests/blender.
"""

import numpy as np

from marrow.core.lattice import build_lattice
from marrow.core.solver_ref import (
    SolverParams,
    make_state,
    precompute,
    solve_attachment,
    step,
)

BLOCK = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))
PINNED = np.array([0, 1], dtype=np.int32)
STILL = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10, attach=1.0, damping=1.0)


def _free_mask(state):
    free = np.ones(state.nodes.shape[0], dtype=bool)
    free[PINNED] = False
    return free


def test_a_kinematic_pin_lands_on_its_target():
    state = make_state(BLOCK.nodes, pinned=PINNED)
    state.predicted[:] = BLOCK.nodes
    targets = BLOCK.nodes + np.array([2.0, 0.0, 0.0])
    solve_attachment(state, targets, 0.0, 1.0 / 240.0, kinematic=True)
    assert np.allclose(state.predicted[PINNED], targets[PINNED], atol=1e-12)


def test_a_static_pin_still_ignores_its_target():
    """The default must not change: kinematic is opt-in, so a pin with the
    flag off keeps outranking the armature the way it always has."""
    state = make_state(BLOCK.nodes, pinned=PINNED)
    state.predicted[:] = BLOCK.nodes
    solve_attachment(state, BLOCK.nodes + 2.0, 0.0, 1.0 / 240.0)
    assert np.array_equal(state.predicted[PINNED], BLOCK.nodes[PINNED])


def test_a_static_pin_never_moves_and_never_gains_velocity():
    """The integrator advances a pinned node's position unconditionally,
    with no kinematic flag guarding it, on the claim that p == x already
    holds for a static pin - predict copies it, every projection scales its
    correction by an inverse mass of zero, and all three contact passes
    return early. If that claim is wrong this drifts. Exact equality, not
    a tolerance, because the claim is bit-identity.
    """
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    params = SolverParams(substeps=10, attach=1.0)
    state = make_state(BLOCK.nodes, pinned=PINNED)
    targets = BLOCK.nodes + np.array([5.0, 5.0, 5.0])
    for _ in range(6):
        step(state, BLOCK.tets, dm_inv, rest_vol, params, targets)
    assert np.array_equal(state.nodes[PINNED], BLOCK.nodes[PINNED])
    assert np.array_equal(state.velocities[PINNED], np.zeros((PINNED.size, 3)))


def test_a_kinematic_pin_carries_the_material_with_it():
    """The gap this closes. A frozen pin leaves the body behind; a driven
    one drags it.

    Attachment is deliberately soft here. Only the pinned nodes are given a
    moving target, which is what a Hook bound to a vertex group produces;
    every other node is aimed at its rest position. At stiffness 1.0 that
    aim is a hard snap, so the free material is nailed to rest and cannot
    follow anything - measured, free travel of exactly 0.00000. A soft pull
    lets the elastic coupling win, which is the setting this feature is
    actually used at.
    """
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=10, attach=0.1, damping=1.0,
        pin_kinematic=True,
    )
    state = make_state(BLOCK.nodes, pinned=PINNED)
    free = _free_mask(state)
    shift = np.array([0.1, 0.0, 0.0])
    for f in range(1, 6):
        targets = BLOCK.nodes.copy()
        targets[PINNED] += f * shift
        step(state, BLOCK.tets, dm_inv, rest_vol, params, targets)

    assert np.allclose(state.nodes[PINNED], BLOCK.nodes[PINNED] + 5 * shift, atol=1e-9)
    assert state.nodes[free][:, 0].mean() > BLOCK.nodes[free][:, 0].mean() + 0.1, (
        "the free material did not follow the driven pin"
    )


def test_a_kinematic_pin_is_driven_not_simulated():
    """It is moved by the animation, so it carries no velocity of its own -
    otherwise freeing the pin would fling it."""
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=10, attach=1.0, pin_kinematic=True
    )
    state = make_state(BLOCK.nodes, pinned=PINNED)
    targets = BLOCK.nodes.copy()
    targets[PINNED] += np.array([1.0, 0.0, 0.0])
    step(state, BLOCK.tets, dm_inv, rest_vol, params, targets)
    assert np.array_equal(state.velocities[PINNED], np.zeros((PINNED.size, 3)))


def test_kinematic_with_no_targets_is_inert():
    """The flag alone drives nothing. Targets are what a driven pin follows,
    and they only exist when Attachment is on, so a kinematic pin with none
    is still a frozen one.

    This used to be stated as "attach at 0 leaves the pin frozen", because
    stiffness 0 disabled the pass outright. It no longer does: 0 now means
    "targets for the pins, hands off the free material", which is the
    setting the feature is actually used at. Absent targets is what makes
    it inert now - see test_stiffness_zero_with_a_kinematic_pin_drives_only
    _the_pins.
    """
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    params = SolverParams(substeps=5, attach=0.0, pin_kinematic=True)
    state = make_state(BLOCK.nodes, pinned=PINNED)
    for _ in range(3):
        step(state, BLOCK.tets, dm_inv, rest_vol, params, None)
    assert np.array_equal(state.nodes[PINNED], BLOCK.nodes[PINNED])


def test_stiffness_zero_with_a_kinematic_pin_drives_only_the_pins():
    """The setting the feature actually wants.

    Attachment aims every node at its evaluated position, and for material
    the animation does not touch that aim is the rest pose - so the pass
    that supplies a driven pin's targets also nails the rest of the body
    down. Stiffness 0 asks for targets on the pins and nothing on anyone
    else, which used to disable the pass outright and leave the pin frozen.
    """
    state = make_state(BLOCK.nodes, pinned=PINNED)
    state.predicted[:] = BLOCK.nodes + 0.25
    targets = BLOCK.nodes + np.array([2.0, 0.0, 0.0])
    solve_attachment(state, targets, 0.0, 1.0 / 240.0,
                     kinematic=True, drive_free=False)
    free = _free_mask(state)
    assert np.allclose(state.predicted[PINNED], targets[PINNED], atol=1e-12)
    assert np.array_equal(state.predicted[free], (BLOCK.nodes + 0.25)[free]), (
        "free material must be left entirely to the elastic solve"
    )


def test_stiffness_zero_carries_the_body_where_a_stiff_attachment_cannot():
    """Measured on the real cage: at Attach Stiffness 0.05 a driven pin
    dragged the body 0.158 against its own 1.473, and at 0.0001 it dragged
    it 1.515 - the attachment grip was the whole obstacle, not mass or
    gravity. Stiffness 0 is that limit without the magic number.
    """
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    shift = np.array([0.1, 0.0, 0.0])

    def travel(attach, kinematic=True):
        params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=10,
                              attach=attach, damping=1.0, pin_kinematic=kinematic)
        state = make_state(BLOCK.nodes, pinned=PINNED)
        for f in range(1, 6):
            targets = BLOCK.nodes.copy()
            targets[PINNED] += f * shift
            step(state, BLOCK.tets, dm_inv, rest_vol, params, targets)
        free = _free_mask(state)
        return (state.nodes[free] - BLOCK.nodes[free])[:, 0].mean()

    gripped = travel(0.5)
    released = travel(0.0)
    assert released > gripped * 2.0, (
        f"pins-only carried the body {released:.4f}, barely better than "
        f"{gripped:.4f} at stiffness 0.5"
    )


def test_stiffness_zero_without_a_kinematic_pin_is_still_no_attachment():
    """Stiffness 0 only means "pins only" when there are driven pins. With
    the flag off it must stay exactly what it always was: no pass at all."""
    dm_inv, rest_vol = precompute(BLOCK.nodes, BLOCK.tets)
    params = SolverParams(substeps=5, attach=0.0)
    plain = make_state(BLOCK.nodes, pinned=PINNED)
    aimed = make_state(BLOCK.nodes, pinned=PINNED)
    for _ in range(3):
        step(plain, BLOCK.tets, dm_inv, rest_vol, params)
        step(aimed, BLOCK.tets, dm_inv, rest_vol, params, BLOCK.nodes + 3.0)
    assert np.array_equal(plain.nodes, aimed.nodes)
