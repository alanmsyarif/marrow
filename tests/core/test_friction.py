"""Coulomb contact friction, on the reference plane contact.

This is the oracle the GLSL friction is diffed against. The algebra is
identical in all three contact kernels - only the source of the normal and
the penetration depth differs - so getting it right here is most of getting
it right on the card.
"""

import numpy as np

from marrow.core.solver_ref import make_state, solve_plane_contact


def _sliding_state(drop=0.1, slide=0.3):
    """One node that started above the plane and moved down and sideways.

    ``nodes`` is the substep-start position, ``predicted`` where the substep
    wants it: ``drop`` below the plane and ``slide`` along +x.
    """
    state = make_state(np.array([[0.0, 0.0, 0.0]]))
    state.predicted[:] = [[slide, 0.0, -drop]]
    return state


def test_high_friction_cancels_the_sliding_entirely():
    """A node pressed into the plane and dragged sideways holds where it was.

    The clamp is friction * depth against the tangential motion, so a large
    coefficient against a real penetration means the whole tangential step
    is given back - static friction, with no separate coefficient for it.
    """
    state = _sliding_state(drop=0.1, slide=0.02)
    solve_plane_contact(state, ground_z=0.0, friction=5.0)

    assert np.isclose(state.predicted[0, 2], 0.0), "not depenetrated"
    assert np.isclose(state.predicted[0, 0], 0.0), "tangential motion survived"


def test_zero_friction_leaves_the_slide_untouched():
    """The current behaviour, and it must stay reachable exactly."""
    state = _sliding_state(drop=0.1, slide=0.3)
    solve_plane_contact(state, ground_z=0.0, friction=0.0)

    assert np.isclose(state.predicted[0, 2], 0.0)
    assert np.isclose(state.predicted[0, 0], 0.3)


def test_the_correction_never_exceeds_the_coulomb_clamp():
    """Past the clamp the contact slips: it resists, it does not weld.

    Tangential motion of 0.3 against a 0.02 penetration at mu = 0.5 buys
    only 0.01 of resistance, so the node keeps 0.29 of its slide.
    """
    state = _sliding_state(drop=0.02, slide=0.3)
    solve_plane_contact(state, ground_z=0.0, friction=0.5)

    assert np.isclose(state.predicted[0, 0], 0.3 - 0.5 * 0.02)


def test_a_node_clear_of_the_plane_is_never_braked():
    """No contact, no friction. Otherwise friction becomes global drag."""
    state = make_state(np.array([[0.0, 0.0, 1.0]]))
    state.predicted[:] = [[0.5, 0.0, 1.0]]
    solve_plane_contact(state, ground_z=0.0, friction=5.0)

    assert np.allclose(state.predicted[0], [0.5, 0.0, 1.0])


def test_deeper_penetration_grips_harder():
    """Friction scales with depth, which is what makes it read as weight."""
    shallow = _sliding_state(drop=0.01, slide=0.3)
    deep = _sliding_state(drop=0.05, slide=0.3)
    solve_plane_contact(shallow, ground_z=0.0, friction=0.5)
    solve_plane_contact(deep, ground_z=0.0, friction=0.5)

    assert deep.predicted[0, 0] < shallow.predicted[0, 0]


def test_a_pinned_node_is_left_alone():
    state = make_state(np.array([[0.0, 0.0, 0.0]]), pinned=[0])
    state.predicted[:] = [[0.3, 0.0, -0.1]]
    solve_plane_contact(state, ground_z=0.0, friction=5.0)

    assert np.allclose(state.predicted[0], [0.3, 0.0, -0.1])


def test_friction_acts_across_the_full_tangent_plane():
    """Not just x. A diagonal drag is resisted along its own direction."""
    state = make_state(np.array([[0.0, 0.0, 0.0]]))
    state.predicted[:] = [[0.3, 0.4, -0.02]]   # tangential length 0.5
    solve_plane_contact(state, ground_z=0.0, friction=0.5)

    # 0.5 * 0.02 = 0.01 of resistance, removed along the drag direction.
    expected = np.array([0.3, 0.4]) * (1.0 - 0.01 / 0.5)
    assert np.allclose(state.predicted[0, :2], expected)
