"""What Tear Strain measures, and what tearing leaves behind.

Three properties the old Frobenius-norm criterion did not have:

  1. the slider reads as a real stretch ratio, the same one in every
     deformation mode, instead of only under isotropic swelling;
  2. torn material keeps its volume constraint, so it goes slack without
     inflating;
  3. no node is ever left with zero intact tets, because such a node has no
     constraint at all and free-falls, dragging a spike through the render
     mesh that the fixed topology can never resolve into debris.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.layout import pack_nodes
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver
from marrow.gpu.textures import upload


def _solver(tear_threshold, stretch=None, substeps=10, mu=1.0e4, lam=1.0e5):
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=substeps, mu=mu, lam=lam
    )
    state = make_state(BLOCK.nodes)
    solver = GPUSolver(BLOCK, state.inv_mass, params, tear_threshold=tear_threshold)
    if stretch is not None:
        solver.tex_x = upload(
            pack_nodes(BLOCK.nodes * np.array(stretch), state.inv_mass)
        )
    return solver


def _cage_volume(nodes, tets):
    p0 = nodes[tets[:, 0]]
    e1, e2, e3 = (nodes[tets[:, i]] - p0 for i in (1, 2, 3))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3)).sum() / 6.0)


# The criterion tests run one substep with the stiffnesses off. Two things
# would otherwise measure the dynamics rather than the threshold: a released
# stretched block is a spring that snaps back and overshoots, and within a
# single substep the first colour's projection has already moved nodes before
# the last colour is even looked at. With mu and lam at zero nothing moves, so
# the tear test is evaluated against exactly the deformation set up here. That
# it still tears at all is the point: tearing no longer needs mu > 0.
def _criterion_solver(threshold, stretch):
    return _solver(threshold, stretch=stretch, substeps=1, mu=0.0, lam=0.0)


def test_the_threshold_is_a_stretch_ratio_in_every_direction():
    """A 1.4x pull must not tear at 1.5, on any axis. Under the old
    norm-based criterion a uniaxial pull survived all the way to about 2.4x."""
    for axis in range(3):
        stretch = [1.0, 1.0, 1.0]
        stretch[axis] = 1.4
        solver = _criterion_solver(1.5, stretch)
        solver.step()
        assert solver.torn_flags().sum() == 0.0, (
            f"a 1.4x pull on axis {axis} tore against a 1.5 threshold"
        )


def test_a_pull_just_past_the_threshold_does_tear():
    solver = _criterion_solver(1.5, (1.8, 1.0, 1.0))
    solver.step()
    assert solver.torn_flags().sum() > 0, "a 1.8x pull must tear at 1.5"


def test_the_threshold_reads_the_same_on_every_axis():
    """Same stretch, different axis, same verdict. A criterion that mixed the
    three directions together could not promise this."""
    torn = []
    for axis in range(3):
        stretch = [1.0, 1.0, 1.0]
        stretch[axis] = 2.0
        solver = _criterion_solver(1.5, stretch)
        solver.step()
        torn.append(float(solver.torn_flags().sum()))
    assert len(set(torn)) == 1, f"axis-dependent tearing: {torn}"


def test_rotating_the_cage_does_not_tear_it():
    """Principal stretch is rotation invariant; a rigid turn is not strain."""
    angle = 0.7
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    state = make_state(BLOCK.nodes)
    solver = _solver(1.05)
    solver.tex_x = upload(pack_nodes(BLOCK.nodes @ rot.T, state.inv_mass))
    solver.step()
    assert solver.torn_flags().sum() == 0.0, "a pure rotation tore the cage"


def test_torn_material_does_not_inflate():
    """Torn tets keep a volume constraint, so the cage settles instead of
    growing. Without one it ballooned to 3.1x on a stretch shot and kept going
    - broken material is not new material.

    The bar is that volume stops growing, not that it returns to rest. A tet
    torn at 3x holds 3x, and that is the design: it stops resisting, it does
    not start pulling. Rest volume would make torn material a spring again.
    """
    solver = _solver(1.2, stretch=(3.0, 1.0, 1.0))
    for _ in range(20):
        solver.step()
    assert solver.torn_flags().sum() > 0, "this case is meant to tear"
    settled = _cage_volume(solver.positions(), BLOCK.tets)
    for _ in range(40):
        solver.step()
    later = _cage_volume(solver.positions(), BLOCK.tets)
    assert later < settled * 1.05, (
        f"torn cage is still inflating: {settled:.4f} -> {later:.4f} over 40 "
        f"more frames ({later / settled:.3f}x)"
    )


def test_no_node_is_ever_left_without_an_intact_tet():
    """The orphan rule. A node with no intact tet free-falls, and fixed render
    topology turns that into a spike rather than debris."""
    solver = _solver(1.05, stretch=(4.0, 2.0, 2.0))
    for _ in range(20):
        solver.step()
    torn = solver.torn_flags() > 0.5
    assert torn.any(), "this case is meant to tear"
    live = np.zeros(BLOCK.n_nodes, dtype=np.int64)
    for tet in BLOCK.tets[~torn]:
        live[tet] += 1
    assert int((live == 0).sum()) == 0, (
        f"{int((live == 0).sum())} of {BLOCK.n_nodes} nodes have no intact tet"
    )


def test_a_shredded_body_stays_finite_and_bounded():
    solver = _solver(1.05, stretch=(6.0, 2.0, 2.0))
    for _ in range(20):
        solver.step()
    out = solver.positions()
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() < 100.0, (
        f"material ran away to {np.abs(out).max():.1f} units"
    )


def test_torn_flags_come_back_in_mesh_tet_order():
    """The texture is in colour order. A caller indexing mesh.tets with a raw
    read would have been reading some other tet's flag."""
    solver = _solver(1.2, stretch=(3.0, 1.0, 1.0))
    solver.step()
    flags = solver.torn_flags()
    assert flags.shape == (BLOCK.n_tets,)
    assert set(np.unique(flags)) <= {0.0, 1.0}, "the torn flag must stay binary"
