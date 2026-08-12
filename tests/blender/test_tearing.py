"""Tearing by constraint failure.

A tet whose deviatoric strain passes the threshold is marked torn and stops
contributing any constraint, permanently. The material goes slack there and
pulls apart as the cage stretches. Nothing about the topology changes, so the
render mesh is never modified - that is what keeps the barycentric-embedding
decision intact.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver


def _solver(tear_threshold, stretch=None, substeps=10):
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps)
    state = make_state(BLOCK.nodes)
    solver = GPUSolver(BLOCK, state.inv_mass, params, tear_threshold=tear_threshold)
    if stretch is not None:
        stretched = BLOCK.nodes * np.array(stretch)
        solver.tex_x = _reupload(solver, stretched, state.inv_mass)
    return solver


def _reupload(solver, nodes, inv_mass):
    from marrow.core.layout import pack_nodes
    from marrow.gpu.textures import upload

    return upload(pack_nodes(nodes, inv_mass))


def test_nothing_tears_when_tearing_is_disabled():
    solver = _solver(0.0, stretch=(4.0, 1.0, 1.0))
    solver.step()
    assert solver.torn_flags().sum() == 0.0, "tearing was off but tets tore anyway"


def test_a_heavily_stretched_body_tears():
    solver = _solver(1.5, stretch=(4.0, 1.0, 1.0))
    solver.step()
    torn = solver.torn_flags()
    assert torn.sum() > 0, "a 4x stretch past a 1.5 threshold should tear something"
    assert set(np.unique(torn)) <= {0.0, 1.0}, "the torn flag must stay binary"


def test_a_gently_stretched_body_does_not_tear():
    solver = _solver(3.0, stretch=(1.05, 1.0, 1.0))
    solver.step()
    assert solver.torn_flags().sum() == 0.0, (
        "a 5% stretch must not tear against a 3.0 threshold"
    )


def test_a_lower_threshold_tears_more():
    forgiving = _solver(3.0, stretch=(3.0, 1.0, 1.0))
    brittle = _solver(1.2, stretch=(3.0, 1.0, 1.0))
    forgiving.step()
    brittle.step()
    assert brittle.torn_flags().sum() > forgiving.torn_flags().sum(), (
        "a lower tear threshold must tear at least as many tets"
    )


def test_tearing_is_permanent():
    """Once torn, a tet stays torn even after the strain is long gone."""
    solver = _solver(1.5, stretch=(4.0, 1.0, 1.0))
    solver.step()
    after_first = solver.torn_flags().sum()
    assert after_first > 0

    for _ in range(10):
        solver.step()
    assert solver.torn_flags().sum() >= after_first, "torn tets healed themselves"


def test_a_torn_body_does_not_spring_back_like_an_intact_one():
    """The point of the feature: torn material stops resisting."""
    intact = _solver(0.0, stretch=(4.0, 1.0, 1.0))
    torn = _solver(1.2, stretch=(4.0, 1.0, 1.0))
    for _ in range(15):
        intact.step()
        torn.step()

    intact_span = float(np.ptp(intact.positions()[:, 0]))
    torn_span = float(np.ptp(torn.positions()[:, 0]))
    assert torn_span > intact_span, (
        f"torn material should stay stretched: torn span {torn_span:.4f} "
        f"vs intact {intact_span:.4f}"
    )


def test_a_torn_solver_stays_finite():
    solver = _solver(1.1, stretch=(6.0, 2.0, 2.0))
    for _ in range(20):
        solver.step()
    assert np.all(np.isfinite(solver.positions()))
