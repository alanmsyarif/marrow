import gpu
import numpy as np

from _oracle_harness import CUBE
from marrow.core.solver_ref import SolverParams
from marrow.gpu.solver import GPUSolver

gpu.init()


def _fibers_along_x(n_tets, phase=0.75):
    fiber = np.zeros((n_tets, 5), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _params(**kw):
    base = dict(gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4, wave_amp=0.4,
                wave_len=1.0, wave_speed=0.0, waveform=1)
    base.update(kw)
    return SolverParams(**base)


def _extent(nodes, axis):
    return float(nodes[:, axis].max() - nodes[:, axis].min())


def test_a_solver_built_without_fibers_still_runs():
    """Every existing body takes this path. The image must still bind."""
    solver = GPUSolver(CUBE, np.ones(CUBE.n_nodes), SolverParams())
    solver.step()
    assert np.all(np.isfinite(solver.positions()))


def test_fibers_contract_the_body_along_their_direction():
    inv_mass = np.ones(CUBE.n_nodes)
    solver = GPUSolver(CUBE, inv_mass, _params(), fiber=_fibers_along_x(CUBE.n_tets))
    before = _extent(CUBE.nodes, 0)
    for _ in range(20):
        solver.step()
    after = _extent(solver.positions(), 0)
    assert after < before - 1e-3, f"fiber did not contract: {before} -> {after}"


def test_sim_time_advances_one_substep_at_a_time():
    params = _params()
    solver = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params,
                       fiber=_fibers_along_x(CUBE.n_tets))
    assert solver.sim_time == 0.0
    solver.step()
    expected = params.dt
    assert abs(solver.sim_time - expected) < 1e-9, (
        f"one frame must advance the clock by dt, got {solver.sim_time}"
    )


def test_zero_fiber_stiffness_matches_a_solver_with_no_fibers():
    params = _params(fiber_k=0.0)
    with_fiber = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params,
                           fiber=_fibers_along_x(CUBE.n_tets))
    without = GPUSolver(CUBE, np.ones(CUBE.n_nodes), params)
    for _ in range(5):
        with_fiber.step()
        without.step()
    assert np.allclose(with_fiber.positions(), without.positions(), atol=1e-6)
