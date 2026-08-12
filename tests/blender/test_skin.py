import gpu
import numpy as np

from _oracle_harness import BLOCK
from marrow.core.bind import bind_points, deform
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.solver import GPUSolver

gpu.init()

TOL = 1e-5


def _solver_with_render(points, params=None):
    params = params or SolverParams(gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0)
    state = make_state(BLOCK.nodes)
    solver = GPUSolver(BLOCK, state.inv_mass, params)
    idx, w = bind_points(BLOCK.nodes, BLOCK.tets, points)
    solver.attach_render(idx, w)
    return solver, idx, w


def test_skin_at_rest_reproduces_the_bound_points():
    rng = np.random.default_rng(7)
    points = rng.uniform(0.05, 0.95, size=(40, 3))
    solver, _, _ = _solver_with_render(points)
    out = solver.skin()
    if not np.allclose(out, points, atol=TOL):
        # Decisive diagnostic: does a second read recover? If yes the write
        # eventually lands and this is a visibility problem. If no, the skin
        # dispatch never wrote at all and retrying cannot help.
        again = solver.skin()
        third = solver.skin()
        raise AssertionError(
            f"skin at rest diverged by {float(np.abs(out - points).max()):.3e}; "
            f"read1 all_zero={bool(np.allclose(out, 0.0))} "
            f"read2 diverged_by={float(np.abs(again - points).max()):.3e} "
            f"read3 diverged_by={float(np.abs(third - points).max()):.3e} "
            f"n_render={solver.n_render} "
            f"cage_finite={bool(np.all(np.isfinite(solver.positions())))}"
        )


def test_skin_matches_the_cpu_deform():
    rng = np.random.default_rng(8)
    points = rng.uniform(0.05, 0.95, size=(40, 3))
    solver, idx, w = _solver_with_render(points)
    solver.step()

    cpu = deform(solver.positions(), BLOCK.tets, idx, w)
    gpu_out = solver.skin()
    assert gpu_out.shape == cpu.shape
    assert np.abs(gpu_out - cpu).max() < TOL, (
        f"skin diverged from the CPU deform by "
        f"{float(np.abs(gpu_out - cpu).max()):.3e}"
    )


def test_skin_follows_a_falling_cage():
    rng = np.random.default_rng(9)
    points = rng.uniform(0.05, 0.95, size=(20, 3))
    params = SolverParams(mu=0.0, lam=0.0)
    solver, _, _ = _solver_with_render(points, params)
    before = solver.skin()
    for _ in range(5):
        solver.step()
    after = solver.skin()
    risen = int(np.count_nonzero(after[:, 2] >= before[:, 2]))
    identical = bool(np.array_equal(after, before))
    assert np.all(after[:, 2] < before[:, 2]), (
        f"render points did not fall: {risen} of {len(after)} did not drop, "
        f"readback identical to the previous one: {identical}. "
        f"before z {before[:3, 2].tolist()}, after z {after[:3, 2].tolist()}"
    )


def test_skin_reads_back_only_render_vertices():
    """R texels, not N. The readback rule is the Python-side ceiling."""
    rng = np.random.default_rng(10)
    points = rng.uniform(0.05, 0.95, size=(7, 3))
    solver, _, _ = _solver_with_render(points)
    out = solver.skin()
    assert out.shape == (7, 3), f"expected (7, 3), got {out.shape}"
    assert BLOCK.n_nodes > 7, "fixture no longer proves anything"


def test_skin_before_attach_render_is_a_clear_error():
    state = make_state(BLOCK.nodes)
    solver = GPUSolver(BLOCK, state.inv_mass, SolverParams())
    try:
        solver.skin()
    except RuntimeError as exc:
        assert "attach_render" in str(exc)
    else:
        raise AssertionError("skin() without bind data must not silently succeed")
