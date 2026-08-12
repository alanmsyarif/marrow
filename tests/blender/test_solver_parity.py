import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.lattice import build_lattice
from marrow.core.layout import color_ordered
from marrow.core.solver_ref import SolverParams, make_state, precompute, step
from marrow.gpu.solver import GPUSolver, MarrowNaNError

gpu.init()

# float32 across 10 frames x 10 substeps on a unit-scale cage. This only holds
# once both sides iterate tets in the same colour order - see _oracle_run.
TOL = 1e-4


def _oracle_run(mesh, params, frames, pinned=None):
    """Run the oracle over colour-ordered tets, exactly as the GPU does.

    XPBD is Gauss-Seidel and therefore order-dependent: a constraint sees the
    positions every earlier constraint already moved. The GPU dispatches
    colour by colour, so the oracle must visit tets in that same order or the
    two are solving subtly different problems.

    An earlier version of this helper passed mesh.tets in its original order
    and the comparison missed by 3.4e-4. That was not float32 - measured, the
    gap was already 3.05e-4 after a single frame, barely grew over 40, was
    identical against a float32-rounded oracle, and shrank as substeps rose
    (6.3e-1 at 2 substeps, 1.2e-4 at 20). Rounding error grows with more
    operations; ordering error shrinks as each correction gets smaller.
    """
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    state = make_state(mesh.nodes, pinned=pinned)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    for _ in range(frames):
        step(state, ordered, dm_inv, rest_vol, params)
    return state.nodes


def _gpu_run(mesh, params, frames, pinned=None):
    state = make_state(mesh.nodes, pinned=pinned)
    solver = GPUSolver(mesh, state.inv_mass, params)
    for _ in range(frames):
        solver.step()
    return solver.positions()


def test_free_fall_matches_the_oracle():
    params = SolverParams(mu=0.0, lam=0.0)
    assert_close(_gpu_run(CUBE, params, 10), _oracle_run(CUBE, params, 10),
                 TOL, "10 frames of free fall")


def test_constrained_block_matches_the_oracle():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    assert_close(_gpu_run(BLOCK, params, 10), _oracle_run(BLOCK, params, 10),
                 TOL, "10 frames of constrained block")


def test_pinned_block_under_gravity_matches_the_oracle():
    params = SolverParams()
    pinned = np.array([0, 1, 2], dtype=np.int32)
    assert_close(_gpu_run(BLOCK, params, 10, pinned), _oracle_run(BLOCK, params, 10, pinned),
                 TOL, "10 frames of pinned block")


def test_pinned_body_settles_rather_than_exploding():
    params = SolverParams()
    pinned = np.arange(4, dtype=np.int32)
    out = _gpu_run(BLOCK, params, 60, pinned)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() < 100.0, "a pinned body drifted absurdly far"


def test_nan_state_is_detected_and_raises():
    """The spec requires a freeze and a refusal, not a quiet cache write.

    Detection lives at the readback boundary, not inside step(): step() must
    not touch PCIe at all, or the readback rule the architecture rests on is
    defeated by its own error check.
    """
    params = SolverParams()
    state = make_state(CUBE.nodes)
    solver = GPUSolver(CUBE, state.inv_mass, params)
    solver.poison_for_test()
    solver.step()
    try:
        solver.positions()
    except MarrowNaNError as exc:
        assert "NaN" in str(exc)
        assert "Substeps" in str(exc), "error must name the knob to change"
    else:
        raise AssertionError("a non-finite state must not pass silently")


def test_dispatch_chain_is_deterministic_at_realistic_scale():
    """The spike proved this at 256 texels. Re-check it at real size.

    There is no barrier API, so ordering between dependent dispatches is the
    driver's to guarantee. Running the same deterministic frame twice must
    give bit-identical results; if it does not, dispatches are racing.
    """
    big = build_lattice(np.zeros(3), 0.1, np.ones((20, 20, 20), dtype=bool))
    params = SolverParams(substeps=2)
    state = make_state(big.nodes)

    first = None
    for _ in range(3):
        solver = GPUSolver(big, state.inv_mass, params)
        solver.step()
        out = solver.positions()
        if first is None:
            first = out
        else:
            assert np.array_equal(first, out), (
                f"non-deterministic across runs at {big.n_nodes} nodes, "
                f"{big.n_tets} tets - dependent dispatches are racing"
            )
    print(f"  barrier check: {big.n_nodes} nodes, {big.n_tets} tets, deterministic")
