import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.layout import color_ordered, pack_nodes, pack_rest, pack_tets, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state, precompute, solve_constraints
from marrow.gpu.kernels import SOLVE_SRC, build
from marrow.gpu.textures import download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-5  # float32 across a full constraint projection on a unit-scale cage

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "rest", {"READ"}),
]
PUSH = [
    ("FLOAT", "h"),
    ("FLOAT", "mu"),
    ("FLOAT", "lam"),
    ("INT", "color_begin"),
    ("INT", "color_end"),
]


def _run_solve(mesh, state, params, h):
    """One GPU substep of constraint solving, colour by colour."""
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, offsets = color_ordered(mesh.tets, colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)

    shader = build("solve", SOLVE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_tets(ordered))
    tex_r = upload(pack_rest(dm_inv, rest_vol))

    for c in range(len(offsets) - 1):
        begin, end = int(offsets[c]), int(offsets[c + 1])
        if end <= begin:
            continue
        shader.bind()
        shader.image("p", tex_p)
        shader.image("tets", tex_t)
        shader.image("rest", tex_r)
        shader.uniform_float("h", h)
        shader.uniform_float("mu", params.mu)
        shader.uniform_float("lam", params.lam)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), mesh.n_nodes)


def _run_oracle(mesh, state, params, h):
    """Same substep on the CPU, over the same colour-ordered tets."""
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    solve_constraints(state, ordered, dm_inv, rest_vol, params, h)
    return state.predicted.copy()


def _paired_states(mesh, deform):
    """Two identical states, one for each side of the comparison."""
    a, b = make_state(mesh.nodes), make_state(mesh.nodes)
    for st in (a, b):
        st.predicted[:] = deform(mesh.nodes.copy())
    return a, b


def test_solve_matches_oracle_on_a_stretched_cube():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps

    def stretch(nodes):
        nodes[:, 0] *= 1.3
        return nodes

    gpu_state, cpu_state = _paired_states(CUBE, stretch)
    assert_close(
        _run_solve(CUBE, gpu_state, params, h),
        _run_oracle(CUBE, cpu_state, params, h),
        TOL,
        "solve on a stretched cube",
    )


def test_solve_matches_oracle_on_a_squashed_block():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps

    def squash(nodes):
        nodes[:, 2] *= 0.7
        return nodes

    gpu_state, cpu_state = _paired_states(BLOCK, squash)
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h),
        _run_oracle(BLOCK, cpu_state, params, h),
        TOL,
        "solve on a squashed block",
    )


def test_solve_matches_oracle_with_pinned_nodes():
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps
    pinned = np.array([0, 1, 2], dtype=np.int32)

    gpu_state = make_state(BLOCK.nodes, pinned=pinned)
    cpu_state = make_state(BLOCK.nodes, pinned=pinned)
    for st in (gpu_state, cpu_state):
        st.predicted[:] = BLOCK.nodes * 1.1

    out = _run_solve(BLOCK, gpu_state, params, h)
    assert_close(out, _run_oracle(BLOCK, cpu_state, params, h), TOL, "solve with pins")
    assert np.allclose(out[pinned], BLOCK.nodes[pinned] * 1.1, atol=TOL), (
        "pinned nodes must not be moved by a constraint projection"
    )


def test_solve_at_rest_matches_the_oracle_residual():
    """At rest the GPU must reproduce the oracle's residual, not beat it.

    The rest state is stress-free in energy but XPBD projects the two
    constraints sequentially, so one substep leaves a real Gauss-Seidel
    residual. Measured on this cube at h = 1/240, the oracle's own drift is
    1.61e-3. An earlier version of this test asserted an invented 1e-3
    absolute bound and failed - it was demanding the GPU be more correct than
    the thing validating it. Parity is the assertion; the loose bound below
    only catches the oracle itself regressing.
    """
    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    h = params.dt / params.substeps
    gpu_state, cpu_state = _paired_states(CUBE, lambda nodes: nodes)

    out = _run_solve(CUBE, gpu_state, params, h)
    ref = _run_oracle(CUBE, cpu_state, params, h)
    assert_close(out, ref, TOL, "solve at rest")

    residual = float(np.abs(ref - CUBE.nodes).max())
    assert residual < 1e-2, (
        f"oracle rest residual {residual:.3e} is far above the measured "
        f"1.61e-3; the constraint formulation has regressed"
    )


def test_zero_stiffness_is_a_noop():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0)
    h = params.dt / params.substeps
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes + 0.1
    out = _run_solve(CUBE, state, params, h)
    assert np.allclose(out, CUBE.nodes + 0.1, atol=TOL)
