import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.layout import (
    color_order,
    color_ordered,
    pack_fiber,
    pack_nodes,
    pack_rest,
    pack_tets,
    unpack_vec3,
)
from marrow.core.solver_ref import SolverParams, make_state, precompute, solve_constraints
from marrow.gpu.kernels import SOLVE_SRC, build
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-5

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "tets", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "rest", {"READ"}),
    ("R32F", "FLOAT_2D", "torn", {"READ", "WRITE"}),
    ("R32F", "FLOAT_2D", "live", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "fiber", {"READ"}),
]
PUSH = [
    ("FLOAT", "h"),
    ("FLOAT", "mu"),
    ("FLOAT", "lam"),
    ("FLOAT", "tear_threshold"),
    ("INT", "color_begin"),
    ("INT", "color_end"),
    ("FLOAT", "fiber_k"),
    ("FLOAT", "wave_amp"),
    ("FLOAT", "wave_len"),
    ("FLOAT", "wave_speed"),
    ("FLOAT", "wave_time"),
    ("INT", "waveform"),
]


def _fibers(n_tets, phase=0.75):
    """Every tet along +X at one phase, so the whole cage fires together."""
    fiber = np.zeros((n_tets, 4), dtype=np.float64)
    fiber[:, 0] = 1.0
    fiber[:, 3] = phase
    return fiber


def _run_solve(mesh, state, params, h, fiber, wave_time=0.0, torn=None):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, offsets = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)

    shader = build("solve", SOLVE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_tets(ordered))
    tex_r = upload(pack_rest(dm_inv, rest_vol))
    # Fiber rows are per-tet, so they must ride the same colour permutation
    # the tets did or every tet contracts along its neighbour's direction.
    tex_f = upload(pack_fiber(fiber[order]))
    tex_torn = blank(mesh.n_tets, fmt="R32F") if torn is None else upload(torn, fmt="R32F")
    tex_live = blank(mesh.n_nodes, fmt="R32F")

    for c in range(len(offsets) - 1):
        begin, end = int(offsets[c]), int(offsets[c + 1])
        if end <= begin:
            continue
        shader.bind()
        shader.image("p", tex_p)
        shader.image("tets", tex_t)
        shader.image("rest", tex_r)
        shader.image("torn", tex_torn)
        shader.image("live", tex_live)
        shader.image("fiber", tex_f)
        shader.uniform_float("h", h)
        shader.uniform_float("tear_threshold", 0.0)
        shader.uniform_float("mu", params.mu)
        shader.uniform_float("lam", params.lam)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        shader.uniform_float("fiber_k", params.fiber_k)
        shader.uniform_float("wave_amp", params.wave_amp)
        shader.uniform_float("wave_len", params.wave_len)
        shader.uniform_float("wave_speed", params.wave_speed)
        shader.uniform_float("wave_time", wave_time)
        shader.uniform_int("waveform", params.waveform)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), mesh.n_nodes)


def _run_oracle(mesh, state, params, h, fiber, wave_time=0.0):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    solve_constraints(
        state, ordered, dm_inv, rest_vol, params, h,
        fiber=fiber[order], t=wave_time,
    )
    return state.predicted.copy()


def _paired_states(mesh, deform):
    a, b = make_state(mesh.nodes), make_state(mesh.nodes)
    for st in (a, b):
        st.predicted[:] = deform(mesh.nodes.copy())
    return a, b


def test_fiber_matches_oracle_with_a_square_wave():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.4, wave_len=1.0, wave_speed=0.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    gpu_state, cpu_state = _paired_states(CUBE, lambda n: n)
    assert_close(
        _run_solve(CUBE, gpu_state, params, h, fiber),
        _run_oracle(CUBE, cpu_state, params, h, fiber),
        TOL,
        "fiber square wave on a cube",
    )


def test_fiber_matches_oracle_with_a_smooth_wave_in_motion():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.35, wave_len=0.4, wave_speed=1.5, waveform=0,
    )
    h = params.dt / params.substeps
    fiber = np.zeros((BLOCK.n_tets, 4), dtype=np.float64)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(BLOCK.n_tets, 3))
    fiber[:, :3] = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    fiber[:, 3] = rng.uniform(0.0, 2.0, size=BLOCK.n_tets)

    def squash(nodes):
        nodes[:, 2] *= 0.85
        return nodes

    gpu_state, cpu_state = _paired_states(BLOCK, squash)
    # A non-zero time is the point: it is what drives the phase negative,
    # where fract and % have to agree.
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h, fiber, wave_time=0.37),
        _run_oracle(BLOCK, cpu_state, params, h, fiber, wave_time=0.37),
        TOL,
        "fiber smooth wave mid-travel",
    )


def test_zero_direction_rows_match_the_oracle():
    """Half the cage has no fiber. Both sides must skip exactly those."""
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), fiber_k=1.0e4,
        wave_amp=0.5, wave_len=1.0, wave_speed=0.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(BLOCK.n_tets)
    fiber[::2, :3] = 0.0
    gpu_state, cpu_state = _paired_states(BLOCK, lambda n: n)
    assert_close(
        _run_solve(BLOCK, gpu_state, params, h, fiber),
        _run_oracle(BLOCK, cpu_state, params, h, fiber),
        TOL,
        "fiber with unassigned tets",
    )


def test_zero_fiber_stiffness_is_a_noop():
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=0.0, wave_amp=0.9, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes + 0.1
    out = _run_solve(CUBE, state, params, h, fiber)
    assert np.allclose(out, CUBE.nodes + 0.1, atol=TOL)


def test_a_torn_tet_ignores_its_fiber():
    """Tearing means the material goes slack. Torn muscle must not pull.

    The oracle has no tearing, so this is GPU-only by construction: mark
    every tet torn, turn the isotropic terms off, and nothing may move.
    """
    from marrow.core.layout import pack_scalar

    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), mu=0.0, lam=0.0,
        fiber_k=1.0e4, wave_amp=0.5, wave_len=1.0, waveform=1,
    )
    h = params.dt / params.substeps
    fiber = _fibers(CUBE.n_tets)
    state = make_state(CUBE.nodes)
    state.predicted[:] = CUBE.nodes
    torn = pack_scalar(np.ones(CUBE.n_tets))
    out = _run_solve(CUBE, state, params, h, fiber, torn=torn)
    assert np.allclose(out, CUBE.nodes, atol=TOL), (
        "a torn tet still contracted"
    )
