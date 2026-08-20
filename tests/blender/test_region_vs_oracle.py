"""Per-tet stiffness on the GPU, diffed against the numpy oracle.

The multiplier scales mu and lam inside the kernel, so a sign or ordering
slip here shows up as a body that is soft where it should be stiff - which
looks plausible on screen and is invisible without this file.
"""

import gpu
import numpy as np

from _oracle_harness import BLOCK, CUBE, assert_close
from marrow.core.coloring import color_tets
from marrow.core.layout import (
    color_order,
    color_ordered,
    pack_nodes,
    pack_rest,
    pack_scalar,
    pack_tets,
    unpack_vec3,
)
from marrow.core.solver_ref import SolverParams, make_state, precompute, solve_constraints
from marrow.gpu.kernels import SOLVE_IMAGES, SOLVE_PUSH, SOLVE_SRC, build
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-5


def _run_solve(mesh, state, params, h, region):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, offsets = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)

    shader = build("solve", SOLVE_SRC, SOLVE_IMAGES, SOLVE_PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_tets(ordered))
    tex_r = upload(pack_rest(dm_inv, rest_vol))
    tex_fiber = blank(mesh.n_tets)
    tex_torn = blank(mesh.n_tets, fmt="R32F")
    tex_live = blank(mesh.n_nodes, fmt="R32F")
    # Per-tet, so it rides the colour permutation the tets did. Feeding it
    # unpermuted is the bug this whole module exists to catch, and on a
    # uniform region it would pass regardless - hence the varying regions
    # below.
    tex_region = upload(pack_scalar(region[order]), fmt="R32F")

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
        shader.image("fiber", tex_fiber)
        shader.image("region", tex_region)
        shader.uniform_float("h", h)
        shader.uniform_float("tear_threshold", 0.0)
        shader.uniform_float("mu", params.mu)
        shader.uniform_float("lam", params.lam)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        shader.uniform_float("fiber_k", 0.0)
        shader.uniform_float("wave_amp", 0.0)
        shader.uniform_float("wave_len", 1.0)
        shader.uniform_float("wave_speed", 0.0)
        shader.uniform_float("wave_time", 0.0)
        shader.uniform_int("waveform", 0)
        shader.uniform_float("wave_noise", 0.0)
        shader.uniform_float("fiber_bend", 0.0)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), mesh.n_nodes)


def _run_oracle(mesh, state, params, h, region):
    colors = color_tets(mesh.tets, mesh.n_nodes)
    ordered, _ = color_ordered(mesh.tets, colors)
    order = color_order(colors)
    dm_inv, rest_vol = precompute(mesh.nodes, ordered)
    solve_constraints(
        state, ordered, dm_inv, rest_vol, params, h, region=region[order]
    )
    return state.predicted.copy()


def _paired_states(mesh, deform):
    a, b = make_state(mesh.nodes), make_state(mesh.nodes)
    for st in (a, b):
        st.predicted[:] = deform(mesh.nodes.copy())
    return a, b


def _stretch_x(nodes):
    nodes[:, 0] *= 1.3
    return nodes


def test_uniform_region_matches_oracle():
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=1.0e4, lam=1.0e5)
    h = params.dt / params.substeps
    region = np.full(CUBE.n_tets, 0.4)
    a, b = _paired_states(CUBE, _stretch_x)
    assert_close(
        _run_solve(CUBE, a, params, h, region),
        _run_oracle(CUBE, b, params, h, region),
        TOL,
        "uniform region multiplier",
    )


def test_varying_region_matches_oracle():
    """Every tet a different multiplier, so a permutation slip cannot hide."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=1.0e4, lam=1.0e5)
    h = params.dt / params.substeps
    rng = np.random.default_rng(11)
    region = rng.uniform(0.02, 1.0, size=BLOCK.n_tets)

    def squash(nodes):
        nodes[:, 2] *= 0.85
        nodes[:, 0] *= 1.2
        return nodes

    a, b = _paired_states(BLOCK, squash)
    assert_close(
        _run_solve(BLOCK, a, params, h, region),
        _run_oracle(BLOCK, b, params, h, region),
        TOL,
        "per-tet region multiplier",
    )


def test_zero_multiplier_tets_match_oracle():
    """Zero is the value that would divide by zero if the gate were wrong."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=1.0e4, lam=1.0e5)
    h = params.dt / params.substeps
    region = np.ones(BLOCK.n_tets)
    region[::2] = 0.0
    a, b = _paired_states(BLOCK, _stretch_x)
    out = _run_solve(BLOCK, a, params, h, region)
    assert np.all(np.isfinite(out)), "a zero multiplier produced NaN or inf"
    assert_close(
        out,
        _run_oracle(BLOCK, b, params, h, region),
        TOL,
        "region with switched-off tets",
    )


def test_all_ones_region_matches_an_unmultiplied_oracle():
    """mu * 1.0 is exact, so the kernel with the image bound must agree with
    the oracle that never multiplies at all."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=1.0e4, lam=1.0e5)
    h = params.dt / params.substeps
    a, b = _paired_states(BLOCK, _stretch_x)
    colors = color_tets(BLOCK.tets, BLOCK.n_nodes)
    ordered, _ = color_ordered(BLOCK.tets, colors)
    dm_inv, rest_vol = precompute(BLOCK.nodes, ordered)
    solve_constraints(b, ordered, dm_inv, rest_vol, params, h)
    assert_close(
        _run_solve(BLOCK, a, params, h, np.ones(BLOCK.n_tets)),
        b.predicted,
        TOL,
        "region of ones against no region",
    )


def test_soft_tets_recover_less_than_stiff_ones():
    """Not a parity check: the sign of the effect. A multiplier applied the
    wrong way round would still match the oracle if both were wrong."""
    params = SolverParams(gravity=(0.0, 0.0, 0.0), mu=1.0e4, lam=1.0e5)
    h = params.dt / params.substeps
    stretched = _stretch_x(BLOCK.nodes.copy())
    stiff, _ = _paired_states(BLOCK, _stretch_x)
    soft, _ = _paired_states(BLOCK, _stretch_x)
    out_stiff = _run_solve(BLOCK, stiff, params, h, np.ones(BLOCK.n_tets))
    out_soft = _run_solve(BLOCK, soft, params, h, np.full(BLOCK.n_tets, 0.05))
    moved_stiff = np.abs(out_stiff - stretched).max()
    moved_soft = np.abs(out_soft - stretched).max()
    assert moved_stiff > moved_soft, (
        f"stiff cage moved {moved_stiff:.4e}, soft cage {moved_soft:.4e} - "
        "the multiplier is applied the wrong way round"
    )
