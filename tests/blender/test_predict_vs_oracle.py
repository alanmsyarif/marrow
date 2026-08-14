import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close, oracle_predict
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.kernels import PREDICT_SRC, build
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 1e-6

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "x", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "v", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "p", {"WRITE"}),
    ("R32F", "FLOAT_2D", "mark", {"WRITE"}),
]
PUSH = [("FLOAT", "h"), ("VEC3", "gravity"), ("INT", "n_nodes")]


def _run_predict(state, params, h):
    shader = build("predict", PREDICT_SRC, IMAGES, PUSH)
    n = state.nodes.shape[0]

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    tex_p = blank(n)
    tex_mark = blank(n, fmt="R32F")

    shader.bind()
    shader.image("x", tex_x)
    shader.image("v", tex_v)
    shader.image("p", tex_p)
    shader.image("mark", tex_mark)
    shader.uniform_float("h", h)
    shader.uniform_float("gravity", tuple(params.gravity))
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), n)


def test_predict_matches_the_oracle_from_rest():
    params = SolverParams()
    state = make_state(CUBE.nodes)
    h = params.dt / params.substeps
    assert_close(_run_predict(state, params, h), oracle_predict(state, params, h),
                 TOL, "predict from rest")


def test_predict_matches_the_oracle_with_velocity():
    params = SolverParams()
    state = make_state(CUBE.nodes)
    rng = np.random.default_rng(2)
    state.velocities[:] = rng.uniform(-2.0, 2.0, size=state.nodes.shape)
    h = params.dt / params.substeps
    assert_close(_run_predict(state, params, h), oracle_predict(state, params, h),
                 TOL, "predict with velocity")


def test_predict_leaves_pinned_nodes_where_they_are():
    params = SolverParams()
    state = make_state(CUBE.nodes, pinned=np.array([0, 3], dtype=np.int32))
    state.velocities[:] = 5.0
    h = params.dt / params.substeps
    out = _run_predict(state, params, h)
    assert np.allclose(out[[0, 3]], CUBE.nodes[[0, 3]], atol=TOL), (
        "a zero-inverse-mass node must not be integrated"
    )


def test_predict_does_not_write_past_the_node_count():
    """The bounds check is what makes a rounded-up dispatch safe."""
    params = SolverParams()
    state = make_state(CUBE.nodes)
    n = state.nodes.shape[0]
    shader = build("predict", PREDICT_SRC, IMAGES, PUSH)

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    tex_p = blank(n)
    tex_mark = blank(n, fmt="R32F")

    shader.bind()
    shader.image("x", tex_x)
    shader.image("v", tex_v)
    shader.image("p", tex_p)
    shader.image("mark", tex_mark)
    shader.uniform_float("h", 1.0)
    shader.uniform_float("gravity", (0.0, 0.0, -9.81))
    shader.uniform_int("n_nodes", n)
    gpu.compute.dispatch(shader, 4, 1, 1)  # 256 threads for 8 nodes

    flush(make_flush_shader("RGBA32F"), tex_p)
    flat = download(tex_p).reshape(-1, 4)
    assert np.all(flat[n:] == 0.0), "kernel wrote past n_nodes"
