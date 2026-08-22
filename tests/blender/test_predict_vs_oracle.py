import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close, oracle_predict
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state
from marrow.gpu.kernels import PREDICT_IMAGES, PREDICT_PUSH, PREDICT_SRC, build
from marrow.gpu.solver import pack_fields
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 1e-6

# The solver's own lists, not a copy: a test that declares its own
# interface stops testing the kernel the addon runs the moment they drift.
IMAGES = PREDICT_IMAGES
PUSH = PREDICT_PUSH


def _run_predict(state, params, h, fields=None):
    shader = build("predict", PREDICT_SRC, IMAGES, PUSH)
    n = state.nodes.shape[0]

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    tex_p = blank(n)
    tex_mark = blank(n, fmt="R32F")
    rows = [] if fields is None else list(fields)
    tex_fields = upload(pack_fields(rows))

    shader.bind()
    shader.image("x", tex_x)
    shader.image("v", tex_v)
    shader.image("p", tex_p)
    shader.image("mark", tex_mark)
    shader.image("fields", tex_fields)
    shader.uniform_float("h", h)
    shader.uniform_float("gravity", tuple(params.gravity))
    shader.uniform_int("n_nodes", n)
    shader.uniform_int("n_fields", len(rows))
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


def _rows():
    """One of each supported kind, placed so every branch does something."""
    from marrow.core.solver_ref import FIELD_FORCE, FIELD_VORTEX, FIELD_WIND

    return [
        (float(FIELD_FORCE), 0.4, -0.3, 0.2, 0.0, 0.0, 1.0, 7.5, 1.5, 0.0),
        (float(FIELD_WIND), 0.0, 0.0, 0.0, 0.3, 0.6, 0.7, -4.0, 0.0, 0.0),
        (float(FIELD_VORTEX), 0.1, 0.1, 0.5, 0.0, 1.0, 0.0, 3.25, 0.5, 0.0),
    ]


def test_predict_matches_the_oracle_with_fields():
    """Force, wind and vortex together, with falloff, in one dispatch. The
    kernel evaluates these where the node is; the oracle does it in float64.
    A sign slip in either would look perfectly plausible on screen."""
    from marrow.core.solver_ref import field_accel

    params = SolverParams()
    state = make_state(CUBE.nodes)
    state.velocities[:] = 0.05
    h = params.dt / params.substeps
    rows = _rows()

    gravity = np.asarray(params.gravity, dtype=np.float64)
    movable = state.inv_mass > 0.0
    want = state.nodes.copy()
    want[movable] += (
        state.velocities[movable] * h
        + (gravity + field_accel(state.nodes, rows)[movable]) * (h * h)
    )
    assert_close(_run_predict(state, params, h, rows), want, TOL, "fields")


def test_a_max_distance_field_reaches_only_part_of_the_cage():
    """The cutoff branch, which a uniformly covered cage would never take."""
    from marrow.core.solver_ref import FIELD_WIND, field_accel

    params = SolverParams(gravity=(0.0, 0.0, 0.0))
    state = make_state(CUBE.nodes)
    h = params.dt / params.substeps
    rows = [(float(FIELD_WIND), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 40.0, 0.0, 0.6)]

    reached = np.linalg.norm(field_accel(state.nodes, rows), axis=1) > 0.0
    assert reached.any() and not reached.all(), (
        "the test cage has to straddle the cutoff or this proves nothing"
    )
    movable = state.inv_mass > 0.0
    want = state.nodes.copy()
    want[movable] += field_accel(state.nodes, rows)[movable] * (h * h)
    assert_close(_run_predict(state, params, h, rows), want, TOL, "max distance")


def test_no_fields_is_bit_identical_to_before_they_existed():
    params = SolverParams()
    state = make_state(CUBE.nodes)
    h = params.dt / params.substeps
    assert np.array_equal(
        _run_predict(state, params, h), _run_predict(state, params, h, [])
    )
