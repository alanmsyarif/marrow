import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state
from marrow.gpu.kernels import INTEGRATE_SRC, build
from marrow.gpu.textures import download, flush, make_flush_shader, upload

gpu.init()

TOL = 1e-5

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "x", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "p", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "v", {"READ", "WRITE"}),
]
PUSH = [("FLOAT", "h"), ("FLOAT", "damping"), ("INT", "n_nodes"),
        ("FLOAT", "max_vel")]


def _run_integrate(state, predicted, h, damping):
    n = state.nodes.shape[0]
    shader = build("integrate", INTEGRATE_SRC, IMAGES, PUSH)

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_p = upload(pack_nodes(predicted, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))

    shader.bind()
    shader.image("x", tex_x)
    shader.image("p", tex_p)
    shader.image("v", tex_v)
    shader.uniform_float("h", h)
    shader.uniform_float("damping", damping)
    shader.uniform_int("n_nodes", n)
    # The oracle has no velocity clamp; keep it disabled so the two agree.
    shader.uniform_float("max_vel", 0.0)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    sync = make_flush_shader("RGBA32F")
    flush(sync, tex_x)
    flush(sync, tex_v)
    return unpack_vec3(download(tex_x), n), unpack_vec3(download(tex_v), n)


def _oracle_integrate(state, predicted, h, damping):
    movable = state.inv_mass > 0.0
    velocities = state.velocities.copy()
    nodes = state.nodes.copy()
    velocities[movable] = (predicted[movable] - nodes[movable]) / h * damping
    nodes[movable] = predicted[movable]
    return nodes, velocities


def test_integrate_matches_the_oracle():
    h, damping = 1 / 240, 0.999
    state = make_state(CUBE.nodes)
    rng = np.random.default_rng(4)
    predicted = CUBE.nodes + rng.uniform(-0.05, 0.05, size=CUBE.nodes.shape)

    gpu_x, gpu_v = _run_integrate(state, predicted, h, damping)
    cpu_x, cpu_v = _oracle_integrate(state, predicted, h, damping)
    assert_close(gpu_x, cpu_x, TOL, "integrate positions")
    assert_close(gpu_v, cpu_v, 1e-3, "integrate velocities")


def test_integrate_scales_velocity_by_damping():
    h = 1 / 240
    state = make_state(CUBE.nodes)
    predicted = CUBE.nodes + 0.01

    _, fast_v = _run_integrate(state, predicted, h, 1.0)
    _, slow_v = _run_integrate(state, predicted, h, 0.5)
    assert np.linalg.norm(slow_v) < np.linalg.norm(fast_v)


def test_integrate_leaves_pinned_nodes_alone():
    h = 1 / 240
    pinned = np.array([2], dtype=np.int32)
    state = make_state(CUBE.nodes, pinned=pinned)
    predicted = CUBE.nodes + 0.5

    gpu_x, gpu_v = _run_integrate(state, predicted, h, 1.0)
    assert np.allclose(gpu_x[pinned], CUBE.nodes[pinned], atol=TOL)
    assert np.allclose(gpu_v[pinned], 0.0, atol=TOL)
