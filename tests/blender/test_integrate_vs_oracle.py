import gpu
import numpy as np

from _oracle_harness import CUBE, assert_close
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state
from marrow.gpu.kernels import INTEGRATE_SRC, build
from marrow.gpu.textures import blank, download, flush, make_flush_shader, upload

gpu.init()

TOL = 1e-5

IMAGES = [
    ("RGBA32F", "FLOAT_2D", "x", {"READ", "WRITE"}),
    ("RGBA32F", "FLOAT_2D", "p", {"READ"}),
    ("RGBA32F", "FLOAT_2D", "v", {"READ", "WRITE"}),
    ("R32F", "FLOAT_2D", "mark", {"READ"}),
]
PUSH = [("FLOAT", "h"), ("FLOAT", "damping"), ("INT", "n_nodes"),
        ("FLOAT", "max_vel"), ("INT", "kinematic")]


def _run_integrate(state, predicted, h, damping, kinematic=0):
    n = state.nodes.shape[0]
    shader = build("integrate", INTEGRATE_SRC, IMAGES, PUSH)

    tex_x = upload(pack_nodes(state.nodes, state.inv_mass))
    tex_p = upload(pack_nodes(predicted, state.inv_mass))
    tex_v = upload(pack_nodes(state.velocities, np.zeros(n)))
    # Zeroed marks: the oracle has no contact passes, so nothing is clamped.
    tex_mark = blank(n, fmt="R32F")

    shader.bind()
    shader.image("x", tex_x)
    shader.image("p", tex_p)
    shader.image("v", tex_v)
    shader.image("mark", tex_mark)
    shader.uniform_float("h", h)
    shader.uniform_float("damping", damping)
    shader.uniform_int("n_nodes", n)
    # The oracle has no velocity clamp; keep it disabled so the two agree.
    shader.uniform_float("max_vel", 0.0)
    # Static pins by default: the integrator refuses to move a pinned node
    # whatever p says. 1 makes it a pass-through for them, which is what a
    # pin driven by the animation needs.
    shader.uniform_int("kinematic", int(kinematic))
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


def test_integrate_carries_a_kinematic_pin_through():
    """The other half of the guard above. A pin driven by the animation has
    its target written into p by the attachment pass, so the integrator has
    to let that through - otherwise it is discarded every substep and the
    pin never moves. It still gains no velocity: it is driven, not
    simulated, and predict reads no velocity for a pinned node.
    """
    h = 1 / 240
    pinned = np.array([2], dtype=np.int32)
    state = make_state(CUBE.nodes, pinned=pinned)
    predicted = CUBE.nodes + 0.5

    gpu_x, gpu_v = _run_integrate(state, predicted, h, 1.0, kinematic=1)
    assert np.allclose(gpu_x[pinned], predicted[pinned], atol=TOL)
    assert np.allclose(gpu_v[pinned], 0.0, atol=TOL)
