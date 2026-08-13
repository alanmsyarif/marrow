import gpu
import numpy as np
from mathutils import Matrix

from _oracle_harness import CUBE
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state
from marrow.gpu.kernels import COLLIDE_IMAGES, COLLIDE_PUSH, COLLIDE_SRC, build
from marrow.gpu.textures import (
    download,
    flush,
    make_flush_shader,
    upload,
    upload3d,
)

gpu.init()

# Shared with the solver. Three copies of these lists used to exist, and
# adding the sdf image broke the two that were not the real one.
IMAGES, PUSH = COLLIDE_IMAGES, COLLIDE_PUSH
SDF_DUMMY = upload3d(np.zeros((1, 1, 1), dtype=np.float32))
IDENTITY = Matrix.Identity(4)


def _run_collide(positions, inv_mass, ground_z, ground_on):
    """Ground plane only. A disabled ground now means no dispatch at all,
    which is how GPUSolver decides it - there is no ground_on uniform."""
    n = positions.shape[0]
    shader = build("collide", COLLIDE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(positions, inv_mass))

    if ground_on:
        shader.bind()
        shader.image("p", tex_p)
        shader.image("stick", upload(np.zeros_like(pack_nodes(positions, inv_mass))))
        shader.image("sdf", SDF_DUMMY)
        shader.uniform_float("ground_z", ground_z)
        shader.uniform_int("kind", 0)
        shader.uniform_int("n_nodes", n)
        shader.uniform_float("to_local", IDENTITY)
        shader.uniform_float("to_world", IDENTITY)
        shader.uniform_int("collider_id", 0)
        shader.uniform_int("sticky", 0)
        shader.uniform_float("break_dist", 1.0e30)
        gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    # Without this the readback intermittently returns the pre-dispatch
    # contents. Built per call rather than cached, so no GPU object outlives
    # the context.
    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), n)


def test_nodes_below_the_ground_are_lifted_onto_it():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.all(out[:, 2] >= -1e-6), (
        f"lift failed. input z {sorted(set(sunk[:, 2].tolist()))}, "
        f"output z {sorted(set(round(float(z), 5) for z in out[:, 2]))}"
    )


def test_nodes_above_the_ground_are_untouched():
    state = make_state(CUBE.nodes)
    lifted = CUBE.nodes + np.array([0.0, 0.0, 5.0])
    out = _run_collide(lifted, state.inv_mass, 0.0, True)
    assert np.allclose(out, lifted, atol=1e-6), (
        f"above-ground nodes moved. max delta {float(np.abs(out - lifted).max()):.3e}"
    )


def test_horizontal_position_is_never_changed():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.allclose(out[:, :2], sunk[:, :2], atol=1e-6), (
        f"horizontal moved. max delta {float(np.abs(out[:, :2] - sunk[:, :2]).max()):.3e}"
    )


def test_disabled_ground_is_a_noop():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, False)
    assert np.allclose(out, sunk, atol=1e-6), (
        f"disabled ground still moved nodes. max delta {float(np.abs(out - sunk).max()):.3e}"
    )


def test_pinned_nodes_are_not_pushed_by_the_ground():
    """A pin outranks a collider: the user put it there deliberately."""
    state = make_state(CUBE.nodes, pinned=np.array([0], dtype=np.int32))
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 0.0, True)
    assert np.allclose(out[0], sunk[0], atol=1e-6), (
        f"pinned node moved: {out[0].tolist()} from {sunk[0].tolist()}"
    )


def test_ground_height_is_respected():
    state = make_state(CUBE.nodes)
    sunk = CUBE.nodes - np.array([0.0, 0.0, 2.0])
    out = _run_collide(sunk, state.inv_mass, 1.5, True)
    assert np.all(out[:, 2] >= 1.5 - 1e-6), (
        f"ground_z=1.5 not applied. input z {sorted(set(sunk[:, 2].tolist()))}, "
        f"output z {sorted(set(round(float(z), 5) for z in out[:, 2]))}"
    )
