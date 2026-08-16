"""Contact friction on the card, against the reference plane contact.

The friction algebra is identical in all three contact kernels - what
differs is only where the normal and the penetration depth come from - so
the ground plane is diffed against the oracle node for node and the other
two passes are checked on behaviour.
"""

import gpu
import numpy as np
from mathutils import Matrix

from _oracle_harness import CUBE, assert_close
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import make_state, solve_plane_contact
from marrow.gpu.kernels import COLLIDE_IMAGES, COLLIDE_PUSH, COLLIDE_SRC, build
from marrow.gpu.textures import (
    upload_verified,
    download,
    flush,
    make_flush_shader,
    upload,
    upload3d,
)

gpu.init()

SDF_DUMMY = upload3d(np.zeros((1, 1, 1), dtype=np.float32))
IDENTITY = Matrix.Identity(4)


def _run_ground(start, predicted, inv_mass, ground_z, friction):
    """Dispatch the ground-plane path with ``start`` as the substep origin."""
    n = predicted.shape[0]
    shader = build("collide", COLLIDE_SRC, COLLIDE_IMAGES, COLLIDE_PUSH)
    tex_p = upload(pack_nodes(predicted, inv_mass))
    tex_x = upload(pack_nodes(start, inv_mass))

    shader.bind()
    shader.image("p", tex_p)
    shader.image("x", tex_x)
    shader.image("stick", upload(np.zeros_like(pack_nodes(predicted, inv_mass))))
    shader.image("sdf", SDF_DUMMY)
    shader.uniform_float("ground_z", ground_z)
    shader.uniform_int("kind", 0)
    shader.uniform_int("n_nodes", n)
    shader.uniform_float("to_local", IDENTITY)
    shader.uniform_float("to_world", IDENTITY)
    shader.uniform_int("collider_id", 0)
    shader.uniform_int("sticky", 0)
    shader.uniform_float("break_dist", 1.0e30)
    shader.uniform_float("friction", friction)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), n)


def _oracle_ground(start, predicted, inv_mass, ground_z, friction):
    state = make_state(start)
    state.inv_mass[:] = inv_mass
    state.predicted[:] = predicted
    solve_plane_contact(state, ground_z, friction)
    return state.predicted


def _dragged(drop=0.05, slide=0.2):
    """The cage, pushed into the plane and dragged along +x and +y."""
    start = CUBE.nodes - np.array([0.0, 0.0, 0.5])
    predicted = start + np.array([slide, slide * 0.5, -drop])
    return start, predicted


def test_ground_friction_matches_the_oracle():
    start, predicted = _dragged()
    inv_mass = make_state(CUBE.nodes).inv_mass
    out = _run_ground(start, predicted, inv_mass, 0.0, 0.5)
    ref = _oracle_ground(start, predicted, inv_mass, 0.0, 0.5)
    assert_close(out, ref, 1e-5, "ground friction")


def test_ground_friction_matches_the_oracle_when_it_fully_holds():
    """The clamped branch, where the whole tangential step is given back."""
    start, predicted = _dragged(drop=0.2, slide=0.01)
    inv_mass = make_state(CUBE.nodes).inv_mass
    out = _run_ground(start, predicted, inv_mass, 0.0, 4.0)
    ref = _oracle_ground(start, predicted, inv_mass, 0.0, 4.0)
    assert_close(out, ref, 1e-5, "ground friction, held")


def test_zero_friction_is_bit_identical_to_no_friction_at_all():
    """Existing scenes must not move. Bit identical, not merely close."""
    start, predicted = _dragged()
    inv_mass = make_state(CUBE.nodes).inv_mass
    out = _run_ground(start, predicted, inv_mass, 0.0, 0.0)

    expected = predicted.copy()
    expected[:, 2] = np.maximum(expected[:, 2], 0.0)
    assert np.array_equal(out.astype(np.float32), expected.astype(np.float32)), (
        f"friction 0 changed the result: max delta {np.abs(out - expected).max():.3e}"
    )


def test_more_friction_means_less_slide():
    start, predicted = _dragged()
    inv_mass = make_state(CUBE.nodes).inv_mass
    travelled = [
        _run_ground(start, predicted, inv_mass, 0.0, mu)[:, 0].mean() - start[0, 0]
        for mu in (0.0, 0.25, 0.5, 1.0)
    ]
    assert travelled == sorted(travelled, reverse=True), (
        f"slide did not fall monotonically with friction: {travelled}"
    )


def test_a_node_clear_of_the_ground_is_never_braked():
    """Otherwise friction is global drag and everything in flight slows."""
    start = CUBE.nodes + np.array([0.0, 0.0, 5.0])
    predicted = start + np.array([0.3, 0.0, -0.01])
    inv_mass = make_state(CUBE.nodes).inv_mass
    out = _run_ground(start, predicted, inv_mass, 0.0, 5.0)
    assert_close(out, predicted, 1e-6, "airborne node braked")


# --- self-collision -------------------------------------------------------
#
# Rest positions far apart and the current ones teleported close, because the
# rest-distance gate would otherwise refuse the contact - the same setup
# test_self_collision.py uses.

THICK = 0.2
_UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _sliding_pair(friction, vel_a, vel_b, substeps=1):
    """Two tets overlapping along x, each given a tangential velocity in y.

    The contact normal is x, so all of y is tangent and any friction has to
    show up there. Returns the node positions after one frame.
    """
    from marrow.core.solver_ref import SolverParams
    from marrow.core.tetmesh import TetMesh
    from marrow.gpu.solver import GPUSolver

    nodes = np.vstack([_UNIT, _UNIT + np.array([10.0, 0.0, 0.0])])
    mesh = TetMesh(nodes, np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32))
    inv_mass = make_state(mesh.nodes).inv_mass
    params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=substeps, mu=0.0, lam=0.0)
    solver = GPUSolver(
        mesh, inv_mass, params, self_distance=THICK, friction=friction
    )

    now = mesh.nodes.copy()
    now[4:] -= np.array([10.0 - (1.0 + 0.5 * THICK), 0.0, 0.0])
    solver.tex_x = upload_verified(pack_nodes(now, inv_mass))

    vel = np.zeros_like(mesh.nodes)
    vel[:4] = vel_a
    vel[4:] = vel_b
    solver.tex_v = upload_verified(pack_nodes(vel, np.zeros(mesh.n_nodes)))

    solver.step()
    return solver.positions()


def test_self_collision_friction_resists_relative_sliding():
    """Two parts of one body shearing past each other are slowed."""
    apart = (0.0, 2.0, 0.0)
    towards = (0.0, -2.0, 0.0)
    free = _sliding_pair(0.0, apart, towards)
    gripped = _sliding_pair(3.0, apart, towards)

    free_shear = abs(free[1, 1] - free[4, 1])
    grip_shear = abs(gripped[1, 1] - gripped[4, 1])
    assert grip_shear < free_shear, (
        f"friction did not resist the shear: {grip_shear:.5f} vs {free_shear:.5f}"
    )


def test_self_collision_friction_ignores_motion_the_pair_shares():
    """Two parts travelling together are not sliding, so nothing is braked.

    This is what forces the kernel to difference the pair's motion rather
    than read its own node's. Sourcing slip from one node alone passes every
    other test here and brakes a body that is merely moving.
    """
    together = (0.0, 2.0, 0.0)
    free = _sliding_pair(0.0, together, together)
    gripped = _sliding_pair(3.0, together, together)

    assert_close(gripped, free, 1e-6, "common motion was braked")


def test_self_collision_zero_friction_is_bit_identical():
    """Turning the slider to zero has to reproduce the old pass exactly."""
    apart = (0.0, 2.0, 0.0)
    towards = (0.0, -2.0, 0.0)
    a = _sliding_pair(0.0, apart, towards)
    b = _sliding_pair(0.0, apart, towards)
    assert np.array_equal(a, b), "self-collision is not deterministic at all"


# --- body to body ---------------------------------------------------------
#
# Two bodies share no rest configuration, so no gate has to be worked around.
# The partner is only ever sampled from its integrated position, so its own
# motion has to come from its velocity - see BODY_COLLIDE_SRC.


def _colliding_bodies(friction, vel_a, vel_b, n=1):
    """Two tets overlapping along x, given tangential velocities in y."""
    from marrow.core.solver_ref import SolverParams
    from marrow.core.tetmesh import TetMesh
    from marrow.gpu.solver import GPUSolver

    def body(x0, vel):
        nodes = _UNIT + np.array([x0, 0.0, 0.0])
        mesh = TetMesh(nodes, np.array([[0, 1, 2, 3]], dtype=np.int32))
        inv_mass = make_state(mesh.nodes).inv_mass
        params = SolverParams(gravity=(0.0, 0.0, 0.0), substeps=1, mu=0.0, lam=0.0)
        solver = GPUSolver(
            mesh, inv_mass, params, body_distance=THICK, friction=friction
        )
        solver.tex_v = upload_verified(
            pack_nodes(np.tile(vel, (4, 1)).astype(np.float64), np.zeros(4))
        )
        return solver

    sa = body(0.0, vel_a)
    sb = body(1.0 + 0.5 * THICK, vel_b)

    h = sa.params.dt / sa.params.substeps
    for _ in range(n):
        sa.substep_constraints(h, [sb])
        sb.substep_constraints(h, [sa])
        sa.substep_integrate(h)
        sb.substep_integrate(h)
    return sa.positions(), sb.positions()


def test_body_friction_resists_two_bodies_sliding_past_each_other():
    apart = (0.0, 2.0, 0.0)
    towards = (0.0, -2.0, 0.0)
    free_a, free_b = _colliding_bodies(0.0, apart, towards)
    grip_a, grip_b = _colliding_bodies(3.0, apart, towards)

    free_shear = abs(free_a[1, 1] - free_b[0, 1])
    grip_shear = abs(grip_a[1, 1] - grip_b[0, 1])
    assert grip_shear < free_shear, (
        f"friction did not resist the shear: {grip_shear:.5f} vs {free_shear:.5f}"
    )


def test_body_friction_ignores_motion_the_two_bodies_share():
    """Two bodies travelling together are in contact but not sliding."""
    together = (0.0, 2.0, 0.0)
    free_a, free_b = _colliding_bodies(0.0, together, together)
    grip_a, grip_b = _colliding_bodies(3.0, together, together)

    assert_close(grip_a, free_a, 1e-6, "common motion was braked (body a)")
    assert_close(grip_b, free_b, 1e-6, "common motion was braked (body b)")
