"""GPU-versus-oracle parity for the attachment pass.

The kernel is a transcription of marrow.core.solver_ref.solve_attachment;
without this diff a sign error in the GLSL is indistinguishable from a
sign error in the numpy.
"""

import gpu
import numpy as np

from _oracle_harness import BLOCK, assert_close
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import attach_compliance, make_state, solve_attachment
from marrow.gpu.kernels import ATTACH_IMAGES, ATTACH_PUSH, ATTACH_SRC, build
from marrow.gpu.textures import download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-6  # float32, one diagonal projection on a unit-scale cage


def _run_attach(state, targets, compliance, h):
    shader = build("attach", ATTACH_SRC, ATTACH_IMAGES, ATTACH_PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_t = upload(pack_nodes(np.asarray(targets), np.zeros(targets.shape[0])))

    shader.bind()
    shader.image("p", tex_p)
    shader.image("target", tex_t)
    shader.uniform_float("h", h)
    shader.uniform_float("compliance", compliance)
    shader.uniform_int("n_nodes", state.predicted.shape[0])
    gpu.compute.dispatch(shader, (state.predicted.shape[0] + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), state.predicted.shape[0])


def _oracle(state, targets, compliance, h):
    solve_attachment(state, targets, compliance, h)
    return state.predicted.copy()


def _targets(mesh, seed):
    """Targets that are not axis-aligned, so a channel mix-up fails."""
    rng = np.random.default_rng(seed)
    return mesh.nodes + rng.uniform(-0.5, 0.5, mesh.nodes.shape)


def test_attach_matches_the_oracle_at_half_stiffness():
    h = 1.0 / 240.0
    compliance = attach_compliance(0.5, 1.0 / 24.0)
    targets = _targets(BLOCK, 11)

    state = make_state(BLOCK.nodes)
    state.predicted[:] = BLOCK.nodes * 1.1
    gpu_out = _run_attach(state, targets, compliance, h)

    state2 = make_state(BLOCK.nodes)
    state2.predicted[:] = BLOCK.nodes * 1.1
    assert_close(gpu_out, _oracle(state2, targets, compliance, h), TOL,
                 "attach at half stiffness")


def test_attach_at_zero_compliance_snaps_like_the_oracle():
    h = 1.0 / 240.0
    targets = _targets(BLOCK, 12)

    state = make_state(BLOCK.nodes)
    state.predicted[:] = BLOCK.nodes + 0.3
    gpu_out = _run_attach(state, targets, 0.0, h)

    state2 = make_state(BLOCK.nodes)
    state2.predicted[:] = BLOCK.nodes + 0.3
    assert_close(gpu_out, _oracle(state2, targets, 0.0, h), TOL,
                 "attach snap")
    assert_close(gpu_out, targets, TOL, "attach snap lands on the targets")


def test_attach_leaves_pinned_nodes_where_they_are():
    h = 1.0 / 240.0
    pinned = np.array([0, 1, 2], dtype=np.int32)
    targets = _targets(BLOCK, 13)

    state = make_state(BLOCK.nodes, pinned=pinned)
    state.predicted[:] = BLOCK.nodes
    gpu_out = _run_attach(state, targets, 0.0, h)

    state2 = make_state(BLOCK.nodes, pinned=pinned)
    state2.predicted[:] = BLOCK.nodes
    assert_close(gpu_out, _oracle(state2, targets, 0.0, h), TOL,
                 "attach with pins")
    assert np.allclose(gpu_out[pinned], BLOCK.nodes[pinned], atol=TOL), (
        "a pin outranks the armature"
    )


def test_attach_inverse_mass_weights_match_the_oracle():
    """Non-uniform inverse mass: heavier nodes are pulled less. The kernel
    reads the weight from the packed texel, so a pack mistake shows here."""
    h = 1.0 / 240.0
    compliance = attach_compliance(0.8, 1.0 / 24.0)
    targets = _targets(BLOCK, 14)

    rng = np.random.default_rng(5)
    inv_mass = rng.uniform(0.2, 3.0, BLOCK.n_nodes)

    state = make_state(BLOCK.nodes)
    state.inv_mass[:] = inv_mass
    state.predicted[:] = BLOCK.nodes
    gpu_out = _run_attach(state, targets, compliance, h)

    state2 = make_state(BLOCK.nodes)
    state2.inv_mass[:] = inv_mass
    state2.predicted[:] = BLOCK.nodes
    assert_close(gpu_out, _oracle(state2, targets, compliance, h), TOL,
                 "attach with non-uniform mass")
