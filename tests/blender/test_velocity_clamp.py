"""Velocity clamp: contact-corrected nodes keep at most 0.2 thicknesses per substep.

The reference self-collision caps particle velocity at 0.2 * thickness / h so
fast material cannot tunnel through thin contact features and wad up instead
of folding. Marrow applies the same cap in integrate, but only to nodes the
contact passes marked this substep - a global cap turned every drop into
slow motion, so free fall keeps its speed.

The cap limits the velocity carried into the next predict; the position
corrections of the substep that produced it stand.
"""

import numpy as np

from _oracle_harness import BLOCK
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.core.solver_ref import SolverParams, make_state
from marrow.core.tetmesh import TetMesh
from marrow.gpu.solver import GPUSolver
from marrow.gpu.textures import download, upload

THICK = 0.2
FAST = 100.0

_UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _two_tets(gap):
    """Two disjoint unit tets, the second offset ``gap`` along x."""
    nodes = np.vstack([_UNIT, _UNIT + np.array([gap, 0.0, 0.0])])
    return TetMesh(nodes, np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32))


def _inert(mesh, distance):
    params = SolverParams(
        gravity=(0.0, 0.0, 0.0), substeps=1, mu=0.0, lam=0.0, damping=1.0
    )
    return GPUSolver(mesh, make_state(mesh.nodes).inv_mass, params,
                     self_distance=distance)


def _velocities(solver):
    return unpack_vec3(download(solver.tex_v), solver.n_nodes)


def test_fast_free_motion_is_not_capped_even_with_thickness_active():
    """The slow-motion regression: no contact, no cap.

    The whole block translates rigidly, so the rest-distance gate keeps the
    self-collision pass out of it and no node is marked.
    """
    solver = _inert(BLOCK, THICK)
    vel = np.zeros_like(BLOCK.nodes)
    vel[:, 0] = FAST
    solver.tex_v = upload(pack_nodes(vel, np.zeros(BLOCK.n_nodes)))

    solver.step()
    v = _velocities(solver)
    assert np.allclose(np.linalg.norm(v, axis=1), FAST, rtol=1e-2), (
        f"free fall was capped: speeds {np.linalg.norm(v, axis=1)}"
    )


def test_only_the_node_in_contact_is_capped():
    """A node the self-collision pass corrects loses its speed; its
    uninvolved neighbour, moving at the same velocity, does not.

    -2 m/s covers 1/12 m per substep, under the 0.2 thickness, so the pair
    is inside the contact distance after predict and the pass fires. A
    faster crash would tunnel clean through in this single substep - the
    clamp exists to keep that from happening on every substep after the
    first contact.
    """
    mesh = _two_tets(10.0)
    solver = _inert(mesh, THICK)

    now = mesh.nodes.copy()
    now[4:] -= np.array([10.0 - (1.0 + 0.5 * THICK), 0.0, 0.0])
    inv_mass = make_state(mesh.nodes).inv_mass
    solver.tex_x = upload(pack_nodes(now, inv_mass))
    vel = np.zeros_like(now)
    vel[4:, 0] = -2.0
    solver.tex_v = upload(pack_nodes(vel, np.zeros(mesh.n_nodes)))

    solver.step()
    v = _velocities(solver)
    cap = 0.2 * THICK / solver.params.dt
    assert np.linalg.norm(v[4]) < cap * 1.01, (
        f"contacted node kept {np.linalg.norm(v[4]):.3f} m/s, cap {cap:.3f}"
    )
    assert np.allclose(np.linalg.norm(v[5]), 2.0, rtol=1e-2), (
        "an unmarked node at the same speed must not be capped"
    )


def test_no_contact_thickness_means_no_cap():
    """Same crash, feature off: the clamp must not fire without a thickness."""
    mesh = _two_tets(10.0)
    solver = _inert(mesh, 0.0)

    now = mesh.nodes.copy()
    now[4:] -= np.array([10.0 - (1.0 + 0.5 * THICK), 0.0, 0.0])
    inv_mass = make_state(mesh.nodes).inv_mass
    solver.tex_x = upload(pack_nodes(now, inv_mass))
    vel = np.zeros_like(now)
    vel[4:, 0] = -2.0
    solver.tex_v = upload(pack_nodes(vel, np.zeros(mesh.n_nodes)))

    solver.step()
    v = _velocities(solver)
    assert np.allclose(np.linalg.norm(v[4]), 2.0, rtol=1e-2), (
        f"velocity changed with the feature off: {np.linalg.norm(v[4]):.3f}"
    )
