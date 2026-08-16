"""Resolution changes the detail, not the physics.

Mass is lumped from material volume at a fixed density (see
marrow.core.tetmesh.MASS_DENSITY), so the same object weighs the same
however finely it is caged. An older version gave every node 1 mass unit,
which made a finer cage a heavier object - a 0.25-to-0.1 unit sphere went
from 461 to 5,104 mass units - and it sagged about ten times further.
"""
import bpy
import gpu
import numpy as np

import marrow
from marrow.blender.session import MarrowSession
from marrow.core.lattice import build_lattice, grid_dims
from marrow.core.solver_ref import SolverParams
from marrow.core.tetmesh import MASS_DENSITY, node_volumes
from marrow.gpu.solver import GPUSolver

gpu.init()


def _column(spacing):
    dims = grid_dims((0, 0, 0), (0.6, 0.6, 1.2), spacing)
    return build_lattice((0, 0, 0), spacing, np.ones(dims, dtype=bool))


def _lumped_inv(mesh):
    mass = node_volumes(mesh.nodes, mesh.tets) * MASS_DENSITY
    inv = 1.0 / mass
    inv[mesh.nodes[:, 2] > 1.2 - 1e-9] = 0.0   # pin the top face
    return inv


def _sag(spacing, frames=24):
    """Mean drop of the free nodes of a hanging column under gravity."""
    mesh = _column(spacing)
    inv = _lumped_inv(mesh)
    solver = GPUSolver(mesh, inv, SolverParams())
    for _ in range(frames):
        solver.step()
    pos = solver.positions()
    free = inv > 0.0
    return float((mesh.nodes[free, 2] - pos[free, 2]).mean())


def test_hanging_column_sag_is_resolution_independent():
    coarse = _sag(0.2)
    fine = _sag(0.1)
    assert coarse > 0.0 and fine > 0.0, "a hanging column must sag"
    # Under uniform per-node mass this ratio was about 10: the fine cage
    # was twice as heavy per cell and sagged ten times further. Lumped
    # mass lands near 1; the remaining gap is solver convergence, which
    # gets worse for finer cages at a fixed iteration budget, not weight.
    ratio = fine / coarse
    assert 0.2 <= ratio <= 2.5, (
        f"sag changed with resolution: coarse {coarse:.5f}, fine {fine:.5f}, "
        f"ratio {ratio:.2f}"
    )


def _tetrahedralised_cube(resolution):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
    return obj


def test_session_total_mass_is_resolution_independent():
    """The session builds inv_mass from lumped volumes: a 2 m cube is
    8 m^3 of material at either resolution, so its total mass is the same.
    Older versions stored inv_mass = 1 for every node, making the fine
    cage of the same object eleven times heavier."""
    totals = {}
    for resolution in (0.5, 0.25):
        obj = _tetrahedralised_cube(resolution)
        s = MarrowSession(obj)
        inv = np.asarray(s.inv_mass, dtype=np.float64)
        assert np.any(inv != inv[0]), "masses must vary with the lattice"
        totals[resolution] = float(np.sum(1.0 / inv))
    assert np.isclose(totals[0.5], totals[0.25], rtol=1e-6), (
        f"total mass changed with resolution: {totals}"
    )
    assert np.isclose(totals[0.5], 8.0 * MASS_DENSITY, rtol=1e-6)
