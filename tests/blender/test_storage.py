import bpy
import numpy as np

from marrow.blender.storage import (
    read_bind,
    read_tetmesh,
    write_bind,
    write_tetmesh,
)
from marrow.core.lattice import build_lattice
from marrow.core.coloring import color_tets

MESH = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def _empty_mesh(name="cage"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.data.meshes.new(name)


def test_tetmesh_round_trip_is_exact():
    mesh = _empty_mesh()
    colors = color_tets(MESH.tets, MESH.n_nodes)
    write_tetmesh(mesh, MESH, colors)

    out, out_colors = read_tetmesh(mesh)
    assert out.n_nodes == MESH.n_nodes
    assert out.n_tets == MESH.n_tets
    assert np.allclose(out.nodes, MESH.nodes, atol=1e-6)
    assert np.array_equal(out.tets, MESH.tets)
    assert np.array_equal(out_colors, colors)
    out.validate()


def test_bind_round_trip_is_exact():
    mesh = _empty_mesh("render")
    n = 12
    mesh.from_pydata([(float(i), 0.0, 0.0) for i in range(n)], [], [])
    mesh.update()

    rng = np.random.default_rng(3)
    idx = rng.integers(0, MESH.n_tets, size=n).astype(np.int32)
    w = rng.random((n, 4))
    w = w / w.sum(axis=1, keepdims=True)

    write_bind(mesh, idx, w)
    out_idx, out_w = read_bind(mesh)
    assert np.array_equal(out_idx, idx)
    assert np.allclose(out_w, w, atol=1e-6)


def test_read_tetmesh_on_bare_mesh_raises():
    mesh = _empty_mesh("bare")
    try:
        read_tetmesh(mesh)
    except KeyError as exc:
        assert "marrow_tets" in str(exc)
    else:
        raise AssertionError("expected KeyError on a mesh with no Marrow data")


def test_empty_tetmesh_round_trips():
    from marrow.core.tetmesh import TetMesh

    mesh = _empty_mesh("empty")
    empty = TetMesh(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32))
    write_tetmesh(mesh, empty, np.zeros(0, dtype=np.int32))
    out, colors = read_tetmesh(mesh)
    assert out.n_tets == 0
    assert colors.shape == (0,)
