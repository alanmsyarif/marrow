import bpy
import numpy as np

from marrow.blender.storage import clear_marrow_data, read_fiber, write_fiber


def _mesh(name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    mesh.update()
    return mesh


def test_fiber_round_trips_through_the_mesh():
    mesh = _mesh("fiber_round_trip")
    fiber = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.25]])
    write_fiber(mesh, fiber)
    out = read_fiber(mesh)
    assert out is not None
    assert out.shape == (2, 4)
    assert np.allclose(out, fiber, atol=1e-6), f"{out} != {fiber}"


def test_a_cage_without_fibers_reads_as_none():
    assert read_fiber(_mesh("fiber_absent")) is None


def test_clear_marrow_data_strips_the_fiber_key():
    mesh = _mesh("fiber_cleared")
    write_fiber(mesh, np.array([[1.0, 0.0, 0.0, 0.0]]))
    clear_marrow_data(mesh)
    assert read_fiber(mesh) is None
