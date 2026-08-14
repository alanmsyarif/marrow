"""Renaming a tetrahedralised object must not kill its simulation.

The cage keeps its original name when the object is renamed, so a name-based
lookup silently loses it - and a session keyed by the old name either leaks
or, if an unrelated object reuses the name, writes another body's cached
frames into the wrong mesh. These tests pin the rename-safe cage lookup and
the stale-session cleanup in the frame handler.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.session import find_cage


def _cube():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 30
    return obj


def _positions(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def test_find_cage_survives_a_rename():
    obj = _cube()
    obj.name = "Jelly"
    assert find_cage(obj) is not None, "renaming the object lost its cage"


def test_detetrahedralize_survives_a_rename():
    obj = _cube()
    obj.name = "Jelly"
    assert bpy.ops.marrow.detetrahedralize() == {"FINISHED"}
    assert find_cage(obj) is None, "the cage was left behind"
    assert obj.data.attributes.get("marrow_rest") is None, "attributes left"


def test_live_simulation_survives_a_rename():
    obj = _cube()
    old_name = obj.name
    obj.name = "Jelly"
    rest = _positions(obj)
    try:
        for frame in range(1, 6):
            bpy.context.scene.frame_set(frame)
        moved = _positions(obj)
        assert not np.allclose(moved, rest, atol=1e-6), (
            "the simulation died when the object was renamed"
        )
        assert handlers.SESSIONS.get(obj.name) is not None, (
            "no session was built under the new name"
        )
        assert old_name not in handlers.SESSIONS, (
            "the stale session under the old name was not dropped"
        )
    finally:
        handlers.unregister_handler()


def test_a_reused_name_does_not_get_the_old_bodys_frames():
    obj = _cube()
    old_name = obj.name
    rest = _positions(obj)
    try:
        for frame in range(1, 4):
            bpy.context.scene.frame_set(frame)
        assert not np.allclose(_positions(obj), rest, atol=1e-6), (
            "fixture did not simulate"
        )

        # Another object takes over the old name. It has no Marrow data, so
        # the old body's session must be dropped, never written into it.
        obj.name = "Jelly"
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        impostor = bpy.context.active_object
        assert impostor.name == old_name, "fixture did not reuse the name"
        impostor_rest = _positions(impostor)

        bpy.context.scene.frame_set(4)
        assert np.allclose(_positions(impostor), impostor_rest, atol=1e-9), (
            "another body's cached frames were written into the wrong mesh"
        )
        assert old_name not in handlers.SESSIONS, (
            "the session keyed by the reused name was not dropped"
        )
    finally:
        handlers.unregister_handler()
