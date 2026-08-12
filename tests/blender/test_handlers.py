import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.session import MarrowSession


def _tetrahedralised_cube():
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
    return obj


def _positions(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def _baked_session(obj, start=1, end=10):
    session = MarrowSession(obj)
    session.bake(start, end)
    handlers.SESSIONS[obj.name] = session
    return session


def test_register_handler_is_idempotent():
    _tetrahedralised_cube()
    handlers.unregister_handler()
    handlers.register_handler()
    handlers.register_handler()
    count = sum(
        1 for h in bpy.app.handlers.frame_change_post if h is handlers.on_frame_change
    )
    assert count == 1, f"handler registered {count} times"
    handlers.unregister_handler()


def test_scrubbing_to_a_baked_frame_updates_the_mesh():
    obj = _tetrahedralised_cube()
    rest = _positions(obj)
    _baked_session(obj)
    handlers.register_handler()
    try:
        bpy.context.scene.frame_set(10)
        moved = _positions(obj)
        assert not np.allclose(moved, rest, atol=1e-6), (
            "frame change did not write the cached frame into the mesh"
        )
        assert np.all(np.isfinite(moved))
    finally:
        handlers.unregister_handler()


def test_scrubbing_outside_the_baked_range_leaves_the_mesh_alone():
    obj = _tetrahedralised_cube()
    _baked_session(obj, 1, 3)
    handlers.register_handler()
    try:
        bpy.context.scene.frame_set(3)
        at_three = _positions(obj)
        bpy.context.scene.frame_set(40)
        assert np.allclose(_positions(obj), at_three, atol=1e-9), (
            "an unbaked frame must leave the mesh where it was"
        )
    finally:
        handlers.unregister_handler()


def test_unregister_removes_only_marrows_handler():
    _tetrahedralised_cube()

    def someone_elses_handler(scene, depsgraph=None):
        pass

    bpy.app.handlers.frame_change_post.append(someone_elses_handler)
    handlers.register_handler()
    handlers.unregister_handler()
    try:
        assert someone_elses_handler in bpy.app.handlers.frame_change_post, (
            "unregister must not remove other addons' handlers"
        )
        assert handlers.on_frame_change not in bpy.app.handlers.frame_change_post
    finally:
        bpy.app.handlers.frame_change_post.remove(someone_elses_handler)


def test_unregister_frees_gpu_state():
    """Module-level state outlives the GPU context, so it must be released.

    A GPUShader still referenced when Blender tears the context down crashes
    at shutdown - measured, EXCEPTION_ACCESS_VIOLATION. SESSIONS is module
    level, so unregister has to empty it.
    """
    obj = _tetrahedralised_cube()
    _baked_session(obj, 1, 2)
    assert handlers.SESSIONS, "fixture did not register a session"
    handlers.unregister_handler()
    assert not handlers.SESSIONS, "unregister must drop every session"


def test_missing_object_is_skipped_not_fatal():
    obj = _tetrahedralised_cube()
    _baked_session(obj, 1, 3)
    handlers.SESSIONS["no_such_object"] = handlers.SESSIONS[obj.name]
    handlers.register_handler()
    try:
        bpy.context.scene.frame_set(2)  # must not raise
    finally:
        handlers.unregister_handler()
