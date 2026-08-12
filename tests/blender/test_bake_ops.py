import bpy
import numpy as np

import marrow
from marrow.blender import handlers


def _fresh_addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _cube(resolution=0.5):
    _fresh_addon()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    return obj


def _positions(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def test_settings_expose_the_simulation_knobs():
    obj = _cube()
    settings = obj.marrow
    for name, default in (
        ("substeps", 10),
        ("stiffness", 1.0e4),
        ("volume_preservation", 1.0e5),
        ("damping", 0.999),
    ):
        assert hasattr(settings, name), f"missing setting {name}"
        assert np.isclose(getattr(settings, name), default), (
            f"{name} default is {getattr(settings, name)}, expected {default}"
        )
    assert hasattr(settings, "ground_enabled")
    assert hasattr(settings, "ground_z")


def test_bake_without_a_cage_names_the_fix():
    obj = _cube()
    try:
        bpy.ops.marrow.bake()
    except RuntimeError as exc:
        assert "Tetrahedralize" in str(exc), f"unhelpful error: {exc}"
    else:
        raise AssertionError("baking without a cage must report an error")


def test_bake_stores_the_scene_frame_range():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 6

    assert bpy.ops.marrow.bake() == {"FINISHED"}
    session = handlers.SESSIONS.get(obj.name)
    assert session is not None, "bake did not register a session"
    assert session.baked_range == (1, 6)
    handlers.unregister_handler()


def test_scrubbing_after_bake_moves_the_mesh():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    rest = _positions(obj)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 8
    bpy.ops.marrow.bake()
    try:
        scene.frame_set(8)
        moved = _positions(obj)
        assert not np.allclose(moved, rest, atol=1e-6), "mesh did not follow the bake"
        assert np.all(np.isfinite(moved))
        assert moved[:, 2].mean() < rest[:, 2].mean(), "body did not fall"
    finally:
        handlers.unregister_handler()


def test_free_clears_the_session():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    bpy.ops.marrow.bake()
    assert handlers.SESSIONS
    assert bpy.ops.marrow.free() == {"FINISHED"}
    assert not handlers.SESSIONS, "free must drop the session"


def test_substeps_setting_reaches_the_solver():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 4

    obj.marrow.substeps = 2
    bpy.ops.marrow.bake()
    coarse = handlers.SESSIONS[obj.name].frame_positions(4).copy()

    obj.marrow.substeps = 20
    bpy.ops.marrow.bake()
    fine = handlers.SESSIONS[obj.name].frame_positions(4).copy()
    handlers.unregister_handler()

    assert not np.allclose(coarse, fine, atol=1e-6), (
        "the substeps slider had no effect on the bake"
    )


def test_unregister_leaves_no_sessions_behind():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    bpy.ops.marrow.bake()
    marrow.unregister()
    assert not handlers.SESSIONS, (
        "addon unregister must release GPU state; module globals outlive the "
        "GPU context and crash Blender at shutdown otherwise"
    )
    marrow.register()
