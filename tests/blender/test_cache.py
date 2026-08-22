"""Bakes that survive closing the file.

The cache used to live only in memory, so reopening a .blend meant playing
the whole shot again. It is written beside the file now, in a
blendcache_<name> folder - the convention Blender's own point caches use.

Only cage nodes are stored; render positions are barycentric combinations of
them and rebuild exactly. Only BAKED caches are written, because a live one
is disposable by design.
"""

import os
import shutil
import tempfile

import bpy
import gpu
import numpy as np

import marrow
from marrow.blender import cache, handlers

gpu.init()

ROOT = os.path.join(tempfile.gettempdir(), "marrow_cache_tests")
FRAMES = 6


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    handlers.free_all()


def _baked_cube(path):
    """A cube baked over a short range, saved to ``path``."""
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 3))
    obj = bpy.context.active_object
    obj.name = "Cube"
    obj.marrow.resolution = 0.5
    obj.marrow.ground_enabled = True
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    assert bpy.ops.marrow.bake() == {"FINISHED"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return obj


def _cleanup(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    folder = os.path.dirname(path)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def test_a_bake_survives_reopening_the_file():
    """The limitation this exists to remove. Reopen and the frames are
    already there, with no simulation run."""
    path = os.path.join(ROOT, "survives", "shot.blend")
    try:
        obj = _baked_cube(path)
        want = handlers.SESSIONS["Cube"].frame_positions(FRAMES).copy()
        assert os.path.exists(cache.path_for(obj)), "nothing was written"

        bpy.ops.wm.open_mainfile(filepath=path)
        assert not handlers.SESSIONS, "the load should have cleared memory"

        reopened = bpy.data.objects["Cube"]
        from marrow.blender.session import MarrowSession

        session = MarrowSession(reopened)
        assert session.baked, "the reopened session should already be baked"
        assert session.baked_range == (1, FRAMES), session.baked_range
        got = session.frame_positions(FRAMES)
        assert np.abs(got - want).max() < 1e-4, (
            f"restored frame differs by {np.abs(got - want).max():.2e}"
        )
    finally:
        _cleanup(path)


def test_an_unsaved_file_writes_nothing():
    """There is nowhere to put it, and guessing at a location would litter."""
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    assert cache.path_for(obj) == ""


def test_a_stale_cache_is_refused():
    """Editing the mesh leaves a sidecar that would unpack into the wrong
    number of vertices. Playing that back would look like a solver fault
    rather than a stale file."""
    path = os.path.join(ROOT, "stale", "shot.blend")
    try:
        obj = _baked_cube(path)
        stored = cache.path_for(obj)
        assert os.path.exists(stored)

        # A different body entirely, under the same name.
        _fresh()
        bpy.ops.wm.open_mainfile(filepath=path)
        victim = bpy.data.objects["Cube"]
        victim.data.vertices.add(1)
        victim.data.update()

        from marrow.blender.session import MarrowSession

        session = MarrowSession(victim)
        assert not session.baked, "a cache from another mesh must be refused"
    finally:
        _cleanup(path)


def test_freeing_removes_the_sidecar():
    """Free means it. Leaving the file would have the next session load back
    the bake that was just discarded."""
    path = os.path.join(ROOT, "freed", "shot.blend")
    try:
        obj = _baked_cube(path)
        stored = cache.path_for(obj)
        assert os.path.exists(stored)
        bpy.context.view_layer.objects.active = obj
        assert bpy.ops.marrow.free() == {"FINISHED"}
        assert not os.path.exists(stored), "the sidecar outlived the bake"
    finally:
        _cleanup(path)


def test_a_live_cache_is_not_written():
    """Live caches are rebuilt as the timeline plays, so writing hundreds of
    megabytes of them on every save would be pure cost."""
    path = os.path.join(ROOT, "live", "shot.blend")
    try:
        _fresh()
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 3))
        obj = bpy.context.active_object
        obj.name = "Cube"
        obj.marrow.resolution = 0.5
        assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
        scene = bpy.context.scene
        scene.frame_start, scene.frame_end = 1, FRAMES
        for f in range(1, FRAMES + 1):
            scene.frame_set(f)
        session = handlers.SESSIONS.get("Cube")
        assert session is not None and not session.baked

        os.makedirs(os.path.dirname(path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=path)
        assert not os.path.exists(cache.path_for(obj)), (
            "a live cache was written to disk"
        )
    finally:
        _cleanup(path)
