"""Live simulation: on by default, restarts from the start frame.

Live is the default, so tetrahedralizing is enough - the handler is armed and
the first frame change builds the session. Returning to the start frame
restarts and re-reads the sliders, so tweaking Stiffness and replaying is the
whole edit loop. A baked cache is exempt: it plays back untouched.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers


def _cube(resolution=0.5):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    bpy.ops.marrow.tetrahedralize()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 30
    return obj


def _positions(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def test_live_is_on_by_default_after_tetrahedralize():
    obj = _cube()
    assert obj.marrow.live_enabled is True, "live should be the default"
    assert handlers.on_frame_change in bpy.app.handlers.frame_change_post, (
        "tetrahedralize should arm the frame handler"
    )
    try:
        bpy.context.scene.frame_set(1)
        session = handlers.SESSIONS.get(obj.name)
        assert session is not None, "a session should appear without pressing anything"
        assert session.live is True
        assert session.baked is False
    finally:
        handlers.unregister_handler()


def test_playing_forward_simulates_without_any_button_press():
    obj = _cube()
    rest = _positions(obj)
    scene = bpy.context.scene
    try:
        for frame in range(1, 11):
            scene.frame_set(frame)
        moved = _positions(obj)
        assert not np.allclose(moved, rest, atol=1e-6), "live playback did not move the mesh"
        assert moved[:, 2].mean() < rest[:, 2].mean(), "body did not fall"
        assert handlers.SESSIONS[obj.name].baked_range == (1, 10)
    finally:
        handlers.unregister_handler()


def test_scrubbing_back_replays_the_cache():
    obj = _cube()
    scene = bpy.context.scene
    try:
        for frame in range(1, 9):
            scene.frame_set(frame)
        at_eight = _positions(obj).copy()
        scene.frame_set(4)
        assert not np.allclose(_positions(obj), at_eight, atol=1e-6)
        scene.frame_set(8)
        assert np.allclose(_positions(obj), at_eight, atol=1e-6), (
            "a cached frame must replay exactly"
        )
    finally:
        handlers.unregister_handler()


def test_replaying_from_the_start_picks_up_changed_settings():
    """The point of the restart rule: no Free, no rebake, just replay."""
    obj = _cube()
    scene = bpy.context.scene
    try:
        obj.marrow.substeps = 2
        for frame in range(1, 7):
            scene.frame_set(frame)
        coarse = handlers.SESSIONS[obj.name].frame_positions(6).copy()

        obj.marrow.substeps = 20
        scene.frame_set(1)          # restart, re-reading the sliders
        for frame in range(2, 7):
            scene.frame_set(frame)
        fine = handlers.SESSIONS[obj.name].frame_positions(6)

        assert not np.allclose(coarse, fine, atol=1e-6), (
            "changing Substeps and replaying from frame 1 had no effect"
        )
    finally:
        handlers.unregister_handler()


def test_a_baked_cache_is_not_regenerated_at_the_start_frame():
    obj = _cube()
    scene = bpy.context.scene
    scene.frame_end = 6
    try:
        assert bpy.ops.marrow.bake() == {"FINISHED"}
        session = handlers.SESSIONS[obj.name]
        assert session.baked is True and session.live is False

        scene.frame_set(6)
        baked_six = _positions(obj).copy()

        obj.marrow.substeps = 2     # would change the result if it re-simulated
        scene.frame_set(1)
        scene.frame_set(6)
        assert np.allclose(_positions(obj), baked_six, atol=1e-9), (
            "a baked cache must replay untouched, not re-simulate"
        )
    finally:
        handlers.unregister_handler()


def test_scrubbing_before_the_start_resets_a_live_body():
    """Below the start there is no frame to serve, so hold nothing back.

    Leaving the last simulated pose on screen reads as a simulated state when
    it is really a stale one, and playback being paused makes no difference -
    a scrub is a scrub.
    """
    obj = _cube()
    rest = _positions(obj).copy()
    scene = bpy.context.scene
    try:
        for frame in range(1, 11):
            scene.frame_set(frame)
        assert not np.allclose(_positions(obj), rest, atol=1e-6), (
            "setup: the body should have moved by frame 10"
        )

        scene.frame_set(0)
        assert np.allclose(_positions(obj), rest, atol=1e-5), (
            "scrubbing before the start must put the body back at rest"
        )
        assert handlers.SESSIONS[obj.name].frame_positions(10) is None, (
            "a reset must drop the cache, or replaying would serve stale frames"
        )
    finally:
        handlers.unregister_handler()


def test_the_reset_follows_the_start_frame_not_frame_one():
    obj = _cube()
    rest = _positions(obj).copy()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 5, 30
    try:
        for frame in range(5, 15):
            scene.frame_set(frame)
        assert not np.allclose(_positions(obj), rest, atol=1e-6)

        scene.frame_set(4)
        assert np.allclose(_positions(obj), rest, atol=1e-5), (
            "frame 4 is before a start frame of 5, so it must reset"
        )
    finally:
        handlers.unregister_handler()


def test_a_baked_cache_is_not_reset_by_scrubbing_before_the_start():
    obj = _cube()
    scene = bpy.context.scene
    scene.frame_end = 6
    try:
        assert bpy.ops.marrow.bake() == {"FINISHED"}
        session = handlers.SESSIONS[obj.name]
        scene.frame_set(6)
        baked_six = _positions(obj).copy()

        scene.frame_set(0)
        scene.frame_set(6)
        assert session.baked is True, "a bake must survive a scrub before the start"
        assert np.allclose(_positions(obj), baked_six, atol=1e-9), (
            "a baked cache must replay untouched after scrubbing below the start"
        )
    finally:
        handlers.unregister_handler()


def test_a_small_skip_is_caught_up():
    obj = _cube()
    scene = bpy.context.scene
    try:
        scene.frame_set(1)
        scene.frame_set(4)
        session = handlers.SESSIONS[obj.name]
        assert session.frame_positions(2) is not None, "skipped frames must be filled in"
        assert session.frame_positions(4) is not None
    finally:
        handlers.unregister_handler()


def test_a_large_jump_leaves_the_mesh_alone():
    obj = _cube()
    scene = bpy.context.scene
    try:
        scene.frame_set(1)
        scene.frame_set(2)
        at_two = _positions(obj).copy()
        scene.frame_set(28)
        assert np.allclose(_positions(obj), at_two, atol=1e-9), (
            "an uncatchable jump must not half-simulate"
        )
    finally:
        handlers.unregister_handler()


def test_a_large_jump_with_an_empty_cache_catches_up_from_the_start():
    """The trap attach made visible: tetrahedralize, scrub to where the bones
    are posed, press play. The first frame change is a jump of the whole
    timeline, and a session that never simulated used to answer None for
    every frame after it - the mesh sat at rest while the bones moved. With
    nothing cached there is no history to protect, so catch up instead.
    """
    obj = _cube()
    rest = _positions(obj).copy()
    scene = bpy.context.scene
    try:
        scene.frame_set(28)  # jump straight in, no cache anywhere
        assert not np.allclose(_positions(obj), rest, atol=1e-6), (
            "a fresh session jumped mid-timeline must catch up, not sit at rest"
        )
        session = handlers.SESSIONS[obj.name]
        assert session.frame_positions(1) is not None, "catch-up fills the range"
        assert session.frame_positions(28) is not None
    finally:
        handlers.unregister_handler()


def test_flipping_attachment_mid_sim_restarts_the_group():
    """The toggle mutes the driving modifiers the instant the box is
    ticked, so a body that is muted but not yet bone-driven reads as
    broken. Flipping Attachment (or its stiffness) must therefore rebuild
    on the next frame change, not wait for a manual trip to the start
    frame. With k=1 and no bones the targets are the rest shape, so a
    restarted body snaps back up to it instead of falling on."""
    obj = _cube()
    rest_z = _positions(obj)[:, 2].mean()
    scene = bpy.context.scene
    try:
        for frame in range(1, 7):
            scene.frame_set(frame)
        session = handlers.SESSIONS[obj.name]
        assert session.attach_enabled is False
        falling_z = _positions(obj)[:, 2].mean()
        assert falling_z < rest_z - 0.05, "setup: the body should be falling"

        obj.marrow.attach_enabled = True
        obj.marrow.attach_stiffness = 1.0
        scene.frame_set(7)

        session = handlers.SESSIONS[obj.name]
        assert session.attach_enabled is True, (
            "the flip must be picked up without returning to the start frame"
        )
        held_z = _positions(obj)[:, 2].mean()
        assert held_z > falling_z, (
            f"the restart must re-attach the body (z {held_z:.3f}, was "
            f"falling at {falling_z:.3f})"
        )
    finally:
        handlers.unregister_handler()


def test_live_toggle_without_a_cage_names_the_fix():
    obj = _cube()
    bpy.ops.marrow.detetrahedralize()   # back to a plain mesh, live now off
    try:
        bpy.ops.marrow.live_toggle()
    except RuntimeError as exc:
        assert "Tetrahedralize" in str(exc), f"unhelpful error: {exc}"
    else:
        raise AssertionError("live without a cage must report an error")
    assert not obj.marrow.live_enabled, "a refused toggle still flipped Live on"


def test_turning_live_off_frees_the_session():
    obj = _cube()
    bpy.context.scene.frame_set(1)
    assert handlers.SESSIONS
    assert bpy.ops.marrow.live_toggle() == {"FINISHED"}
    assert obj.marrow.live_enabled is False
    assert not handlers.SESSIONS, "turning live off must release the session"


def test_no_session_is_built_when_live_is_off():
    obj = _cube()
    obj.marrow.live_enabled = False
    try:
        handlers.register_handler()
        bpy.context.scene.frame_set(3)
        assert obj.name not in handlers.SESSIONS
    finally:
        handlers.unregister_handler()


def test_a_failing_live_frame_stops_live_instead_of_raising_every_frame():
    obj = _cube()
    scene = bpy.context.scene
    try:
        scene.frame_set(1)
        session = handlers.SESSIONS[obj.name]

        def explode(frame, frame_start):
            raise RuntimeError("boom")

        session.ensure_frame = explode
        scene.frame_set(2)  # must not propagate out of the handler
        assert session.live is False, "a live failure must disable live mode"
    finally:
        handlers.unregister_handler()
