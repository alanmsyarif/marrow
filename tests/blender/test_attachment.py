"""End-to-end: an animated bone drives the soft body from inside the solver.

Builds a one-bone armature with a fully skinned cube, turns attachment on,
and bakes. The bone translation is the only force that could move the cube
sideways, so any tracked motion went through the attachment pass.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.attach import DISPLAY_KEY
from marrow.blender.session import find_cage
from marrow.blender.storage import ATTACH_IDX

FRAMES = (1, 8)


def _fresh_addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _skinned_body(shift):
    """Armature with one linearly animated bone plus a cube weighted to it."""
    _fresh_addon()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FRAMES

    bpy.ops.object.armature_add()
    arm = bpy.context.active_object
    bone = arm.pose.bones[0]

    scene.frame_set(FRAMES[0])
    bone.location = (0.0, 0.0, 0.0)
    bone.keyframe_insert("location")
    scene.frame_set(FRAMES[1])
    bone.location = shift
    bone.keyframe_insert("location")
    # Slotted actions (4.4+): the fcurves live in the strip's channelbag,
    # not on the action. LINEAR so mid-bake frames see a steady drag.
    action = arm.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag is None:
                    continue
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"
    scene.frame_set(FRAMES[0])

    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    weights = obj.vertex_groups.new(name=bone.name)
    weights.add(range(len(obj.data.vertices)), 1.0, "REPLACE")
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
    obj.marrow.resolution = 0.5
    return obj


def _centroid(positions):
    return np.asarray(positions, dtype=np.float64).mean(axis=0)


def _bake(obj):
    assert bpy.ops.marrow.bake() == {"FINISHED"}
    return handlers.SESSIONS[obj.name]


def test_a_hard_attachment_rides_the_bone():
    """k=1 snaps every substep, so the baked centroid moves exactly as far
    as the bone did - gravity included, since the snap has the last word."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    bpy.ops.marrow.tetrahedralize()
    session = _bake(obj)
    try:
        first = _centroid(session.frame_positions(FRAMES[0]))
        last = _centroid(session.frame_positions(FRAMES[1]))
        delta = last - first
        assert abs(delta[0] - 2.0) < 1e-2, (
            f"body tracked the bone {delta[0]:+.3f} m, expected 2.0"
        )
        assert abs(delta[1]) < 1e-2 and abs(delta[2]) < 1e-2, (
            f"body drifted sideways while riding the bone: {delta}"
        )
    finally:
        handlers.unregister_handler()


def test_soft_attachment_lags_and_settles_behind_the_bone():
    """Low stiffness drags the body along but cannot hold it rigid. Mid
    chase the follower sits in a steady lag behind the constant-speed
    bone; when the bone stops dead the inertia may even carry the body
    past it, so only the mid-chase lag is asserted sharply."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.3
    bpy.ops.marrow.tetrahedralize()
    session = _bake(obj)
    try:
        first = _centroid(session.frame_positions(FRAMES[0]))
        mid = _centroid(session.frame_positions(4))
        last = _centroid(session.frame_positions(FRAMES[1]))
        # LINEAR keys: at frame 4 the bone has covered 3/7 of its 2 m
        # travel. Mid-chase a soft follower sits in a steady lag behind a
        # constant-speed target, so it must have moved, but less.
        dx_mid = mid[0] - first[0]
        assert 0.1 < dx_mid < 2.0 * 3.0 / 7.0, (
            f"mid-bake centroid at {dx_mid:+.3f} m, bone at "
            f"{2.0 * 3.0 / 7.0:+.3f} m - expected a visible lag"
        )
        # At the end the bone stops dead and the inertia may carry the
        # body past it; only require that it was dragged along at all.
        assert last[0] - first[0] > 0.5, (
            f"soft attachment barely moved the body: {last[0] - first[0]:+.3f} m"
        )
    finally:
        handlers.unregister_handler()


def test_the_ground_plane_outranks_the_bone():
    """The bone drags the body 4 m down, straight through the ground plane.
    Attachment runs before collision, so the plane keeps the last word.
    Free fall alone covers ~0.42 m in 7 frames, so a body that reaches the
    plane was dragged there by the bone, not by gravity.

    Pose-bone location is bone-space, and the default bone's local Y runs
    along its axis (world +z) - measured, not assumed. So a world drop of
    (0, 0, -4) is keyed as bone location (0, -4, 0).
    """
    obj = _skinned_body(shift=(0.0, -4.0, 0.0))
    obj.location = (0.0, 0.0, 2.0)
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    obj.marrow.ground_enabled = True
    obj.marrow.ground_z = 0.0
    bpy.ops.marrow.tetrahedralize()
    session = _bake(obj)
    try:
        first = session.frame_positions(FRAMES[0])
        last = session.frame_positions(FRAMES[1])
        # Gravity alone drops the body ~0.42 m in these 7 frames; falling
        # further than that means the bone did the dragging.
        drop = float(last[:, 2].mean() - first[:, 2].mean())
        assert drop < -1.0, f"bone did not drag the body down: {drop:+.3f}"
        min_z = float(last[:, 2].min())
        assert min_z > -1e-3, f"bone dragged the body underground: {min_z:.4f}"
        assert min_z < 0.5, (
            f"body never reached the plane (min z {min_z:.3f}); the bone "
            f"did not drive the sim at all"
        )
    finally:
        handlers.unregister_handler()


def test_the_displayed_mesh_is_not_deformed_twice():
    """The armature modifier feeds the targets; the written simulation is
    the display. Leaving the modifier shown would bend the result a second
    time - measured at exactly twice the bone travel - so attachment mutes
    it in viewport and render while it is on, and what the viewport shows
    must be the cached simulation, nothing more."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    bpy.ops.marrow.tetrahedralize()
    modifier = obj.modifiers[0]
    assert modifier.show_viewport is False and modifier.show_render is False, (
        "attachment on must mute the object's own modifiers"
    )
    session = _bake(obj)
    try:
        bpy.context.scene.frame_set(FRAMES[1])
        simulated = _centroid(session.frame_positions(FRAMES[1]))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        me = evaluated.to_mesh()
        try:
            verts = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", verts)
        finally:
            evaluated.to_mesh_clear()
        displayed = _centroid(verts.reshape(-1, 3))
        assert abs(displayed[0] - 2.0) < 1e-2, (
            f"displayed mesh at {displayed[0]:+.3f} m, bone travelled 2.0 - "
            "the modifier bent the simulation a second time"
        )
        assert np.allclose(displayed, simulated, atol=1e-3), (
            f"displayed {displayed} is not the cached simulation {simulated}"
        )
    finally:
        handlers.unregister_handler()


def test_toggling_attach_off_hands_the_modifiers_back():
    """The mute is borrowed visibility, not owned: the original state is
    stored on the object and handed back by the toggle and by
    De-tetrahedralize."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    bpy.ops.marrow.tetrahedralize()
    modifier = obj.modifiers[0]
    assert modifier.show_viewport is False
    assert DISPLAY_KEY in obj

    obj.marrow.attach_enabled = False
    assert modifier.show_viewport is True and modifier.show_render is True
    assert DISPLAY_KEY not in obj

    obj.marrow.attach_enabled = True
    assert modifier.show_viewport is False
    bpy.ops.marrow.detetrahedralize()
    assert obj.marrow.attach_enabled is False
    assert modifier.show_viewport is True and modifier.show_render is True
    assert DISPLAY_KEY not in obj


def test_a_count_changing_modifier_is_display_only():
    """Subdivision cannot feed the targets - its vertices have no
    per-base-vertex meaning - so it stays shown and smooths the display,
    while the armature modifier is muted and drives the simulation. A
    body under a Subdivision must bake without the vertex-count error
    and still ride the bone."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    bpy.ops.marrow.tetrahedralize()
    session = _bake(obj)
    try:
        armature = obj.modifiers["Armature"]
        assert armature.show_viewport is False, (
            "the driving modifier must stay muted under attachment"
        )
        assert sub.show_viewport is True, (
            "a count-changing modifier must be handed back to the display"
        )
        rows = obj[DISPLAY_KEY]
        assert len(rows["Subdivision"]) == 3 and not rows["Subdivision"][2], (
            f"the probe must record Subdivision as count-changing: {rows}"
        )
        first = _centroid(session.frame_positions(FRAMES[0]))
        last = _centroid(session.frame_positions(FRAMES[1]))
        assert abs((last - first)[0] - 2.0) < 1e-2, (
            "the body must still ride the bone under a Subdivision modifier"
        )
        # The display is the smoothed simulation: evaluated mesh is denser.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        me = evaluated.to_mesh()
        try:
            assert len(me.vertices) > len(obj.data.vertices), (
                "Subdivision must still smooth the displayed mesh"
            )
        finally:
            evaluated.to_mesh_clear()
    finally:
        handlers.unregister_handler()


def test_attachment_survives_moving_the_object_after_tetrahedralize():
    """Weights are synthesized in object space and targets are sampled into
    the bind frame, so relocating the body after Tetrahedralize must not
    scramble the blend into a featureless blob. The simulation, like the
    rest of Marrow, simply stays in the world frame it started in."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    bpy.ops.marrow.tetrahedralize()
    obj.location = (3.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    session = _bake(obj)
    try:
        first = session.frame_positions(FRAMES[0])
        last = session.frame_positions(FRAMES[1])
        spread = last.max(axis=0) - last.min(axis=0)
        assert np.all(spread > 1.5) and np.all(spread < 2.5), (
            f"moving the object after Tetrahedralize scrambled the body "
            f"into a blob: spread {spread}"
        )
        delta = _centroid(last) - _centroid(first)
        assert abs(delta[0] - 2.0) < 1e-2, (
            f"body tracked the bone {delta[0]:+.3f} m, expected 2.0"
        )
    finally:
        handlers.unregister_handler()


def test_tetrahedralizing_at_a_posed_frame_still_binds_the_rest_shape():
    """The lattice fill evaluates the modifier stack, but the bind and the
    stored rest read the unposed base mesh. Tetrahedralizing while the bone
    plays a pose must therefore park the modifiers for the capture - else
    the cage fills the posed silhouette, the weights collapse onto a handful
    of vertices, and every target becomes a blob. With the capture parked,
    a body tetrahedralized mid-pose still rides the bone at full spread."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 1.0
    bpy.context.scene.frame_set(FRAMES[1])  # tet into the pose, not the rest
    bpy.ops.marrow.tetrahedralize()
    assert obj.modifiers[0].show_viewport is False, (
        "attachment on must keep the modifier muted across the capture"
    )
    session = _bake(obj)
    try:
        first = _centroid(session.frame_positions(FRAMES[0]))
        last = _centroid(session.frame_positions(FRAMES[1]))
        delta = last - first
        assert abs(delta[0] - 2.0) < 1e-2, (
            f"body tracked the bone {delta[0]:+.3f} m, expected 2.0"
        )
        spread = session.frame_positions(FRAMES[1])
        spread = spread.max(axis=0) - spread.min(axis=0)
        assert np.all(spread > 1.5) and np.all(spread < 2.5), (
            f"posed-frame tetrahedralize scrambled the body: spread {spread}"
        )
    finally:
        handlers.unregister_handler()


def test_attach_off_leaves_the_classic_trajectory_alone():
    """Zero-cost guard: with the toggle off there is no shader, no cached
    weights, and the bake is bit-identical to the previous bake."""
    obj = _skinned_body(shift=(2.0, 0.0, 0.0))
    obj.marrow.attach_enabled = False
    bpy.ops.marrow.tetrahedralize()
    session = _bake(obj)
    baseline = {
        frame: session.frame_positions(frame).copy() for frame in FRAMES
    }
    assert getattr(session.solver, "sh_attach", None) is None, (
        "attach off still built the attachment shader"
    )
    cage = find_cage(obj)
    assert cage.data.attributes.get(ATTACH_IDX) is None, (
        "attach off still synthesized and cached skin weights"
    )
    handlers.unregister_handler()

    session = _bake(obj)
    try:
        for frame in FRAMES:
            assert np.array_equal(session.frame_positions(frame), baseline[frame]), (
                f"attach off changed the bake at frame {frame}"
            )
    finally:
        handlers.unregister_handler()
