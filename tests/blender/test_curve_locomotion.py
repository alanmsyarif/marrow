"""End-to-end: a Curve modifier steers the body, Attachment carries it.

The recipe tools/curve_path_demo.py builds, asserted. A one-bone armature
advances the mesh along its own local X, the Curve modifier bends that
advance onto a path, and Attachment is the only thing that could move the
simulation - so any travel the cage shows arrived through the attachment
pass, along the curve.

The second test is the reason the armature is there at all: targets are read
in the bind frame (attach.sample_targets), so driving the same rig from the
object's transform instead cancels out and the body wriggles on the spot.
That is deliberate, and this pins it - if targets ever move to the object's
current frame, the demo's conveyor becomes wrong and this test says so.
"""

import math

import bpy
import numpy as np
from mathutils import Vector

import marrow
from marrow.blender import handlers

LENGTH = 6.0
RADIUS = 0.35
END = 40
TRAVEL = 6.0
PATH_LENGTH = 20.0
AMPLITUDE = 0.9
FREQUENCY = 0.9


def _fresh_addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _linear(obj):
    """Slotted actions (4.4+) keep fcurves in the strip's channelbag."""
    action = obj.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag is None:
                    continue
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"


def _path():
    data = bpy.data.curves.new("Path", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    points = 100
    spline.points.add(points - 1)
    for i in range(points):
        x = -LENGTH / 2.0 + i * (PATH_LENGTH / (points - 1))
        spline.points[i].co = (
            x, AMPLITUDE * math.sin(x * FREQUENCY), RADIUS, 1.0,
        )
    obj = bpy.data.objects.new("Path", data)
    bpy.context.collection.objects.link(obj)
    return obj


def _body():
    """A cylinder whose LENGTH runs along local X, which is the axis the
    Curve modifier deforms along. The rotation is applied, not kept on the
    object, or the deform would send the body sideways down the path."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=RADIUS, depth=LENGTH, location=(0.0, 0.0, RADIUS),
        rotation=(0.0, math.pi / 2.0, 0.0), vertices=16,
    )
    obj = bpy.context.active_object
    obj.name = "Snake"
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return obj


def _conveyor():
    """One bone sliding TRAVEL along world +X, keyed LINEAR."""
    scene = bpy.context.scene
    bpy.ops.object.armature_add(location=(0.0, 0.0, RADIUS))
    arm = bpy.context.active_object
    arm.name = "Drive"
    bone = arm.pose.bones[0]
    # Pose-bone location is bone space and the default bone's axes are not
    # the world's, so the drive is converted through the rest matrix.
    to_bone = bone.bone.matrix_local.to_3x3().inverted()
    scene.frame_set(1)
    bone.location = (0.0, 0.0, 0.0)
    bone.keyframe_insert("location")
    scene.frame_set(END)
    bone.location = to_bone @ Vector((TRAVEL, 0.0, 0.0))
    bone.keyframe_insert("location")
    _linear(arm)
    scene.frame_set(1)
    return arm


def _scene(drive):
    """The demo's rig. ``drive`` picks what advances the body:

    "local" is the demo's conveyor - a bone, so the advance lands in the
    mesh's local space where the Curve modifier and the bind frame can both
    see it. "object" keys the object's own location instead, which is the
    intuitive thing to reach for and the thing that does not work.
    """
    _fresh_addon()
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, END

    arm = _conveyor() if drive == "local" else None
    body = _body()
    if arm is not None:
        group = body.vertex_groups.new(name=arm.pose.bones[0].name)
        group.add(range(len(body.data.vertices)), 1.0, "REPLACE")
        modifier = body.modifiers.new("Armature", "ARMATURE")
        modifier.object = arm
    else:
        scene.frame_set(1)
        body.location = (0.0, 0.0, RADIUS)
        body.keyframe_insert("location")
        scene.frame_set(END)
        body.location = (TRAVEL, 0.0, RADIUS)
        body.keyframe_insert("location")
        _linear(body)
        scene.frame_set(1)

    curve = body.modifiers.new("Curve", "CURVE")
    curve.object = _path()
    curve.deform_axis = "POS_X"

    settings = body.marrow
    settings.resolution = 0.4
    settings.attach_enabled = True
    settings.attach_stiffness = 0.4
    settings.ground_enabled = True
    settings.ground_z = 0.0
    settings.friction = 0.5
    settings.substeps = 8
    bpy.context.view_layer.objects.active = body
    return body


def _spine():
    data = bpy.data.curves.new("Spine", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (-LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spline.points[1].co = (LENGTH / 2.0, 0.0, RADIUS, 1.0)
    obj = bpy.data.objects.new("Spine", data)
    bpy.context.collection.objects.link(obj)
    return obj


def _bake(obj):
    assert bpy.ops.marrow.bake() == {"FINISHED"}
    return handlers.SESSIONS[obj.name]


def _track(session):
    """(centroid, x-spread) at the first and last frame."""
    first = session.frame_positions(1)
    last = session.frame_positions(END)
    return (
        first.mean(axis=0), last.mean(axis=0),
        float(first.max(axis=0)[0] - first.min(axis=0)[0]),
        float(last.max(axis=0)[0] - last.min(axis=0)[0]),
    )


def test_the_body_is_carried_along_the_curve():
    """Attachment is the only pass that could move the cage sideways, so the
    travel it shows came through the curve. Most of the drive has to land -
    a soft attachment lags, it does not stall - and the body has to arrive
    still a body, not a collapsed clump."""
    body = _scene("local")
    bpy.ops.marrow.tetrahedralize()
    session = _bake(body)
    try:
        first, last, span_first, span_last = _track(session)
        travelled = last[0] - first[0]
        assert travelled > 0.6 * TRAVEL, (
            f"body covered {travelled:+.3f} m of the {TRAVEL} m drive - "
            f"the curve is not carrying it"
        )
        assert travelled < 1.4 * TRAVEL, (
            f"body overshot the drive by a long way: {travelled:+.3f} m"
        )
        assert span_last > 0.6 * span_first, (
            f"body collapsed along its own length: {span_first:.2f} m to "
            f"{span_last:.2f} m"
        )
        # The path swings +-AMPLITUDE, so a body riding it stays inside that
        # band with margin for the lag. A body that had come off the deform
        # would drift out of it and keep going.
        for frame in (1, END // 2, END):
            y = session.frame_positions(frame)[:, 1]
            assert np.abs(y).max() < 3.0 * AMPLITUDE, (
                f"frame {frame}: body left the path laterally, "
                f"max |y| {np.abs(y).max():.2f}"
            )
    finally:
        handlers.unregister_handler()


def test_the_object_transform_cannot_drive_the_curve():
    """The bind-frame rule, pinned. Targets are read in the frame the cage
    was tetrahedralized in, so the Curve modifier bending the mesh forward
    in local space and the object transform sliding it back cancel exactly.
    The body wriggles on the spot instead of travelling - which is why the
    demo drives from a bone and not from the object."""
    body = _scene("object")
    bpy.ops.marrow.tetrahedralize()
    session = _bake(body)
    try:
        first, last, _, _ = _track(session)
        travelled = last[0] - first[0]
        assert abs(travelled) < 0.25 * TRAVEL, (
            f"the object transform moved the simulation {travelled:+.3f} m. "
            f"Targets are supposed to be bind-frame - if that changed on "
            f"purpose, tools/curve_path_demo.py no longer needs its armature"
        )
    finally:
        handlers.unregister_handler()


def test_the_fiber_wave_rides_on_a_curve_driven_body():
    """The two features have to compose: the curve steers, the fiber wave
    squashes along the body while it is steered. Fibers are sampled against
    the REST cage, so the straight spine stays a valid source however far
    the body is later bent - and turning them on must not cost the travel."""
    body = _scene("local")
    bpy.ops.marrow.tetrahedralize()
    body.marrow.fiber_curve = _spine()
    bpy.ops.marrow.tetrahedralize()
    settings = body.marrow
    settings.fiber_enabled = True
    settings.fiber_stiffness = 3.0e4
    settings.wave_amplitude = 0.2
    settings.wave_length = 2.0
    settings.wave_speed = 1.0
    session = _bake(body)
    try:
        first, last, _, _ = _track(session)
        travelled = last[0] - first[0]
        assert travelled > 0.6 * TRAVEL, (
            f"fibers on cost the body its ride: {travelled:+.3f} m of "
            f"{TRAVEL} m"
        )
        positions = session.frame_positions(END)
        assert np.isfinite(positions).all(), (
            "fiber wave on a curve-driven body produced non-finite nodes"
        )
    finally:
        handlers.unregister_handler()
