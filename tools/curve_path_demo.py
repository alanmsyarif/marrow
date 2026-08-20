"""Build a curve-driven locomotion scene. Run from Blender's Scripting tab.

The companion to fiber_demo.py, and the answer to "can the body follow a
curve?". Fibers alone give emergent locomotion - contraction plus friction,
nobody steering - which crawls but wanders. This scene is the art-directed
half: a curve says where the body goes, Attachment drags the flesh there,
and the fiber wave rides on top so the travel still looks muscular rather
than like a mesh being slid along a rail.

THE ONE RULE THAT MAKES THIS WORK. Attachment samples the object's
EVALUATED mesh and reads it in the bind frame - the world frame the cage was
tetrahedralized in - not in the object's current frame (see
`attach.sample_targets`). So animating the object's own location buys
nothing: the Curve modifier bends the mesh forward in local space and the
object transform slides it straight back again, and the body wriggles on the
spot. The travel has to happen in the mesh's LOCAL space, which is what the
one-bone armature below is for. It is a conveyor, not a rig: a single bone
sliding along +X, the whole body weighted to it, feeding the Curve modifier
a body that is already further along its own length.

Modifier order is load-bearing: Armature first, Curve second. Reversed, the
bone would drag the already-bent body sideways off the path.
"""

import importlib
import math

import bpy
from mathutils import Vector

# Installed as an extension the package is bl_ext.user_default.marrow; run
# from a clone of the repo it is plain marrow. Try both rather than make the
# user care which one they have.
for _pkg in ("bl_ext.user_default.marrow", "marrow"):
    try:
        _addon = importlib.import_module(_pkg)
        _session = importlib.import_module(f"{_pkg}.blender.session")
        _storage = importlib.import_module(f"{_pkg}.blender.storage")
        break
    except ImportError:
        continue
else:
    raise SystemExit(
        "Marrow not importable. Enable the add-on, or run this from the "
        "repo root so that `marrow` is on sys.path."
    )

if not hasattr(bpy.types.Object, "marrow"):
    _addon.register()

CAGE_SUFFIX = _session.CAGE_SUFFIX
read_fiber = _storage.read_fiber

LENGTH = 6.0
RADIUS = 0.35
TRAVEL = 12.0        # how far along the path the body is driven
END = 96             # last frame of the drive
PATH_LENGTH = 30.0   # the path has to outrun body + travel, or the body
                     # runs off the end and the deform extrapolates
AMPLITUDE = 0.9      # lateral swing of the path
FREQUENCY = 0.9      # radians per world unit; ~7 m per full S


def _linear(obj):
    """Every keyframe on ``obj`` set to LINEAR, so the drive is steady.

    Slotted actions (4.4+): the fcurves live in the strip's channelbag, not
    on the action, so this walks layers and slots rather than action.fcurves.
    """
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


def build():
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)

    scene = bpy.context.scene
    # SolverParams.dt is fixed at 1/24 and nothing reads scene.render.fps,
    # so the wave's "cycles per second" is cycles per 24 frames. At any
    # other frame rate the motion plays back off its own clock.
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, END

    # --- the conveyor ---
    bpy.ops.object.armature_add(location=(0.0, 0.0, RADIUS))
    arm = bpy.context.active_object
    arm.name = "Drive"
    bone = arm.pose.bones[0]
    # Pose-bone location is bone space, and the default bone's axes are not
    # the world's - measured, not assumed. Converting through the rest
    # matrix is what keeps this a +X drive whatever the bone is doing.
    to_bone = bone.bone.matrix_local.to_3x3().inverted()

    scene.frame_set(1)
    bone.location = (0.0, 0.0, 0.0)
    bone.keyframe_insert("location")
    scene.frame_set(END)
    bone.location = to_bone @ Vector((TRAVEL, 0.0, 0.0))
    bone.keyframe_insert("location")
    _linear(arm)
    scene.frame_set(1)

    # --- the body ---
    bpy.ops.mesh.primitive_cylinder_add(
        radius=RADIUS, depth=LENGTH, location=(0.0, 0.0, RADIUS),
        rotation=(0.0, math.pi / 2.0, 0.0), vertices=24,
    )
    body = bpy.context.active_object
    body.name = "Snake"
    # The rotation is applied rather than kept on the object: the Curve
    # modifier's Deform Axis is a LOCAL axis, so the body's length has to
    # run along local X or the deform sends it sideways down the path.
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    group = body.vertex_groups.new(name=bone.name)
    group.add(range(len(body.data.vertices)), 1.0, "REPLACE")

    # --- the path ---
    data = bpy.data.curves.new("Path", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    points = 140
    spline.points.add(points - 1)
    for i in range(points):
        x = -LENGTH / 2.0 + i * (PATH_LENGTH / (points - 1))
        spline.points[i].co = (
            x, AMPLITUDE * math.sin(x * FREQUENCY), RADIUS, 1.0,
        )
    path = bpy.data.objects.new("Path", data)
    bpy.context.collection.objects.link(path)

    armature_mod = body.modifiers.new("Armature", "ARMATURE")
    armature_mod.object = arm
    curve_mod = body.modifiers.new("Curve", "CURVE")
    curve_mod.object = path
    curve_mod.deform_axis = "POS_X"

    # --- the fiber spine ---
    # Straight, and along the body's REST shape: fibers are sampled at
    # Tetrahedralize against the undeformed cage, so this is a source of
    # direction and phase, not a second path. The Path above is what steers.
    spine_data = bpy.data.curves.new("Spine", type="CURVE")
    spine_data.dimensions = "3D"
    spine_spline = spine_data.splines.new("POLY")
    spine_spline.points.add(1)
    spine_spline.points[0].co = (-LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spine_spline.points[1].co = (LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spine = bpy.data.objects.new("Spine", spine_data)
    bpy.context.collection.objects.link(spine)

    settings = body.marrow
    settings.resolution = 0.18
    settings.attach_enabled = True
    # Soft on purpose. At 1.0 the flesh rides the curve exactly and the
    # result is a mesh on a rail; this low it lags, overshoots the corners
    # and settles, which is the whole reason to simulate it at all.
    settings.attach_stiffness = 0.4
    settings.ground_enabled = True
    settings.ground_z = 0.0
    settings.friction = 0.5
    settings.substeps = 20

    bpy.context.view_layer.objects.active = body

    # Pass 1 only creates the cage - the Curve slot is empty, so it bakes no
    # fibers. Pass 2 is the one that samples the spine. Same two-pass dance
    # as fiber_demo.py, and for the same reason.
    bpy.ops.marrow.tetrahedralize()
    settings.fiber_curve = spine
    bpy.ops.marrow.tetrahedralize()

    # Mild. The curve is doing the travelling here, so the fiber term is
    # seasoning - a muscular squash along a body that is already moving -
    # rather than the thing that has to produce locomotion by itself.
    settings.fiber_enabled = True
    settings.fiber_stiffness = 3.0e4
    settings.wave_amplitude = 0.2
    settings.wave_length = 2.0
    settings.wave_speed = 1.0
    settings.waveform = "SMOOTH"

    cage = bpy.data.objects[f"Snake{CAGE_SUFFIX}"]
    fiber = read_fiber(cage.data)
    if fiber is None:
        raise RuntimeError(
            "No fibers baked onto the cage. The spine curve this script "
            "built should always bake - if it did not, something upstream "
            "of this script broke."
        )

    print(
        f"Curve-driven snake built: {len(cage.data.vertices)} nodes, "
        f"{fiber.shape[0]} fiber rows, {TRAVEL} m of travel over {END} "
        f"frames. Press play, or Bake for the full quality."
    )


build()
