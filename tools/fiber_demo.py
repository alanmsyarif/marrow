"""Build a fiber locomotion scene. Run from Blender's Scripting tab.

Not a test: a snake crawling forward is the acceptance criterion for this
feature but makes a slow, flaky assertion. This puts the scene on screen so
it can be judged by eye, which is the honest way to judge it.

Tetrahedralize runs twice, on purpose. The Curve slot only appears on a
cage that already exists, and fibers are baked from whatever curve is set
at Tetrahedralize time - so the first pass exists only to create the cage,
and the second is the one that actually bakes fibers. One call would leave
the snake with no fibers and no motion, silently.
"""

import importlib

import bpy

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

# The Scripting tab always has this registered already, because it comes
# from an enabled add-on. Headless `--factory-startup` runs do not enable
# any add-on, so `obj.marrow` would not exist yet without this.
if not hasattr(bpy.types.Object, "marrow"):
    _addon.register()

CAGE_SUFFIX = _session.CAGE_SUFFIX
read_fiber = _storage.read_fiber

LENGTH = 6.0
RADIUS = 0.35


def build():
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=RADIUS, depth=LENGTH, location=(0.0, 0.0, RADIUS),
        rotation=(0.0, 1.5708, 0.0), vertices=24,
    )
    body = bpy.context.active_object
    body.name = "Snake"

    data = bpy.data.curves.new("Spine", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (-LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spline.points[1].co = (LENGTH / 2.0, 0.0, RADIUS, 1.0)
    spine = bpy.data.objects.new("Spine", data)
    bpy.context.collection.objects.link(spine)

    settings = body.marrow
    settings.resolution = 0.18
    bpy.context.view_layer.objects.active = body

    # Pass 1: no Curve is set yet, so this only creates the cage - it bakes
    # no fibers. It exists to make the Curve slot appear at all.
    bpy.ops.marrow.tetrahedralize()

    settings.fiber_curve = spine

    # Pass 2: the Curve is set now, so this is the tetrahedralize that
    # actually samples it and bakes fiber rows onto the cage.
    bpy.ops.marrow.tetrahedralize()

    settings.fiber_enabled = True
    settings.fiber_stiffness = 3.0e4
    settings.wave_amplitude = 0.35
    settings.wave_length = 1.5
    settings.wave_speed = 1.2
    settings.waveform = "SMOOTH"
    # Locomotion is contraction plus grip. With friction at zero the wave
    # travels and the body goes nowhere.
    settings.friction = 0.8
    settings.ground_enabled = True
    settings.ground_z = 0.0
    settings.substeps = 20

    cage = bpy.data.objects[f"Snake{CAGE_SUFFIX}"]
    fiber = read_fiber(cage.data)
    if fiber is None:
        raise RuntimeError(
            "No fibers baked onto the cage. The spine curve this script "
            "built should always bake - if it did not, something upstream "
            "of this script broke."
        )

    print(
        f"Snake built: {len(cage.data.vertices)} nodes, "
        f"{fiber.shape[0]} fiber rows. Press play."
    )


build()
