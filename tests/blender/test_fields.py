"""Gravity comes from the scene, and force fields push the body.

Marrow used to hardcode -9.81 and ignore Scene Properties entirely, so
turning gravity off in the place every other Blender physics system reads it
did nothing here. Force fields are the other half: Blender exposes no way to
ask it for the combined field at a point, so Force, Wind and Vortex are
reimplemented against the field settings on each object.
"""

import bpy
import gpu
import numpy as np

import marrow
from marrow.blender import handlers

gpu.init()

FRAMES = 8


def _body(resolution=0.5):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    handlers.free_all()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    obj.marrow.ground_enabled = False
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    return obj


def _travel(obj):
    """Where the mesh centroid moved to over the shot, in object space."""
    n = len(obj.data.vertices)
    before = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", before)
    before = before.reshape(-1, 3).mean(axis=0)
    try:
        for f in range(1, FRAMES + 1):
            bpy.context.scene.frame_set(f)
        after = np.empty(n * 3)
        obj.data.vertices.foreach_get("co", after)
        return after.reshape(-1, 3).mean(axis=0) - before
    finally:
        handlers.unregister_handler()


def _field(kind, location, strength, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.object.empty_add(location=location, rotation=rotation)
    ob = bpy.context.active_object
    bpy.ops.object.forcefield_toggle()
    ob.field.type = kind
    ob.field.strength = strength
    ob.field.falloff_power = 0.0
    return ob


def _collection_with(*objects):
    coll = bpy.data.collections.new("Fields")
    bpy.context.scene.collection.children.link(coll)
    for ob in objects:
        coll.objects.link(ob)
        bpy.context.scene.collection.objects.unlink(ob)
    return coll


def test_gravity_comes_from_the_scene():
    obj = _body()
    assert bpy.context.scene.use_gravity
    assert _travel(obj)[2] < -0.1, "the body should fall on the scene default"


def test_turning_scene_gravity_off_stops_the_fall():
    """The setting every other Blender physics system obeys. Marrow ignored
    it entirely before, which made it look broken rather than opinionated."""
    obj = _body()
    bpy.context.scene.use_gravity = False
    assert abs(_travel(obj)[2]) < 1e-3, "gravity off should leave it hanging"


def test_the_gravity_scale_makes_a_body_weightless():
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    assert abs(_travel(obj)[2]) < 1e-3


def test_a_negative_gravity_scale_falls_upwards():
    obj = _body()
    obj.marrow.gravity_scale = -1.0
    assert _travel(obj)[2] > 0.1


def test_a_wind_field_pushes_the_body_along_its_axis():
    """An empty rotated so its local +Z lies along world +X, which is the
    axis Blender blows a Wind field down."""
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    wind = _field("WIND", (0.0, 0.0, 0.0), 30.0, rotation=(0.0, 1.5707963, 0.0))
    obj.marrow.field_collection = _collection_with(wind)

    travel = _travel(obj)
    assert travel[0] > 0.1, f"wind did not carry the body: {np.round(travel, 3)}"
    assert abs(travel[1]) < 0.05 and abs(travel[2]) < 0.05, (
        f"wind should blow down one axis only: {np.round(travel, 3)}"
    )


def test_a_force_field_repels():
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    push = _field("FORCE", (0.0, 0.0, -4.0), 40.0)
    obj.marrow.field_collection = _collection_with(push)
    assert _travel(obj)[2] > 0.1, "a positive Force strength should push away"


def test_an_empty_field_collection_changes_nothing():
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    obj.marrow.field_collection = _collection_with()
    assert np.linalg.norm(_travel(obj)) < 1e-3


def test_an_unsupported_field_type_is_ignored():
    """Magnetic needs a velocity cross product and a charge model that
    nothing here has. Skipping it beats approximating it with the wrong one
    and calling that wind."""
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    magnet = _field("MAGNET", (0.0, 0.0, 0.0), 50.0)
    obj.marrow.field_collection = _collection_with(magnet)
    assert np.linalg.norm(_travel(obj)) < 1e-3


def test_turbulence_moves_the_body():
    obj = _body()
    obj.marrow.gravity_scale = 0.0
    noise = _field("TURBULENCE", (0.0, 0.0, 0.0), 60.0)
    noise.field.size = 1.0
    obj.marrow.field_collection = _collection_with(noise)
    assert np.linalg.norm(_travel(obj)) > 0.05


def test_turbulence_is_not_a_uniform_push():
    """The whole point: it has to vary across the body, or it is just a wind
    with extra settings. Compares how far each render vertex travelled."""
    obj = _body(resolution=0.4)
    obj.marrow.gravity_scale = 0.0
    noise = _field("TURBULENCE", (0.0, 0.0, 0.0), 60.0)
    noise.field.size = 0.7
    obj.marrow.field_collection = _collection_with(noise)

    n = len(obj.data.vertices)
    before = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", before)
    _travel(obj)
    after = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", after)
    moved = np.linalg.norm(
        after.reshape(-1, 3) - before.reshape(-1, 3), axis=1
    )
    assert moved.std() > 0.01, (
        f"every vertex moved alike ({moved.std():.4f}) - that is a wind, "
        f"not turbulence"
    )


def test_the_seed_changes_the_pattern():
    """Two turbulence fields in the same place must not push identically,
    which is the only thing Seed is for."""
    travels = []
    for seed in (0, 7):
        obj = _body()
        obj.marrow.gravity_scale = 0.0
        noise = _field("TURBULENCE", (0.0, 0.0, 0.0), 60.0)
        noise.field.size = 1.0
        noise.field.seed = seed
        obj.marrow.field_collection = _collection_with(noise)
        travels.append(_travel(obj))
    assert np.linalg.norm(travels[0] - travels[1]) > 0.01, (
        f"seed made no difference: {np.round(travels[0], 4)}"
    )


def test_size_changes_how_tightly_it_swirls():
    travels = []
    for size in (0.3, 4.0):
        obj = _body()
        obj.marrow.gravity_scale = 0.0
        noise = _field("TURBULENCE", (0.0, 0.0, 0.0), 60.0)
        noise.field.size = size
        obj.marrow.field_collection = _collection_with(noise)
        travels.append(_travel(obj))
    assert np.linalg.norm(travels[0] - travels[1]) > 0.01
