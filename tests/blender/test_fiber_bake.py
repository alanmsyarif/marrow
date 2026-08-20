"""Sampling a Curve, and baking the result at Tetrahedralize."""

import bpy
import numpy as np

import marrow
from marrow.blender.curve import polyline_from_curve


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _straight_curve(name="spine", length=4.0):
    """A two-point poly curve along +X, at the origin."""
    data = bpy.data.curves.new(name, type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("POLY")
    spline.points.add(1)                      # a new spline starts with one
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (length, 0.0, 0.0, 1.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def test_polyline_runs_along_the_curve_in_order():
    _fresh()
    obj = _straight_curve()
    points = polyline_from_curve(bpy.context, obj)
    assert points.shape[1] == 3
    assert points.shape[0] >= 2
    xs = points[:, 0]
    assert np.all(np.diff(xs) > 0.0), f"points are not in path order: {xs}"
    assert abs(xs[0]) < 1e-6 and abs(xs[-1] - 4.0) < 1e-6


def test_polyline_is_world_space():
    _fresh()
    obj = _straight_curve("moved_spine")
    obj.location = (10.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    points = polyline_from_curve(bpy.context, obj)
    assert abs(points[:, 0].min() - 10.0) < 1e-6, "curve was not taken to world space"


def test_a_non_curve_object_yields_nothing():
    _fresh()
    mesh_obj = bpy.data.objects.new("not_a_curve", bpy.data.meshes.new("m"))
    bpy.context.collection.objects.link(mesh_obj)
    assert polyline_from_curve(bpy.context, mesh_obj).shape == (0, 3)


def test_a_curve_with_no_splines_yields_nothing():
    """Blender hands back None from to_mesh() for an empty curve, and
    the bake runs after the old cage is gone - so raising here would
    cost the user the cage as well as the fibers."""
    _fresh()
    data = bpy.data.curves.new("empty", type="CURVE")
    data.dimensions = "3D"
    obj = bpy.data.objects.new("empty", data)
    bpy.context.collection.objects.link(obj)
    assert polyline_from_curve(bpy.context, obj).shape == (0, 3)


def test_tetrahedralize_bakes_fibers_when_a_curve_is_set():
    from marrow.blender.session import find_cage
    from marrow.blender.storage import read_fiber

    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    body.marrow.fiber_curve = _straight_curve("body_spine")
    bpy.ops.marrow.tetrahedralize()

    cage = find_cage(body)
    assert cage is not None
    fiber = read_fiber(cage.data)
    assert fiber is not None, "a curve was set but no fibers were baked"
    assert fiber.shape[1] == 5
    lengths = np.linalg.norm(fiber[:, :3], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5), "directions must be unit length"
    assert np.allclose(fiber[:, 1:3], 0.0, atol=1e-5), "a +X curve gives +X fibers"


def test_tetrahedralize_without_a_curve_bakes_nothing():
    from marrow.blender.session import find_cage
    from marrow.blender.storage import read_fiber

    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(20.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    bpy.ops.marrow.tetrahedralize()
    assert read_fiber(find_cage(body).data) is None
