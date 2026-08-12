import bpy
import numpy as np

import marrow
from marrow.blender.storage import read_bind, read_tetmesh


def _setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    return bpy.context.active_object


def test_operator_creates_a_valid_cage():
    obj = _setup()
    obj.marrow.resolution = 0.5
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}

    cage_obj = bpy.data.objects.get(f"{obj.name}_marrow_cage")
    assert cage_obj is not None, "no cage object was created"
    tetmesh, colors = read_tetmesh(cage_obj.data)
    tetmesh.validate()
    assert tetmesh.n_tets > 0
    assert colors.shape == (tetmesh.n_tets,)


def test_operator_binds_every_render_vertex():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()

    idx, w = read_bind(obj.data)
    assert idx.shape[0] == len(obj.data.vertices)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert idx.min() >= 0


def test_cage_is_hidden_and_parented():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    cage_obj = bpy.data.objects[f"{obj.name}_marrow_cage"]
    assert cage_obj.parent == obj
    assert cage_obj.display_type == "WIRE"
    assert cage_obj.hide_render


def test_resolution_too_coarse_reports_an_error_not_a_crash():
    obj = _setup()
    obj.marrow.resolution = 100.0
    # An operator that reports {"ERROR"} and returns CANCELLED surfaces in
    # Python as RuntimeError: that is Blender promoting the report, not the
    # operator crashing, so the return value never reaches the caller. What
    # matters is that the user gets the fix instruction rather than a
    # traceback from somewhere inside numpy.
    try:
        bpy.ops.marrow.tetrahedralize()
    except RuntimeError as exc:
        assert "Lower Resolution" in str(exc), f"unhelpful error text: {exc}"
    else:
        raise AssertionError("expected a reported error at 100.0 resolution")


def test_rerunning_replaces_the_previous_cage():
    obj = _setup()
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    bpy.ops.marrow.tetrahedralize()
    matching = [o for o in bpy.data.objects if o.name.startswith(f"{obj.name}_marrow_cage")]
    assert len(matching) == 1, f"expected one cage, found {[o.name for o in matching]}"
