import bpy
import numpy as np

from marrow.blender.inside_bvh import cell_mask_from_object


def _fresh_cube(size=2.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=size)
    return bpy.context.active_object


def test_cube_mask_is_fully_occupied_in_the_interior():
    obj = _fresh_cube(size=2.0)
    mask, bounds_min = cell_mask_from_object(obj, spacing=0.5)
    assert mask.dtype == bool
    assert mask.any(), "cube produced an empty mask"
    # Cube spans -1..1, so a 0.5 grid is 4x4x4 and every cell centre is inside.
    assert mask.shape == (4, 4, 4)
    assert mask.all()
    assert np.allclose(bounds_min, [-1.0, -1.0, -1.0], atol=1e-6)


def test_sphere_mask_excludes_corners():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    obj = bpy.context.active_object
    mask, _ = cell_mask_from_object(obj, spacing=0.25)
    assert mask.any()
    assert not mask.all(), "a sphere must leave the bounding-box corners empty"
    assert not mask[0, 0, 0]


def test_spacing_too_coarse_yields_empty_mask_not_a_crash():
    """A spacing far larger than the object must return cleanly, not raise."""
    obj = _fresh_cube(size=0.01)
    mask, bounds_min = cell_mask_from_object(obj, spacing=10.0)
    assert mask.shape == (1, 1, 1)
    assert mask.dtype == bool
    assert bounds_min.shape == (3,)
    assert np.all(np.isfinite(bounds_min))
