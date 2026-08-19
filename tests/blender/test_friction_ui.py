"""The Friction sliders reach the solver.

Separate from test_friction.py on purpose. These reset Blender with
read_factory_settings and re-register the addon, which frees datablocks out
from under any GPU state a kernel test is holding - mixing the two in one
module made the contact tests fail intermittently. test_collider_ui.py keeps
the same separation.
"""

import numpy as np


def test_a_colliders_own_friction_reaches_the_solver():
    """Per-collider Friction is authored on the collider, not on the body.

    The value has to survive collider_objects_of, _collider_specs and the
    solver's collider tuple to mean anything, and every one of those is a
    positional tuple that has been extended before.
    """
    import bpy

    import marrow
    from marrow.blender.ops import collider_objects_of
    from marrow.blender.session import MarrowSession

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 3))
    body = bpy.context.active_object
    body.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()

    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    ball = bpy.context.active_object
    ball.marrow_collider.shape = "SPHERE"
    ball.marrow_collider.friction = 0.75

    bpy.context.view_layer.objects.active = body
    bpy.ops.marrow.collider_add()

    got = collider_objects_of(body)
    assert len(got) == 1 and got[0][:3] == (ball, "SPHERE", False), got
    assert np.isclose(got[0][3], 0.75), f"friction lost on the way out: {got}"

    session = MarrowSession(body)
    session.refresh_from_object()
    session._build_solver()
    entry = session.solver.colliders[0]
    assert len(entry) == 6, f"collider tuple lost a field: {len(entry)}"
    assert np.isclose(entry[5], 0.75), f"collider friction did not arrive: {entry[5]}"


def test_the_bodys_own_friction_reaches_the_solver():
    """The body-level slider, which the ground and both contact passes read."""
    import bpy

    import marrow
    from marrow.blender.session import MarrowSession

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 3))
    body = bpy.context.active_object
    body.marrow.resolution = 0.5
    body.marrow.friction = 0.4
    bpy.ops.marrow.tetrahedralize()

    session = MarrowSession(body)
    session.refresh_from_object()
    session._build_solver()
    assert np.isclose(session.solver.friction, 0.4), (
        f"body friction did not arrive: {session.solver.friction}"
    )
