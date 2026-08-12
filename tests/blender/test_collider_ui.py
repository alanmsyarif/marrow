"""Colliders are picked from the soft body, not tagged on each object.

Walking to every object to tick a checkbox was backwards: colliders belong to
the body being simulated. The soft body owns a list of slots, each holding an
object and the shape to treat it as.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.ops import collider_objects_of
from marrow.blender.session import MarrowSession
from marrow.blender.ui import MARROW_PT_panel


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _soft_body(location=(0, 0, 3)):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    return obj


def test_a_body_starts_with_no_colliders():
    _fresh()
    obj = _soft_body()
    assert len(obj.marrow.colliders) == 0
    assert collider_objects_of(obj) == []


def test_add_and_remove_collider_slots():
    _fresh()
    obj = _soft_body()
    assert bpy.ops.marrow.collider_add() == {"FINISHED"}
    assert bpy.ops.marrow.collider_add() == {"FINISHED"}
    assert len(obj.marrow.colliders) == 2
    assert obj.marrow.active_collider == 1

    assert bpy.ops.marrow.collider_remove() == {"FINISHED"}
    assert len(obj.marrow.colliders) == 1


def test_remove_is_unavailable_with_an_empty_list():
    _fresh()
    _soft_body()
    from marrow.blender.ops import MARROW_OT_collider_remove

    assert MARROW_OT_collider_remove.poll(bpy.context) is False


def test_picking_an_object_into_a_slot_reaches_the_solver():
    _fresh()
    obj = _soft_body()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    ball = bpy.context.active_object

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()
    obj.marrow.colliders[0].object = ball
    obj.marrow.colliders[0].shape = "SPHERE"

    assert collider_objects_of(obj) == [(ball, "SPHERE")]
    session = MarrowSession(obj)
    session.refresh_from_object()
    session._build_solver()
    assert len(session.solver.colliders) == 1
    kind, _to_local, _to_world = session.solver.colliders[0]
    assert kind == 1, "SPHERE must reach the kernel as kind 1"


def test_an_empty_works_as_a_collider():
    """A primitive collider needs a transform and nothing else."""
    _fresh()
    obj = _soft_body()
    bpy.ops.object.empty_add(location=(0, 0, 0))
    empty = bpy.context.active_object

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()
    obj.marrow.colliders[0].object = empty
    obj.marrow.colliders[0].shape = "BOX"
    assert collider_objects_of(obj) == [(empty, "BOX")]


def test_an_empty_slot_is_skipped_not_an_error():
    _fresh()
    obj = _soft_body()
    bpy.ops.marrow.collider_add()          # left unset
    assert collider_objects_of(obj) == []
    session = MarrowSession(obj)
    session.refresh_from_object()
    session._build_solver()                 # must not raise


def test_a_body_pointed_at_itself_is_skipped():
    _fresh()
    obj = _soft_body()
    bpy.ops.marrow.collider_add()
    obj.marrow.colliders[0].object = obj
    assert collider_objects_of(obj) == [], "a body must not collide with itself"


def test_a_picked_collider_actually_stops_the_body():
    _fresh()
    obj = _soft_body(location=(0, 0, 3))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    ball = bpy.context.active_object

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()
    obj.marrow.colliders[0].object = ball

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 40
    assert bpy.ops.marrow.bake() == {"FINISHED"}
    session = handlers.SESSIONS[obj.name]
    session.write_to_mesh(obj, 40)

    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", co)
    local = co.reshape(n, 3)
    m = np.array(obj.matrix_world)
    world_z = (local @ m[:3, :3].T + m[:3, 3])[:, 2]
    handlers.unregister_handler()

    assert world_z.min() > 0.5, (
        f"body fell through the collider: lowest z {world_z.min():.3f}"
    )


def test_the_panel_is_for_meshes_again():
    _fresh()
    bpy.ops.object.empty_add()
    assert MARROW_PT_panel.poll(bpy.context) is False, (
        "colliders are picked from the body now, so an Empty needs no panel"
    )
    bpy.ops.mesh.primitive_cube_add()
    assert MARROW_PT_panel.poll(bpy.context) is True
