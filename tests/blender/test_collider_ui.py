"""Colliders come from a collection on the soft body.

The body points at one Blender collection and every object in it is a
collider, nested collections included. Shape and stickiness live on the
collider object itself, so the same object can serve several bodies without
its settings being duplicated per body.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.ops import collider_objects_of, migrate_collider_slots
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


def _ball(location=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=location)
    return bpy.context.active_object


def test_a_body_starts_with_no_collider_collection():
    _fresh()
    obj = _soft_body()
    assert obj.marrow.collider_collection is None
    assert collider_objects_of(obj) == []


def test_collider_add_links_the_selected_objects():
    _fresh()
    obj = _soft_body()
    ball = _ball()

    bpy.context.view_layer.objects.active = obj
    assert bpy.ops.marrow.collider_add() == {"FINISHED"}

    collection = obj.marrow.collider_collection
    assert collection is not None, "add must create a collection to hold them"
    assert ball.name in collection.objects, "the selected object was not linked"
    assert obj.name not in collection.objects, "the body must not collide with itself"


def test_collider_remove_unlinks_the_active_row():
    _fresh()
    obj = _soft_body()
    ball = _ball()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()

    obj.marrow.active_collider = 0
    assert bpy.ops.marrow.collider_remove() == {"FINISHED"}
    assert ball.name not in obj.marrow.collider_collection.objects


def test_remove_is_unavailable_with_no_collection():
    _fresh()
    _soft_body()
    from marrow.blender.ops import MARROW_OT_collider_remove

    assert MARROW_OT_collider_remove.poll(bpy.context) is False


def test_a_collider_in_the_collection_reaches_the_solver():
    _fresh()
    obj = _soft_body()
    ball = _ball()
    ball.marrow_collider.shape = "SPHERE"

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()

    assert collider_objects_of(obj) == [(ball, "SPHERE", False)]
    session = MarrowSession(obj)
    session.refresh_from_object()
    session._build_solver()
    assert len(session.solver.colliders) == 1
    kind, _to_local, _to_world, sticky, _field = session.solver.colliders[0]
    assert sticky is False, "a collider is not sticky until the object says so"
    assert kind == 1, "SPHERE must reach the kernel as kind 1"


def test_an_empty_works_as_a_collider():
    """A primitive collider needs a transform and nothing else."""
    _fresh()
    obj = _soft_body()
    bpy.ops.object.empty_add(location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.marrow_collider.shape = "BOX"

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()
    assert collider_objects_of(obj) == [(empty, "BOX", False)]


def test_an_empty_collection_is_skipped_not_an_error():
    _fresh()
    obj = _soft_body()
    obj.marrow.collider_collection = bpy.data.collections.new("Empty Colliders")

    assert collider_objects_of(obj) == []
    session = MarrowSession(obj)
    session.refresh_from_object()
    session._build_solver()                 # must not raise


def test_a_body_inside_its_own_collider_collection_is_skipped():
    _fresh()
    obj = _soft_body()
    collection = bpy.data.collections.new("Colliders")
    collection.objects.link(obj)
    obj.marrow.collider_collection = collection

    assert collider_objects_of(obj) == [], "a body must not collide with itself"


def test_a_nested_collection_counts_as_colliders():
    _fresh()
    obj = _soft_body()
    ball = _ball()
    inner = bpy.data.collections.new("Inner")
    inner.objects.link(ball)
    outer = bpy.data.collections.new("Outer")
    outer.children.link(inner)
    obj.marrow.collider_collection = outer

    assert collider_objects_of(obj) == [(ball, "MESH", False)], (
        "objects in a nested collection must collide too"
    )


def test_old_collider_slots_migrate_to_a_collection():
    """A .blend saved before the collection rewrite keeps its colliders."""
    _fresh()
    obj = _soft_body()
    ball = _ball()

    slot = obj.marrow.colliders.add()
    slot.object = ball
    slot.shape = "SPHERE"
    slot.sticky = True

    migrate_collider_slots()

    assert len(obj.marrow.colliders) == 0, "migrated slots must be cleared"
    assert obj.marrow.collider_collection is not None
    assert ball.name in obj.marrow.collider_collection.objects
    assert ball.marrow_collider.shape == "SPHERE"
    assert ball.marrow_collider.sticky is True
    assert collider_objects_of(obj) == [(ball, "SPHERE", True)]


def test_a_collider_actually_stops_the_body():
    _fresh()
    obj = _soft_body(location=(0, 0, 3))
    _ball()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.marrow.collider_add()

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
    world = local @ m[:3, :3].T + m[:3, 3]
    handlers.unregister_handler()

    # The body was lighter when every node carried one mass unit, and then
    # it stopped on top of the ball with room to spare. Lumped mass makes
    # it heavy enough to squash and drape around the contact, so "stopped"
    # now means no vertex inside the unit sphere, not a height floor.
    assert np.linalg.norm(world, axis=1).min() > 0.95, (
        "body fell through the collider: closest vertex to the sphere "
        f"centre {np.linalg.norm(world, axis=1).min():.3f}"
    )


def test_the_panel_is_for_meshes_again():
    _fresh()
    bpy.ops.object.empty_add()
    assert MARROW_PT_panel.poll(bpy.context) is False, (
        "colliders are picked from the body now, so an Empty needs no panel"
    )
    bpy.ops.mesh.primitive_cube_add()
    assert MARROW_PT_panel.poll(bpy.context) is True
