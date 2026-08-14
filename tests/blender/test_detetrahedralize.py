"""Taking a simulated object back to the mesh the user started with.

See docs/superpowers/specs/2026-08-13-marrow-de-tetrahedralize-design.md.

Simulation writes straight into mesh.vertices, so the restore is the whole
feature; removing the cage and the attributes is just deletion.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.storage import BIND_IDX, BIND_W, REST_KEY, read_rest

ATTRS = (REST_KEY, BIND_IDX) + BIND_W


def _setup(resolution=0.5):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    return obj


def _verts(obj):
    out = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", out)
    return out.reshape(-1, 3)


def _deform(obj, frames=4):
    """Simulate until the mesh has visibly moved."""
    before = _verts(obj)
    session = handlers.SESSIONS.get(obj.name)
    if session is None:
        from marrow.blender.ops import session_for

        session = session_for(obj)
        handlers.SESSIONS[obj.name] = session
        session.live = True
    for frame in range(1, frames + 1):
        session.write_to_mesh(obj, frame, frame_start=1)
    moved = float(np.abs(_verts(obj) - before).max())
    assert moved > 1e-4, f"the simulation did not deform the mesh: {moved}"
    return moved


def test_tetrahedralize_stores_the_rest_shape():
    obj = _setup()
    original = _verts(obj)
    bpy.ops.marrow.tetrahedralize()

    stored = read_rest(obj.data)
    assert stored is not None, "no rest attribute was written"
    assert np.array_equal(stored, original), "rest shape is not the modelled shape"


def test_round_trip_restores_the_mesh_exactly():
    """The test this feature exists for."""
    obj = _setup()
    original = _verts(obj)
    bpy.ops.marrow.tetrahedralize()
    _deform(obj)

    assert bpy.ops.marrow.detetrahedralize() == {"FINISHED"}
    assert np.array_equal(_verts(obj), original), (
        f"mesh not restored: max error "
        f"{float(np.abs(_verts(obj) - original).max())}"
    )


def test_it_removes_the_cage_and_every_attribute():
    obj = _setup()
    bpy.ops.marrow.tetrahedralize()
    cage_mesh_name = bpy.data.objects[f"{obj.name}_marrow_cage"].data.name

    bpy.ops.marrow.detetrahedralize()

    assert bpy.data.objects.get(f"{obj.name}_marrow_cage") is None, "cage object left"
    assert bpy.data.meshes.get(cage_mesh_name) is None, "cage mesh datablock left"
    for name in ATTRS:
        assert obj.data.attributes.get(name) is None, f"attribute {name} left behind"


def test_it_releases_the_session_and_stops_live():
    obj = _setup()
    bpy.ops.marrow.tetrahedralize()
    _deform(obj)
    obj.marrow.live_enabled = True
    assert obj.name in handlers.SESSIONS

    bpy.ops.marrow.detetrahedralize()

    assert obj.name not in handlers.SESSIONS, "session left registered"
    assert not obj.marrow.live_enabled, "live was left on and will rebuild a session"


def test_it_reports_rather_than_raising_on_a_plain_mesh():
    obj = _setup()
    assert bpy.ops.marrow.detetrahedralize() == {"CANCELLED"}


def test_a_no_op_run_does_not_touch_live_or_sessions():
    """Reporting 'nothing to remove' must not come with side effects."""
    obj = _setup()
    obj.marrow.live_enabled = True
    assert bpy.ops.marrow.detetrahedralize() == {"CANCELLED"}
    assert obj.marrow.live_enabled, "a no-op run flipped Live off"
    assert obj.name not in handlers.SESSIONS


def test_a_second_run_is_a_no_op():
    obj = _setup()
    bpy.ops.marrow.tetrahedralize()
    assert bpy.ops.marrow.detetrahedralize() == {"FINISHED"}
    assert bpy.ops.marrow.detetrahedralize() == {"CANCELLED"}


def test_it_still_cleans_up_when_the_cage_was_deleted_by_hand():
    obj = _setup()
    bpy.ops.marrow.tetrahedralize()
    bpy.data.objects.remove(
        bpy.data.objects[f"{obj.name}_marrow_cage"], do_unlink=True
    )

    assert bpy.ops.marrow.detetrahedralize() == {"FINISHED"}
    for name in ATTRS:
        assert obj.data.attributes.get(name) is None, f"attribute {name} left behind"


def test_re_tetrahedralizing_after_a_simulation_does_not_bake_the_pose_in():
    """Before the rest attribute existed, the second cage was built from the
    deformed mesh and the drift compounded on every Resolution change."""
    obj = _setup()
    bpy.ops.marrow.tetrahedralize()
    clean = np.array(
        [tuple(v.co) for v in bpy.data.objects[f"{obj.name}_marrow_cage"].data.vertices]
    )

    _deform(obj)
    bpy.ops.marrow.tetrahedralize()
    after = np.array(
        [tuple(v.co) for v in bpy.data.objects[f"{obj.name}_marrow_cage"].data.vertices]
    )

    assert after.shape == clean.shape, (
        f"cage changed size after a simulation: {clean.shape} then {after.shape}"
    )
    assert np.allclose(after, clean, atol=1e-9), "the deformed pose leaked into the cage"
