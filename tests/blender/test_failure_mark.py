"""The marrow_torn attribute: what makes failed material visible downstream.

Marrow never changes your topology, so the render mesh cannot come apart on
its own however much of the cage fails - the two halves stay joined by a
very thin filament. This attribute marks the failed material per vertex so
Geometry Nodes can delete it, which is where the break actually happens.
"""

import bpy
import gpu
import numpy as np

import marrow
from marrow.blender import handlers

gpu.init()


def _hanging_bar(tearing, strain=1.2, resolution=0.12):
    """A bar pinned at the top under heavy gravity, so it fails on its own."""
    import bmesh

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    handlers.free_all()

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -1.0))
    body = bpy.context.active_object
    body.scale = (0.3, 0.3, 2.0)
    bpy.ops.object.transform_apply(scale=True)
    for _ in range(2):
        bpy.ops.object.modifier_add(type="SUBSURF")
        body.modifiers[-1].subdivision_type = "SIMPLE"
        body.modifiers[-1].levels = 1
        bpy.ops.object.modifier_apply(modifier=body.modifiers[-1].name)

    body.marrow.resolution = resolution
    body.marrow.stiffness = 2000.0
    body.marrow.substeps = 20
    body.marrow.ground_enabled = False
    body.marrow.gravity_scale = 6.0
    body.marrow.tearing_enabled = tearing
    body.marrow.tear_threshold = strain
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}

    v = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    top = np.nonzero(v[:, 2] > v[:, 2].max() - 0.12)[0]
    group = body.vertex_groups.new(name="hold")
    group.add(top.tolist(), 1.0, "REPLACE")
    body.marrow.pin_group = "hold"

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 30
    return body


def _torn_values(obj):
    from marrow.blender.storage import TORN_ATTR

    attr = obj.data.attributes.get(TORN_ATTR)
    if attr is None:
        return None
    out = np.empty(len(obj.data.vertices), dtype=np.float32)
    attr.data.foreach_get("value", out)
    return out


def test_failure_marks_the_render_mesh():
    """Marrow will not change your topology, so the mesh cannot come apart on
    its own. The attribute is what lets Geometry Nodes do it instead - without
    it the failed material is invisible to everything downstream."""
    body = _hanging_bar(True)
    try:
        for f in range(1, 31):
            bpy.context.scene.frame_set(f)
        values = _torn_values(body)
    finally:
        handlers.unregister_handler()

    assert values is not None, "no marrow_torn attribute was written"
    share = float(values.mean())
    assert 0.0 < share < 1.0, (
        f"{share:.2%} of vertices marked - a useful selection is neither "
        f"nothing nor everything"
    )


def test_no_attribute_without_failure():
    """The readback behind it is an image the size of the cage, so a body
    that can never fail must not pay for it."""
    body = _hanging_bar(False)
    try:
        for f in range(1, 31):
            bpy.context.scene.frame_set(f)
        assert _torn_values(body) is None
    finally:
        handlers.unregister_handler()


def test_the_mark_grows_over_the_shot():
    """Read from the recorded failure frames, not from the GPU, so scrubbing
    shows what had failed by then rather than what fails by the end."""
    body = _hanging_bar(True)
    try:
        for f in range(1, 31):
            bpy.context.scene.frame_set(f)
        end = float(_torn_values(body).mean())
        bpy.context.scene.frame_set(2)
        early = float(_torn_values(body).mean())
    finally:
        handlers.unregister_handler()

    assert early < end, (
        f"frame 2 shows {early:.2%} and frame 30 shows {end:.2%} - scrubbing "
        f"back must not show the final state"
    )


def test_de_tetrahedralize_removes_the_mark():
    body = _hanging_bar(True)
    try:
        for f in range(1, 31):
            bpy.context.scene.frame_set(f)
        assert _torn_values(body) is not None
    finally:
        handlers.unregister_handler()
    assert bpy.ops.marrow.detetrahedralize() == {"FINISHED"}
    assert _torn_values(body) is None, "a freed body must not keep the mark"
