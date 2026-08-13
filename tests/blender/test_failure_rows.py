"""The spec's remaining failure-handling rows."""

import bpy

import marrow
from marrow.blender import group as group_mod
from marrow.blender import ops as ops_mod
from marrow.blender import session as session_mod
from marrow.gpu import capability
from marrow.gpu.solver import MarrowNaNError


def _cube():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = 0.5
    return obj


def test_gpu_is_available_on_this_machine():
    assert capability.gpu_available() is True


def test_bake_refuses_plainly_when_there_is_no_gpu():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    original = capability.gpu_available
    capability.gpu_available = lambda: False
    try:
        bpy.ops.marrow.bake()
    except RuntimeError as exc:
        assert "GPU" in str(exc), f"error must name the GPU: {exc}"
    else:
        raise AssertionError("bake must refuse without a usable GPU context")
    finally:
        capability.gpu_available = original


def test_register_succeeds_even_without_a_gpu():
    """The panel must still appear so the user can read the message."""
    original = capability.gpu_available
    capability.gpu_available = lambda: False
    try:
        marrow.unregister()
        marrow.register()  # must not raise
        assert hasattr(bpy.types.Object, "marrow")
    finally:
        capability.gpu_available = original


def test_a_nan_bake_reports_an_error_not_a_traceback():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()

    # Bake drives group.bake, because bodies that collide are simulated
    # together and a session cannot bake itself out of that.
    original = group_mod.bake

    def exploding_bake(members, start, end, scene=None):
        raise MarrowNaNError(
            "Marrow produced a non-finite frame at 3. Nothing was cached. "
            "Raise Substeps, or lower Stiffness and Volume Preservation, "
            "and bake again."
        )

    group_mod.bake = exploding_bake
    try:
        bpy.ops.marrow.bake()
    except RuntimeError as exc:
        assert "Raise Substeps" in str(exc), f"guidance missing: {exc}"
    else:
        raise AssertionError("a NaN bake must report an error")
    finally:
        group_mod.bake = original


def test_node_budget_refusal_names_the_count():
    obj = _cube()
    bpy.ops.marrow.tetrahedralize()
    original = session_mod.MAX_NODES
    session_mod.MAX_NODES = 10
    try:
        bpy.ops.marrow.bake()
    except RuntimeError as exc:
        assert "125" in str(exc) and "10" in str(exc), f"counts missing: {exc}"
    else:
        raise AssertionError("an over-budget cage must be refused")
    finally:
        session_mod.MAX_NODES = original
