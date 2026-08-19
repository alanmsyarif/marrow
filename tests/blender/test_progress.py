"""The chunked voxel pass and the modal Tetrahedralize pipeline.

At Resolution 0.08 on a 34k-vertex mesh the voxel pass is 54.7s of a 66.2s
run - a triple Python loop doing three ray casts per cell over 6.5M cells.
That is the freeze, so that is what has to yield.

The modal operator itself cannot run headless: it needs a window, a timer
and an event loop. The logic therefore lives in a plain generator that these
tests drive directly, and the operator stays a thin shell around it.
"""

import bpy
import marrow
import numpy as np

from marrow.blender.inside_bvh import cell_mask_from_object, cell_mask_iter
from marrow.core.progress import drain


def _blob(name="progress_blob"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.6)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _fractions(work):
    seen = []
    while True:
        try:
            seen.append(next(work))
        except StopIteration as done:
            return seen, done.value


def _assert_well_behaved(seen, what):
    assert seen, f"{what}: reported no progress at all"
    assert all(0.0 <= f <= 1.0 for f in seen), f"{what}: fraction out of range"
    assert seen == sorted(seen), f"{what}: progress went backwards"
    assert seen[-1] == 1.0, f"{what}: finished at {seen[-1]}, not 1.0"


def test_chunked_voxel_pass_matches_the_blocking_call():
    obj = _blob()
    try:
        seen, (mask, bounds_min) = _fractions(cell_mask_iter(obj, 0.2))
        _assert_well_behaved(seen, "cell_mask_iter")

        want_mask, want_min = cell_mask_from_object(obj, 0.2)
        assert np.array_equal(mask, want_mask), (
            f"occupancy differs: {int(mask.sum())} cells vs {int(want_mask.sum())}"
        )
        assert np.allclose(bounds_min, want_min)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_the_voxel_pass_yields_more_than_once_so_it_can_be_interrupted():
    """A generator that only yields at the end would still freeze Blender."""
    obj = _blob()
    try:
        seen, _ = _fractions(cell_mask_iter(obj, 0.1))
        assert len(seen) > 4, (
            f"only {len(seen)} yields - too coarse to keep the window alive"
        )
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_abandoning_the_voxel_pass_early_costs_nothing():
    """ESC has to be able to walk away mid-pass without a partial result."""
    obj = _blob()
    try:
        work = cell_mask_iter(obj, 0.1)
        next(work)
        work.close()          # what dropping the generator on ESC does

        # Nothing was written to the object, and a fresh pass is unaffected
        # by the abandoned one.
        assert "marrow_rest" not in obj.data.attributes
        mask, _ = drain(cell_mask_iter(obj, 0.2))
        assert np.array_equal(mask, cell_mask_from_object(obj, 0.2)[0])
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


# --- the pipeline the modal operator drives -------------------------------

from marrow.blender import ops as ops_mod          # noqa: E402
from marrow.blender import session as session_mod  # noqa: E402
from marrow.blender.session import find_cage       # noqa: E402


def _pipeline_obj():
    obj = _blob("progress_pipe")
    obj.marrow.resolution = 0.3
    return obj


def test_the_pipeline_reports_named_stages_and_finishes():
    obj = _pipeline_obj()
    try:
        work = ops_mod._tetrahedralize_iter(bpy.context, obj)
        labels, fractions = [], []
        while True:
            try:
                label, fraction = next(work)
            except StopIteration as done:
                level, message = done.value
                break
            labels.append(label)
            fractions.append(fraction)

        assert all(0.0 <= f <= 1.0 for f in fractions), "fraction out of range"
        assert "voxelising" in labels and "binding" in labels, (
            f"stages not reported: {sorted(set(labels))}"
        )
        assert level == "INFO", level
        assert "tets" in message and "nodes" in message, message
        assert find_cage(obj) is not None, "no cage was built"
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_cancelling_the_pipeline_builds_no_cage_and_unparks_modifiers():
    """What Esc does. The half-built state must not survive it."""
    obj = _pipeline_obj()
    obj.modifiers.new("subsurf", "SUBSURF")
    try:
        work = ops_mod._tetrahedralize_iter(bpy.context, obj)
        next(work)                      # into the voxel pass
        assert obj.modifiers[0].show_viewport is False, "modifier was not parked"

        work.close()                    # Esc

        assert obj.modifiers[0].show_viewport is True, (
            "modifier left parked after cancel - the mesh would render wrong"
        )
        assert find_cage(obj) is None, "cancel left a partial cage behind"
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


def test_a_cage_over_the_node_budget_warns_instead_of_reporting_success():
    """Bake refuses over-budget cages, and used to be the first to say so."""
    obj = _pipeline_obj()
    original = session_mod.MAX_NODES
    session_mod.MAX_NODES = 10
    try:
        level, message = drain(ops_mod._tetrahedralize_iter(bpy.context, obj))
        assert level == "WARNING", f"over-budget cage reported as {level}"
        assert "Bake will refuse" in message, message
        assert "10" in message, f"budget not named: {message}"
    finally:
        session_mod.MAX_NODES = original
        bpy.data.objects.remove(obj, do_unlink=True)


def test_a_cancelled_bake_keeps_the_frames_it_already_simulated():
    """Esc during a bake is an interruption, not a loss.

    The cache is keyed by frame, so stopping partway leaves whole frames
    behind. Throwing them away would make a long bake all-or-nothing, which
    is what the frame-sized slice exists to avoid.
    """
    obj = _blob("progress_bake")
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 40
    try:
        work = ops_mod._bake_iter(bpy.context, obj)
        for _ in range(3):
            next(work)
        work.close()

        session = ops_mod.handlers.SESSIONS.get(obj.name)
        assert session is not None, "cancel dropped the session entirely"
        assert len(session._cache) > 0, "cancel discarded every simulated frame"
        assert len(session._cache) < 40, (
            f"cache holds {len(session._cache)} frames - the bake did not stop"
        )
    finally:
        ops_mod.handlers.SESSIONS.clear()
        bpy.data.objects.remove(obj, do_unlink=True)
