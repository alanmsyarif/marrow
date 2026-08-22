import bpy
import numpy as np

import marrow
from marrow.blender import session as session_mod
from marrow.blender.session import MarrowSession
from marrow.core.solver_ref import SolverParams


def _tetrahedralised_cube(resolution=0.5):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
    return obj


def _rest_positions(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def test_bake_stores_one_array_per_frame():
    obj = _tetrahedralised_cube()
    s = MarrowSession(obj)
    stored = s.bake(1, 5)
    assert stored == 5, f"expected 5 frames stored, got {stored}"
    assert s.baked_range == (1, 5)
    for frame in range(1, 6):
        arr = s.frame_positions(frame)
        assert arr is not None, f"frame {frame} missing from the cache"
        assert arr.shape == (len(obj.data.vertices), 3)
        assert np.all(np.isfinite(arr))


def test_baked_body_falls_under_gravity():
    obj = _tetrahedralised_cube()
    s = MarrowSession(obj)
    s.bake(1, 10)
    first = s.frame_positions(1)
    last = s.frame_positions(10)
    risen = np.flatnonzero(last[:, 2] >= first[:, 2])
    assert risen.size == 0, (
        f"body did not fall: mean z {first[:, 2].mean():.4f} -> "
        f"{last[:, 2].mean():.4f}; {risen.size} of {len(last)} vertices rose. "
        f"worst idx {risen[:4].tolist()} "
        f"first_z {np.round(first[risen[:4], 2], 5).tolist()} "
        f"last_z {np.round(last[risen[:4], 2], 5).tolist()} "
        f"first_all_zero={bool(np.allclose(first, 0.0))} "
        f"last_all_zero={bool(np.allclose(last, 0.0))}"
    )


def test_frame_outside_the_baked_range_is_none():
    obj = _tetrahedralised_cube()
    s = MarrowSession(obj)
    s.bake(5, 8)
    assert s.frame_positions(4) is None
    assert s.frame_positions(9) is None
    assert s.frame_positions(5) is not None


def test_write_to_mesh_moves_the_render_vertices():
    obj = _tetrahedralised_cube()
    rest = _rest_positions(obj)
    s = MarrowSession(obj)
    s.bake(1, 10)

    assert s.write_to_mesh(obj, 10) is True
    moved = _rest_positions(obj)
    assert not np.allclose(moved, rest, atol=1e-6), "mesh did not move"
    assert np.all(np.isfinite(moved))


def test_write_to_mesh_declines_an_unbaked_frame():
    obj = _tetrahedralised_cube()
    rest = _rest_positions(obj)
    s = MarrowSession(obj)
    s.bake(1, 3)
    assert s.write_to_mesh(obj, 99) is False
    assert np.allclose(_rest_positions(obj), rest, atol=1e-9), (
        "an unbaked frame must leave the mesh untouched"
    )


def test_cage_over_the_node_budget_is_refused_with_the_count():
    obj = _tetrahedralised_cube()
    original = session_mod.MAX_NODES
    session_mod.MAX_NODES = 10
    try:
        MarrowSession(obj)
    except ValueError as exc:
        text = str(exc)
        assert "125" in text, f"error must name the actual node count: {text}"
        assert "10" in text, f"error must name the budget: {text}"
    else:
        raise AssertionError("a cage over budget must be refused")
    finally:
        session_mod.MAX_NODES = original


def test_free_makes_the_session_refuse_further_work():
    obj = _tetrahedralised_cube()
    s = MarrowSession(obj)
    s.bake(1, 2)
    s.free()
    try:
        s.bake(1, 2)
    except RuntimeError as exc:
        assert "freed" in str(exc)
    else:
        raise AssertionError("a freed session must not silently keep working")


def test_substeps_setting_is_honoured():
    obj = _tetrahedralised_cube()
    slow = MarrowSession(obj, SolverParams(substeps=2))
    fast = MarrowSession(obj, SolverParams(substeps=20))
    slow.bake(1, 3)
    fast.bake(1, 3)
    assert not np.allclose(slow.frame_positions(3), fast.frame_positions(3), atol=1e-6), (
        "substep count had no effect on the result"
    )


def test_an_adaptive_cage_from_an_older_version_is_refused():
    """Cages the removed octree built carry hanging nodes at every
    fine-to-coarse face, held together by a blend pass that no longer
    exists. Running one anyway does not degrade gracefully - the unglued
    nodes drift off and the cage tears itself apart, which reads as a solver
    bug rather than a stale file. Refusing names the fix instead.
    """
    from marrow.blender.session import find_cage
    from marrow.blender.storage import BLEND_KEY, BLEND_W_KEY, has_legacy_blend

    obj = _tetrahedralised_cube()
    cage = find_cage(obj)
    assert not has_legacy_blend(cage.data), "a fresh cage has no glue rows"

    # Exactly what an adaptive cage left on disk.
    cage.data[BLEND_KEY] = [0, 1, 2, 3, 4]
    cage.data[BLEND_W_KEY] = [0.25, 0.25, 0.25, 0.25]
    assert has_legacy_blend(cage.data)

    try:
        MarrowSession(obj)
    except ValueError as exc:
        assert "Tetrahedralize" in str(exc), f"no fix offered: {exc}"
    else:
        raise AssertionError("an adaptive cage should be refused, not simulated")


def test_re_tetrahedralizing_clears_the_old_glue_rows():
    """The way out has to actually work: a rebuild must leave no trace of the
    old cage behind, or the refusal would be permanent."""
    from marrow.blender.session import find_cage
    from marrow.blender.storage import BLEND_KEY, BLEND_W_KEY, has_legacy_blend

    obj = _tetrahedralised_cube()
    cage = find_cage(obj)
    cage.data[BLEND_KEY] = [0, 1, 2, 3, 4]
    cage.data[BLEND_W_KEY] = [0.25, 0.25, 0.25, 0.25]

    assert bpy.ops.marrow.tetrahedralize() == {"FINISHED"}
    assert not has_legacy_blend(find_cage(obj).data)
    MarrowSession(obj)  # builds without raising


def _world_centre(obj):
    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world.to_4x4())
    return (co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]).mean(axis=0)


def _played(obj, frames=6):
    from marrow.blender import handlers

    # Belt and braces. A load clears SESSIONS now - see free_on_load - but
    # this helper builds its cube without one, and a leftover session keyed
    # by the same object name is exactly what this file is measuring against.
    handlers.free_all()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    try:
        for f in range(1, frames + 1):
            scene.frame_set(f)
        return _world_centre(obj)
    finally:
        handlers.unregister_handler()


def test_moving_the_object_and_restarting_moves_the_simulation():
    """Cage nodes are stored in the world frame Tetrahedralize saw. Without
    rebasing, moving the object left the body simulating and drawing where it
    used to be while its origin sat somewhere else - the mesh visibly
    detached from its own gizmo."""
    obj = _tetrahedralised_cube()
    obj.marrow.ground_enabled = False
    obj.location = (5.0, 3.0, 0.0)
    bpy.context.view_layer.update()

    centre = _played(obj)
    assert abs(centre[0] - 5.0) < 0.05 and abs(centre[1] - 3.0) < 0.05, (
        f"body did not follow the object: centre {np.round(centre, 3)}"
    )


def test_rotating_the_object_and_restarting_turns_the_simulation():
    obj = _tetrahedralised_cube()
    obj.marrow.ground_enabled = False
    obj.location = (0.0, 4.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 1.5707963)
    bpy.context.view_layer.update()

    centre = _played(obj)
    assert abs(centre[1] - 4.0) < 0.05, (
        f"a rotated object should still sit at its own origin: "
        f"{np.round(centre, 3)}"
    )


def test_scaling_after_tetrahedralize_is_refused():
    """A scaled delta would resize the cage under a rest shape measured
    before it, changing both the mass and what Stiffness means. Saying so
    beats simulating a different object."""
    obj = _tetrahedralised_cube()
    obj.scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    try:
        MarrowSession(obj)
    except ValueError as exc:
        assert "scale" in str(exc).lower(), f"unhelpful message: {exc}"
    else:
        raise AssertionError("a scaled object should be refused")


def test_an_untouched_object_is_unaffected():
    """The identity case has to stay exactly as it was - no rebase, no
    rounding through a matrix that is not doing anything."""
    obj = _tetrahedralised_cube()
    obj.marrow.ground_enabled = False
    centre = _played(obj)
    # Not zero: the solver is float32 and a readback lands a few parts in a
    # hundred thousand off. Three orders below that, and five below the 5.0
    # the rebase exists to remove.
    assert np.linalg.norm(centre[:2]) < 1e-3, np.round(centre, 6)
