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
