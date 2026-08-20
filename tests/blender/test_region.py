"""Per-region stiffness: a painted vertex group makes one body two materials.

The group rides the same k-nearest map the pin group and the attachment
pass already use, so nothing new is baked and nothing is stored on the cage.
That is the whole point of the design, and it is what these tests pin down:
repaint, restart, and the solver sees it.
"""

import bpy
import gpu
import numpy as np

import marrow
from marrow.blender.session import MarrowSession

gpu.init()


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _caged_cube(resolution=0.5):
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    bpy.ops.marrow.tetrahedralize()
    return obj


def _paint(obj, name, indices, weight=1.0):
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    group.add(list(indices), float(weight), "REPLACE")
    return group


def _session(obj):
    """A session built the way Live and Bake build one: panel first."""
    session = MarrowSession(obj)
    session.refresh_from_object()
    session._build_solver()
    return session


def test_defaults_leave_the_body_uniform():
    obj = _caged_cube()
    assert obj.marrow.region_group == ""
    # float32 on the way in, so 0.1 does not come back as 0.1.
    assert abs(obj.marrow.region_softest - 0.1) < 1e-6
    assert _session(obj).region is None


def test_an_unpainted_group_name_is_still_uniform():
    """A group that exists but has no weights must not cost a k-nearest
    synthesis, and must not hand the solver an array of zeros."""
    obj = _caged_cube()
    obj.vertex_groups.new(name="stiff")
    obj.marrow.region_group = "stiff"
    assert _session(obj).region is None


def test_a_renamed_group_falls_back_to_uniform():
    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)))
    obj.marrow.region_group = "gone"
    assert _session(obj).region is None


def test_a_fully_painted_group_is_all_ones():
    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)), 1.0)
    obj.marrow.region_group = "stiff"
    region = _session(obj).region
    assert region is not None
    assert np.allclose(region, 1.0), region.min()


def test_an_unpainted_body_bottoms_out_at_softest():
    """Weight 0 everywhere maps to Softest, not to zero. The alternative
    turns every body with one painted region into soup."""
    obj = _caged_cube()
    group = obj.vertex_groups.new(name="stiff")
    # One vertex painted, so the group is non-empty; the rest stay at 0.
    group.add([0], 1.0, "REPLACE")
    obj.marrow.region_group = "stiff"
    obj.marrow.region_softest = 0.25
    region = _session(obj).region
    assert region is not None
    assert region.min() >= 0.25 - 1e-9, region.min()
    assert region.max() <= 1.0 + 1e-9, region.max()
    # The painted corner has to actually be stiffer than the far side.
    assert region.max() > region.min() + 1e-6


def test_softest_zero_reaches_zero():
    obj = _caged_cube()
    group = obj.vertex_groups.new(name="stiff")
    group.add([0], 1.0, "REPLACE")
    obj.marrow.region_group = "stiff"
    obj.marrow.region_softest = 0.0
    region = _session(obj).region
    assert region.min() < 1.0e-6, region.min()


def test_repainting_takes_effect_on_the_next_build():
    """No re-tetrahedralize: the group is read at solver build time, which
    is the difference between this and the fiber curve."""
    obj = _caged_cube()
    group = _paint(obj, "stiff", [0], 1.0)
    obj.marrow.region_group = "stiff"
    before = _session(obj).region.copy()
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    after = _session(obj).region
    assert not np.allclose(before, after)
    assert np.allclose(after, 1.0)


def test_the_region_is_one_value_per_tet():
    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)), 1.0)
    obj.marrow.region_group = "stiff"
    session = _session(obj)
    assert session.region.shape == (session.tetmesh.n_tets,)


def test_a_soft_body_deforms_further_than_a_stiff_one():
    """End to end through the GPU: same cage, same gravity, one softened.

    Not a parity check - the oracle diff owns that. This is the one test
    that would fail if the multiplier were computed correctly and then
    never reached tex_region."""
    spread = {}
    for softest in (1.0, 0.01):
        obj = _caged_cube()
        group = obj.vertex_groups.new(name="stiff")
        group.add([0], 1.0, "REPLACE")
        obj.marrow.region_group = "stiff"
        obj.marrow.region_softest = softest
        obj.marrow.ground_enabled = False
        session = _session(obj)
        for _ in range(8):
            session.solver.step()
        nodes = session.solver.positions()
        rest = session.tetmesh.nodes
        spread[softest] = float(np.abs(nodes - rest).max())
    assert spread[0.01] > spread[1.0], (
        f"soft cage moved {spread[0.01]:.4f}, stiff {spread[1.0]:.4f} - "
        "the multiplier is not reaching the solver"
    )


def test_switching_the_group_on_a_live_session_takes_effect():
    """The restart path, not a fresh session.

    Every other test here builds a new MarrowSession, whose constructor
    reads the panel - so none of them can see refresh_from_object failing
    to. That gap is what let a Stiffness Group work in a Bake and do
    nothing in Live.
    """
    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)), 1.0)
    session = _session(obj)
    assert session.region is None, "no group set yet"
    obj.marrow.region_group = "stiff"
    session.refresh_from_object()
    session._build_solver()
    assert session.region is not None, "refresh_from_object did not re-read the group"
    assert np.allclose(session.region, 1.0)


def test_changing_softest_on_a_live_session_takes_effect():
    obj = _caged_cube()
    group = obj.vertex_groups.new(name="stiff")
    group.add([0], 1.0, "REPLACE")
    obj.marrow.region_group = "stiff"
    obj.marrow.region_softest = 0.1
    session = _session(obj)
    assert session.region.min() < 0.2, session.region.min()
    obj.marrow.region_softest = 0.9
    session.refresh_from_object()
    session._build_solver()
    assert session.region.min() > 0.85, session.region.min()


def test_clearing_the_group_on_a_live_session_returns_to_uniform():
    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)), 1.0)
    obj.marrow.region_group = "stiff"
    session = _session(obj)
    assert session.region is not None
    obj.marrow.region_group = ""
    session.refresh_from_object()
    session._build_solver()
    assert session.region is None


def test_picking_a_group_mid_playback_restarts_the_body():
    """group.advance watches a short list of settings that change what
    _build_solver produces, so they take effect on the next frame instead of
    waiting for a manual trip to the start frame. Attachment and the pin
    settings were on that list; the Stiffness Group has to be too, or picking
    one at frame 20 does nothing and the feature reads as broken."""
    from marrow.blender import handlers

    obj = _caged_cube()
    _paint(obj, "stiff", range(len(obj.data.vertices)), 1.0)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 30
    try:
        for frame in range(1, 8):
            scene.frame_set(frame)
        session = handlers.SESSIONS.get(obj.name)
        assert session is not None
        assert session.region is None, "no group picked yet"

        obj.marrow.region_group = "stiff"
        scene.frame_set(8)
        assert session.region is not None, (
            "picking a Stiffness Group mid-playback did not restart the body"
        )
    finally:
        handlers.unregister_handler()


def test_changing_softest_mid_playback_restarts_the_body():
    from marrow.blender import handlers

    obj = _caged_cube()
    group = obj.vertex_groups.new(name="stiff")
    group.add([0], 1.0, "REPLACE")
    obj.marrow.region_group = "stiff"
    obj.marrow.region_softest = 0.1
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 30
    try:
        for frame in range(1, 8):
            scene.frame_set(frame)
        session = handlers.SESSIONS.get(obj.name)
        assert session.region.min() < 0.2

        obj.marrow.region_softest = 0.9
        scene.frame_set(8)
        assert session.region.min() > 0.85, (
            "changing Softest mid-playback did not restart the body"
        )
    finally:
        handlers.unregister_handler()
