"""The fiber properties, the panel box that gates them, and the wiring
that carries them from the panel to the solver."""

import bpy
import gpu

import marrow

gpu.init()


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _body_with_cage(location, with_curve=True):
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    body = bpy.context.active_object
    body.marrow.resolution = 1.0
    if with_curve:
        data = bpy.data.curves.new(f"spine{location[0]}", type="CURVE")
        data.dimensions = "3D"
        spline = data.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (location[0] - 2.0, location[1], location[2], 1.0)
        spline.points[1].co = (location[0] + 2.0, location[1], location[2], 1.0)
        curve = bpy.data.objects.new(f"spine{location[0]}", data)
        bpy.context.collection.objects.link(curve)
        body.marrow.fiber_curve = curve
    bpy.ops.marrow.tetrahedralize()
    return body


def _session_for(body):
    """A session that has read the panel, the way Live and Bake both build one.

    MarrowSession's constructor takes explicit params; the panel is only
    consulted by refresh_from_object, so a test that skips it would be
    reading the defaults back rather than what it set.
    """
    from marrow.blender.session import MarrowSession

    session = MarrowSession(body)
    session.refresh_from_object()
    session._build_solver()
    return session


def test_the_defaults_are_inert():
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(40.0, 0.0, 0.0))
    settings = bpy.context.active_object.marrow
    assert settings.fiber_enabled is False
    assert settings.fiber_curve is None
    assert settings.waveform == "SMOOTH"


def test_the_wavelength_cannot_reach_zero():
    """The oracle and the kernel both divide by it unguarded, on purpose so
    the two stay identical. This minimum is the whole guard."""
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(45.0, 0.0, 0.0))
    settings = bpy.context.active_object.marrow
    settings.wave_length = 0.0
    assert settings.wave_length > 0.0, "wave_length must not clamp to zero"


def test_the_curve_slot_only_offers_curves():
    """The poll keeps a mesh out of the picker.

    Measured on Blender 5.2: a pointer poll is NOT run on assignment from
    script, so `settings.fiber_curve = some_mesh` sticks. The poll is
    therefore tested for what it does - filter the picker, the only way a
    user fills the slot by hand - and the script route is covered by
    test_fiber_bake, where a non-curve bakes no fibers.
    """
    from marrow.blender.ui import _poll_curve

    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(50.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    curve_obj = bpy.data.objects.new(
        "spine", bpy.data.curves.new("spine", type="CURVE")
    )
    bpy.context.collection.objects.link(curve_obj)

    assert _poll_curve(None, curve_obj) is True, "a curve must be offered"
    assert _poll_curve(None, mesh_obj) is False, "a mesh must not be offered"


def test_session_passes_fiber_stiffness_through_when_enabled():
    body = _body_with_cage((60.0, 0.0, 0.0))
    body.marrow.fiber_enabled = True
    body.marrow.fiber_stiffness = 1234.0
    session = _session_for(body)
    assert abs(session.params.fiber_k - 1234.0) < 1e-6
    assert session.solver is not None


def test_fiber_disabled_zeroes_the_stiffness():
    body = _body_with_cage((70.0, 0.0, 0.0))
    body.marrow.fiber_enabled = False
    body.marrow.fiber_stiffness = 1234.0
    session = _session_for(body)
    assert session.params.fiber_k == 0.0


def test_a_cage_with_no_fibers_still_builds_a_session():
    body = _body_with_cage((80.0, 0.0, 0.0), with_curve=False)
    body.marrow.fiber_enabled = True
    session = _session_for(body)
    assert session.solver is not None, "fiber on without baked data must not crash"
    assert session.fiber is None


def _drawn_for(obj):
    """Panel controls actually offered, via the walker test_panel_gating
    owns. Fiber lives in a sub-panel now, so a helper that drew only the
    parent would see none of it - and the copy that used to live here would
    have gone on passing by asserting nothing."""
    from test_panel_gating import _drawn_for as walk

    return walk(obj)


def test_the_panel_offers_the_wave_controls_once_fibers_are_baked():
    body = _body_with_cage((90.0, 0.0, 0.0))
    body.marrow.fiber_enabled = True
    drawn = _drawn_for(body)
    for name in ("fiber_enabled", "fiber_curve", "wave_amplitude",
                 "wave_length", "wave_speed", "waveform"):
        assert ("prop", name) in drawn, f"{name} was not offered"


def test_the_panel_explains_itself_when_no_fibers_are_baked():
    """The curve is baked at Tetrahedralize, so setting it in the panel does
    nothing on its own. That must be said, not left to look broken."""
    body = _body_with_cage((100.0, 0.0, 0.0), with_curve=False)
    drawn = _drawn_for(body)
    labels = [text for kind, text in drawn if kind == "label"]
    assert any("Tetrahedralize" in text for text in labels), (
        f"no explanation offered, got {labels}"
    )
    assert ("prop", "wave_amplitude") not in drawn, (
        "wave controls drive nothing without baked fibers"
    )
