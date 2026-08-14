"""The panel shows nothing but the Cage box until a cage exists.

Every control below Cage acts on a cage: the sliders feed a solver that has
none, and Live, Bake and Free can only report the same "run Tetrahedralize
first" back. Drawing them on a plain mesh offers work that cannot be done.
"""

import bpy

import marrow
from marrow.blender.ui import MARROW_PT_panel


class _Layout:
    """The smallest stand-in for UILayout that ``draw`` needs.

    Blender hands out a real UILayout only inside a draw callback, so the
    panel is driven against a recorder instead. Boxes, rows and columns all
    append into the one ``drawn`` list: what matters here is whether a
    control was offered at all, not which sub-layout it landed in.
    """

    def __init__(self, drawn=None):
        self.drawn = [] if drawn is None else drawn
        self.enabled = True

    def box(self):
        return _Layout(self.drawn)

    def row(self, align=False):
        return _Layout(self.drawn)

    def column(self, align=False):
        return _Layout(self.drawn)

    def label(self, text="", icon=""):
        self.drawn.append(("label", text))

    def prop(self, data, name, **kwargs):
        self.drawn.append(("prop", name))

    def operator(self, idname, **kwargs):
        self.drawn.append(("operator", idname))

    def template_list(self, listtype, list_id, *args, **kwargs):
        self.drawn.append(("template_list", listtype))


class _Panel:
    """``draw`` only ever reaches for ``self.layout``."""

    def __init__(self, layout):
        self.layout = layout


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _drawn_for(obj):
    bpy.context.view_layer.objects.active = obj
    layout = _Layout()
    MARROW_PT_panel.draw(_Panel(layout), bpy.context)
    return layout.drawn


def test_a_mesh_with_no_cage_shows_only_the_cage_box():
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object

    drawn = _drawn_for(obj)

    assert ("operator", "marrow.tetrahedralize") in drawn, (
        "the way out of this state must still be offered"
    )
    assert ("label", "Simulation") not in drawn
    assert ("label", "Display") not in drawn
    assert ("operator", "marrow.bake") not in drawn
    assert ("operator", "marrow.live_toggle") not in drawn
    assert ("operator", "marrow.free") not in drawn
    assert ("prop", "substeps") not in drawn
    assert ("prop", "collider_collection") not in drawn


def test_a_mesh_with_no_cage_says_why_the_settings_are_missing():
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object

    labels = [text for kind, text in _drawn_for(obj) if kind == "label"]
    assert any("Tetrahedralize" in text for text in labels), (
        f"an empty panel needs a reason, got {labels}"
    )


def test_a_tetrahedralized_body_shows_the_simulation_settings():
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()

    drawn = _drawn_for(obj)

    assert ("label", "Simulation") in drawn
    assert ("label", "Display") in drawn
    assert ("prop", "substeps") in drawn
    assert ("prop", "collider_collection") in drawn
    assert ("operator", "marrow.bake") in drawn
    assert ("operator", "marrow.detetrahedralize") in drawn


def test_de_tetrahedralizing_hides_the_settings_again():
    _fresh()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = 0.5
    bpy.ops.marrow.tetrahedralize()
    assert ("label", "Simulation") in _drawn_for(obj)

    bpy.ops.marrow.detetrahedralize()
    assert ("label", "Simulation") not in _drawn_for(obj)
