"""False-color material for the stretch display.

Vellum-style: a point attribute carries one scalar per render vertex and a
generated material ramps it through a rainbow. While a mode is active the
material sits in slot 0 and the object's previous material is remembered in
an ID property, so choosing Off puts the user's shading back exactly as it
was - including the case of an object that had no material at all.
"""

import bpy
import numpy as np

ATTR = "marrow_falsecolor"
ORIG_KEY = "marrow_falsecolor_orig"
ORIG_COUNT = "marrow_falsecolor_slots"

# Map-range input span per mode. Stretch is a ratio around 1.
RANGES = {"STRETCH": (0.8, 1.2)}
# The value an undeformed body shows, so enabling a mode before simulating
# already displays a uniform neutral green rather than an empty attribute.
NEUTRAL = {"STRETCH": 1.0}

_RAINBOW = (
    (0.00, (0.0, 0.0, 1.0, 1.0)),
    (0.25, (0.0, 1.0, 1.0, 1.0)),
    (0.50, (0.0, 1.0, 0.0, 1.0)),
    (0.75, (1.0, 1.0, 0.0, 1.0)),
    (1.00, (1.0, 0.0, 0.0, 1.0)),
)


def _material(mode: str):
    """One shared material per mode, rebuilt so the range follows the mode."""
    name = f"Marrow False Color ({mode.title()})"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emit = tree.nodes.new("ShaderNodeEmission")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    map_range = tree.nodes.new("ShaderNodeMapRange")
    attr = tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = ATTR

    lo, hi = RANGES[mode]
    map_range.inputs["From Min"].default_value = lo
    map_range.inputs["From Max"].default_value = hi

    elements = ramp.color_ramp.elements
    elements[0].position, elements[0].color = _RAINBOW[0]
    elements[1].position, elements[1].color = _RAINBOW[-1]
    for position, color in _RAINBOW[1:-1]:
        element = ramp.color_ramp.elements.new(position)
        element.color = color

    tree.links.new(attr.outputs["Fac"], map_range.inputs["Value"])
    tree.links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def apply(obj, mode: str) -> None:
    """Swap the false-color material into slot 0, remembering what was there."""
    if ORIG_KEY not in obj:
        original = obj.material_slots[0].material if obj.material_slots else None
        obj[ORIG_KEY] = original.name if original is not None else ""
        obj[ORIG_COUNT] = len(obj.material_slots)

    material = _material(mode)
    if not obj.material_slots:
        obj.data.materials.append(material)
    else:
        obj.material_slots[0].material = material


def restore(obj) -> None:
    """Put back the material slot 0 held before apply(). Idempotent."""
    if ORIG_KEY not in obj:
        return
    name = obj[ORIG_KEY]
    count = int(obj.get(ORIG_COUNT, 0))
    if obj.material_slots:
        obj.material_slots[0].material = (
            bpy.data.materials.get(name) if name else None
        )
    # A slot we appended for an object that had no material goes again.
    # Popping mesh.materials by hand leaves the object's slot array out of
    # step with the data list - measured, a slot that nothing removes - so
    # removal goes through the same operator the UI uses, which must run
    # while the slot still holds our material.
    while len(obj.material_slots) > count:
        obj.active_material_index = len(obj.material_slots) - 1
        with bpy.context.temp_override(
            active_object=obj,
            object=obj,
            selected_objects=[obj],
            selected_editable_objects=[obj],
        ):
            bpy.ops.object.material_slot_remove()
    del obj[ORIG_KEY]
    if ORIG_COUNT in obj:
        del obj[ORIG_COUNT]


def write_attribute(mesh, values) -> None:
    attr = mesh.attributes.get(ATTR)
    if attr is None or attr.data_type != "FLOAT" or attr.domain != "POINT":
        if attr is not None:
            mesh.attributes.remove(attr)
        attr = mesh.attributes.new(name=ATTR, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", np.asarray(values, dtype=np.float32))
    mesh.update()


def clear_attribute(mesh) -> None:
    attr = mesh.attributes.get(ATTR)
    if attr is not None:
        mesh.attributes.remove(attr)
        mesh.update()


def prime(obj, mode: str) -> None:
    """Fill the attribute with the neutral value so the ramp shows something
    before the first simulated frame overwrites it."""
    write_attribute(
        obj.data, np.full(len(obj.data.vertices), NEUTRAL[mode], dtype=np.float32)
    )
