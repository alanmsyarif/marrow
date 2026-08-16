"""False color: stretch rainbow display on the render mesh.

Enabling a mode swaps a generated material into slot 0 and primes the point
attribute; simulated frames overwrite it with the per-tet metric; Off and
de-tetrahedralize both put the object's own shading back.
"""

import bpy
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.false_color import ATTR


def _cube(resolution=0.5):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    # High enough to fall a metre and squash on the ground plane before the
    # test's frame range runs out.
    obj.location = (0.0, 0.0, 2.0)
    obj.marrow.resolution = resolution
    bpy.ops.marrow.tetrahedralize()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 30
    return obj


def _values(obj):
    attr = obj.data.attributes.get(ATTR)
    assert attr is not None, "false color attribute missing"
    out = np.empty(len(obj.data.vertices), dtype=np.float32)
    attr.data.foreach_get("value", out)
    return out


def _drop_and_squash(obj, frames=16):
    obj.marrow.ground_enabled = True
    scene = bpy.context.scene
    for frame in range(1, frames + 1):
        scene.frame_set(frame)


def _max_deviation(obj, frames=16):
    """Largest |metric - rest| seen at any frame.

    Asserting on the last frame alone pins the test to the rebound timing,
    which the mass model moves: a lumped-mass body is heavier, hits harder,
    and is back near rest by frame 16 where a uniform-mass one was still
    squashed. Deformation at any frame proves the metric writes.
    """
    scene = bpy.context.scene
    peak = 0.0
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        values = _values(obj)
        assert np.all(np.isfinite(values))
        peak = max(peak, float(np.abs(values - 1.0).max()))
    return peak


def test_enabling_a_mode_swaps_the_material_and_primes_the_attribute():
    obj = _cube()
    original = bpy.data.materials.new("Original")
    obj.data.materials.append(original)
    try:
        obj.marrow.false_color = "STRETCH"
        assert obj.material_slots[0].material.name == "Marrow False Color (Stretch)"
        assert np.allclose(_values(obj), 1.0), "stretch must prime to its rest value"

        obj.marrow.false_color = "OFF"
        assert obj.material_slots[0].material == original
    finally:
        handlers.unregister_handler()


def test_an_object_without_a_material_gets_its_slot_back():
    obj = _cube()
    try:
        assert not obj.material_slots
        obj.marrow.false_color = "STRETCH"
        assert len(obj.material_slots) == 1
        obj.marrow.false_color = "OFF"
        assert not obj.material_slots, "the appended slot must go again"
    finally:
        handlers.unregister_handler()


def test_a_simulated_frame_writes_the_metric():
    obj = _cube()
    try:
        obj.marrow.false_color = "STRETCH"
        obj.marrow.ground_enabled = True
        assert _max_deviation(obj) > 0.01, (
            "a body squashed on the ground must not read rest everywhere"
        )
    finally:
        handlers.unregister_handler()


def test_a_mode_switched_on_after_a_bake_still_colours_cached_frames():
    obj = _cube()
    try:
        obj.marrow.ground_enabled = True
        bpy.context.scene.frame_end = 16
        assert bpy.ops.marrow.bake() == {"FINISHED"}

        obj.marrow.false_color = "STRETCH"
        scene = bpy.context.scene
        peak = 0.0
        for frame in range(1, 17):
            scene.frame_set(frame)
            peak = max(peak, float(np.abs(_values(obj) - 1.0).max()))
        assert peak > 0.01, (
            "cached frames must colour even though the mode came late"
        )
    finally:
        handlers.unregister_handler()


def test_detetrahedralize_restores_the_material_and_clears_the_attribute():
    obj = _cube()
    original = bpy.data.materials.new("Original")
    obj.data.materials.append(original)
    try:
        obj.marrow.false_color = "STRETCH"
        bpy.ops.marrow.detetrahedralize()
        assert obj.marrow.false_color == "OFF"
        assert obj.material_slots[0].material == original
        assert obj.data.attributes.get(ATTR) is None
    finally:
        handlers.unregister_handler()
def test_resetting_below_the_start_frame_repaints_the_rest_colours():
    """A reset puts the cage at rest, so the stretch display must say so.

    Without this the surface keeps the colours of the last simulated frame,
    which reads as a deformed body while the mesh on screen is back at rest.
    """
    obj = _cube()
    obj.marrow.false_color = "STRETCH"
    try:
        _drop_and_squash(obj)
        squashed = _values(obj).copy()
        assert not np.allclose(squashed, 1.0, atol=1e-3), (
            "setup: a squashed body should not be painted at rest"
        )

        bpy.context.scene.frame_set(0)
        assert np.allclose(_values(obj), 1.0, atol=1e-3), (
            f"reset left stale colours, range "
            f"{_values(obj).min():.3f}..{_values(obj).max():.3f}"
        )
    finally:
        handlers.unregister_handler()
