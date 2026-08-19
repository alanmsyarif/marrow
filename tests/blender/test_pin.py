"""Pinning: a painted vertex group holds material in place.

Zero inverse mass was always in the solver - predict, integrate and every
contact pass branch on it, and the collide and attach kernels both say a
pin outranks them. Nothing exposed it, so a sticky collider was the only
way to hold material still. This is the exposure: a vertex group on the
render mesh, blended onto the cage through the same k-nearest map the
attachment pass already uses.

Weight scales inverse mass rather than switching it. 1.0 is a true pin;
lower is a heavier node, which smears the mass discontinuity at the edge
of the painted region instead of leaving a hard stress ring. A partial
weight is NOT partial holding: gravity is an acceleration, so a heavy
node still free-falls. It only resists constraints and contacts harder.
"""

import bpy
import gpu
import numpy as np

import marrow
from marrow.blender import handlers
from marrow.blender.session import MarrowSession
from marrow.core.tetmesh import MASS_DENSITY, node_volumes

gpu.init()

FRAMES = (1, 8)


def _fresh_addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()


def _cube(resolution=0.5):
    """A caged cube spanning -1..1, with the scene set to a short range."""
    _fresh_addon()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FRAMES
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.marrow.resolution = resolution
    return obj


def _paint(obj, name, indices, weight=1.0):
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    group.add(list(indices), float(weight), "REPLACE")
    return group


def _top_verts(obj):
    return [v.index for v in obj.data.vertices if v.co.z > 0.9]


def _free_inv_mass(session):
    """What inverse mass would be with nothing pinned."""
    mesh = session.tetmesh
    mass = node_volumes(mesh.nodes, mesh.tets) * MASS_DENSITY
    return 1.0 / np.maximum(mass, 1e-12)


def _session(obj):
    bpy.ops.marrow.tetrahedralize()
    return MarrowSession(obj)


def _panel_session(obj):
    """A session built the way the operators build one.

    MarrowSession(obj) reads only a few settings off the panel - resolution
    and the pin fields. Attachment arrives as an explicit kwarg, so anything
    testing attachment has to go through the panel path or the checkbox is
    silently ignored.
    """
    from marrow.blender.ops import session_for

    bpy.ops.marrow.tetrahedralize()
    return session_for(obj)


def test_a_fully_painted_group_pins_every_node():
    """Every render vertex at 1.0 must leave no node with any mobility.

    The blend rows are normalised floats, so a row can sum to
    0.9999999999999999 and leave 1e-16 of inverse mass behind. Predict
    only asks ``w > 0.0``, so that speck still takes the full gravity
    step and the "pinned" body sails away. Rounding has to be snapped.
    """
    obj = _cube()
    _paint(obj, "Pin", range(len(obj.data.vertices)), 1.0)
    obj.marrow.pin_group = "Pin"
    session = _session(obj)
    try:
        assert np.all(session.inv_mass == 0.0), (
            f"{int(np.count_nonzero(session.inv_mass))} of "
            f"{session.inv_mass.size} nodes kept inverse mass under a "
            f"fully painted pin group"
        )
    finally:
        session.free()


def test_pinned_material_holds_while_the_rest_hangs():
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    session = _session(obj)
    try:
        start = session.tetmesh.nodes.copy()
        pinned = session.inv_mass == 0.0
        assert pinned.any(), "painting the top face pinned no cage node"
        assert not pinned.all(), "the whole cage was pinned; nothing can hang"
        for frame in range(FRAMES[0], FRAMES[1] + 1):
            session.solver.step()
        end = session.solver.positions()
        moved = np.abs(end - start).max(axis=1)
        assert moved[pinned].max() < 1e-6, (
            f"a pinned node moved {moved[pinned].max():.6f}"
        )
        assert moved[~pinned].max() > 1e-3, (
            "nothing hung: the free half of the cage never moved"
        )
    finally:
        session.free()


def test_no_pin_group_leaves_every_node_free():
    """Control. The default path must be exactly what it was before."""
    obj = _cube()
    session = _session(obj)
    try:
        assert np.array_equal(session.inv_mass, _free_inv_mass(session))
    finally:
        session.free()


def test_a_group_name_that_no_longer_exists_is_ignored():
    """Renaming or deleting a group must not break the bake."""
    obj = _cube()
    obj.marrow.pin_group = "Gone"
    session = _session(obj)
    try:
        assert np.array_equal(session.inv_mass, _free_inv_mass(session))
    finally:
        session.free()


def test_a_partial_weight_is_a_heavier_node_not_a_pin():
    obj = _cube()
    _paint(obj, "Pin", range(len(obj.data.vertices)), 0.5)
    obj.marrow.pin_group = "Pin"
    session = _session(obj)
    try:
        assert np.allclose(session.inv_mass, _free_inv_mass(session) * 0.5), (
            "a 0.5 weight must halve inverse mass everywhere"
        )
        assert np.all(session.inv_mass > 0.0), "0.5 is not a pin"
    finally:
        session.free()


def test_a_restart_picks_up_a_repainted_group():
    """Inverse mass is built at solver build, not at session construction.

    Live restarts and Bake both go through refresh_from_object plus
    _build_solver, so a group repainted between takes has to land there or
    editing the weights would silently do nothing until Free.
    """
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    session = _session(obj)
    try:
        before = int(np.count_nonzero(session.inv_mass == 0.0))
        _paint(obj, "Pin", range(len(obj.data.vertices)), 1.0)
        session.refresh_from_object()
        session._build_solver()
        after = int(np.count_nonzero(session.inv_mass == 0.0))
        assert after > before, (
            f"repainting the group changed nothing: {before} pinned before, "
            f"{after} after"
        )
        assert after == session.inv_mass.size
    finally:
        session.free()



def test_the_follows_flag_reaches_the_solver():
    """The panel checkbox has to arrive at the GLSL uniform, or the pin is
    still frozen however the box is ticked."""
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.5
    obj.marrow.pin_follows = True
    session = _session(obj)
    try:
        assert session.pin_kinematic is True
        assert session.solver.pin_kinematic is True
    finally:
        session.free()


def test_pins_are_frozen_unless_the_flag_is_set():
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.5
    session = _session(obj)
    try:
        assert session.pin_kinematic is False
        assert session.solver.pin_kinematic is False
    finally:
        session.free()


def test_a_restart_picks_up_the_follows_flag():
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.5
    session = _session(obj)
    try:
        assert session.solver.pin_kinematic is False
        obj.marrow.pin_follows = True
        session.refresh_from_object()
        session._build_solver()
        assert session.solver.pin_kinematic is True
    finally:
        session.free()


def test_zero_attach_stiffness_still_drives_the_pins():
    """Attach Stiffness 0 with Follows Animation on is the setting the
    feature is used at: targets for the pins, hands off the free material.
    It used to disable the attachment pass outright, which left the pin
    frozen and made the useful region of the slider reachable only by
    typing a near-zero number.
    """
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.0
    obj.marrow.pin_follows = True
    session = _panel_session(obj)
    try:
        assert session.solver.attach_enabled is True, (
            "the pass must run so the pins get their targets"
        )
        assert session.solver.drive_free is False, (
            "free material must be left to the elastic solve"
        )
        assert session.solver.pin_kinematic is True
    finally:
        session.free()


def test_zero_attach_stiffness_without_a_pin_runs_no_attachment():
    obj = _cube()
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = 0.0
    session = _panel_session(obj)
    try:
        assert session.solver.attach_enabled is False
    finally:
        session.free()


def test_follows_animation_still_needs_attachment():
    obj = _cube()
    _paint(obj, "Pin", _top_verts(obj), 1.0)
    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = False
    obj.marrow.pin_follows = True
    session = _panel_session(obj)
    try:
        assert session.solver.attach_enabled is False, (
            "no targets exist without Attachment, so the pin stays frozen"
        )
    finally:
        session.free()


def _hooked_cube(attach_k, follows):
    """A cube whose top face is hooked to a linearly animated Empty.

    The user-facing shape of this feature: a hook drags a painted region and
    the pin is expected to ride it.
    """
    obj = _cube(resolution=0.4)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FRAMES
    top = _top_verts(obj)
    _paint(obj, "Pin", top, 1.0)

    bpy.ops.object.empty_add(location=(0.0, 0.0, 1.0))
    empty = bpy.context.active_object
    scene.frame_set(FRAMES[0])
    empty.location = (0.0, 0.0, 1.0)
    empty.keyframe_insert("location")
    scene.frame_set(FRAMES[1])
    empty.location = (2.0, 0.0, 1.0)
    empty.keyframe_insert("location")
    action = empty.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                bag = strip.channelbag(slot)
                if bag is None:
                    continue
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = "LINEAR"
    scene.frame_set(FRAMES[0])

    bpy.context.view_layer.objects.active = obj
    hook = obj.modifiers.new("Hook", "HOOK")
    hook.object = empty
    hook.vertex_group = "Pin"
    hook.matrix_inverse = empty.matrix_world.inverted()

    obj.marrow.pin_group = "Pin"
    obj.marrow.attach_enabled = True
    obj.marrow.attach_stiffness = attach_k
    obj.marrow.pin_follows = follows
    bpy.ops.marrow.tetrahedralize()
    return obj


def test_a_pins_only_bake_still_advances_the_scene():
    """Targets are resampled from the evaluated mesh, which only changes
    when the bake walks the scene forward. That walk was gated on
    ``attach_stiffness > 0``, so pins-only baked every frame against the
    start pose and the pin never moved - the animation was invisible to a
    mode built entirely to follow it.
    """
    obj = _hooked_cube(attach_k=0.0, follows=True)
    assert bpy.ops.marrow.bake() == {"FINISHED"}
    session = handlers.SESSIONS[obj.name]
    try:
        pinned = session.inv_mass == 0.0
        assert pinned.any()
        first = session._cache_nodes[FRAMES[0]].astype(float)
        last = session._cache_nodes[FRAMES[1]].astype(float)
        travel = np.linalg.norm(last[pinned] - first[pinned], axis=1).mean()
        assert travel > 0.5, (
            f"a driven pin moved {travel:.4f} while its Empty moved 2.0"
        )
    finally:
        handlers.free_all()
