"""Marrow operators."""

import time

import bpy
import numpy as np

from ..blender import false_color, group, handlers
from ..blender.curve import polyline_from_curve
from ..blender.inside_bvh import cell_mask_iter, cell_oracle_from_object
from ..blender import session as session_mod
from ..blender.session import CAGE_SUFFIX, MarrowSession, find_cage
from ..blender.storage import (
    BIND_IDX,
    REST_KEY,
    clear_marrow_data,
    restore_rest,
    write_bind,
    write_blend,
    write_fiber,
    write_rest,
    write_tetmesh,
)
from ..core.adaptive import build_adaptive_lattice
from ..core.bind import bind_points_iter
from ..core.coloring import color_sets_iter
from ..core.fiber import fiber_from_polyline, tet_centroids
from ..core.lattice import build_lattice
from ..core.progress import drain
from ..core.solver_ref import SolverParams
from ..gpu import capability
from ..gpu.solver import MarrowNaNError


# How long one modal tick is allowed to work before handing control back.
# 50ms keeps the window repainting at ~20fps during a long cage build, which
# is the whole point: a blocking run reports Not Responding to Windows and
# reads as a crash, and people kill it.
_SLICE_SECONDS = 0.05


class _Abort(Exception):
    """A stage refused. The message is written for the user, not the log."""


def _stage(label, work):
    """Re-yield a sub-generator's progress under ``label``, return its value."""
    while True:
        try:
            fraction = next(work)
        except StopIteration as done:
            return done.value
        yield label, fraction


class _ModalPipeline:
    """Runs a (label, fraction) generator in slices, keeping Blender alive.

    Both long operators are the same shape: minutes of CPU work that used to
    sit in one uninterruptible execute(), which makes Windows mark the window
    Not Responding and reads to everyone as a crash. Subclasses supply the
    generator through _make_work and get two entry points for free -
    execute() drains it (scripts and tests, where nothing is watching) and
    invoke() runs it modally on a timer, which is what a panel click uses.
    """

    # Class-level defaults so _release is safe however the operator exits.
    _timer = None
    _work = None
    _label = ""

    def _make_work(self, context):
        raise NotImplementedError

    def _cancel_message(self):
        return "Marrow: cancelled"

    def execute(self, context):
        """Blocking path: scripts, tests, anything not driven by a click."""
        try:
            level, message = drain(self._make_work(context))
        except _Abort as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({level}, message)
        return {"FINISHED"}

    def invoke(self, context, event):
        """Clicked from the panel: run modally so the window stays alive."""
        self._work = self._make_work(context)
        self._label = "starting"
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 60.0, window=context.window)
        wm.modal_handler_add(self)
        self._show(context, 0.0)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._release(context)
            # close() raises GeneratorExit at the live yield, so the
            # generator's own finally blocks run and nothing is left parked.
            self._work.close()
            self.report({"INFO"}, self._cancel_message())
            return {"CANCELLED"}

        if event.type != "TIMER":
            # Swallow everything else: the scene must not be edited out from
            # under a half-finished run.
            return {"RUNNING_MODAL"}

        fraction = 0.0
        deadline = time.perf_counter() + _SLICE_SECONDS
        try:
            while time.perf_counter() < deadline:
                self._label, fraction = next(self._work)
        except StopIteration as done:
            self._release(context)
            level, message = done.value
            self.report({level}, message)
            return {"FINISHED"}
        except _Abort as exc:
            self._release(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self._release(context)
            self.report({"ERROR"}, f"Marrow: {exc}")
            return {"CANCELLED"}

        self._show(context, fraction)
        return {"RUNNING_MODAL"}

    def _show(self, context, fraction):
        context.workspace.status_text_set(
            f"Marrow: {self._label} {fraction * 100:.0f}%      [Esc] cancel"
        )

    def _release(self, context):
        context.workspace.status_text_set(None)
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


def _tetrahedralize_iter(context, obj):
    """The whole of Tetrahedralize, resumable, as (label, fraction) pairs.

    Written as a generator so the operator can run it a slice at a time and
    let Blender repaint in between - see MARROW_OT_tetrahedralize.modal. The
    blocking path drains the same generator, so there is one implementation
    of the pipeline rather than a fast one and a responsive one that drift.

    Nothing reaches the blend file before the bind write near the end, so
    abandoning this generator cannot leave a half-built cage. Esc closes it,
    which raises GeneratorExit at whichever yield is live and runs the
    finally below - so the parked modifiers come back on every exit path.
    """
    spacing = float(obj.marrow.resolution)

    # Build from the shape the user modelled, never from a simulated
    # pose. Without this, tetrahedralizing again after playing the
    # timeline would make the deformed mesh the new rest shape, and the
    # drift would compound every time Resolution was changed.
    restore_rest(obj.data)

    # The lattice, the bind and the stored rest must all describe the
    # SAME shape: the modelled base mesh. The inside test evaluates the
    # modifier stack, so with an armature playing a pose the cage would
    # fill the POSED silhouette while the bind reads the unposed base
    # mesh - two different frames, and the attachment weights synthesized
    # between them collapse onto a handful of vertices. Park every
    # modifier for the capture; sampling applies the same rule per frame.
    states = [(m, m.show_viewport) for m in obj.modifiers]
    for m, _ in states:
        m.show_viewport = False
    try:
        context.view_layer.update()
        blend_rows = None
        if obj.marrow.adaptive:
            # The octree follows the surface: boundary layer and thin
            # features at Min Size, bulk at Resolution.
            # ponytail: the octree build is one blocking call, so this stage
            # shows a label but no motion. Chunk it if adaptive cages start
            # taking long enough to look hung.
            yield "building octree", 0.0
            bounds_min, oracle = cell_oracle_from_object(obj)
            tetmesh, blend_idx, blend_w = build_adaptive_lattice(
                spacing, float(obj.marrow.min_resolution), oracle
            )
            if tetmesh.n_nodes == 0:
                raise _Abort(
                    "No cells inside the mesh. Lower Resolution in the "
                    "Marrow panel until the cage fills the object."
                )
            blend_rows = (blend_idx, blend_w)
        else:
            mask, bounds_min = yield from _stage(
                "voxelising", cell_mask_iter(obj, spacing)
            )
            if not mask.any():
                raise _Abort(
                    "No cells inside the mesh. Lower Resolution in the Marrow "
                    "panel until the cage fills the object."
                )
            tetmesh = build_lattice(bounds_min, spacing, mask)

        try:
            tetmesh.validate()
        except ValueError as exc:
            raise _Abort(f"Invalid cage: {exc}") from exc

        colors = yield from _stage(
            "colouring", color_sets_iter(tetmesh.tets, tetmesh.n_nodes)
        )

        render_verts = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", render_verts)
        render_verts = render_verts.reshape(-1, 3)
        world = np.array(obj.matrix_world.to_4x4())
        world_verts = render_verts @ world[:3, :3].T + world[:3, 3]

        bind_idx, bind_w = yield from _stage(
            "binding", bind_points_iter(tetmesh.nodes, tetmesh.tets, world_verts)
        )
        write_bind(obj.data, bind_idx, bind_w)
        # Captured after the restore above, so it is the modelled shape.
        write_rest(obj.data)
    finally:
        for m, state in states:
            m.show_viewport = state
        context.view_layer.update()

    # Fibers are baked here and frozen. The direction is rest-space, because
    # the constraint measures F a and F maps rest to world, so an animated
    # curve would have no meaning as a source. Changing the curve means
    # tetrahedralizing again, and the panel says so.
    #
    # Sampled in world space: that is the space tetmesh.nodes are in, and
    # therefore the space dm_inv is built from.
    #
    # Computed BEFORE remove_cage, written after. Curve evaluation depends
    # on nothing remove_cage touches, and putting it here means a raise in
    # it - an empty curve, a shape nobody anticipated - costs the user the
    # fibers and nothing else, instead of the cage that was already deleted.
    # It also reads the depsgraph before an object deletion invalidates it.
    spine = polyline_from_curve(context, obj.marrow.fiber_curve)
    fiber_rows = (
        fiber_from_polyline(spine, tet_centroids(tetmesh.nodes, tetmesh.tets))
        if spine.shape[0] >= 2
        else None
    )

    cage_name = f"{obj.name}{CAGE_SUFFIX}"
    remove_cage(obj)

    cage_mesh = bpy.data.meshes.new(cage_name)
    write_tetmesh(cage_mesh, tetmesh, colors)
    if blend_rows is not None:
        write_blend(cage_mesh, *blend_rows)
    if fiber_rows is not None:
        write_fiber(cage_mesh, fiber_rows)

    cage_obj = bpy.data.objects.new(cage_name, cage_mesh)
    context.collection.objects.link(cage_obj)
    cage_obj.parent = obj
    # Cage nodes are stored in world space. Assigning .parent in Python
    # leaves matrix_parent_inverse at identity, so the body's transform
    # gets applied on top and the cage draws in the wrong place - the
    # operator does this for you, plain assignment does not.
    cage_obj.matrix_parent_inverse = obj.matrix_world.inverted()
    cage_obj.display_type = "WIRE"
    cage_obj.hide_render = True
    cage_obj.hide_select = True

    # Live is the default, so a fresh cage is ready to play at once - no
    # button press, no bake.
    stale = handlers.SESSIONS.pop(obj.name, None)
    if stale is not None:
        stale.free()
    if obj.marrow.live_enabled:
        handlers.register_handler()

    message = (
        f"Marrow: {tetmesh.n_tets} tets, {tetmesh.n_nodes} nodes, "
        f"{int(colors.max()) + 1 if colors.size else 0} colours"
    )
    if blend_rows is not None:
        message += f", {blend_rows[0].shape[0]} blend rows"

    # Say it here rather than let Bake refuse after the wait. The budget is
    # checked when a session is built, which is long after this point.
    if tetmesh.n_nodes > session_mod.MAX_NODES:
        return "WARNING", (
            f"{message}. Over the {session_mod.MAX_NODES} node budget, so Bake will "
            f"refuse this cage - raise Resolution."
        )
    return "INFO", message


class MARROW_OT_tetrahedralize(_ModalPipeline, bpy.types.Operator):
    bl_idname = "marrow.tetrahedralize"
    bl_label = "Tetrahedralize"
    bl_description = "Fill the selected mesh with a tetrahedral cage"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def _make_work(self, context):
        return _tetrahedralize_iter(context, context.active_object)

    def _cancel_message(self):
        return "Marrow: tetrahedralize cancelled"


def remove_cage(obj) -> bool:
    """Delete ``obj``'s cage object and its mesh. False if it had none."""
    cage = find_cage(obj)
    if cage is None:
        return False
    cage_mesh = cage.data
    bpy.data.objects.remove(cage, do_unlink=True)
    if cage_mesh is not None and cage_mesh.users == 0:
        bpy.data.meshes.remove(cage_mesh)
    return True


class MARROW_OT_detetrahedralize(bpy.types.Operator):
    bl_idname = "marrow.detetrahedralize"
    bl_label = "De-tetrahedralize"
    bl_description = "Remove the cage and restore the object's original shape"
    bl_options = {"REGISTER", "UNDO"}

    # Modal state. Class-level defaults so _release is safe on any path.
    _timer = None
    _work = None
    _label = ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object

        # Look before touching anything. On a plain mesh this operator must be
        # a true no-op, not flip Live off and free a session as side effects
        # of reporting there was nothing to do.
        has_work = (
            find_cage(obj) is not None
            or obj.data.attributes.get(REST_KEY) is not None
            or obj.data.attributes.get(BIND_IDX) is not None
        )
        if not has_work:
            self.report({"INFO"}, "Marrow: nothing to remove")
            return {"CANCELLED"}

        # The session goes first. Restoring positions while a live session is
        # still registered would have the next frame change overwrite them.
        session = handlers.SESSIONS.pop(obj.name, None)
        if session is not None:
            session.free()
        obj.marrow.live_enabled = False
        # The update callback puts the object's own material back in slot 0.
        if obj.marrow.false_color != "OFF":
            obj.marrow.false_color = "OFF"
        # The update callback hands the modifiers' visibility back.
        if obj.marrow.attach_enabled:
            obj.marrow.attach_enabled = False

        remove_cage(obj)
        restored = restore_rest(obj.data)
        clear_marrow_data(obj.data)
        false_color.clear_attribute(obj.data)

        shape = "shape restored" if restored else "no stored shape to restore"
        self.report({"INFO"}, f"Marrow: cage removed, {shape}")
        return {"FINISHED"}


def collider_objects_of(obj):
    """(object, shape, sticky) for every collider in ``obj``'s collection.

    ``all_objects`` rather than ``objects``, so a collection of collections
    works the way the outliner suggests it should. The body itself is skipped
    rather than treated as an error - a body sitting in its own collider
    collection is just a half-finished edit.
    """
    collection = obj.marrow.collider_collection
    if collection is None:
        return []
    return [
        (ob, ob.marrow_collider.shape, ob.marrow_collider.sticky,
         ob.marrow_collider.friction)
        for ob in collection.all_objects
        if ob is not obj
    ]


def _collection_for(obj):
    """``obj``'s collider collection, made and shown in the outliner if new."""
    settings = obj.marrow
    if settings.collider_collection is None:
        collection = bpy.data.collections.new(f"{obj.name} Colliders")
        scene = getattr(bpy.context, "scene", None)
        if scene is not None:
            # Unlinked, it would be invisible in the outliner and impossible
            # to edit by hand, which is the whole point of a collection.
            scene.collection.children.link(collection)
        settings.collider_collection = collection
    return settings.collider_collection


@bpy.app.handlers.persistent
def migrate_collider_slots(*_args):
    """Carry a pre-collection .blend over to the collider collection.

    Registered as a load_post handler. Idempotent: a body whose slots are
    already drained does nothing, so it is safe on every load. Shape and
    stickiness used to be per slot and are per object now, so a collider
    shared by two bodies that disagreed keeps whichever body migrates last.
    """
    for obj in bpy.data.objects:
        settings = getattr(obj, "marrow", None)
        if settings is None or not settings.colliders:
            continue
        collection = _collection_for(obj)
        for slot in settings.colliders:
            if slot.object is None or slot.object is obj:
                continue
            slot.object.marrow_collider.shape = slot.shape
            slot.object.marrow_collider.sticky = slot.sticky
            if slot.object.name not in collection.objects:
                collection.objects.link(slot.object)
        settings.colliders.clear()


def session_for(obj) -> MarrowSession:
    """A session for ``obj`` built from its panel settings.

    Shared with the bake operator's group path, which needs the same session
    for every body in the group and not only for the active object.
    """
    settings = obj.marrow
    thickness = float(settings.self_thickness) * float(settings.resolution)
    return MarrowSession(
        obj,
        _params_from(settings),
        ground_z=float(settings.ground_z),
        ground_on=bool(settings.ground_enabled),
        collider_objects=collider_objects_of(obj),
        tear_threshold=(
            float(settings.tear_threshold) if settings.tearing_enabled else 0.0
        ),
        stick_break=float(settings.stick_break),
        self_distance=thickness if settings.self_collision else 0.0,
        body_distance=thickness if settings.body_collision else 0.0,
        friction=float(settings.friction),
        attach_enabled=bool(settings.attach_enabled),
        attach_stiffness=float(settings.attach_stiffness),
        pin_group=str(settings.pin_group),
        pin_kinematic=bool(settings.pin_follows),
    )


def _params_from(settings) -> SolverParams:
    """Map the panel sliders onto the solver's parameters.

    Both session paths come through here - Bake builds params up front in
    session_for, Live re-reads them in refresh_from_object - so a setting
    added in one place reaches the other for free.
    """
    return SolverParams(
        substeps=int(settings.substeps),
        mu=float(settings.stiffness),
        lam=float(settings.volume_preservation),
        damping=float(settings.damping),
        # Zero when the toggle is off, so the kernel branch goes dead
        # rather than the pass being conditionally dispatched.
        fiber_k=(
            float(settings.fiber_stiffness) if settings.fiber_enabled else 0.0
        ),
        wave_amp=float(settings.wave_amplitude),
        wave_len=float(settings.wave_length),
        wave_speed=float(settings.wave_speed),
        waveform=0 if settings.waveform == "SMOOTH" else 1,
    )


def _bake_iter(context, obj):
    """Bake the whole group, resumable, one frame per yield.

    Esc keeps the frames already simulated rather than throwing the wait
    away: the cache is keyed by frame, so a partial bake is playable up to
    where it stopped. That is the difference between an interruption and a
    loss, and it is why the frame is the slice.
    """
    scene = context.scene

    if not capability.gpu_available():
        raise _Abort(
            "Marrow needs a working GPU compute context and this build of "
            "Blender could not provide one. Simulation is unavailable."
        )

    # Bodies that collide are simulated together, so baking one has to
    # bake all of them. Membership comes from the scene rather than from
    # the live sessions, which are about to be cleared.
    bodies = [obj] + group.partners_in_scene(obj)

    # Drop the previous sessions and stop the handler before baking:
    # baking steps the scene frame to sample animated colliders, and a
    # live handler would write stale cache frames back into the mesh
    # while we are mid-bake.
    for body in bodies:
        previous = handlers.SESSIONS.pop(body.name, None)
        if previous is not None:
            previous.free()
    handlers.unregister_handler()

    try:
        sessions = [session_for(body) for body in bodies]
    except ValueError as exc:
        handlers.register_handler()
        raise _Abort(str(exc)) from exc

    try:
        frames = yield from _stage(
            "baking", group.bake_iter(
                sessions, scene.frame_start, scene.frame_end, scene=scene
            )
        )
    except GeneratorExit:
        # Cancelled. Keep the sessions so what was simulated stays playable,
        # and re-arm the handler, then let the exit continue.
        for session in sessions:
            handlers.SESSIONS[session.object_name] = session
        handlers.register_handler()
        raise
    except MarrowNaNError as exc:
        for session in sessions:
            session.free()
        # Re-arm the handler so live simulation, which was torn down for
        # the bake, rebuilds itself on the next frame change.
        handlers.register_handler()
        raise _Abort(str(exc)) from exc
    except Exception as exc:
        # A wedged GPU queue raises StaleReadError, not MarrowNaNError.
        # Report it the same way rather than leaking the sessions and
        # leaving the frame handler unregistered behind a traceback.
        for session in sessions:
            session.free()
        handlers.register_handler()
        raise _Abort(f"Marrow bake failed: {exc}") from exc

    for session in sessions:
        handlers.SESSIONS[session.object_name] = session
    handlers.register_handler()

    extra = f" for {len(sessions)} bodies" if len(sessions) > 1 else ""
    return "INFO", f"Marrow: baked {frames} frames{extra}"


class MARROW_OT_bake(_ModalPipeline, bpy.types.Operator):
    bl_idname = "marrow.bake"
    bl_label = "Bake"
    bl_description = "Simulate the scene frame range and cache the result"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def _make_work(self, context):
        return _bake_iter(context, context.active_object)

    def _cancel_message(self):
        return "Marrow: bake stopped - frames simulated so far are cached"


class MARROW_OT_free(bpy.types.Operator):
    bl_idname = "marrow.free"
    bl_label = "Free Bake"
    bl_description = "Discard the cached simulation and release GPU memory"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = context.active_object
        name = obj.name if obj is not None else None
        session = handlers.SESSIONS.pop(name, None)
        if session is not None:
            session.free()
            self.report({"INFO"}, "Marrow: bake freed")
        else:
            self.report({"INFO"}, "Marrow: nothing baked")
        return {"FINISHED"}


class MARROW_OT_live(bpy.types.Operator):
    bl_idname = "marrow.live_toggle"
    bl_label = "Live"
    bl_description = (
        "Simulate as the timeline plays. Returning to the start frame restarts "
        "and picks up changed settings"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        settings = obj.marrow

        if settings.live_enabled:
            settings.live_enabled = False
            session = handlers.SESSIONS.pop(obj.name, None)
            if session is not None:
                session.free()
            if not handlers.SESSIONS:
                handlers.unregister_handler()
            self.report({"INFO"}, "Marrow: live simulation off")
            return {"FINISHED"}

        if find_cage(obj) is None:
            self.report(
                {"ERROR"},
                "Marrow: run Tetrahedralize in the Marrow panel first; there "
                "is no cage to simulate.",
            )
            return {"CANCELLED"}

        if not capability.gpu_available():
            self.report(
                {"ERROR"},
                "Marrow needs a working GPU compute context and this build of "
                "Blender could not provide one. Simulation is unavailable.",
            )
            return {"CANCELLED"}

        settings.live_enabled = True
        previous = handlers.SESSIONS.pop(obj.name, None)
        if previous is not None:
            previous.free()

        handlers.register_handler()
        handlers.ensure_sessions(context.scene)
        context.scene.frame_set(context.scene.frame_start)
        self.report({"INFO"}, "Marrow: live. Play the timeline")
        return {"FINISHED"}


class MARROW_OT_collider_add(bpy.types.Operator):
    bl_idname = "marrow.collider_add"
    bl_label = "Add Collider"
    bl_description = (
        "Link the other selected objects into this soft body's collider "
        "collection, creating one if it has none yet"
    )
    bl_options = {"REGISTER", "UNDO"}

    # Modal state. Class-level defaults so _release is safe on any path.
    _timer = None
    _work = None
    _label = ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        picked = [ob for ob in context.selected_objects if ob is not obj]
        if not picked:
            self.report(
                {"ERROR"},
                "Marrow: select the objects to collide against as well as the body",
            )
            return {"CANCELLED"}

        collection = _collection_for(obj)
        for ob in picked:
            if ob.name not in collection.objects:
                collection.objects.link(ob)
        obj.marrow.active_collider = max(0, len(collection.all_objects) - 1)
        return {"FINISHED"}


class MARROW_OT_collider_remove(bpy.types.Operator):
    bl_idname = "marrow.collider_remove"
    bl_label = "Remove Collider"
    bl_description = "Unlink the selected collider from the collection"
    bl_options = {"REGISTER", "UNDO"}

    # Modal state. Class-level defaults so _release is safe on any path.
    _timer = None
    _work = None
    _label = ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and obj.marrow.collider_collection is not None
            and len(obj.marrow.collider_collection.all_objects) > 0
        )

    def execute(self, context):
        settings = context.active_object.marrow
        collection = settings.collider_collection
        members = list(collection.all_objects)
        index = min(settings.active_collider, len(members) - 1)
        target = members[index]
        # all_objects reaches into nested collections, so unlink the object
        # from whichever one actually holds it, not just the top level.
        for holder in (collection, *collection.children_recursive):
            if target.name in holder.objects:
                holder.objects.unlink(target)
        settings.active_collider = max(0, index - 1)
        return {"FINISHED"}
