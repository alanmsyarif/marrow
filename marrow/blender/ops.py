"""Marrow operators."""

import bpy
import numpy as np

from ..blender import group, handlers
from ..blender.inside_bvh import cell_mask_from_object
from ..blender.session import CAGE_SUFFIX, MarrowSession, find_cage
from ..blender.storage import (
    BIND_IDX,
    REST_KEY,
    clear_marrow_data,
    restore_rest,
    write_bind,
    write_rest,
    write_tetmesh,
)
from ..core.bind import bind_points
from ..core.coloring import color_tets
from ..core.lattice import build_lattice
from ..core.solver_ref import SolverParams
from ..gpu import capability
from ..gpu.solver import MarrowNaNError


class MARROW_OT_tetrahedralize(bpy.types.Operator):
    bl_idname = "marrow.tetrahedralize"
    bl_label = "Tetrahedralize"
    bl_description = "Fill the selected mesh with a tetrahedral cage"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        spacing = float(obj.marrow.resolution)

        # Build from the shape the user modelled, never from a simulated
        # pose. Without this, tetrahedralizing again after playing the
        # timeline would make the deformed mesh the new rest shape, and the
        # drift would compound every time Resolution was changed.
        restore_rest(obj.data)

        mask, bounds_min = cell_mask_from_object(obj, spacing)
        if not mask.any():
            self.report(
                {"ERROR"},
                "No cells inside the mesh. Lower Resolution in the Marrow panel "
                "until the cage fills the object.",
            )
            return {"CANCELLED"}

        tetmesh = build_lattice(bounds_min, spacing, mask)
        try:
            tetmesh.validate()
        except ValueError as exc:
            self.report({"ERROR"}, f"Invalid cage: {exc}")
            return {"CANCELLED"}

        colors = color_tets(tetmesh.tets, tetmesh.n_nodes)

        render_verts = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", render_verts)
        render_verts = render_verts.reshape(-1, 3)
        world = np.array(obj.matrix_world.to_4x4())
        world_verts = render_verts @ world[:3, :3].T + world[:3, 3]

        bind_idx, bind_w = bind_points(tetmesh.nodes, tetmesh.tets, world_verts)
        write_bind(obj.data, bind_idx, bind_w)
        # Captured after the restore above, so it is the modelled shape.
        write_rest(obj.data)

        cage_name = f"{obj.name}{CAGE_SUFFIX}"
        remove_cage(obj)

        cage_mesh = bpy.data.meshes.new(cage_name)
        write_tetmesh(cage_mesh, tetmesh, colors)
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

        self.report(
            {"INFO"},
            f"Marrow: {tetmesh.n_tets} tets, {tetmesh.n_nodes} nodes, "
            f"{int(colors.max()) + 1 if colors.size else 0} colours",
        )
        return {"FINISHED"}


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

        remove_cage(obj)
        restored = restore_rest(obj.data)
        clear_marrow_data(obj.data)

        shape = "shape restored" if restored else "no stored shape to restore"
        self.report({"INFO"}, f"Marrow: cage removed, {shape}")
        return {"FINISHED"}


def collider_objects_of(obj):
    """(object, shape, sticky) for each usable collider slot on ``obj``.

    Empty slots and a body pointed at itself are skipped rather than treated
    as errors - both are just a half-finished edit in the list.
    """
    pairs = []
    for slot in obj.marrow.colliders:
        if slot.object is None or slot.object is obj:
            continue
        pairs.append((slot.object, slot.shape, slot.sticky))
    return pairs


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
    )


def _params_from(settings) -> SolverParams:
    """Map the panel sliders onto the solver's parameters."""
    return SolverParams(
        substeps=int(settings.substeps),
        mu=float(settings.stiffness),
        lam=float(settings.volume_preservation),
        damping=float(settings.damping),
    )


class MARROW_OT_bake(bpy.types.Operator):
    bl_idname = "marrow.bake"
    bl_label = "Bake"
    bl_description = "Simulate the scene frame range and cache the result"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        settings = obj.marrow

        if not capability.gpu_available():
            self.report(
                {"ERROR"},
                "Marrow needs a working GPU compute context and this build of "
                "Blender could not provide one. Simulation is unavailable.",
            )
            return {"CANCELLED"}

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
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        try:
            frames = group.bake(
                sessions, scene.frame_start, scene.frame_end, scene=scene
            )
        except MarrowNaNError as exc:
            for session in sessions:
                session.free()
            # Re-arm the handler so live simulation, which was torn down for
            # the bake, rebuilds itself on the next frame change.
            handlers.register_handler()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            # A wedged GPU queue raises StaleReadError, not MarrowNaNError.
            # Report it the same way rather than leaking the sessions and
            # leaving the frame handler unregistered behind a traceback.
            for session in sessions:
                session.free()
            handlers.register_handler()
            self.report({"ERROR"}, f"Marrow bake failed: {exc}")
            return {"CANCELLED"}

        for session in sessions:
            handlers.SESSIONS[session.object_name] = session
        handlers.register_handler()

        extra = f" for {len(sessions)} bodies" if len(sessions) > 1 else ""
        self.report({"INFO"}, f"Marrow: baked {frames} frames{extra}")
        return {"FINISHED"}


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
    bl_description = "Add a collider slot to this soft body"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        settings = context.active_object.marrow
        settings.colliders.add()
        settings.active_collider = len(settings.colliders) - 1
        return {"FINISHED"}


class MARROW_OT_collider_remove(bpy.types.Operator):
    bl_idname = "marrow.collider_remove"
    bl_label = "Remove Collider"
    bl_description = "Remove the selected collider slot"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and len(obj.marrow.colliders) > 0
        )

    def execute(self, context):
        settings = context.active_object.marrow
        index = min(settings.active_collider, len(settings.colliders) - 1)
        settings.colliders.remove(index)
        settings.active_collider = max(0, index - 1)
        return {"FINISHED"}
