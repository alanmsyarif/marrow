"""Marrow properties and sidebar panel."""

import bpy


class MarrowColliderSlot(bpy.types.PropertyGroup):
    """Legacy: one collider slot, from before colliders became a collection.

    Kept registered only so a .blend saved by an older Marrow still has its
    slots to read on load - see ``ops.migrate_collider_slots``, which drains
    them into a collection and clears them. Nothing writes slots any more.
    """

    object: bpy.props.PointerProperty(
        name="Object",
        description="Object whose transform drives this collider",
        type=bpy.types.Object,
    )
    shape: bpy.props.EnumProperty(
        name="Shape",
        description="Collision shape, sized and oriented by the object's transform",
        items=[
            ("MESH", "Mesh", "The object's own shape, as a signed distance field"),
            ("SPHERE", "Sphere", "Unit sphere shaped by the object transform"),
            ("BOX", "Box", "Unit box shaped by the object transform"),
        ],
        default="MESH",
    )
    sticky: bpy.props.BoolProperty(
        name="Sticky",
        description=(
            "Material that touches this collider is held to the surface and "
            "dragged along as the collider moves, instead of only being "
            "pushed out of it"
        ),
        default=False,
    )


class MarrowColliderSettings(bpy.types.PropertyGroup):
    """How one object behaves when a body's collider collection holds it.

    On the collider rather than on the body, so an object dropped into two
    bodies' collections is described once instead of twice.
    """

    shape: bpy.props.EnumProperty(
        name="Shape",
        description="Collision shape, sized and oriented by the object's transform",
        items=[
            ("MESH", "Mesh", "The object's own shape, as a signed distance field"),
            ("SPHERE", "Sphere", "Unit sphere shaped by the object transform"),
            ("BOX", "Box", "Unit box shaped by the object transform"),
        ],
        default="MESH",
    )
    sticky: bpy.props.BoolProperty(
        name="Sticky",
        description=(
            "Material that touches this collider is held to the surface and "
            "dragged along as the collider moves, instead of only being "
            "pushed out of it"
        ),
        default=False,
    )


class MARROW_UL_colliders(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        row.label(text=item.name, icon="OBJECT_DATA")
        sub = row.row(align=True)
        sub.prop(item.marrow_collider, "shape", text="")
        sub.prop(item.marrow_collider, "sticky", text="", icon="SNAP_ON", toggle=True)


def _update_false_color(self, context):
    """Swap the false-color material in on mode change, the original back out."""
    from . import false_color

    obj = self.id_data
    if self.false_color == "OFF":
        false_color.restore(obj)
    else:
        false_color.apply(obj, self.false_color)
        false_color.prime(obj, self.false_color)


def _update_attach(self, context):
    """Attachment owns the display: mute the object's modifiers, or hand back."""
    from . import attach

    attach.mute_modifiers(self.id_data, self.attach_enabled)


class MarrowSettings(bpy.types.PropertyGroup):
    resolution: bpy.props.FloatProperty(
        name="Resolution",
        description="Cage cell size in world units. Smaller fills finer detail",
        default=0.25,
        min=0.001,
        soft_max=1.0,
        unit="LENGTH",
    )
    live_enabled: bpy.props.BoolProperty(
        name="Live",
        description=(
            "Simulate as the timeline plays. Returning to the start frame "
            "restarts the simulation and picks up any changed settings"
        ),
        default=True,
    )
    substeps: bpy.props.IntProperty(
        name="Substeps",
        description=(
            "XPBD substeps per frame. More is stabler and slower - this is the "
            "most expensive knob here"
        ),
        default=10,
        min=1,
        max=100,
    )
    stiffness: bpy.props.FloatProperty(
        name="Stiffness",
        description="Resistance to distortion (deviatoric compliance)",
        default=1.0e4,
        min=0.0,
        soft_max=1.0e6,
    )
    volume_preservation: bpy.props.FloatProperty(
        name="Volume Preservation",
        description="Resistance to volume change (hydrostatic compliance)",
        default=1.0e5,
        min=0.0,
        soft_max=1.0e7,
    )
    damping: bpy.props.FloatProperty(
        name="Damping",
        description="Velocity retained each substep. 1.0 is undamped",
        default=0.999,
        min=0.0,
        max=1.0,
    )
    ground_enabled: bpy.props.BoolProperty(
        name="Ground Plane",
        description="Stop the body falling through a horizontal plane",
        default=False,
    )
    tearing_enabled: bpy.props.BoolProperty(
        name="Tearing",
        description=(
            "Let over-stretched material fail permanently. Torn tets stop "
            "resisting, so the body goes slack and pulls apart"
        ),
        default=False,
    )
    tear_threshold: bpy.props.FloatProperty(
        name="Tear Strain",
        description=(
            "Largest stretch ratio material survives. 1.5 means a tet fails "
            "once anything in it is pulled to 1.5x its rest length, in any "
            "direction. Lower is more brittle. Volume-preserving squashing "
            "stretches sideways and counts, so a heavy press can tear too"
        ),
        default=1.5,
        min=1.01,
        soft_max=5.0,
    )
    self_collision: bpy.props.BoolProperty(
        name="Self Collision",
        description=(
            "Stop the body passing through itself where it folds. Costs "
            "roughly 5ms a frame at Resolution 0.1 and grows with the square "
            "of the cage's surface"
        ),
        default=False,
    )
    attach_enabled: bpy.props.BoolProperty(
        name="Attachment",
        description=(
            "Pull the cage towards the object's animated shape each frame, "
            "so an armature or other deforming modifiers drive the "
            "simulation from the inside instead of bending the result "
            "afterwards. The object's own modifiers are muted in the "
            "display while this is on. Requires the object to keep its "
            "vertex count"
        ),
        default=False,
        update=_update_attach,
    )
    attach_stiffness: bpy.props.FloatProperty(
        name="Attach Stiffness",
        description=(
            "How hard the flesh follows the animation. 1.0 rides the bones "
            "exactly; lower values let the bones lead and the flesh lag, "
            "jiggle and overshoot"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
    )
    body_collision: bpy.props.BoolProperty(
        name="Collide With Bodies",
        description=(
            "Collide with every other Marrow object that also has this on, "
            "and deform both. They are simulated together, so baking one "
            "bakes all of them and the group runs at its highest Substeps"
        ),
        default=False,
    )
    self_thickness: bpy.props.FloatProperty(
        name="Thickness",
        description=(
            "Contact gap, as a multiple of Resolution, for both Self "
            "Collision and Collide With Bodies. 1.0 keeps the render surface "
            "from visibly interpenetrating. Below 1.0 a fold can slip through "
            "between cage nodes"
        ),
        default=1.0,
        min=0.1,
        soft_max=3.0,
    )
    collider_collection: bpy.props.PointerProperty(
        name="Colliders",
        description=(
            "Every object in this collection is collided against, nested "
            "collections included. Shape and Sticky are set on each object"
        ),
        type=bpy.types.Collection,
    )
    # Drained into collider_collection on load - see MarrowColliderSlot.
    colliders: bpy.props.CollectionProperty(type=MarrowColliderSlot)
    active_collider: bpy.props.IntProperty(default=0)
    stick_break: bpy.props.FloatProperty(
        name="Stick Break",
        description=(
            "How far the material may drag a sticky contact point before it "
            "lets go. Zero never lets go. Tune it against the shot - the "
            "distance a contact settles at depends on Stiffness and Substeps"
        ),
        default=0.0,
        min=0.0,
        soft_max=1.0,
        unit="LENGTH",
    )
    ground_z: bpy.props.FloatProperty(
        name="Ground Height",
        description="Height of the ground plane in world units",
        default=0.0,
        unit="LENGTH",
    )
    false_color: bpy.props.EnumProperty(
        name="False Color",
        description=(
            "Rainbow-shade the surface by how far the material is stretched, "
            "like Vellum's false color mode. Swaps a generated material into "
            "slot 0 while active; Off restores the object's own material"
        ),
        items=[
            ("OFF", "Off", "Show the object's own material"),
            (
                "STRETCH",
                "Stretch",
                "Edge stretch ratio: 1 at rest, hot where pulled past rest "
                "length, cold where compressed",
            ),
        ],
        default="OFF",
        update=_update_false_color,
    )


class MARROW_PT_panel(bpy.types.Panel):
    bl_label = "Marrow"
    bl_idname = "MARROW_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Marrow"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.marrow

        cage = layout.box()
        cage.label(text="Cage")
        cage.prop(settings, "resolution")
        cage.operator("marrow.tetrahedralize", icon="MESH_ICOSPHERE")

        from .session import find_cage

        if find_cage(obj) is None:
            # Everything below acts on a cage. The sliders would feed a solver
            # that does not exist and Live, Bake and Free can only report the
            # same "run Tetrahedralize first" back, so offer none of it.
            layout.label(text="Tetrahedralize to simulate", icon="INFO")
            return

        cage.operator("marrow.detetrahedralize", icon="X")

        sim = layout.box()
        sim.label(text="Simulation")
        sim.prop(settings, "substeps")
        sim.prop(settings, "stiffness")
        sim.prop(settings, "volume_preservation")
        sim.prop(settings, "damping")

        ground = sim.box()
        ground.prop(settings, "ground_enabled")
        row = ground.row()
        row.enabled = settings.ground_enabled
        row.prop(settings, "ground_z")

        tearing = sim.box()
        tearing.prop(settings, "tearing_enabled")
        row = tearing.row()
        row.enabled = settings.tearing_enabled
        row.prop(settings, "tear_threshold")

        attach = sim.box()
        attach.prop(settings, "attach_enabled")
        row = attach.row()
        row.enabled = settings.attach_enabled
        row.prop(settings, "attach_stiffness")

        contact = sim.box()
        contact.prop(settings, "self_collision")
        contact.prop(settings, "body_collision")
        row = contact.row()
        row.enabled = settings.self_collision or settings.body_collision
        row.prop(settings, "self_thickness")

        # Colliders belong to the body being simulated: point it at a
        # collection here rather than walking to each object and tagging it.
        box = sim.box()
        box.label(text="Colliders", icon="PHYSICS")
        box.prop(settings, "collider_collection", text="")
        collection = settings.collider_collection
        row = box.row()
        if collection is None:
            row.label(
                text="Select objects and press + to collide against them",
                icon="INFO",
            )
        else:
            row.template_list(
                "MARROW_UL_colliders", "",
                collection, "all_objects",
                settings, "active_collider",
                rows=2,
            )
        col = row.column(align=True)
        col.operator("marrow.collider_add", icon="ADD", text="")
        col.operator("marrow.collider_remove", icon="REMOVE", text="")
        row = box.row()
        row.enabled = collection is not None and any(
            ob.marrow_collider.sticky for ob in collection.all_objects
        )
        row.prop(settings, "stick_break")

        display = layout.box()
        display.label(text="Display", icon="COLOR")
        display.prop(settings, "false_color")

        from . import handlers

        session = handlers.SESSIONS.get(obj.name)

        row = layout.row(align=True)
        row.operator(
            "marrow.live_toggle",
            text="Live",
            icon="PLAY",
            depress=bool(settings.live_enabled),
        )
        row.operator("marrow.bake", icon="PHYSICS")
        row.operator("marrow.free", icon="TRASH")

        cached = session.baked_range if session is not None else None
        if session is not None and getattr(session, "baked", False):
            layout.label(
                text=f"Baked {cached[0]}-{cached[1]}" if cached else "Baked",
                icon="FILE_TICK",
            )
        elif settings.live_enabled:
            layout.label(
                text=f"Live, cached {cached[0]}-{cached[1]}" if cached else
                     "Live: play the timeline",
                icon="REC",
            )
