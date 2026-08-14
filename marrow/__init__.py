"""Marrow: GPU tetrahedral soft body for Blender."""

__version__ = "0.6.2"

# bpy is imported inside register()/unregister(), not at module scope. Any
# `from .core.x import ...` executes this file first, so a top-level
# `import bpy` takes the entire pytest core suite down with
# ModuleNotFoundError outside Blender - measured, all six core modules failed
# at collection. Deferring is the ordinary addon idiom and costs nothing:
# Blender only ever calls these two functions, and by then bpy is present.

_registered = []


def register():
    import bpy

    from .blender.ops import (
        MARROW_OT_bake,
        MARROW_OT_collider_add,
        MARROW_OT_collider_remove,
        MARROW_OT_detetrahedralize,
        MARROW_OT_free,
        MARROW_OT_live,
        MARROW_OT_tetrahedralize,
    )
    from .blender.ui import (
        MARROW_PT_panel,
        MARROW_UL_colliders,
        MarrowColliderSlot,
        MarrowSettings,
    )

    # MarrowColliderSlot must land before MarrowSettings, which points at it.
    classes = (
        MarrowColliderSlot,
        MarrowSettings,
        MARROW_UL_colliders,
        MARROW_OT_collider_add,
        MARROW_OT_collider_remove,
        MARROW_OT_tetrahedralize,
        MARROW_OT_detetrahedralize,
        MARROW_OT_bake,
        MARROW_OT_live,
        MARROW_OT_free,
        MARROW_PT_panel,
    )
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.marrow = bpy.props.PointerProperty(type=MarrowSettings)
    _registered[:] = classes


def unregister():
    import bpy

    from .blender import handlers

    # Releases every session's GPU objects. Module globals are collected after
    # Blender tears the GPU context down, and freeing a shader against a dead
    # context crashes at shutdown.
    handlers.unregister_handler()

    del bpy.types.Object.marrow
    for cls in reversed(_registered):
        bpy.utils.unregister_class(cls)
    _registered.clear()
