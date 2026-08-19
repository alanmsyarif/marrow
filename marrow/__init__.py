"""Marrow: GPU tetrahedral soft body for Blender."""

import tomllib
from pathlib import Path

# blender_manifest.toml is the single source of truth for the version. Blender
# reads that file without running any Python, so it cannot be generated from
# here - the only direction that leaves one copy is to read it back. A second
# literal in this file is exactly what drifted before: manifest said 1.0.0
# while __version__ still said 0.9.2.
__version__ = tomllib.loads(
    (Path(__file__).parent / "blender_manifest.toml").read_text(encoding="utf-8")
)["version"]

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
        migrate_collider_slots,
    )
    from .blender.ui import (
        MARROW_PT_panel,
        MARROW_UL_colliders,
        MarrowColliderSettings,
        MarrowColliderSlot,
        MarrowSettings,
    )

    # MarrowColliderSlot must land before MarrowSettings, which points at it.
    classes = (
        MarrowColliderSlot,
        MarrowColliderSettings,
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
    bpy.types.Object.marrow_collider = bpy.props.PointerProperty(
        type=MarrowColliderSettings
    )
    # Pre-collection .blend files still carry collider slots. Drain them the
    # moment one is opened, so a body that plainly had colliders before does
    # not come back with an empty collider list.
    if migrate_collider_slots not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(migrate_collider_slots)
    _registered[:] = classes


def unregister():
    import bpy

    from .blender import handlers

    # Releases every session's GPU objects. Module globals are collected after
    # Blender tears the GPU context down, and freeing a shader against a dead
    # context crashes at shutdown.
    handlers.unregister_handler()

    from .blender.ops import migrate_collider_slots

    while migrate_collider_slots in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(migrate_collider_slots)

    del bpy.types.Object.marrow_collider
    del bpy.types.Object.marrow
    for cls in reversed(_registered):
        bpy.utils.unregister_class(cls)
    _registered.clear()
