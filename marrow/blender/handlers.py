"""Frame handler that plays baked Marrow caches back into render meshes.

Nothing simulates here. A frame change is a dictionary lookup and a
foreach_set, which is what keeps scrubbing interactive.

SESSIONS is module-level and holds objects that own GPU resources. Module
globals are collected *after* Blender tears the GPU context down, and a
GPUShader freed against a dead context crashes at shutdown - measured,
EXCEPTION_ACCESS_VIOLATION in MSVCP140.dll. So unregister_handler() releases
every session, and marrow.unregister() must call it.
"""

import bpy

SESSIONS: dict = {}


def ensure_sessions(scene) -> None:
    """Give every live-enabled tetrahedralised object a session.

    Live is the default, so a file can be reopened and played without anyone
    pressing a button. Building a session is not cheap, hence once per object.
    """
    from .session import CAGE_SUFFIX, MarrowSession

    for obj in scene.objects:
        settings = getattr(obj, "marrow", None)
        if settings is None or not settings.live_enabled:
            continue
        if obj.name in SESSIONS:
            continue
        if bpy.data.objects.get(f"{obj.name}{CAGE_SUFFIX}") is None:
            continue
        try:
            session = MarrowSession(obj)
            session.refresh_from_object()
            session.live = True
            SESSIONS[obj.name] = session
        except Exception as exc:
            settings.live_enabled = False
            print(f"marrow: could not start live simulation for {obj.name!r}: {exc}")


def on_frame_change(scene, depsgraph=None) -> None:
    """Write each session's cached frame into its object."""
    ensure_sessions(scene)
    frame = scene.frame_current
    for name, session in list(SESSIONS.items()):
        obj = bpy.data.objects.get(name)
        if obj is None:
            # The object was renamed or deleted; the session is stale but
            # harmless. Skipping beats raising inside a frame handler, which
            # Blender would surface on every single frame change.
            continue
        try:
            session.write_to_mesh(obj, frame, frame_start=scene.frame_start)
        except Exception as exc:
            # A raise here fires on every frame change and makes Blender
            # unusable. Stop this session's live simulation and say why once.
            session.live = False
            print(f"marrow: live simulation stopped for {name!r}: {exc}")


def register_handler() -> None:
    """Idempotent: Blender will happily hold the same handler twice."""
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)


def unregister_handler() -> None:
    """Remove only our handler, then release all GPU state."""
    while on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
    free_all()


def free_all() -> None:
    for session in SESSIONS.values():
        session.free()
    SESSIONS.clear()
