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

from .storage import BIND_IDX

SESSIONS: dict = {}

_atexit_installed = False


def _install_atexit_guard() -> None:
    """Free GPU state at interpreter exit, while the context is still alive.

    Blender does not call unregister() on quit, and module globals are
    collected after the GPU context is torn down - a GPUTexture freed that
    late crashes the whole process at shutdown (EXCEPTION_ACCESS_VIOLATION in
    MSVCP140.dll, measured). atexit callbacks run early enough in
    finalisation that freeing still works.
    """
    global _atexit_installed
    if _atexit_installed:
        return
    import atexit

    atexit.register(free_all)
    _atexit_installed = True


def ensure_sessions(scene) -> None:
    """Give every live-enabled tetrahedralised object a session.

    Live is the default, so a file can be reopened and played without anyone
    pressing a button. Building a session is not cheap, hence once per object.
    """
    from .session import MarrowSession, find_cage

    for obj in scene.objects:
        settings = getattr(obj, "marrow", None)
        if settings is None or not settings.live_enabled:
            continue
        if obj.name in SESSIONS:
            continue
        if find_cage(obj) is None:
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
        mesh = getattr(obj, "data", None) if obj is not None else None
        if mesh is None or mesh.attributes.get(BIND_IDX) is None:
            # The object was deleted or renamed, or an unrelated mesh now owns
            # the name. Kept around, the session only leaks its GPU memory;
            # worse, if a different object reuses the name it would have
            # another body's cached frames written into the wrong mesh. A
            # renamed object has already been given a fresh session under its
            # new name by ensure_sessions above, so dropping this one loses
            # nothing.
            session.free()
            del SESSIONS[name]
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
    _install_atexit_guard()
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


@bpy.app.handlers.persistent
def free_on_load(*_args):
    """Drop every session when a file is loaded. Registered on load_pre.

    SESSIONS is keyed by object NAME and lives in module scope, so it
    outlives the file it was built for. Open a second file holding an object
    of the same name that already has a cage, and on_frame_change finds that
    entry and simulates the new mesh with the old file's GPU state -
    measured, a 125-node session answering for an 8-node body. Tetrahedralize
    pops the stale entry, so the path only bites when a body is NOT rebuilt,
    which is exactly the case of opening a finished file and pressing play.

    It leaks as well: about 25 GPU textures a session, held for the life of
    the Blender process however many files are opened.

    load_pre rather than load_post, because these sessions belong to the file
    being closed - freeing them before the next one arrives means nothing can
    look one up in between. Persistent, or Blender would drop this handler on
    the first load and it would protect exactly one file.
    """
    free_all()
