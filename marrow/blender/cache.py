"""Bake caches that survive closing the file.

The cache used to live only in memory, so reopening a .blend meant playing
the whole shot again. It is written beside the file instead, in a
``blendcache_<name>`` folder - the same convention Blender's own point caches
use, and for the same reason: a bake of a real body is hundreds of megabytes
of float, and putting that inside the .blend would make every save carry it.

Only cage node positions are stored. The render positions are barycentric
combinations of them, so they rebuild exactly - measured at 1.4e-7 against
what the skin kernel produced - and a cage is smaller than the mesh wrapped
around it. Storing both would be storing the same information twice.

Only BAKED caches are written. A live cache is disposable by design and is
rebuilt as the timeline plays; a bake is something the user waited for.
"""

import os

import numpy as np

# Bumped when the stored layout changes, so an old sidecar is ignored rather
# than unpacked into the wrong shapes.
FORMAT = 1


def path_for(obj) -> str:
    """Where ``obj``'s bake lives, or "" when the .blend is unsaved.

    Keyed by object name, so renaming an object orphans its cache. That is
    the same rule the rest of Marrow follows for sessions, and the cost of
    getting it wrong is a rebake rather than a wrong simulation - the
    validation in ``load`` refuses anything whose shape does not match.
    """
    import bpy

    blend = bpy.data.filepath
    if not blend:
        return ""
    folder = os.path.join(
        os.path.dirname(blend),
        "blendcache_" + os.path.splitext(os.path.basename(blend))[0],
    )
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name)
    return os.path.join(folder, f"marrow_{safe}.npz")


def save(session, obj) -> str:
    """Write a baked session beside the .blend. Returns the path, or "".

    Silent when there is nothing to write: an unsaved file has nowhere to put
    it, and a live cache is not worth keeping.
    """
    if not getattr(session, "baked", False) or not session._cache_nodes:
        return ""
    path = path_for(obj)
    if not path:
        return ""
    frames = np.array(sorted(session._cache_nodes), dtype=np.int32)
    nodes = np.stack([session._cache_nodes[int(f)] for f in frames])
    torn = session._torn_frame
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        format=np.array([FORMAT], dtype=np.int32),
        frames=frames,
        nodes=nodes.astype(np.float32),
        torn_frame=(np.zeros(0) if torn is None else np.asarray(torn)),
        n_render=np.array([len(obj.data.vertices)], dtype=np.int32),
        n_tets=np.array([session.tetmesh.tets.shape[0]], dtype=np.int32),
    )
    return path


def load(session, obj) -> bool:
    """Restore a baked session from disk. False if there was nothing usable.

    Every stored shape is checked against the cage that exists now. Editing
    the mesh or re-tetrahedralizing leaves a cache that would unpack into the
    wrong number of nodes, and playing that back would look like a solver
    fault rather than a stale file, so it is refused instead.
    """
    path = path_for(obj)
    if not path or not os.path.exists(path):
        return False
    try:
        with np.load(path) as data:
            if int(data["format"][0]) != FORMAT:
                return False
            if int(data["n_render"][0]) != len(obj.data.vertices):
                return False
            if int(data["n_tets"][0]) != session.tetmesh.tets.shape[0]:
                return False
            frames = data["frames"]
            nodes = data["nodes"]
            torn = data["torn_frame"]
            if nodes.shape[1] != session.tetmesh.nodes.shape[0]:
                return False
    except Exception:
        # A truncated or foreign .npz is a stale cache, not a crash.
        return False

    from ..core.bind import deform

    for i, frame in enumerate(frames):
        node_positions = nodes[i].astype(np.float64)
        session._cache_nodes[int(frame)] = nodes[i]
        session._cache[int(frame)] = deform(
            node_positions, session.tetmesh.tets, session.bind_idx, session.bind_w
        ).astype(np.float32)
    session._torn_frame = None if torn.size == 0 else torn
    session.baked = True
    session._last_simulated = int(frames.max()) if frames.size else None
    return True


def remove(obj) -> bool:
    """Delete ``obj``'s sidecar. True if one was there.

    Free and De-tetrahedralize both mean it: leaving the file behind would
    have the next session load the bake the user just discarded.
    """
    path = path_for(obj)
    if not path or not os.path.exists(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    folder = os.path.dirname(path)
    try:
        if not os.listdir(folder):
            os.rmdir(folder)
    except OSError:
        pass
    return True
