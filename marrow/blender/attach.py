"""Attachment weights and per-frame target sampling.

The trick that lets an armature drive the simulation from the inside:
cage nodes get skinning weights synthesized against the rest shape, and
every frame those weights are blended over the EVALUATED mesh - the one
the armature modifier has already deformed - to produce per-node targets.
Targets come from Blender's own evaluated mesh, so linear and dual
quaternion skinning both work without reimplementing either.
"""

import numpy as np

from ..blender.storage import read_attach, read_rest, write_attach
from ..core.attach import synth_weights, targets_from

# Where the modifiers' pre-mute visibility is stored on the object, so
# toggling Attachment off or De-tetrahedralize can hand the display back.
# Rows are name -> [show_viewport, show_render, preserves_count].
DISPLAY_KEY = "marrow_attach_mod_display"


def _preserves_count(obj, modifier) -> bool:
    """Whether one modifier keeps the vertex count, probed by evaluating it
    alone against the mesh as it is. Count behaviour depends on the modifier
    and the topology, not on the positions, so the current mesh is fine.
    """
    import bpy

    mesh = obj.data
    states = [(m, m.show_viewport) for m in obj.modifiers]
    for m, _ in states:
        # Name match, not `is`: collection iteration hands out fresh RNA
        # wrappers, so an identity check never matches and everything
        # would end up muted.
        m.show_viewport = m.name == modifier.name
    try:
        # view_layer.update(), not mesh.update(): a flag flip alone is not
        # always enough for the depsgraph to re-evaluate the stack.
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        me = evaluated.to_mesh()
        count = 0 if me is None else len(me.vertices)
        evaluated.to_mesh_clear()
        return count == len(mesh.vertices)
    finally:
        for m, state in states:
            m.show_viewport = state
        bpy.context.view_layer.update()


def mute_modifiers(obj, muted: bool) -> None:
    """While attachment is on, count-preserving modifiers feed the
    simulation instead of bending its display.

    The targets sample the evaluated mesh they produce, and the written
    simulation IS the display - leaving them shown would deform the result
    a second time, measured at exactly twice the bone travel. Count-changing
    modifiers (Subdivision, Decimate) are left alone instead: their vertices
    have no per-base-vertex meaning, so they cannot feed the targets, and on
    the display they smooth the simulated shape rather than re-bending it.

    The original visibility is stored on the object so it can be handed
    back; modifiers added while muted are adopted at sample time.
    """
    if muted:
        if DISPLAY_KEY not in obj:
            # Two-element rows are unprobed: probing needs a depsgraph that
            # actually re-evaluates, and the property update callback this
            # runs from does not have one. Everything is muted for now; the
            # first sample_targets probes and wakes the count-changing
            # modifiers back up.
            obj[DISPLAY_KEY] = {
                m.name: [bool(m.show_viewport), bool(m.show_render)]
                for m in obj.modifiers
            }
        for m in obj.modifiers:
            m.show_viewport = False
            m.show_render = False
        return
    stored = obj.get(DISPLAY_KEY)
    if stored is None:
        return
    for m in obj.modifiers:
        row = stored.get(m.name)
        if row is not None:
            m.show_viewport = bool(row[0])
            m.show_render = bool(row[1])
    del obj[DISPLAY_KEY]


def ensure_weights(obj, tetmesh):
    """``(idx, w)`` for ``obj``'s cage, computing and caching them if new.

    Weights are against the REST shape, synthesized in OBJECT space: the
    cage nodes arrive in bind-time world space, and the bind matrix - kept
    on the cage as matrix_parent_inverse - brings them home. Object space
    never changes with the object transform, so moving or rotating the body
    after Tetrahedralize cannot scramble the node-vertex correspondence the
    way a world-space synthesis would.

    KNOWN LIMITATION: computed once and reused. Bone transform motion is
    captured every frame through the evaluated positions, but an edit
    that moves rest vertices without re-tetrahedralizing leaves stale
    weights - the same rule as the bind data.
    """
    from ..blender.session import find_cage

    cage = find_cage(obj)
    if cage is None:
        raise ValueError(
            f"{obj.name!r} has no Marrow cage. Run Tetrahedralize first."
        )
    cached = read_attach(cage.data)
    if cached is not None and cached[0].shape[0] == tetmesh.n_nodes:
        return cached

    rest = read_rest(obj.data)
    if rest is None or rest.shape[0] != len(obj.data.vertices):
        raise ValueError(
            f"{obj.name!r} has no stored rest shape; re-run Tetrahedralize."
        )
    to_local = np.array(cage.matrix_parent_inverse.to_4x4())
    nodes_local = tetmesh.nodes @ to_local[:3, :3].T + to_local[:3, 3]
    idx, w = synth_weights(nodes_local, rest)
    write_attach(cage.data, idx, w)
    return idx, w


def sample_targets(obj, idx, w):
    """This frame's world-space targets for every cage node.

    Sampling has to see the armature deforming the REST shape, but Marrow
    writes its simulated positions straight into the mesh every frame -
    evaluating now would deform the simulation and feed it back into its
    own targets. So the stored rest shape is swapped in, the depsgraph is
    evaluated, and the mesh is swapped back to where it was. One
    evaluated-mesh read per frame; the blend itself is one sparse matmul.
    """
    import bpy

    mesh = obj.data
    rest = read_rest(mesh)
    count = len(mesh.vertices)
    if rest is None or rest.shape[0] != count:
        raise ValueError(
            f"{obj.name!r} has no stored rest shape; re-run Tetrahedralize."
        )

    current = np.empty(count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", current)

    stored = obj.get(DISPLAY_KEY)

    # Unprobed rows (stored by the toggle, which cannot probe) are sorted
    # out here, where the depsgraph works: count-changing modifiers wake
    # back up, because they only smooth the display - their vertices have
    # no per-base-vertex meaning, so they cannot feed the targets.
    if stored is not None and any(len(row) < 3 for row in stored.values()):
        rows = {}
        for m in obj.modifiers:
            row = stored.get(m.name)
            if row is None:
                continue
            if len(row) < 3:
                preserves = _preserves_count(obj, m)
                row = [bool(row[0]), bool(row[1]), preserves]
                if not preserves:
                    m.show_viewport = row[0]
                    m.show_render = row[1]
            rows[m.name] = [bool(row[0]), bool(row[1]), bool(row[2])]
        obj[DISPLAY_KEY] = rows
        stored = obj[DISPLAY_KEY]

    # Count-preserving modifiers are muted in the display while attachment
    # is on, but sampling is exactly the evaluation that needs them on.
    # Count-changing ones are the opposite: on for display, off here, since
    # targets are weighted per base vertex. Modifiers added after the mute
    # are probed once and sorted the same way.
    swapped, parked, adopted = [], [], []
    for m in obj.modifiers:
        row = stored.get(m.name) if stored is not None else None
        if row is None and not m.show_viewport:
            continue  # the user's own disabled modifier stays out of it
        preserves = bool(row[2]) if row is not None else _preserves_count(obj, m)
        if row is None and preserves:
            # Adopted into the mute: it would re-bend the display.
            adopted.append((m, bool(m.show_render)))
            m.show_viewport = False
            m.show_render = False
            row = [True, True, True]
        if preserves and not m.show_viewport:
            m.show_viewport = True
            swapped.append(m)
        elif not preserves and m.show_viewport:
            m.show_viewport = False
            parked.append(m)
    if adopted:
        rows = {name: list(row) for name, row in (stored or {}).items()}
        for m, render in adopted:
            rows[m.name] = [True, render, True]
        obj[DISPLAY_KEY] = rows

    # Restore only after the evaluated mesh has been read; an exception in
    # between must not strand the object at its rest shape.
    targets = None
    try:
        mesh.vertices.foreach_set("co", rest.ravel())
        mesh.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        try:
            eval_mesh = evaluated.to_mesh()
            if eval_mesh is None or len(eval_mesh.vertices) != count:
                got = 0 if eval_mesh is None else len(eval_mesh.vertices)
                raise ValueError(
                    f"{obj.name!r}: evaluated mesh has {got} vertices, "
                    f"attachment weights were built for {count}. A modifier "
                    f"changed the vertex count; re-run Tetrahedralize."
                )
            verts = np.empty(count * 3, dtype=np.float64)
            eval_mesh.vertices.foreach_get("co", verts)
        finally:
            evaluated.to_mesh_clear()
        verts = verts.reshape(-1, 3)
        # Targets live in the solver's frame - the world frame at
        # tetrahedralize time, kept on the cage as the inverse of
        # matrix_parent_inverse - not in the object's current world frame.
        # The evaluated mesh is local, the same space the object-space
        # weights were measured against, so the bind matrix is the only
        # transform in the blend. Moving the object after Tetrahedralize
        # therefore cannot scramble the targets, matching the rest of
        # Marrow: the simulation stays where it was tetrahedralized.
        from ..blender.session import find_cage

        cage = find_cage(obj)
        if cage is None:
            raise ValueError(
                f"{obj.name!r} has no Marrow cage. Run Tetrahedralize first."
            )
        bind = np.array(cage.matrix_parent_inverse.inverted().to_4x4())
        bind_verts = verts @ bind[:3, :3].T + bind[:3, 3]
        targets = targets_from(idx, w, bind_verts)
    finally:
        for m in swapped:
            m.show_viewport = False
        for m in parked:
            m.show_viewport = True
        mesh.vertices.foreach_set("co", current)
        mesh.update()
    return targets
