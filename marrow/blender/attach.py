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


def ensure_weights(obj, tetmesh):
    """``(idx, w)`` for ``obj``'s cage, computing and caching them if new.

    Weights are against the REST shape in world space: cage nodes are
    stored world-space, and so must be the vertices they are measured to.
    Computed lazily when attachment is first enabled rather than at
    tetrahedralize time, so a body that never attaches pays nothing.

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
    world = np.array(obj.matrix_world.to_4x4())
    world_rest = rest @ world[:3, :3].T + world[:3, 3]
    idx, w = synth_weights(tetmesh.nodes, world_rest)
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
            world = np.array(evaluated.matrix_world.to_4x4())
        finally:
            evaluated.to_mesh_clear()
        verts = verts.reshape(-1, 3)
        world_verts = verts @ world[:3, :3].T + world[:3, 3]
        targets = targets_from(idx, w, world_verts)
    finally:
        mesh.vertices.foreach_set("co", current)
        mesh.update()
    return targets
