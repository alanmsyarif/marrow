"""A Curve object as an ordered world-space polyline.

Fibers are sampled against the curve after Blender has evaluated it, so
resolution, modifiers and shape keys are all accounted for. Evaluating to a
mesh is what makes that possible - but a mesh has no notion of "along", so
the vertices have to be walked back into path order through the edges.
"""

import numpy as np


def polyline_from_curve(context, curve_obj) -> np.ndarray:
    """Ordered world-space points along ``curve_obj``, (S, 3).

    Returns an empty (0, 3) array for anything that is not a curve, or for a
    curve that does not evaluate to a single open path - a bevelled or
    extruded curve becomes a tube, and a tube has no unambiguous direction
    to hand a fiber. Callers treat empty as "no fibers", which is the same
    thing they do when no curve was set at all.
    """
    if curve_obj is None or curve_obj.type != "CURVE":
        return np.zeros((0, 3), dtype=np.float64)

    depsgraph = context.evaluated_depsgraph_get()
    evaluated = curve_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    # None for a curve datablock with no splines - deleting every
    # control point in edit mode is enough. Reading .vertices off that
    # raises from inside Tetrahedralize, after the old cage has been
    # removed and before the new one is linked, which costs the user a
    # whole cage rebuild for an empty curve.
    if mesh is None:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        count = len(mesh.vertices)
        if count < 2:
            return np.zeros((0, 3), dtype=np.float64)

        coords = np.empty(count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)

        neighbours = [[] for _ in range(count)]
        for edge in mesh.edges:
            a, b = edge.vertices[0], edge.vertices[1]
            neighbours[a].append(b)
            neighbours[b].append(a)

        order = _walk_chain(neighbours, count)
        if order is None:
            return np.zeros((0, 3), dtype=np.float64)
        coords = coords[order]
    finally:
        # to_mesh() allocates; the matching free is not optional.
        evaluated.to_mesh_clear()

    world = np.array(curve_obj.matrix_world.to_4x4())
    return coords @ world[:3, :3].T + world[:3, 3]


def _walk_chain(neighbours, count):
    """Vertex indices from one end of an open chain to the other, or None.

    None means the evaluated geometry is not a simple path: a branch, a ring
    or several disconnected pieces. Guessing a direction through any of
    those would put fibers somewhere the user did not ask for, so the caller
    declines instead.
    """
    degrees = [len(n) for n in neighbours]
    if any(d == 0 or d > 2 for d in degrees):
        return None
    ends = [i for i, d in enumerate(degrees) if d == 1]
    if len(ends) != 2:
        return None

    order = [ends[0]]
    previous = -1
    current = ends[0]
    while True:
        nexts = [n for n in neighbours[current] if n != previous]
        if not nexts:
            break
        previous, current = current, nexts[0]
        order.append(current)
        if len(order) > count:
            return None
    return order if len(order) == count else None
