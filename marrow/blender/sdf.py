"""Bake a mesh collider into a signed distance field.

Analytic primitives cannot express an arbitrary shape. An SDF can, and
depenetration by walking its gradient handles concavity, holes and thin
features the same way it handles a convex blob.

The field is baked in the collider's LOCAL space, so the object transform
places, rotates and scales it exactly as it already does the unit sphere and
the unit box. A collider that moves needs no rebake.
"""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

_EPS = 1e-6
# Three directions, none axis-aligned and none a multiple of another, so a ray
# that grazes an edge or a vertex is outvoted rather than believed.
_RAYS = (
    Vector((0.5773502691896258, 0.4082482904638631, 0.7071067811865476)),
    Vector((-0.3333333333333333, 0.6666666666666666, -0.6666666666666666)),
    Vector((0.8017837257372732, -0.5345224838248488, 0.2672612419124244)),
)


def _hits_odd(bvh, point, ray) -> bool:
    hits = 0
    origin = point.copy()
    while True:
        location, _normal, _index, _dist = bvh.ray_cast(origin, ray)
        if location is None:
            break
        hits += 1
        origin = location + ray * _EPS
    return hits % 2 == 1


def _is_inside(bvh, point) -> bool:
    """Majority of three ray-parity tests.

    inside_bvh casts one fixed ray, which is enough for cage occupancy, where
    cell centres are spaced a whole Resolution apart. An SDF samples densely
    right against the surface, and there a single ray grazing a face or an
    edge miscounts: measured 113 wrong signs on a UV sphere at cell 0.1, and
    a wrong sign is a 2x radius error in the field. Three rays fix it.
    """
    votes = sum(1 for ray in _RAYS if _hits_odd(bvh, point, ray))
    return votes >= 2


MIN_CELLS = 16
MAX_CELLS = 96
PAD_CELLS = 2

# Keyed by (mesh name, vertex count, dims). A live restart rebuilds solvers
# every time the timeline returns to the start, and re-paying the bake each
# time is avoidable. The vertex count is a cheap guard against the obvious
# edit, not full change detection - see the spec's ceilings.
_CACHE: dict = {}


def _local_bvh(obj):
    """BVH of the evaluated object in its own local space, plus its bounds."""
    import bpy

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    try:
        mesh = evaluated.to_mesh()
    except (RuntimeError, AttributeError):
        # An empty, a camera, a light. Mesh is the default collider shape, so
        # this is reachable by picking any object at all, and it must skip
        # rather than take the whole session down.
        return None, None
    if mesh is None:
        return None, None
    try:
        verts = [v.co.copy() for v in mesh.vertices]
        tris = []
        for poly in mesh.polygons:
            loop = tuple(poly.vertices)
            for i in range(1, len(loop) - 1):
                tris.append((loop[0], loop[i], loop[i + 1]))
        if not verts or not tris:
            return None, None
        bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
        coords = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()
    return bvh, coords


def bake(obj, cell_size: float):
    """``(field, grid_from_local)`` for ``obj``, or ``(None, None)``.

    ``field`` is (nz, ny, nx) float32 of signed distance in grid units,
    negative inside. ``grid_from_local`` is the 4x4 mapping the padded local
    bounding box onto the unit cube, which the caller composes into the
    collider's to_local so the kernel needs no extra push constants.
    """
    bvh, coords = _local_bvh(obj)
    if bvh is None:
        return None, None

    cell = max(float(cell_size), 1e-6)
    low, high = coords.min(axis=0), coords.max(axis=0)
    # A CUBIC box. Mapping a non-cubic one onto the unit cube would scale each
    # axis differently, and a distance field does not survive that - the
    # gradient would point somewhere that is not the nearest surface. A cube
    # makes grid space a uniform scale of local space, so dividing by the span
    # leaves a true distance. Padded so the zero isosurface is never clipped
    # and the gradient has a texel of room at the boundary.
    half = float(np.max(high - low)) / 2.0 + PAD_CELLS * cell
    centre = (low + high) / 2.0
    low = centre - half
    width = max(2.0 * half, 1e-6)
    n = int(np.clip(int(np.ceil(width / cell)), MIN_CELLS, MAX_CELLS))

    key = (obj.data.name, len(coords), n)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    nx = ny = nz = n
    # Cell centres, so no sample sits exactly on a face, where parity is
    # least reliable.
    ticks = (np.arange(n) + 0.5) * (width / n)
    axes = [low[a] + ticks for a in range(3)]
    field = np.empty((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                point = Vector((axes[0][i], axes[1][j], axes[2][k]))
                location, _normal, _index, distance = bvh.find_nearest(point)
                if location is None:
                    field[k, j, i] = 1.0e9
                    continue
                # Ray parity for the sign. Measured cheaper than find_nearest
                # itself, and the nearest-face-normal shortcut most bakers use
                # disagrees with it on 2.9% of Suzanne's voxels.
                signed = -distance if _is_inside(bvh, point) else distance
                field[k, j, i] = signed

    # Distances are in local units; the kernel works in grid units, where the
    # box is the unit cube. One uniform divide, because the box is a cube.
    field /= width

    grid = np.eye(4)
    grid[0, 0] = grid[1, 1] = grid[2, 2] = 1.0 / width
    grid[:3, 3] = -low / width

    _CACHE[key] = (field, grid)
    return field, grid


def clear_cache() -> None:
    _CACHE.clear()
