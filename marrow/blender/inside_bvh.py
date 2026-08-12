"""Inside/outside testing against a Blender object, via BVH ray parity."""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core.lattice import grid_dims

_RAY = Vector((0.5773502691896258, 0.4082482904638631, 0.7071067811865476))
_EPS = 1e-6


def _world_bvh(obj):
    """BVH of the evaluated object in world space, plus its world bounds."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = obj.matrix_world
        verts = [matrix @ v.co for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
        tris = []
        for poly in polys:  # fan-triangulate, BVHTree wants tris or quads
            for i in range(1, len(poly) - 1):
                tris.append((poly[0], poly[i], poly[i + 1]))
        bvh = BVHTree.FromPolygons(verts, tris, all_triangles=True)
        coords = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()
    return bvh, coords


def _is_inside(bvh, point: Vector) -> bool:
    """Odd number of forward hits means inside."""
    hits = 0
    origin = point.copy()
    while True:
        location, _normal, _index, _dist = bvh.ray_cast(origin, _RAY)
        if location is None:
            break
        hits += 1
        origin = location + _RAY * _EPS
    return hits % 2 == 1


def cell_mask_from_object(obj, spacing: float):
    """Voxel occupancy of ``obj`` at ``spacing``, by cell-centre inside test."""
    bvh, coords = _world_bvh(obj)
    bounds_min = coords.min(axis=0)
    bounds_max = coords.max(axis=0)
    dims = grid_dims(bounds_min, bounds_max, spacing)

    mask = np.zeros(dims, dtype=bool)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                centre = bounds_min + (np.array([i, j, k]) + 0.5) * spacing
                mask[i, j, k] = _is_inside(bvh, Vector(centre.tolist()))
    return mask, bounds_min
