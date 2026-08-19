"""Inside/outside testing against a Blender object, via BVH ray parity."""

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core.lattice import grid_dims
from ..core.progress import drain

_EPS = 1e-6
# Three directions, none axis-aligned and none a multiple of another, so a ray
# that grazes an edge or a vertex is outvoted rather than believed.
_RAYS = (
    Vector((0.5773502691896258, 0.4082482904638631, 0.7071067811865476)),
    Vector((-0.3333333333333333, 0.6666666666666666, -0.6666666666666666)),
    Vector((0.8017837257372732, -0.5345224838248488, 0.2672612419124244)),
)


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


def _hits_odd(bvh, point: Vector, ray: Vector) -> bool:
    """Odd number of forward hits means inside."""
    hits = 0
    origin = point.copy()
    while True:
        location, _normal, _index, _dist = bvh.ray_cast(origin, ray)
        if location is None:
            break
        hits += 1
        origin = location + ray * _EPS
    return hits % 2 == 1


def is_inside(bvh, point: Vector) -> bool:
    """Majority of three ray-parity tests.

    A single ray grazing a face, an edge or a vertex miscounts the hits, and
    a wrong sign either punches a hole in the cage or floats a detached cube
    of tets outside the mesh. Measured on a UV sphere at cell 0.1: 113 wrong
    signs with one ray, none with three. It costs three casts on what is a
    one-time CPU operation either way.
    """
    votes = sum(1 for ray in _RAYS if _hits_odd(bvh, point, ray))
    return votes >= 2


def cell_mask_from_object(obj, spacing: float):
    """Voxel occupancy of ``obj`` at ``spacing``, by cell-centre inside test."""
    return drain(cell_mask_iter(obj, spacing))


def cell_mask_iter(obj, spacing: float):
    """cell_mask_from_object as a generator, yielding 0..1 per x-plane.

    This pass is the freeze: three ray casts per cell, in Python, over the
    whole bounding box. Measured at 54.7s of a 66.2s Tetrahedralize on a
    34k-vertex mesh at Resolution 0.08 - 6.5M cells - with Blender's window
    reporting Not Responding for the duration.

    One yield per x-plane rather than per cell: a plane is a few milliseconds
    of work at any sane resolution, which is fine grained enough for the
    operator to repaint between slices, while a yield per cell would put
    generator overhead on the hottest loop in the addon.
    """
    bvh, coords = _world_bvh(obj)
    bounds_min = coords.min(axis=0)
    bounds_max = coords.max(axis=0)
    dims = grid_dims(bounds_min, bounds_max, spacing)

    mask = np.zeros(dims, dtype=bool)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                centre = bounds_min + (np.array([i, j, k]) + 0.5) * spacing
                mask[i, j, k] = is_inside(bvh, Vector(centre.tolist()))
        if i + 1 < dims[0]:
            yield (i + 1) / dims[0]

    yield 1.0
    return mask, bounds_min


def cell_oracle_from_object(obj):
    """(bounds_min, oracle) driving the adaptive octree.

    The oracle answers the two questions refinement asks about a cell
    centre: how far is the surface (BVH nearest point) and which side of
    it are we on (the three-ray parity vote). Both run in world space,
    like the uniform mask. Distance to the nearest triangle is exact
    enough for the refine rule - it only decides cell sizes.
    """
    bvh, coords = _world_bvh(obj)

    class _Oracle:
        bounds_min = coords.min(axis=0)
        bounds_max = coords.max(axis=0)

        def distance(self, point):
            location, _normal, _index, dist = bvh.find_nearest(Vector(point))
            if location is None:
                return float("inf")
            return float(dist)

        def inside(self, point):
            return is_inside(bvh, Vector(point))

    return _Oracle.bounds_min.copy(), _Oracle()

