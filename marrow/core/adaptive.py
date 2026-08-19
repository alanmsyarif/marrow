"""Adaptive tet lattice: octree refinement with hanging-node glue.

A uniform grid resolves the thinnest feature it must capture and pays for it
everywhere. Here an octree refines a cell while its size exceeds the distance
from its centre to the surface, so the boundary layer and thin features sit
at the min size while the bulk stays at the max. Leaves are tetrahedralized
with the same parity-alternated 5-tet split as the uniform lattice, so
equal-size neighbours conform exactly as they do there.

Where a fine face meets a 2x coarse face, the extra nodes - coarse edge
midpoints and the coarse face centre - become hanging nodes. Axis-aligned
cube faces are planar, so every triangulation of a face agrees on the linear
displacement field and the hanging nodes are glued by plain interpolation
rows (x_h = sum w_i x_i) that the solver projects as an extra XPBD pass.
"""

import math

import numpy as np

from .lattice import _CORNER_OFFSETS, _SPLIT_EVEN, _SPLIT_ODD, grid_dims
from .tetmesh import TetMesh, repair_orientation

_REFINE_RATIO = 1.0

# The six face directions, as cell-coordinate offsets.
_FACES = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


def levels(max_size: float, min_size: float) -> int:
    """Refinement depth: leaf sizes are exact halvings of the max size."""
    return max(0, math.ceil(math.log2(float(max_size) / float(min_size))))


def refine(max_size, min_size, oracle) -> np.ndarray:
    """Octree leaves as (M, 4) int64 of (level, i, j, k).

    ``oracle`` exposes ``bounds_min``/``bounds_max`` plus ``distance(point)``
    and ``inside(point)``. A cell refines while its size exceeds
    ``_REFINE_RATIO`` times the centre's distance to the surface, down to the
    min size. Cells far outside or deep inside stop early; the boundary layer
    and thin features run to the min level. The inside test is deliberately
    not run here - a cell that will refine further does not need it.
    """
    n_levels = levels(max_size, min_size)
    bounds_min = np.asarray(oracle.bounds_min, dtype=np.float64)
    dims = grid_dims(oracle.bounds_min, oracle.bounds_max, max_size)

    frontier = [
        (0, i, j, k)
        for i in range(dims[0])
        for j in range(dims[1])
        for k in range(dims[2])
    ]
    leaves = []
    for level in range(n_levels + 1):
        size = float(max_size) / (1 << level)
        children = []
        for lv, i, j, k in frontier:
            centre = bounds_min + (np.array([i, j, k]) + 0.5) * size
            if level < n_levels and size > _REFINE_RATIO * oracle.distance(centre):
                bi, bj, bk = i * 2, j * 2, k * 2
                for off in _CORNER_OFFSETS:
                    children.append(
                        (level + 1, bi + off[0], bj + off[1], bk + off[2])
                    )
            else:
                leaves.append((lv, i, j, k))
        frontier = children
    return np.array(leaves, dtype=np.int64)


def _split(leaf_set, queue, cell, target) -> None:
    """Split ``cell`` down to ``target``, enqueueing the new leaves."""
    leaf_set.discard(cell)
    stack = [cell]
    while stack:
        lv, i, j, k = stack.pop()
        if lv == target:
            leaf_set.add((lv, i, j, k))
            queue.append((lv, i, j, k))
            continue
        bi, bj, bk = i * 2, j * 2, k * 2
        for off in _CORNER_OFFSETS:
            stack.append((lv + 1, bi + off[0], bj + off[1], bk + off[2]))


def balance(leaves: np.ndarray, root_dims) -> np.ndarray:
    """Enforce 2:1 - no face-adjacent leaves differ by more than one level.

    For a leaf at level l, its face neighbour region must be covered by a
    leaf at level l-1 or finer; a coarser covering leaf is split down to
    l-1. Checking the leaf's own neighbours (not just its parent's) also
    catches the root - a level-1 leaf has no parent face neighbours to
    check against. Processing is queued, so splits that create new
    violations are themselves checked.
    """
    root_dims = tuple(int(d) for d in root_dims)
    leaf_set = set(map(tuple, leaves))

    def covering(level, ijk):
        """The leaf containing cell ``ijk`` at ``level``, walking up."""
        while level > 0 and (level, *ijk) not in leaf_set:
            level -= 1
            ijk = (ijk[0] >> 1, ijk[1] >> 1, ijk[2] >> 1)
        return (level, *ijk) if (level, *ijk) in leaf_set else None

    queue = list(leaf_set)
    head = 0
    while head < len(queue):
        lv, i, j, k = queue[head]
        head += 1
        if lv == 0:
            continue
        span = 1 << lv
        for di, dj, dk in _FACES:
            nb = (i + di, j + dj, k + dk)
            if not (
                0 <= nb[0] < root_dims[0] * span
                and 0 <= nb[1] < root_dims[1] * span
                and 0 <= nb[2] < root_dims[2] * span
            ):
                continue
            cov = covering(lv, nb)
            if cov is not None and cov[0] < lv - 1:
                _split(leaf_set, queue, cov, lv - 1)
    return np.array(sorted(leaf_set), dtype=np.int64)


def build_adaptive_lattice(max_size, min_size, oracle):
    """(TetMesh, blend_idx, blend_w) over the inside leaves of the octree.

    ``blend_idx`` is (R, 5) int32 of [hanging, m0..m3] and ``blend_w`` the
    (R, 4) master weights, summing to one; edge rows carry masters
    (a, b, a, b) with weights (0.5, 0.5, 0, 0) and the kernel skips the
    zero-weight slots. Rows are sorted by master level so a Gauss-Seidel
    sweep sees fresh master values down the hanging chains.
    """
    n_levels = levels(max_size, min_size)
    bounds_min = np.asarray(oracle.bounds_min, dtype=np.float64)
    root_dims = grid_dims(oracle.bounds_min, oracle.bounds_max, max_size)
    empty = (
        TetMesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)),
        np.zeros((0, 5), dtype=np.int32),
        np.zeros((0, 4), dtype=np.float64),
    )

    leaves = refine(max_size, min_size, oracle)
    if leaves.size == 0:
        return empty
    leaves = balance(leaves, root_dims)

    # Inside test on leaf centres only - the same occupancy rule as the
    # uniform mask, applied at each leaf's own size.
    kept = []
    for lv, i, j, k in leaves:
        size = float(max_size) / (1 << lv)
        centre = bounds_min + (np.array([i, j, k]) + 0.5) * size
        if oracle.inside(centre):
            kept.append((lv, i, j, k))
    if not kept:
        return empty
    kept_set = set(kept)

    # Node dedup on exact integer coordinates at the finest level, so a
    # hanging node named by several faces is one node, never a near-miss.
    node_index: dict = {}
    nodes = []

    def node_id(key):
        idx = node_index.get(key)
        if idx is None:
            idx = len(nodes)
            node_index[key] = idx
            nodes.append(key)
        return idx

    tets = []
    for lv, i, j, k in kept:
        step = 1 << (n_levels - lv)
        base = np.array([i, j, k]) * step
        local = np.empty((8,), dtype=np.int64)
        for c, off in enumerate(_CORNER_OFFSETS):
            key = (
                int(base[0] + off[0] * step),
                int(base[1] + off[1] * step),
                int(base[2] + off[2] * step),
            )
            local[c] = node_id(key)
        parity = (i + j + k) & 1
        split = _SPLIT_ODD if parity else _SPLIT_EVEN
        for tet in split:
            tets.append(local[tet])

    # Finest-unit positions.
    scale = float(max_size) / (1 << n_levels)
    positions = (np.array(nodes, dtype=np.float64) * scale) + bounds_min

    # Hanging rows: for each kept leaf face whose neighbour is one level
    # coarser, the two fine corners that are not coarse corners interpolate
    # the coarse face. Emit from the fine side only, dedup by hanging node.
    rows: dict = {}

    def covering(level, ijk):
        while level > 0 and (level, *ijk) not in kept_set:
            level -= 1
            ijk = (ijk[0] >> 1, ijk[1] >> 1, ijk[2] >> 1)
        return (level, *ijk) if (level, *ijk) in kept_set else None

    for lv, i, j, k in kept:
        if lv == 0:
            continue
        step = 1 << (n_levels - lv)
        base = np.array([i, j, k]) * step
        for axis, side in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)):
            direction = [0, 0, 0]
            direction[axis] = 1 if side else -1
            nb = (i + direction[0], j + direction[1], k + direction[2])
            if (lv, *nb) in kept_set:
                continue  # conforming same-level face
            span = 1 << lv
            if not (
                0 <= nb[0] < root_dims[0] * span
                and 0 <= nb[1] < root_dims[1] * span
                and 0 <= nb[2] < root_dims[2] * span
            ):
                continue  # face on the outer boundary
            if (lv + 1, i * 2 + direction[0], j * 2 + direction[1],
                    k * 2 + direction[2]) in kept_set:
                continue  # we are the coarse side; the fine leaf emits
            cov = covering(lv, nb)
            if cov is None or cov[0] != lv - 1:
                continue  # balance makes this unreachable; refuse anyway
            ci, cj, ck = cov[1:]
            cbase = np.array([ci, cj, ck]) * (step * 2)
            tangents = [t for t in (0, 1, 2) if t != axis]

            def corner_id(u2, v2):
                off = [0, 0, 0]
                # The coarse leaf sits on the far side of the face, so its
                # coincident face carries the opposite local offset.
                off[axis] = 1 - side
                off[tangents[0]] = u2 >> 1
                off[tangents[1]] = v2 >> 1
                return node_index[
                    (int(cbase[0] + off[0] * step * 2),
                     int(cbase[1] + off[1] * step * 2),
                     int(cbase[2] + off[2] * step * 2))
                ]

            corners = [c for c, off in enumerate(_CORNER_OFFSETS)
                       if off[axis] == side]
            for c in corners:
                off = _CORNER_OFFSETS[c]
                # Twice the in-face coordinate relative to the coarse cell,
                # in units of the fine size: 0, 1 or 2.
                u2 = (int(base[tangents[0]] + off[tangents[0]] * step)
                      - int(cbase[tangents[0]])) // step
                v2 = (int(base[tangents[1]] + off[tangents[1]] * step)
                      - int(cbase[tangents[1]])) // step
                if (u2, v2) in ((0, 0), (2, 0), (0, 2), (2, 2)):
                    continue  # a coarse corner: conforming
                h = node_index[
                    (int(base[0] + off[0] * step),
                     int(base[1] + off[1] * step),
                     int(base[2] + off[2] * step))
                ]
                if h in rows:
                    continue
                if u2 == 1 and v2 == 1:
                    masters = (corner_id(0, 0), corner_id(2, 0),
                               corner_id(0, 2), corner_id(2, 2))
                    w = (0.25, 0.25, 0.25, 0.25)
                elif u2 == 1:
                    a, b = corner_id(0, v2), corner_id(2, v2)
                    masters, w = (a, b, a, b), (0.5, 0.5, 0.0, 0.0)
                else:
                    a, b = corner_id(u2, 0), corner_id(u2, 2)
                    masters, w = (a, b, a, b), (0.5, 0.5, 0.0, 0.0)
                rows[h] = (lv - 1, (h, *masters), w)

    ordered = sorted(rows.values(), key=lambda r: r[0])
    blend_idx = np.array([r[1] for r in ordered], dtype=np.int32).reshape(-1, 5)
    blend_w = np.array([r[2] for r in ordered], dtype=np.float64).reshape(-1, 4)

    tets = repair_orientation(np.array(tets, dtype=np.int32), positions)
    return TetMesh(positions, tets), blend_idx, blend_w
