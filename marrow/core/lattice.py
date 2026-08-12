"""Conforming tet lattice by 5-tet cube subdivision with parity alternation."""

import numpy as np

from ..core.tetmesh import TetMesh, repair_orientation

# Cube corner order: bit 0 = x, bit 1 = y, bit 2 = z.
_CORNER_OFFSETS = np.array(
    [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.int64
)

# Five-tet splits. The two patterns are mirror images; alternating them by
# (i+j+k) parity makes neighbouring cells agree on shared face diagonals.
_SPLIT_EVEN = np.array(
    [[0, 1, 2, 4], [1, 3, 2, 7], [1, 4, 5, 7], [2, 4, 6, 7], [1, 2, 4, 7]],
    dtype=np.int64,
)
_SPLIT_ODD = np.array(
    [[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 4, 6], [3, 5, 7, 6], [0, 3, 5, 6]],
    dtype=np.int64,
)


def grid_dims(bounds_min, bounds_max, spacing: float) -> tuple[int, int, int]:
    """Cell counts covering the bounds, at least 1 per axis."""
    extent = np.asarray(bounds_max, dtype=np.float64) - np.asarray(
        bounds_min, dtype=np.float64
    )
    counts = np.ceil(extent / float(spacing)).astype(np.int64)
    counts = np.maximum(counts, 1)
    return tuple(int(c) for c in counts)


def build_lattice(bounds_min, spacing: float, cell_mask: np.ndarray) -> TetMesh:
    """Build a conforming tet mesh over every True cell of ``cell_mask``."""
    mask = np.asarray(cell_mask, dtype=bool)
    nx, ny, nz = mask.shape
    occupied = np.argwhere(mask)  # (M, 3) cell coordinates

    if occupied.size == 0:
        return TetMesh(
            np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)
        )

    # Global corner-lattice index: (nx+1, ny+1, nz+1) grid of potential nodes.
    def corner_id(ijk):
        return (ijk[..., 0] * (ny + 1) + ijk[..., 1]) * (nz + 1) + ijk[..., 2]

    # (M, 8, 3) corner coordinates for each occupied cell.
    cell_corners = occupied[:, None, :] + _CORNER_OFFSETS[None, :, :]
    cell_corner_ids = corner_id(cell_corners)  # (M, 8)

    used_ids, inverse = np.unique(cell_corner_ids.ravel(), return_inverse=True)
    local = inverse.reshape(cell_corner_ids.shape)  # (M, 8) compacted indices

    # Recover lattice coordinates of the used corners to build positions.
    zi = used_ids % (nz + 1)
    yi = (used_ids // (nz + 1)) % (ny + 1)
    xi = used_ids // ((nz + 1) * (ny + 1))
    lattice = np.stack([xi, yi, zi], axis=1).astype(np.float64)
    nodes = np.asarray(bounds_min, dtype=np.float64) + lattice * float(spacing)

    parity = (occupied.sum(axis=1) % 2).astype(bool)  # (M,)
    splits = np.where(parity[:, None, None], _SPLIT_ODD[None], _SPLIT_EVEN[None])

    # local[m, splits[m, t, c]] -> (M, 5, 4)
    tets = np.take_along_axis(local[:, None, :], splits, axis=2)
    tets = tets.reshape(-1, 4).astype(np.int32)
    tets = repair_orientation(tets, nodes)

    return TetMesh(nodes, tets)
