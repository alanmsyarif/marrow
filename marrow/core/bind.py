"""Bind arbitrary points into a tet cage and deform them with it."""

import numpy as np

from .progress import drain
from .tetmesh import signed_volumes

# A tet flatter than this makes its shape matrix singular, and because the
# solve is batched over every tet at once, one such tet fails the bind for
# every point in the mesh - including points sitting inside healthy tets.
_MIN_TET_VOLUME = 1e-12

_MAX_REPORTED = 8


def _check_bindable(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray) -> None:
    """Reject inputs that would fail opaquely or, worse, quietly."""
    if not np.all(np.isfinite(nodes)):
        raise ValueError("cage nodes contain NaN or inf")
    if not np.all(np.isfinite(points)):
        raise ValueError("points to bind contain NaN or inf")

    if tets.shape[0] == 0:
        return
    flat = np.flatnonzero(np.abs(signed_volumes(nodes, tets)) < _MIN_TET_VOLUME)
    if flat.size:
        shown = ", ".join(str(int(t)) for t in flat[:_MAX_REPORTED])
        more = "" if flat.size <= _MAX_REPORTED else f", and {flat.size - _MAX_REPORTED} more"
        raise ValueError(
            f"{flat.size} degenerate tets cannot be bound against "
            f"(volume below {_MIN_TET_VOLUME}): {shown}{more}"
        )


def _barycentric_all(nodes: np.ndarray, tets: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Barycentric coords of one point in every tet given. Returns (T, 4).

    Called with a handful of candidate tets rather than the whole cage - see
    _TetGrid. It still costs one 3x3 solve per row, which is why the row
    count is what had to come down.
    """
    p0 = nodes[tets[:, 0]]
    m = np.stack(
        [
            nodes[tets[:, 1]] - p0,
            nodes[tets[:, 2]] - p0,
            nodes[tets[:, 3]] - p0,
        ],
        axis=2,
    )  # (T, 3, 3) columns are the edge vectors
    rhs = point[None, :] - p0  # (T, 3)
    solved = np.linalg.solve(m, rhs[..., None])[..., 0]  # (T, 3)
    first = 1.0 - solved.sum(axis=1)
    return np.concatenate([first[:, None], solved], axis=1)


class _TetGrid:
    """Uniform grid over tet bounding boxes, for candidate lookup.

    Binding used to solve a 3x3 system for every (point, tet) pair: 200ns per
    pair measured, so a 34k-vertex mesh against a 500k-tet cage sat for 57
    minutes with Blender's main thread blocked. Points only ever bind to a
    tet whose bounding box covers them, so all but a handful of those solves
    were answering a question the bounding boxes had already settled.

    The pitch is the largest tet's longest side. That is what makes the
    lookup exact with a single cell read: a tet can then straddle at most two
    cells per axis, it is registered in every cell it touches, so any tet
    whose box covers the query point is registered in the point's own cell.
    An adaptive cage of mixed sizes is bounded by its coarsest tet and so
    gets a coarser grid than its fine region would like - still bounded work,
    just less of a win there.
    """

    def __init__(self, nodes: np.ndarray, tets: np.ndarray):
        corners = nodes[tets]                      # (T, 4, 3)
        lo = corners.min(axis=1)
        hi = corners.max(axis=1)

        self.origin = lo.min(axis=0)
        # A cage of one degenerate-thin tet would give a zero pitch and a
        # division by zero; the floor keeps the grid valid rather than fast.
        self.pitch = max(float((hi - lo).max()), 1e-9)

        span = hi.max(axis=0) - self.origin
        self.dims = np.maximum(np.floor(span / self.pitch).astype(np.int64) + 1, 1)

        lo_i = self._cell_of(lo)
        hi_i = self._cell_of(hi)

        # Every cell each box touches. The pitch guarantees at most two per
        # axis, so the eight corner offsets cover it exactly and this stays
        # vectorised instead of becoming a ragged per-tet loop.
        keys, owners = [], []
        ids = np.arange(tets.shape[0], dtype=np.int64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    cell = lo_i + np.array([dx, dy, dz], dtype=np.int64)
                    live = np.all(cell <= hi_i, axis=1)
                    if not np.any(live):
                        continue
                    keys.append(self._key(cell[live]))
                    owners.append(ids[live])

        key = np.concatenate(keys)
        owner = np.concatenate(owners)
        # Sorted by cell, then by tet index within a cell, so a candidate
        # list always arrives in ascending tet order. The tie-breaks below
        # are index order, and they have to be reproducible.
        order = np.lexsort((owner, key))
        self._key_sorted = key[order]
        self._tet_sorted = owner[order]

        self._centroids = corners.mean(axis=1)
        ckey = self._key(self._cell_of(self._centroids))
        corder = np.lexsort((ids, ckey))
        self._ckey_sorted = ckey[corder]
        self._ctet_sorted = ids[corder]

    def _cell_of(self, points: np.ndarray) -> np.ndarray:
        cell = np.floor((points - self.origin) / self.pitch).astype(np.int64)
        return np.clip(cell, 0, self.dims - 1)

    def _key(self, cell: np.ndarray) -> np.ndarray:
        return (cell[..., 0] * self.dims[1] + cell[..., 1]) * self.dims[2] + cell[..., 2]

    @staticmethod
    def _slice(keys_sorted, values_sorted, key):
        lo = np.searchsorted(keys_sorted, key, side="left")
        hi = np.searchsorted(keys_sorted, key, side="right")
        return values_sorted[lo:hi]

    def candidates(self, point: np.ndarray) -> np.ndarray:
        """Tets whose bounding box may cover ``point``, in index order."""
        key = int(self._key(self._cell_of(np.asarray(point)[None, :]))[0])
        return self._slice(self._key_sorted, self._tet_sorted, key)

    def nearest_centroid(self, point: np.ndarray) -> int:
        """Index of the tet whose centroid is nearest, ties to lowest index.

        Rings outward from the point's own cell. A centroid sitting in a cell
        r rings away is at least (r - 1) pitches off, so once the best seen
        beats that bound nothing further out can win - and the ring that
        proves it is still scanned in full, so an exact tie with a
        lower-numbered tet is found rather than missed.
        """
        point = np.asarray(point, dtype=np.float64)
        home = self._cell_of(point[None, :])[0]
        reach = int(self.dims.max())

        found = []
        best = np.inf
        for ring in range(reach + 1):
            if best < (ring - 1) * self.pitch:
                break
            cells = _ring_cells(home, ring, self.dims)
            if cells.size == 0:
                continue
            for key in np.unique(self._key(cells)):
                tets = self._slice(self._ckey_sorted, self._ctet_sorted, int(key))
                if tets.size:
                    found.append(tets)
                    d = np.linalg.norm(self._centroids[tets] - point, axis=1).min()
                    best = min(best, float(d))

        if not found:                      # cannot happen with tets present
            return 0
        pool = np.unique(np.concatenate(found))
        dist = np.linalg.norm(self._centroids[pool] - point, axis=1)
        return int(pool[int(np.argmin(dist))])


def _ring_cells(home: np.ndarray, ring: int, dims: np.ndarray) -> np.ndarray:
    """In-bounds cells exactly ``ring`` steps from ``home`` in Chebyshev."""
    lo = np.maximum(home - ring, 0)
    hi = np.minimum(home + ring, dims - 1)
    grid = np.stack(
        np.meshgrid(
            np.arange(lo[0], hi[0] + 1),
            np.arange(lo[1], hi[1] + 1),
            np.arange(lo[2], hi[2] + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    if ring == 0:
        return grid
    return grid[np.abs(grid - home).max(axis=1) == ring]


def bind_points(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray):
    """Bind each point to a containing tet, falling back to the nearest one."""
    return drain(bind_points_iter(nodes, tets, points))


def bind_points_iter(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray,
                     block: int = 4096):
    """bind_points as a generator, yielding 0..1 every ``block`` points.

    The grid is built once, before the first yield, and reused across every
    chunk - rebuilding it per chunk would cost O(tets) each time and undo the
    speedup it exists to provide.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    _check_bindable(nodes, tets, points)

    n_points = points.shape[0]
    idx = np.zeros(n_points, dtype=np.int32)
    weights = np.zeros((n_points, 4), dtype=np.float64)

    if tets.shape[0] == 0 or n_points == 0:
        yield 1.0
        return idx, weights

    grid = _TetGrid(nodes, tets)

    for i in range(n_points):
        # Only the tets whose bounding box covers the point can contain it,
        # so only those are solved. This is the whole speedup.
        near = grid.candidates(points[i])
        best = None
        if near.size:
            bary = _barycentric_all(nodes, tets[near], points[i])
            inside = np.all(bary >= -1e-9, axis=1)
            if np.any(inside):
                # Prefer the most interior containment for numerical comfort.
                candidates = np.flatnonzero(inside)
                pick = candidates[np.argmax(bary[candidates].min(axis=1))]
                best = int(near[pick])
                w = bary[pick]

        if best is None:
            best = grid.nearest_centroid(points[i])
            w = np.clip(_barycentric_all(nodes, tets[best][None, :], points[i])[0],
                        0.0, None)
            total = w.sum()
            # Barycentric coordinates sum to 1, so after clipping at least one
            # is positive and this cannot trip. It used to fall back to a
            # uniform 0.25 instead, which a NaN reached and returned quietly -
            # the point was silently bound to the tet centroid. Raise instead.
            if not total > 0.0:
                raise ValueError(
                    f"point {i} produced unusable bind weights against tet {best}"
                )
            w = w / total
        idx[i] = best
        weights[i] = w

        if (i + 1) % block == 0 and (i + 1) < n_points:
            yield (i + 1) / n_points

    yield 1.0
    return idx, weights


def deform(nodes: np.ndarray, tets: np.ndarray, bind_idx: np.ndarray, bind_w: np.ndarray):
    """Interpolate point positions from the current cage node positions."""
    corners = nodes[tets[bind_idx]]  # (P, 4, 3)
    return np.einsum("pij,pi->pj", corners, bind_w)


def bind_slip(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray,
              bind_idx: np.ndarray, bind_w: np.ndarray) -> np.ndarray:
    """How far each point had to move to reach the cage, at rest.

    A point inside a tet is reproduced exactly by its own barycentric
    coordinates, so its slip is zero. A point outside every tet is not bound
    at all in that sense - bind_points_iter clips its coordinates onto the
    nearest tet - and the slip is how far away that tet was.

    Which makes this the honest measure of geometry the cage failed to
    cover. A surface vertex sitting just proud of the voxel hull slips a
    fraction of a Resolution and deforms fine. A tentacle tip thinner than
    Resolution gets no cells at all, and every vertex out there slips the
    whole distance back to where the cage stopped - then rides that one
    distant tet, which is what smears a tip into a spike.
    """
    return np.linalg.norm(points - deform(nodes, tets, bind_idx, bind_w), axis=1)
