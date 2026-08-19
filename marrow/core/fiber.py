"""Fiber directions and wave phase, sampled from a polyline. Pure numpy.

The fiber constraint needs two things per tet: a direction to contract
along, and a scalar that says where along the creature this tet sits so the
wave can reach it at the right moment. A curve gives both from one sample -
the tangent at the nearest point, and the arclength at that point - which is
why the direction is not simply painted.

Directions are rest-space, because the constraint measures F a and F maps
rest to world. Callers sample against the cage's rest nodes, once.
"""

import numpy as np

# Below this a segment carries no direction: normalizing it would divide by
# roughly zero and hand the solver a NaN that spreads through the whole cage.
_MIN_SEGMENT = 1e-12


def tet_centroids(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The mean of each tet's four nodes, (T, 3)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return nodes[tets].mean(axis=1)


def fiber_from_polyline(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Per-tet (direction, arclength) from the nearest point on a polyline.

    Returns (T, 4): xyz is the unit tangent of the nearest segment, w is the
    arclength from the start of the polyline to the nearest point. A row of
    zeros means no fiber could be assigned, which every consumer reads as
    "skip this tet" rather than as a direction.
    """
    points = np.asarray(points, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.zeros((centroids.shape[0], 4), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or centroids.shape[0] == 0:
        return out

    starts = points[:-1]
    deltas = points[1:] - starts
    lengths = np.linalg.norm(deltas, axis=1)

    # Zero-length segments are dropped rather than repaired. A curve
    # evaluated with duplicate control points is common and harmless; what
    # is not harmless is letting one become the nearest "segment" and
    # normalizing it.
    keep = lengths > _MIN_SEGMENT
    if not np.any(keep):
        return out

    # Arclength is measured along the WHOLE polyline, including the segments
    # dropped above, so phase stays continuous across a duplicate point.
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]

    starts, deltas = starts[keep], deltas[keep]
    lengths, cumulative = lengths[keep], cumulative[keep]
    tangents = deltas / lengths[:, None]

    # Closest approach of every centroid to every segment, as a running
    # minimum over segments rather than a dense (T, S, 3) stack. The dense
    # form peaks at ~96 bytes per (tet, segment) pair - 118k tets against a
    # 300-point spine is 3.4GB - and the loop is over S, the small axis, so
    # the vectorised work per pass is still the whole tet array.
    #
    # Strictly `<` against the running best, never `<=`: that reproduces
    # np.argmin's lowest-index tie-break, which is what keeps the segment
    # either side of a duplicate control point from swapping the winner.
    best = np.full(centroids.shape[0], np.inf)
    pick = np.zeros(centroids.shape[0], dtype=np.int64)
    t_best = np.zeros(centroids.shape[0])
    for s in range(starts.shape[0]):
        # einsum, not `@` or (a * b).sum(): only einsum sums the three
        # components in the order the dense form did, and only then is the
        # result bit-for-bit what this function returned before.
        proj = np.einsum("tc,c->t", centroids - starts[s], deltas[s])
        t = np.clip(proj / (lengths[s] * lengths[s]), 0.0, 1.0)
        d = np.linalg.norm(centroids - (starts[s] + t[:, None] * deltas[s]), axis=1)
        m = d < best
        best[m], pick[m], t_best[m] = d[m], s, t[m]

    out[:, :3] = tangents[pick]
    out[:, 3] = cumulative[pick] + t_best * lengths[pick]
    return out
