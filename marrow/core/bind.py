"""Bind arbitrary points into a tet cage and deform them with it."""

import numpy as np

from ..core.tetmesh import signed_volumes

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
    """Barycentric coords of one point in every tet. Returns (T, 4)."""
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


def bind_points(nodes: np.ndarray, tets: np.ndarray, points: np.ndarray):
    """Bind each point to a containing tet, falling back to the nearest one."""
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    _check_bindable(nodes, tets, points)

    n_points = points.shape[0]
    idx = np.zeros(n_points, dtype=np.int32)
    weights = np.zeros((n_points, 4), dtype=np.float64)

    centroids = nodes[tets].mean(axis=1)  # (T, 3)

    for i in range(n_points):
        bary = _barycentric_all(nodes, tets, points[i])
        inside = np.all(bary >= -1e-9, axis=1)
        if np.any(inside):
            # Prefer the most interior containment for numerical comfort.
            candidates = np.flatnonzero(inside)
            best = candidates[np.argmax(bary[candidates].min(axis=1))]
            w = bary[best]
        else:
            best = int(np.argmin(np.linalg.norm(centroids - points[i], axis=1)))
            w = np.clip(bary[best], 0.0, None)
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

    return idx, weights


def deform(nodes: np.ndarray, tets: np.ndarray, bind_idx: np.ndarray, bind_w: np.ndarray):
    """Interpolate point positions from the current cage node positions."""
    corners = nodes[tets[bind_idx]]  # (P, 4, 3)
    return np.einsum("pij,pi->pj", corners, bind_w)
