"""Per-tet deformation metrics for the false-color display.

One scalar per tet, neutral at rest and invariant to rotation and translation,
so it reads the same whichever space the cage is tracked in:

- stretch: the largest current/rest edge-length ratio over the six edges.
  1.0 at rest, 1.2 where something is pulled to 1.2x its rest length.

Pure numpy on purpose: the values are computed on the CPU from the cached
cage positions, which keeps them available for cached frames after a mode
change and testable without a GPU. The Blender side maps tet values onto
render vertices through the bind and writes them as a point attribute.
"""

import numpy as np

_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _gather(points: np.ndarray, tets) -> np.ndarray:
    return np.asarray(points)[np.asarray(tets, dtype=np.int64)]


def tet_stretch(nodes, tets, rest) -> np.ndarray:
    """Largest edge ratio per tet. 1.0 at rest, >1 pulled, <1 compressed."""
    cur = _gather(nodes, tets)
    rst = _gather(rest, tets)
    ratios = np.stack([
        np.linalg.norm(cur[:, a] - cur[:, b], axis=1)
        / np.linalg.norm(rst[:, a] - rst[:, b], axis=1)
        for a, b in _EDGES
    ])
    return ratios.max(axis=0)


def vertex_values(tet_values, bind_idx) -> np.ndarray:
    """The owning tet's value at every render vertex."""
    return np.asarray(tet_values)[np.asarray(bind_idx, dtype=np.int64)]
