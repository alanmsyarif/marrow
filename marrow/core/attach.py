"""Synthesize attachment weights for cage nodes.

Render vertices carry the object's vertex groups, so their evaluated
positions follow the armature. Cage nodes are interior lattice points with
no bone weights of their own; each one borrows motion from its nearest
render vertices instead. The weights are computed once against the rest
shape and reused every frame: only the vertex positions change with the
pose, so the per-frame target is a single sparse blend.
"""

import numpy as np

# Block sizes for the pairwise distance scan. The full (N, V) distance
# matrix of a real body is gigabytes; scanning it node-block by
# vertex-block keeps peak memory in the tens of megabytes.
_NODE_CHUNK = 512
_VERT_CHUNK = 4096

# Closer than this and a cage node is sitting on a render vertex. Inverse
# distance would then divide by ~0 and float32 rounding decides which of
# the "equal" vertices wins; give the nearest one the whole weight instead.
_COINCIDENT = 1e-9


def synth_weights(cage_nodes: np.ndarray, render_verts: np.ndarray, k: int = 4):
    """Inverse-distance weights from every cage node to its k nearest verts.

    Returns ``(idx, w)``: ``idx`` is (N, k) int32 vertex indices and ``w``
    is (N, k) float64, non-negative, rows summing to 1.
    """
    nodes = np.asarray(cage_nodes, dtype=np.float64)
    verts = np.asarray(render_verts, dtype=np.float64)
    k = int(k)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"cage_nodes must be (N, 3), got {nodes.shape}")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"render_verts must be (V, 3), got {verts.shape}")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("cage nodes contain NaN or inf")
    if not np.all(np.isfinite(verts)):
        raise ValueError("render vertices contain NaN or inf")
    if verts.shape[0] < k:
        raise ValueError(
            f"attachment needs at least k={k} render vertices, got {verts.shape[0]}"
        )

    n_nodes = nodes.shape[0]
    n_verts = verts.shape[0]
    best_d = np.full((n_nodes, k), np.inf, dtype=np.float64)
    best_i = np.zeros((n_nodes, k), dtype=np.int64)

    for a in range(0, n_nodes, _NODE_CHUNK):
        b = min(a + _NODE_CHUNK, n_nodes)
        chunk_d = np.full((b - a, k), np.inf, dtype=np.float64)
        chunk_i = np.zeros((b - a, k), dtype=np.int64)
        for va in range(0, n_verts, _VERT_CHUNK):
            vb = min(va + _VERT_CHUNK, n_verts)
            d = np.linalg.norm(
                nodes[a:b, None, :] - verts[None, va:vb, :], axis=2
            )
            cand_d = np.concatenate([chunk_d, d], axis=1)
            cand_i = np.concatenate(
                [chunk_i, np.broadcast_to(np.arange(va, vb), (b - a, vb - va))],
                axis=1,
            )
            take = np.argpartition(cand_d, k - 1, axis=1)[:, :k]
            rows = np.arange(b - a)[:, None]
            chunk_d = cand_d[rows, take]
            chunk_i = cand_i[rows, take]
        best_d[a:b] = chunk_d
        best_i[a:b] = chunk_i

    w = np.zeros((n_nodes, k), dtype=np.float64)
    near = best_d < _COINCIDENT
    near_rows = np.any(near, axis=1)
    if near_rows.any():
        # Sitting on a vertex: that vertex takes the entire weight.
        pick = np.argmin(np.where(near, best_d, np.inf), axis=1)
        w[near_rows, pick[near_rows]] = 1.0
    free = ~near_rows
    inv = 1.0 / best_d[free]
    w[free] = inv / inv.sum(axis=1, keepdims=True)

    return best_i.astype(np.int32), w


def targets_from(idx: np.ndarray, w: np.ndarray, evaluated_verts: np.ndarray) -> np.ndarray:
    """Blend evaluated vertex positions into per-cage-node targets."""
    idx = np.asarray(idx, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    verts = np.asarray(evaluated_verts, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"evaluated_verts must be (V, 3), got {verts.shape}")
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if int(idx.max()) >= verts.shape[0]:
        raise ValueError(
            f"attachment weights index vertex {int(idx.max())} but the "
            f"evaluated mesh has only {verts.shape[0]} vertices"
        )
    return np.einsum("nk,nkj->nj", w, verts[idx])


def blend_scalar(idx: np.ndarray, w: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Blend a per-render-vertex scalar into one value per cage node.

    The scalar twin of ``targets_from``, and the whole of how a painted
    vertex group reaches the cage: pin weights are authored on the render
    mesh, where they can be seen and painted, and each cage node reads the
    weighted mean of the same handful of vertices its motion already comes
    from. Rows sum to 1, so a uniformly painted mesh blends to that value
    everywhere rather than fading off into the interior.
    """
    idx = np.asarray(idx, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values must be (V,), got {values.shape}")
    if idx.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if int(idx.max()) >= values.shape[0]:
        raise ValueError(
            f"weights index vertex {int(idx.max())} but the mesh has only "
            f"{values.shape[0]} vertices"
        )
    return np.einsum("nk,nk->n", w, values[idx])
