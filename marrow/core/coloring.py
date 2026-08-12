"""Greedy colouring so tets in a colour never share a node.

This is what lets the GPU solve kernel do plain read-modify-write on node
positions with no atomics: within one dispatch, no two threads touch the
same node.
"""

import numpy as np


def color_tets(tets: np.ndarray, n_nodes: int) -> np.ndarray:
    """Assign each tet a colour such that a colour's tets are node-disjoint."""
    n_tets = int(tets.shape[0])
    colors = np.full(n_tets, -1, dtype=np.int32)
    if n_tets == 0:
        return colors

    # node_color_used[node] is the set of colours already claimed at that node.
    node_colors: list[set[int]] = [set() for _ in range(int(n_nodes))]

    for t in range(n_tets):
        nodes = tets[t]
        taken = set()
        for n in nodes:
            taken |= node_colors[int(n)]
        c = 0
        while c in taken:
            c += 1
        colors[t] = c
        for n in nodes:
            node_colors[int(n)].add(c)

    return colors


def color_groups(colors: np.ndarray) -> list[np.ndarray]:
    """Split tet indices into one int32 array per colour, in colour order."""
    if colors.size == 0:
        return []
    return [
        np.flatnonzero(colors == c).astype(np.int32)
        for c in range(int(colors.max()) + 1)
    ]
