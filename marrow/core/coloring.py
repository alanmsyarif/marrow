"""Greedy colouring so tets in a colour never share a node.

This is what lets the GPU solve kernel do plain read-modify-write on node
positions with no atomics: within one dispatch, no two threads touch the
same node.
"""

import numpy as np


def color_sets(sets, n_nodes: int) -> np.ndarray:
    """Assign each row a colour such that a colour's rows are node-disjoint.

    Rows are variable-length node index sequences; negative entries (the
    padding blend rows use for unused master slots) are ignored.
    """
    colors = np.full(len(list(sets)), -1, dtype=np.int32)

    # node_color_used[node] is the set of colours already claimed at that node.
    node_colors: list[set[int]] = [set() for _ in range(int(n_nodes))]

    for r, row in enumerate(sets):
        taken = set()
        for n in row:
            if n >= 0:
                taken |= node_colors[int(n)]
        c = 0
        while c in taken:
            c += 1
        colors[r] = c
        for n in row:
            if n >= 0:
                node_colors[int(n)].add(c)

    return colors


def color_tets(tets: np.ndarray, n_nodes: int) -> np.ndarray:
    """Assign each tet a colour such that a colour's tets are node-disjoint."""
    return color_sets(tets, n_nodes)


def color_groups(colors: np.ndarray) -> list[np.ndarray]:
    """Split tet indices into one int32 array per colour, in colour order."""
    if colors.size == 0:
        return []
    return [
        np.flatnonzero(colors == c).astype(np.int32)
        for c in range(int(colors.max()) + 1)
    ]
