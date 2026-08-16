"""Greedy colouring so tets in a colour never share a node.

This is what lets the GPU solve kernel do plain read-modify-write on node
positions with no atomics: within one dispatch, no two threads touch the
same node.
"""

import numpy as np

from .progress import drain


def color_sets(sets, n_nodes: int) -> np.ndarray:
    """Assign each row a colour such that a colour's rows are node-disjoint.

    Rows are variable-length node index sequences; negative entries (the
    padding blend rows use for unused master slots) are ignored.
    """
    return drain(color_sets_iter(sets, n_nodes))


def color_sets_iter(sets, n_nodes: int, block: int = 20_000):
    """color_sets as a generator, yielding 0..1 every ``block`` rows.

    Greedy colouring is order dependent - a row takes the lowest colour its
    nodes have not already claimed - so the chunking must not disturb the
    order rows are visited in. It does not: the loop is unchanged and the
    yield is only a pause inside it. Chunk size therefore cannot change the
    result, which is what the tests assert.
    """
    rows = list(sets)
    total = len(rows)
    colors = np.full(total, -1, dtype=np.int32)

    if total == 0:
        yield 1.0
        return colors

    # node_color_used[node] is the set of colours already claimed at that node.
    node_colors: list[set[int]] = [set() for _ in range(int(n_nodes))]

    for r, row in enumerate(rows):
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

        if (r + 1) % block == 0 and (r + 1) < total:
            yield (r + 1) / total

    yield 1.0
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
