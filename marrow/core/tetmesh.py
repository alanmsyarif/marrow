"""Tetrahedral mesh container and orientation invariants."""

from dataclasses import dataclass

import numpy as np


def signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of every tet. Positive means correct winding."""
    p0 = nodes[tets[:, 0]]
    e1 = nodes[tets[:, 1]] - p0
    e2 = nodes[tets[:, 2]] - p0
    e3 = nodes[tets[:, 3]] - p0
    return np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0


def repair_orientation(tets: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Swap the last two indices of any tet with negative signed volume."""
    fixed = np.array(tets, dtype=np.int32, copy=True)
    negative = signed_volumes(nodes, fixed) < 0.0
    fixed[negative, 2], fixed[negative, 3] = fixed[negative, 3], fixed[negative, 2].copy()
    return fixed


_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.intp)


def surface_nodes(tets: np.ndarray) -> np.ndarray:
    """Cage nodes on the boundary, ascending.

    A tet face owned by exactly one tet is on the boundary; a face between two
    tets is named by both. Sorting each face's three indices makes the two
    namings of a shared face identical, so counting duplicates finds the hull.
    """
    tets = np.asarray(tets)
    if tets.size == 0:
        return np.zeros(0, dtype=np.int32)
    faces = np.sort(tets[:, _TET_FACES].reshape(-1, 3), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return np.unique(unique[counts == 1]).astype(np.int32)


@dataclass(frozen=True)
class TetMesh:
    nodes: np.ndarray  # (N, 3) float64
    tets: np.ndarray   # (T, 4) int32

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    def validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must be (N,3), got {self.nodes.shape}")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError(f"tets must be (T,4), got {self.tets.shape}")
        if self.nodes.dtype != np.float64:
            raise ValueError(f"nodes must be float64, got {self.nodes.dtype}")
        if self.tets.dtype != np.int32:
            raise ValueError(f"tets must be int32, got {self.tets.dtype}")
        if self.n_tets and (self.tets.min() < 0 or self.tets.max() >= self.n_nodes):
            raise ValueError("tet node index out of range")
        for row in self.tets:
            if len(set(row.tolist())) != 4:
                raise ValueError(f"tet has a repeated node: {row.tolist()}")
        bad = int(np.count_nonzero(signed_volumes(self.nodes, self.tets) <= 0.0))
        if bad:
            raise ValueError(f"{bad} tets have negative volume or are degenerate")
