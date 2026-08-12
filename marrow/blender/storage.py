"""Persist tet cage and bind data on Blender datablocks.

Addons cannot register new datablock types, so cage nodes live as mesh
vertices, connectivity and colours live in ID properties, and per-point bind
data lives in POINT-domain attributes. All of it survives save and load.
"""

import numpy as np

from ..core.tetmesh import TetMesh

TETS_KEY = "marrow_tets"
COLORS_KEY = "marrow_colors"
BIND_IDX = "marrow_bind_idx"
BIND_W = ("marrow_bind_w0", "marrow_bind_w1", "marrow_bind_w2", "marrow_bind_w3")


def write_tetmesh(mesh, tetmesh: TetMesh, colors: np.ndarray) -> None:
    mesh.clear_geometry()
    mesh.from_pydata([tuple(p) for p in tetmesh.nodes], [], [])
    mesh.update()
    mesh[TETS_KEY] = tetmesh.tets.ravel().astype(np.int32).tolist()
    mesh[COLORS_KEY] = np.asarray(colors, dtype=np.int32).tolist()


def read_tetmesh(mesh):
    if TETS_KEY not in mesh.keys():
        raise KeyError(f"mesh has no {TETS_KEY}; it is not a Marrow cage")

    n_nodes = len(mesh.vertices)
    nodes = np.empty(n_nodes * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", nodes)
    nodes = nodes.reshape(n_nodes, 3)

    flat = np.array(mesh[TETS_KEY], dtype=np.int32)
    tets = flat.reshape(-1, 4) if flat.size else np.zeros((0, 4), dtype=np.int32)
    colors = np.array(mesh.get(COLORS_KEY, []), dtype=np.int32)
    return TetMesh(nodes, tets), colors


def _ensure_attr(mesh, name, data_type):
    attr = mesh.attributes.get(name)
    if attr is None or attr.data_type != data_type or attr.domain != "POINT":
        if attr is not None:
            mesh.attributes.remove(attr)
        attr = mesh.attributes.new(name=name, type=data_type, domain="POINT")
    return attr


def write_bind(mesh, bind_idx: np.ndarray, bind_w: np.ndarray) -> None:
    _ensure_attr(mesh, BIND_IDX, "INT").data.foreach_set(
        "value", np.asarray(bind_idx, dtype=np.int32).tolist()
    )
    for i, name in enumerate(BIND_W):
        _ensure_attr(mesh, name, "FLOAT").data.foreach_set(
            "value", np.asarray(bind_w[:, i], dtype=np.float32).tolist()
        )
    mesh.update()


def read_bind(mesh):
    n = len(mesh.vertices)
    idx = np.empty(n, dtype=np.int32)
    mesh.attributes[BIND_IDX].data.foreach_get("value", idx)

    weights = np.empty((n, 4), dtype=np.float64)
    for i, name in enumerate(BIND_W):
        column = np.empty(n, dtype=np.float32)
        mesh.attributes[name].data.foreach_get("value", column)
        weights[:, i] = column
    return idx, weights
