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
REST_KEY = "marrow_rest"


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


def write_rest(mesh) -> None:
    """Record the mesh's current vertex positions as its rest shape.

    Simulation writes straight into mesh.vertices, so without this there is
    nothing to go back to: freeing a bake would leave the object deformed,
    and re-tetrahedralizing would make the deformed pose the new rest shape.

    Object space, which is what mesh.vertices holds, so a restore does not
    depend on the object transform having stayed where it was.
    """
    count = len(mesh.vertices)
    positions = np.empty(count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", positions)
    _ensure_attr(mesh, REST_KEY, "FLOAT_VECTOR").data.foreach_set(
        "vector", positions.tolist()
    )
    mesh.update()


def read_rest(mesh):
    """The stored rest positions as (N, 3), or None if there are none."""
    attr = mesh.attributes.get(REST_KEY)
    if attr is None:
        return None
    positions = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    attr.data.foreach_get("vector", positions)
    return positions.reshape(-1, 3)


def restore_rest(mesh) -> bool:
    """Put the mesh back into its rest shape. False if none was stored."""
    positions = read_rest(mesh)
    if positions is None or positions.shape[0] != len(mesh.vertices):
        return False
    mesh.vertices.foreach_set("co", positions.ravel())
    mesh.update()
    return True


def clear_marrow_data(mesh) -> None:
    """Remove every attribute Marrow wrote. Tolerates any being absent."""
    for name in (REST_KEY, BIND_IDX) + BIND_W:
        attr = mesh.attributes.get(name)
        if attr is not None:
            mesh.attributes.remove(attr)
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
