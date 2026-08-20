"""Persist tet cage and bind data on Blender datablocks.

Addons cannot register new datablock types, so cage nodes live as mesh
vertices, connectivity and colours live in ID properties, and per-point bind
data lives in POINT-domain attributes. All of it survives save and load.
"""

import numpy as np

from ..core.tetmesh import TetMesh

TETS_KEY = "marrow_tets"
COLORS_KEY = "marrow_colors"
# Adaptive hanging-node glue rows, stored on the CAGE mesh like the tets:
# indices flat (R*5 ints, [hanging, m0..m3] per row) and weights flat
# (R*4 floats). Absent on a uniform cage, which has no hanging nodes.
BLEND_KEY = "marrow_blend"
BLEND_W_KEY = "marrow_blend_w"
# Per-tet fiber rows, stored on the CAGE mesh like the tets: T*5 floats,
# (dir x, dir y, dir z, arclength, side). An ID property rather than an
# attribute because the cage mesh has one vertex per NODE, and there is no
# per-tet domain to hang this on. Absent on a body with no fiber curve,
# which the solver takes as "no fiber pass".
#
# The "2" generation carries the side column, which is what lets the wave
# bend the body rather than only squeeze it. A generation-1 blob is four
# wide, and four and five wide are not distinguishable from the length
# alone, so old caches must be ignored rather than guessed at - hence the
# new name, the same rule ATTACH_IDX follows. A body with only the old key
# reads as unfibered and the panel asks for a Tetrahedralize, which is
# needed anyway to compute a side.
FIBER_KEY = "marrow_fiber2"
_LEGACY_FIBER = "marrow_fiber"
BIND_IDX = "marrow_bind_idx"
BIND_W = ("marrow_bind_w0", "marrow_bind_w1", "marrow_bind_w2", "marrow_bind_w3")
REST_KEY = "marrow_rest"
# Cage-node attachment weights, stored on the CAGE mesh: k nearest render
# vertex indices and their weights per node. The "2" generation was
# synthesized in object space; generation 1 measured world-space vertices
# against bind-space nodes, so any object transform change after
# Tetrahedralize scrambled the correspondence. Old caches must be ignored,
# hence the new names.
ATTACH_IDX = "marrow_attach_idx2"
ATTACH_K = 4
ATTACH_W = tuple(f"marrow_attach_w2_{i}" for i in range(ATTACH_K))
_LEGACY_ATTACH = (
    "marrow_attach_idx",
    "marrow_attach_idx_1",
    "marrow_attach_idx_2",
    "marrow_attach_idx_3",
    "marrow_attach_w0",
    "marrow_attach_w1",
    "marrow_attach_w2",
    "marrow_attach_w3",
)


def _attach_idx_names():
    """The k index attributes, in column order."""
    return (ATTACH_IDX,) + tuple(
        f"{ATTACH_IDX}_{i}" for i in range(1, ATTACH_K)
    )


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


def write_blend(mesh, blend_idx: np.ndarray, blend_w: np.ndarray) -> None:
    mesh[BLEND_KEY] = np.asarray(blend_idx, dtype=np.int32).ravel().tolist()
    mesh[BLEND_W_KEY] = np.asarray(blend_w, dtype=np.float32).ravel().tolist()


def read_blend(mesh):
    """Stored glue rows as ``(blend_idx, blend_w)``, or None on a uniform
    cage. The solver takes None as "no blend pass", bit-identical to before."""
    if BLEND_KEY not in mesh.keys() or BLEND_W_KEY not in mesh.keys():
        return None
    flat = np.array(mesh[BLEND_KEY], dtype=np.int32)
    idx = flat.reshape(-1, 5) if flat.size else np.zeros((0, 5), dtype=np.int32)
    weights = np.array(mesh[BLEND_W_KEY], dtype=np.float32).astype(np.float64)
    w = weights.reshape(-1, 4) if weights.size else np.zeros((0, 4), dtype=np.float64)
    return idx, w


def write_fiber(mesh, fiber: np.ndarray) -> None:
    fiber = np.asarray(fiber, dtype=np.float32)
    if fiber.ndim != 2 or fiber.shape[1] != 5:
        raise ValueError(f"fiber rows must be (T, 5), got {fiber.shape}")
    mesh[FIBER_KEY] = fiber.ravel().tolist()


def read_fiber(mesh):
    """Stored fiber rows as (T, 5), or None if this cage has none."""
    if FIBER_KEY not in mesh.keys():
        return None
    flat = np.array(mesh[FIBER_KEY], dtype=np.float32).astype(np.float64)
    if flat.size == 0 or flat.size % 5:
        return None
    return flat.reshape(-1, 5)


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
    for name in (
        (REST_KEY, BIND_IDX)
        + BIND_W
        + _attach_idx_names()
        + ATTACH_W
        + _LEGACY_ATTACH
    ):
        attr = mesh.attributes.get(name)
        if attr is not None:
            mesh.attributes.remove(attr)
    for key in (BLEND_KEY, BLEND_W_KEY, FIBER_KEY, _LEGACY_FIBER):
        if key in mesh.keys():
            del mesh[key]
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


def write_attach(mesh, attach_idx: np.ndarray, attach_w: np.ndarray) -> None:
    """Persist cage-node attachment weights on the cage mesh.

    Indices ride in k INT attributes rather than one, mirroring how bind
    weights are stored: a POINT attribute holds one value per element.
    """
    attach_idx = np.asarray(attach_idx, dtype=np.int32)
    attach_w = np.asarray(attach_w, dtype=np.float32)
    if attach_idx.shape[1] != ATTACH_K or attach_w.shape[1] != ATTACH_K:
        raise ValueError(
            f"attachment data must have {ATTACH_K} columns, got "
            f"{attach_idx.shape[1]} and {attach_w.shape[1]}"
        )
    _ensure_attr(mesh, ATTACH_IDX, "INT").data.foreach_set(
        "value", attach_idx[:, 0].tolist()
    )
    for i, name in enumerate(_attach_idx_names()[1:], start=1):
        _ensure_attr(mesh, name, "INT").data.foreach_set(
            "value", attach_idx[:, i].tolist()
        )
    for i, name in enumerate(ATTACH_W):
        _ensure_attr(mesh, name, "FLOAT").data.foreach_set(
            "value", attach_w[:, i].tolist()
        )
    mesh.update()


def read_attach(mesh):
    """Stored attachment weights as ``(idx, w)``, or None if absent.

    None means "never computed or removed"; the caller synthesizes and
    writes them back. A partial set (any attribute missing) is treated as
    absent rather than trusted.
    """
    names = _attach_idx_names() + ATTACH_W
    if any(mesh.attributes.get(name) is None for name in names):
        return None
    n = len(mesh.vertices)
    idx = np.empty((n, ATTACH_K), dtype=np.int32)
    for i, name in enumerate(names[:ATTACH_K]):
        column = np.empty(n, dtype=np.int32)
        mesh.attributes[name].data.foreach_get("value", column)
        idx[:, i] = column
    weights = np.empty((n, ATTACH_K), dtype=np.float64)
    for i, name in enumerate(ATTACH_W):
        column = np.empty(n, dtype=np.float32)
        mesh.attributes[name].data.foreach_get("value", column)
        weights[:, i] = column
    return idx, weights
