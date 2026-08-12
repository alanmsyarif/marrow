"""Texel layout for the GPU solver. Pure numpy, no Blender.

Everything here exists because Blender's Python GPU API has no storage
buffers, so all bulk state rides in 2D images. A 1D layout would cap at the
32768 max texture size; 2D indexing removes the ceiling.

Integer data rides in float32 images: GPUTexture refuses any upload Buffer
that is not FLOAT, so RGBA32I is unreachable from Python. float32 represents
every integer up to 2**24 exactly, which is about a thousand times the node
budget, so tet indices are stored as floats and read back with ivec4().
"""

import numpy as np

TEX_WIDTH = 4096


def texture_shape(count: int) -> tuple[int, int]:
    """(width, height) of an image holding ``count`` texels."""
    rows = max(1, -(-int(count) // TEX_WIDTH))  # ceil division
    return (TEX_WIDTH, rows)


def texel_index(i: int) -> tuple[int, int]:
    """(x, y) of element ``i``. Mirrors the shader's texel() exactly."""
    i = int(i)
    return (i % TEX_WIDTH, i // TEX_WIDTH)


def color_order(colors: np.ndarray) -> np.ndarray:
    """The permutation that sorts tets into contiguous colour slices.

    Split out so that a caller mapping a colour-ordered result back to mesh tet
    order uses the same sort color_ordered did, rather than a second copy of it
    that can drift.
    """
    return np.argsort(np.asarray(colors, dtype=np.int32), kind="stable")


def color_ordered(tets: np.ndarray, colors: np.ndarray):
    """Permute tets so each colour is one contiguous slice.

    The solve kernel dispatches once per colour over a half-open range, so a
    colour must be a slice, not a scatter list. Returns the permuted tets and
    an offsets array where colour c is ``[offsets[c], offsets[c + 1])``.
    """
    colors = np.asarray(colors, dtype=np.int32)
    if colors.size == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        )

    order = color_order(colors)
    ordered = np.asarray(tets, dtype=np.int32)[order]
    counts = np.bincount(colors, minlength=int(colors.max()) + 1)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return ordered, offsets


def _blank(count: int) -> np.ndarray:
    width, height = texture_shape(count)
    return np.zeros((height, width, 4), dtype=np.float32)


def _write(image: np.ndarray, values: np.ndarray) -> None:
    """Fill the first len(values) texels of image, row-major."""
    flat = image.reshape(-1, 4)
    flat[: values.shape[0]] = values


def pack_nodes(nodes: np.ndarray, inv_mass: np.ndarray) -> np.ndarray:
    """Positions in rgb, inverse mass in a. Zero mass means pinned."""
    nodes = np.asarray(nodes, dtype=np.float64)
    inv_mass = np.asarray(inv_mass, dtype=np.float64)
    image = _blank(nodes.shape[0])
    values = np.concatenate([nodes, inv_mass[:, None]], axis=1)
    _write(image, values.astype(np.float32))
    return image


def pack_scalar(values: np.ndarray) -> np.ndarray:
    """One float per element, for a single-channel R32F image."""
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[0]
    width, height = texture_shape(count)
    image = np.zeros((height, width, 1), dtype=np.float32)
    image.reshape(-1, 1)[:count, 0] = values.astype(np.float32)
    return image


def pack_tets(tets: np.ndarray) -> np.ndarray:
    """Four node indices per texel, as floats. Exact below 2**24."""
    tets = np.asarray(tets, dtype=np.int64)
    image = _blank(tets.shape[0])
    _write(image, tets.astype(np.float32))
    return image


def pack_rest(dm_inv: np.ndarray, rest_vol: np.ndarray) -> np.ndarray:
    """Three texels per tet: texel 3t+j is column j of dm_inv[t].

    Column-major on purpose. GLSL's mat3(c0, c1, c2) takes columns, so the
    shader can rebuild the matrix with no transpose and F = Ds * DmInv then
    matches numpy's ds @ dm_inv term for term.
    """
    dm_inv = np.asarray(dm_inv, dtype=np.float64)
    rest_vol = np.asarray(rest_vol, dtype=np.float64)
    n_tets = dm_inv.shape[0]
    image = _blank(3 * n_tets)

    values = np.zeros((3 * n_tets, 4), dtype=np.float64)
    for j in range(3):
        values[j::3, :3] = dm_inv[:, :, j]
    values[0::3, 3] = rest_vol
    _write(image, values.astype(np.float32))
    return image


def unpack_vec3(image: np.ndarray, count: int) -> np.ndarray:
    """First ``count`` texels of an image as (count, 3) float64."""
    flat = np.asarray(image).reshape(-1, 4)
    if count > flat.shape[0]:
        raise ValueError(
            f"image cannot hold {count} texels, it has {flat.shape[0]}"
        )
    return flat[:count, :3].astype(np.float64)
