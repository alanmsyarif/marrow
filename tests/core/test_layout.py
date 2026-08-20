import numpy as np
import pytest

from marrow.core.coloring import color_tets
from marrow.core.lattice import build_lattice
from marrow.core.layout import (
    TEX_WIDTH,
    color_ordered,
    pack_nodes,
    pack_rest,
    pack_tets,
    texel_index,
    texture_shape,
    unpack_vec3,
)
from marrow.core.solver_ref import precompute

MESH = build_lattice(np.zeros(3), 0.5, np.ones((2, 2, 2), dtype=bool))


def test_texture_width_is_the_documented_constant():
    assert TEX_WIDTH == 4096


def test_texture_shape_is_one_row_when_it_fits():
    assert texture_shape(10) == (TEX_WIDTH, 1)


def test_texture_shape_rounds_up_to_whole_rows():
    assert texture_shape(TEX_WIDTH + 1) == (TEX_WIDTH, 2)
    assert texture_shape(TEX_WIDTH * 3) == (TEX_WIDTH, 3)


def test_texture_shape_of_zero_still_has_one_row():
    """A zero-height texture cannot be allocated."""
    assert texture_shape(0) == (TEX_WIDTH, 1)


def test_texel_index_wraps_at_the_row_width():
    assert texel_index(0) == (0, 0)
    assert texel_index(TEX_WIDTH - 1) == (TEX_WIDTH - 1, 0)
    assert texel_index(TEX_WIDTH) == (0, 1)
    assert texel_index(TEX_WIDTH + 5) == (5, 1)


def test_color_ordered_groups_every_color_contiguously():
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, offsets = color_ordered(MESH.tets, colors)

    assert ordered.shape == MESH.tets.shape
    assert offsets.shape == (int(colors.max()) + 2,)
    assert offsets[0] == 0
    assert offsets[-1] == MESH.n_tets
    assert np.all(np.diff(offsets) >= 0)


def test_color_ordered_slices_are_node_disjoint():
    """The whole point: one colour's slice can be dispatched race-free."""
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, offsets = color_ordered(MESH.tets, colors)

    for c in range(len(offsets) - 1):
        seen = set()
        for tet in ordered[offsets[c]:offsets[c + 1]]:
            nodes = set(tet.tolist())
            assert not (seen & nodes), f"colour {c} slice shares a node"
            seen |= nodes


def test_color_ordered_is_a_permutation_not_a_rewrite():
    colors = color_tets(MESH.tets, MESH.n_nodes)
    ordered, _ = color_ordered(MESH.tets, colors)
    before = sorted(tuple(sorted(t)) for t in MESH.tets.tolist())
    after = sorted(tuple(sorted(t)) for t in ordered.tolist())
    assert before == after


def test_pack_nodes_carries_position_and_inverse_mass():
    inv_mass = np.arange(MESH.n_nodes, dtype=np.float64)
    img = pack_nodes(MESH.nodes, inv_mass)

    assert img.dtype == np.float32
    assert img.shape == (1, TEX_WIDTH, 4)
    x, y = texel_index(7)
    assert np.allclose(img[y, x, :3], MESH.nodes[7], atol=1e-6)
    assert np.isclose(img[y, x, 3], inv_mass[7])


def test_pack_tets_indices_survive_the_float_round_trip():
    img = pack_tets(MESH.tets)
    assert img.dtype == np.float32
    for t in (0, 5, MESH.n_tets - 1):
        x, y = texel_index(t)
        assert img[y, x].astype(np.int64).tolist() == MESH.tets[t].tolist()


def test_pack_tets_is_exact_at_the_float32_integer_limit():
    """float32 holds every integer to 2**24; the cage budget is far below."""
    big = np.array([[0, 1, 2, 2**24 - 1]], dtype=np.int32)
    img = pack_tets(big)
    x, y = texel_index(0)
    assert img[y, x].astype(np.int64).tolist() == [0, 1, 2, 2**24 - 1]


def test_pack_rest_uses_three_texels_per_tet():
    dm_inv, rest_vol = precompute(MESH.nodes, MESH.tets)
    img = pack_rest(dm_inv, rest_vol)

    t = 4
    cols = []
    for j in range(3):
        x, y = texel_index(3 * t + j)
        cols.append(img[y, x, :3])
    assert np.allclose(np.stack(cols, axis=1), dm_inv[t], atol=1e-5)

    x, y = texel_index(3 * t)
    assert np.isclose(img[y, x, 3], rest_vol[t], atol=1e-6)


def test_unpack_vec3_inverts_pack_nodes():
    inv_mass = np.ones(MESH.n_nodes)
    img = pack_nodes(MESH.nodes, inv_mass)
    out = unpack_vec3(img, MESH.n_nodes)
    assert out.shape == (MESH.n_nodes, 3)
    assert out.dtype == np.float64
    assert np.allclose(out, MESH.nodes, atol=1e-6)


def test_unpack_vec3_rejects_a_count_the_image_cannot_hold():
    img = pack_nodes(MESH.nodes, np.ones(MESH.n_nodes))
    with pytest.raises(ValueError, match="cannot hold"):
        unpack_vec3(img, TEX_WIDTH * 99)


def test_pack_fiber_is_two_texels_per_tet():
    from marrow.core.layout import pack_fiber

    fiber = np.array(
        [[1.0, 0.0, 0.0, 0.25, -1.0], [0.0, 1.0, 0.0, 1.75, 0.5]]
    )
    image = pack_fiber(fiber)
    flat = image.reshape(-1, 4)
    assert image.dtype == np.float32
    assert np.allclose(flat[0], [1.0, 0.0, 0.0, 0.25])
    assert flat[1][0] == -1.0, "side rides the second texel of the pair"
    assert np.allclose(flat[2], [0.0, 1.0, 0.0, 1.75])
    assert flat[3][0] == 0.5
    assert np.allclose(flat[4], 0.0), "unused texels must be zero, which reads as no fiber"


def test_pack_fiber_rejects_the_old_four_wide_rows():
    """Generation 1 rows carry no side column. Packing them would put an
    arclength where the kernel reads a direction."""
    from marrow.core.layout import pack_fiber

    try:
        pack_fiber(np.zeros((3, 4)))
    except ValueError as exc:
        assert "(T, 5)" in str(exc)
    else:
        raise AssertionError("expected a ValueError for four-wide rows")
