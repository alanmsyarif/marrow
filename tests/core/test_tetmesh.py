import numpy as np
import pytest

from marrow.core.tetmesh import TetMesh, repair_orientation, signed_volumes

# Unit tet: volume 1/6
UNIT_NODES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
UNIT_TET = np.array([[0, 1, 2, 3]], dtype=np.int32)


def test_signed_volume_of_unit_tet():
    vols = signed_volumes(UNIT_NODES, UNIT_TET)
    assert vols.shape == (1,)
    assert np.isclose(vols[0], 1.0 / 6.0)


def test_signed_volume_is_negative_when_wound_backwards():
    flipped = np.array([[0, 1, 3, 2]], dtype=np.int32)
    assert signed_volumes(UNIT_NODES, flipped)[0] < 0


def test_repair_orientation_makes_all_volumes_positive():
    flipped = np.array([[0, 1, 3, 2]], dtype=np.int32)
    fixed = repair_orientation(flipped, UNIT_NODES)
    assert np.all(signed_volumes(UNIT_NODES, fixed) > 0)


def test_repair_orientation_leaves_good_tets_untouched():
    fixed = repair_orientation(UNIT_TET, UNIT_NODES)
    assert np.array_equal(fixed, UNIT_TET)


def test_validate_rejects_negative_volume():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 3, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="negative volume"):
        mesh.validate()


def test_validate_rejects_out_of_range_index():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 2, 9]], dtype=np.int32))
    with pytest.raises(ValueError, match="out of range"):
        mesh.validate()


def test_validate_rejects_duplicate_node_in_tet():
    mesh = TetMesh(UNIT_NODES, np.array([[0, 1, 1, 3]], dtype=np.int32))
    with pytest.raises(ValueError, match="repeated node"):
        mesh.validate()


def test_validate_rejects_float32_nodes():
    nodes_f32 = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32
    )
    mesh = TetMesh(nodes_f32, UNIT_TET)
    with pytest.raises(ValueError, match="nodes must be float64"):
        mesh.validate()


def test_validate_rejects_int64_tets():
    tets_i64 = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = TetMesh(UNIT_NODES, tets_i64)
    with pytest.raises(ValueError, match="tets must be int32"):
        mesh.validate()


def test_validate_accepts_good_mesh():
    TetMesh(UNIT_NODES, UNIT_TET).validate()
