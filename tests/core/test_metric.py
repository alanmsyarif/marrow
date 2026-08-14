"""False-color metric: neutral at rest, sane under known deformations.

The metric is rotation- and translation-invariant, because the session
compares a world-space cached cage against a world-space rest cage and the
object transform may have moved since.
"""

import numpy as np

from marrow.core.metric import tet_stretch, vertex_values

REST = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
TETS = np.array([[0, 1, 2, 3]], dtype=np.int32)


def test_rest_shape_is_neutral():
    assert np.allclose(tet_stretch(REST, TETS, REST), 1.0)


def test_uniform_scale_reads_the_scale():
    nodes = REST * 1.2
    assert np.allclose(tet_stretch(nodes, TETS, REST), 1.2)


def test_compression_shows_in_stretch():
    nodes = REST.copy()
    nodes[:, 0] *= 0.5
    # The x edges halve, but the slant edges shrink less; the max ratio is
    # the untouched unit edge, while some edge must read 0.5.
    stretch = tet_stretch(nodes, TETS, REST)
    assert stretch[0] > 0.5


def test_rigid_rotation_and_translation_change_nothing():
    theta = np.radians(30.0)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    nodes = REST @ rot.T + np.array([4.0, -2.0, 9.0])
    assert np.allclose(tet_stretch(nodes, TETS, REST), 1.0)


def test_vertex_values_take_the_owning_tet():
    values = np.array([0.25, 0.75])
    bind = np.array([0, 1, 1, 0], dtype=np.int64)
    assert np.allclose(vertex_values(values, bind), [0.25, 0.75, 0.75, 0.25])
