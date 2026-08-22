import numpy as np

from marrow.core.solver_ref import (
    FIELD_FORCE,
    FIELD_VORTEX,
    FIELD_WIND,
    field_accel,
)


def _row(kind, origin=(0, 0, 0), axis=(0, 0, 1), strength=1.0, power=0.0, max_dist=0.0):
    return (float(kind), *origin, *axis, strength, power, max_dist)


def test_no_fields_is_no_acceleration():
    pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.array_equal(field_accel(pts, None), np.zeros_like(pts))
    assert np.array_equal(field_accel(pts, []), np.zeros_like(pts))


def test_force_pushes_along_the_radius():
    """Positive strength pushes away from the field, which is how Blender
    reads it - a repulsor. Negative pulls in."""
    pts = np.array([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    out = field_accel(pts, [_row(FIELD_FORCE, strength=5.0)])
    assert np.allclose(out[0], [5.0, 0.0, 0.0])
    assert np.allclose(out[1], [0.0, -5.0, 0.0])

    pull = field_accel(pts, [_row(FIELD_FORCE, strength=-5.0)])
    assert np.allclose(pull[0], [-5.0, 0.0, 0.0])


def test_a_point_on_the_force_origin_is_left_alone():
    """The radius has no direction there, and normalising it would hand the
    solver a NaN that spreads through the whole cage."""
    out = field_accel(np.zeros((1, 3)), [_row(FIELD_FORCE, strength=9.0)])
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_wind_is_the_same_everywhere_along_its_axis():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, -4.0, 7.0]])
    out = field_accel(pts, [_row(FIELD_WIND, axis=(0, 1, 0), strength=3.0)])
    assert np.allclose(out[0], [0.0, 3.0, 0.0])
    assert np.allclose(out[1], out[0]), "wind must not depend on position"


def test_vortex_turns_around_the_axis():
    """Tangential, so it is perpendicular both to the axis and to the arm
    from it - that is what makes material orbit rather than fly outwards."""
    pts = np.array([[2.0, 0.0, 0.0]])
    out = field_accel(pts, [_row(FIELD_VORTEX, axis=(0, 0, 1), strength=4.0)])
    assert np.allclose(out[0], [0.0, 4.0, 0.0])
    assert abs(np.dot(out[0], [0.0, 0.0, 1.0])) < 1e-12
    assert abs(np.dot(out[0], pts[0])) < 1e-12


def test_a_point_on_the_vortex_axis_is_left_alone():
    out = field_accel(np.array([[0.0, 0.0, 5.0]]),
                      [_row(FIELD_VORTEX, axis=(0, 0, 1), strength=4.0)])
    assert np.all(np.isfinite(out)) and np.allclose(out, 0.0)


def test_falloff_power_weakens_with_range():
    near = field_accel(np.array([[1.0, 0.0, 0.0]]),
                       [_row(FIELD_FORCE, strength=1.0, power=2.0)])
    far = field_accel(np.array([[2.0, 0.0, 0.0]]),
                      [_row(FIELD_FORCE, strength=1.0, power=2.0)])
    assert np.allclose(near[0, 0], 1.0)
    assert np.allclose(far[0, 0], 0.25), "inverse square, so twice out is a quarter"


def test_zero_power_does_not_weaken_at_all():
    a = field_accel(np.array([[1.0, 0.0, 0.0]]), [_row(FIELD_FORCE, strength=2.0)])
    b = field_accel(np.array([[50.0, 0.0, 0.0]]), [_row(FIELD_FORCE, strength=2.0)])
    assert np.allclose(a[0, 0], b[0, 0])


def test_max_distance_cuts_the_field_off():
    rows = [_row(FIELD_WIND, axis=(1, 0, 0), strength=6.0, max_dist=5.0)]
    inside = field_accel(np.array([[0.0, 0.0, 4.0]]), rows)
    outside = field_accel(np.array([[0.0, 0.0, 6.0]]), rows)
    assert np.allclose(inside[0], [6.0, 0.0, 0.0])
    assert np.allclose(outside[0], 0.0)


def test_fields_add_together():
    rows = [
        _row(FIELD_WIND, axis=(1, 0, 0), strength=2.0),
        _row(FIELD_WIND, axis=(0, 1, 0), strength=3.0),
    ]
    out = field_accel(np.zeros((1, 3)), rows)
    assert np.allclose(out[0], [2.0, 3.0, 0.0])
