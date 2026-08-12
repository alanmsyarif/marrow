"""Sphere and box primitive colliders.

No oracle - solver_ref has no collision - so these are tested on their own
behaviour. Primitives are unit-sized in the collider's local space and shaped
entirely by its transform, so a default Blender sphere (radius 1) or cube
(size 2) maps exactly.
"""

import gpu
import numpy as np
from mathutils import Matrix

from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.gpu.kernels import COLLIDE_SRC, build
from marrow.gpu.textures import download, flush, make_flush_shader, upload

gpu.init()

IMAGES = [("RGBA32F", "FLOAT_2D", "p", {"READ", "WRITE"})]
PUSH = [
    ("FLOAT", "ground_z"),
    ("INT", "kind"),
    ("INT", "n_nodes"),
    ("MAT4", "to_local"),
    ("MAT4", "to_world"),
]

SPHERE, BOX = 1, 2


def _collide(points, kind, xform, inv_mass=None):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if inv_mass is None:
        inv_mass = np.ones(n)

    shader = build("collide", COLLIDE_SRC, IMAGES, PUSH)
    tex_p = upload(pack_nodes(points, inv_mass))

    shader.bind()
    shader.image("p", tex_p)
    shader.uniform_float("ground_z", 0.0)
    shader.uniform_int("kind", kind)
    shader.uniform_int("n_nodes", n)
    shader.uniform_float("to_local", xform.inverted())
    shader.uniform_float("to_world", xform)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), n)


def test_sphere_pushes_interior_points_to_its_surface():
    inside = np.array([[0.0, 0.0, 0.5], [0.3, 0.0, 0.0], [0.0, -0.2, 0.1]])
    out = _collide(inside, SPHERE, Matrix.Identity(4))
    radii = np.linalg.norm(out, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-5), f"radii {radii.tolist()} should all be 1.0"


def test_sphere_leaves_exterior_points_alone():
    outside = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, -2.5]])
    out = _collide(outside, SPHERE, Matrix.Identity(4))
    assert np.allclose(out, outside, atol=1e-6)


def test_sphere_follows_its_transform():
    """A sphere moved to (5,0,0) must push points there, not at the origin."""
    xform = Matrix.Translation((5.0, 0.0, 0.0))
    out = _collide(np.array([[5.0, 0.0, 0.4]]), SPHERE, xform)
    assert np.isclose(np.linalg.norm(out[0] - np.array([5.0, 0.0, 0.0])), 1.0, atol=1e-5)


def test_sphere_scale_changes_its_radius():
    xform = Matrix.Diagonal((2.0, 2.0, 2.0, 1.0))
    out = _collide(np.array([[0.0, 0.0, 1.0]]), SPHERE, xform)
    assert np.isclose(np.linalg.norm(out[0]), 2.0, atol=1e-5), (
        f"a 2x scaled unit sphere should have radius 2, got {np.linalg.norm(out[0]):.4f}"
    )


def test_sphere_centre_does_not_produce_nan():
    """Dead centre has no defined push direction - it must not divide by zero."""
    out = _collide(np.array([[0.0, 0.0, 0.0]]), SPHERE, Matrix.Identity(4))
    assert np.all(np.isfinite(out)), f"centre point produced {out}"
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_box_pushes_interior_points_to_the_nearest_face():
    """Unit box spans -1..1; (0, 0, 0.9) is nearest the +z face."""
    out = _collide(np.array([[0.0, 0.0, 0.9]]), BOX, Matrix.Identity(4))
    assert np.isclose(out[0][2], 1.0, atol=1e-5), f"expected z=1.0, got {out[0].tolist()}"
    assert np.allclose(out[0][:2], 0.0, atol=1e-6), "only the nearest axis should move"


def test_box_picks_the_nearest_face_per_point():
    points = np.array([[0.95, 0.0, 0.0], [0.0, -0.9, 0.0], [0.0, 0.0, -0.8]])
    out = _collide(points, BOX, Matrix.Identity(4))
    assert np.isclose(out[0][0], 1.0, atol=1e-5)
    assert np.isclose(out[1][1], -1.0, atol=1e-5)
    assert np.isclose(out[2][2], -1.0, atol=1e-5)


def test_box_leaves_exterior_points_alone():
    outside = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    out = _collide(outside, BOX, Matrix.Identity(4))
    assert np.allclose(out, outside, atol=1e-6)


def test_a_pin_outranks_a_collider():
    inv_mass = np.array([0.0, 1.0])
    points = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]])
    out = _collide(points, SPHERE, Matrix.Identity(4), inv_mass)
    assert np.allclose(out[0], points[0], atol=1e-6), "a pinned node must not be pushed"
    assert np.isclose(np.linalg.norm(out[1]), 1.0, atol=1e-5)
