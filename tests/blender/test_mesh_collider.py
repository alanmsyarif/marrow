"""Colliding against an arbitrary mesh, via a signed distance field.

See docs/superpowers/specs/2026-08-13-marrow-mesh-colliders-design.md.

The field is baked in the collider's local space, so the object transform
places it exactly as it does the unit sphere and the unit box, and a collider
that moves needs no rebake.
"""

import bpy
import gpu
import numpy as np
from mathutils import Matrix

from marrow.blender import sdf
from marrow.core.layout import pack_nodes, unpack_vec3
from marrow.gpu.kernels import COLLIDE_IMAGES, COLLIDE_PUSH, COLLIDE_SRC, build
from marrow.gpu.textures import (
    download,
    flush,
    make_flush_shader,
    upload,
    upload3d,
)

gpu.init()

MESH = 3


def _object(add, **kwargs):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    add(**kwargs)
    return bpy.context.active_object


def _collide(points, collider, resolution=0.25):
    """Run the real collide kernel against ``collider`` as a mesh collider."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    field, grid = sdf.bake(collider, resolution)
    assert field is not None, "nothing was baked"

    grid_m = Matrix([list(row) for row in grid])
    world = collider.matrix_world.copy()

    shader = build("collide", COLLIDE_SRC, COLLIDE_IMAGES, COLLIDE_PUSH)
    # Every texture must outlive the dispatch. A temporary passed straight
    # into shader.image() is collected before the kernel runs and the image
    # unit is left pointing at nothing - imageSize then reports zero and the
    # branch silently does nothing.
    tex_p = upload(pack_nodes(points, np.ones(n)))
    tex_stick = upload(np.zeros_like(pack_nodes(points, np.ones(n))))
    tex_sdf = upload3d(field)

    shader.bind()
    shader.image("p", tex_p)
    shader.image("stick", tex_stick)
    shader.image("sdf", tex_sdf)
    shader.uniform_float("ground_z", 0.0)
    shader.uniform_int("kind", MESH)
    shader.uniform_int("n_nodes", n)
    shader.uniform_float("to_local", grid_m @ world.inverted())
    shader.uniform_float("to_world", world @ grid_m.inverted())
    shader.uniform_int("collider_id", 1)
    shader.uniform_int("sticky", 0)
    shader.uniform_float("break_dist", 1.0e30)
    gpu.compute.dispatch(shader, (n + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), n)


def test_the_baked_field_matches_an_analytic_sphere():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    sdf.clear_cache()
    field, grid = sdf.bake(ball, 0.25)

    n = field.shape[0]
    width = 1.0 / grid[0, 0]
    low = -grid[:3, 3] * width
    ticks = low[0] + (np.arange(n) + 0.5) * (width / n)
    X, Y, Z = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    got = np.transpose(field, (2, 1, 0)).astype(np.float64) * width
    want = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 1.0

    # The residual is the UV sphere's own faceting, not the bake.
    assert np.abs(got - want).max() < 0.02, (
        f"field is wrong by {np.abs(got - want).max():.4f}"
    )


def test_signs_are_right_away_from_the_surface():
    """One fixed ray miscounts for samples sitting against a face. An SDF
    samples densely there, so the parity test votes over three rays."""
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    sdf.clear_cache()
    field, grid = sdf.bake(ball, 0.1)

    n = field.shape[0]
    width = 1.0 / grid[0, 0]
    low = -grid[:3, 3] * width
    ticks = low[0] + (np.arange(n) + 0.5) * (width / n)
    X, Y, Z = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    got = np.transpose(field, (2, 1, 0)).astype(np.float64) * width
    want = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 1.0

    clear = np.abs(want) > 0.02
    wrong = int((np.sign(got[clear]) != np.sign(want[clear])).sum())
    assert wrong == 0, f"{wrong} voxels signed wrongly"


def test_interior_points_are_pushed_to_the_surface():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    out = _collide([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5], [0.6, 0.0, 0.0]], ball)
    radii = np.linalg.norm(out, axis=1)
    assert np.all(radii > 0.9), f"not pushed out far enough: {radii}"
    assert np.all(radii < 1.1), f"pushed past the surface: {radii}"


def test_the_push_is_along_the_surface_normal():
    """Central differences sample at the cell corner rather than the sample
    point and skew the push diagonally; the analytic trilinear gradient does
    not. A node on the axis must come back on the axis."""
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    out = _collide([[0.0, 0.0, 0.5]], ball)
    assert abs(out[0][0]) < 1e-4 and abs(out[0][1]) < 1e-4, (
        f"a push straight up came out skewed: {out[0]}"
    )


def test_exterior_points_are_left_alone():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    points = [[0.0, 0.0, 1.5], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    out = _collide(points, ball)
    assert np.allclose(out, points, atol=1e-6), f"moved a point outside: {out}"


def test_a_node_in_the_hole_of_a_torus_is_not_pushed():
    """The test this design exists for.

    A bounding box, a convex hull and a vertex cloud all fail it. Only a real
    signed distance field knows the middle of a torus is outside the solid.
    """
    torus = _object(
        bpy.ops.mesh.primitive_torus_add, major_radius=1.0, minor_radius=0.25
    )
    out = _collide([[0.0, 0.0, 0.0]], torus, resolution=0.1)
    assert np.allclose(out[0], [0.0, 0.0, 0.0], atol=1e-6), (
        f"the hole was treated as solid: {out[0]}"
    )


def test_a_node_in_the_ring_of_a_torus_is_pushed_out():
    """Control for the hole test: the solid part must still collide."""
    torus = _object(
        bpy.ops.mesh.primitive_torus_add, major_radius=1.0, minor_radius=0.25
    )
    out = _collide([[1.0, 0.0, 0.0]], torus, resolution=0.1)
    moved = float(np.linalg.norm(out[0] - np.array([1.0, 0.0, 0.0])))
    assert moved > 0.1, f"a node inside the ring was not pushed: moved {moved:.4f}"


def test_it_follows_the_collider_transform_with_no_rebake():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    sdf.clear_cache()
    ball.matrix_world = Matrix.Translation((5.0, 0.0, 0.0))
    bpy.context.view_layer.update()

    # A point inside the moved collider is pushed out; the old location is not.
    out = _collide([[5.0, 0.0, 0.5], [0.0, 0.0, 0.0]], ball)
    assert np.linalg.norm(out[0] - np.array([5.0, 0.0, 0.5])) > 0.3, (
        "the collider did not move with its object"
    )
    assert np.allclose(out[1], [0.0, 0.0, 0.0], atol=1e-6), (
        "the collider was still acting at its old position"
    )


def test_scale_changes_the_collider_size():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    sdf.clear_cache()
    ball.matrix_world = Matrix.Diagonal((2.0, 2.0, 2.0, 1.0))
    bpy.context.view_layer.update()

    out = _collide([[0.0, 0.0, 1.5]], ball)
    radius = float(np.linalg.norm(out[0]))
    assert radius > 1.8, f"a doubled collider should reach ~2.0, got {radius:.3f}"


def test_the_bake_is_cached():
    ball = _object(bpy.ops.mesh.primitive_uv_sphere_add, radius=1.0)
    sdf.clear_cache()
    first, _ = sdf.bake(ball, 0.25)
    second, _ = sdf.bake(ball, 0.25)
    assert first is second, "the field was baked twice for the same collider"


def test_an_object_with_no_geometry_bakes_nothing():
    """An empty has a transform and no mesh. Skipping beats raising."""
    empty = _object(bpy.ops.object.empty_add)
    sdf.clear_cache()
    field, grid = sdf.bake(empty, 0.25)
    assert field is None and grid is None
