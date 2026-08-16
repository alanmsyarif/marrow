"""Blender-side tests for the adaptive lattice: the GPU blend pass.

Item 7 diffs the blend kernel against its numpy oracle - without that, a
sign error in the GLSL is indistinguishable from a sign error in the
algebra. Item 8 runs a full adaptive body through the real solver to show
the glue holds under rigid motion.
"""

import bpy
import gpu
import numpy as np

import marrow
from _oracle_harness import assert_close
from marrow.blender import handlers
from marrow.blender.session import find_cage
from marrow.blender.storage import read_blend, read_tetmesh
from marrow.core.adaptive import build_adaptive_lattice
from marrow.core.coloring import color_sets
from marrow.core.layout import color_order, pack_blend, pack_nodes, unpack_vec3
from marrow.core.lattice import build_lattice, grid_dims
from marrow.core.solver_ref import (
    SolverParams,
    blend_project,
    make_state,
)
from marrow.gpu.kernels import BLEND_IMAGES, BLEND_PUSH, BLEND_SRC, build
from marrow.gpu.solver import GPUSolver
from marrow.gpu.textures import download, flush, make_flush_shader, upload

gpu.init()

TOL = 2e-6  # float32, one zero-compliance projection on a unit-scale cage


class BoxOracle:
    """Analytic oracle for a union of axis-aligned boxes; the core tests use
    the same one. Distance underestimates near box junctions, which only
    over-refines."""

    def __init__(self, boxes):
        self.boxes = [(np.asarray(lo, float), np.asarray(hi, float))
                      for lo, hi in boxes]
        self.bounds_min = np.min([lo for lo, _ in self.boxes], axis=0)
        self.bounds_max = np.max([hi for _, hi in self.boxes], axis=0)

    def distance(self, p):
        p = np.asarray(p, dtype=np.float64)
        best = np.inf
        for lo, hi in self.boxes:
            if np.all(p > lo) and np.all(p < hi):
                d = float(np.min(np.minimum(p - lo, hi - p)))
            else:
                d = float(np.linalg.norm(p - np.clip(p, lo, hi)))
            best = min(best, d)
        return best

    def inside(self, p):
        p = np.asarray(p, dtype=np.float64)
        return any(np.all(p > lo) and np.all(p < hi) for lo, hi in self.boxes)


def _stub_oracle():
    return BoxOracle([
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([1.0, 0.4, 0.4], [1.2, 0.6, 0.6]),
    ])


def _ordered_rows(idx, w, n_nodes):
    """The same colouring the GPUSolver applies, so both sides sweep the
    rows in the same order and parity is exact."""
    color_rows = [
        [int(idx[r, 0])]
        + [int(idx[r, 1 + s]) if w[r, s] > 0.0 else -1 for s in range(4)]
        for r in range(idx.shape[0])
    ]
    colors = color_sets(color_rows, n_nodes)
    order = color_order(colors)
    counts = np.bincount(colors, minlength=int(colors.max()) + 1)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return idx[order], w[order], offsets


def _run_blend(state, idx, w, offsets):
    """One colour-by-colour GPU sweep of the blend kernel."""
    shader = build("blend", BLEND_SRC, BLEND_IMAGES, BLEND_PUSH)
    tex_p = upload(pack_nodes(state.predicted, state.inv_mass))
    tex_b = upload(pack_blend(idx, w))

    for c in range(len(offsets) - 1):
        begin, end = int(offsets[c]), int(offsets[c + 1])
        if end <= begin:
            continue
        shader.bind()
        shader.image("p", tex_p)
        shader.image("blend", tex_b)
        shader.uniform_int("color_begin", begin)
        shader.uniform_int("color_end", end)
        gpu.compute.dispatch(shader, (end - begin + 63) // 64, 1, 1)

    flush(make_flush_shader("RGBA32F"), tex_p)
    return unpack_vec3(download(tex_p), state.predicted.shape[0])


def test_blend_kernel_matches_the_oracle():
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    assert idx.shape[0] > 0
    o_idx, o_w, offsets = _ordered_rows(idx, w, mesh.n_nodes)

    rng = np.random.default_rng(7)
    deform = mesh.nodes + rng.uniform(-0.05, 0.05, mesh.nodes.shape)

    state = make_state(mesh.nodes)
    state.predicted[:] = deform
    gpu_out = _run_blend(state, o_idx, o_w, offsets)

    oracle = make_state(mesh.nodes)
    oracle.predicted[:] = deform
    blend_project(oracle, o_idx, o_w)
    assert_close(gpu_out, oracle.predicted, TOL, "blend projection")

    # The projection must actually reduce the glue error, not merely match
    # the oracle in doing nothing. One sweep over rows that share masters
    # across colours only partially closes it - parity above is the real
    # assertion, this just proves the pass does work.
    before = np.abs(np.einsum("rw,rwd->rd", o_w, deform[o_idx[:, 1:]])
                    - deform[o_idx[:, 0]]).max()
    after = np.abs(np.einsum("rw,rwd->rd", o_w, gpu_out[o_idx[:, 1:]])
                   - gpu_out[o_idx[:, 0]]).max()
    assert after < before, (
        f"glue error did not shrink: {before:.3e} -> {after:.3e}"
    )


def test_blend_kernel_matches_the_oracle_with_pinned_hanging_nodes():
    """A pinned hanging node keeps its place while its masters are dragged
    towards it - the no-early-return path."""
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    o_idx, o_w, offsets = _ordered_rows(idx, w, mesh.n_nodes)
    pinned = np.unique(o_idx[:, 0]).astype(np.int32)

    rng = np.random.default_rng(9)
    deform = mesh.nodes + rng.uniform(-0.05, 0.05, mesh.nodes.shape)

    state = make_state(mesh.nodes, pinned=pinned)
    state.predicted[:] = deform
    gpu_out = _run_blend(state, o_idx, o_w, offsets)

    oracle = make_state(mesh.nodes, pinned=pinned)
    oracle.predicted[:] = deform
    blend_project(oracle, o_idx, o_w)
    assert_close(gpu_out, oracle.predicted, TOL, "blend with pinned hangings")
    assert np.allclose(gpu_out[pinned], deform[pinned], atol=TOL), (
        "pinned hanging nodes must not move"
    )


def test_blend_kernel_matches_the_oracle_with_nonuniform_mass():
    """Uneven inverse masses change every share; a pack mistake in the
    weight channel shows up here."""
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    o_idx, o_w, offsets = _ordered_rows(idx, w, mesh.n_nodes)

    rng = np.random.default_rng(11)
    inv_mass = rng.uniform(0.2, 3.0, mesh.n_nodes)
    deform = mesh.nodes + rng.uniform(-0.05, 0.05, mesh.nodes.shape)

    state = make_state(mesh.nodes)
    state.inv_mass[:] = inv_mass
    state.predicted[:] = deform
    gpu_out = _run_blend(state, o_idx, o_w, offsets)

    oracle = make_state(mesh.nodes)
    oracle.inv_mass[:] = inv_mass
    oracle.predicted[:] = deform
    blend_project(oracle, o_idx, o_w)
    assert_close(gpu_out, oracle.predicted, TOL, "blend with uneven mass")


def test_free_fall_keeps_hanging_nodes_glued():
    """A rigid adaptive body through the full solver: the glue rows must
    keep every hanging node on its masters' interpolant frame after frame,
    and the fall must stay ballistic."""
    mesh, idx, w = build_adaptive_lattice(0.25, 0.03, _stub_oracle())
    params = SolverParams(mu=0.0, lam=0.0, damping=1.0)
    solver = GPUSolver(mesh, np.ones(mesh.n_nodes), params,
                       blend_rows=(idx, w))
    assert solver.sh_blend is not None

    frames = 5
    for _ in range(frames):
        solver.step()
    pos = solver.positions()
    assert np.isfinite(pos).all()

    # The discrete scheme is explicit-Euler-on-predict: substep k falls
    # g*h^2*(k+1), so after n substeps the drop is g*h^2*n(n+1)/2 - not
    # the continuous 1/2*g*t^2 (they differ by exactly 1/2*g*h*t).
    h = params.dt / params.substeps
    n = frames * params.substeps
    drop = np.asarray(params.gravity) * (h * h * n * (n + 1) / 2.0)
    expected = mesh.nodes + drop
    assert_close(pos, expected, 2e-4, "ballistic free fall")

    blended = np.einsum("rw,rwd->rd", w, pos[idx[:, 1:]])
    residual = np.abs(blended - pos[idx[:, 0]]).max()
    assert residual < 2e-5, f"hanging nodes tore off the face: {residual:.3e}"


def test_no_blend_rows_allocates_nothing():
    """The uniform path: blend_rows=None must leave the pass inert."""
    mesh, _, _ = build_adaptive_lattice(0.25, 0.25, _stub_oracle())
    solver = GPUSolver(mesh, np.ones(mesh.n_nodes), SolverParams())
    assert solver.sh_blend is None
    assert solver.blend_offsets is None


class _SphereRodOracle:
    """A chunky sphere bulk with a thin rod out one side - the cantilever.

    The bulk is where the adaptive lattice saves nodes; the rod is the
    thin end that forces the min size and carries the measured tip
    deflection. Analytic distance and inside, like BoxOracle.
    """

    def __init__(self, radius, rod_length):
        self.c = np.zeros(3)
        self.r = float(radius)
        self.lo = np.array([radius - 0.2, -0.1, -0.1])
        self.hi = np.array([radius + rod_length, 0.1, 0.1])
        self.bounds_min = np.minimum(self.c - self.r, self.lo)
        self.bounds_max = np.maximum(self.c + self.r, self.hi)

    def distance(self, p):
        p = np.asarray(p, dtype=np.float64)
        d_sphere = abs(float(np.linalg.norm(p - self.c)) - self.r)
        if np.all(p > self.lo) and np.all(p < self.hi):
            d_box = float(np.min(np.minimum(p - self.lo, self.hi - p)))
        else:
            d_box = float(np.linalg.norm(p - np.clip(p, self.lo, self.hi)))
        return min(d_sphere, d_box)

    def inside(self, p):
        p = np.asarray(p, dtype=np.float64)
        return float(np.linalg.norm(p - self.c)) < self.r or (
            np.all(p > self.lo) and np.all(p < self.hi)
        )


_CANT_MIN = 0.0625


def _tip_drop(mesh, blend_rows):
    """Settled z-deflection of the rod's free end under gravity, the
    sphere cap pinned. Damping bleeds the oscillation out so both cages
    are compared at the same quasi-static point."""
    params = SolverParams(substeps=3, damping=0.85)
    inv_mass = np.ones(mesh.n_nodes)
    inv_mass[mesh.nodes[:, 0] < -2.6] = 0.0
    solver = GPUSolver(mesh, inv_mass, params, blend_rows=blend_rows)
    for _ in range(20):
        solver.step()
    pos = solver.positions()
    assert np.isfinite(pos).all()
    tip = np.abs(mesh.nodes[:, 0] - mesh.nodes[:, 0].max()) < 1e-9
    return mesh.nodes[tip, 2].mean() - pos[tip, 2].mean()


def test_cantilever_matches_uniform_fine_with_fewer_nodes():
    """Item 9: the adaptive cantilever deflects like a uniform-at-min one
    at a quarter of its nodes. The uniform reference comes straight from
    the mask path (vectorized), not the oracle loop."""
    oracle = _SphereRodOracle(2.9, 1.2)
    mesh_a, idx, w = build_adaptive_lattice(0.5, _CANT_MIN, oracle)
    assert idx.shape[0] > 0

    dims = grid_dims(oracle.bounds_min, oracle.bounds_max, _CANT_MIN)
    ii, jj, kk = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]),
        indexing="ij",
    )
    centres = oracle.bounds_min + (np.stack([ii, jj, kk], axis=-1) + 0.5) * _CANT_MIN
    mask = (np.linalg.norm(centres - oracle.c, axis=-1) < oracle.r) | np.all(
        (centres > oracle.lo) & (centres < oracle.hi), axis=-1
    )
    mesh_f = build_lattice(oracle.bounds_min, _CANT_MIN, mask)

    assert mesh_a.n_nodes * 4 <= mesh_f.n_nodes, (
        f"adaptive {mesh_a.n_nodes} nodes is not 4x under uniform-fine "
        f"{mesh_f.n_nodes}"
    )

    drop_f = _tip_drop(mesh_f, None)
    drop_a = _tip_drop(mesh_a, (idx, w))
    assert drop_f > 1e-3, f"the tip barely moved: {drop_f:.3e}"
    assert abs(drop_a - drop_f) <= 0.15 * drop_f, (
        f"tip deflection diverged: adaptive {drop_a:.4e} vs "
        f"uniform-fine {drop_f:.4e}"
    )


def _union_hull_object(name, boxes, spacing):
    """A watertight quad mesh of the union of grid-aligned boxes: one quad
    per cell boundary face with air on the far side. Winding is irrelevant
    to the BVH inside test."""
    inside = set()
    for lo, hi in boxes:
        lo_i = [round(v / spacing) for v in lo]
        hi_i = [round(v / spacing) for v in hi]
        for i in range(lo_i[0], hi_i[0]):
            for j in range(lo_i[1], hi_i[1]):
                for k in range(lo_i[2], hi_i[2]):
                    inside.add((i, j, k))
    verts: dict = {}
    faces = []
    for i, j, k in inside:
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            nb = [i, j, k]
            nb[axis] += sign
            if tuple(nb) in inside:
                continue
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = [i, j, k]
                c[(axis + 1) % 3] += du
                c[(axis + 2) % 3] += dv
                key = tuple(c)
                if key not in verts:
                    verts[key] = len(verts)
                corners.append(verts[key])
            faces.append(corners if sign > 0 else corners[::-1])
    coords = np.zeros((len(verts), 3))
    for key, index in verts.items():
        coords[index] = np.asarray(key) * spacing
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(coords.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def test_operator_thick_plus_thin_with_adaptive_on():
    """Item 10: tetrahedralize a thick-plus-thin object with Adaptive on.
    The cage carries blend rows, the thin bar holds at least two cells
    across, and three live frames stay finite all the way to the skin."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        marrow.unregister()
    except Exception:
        pass
    marrow.register()

    obj = _union_hull_object("thick_thin", [
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.4, 1.0, 0.4), (0.6, 1.2, 0.6)),
    ], 0.05)
    obj.marrow.resolution = 0.25
    obj.marrow.adaptive = True
    obj.marrow.min_resolution = 0.06
    bpy.ops.marrow.tetrahedralize()

    cage = find_cage(obj)
    assert cage is not None
    blend = read_blend(cage.data)
    assert blend is not None and blend[0].shape[0] > 0, (
        "an adaptive cage must store blend rows"
    )
    tetmesh, _ = read_tetmesh(cage.data)

    # The bar core away from its edges: node planes along its thickness.
    nodes = tetmesh.nodes
    bar = nodes[
        (nodes[:, 0] >= 0.45) & (nodes[:, 0] <= 0.55)
        & (nodes[:, 2] >= 0.45) & (nodes[:, 2] <= 0.55)
        & (nodes[:, 1] >= 1.0 - 1e-9) & (nodes[:, 1] <= 1.2 + 1e-9)
    ]
    assert bar.shape[0] > 0, "no cage nodes inside the thin bar"
    finest = 0.25 / 8  # levels(0.25, 0.06) == 3
    planes = np.unique(np.round(bar[:, 1] / finest))
    assert len(planes) >= 4, (
        f"the thin bar resolved to {len(planes)} planes; two cells "
        "across need at least four"
    )

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 8
    try:
        for frame in range(1, 4):
            scene.frame_set(frame)
        session = handlers.SESSIONS[obj.name]
        assert session.solver.sh_blend is not None, (
            "the live solver must pick up the stored blend rows"
        )
        skin = session.frame_positions(3)
        assert skin is not None and np.isfinite(skin).all()
        assert np.isfinite(session.solver.positions()).all()
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        assert np.isfinite(co).all()
    finally:
        handlers.unregister_handler()
