"""What will Tetrahedralize cost on the selected object, before you run it.

Tetrahedralize blocks Blender's main thread with no progress bar, so a cage
that takes forty minutes to bind is indistinguishable from a hang - you kill
it and call it a crash. This runs only the cheap stage (the voxel occupancy
test, a couple of seconds) and projects the expensive one from it.

Run it from Blender's Scripting tab with the mesh selected, or:

    blender myfile.blend --background --python tools/estimate_cage.py
"""

import importlib
import time

import bpy

# Installed as an extension the package is bl_ext.user_default.marrow; run
# from a clone of the repo it is plain marrow. Try both rather than make the
# user care which one they have.
for _pkg in ("bl_ext.user_default.marrow", "marrow"):
    try:
        _inside = importlib.import_module(f"{_pkg}.blender.inside_bvh")
        _lattice = importlib.import_module(f"{_pkg}.core.lattice")
        break
    except ImportError:
        continue
else:
    raise SystemExit(
        "Marrow not importable. Enable the add-on, or run this from the "
        "repo root so that `marrow` is on sys.path."
    )

cell_mask_from_object = _inside.cell_mask_from_object
build_lattice = _lattice.build_lattice
_bind_points = importlib.import_module(f"{_pkg}.core.bind").bind_points

# The bind cost is measured on the real mesh rather than assumed. It used to
# be a flat 200ns per (vertex, tet) pair, but bind_points now narrows the
# search with a spatial grid, so the cost depends on how many of this mesh's
# vertices land outside the cage and take the slower nearest-tet path - a
# thin-limbed shape has far more of those than a chunky one. Two samples of
# different sizes separate the one-off grid build from the per-vertex cost.
_SAMPLE_SMALL = 50
_SAMPLE_LARGE = 550

# session.MAX_NODES. Duplicated rather than imported so this script still
# runs against an install that has not been imported yet.
MAX_NODES = 200_000


def _time_once(fn, tetmesh, sample):
    t0 = time.perf_counter()
    fn(tetmesh.nodes, tetmesh.tets, sample)
    return time.perf_counter() - t0


def _project_bind(obj, tetmesh, n_verts):
    """Seconds bind_points will take, from two timed samples of this mesh.

    Sampling evenly across the vertex buffer, not the first N: mesh vertices
    are ordered by construction, so a prefix can be entirely head or
    entirely limb and miss the mix of inside-the-cage and outside-the-cage
    points that sets the cost.
    """
    import numpy as np

    co = np.empty(n_verts * 3)
    obj.data.vertices.foreach_get("co", co)
    world = np.array(obj.matrix_world.to_4x4())
    verts = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]

    timings = []
    for count in (_SAMPLE_SMALL, _SAMPLE_LARGE):
        take = min(count, n_verts)
        sample = verts[np.linspace(0, n_verts - 1, take).astype(int)]
        # Best of two. A single timing of a sub-second sample is noisy enough
        # that a coarser cage could be reported as slower than a finer one,
        # which is exactly backwards from the advice this script exists to
        # give. Noise only ever adds, so the minimum is the honest estimate.
        best = min(
            _time_once(_bind_points, tetmesh, sample),
            _time_once(_bind_points, tetmesh, sample),
        )
        timings.append((take, best))

    (n_a, t_a), (n_b, t_b) = timings
    if n_b <= n_a:                       # tiny mesh, one sample is the answer
        return t_b / max(n_b, 1) * n_verts
    # Slope is the per-vertex cost; the intercept is the one-off grid build.
    per_vert = (t_b - t_a) / (n_b - n_a)
    build = max(t_a - per_vert * n_a, 0.0)
    return build + per_vert * n_verts


def estimate(obj, spacing=None):
    if spacing is None:
        # obj.marrow only exists once the add-on has registered.
        spacing = getattr(getattr(obj, "marrow", None), "resolution", 0.25)
    spacing = float(spacing)
    n_verts = len(obj.data.vertices)

    t0 = time.perf_counter()
    mask, bounds_min = cell_mask_from_object(obj, spacing)
    mask_s = time.perf_counter() - t0

    occupied = int(mask.sum())
    if occupied == 0:
        print(f"\n  No cells inside at Resolution {spacing}. Lower it.")
        return

    tetmesh = build_lattice(bounds_min, spacing, mask)
    bind_s = _project_bind(obj, tetmesh, n_verts)

    print(f"\n  object            {obj.name}")
    print(f"  render vertices   {n_verts:,}")
    print(f"  resolution        {spacing} m")
    print(f"  grid              {mask.shape[0]} x {mask.shape[1]} x "
          f"{mask.shape[2]}  ({mask.size:,} cells)")
    print(f"  occupied          {occupied:,} ({100 * occupied / mask.size:.1f}%)")
    print(f"  cage nodes        {tetmesh.n_nodes:,}")
    print(f"  cage tets         {tetmesh.n_tets:,}")
    print(f"\n  voxel pass        {mask_s:.1f}s   (already done, above)")
    print(f"  bind pass         {bind_s:.0f}s ({bind_s / 60:.1f} min)")

    if tetmesh.n_nodes > MAX_NODES:
        print(f"\n  REFUSED AT BAKE: {tetmesh.n_nodes:,} nodes is over the "
              f"{MAX_NODES:,} budget.")
        print("  Tetrahedralize would still spend the time above, then Bake "
              "would reject it.")

    if bind_s > 120:
        coarser = spacing * 2
        print(f"\n  At Resolution {coarser} the cage is ~8x smaller and the "
              f"bind is ~{bind_s / 8 / 60:.1f} min.")
        print("  Adaptive also helps: it keeps the fine cells near the "
              "surface and leaves the bulk coarse.")


def main():
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        print("Select a mesh object first.")
        return
    estimate(obj)


if __name__ == "__main__":
    main()
