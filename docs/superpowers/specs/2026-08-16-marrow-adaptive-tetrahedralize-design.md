# Marrow: Adaptive Tetrahedralization

Design spec, 2026-08-16. Extends the v1 spec, `2026-08-11-marrow-gpu-tet-soft-body-design.md`, whose cage was a single uniform grid.

## Summary

The cage resolution follows the shape. An octree refines cells towards the surface and through thin features, driven by BVH distance queries, and every leaf is tetrahedralized with the existing parity-alternated 5-tet split. Level boundaries are glued by a new XPBD constraint pass that holds each hanging node at the bilinear interpolation of the coarse face's corners. One new core module, one new kernel, two new settings. The uniform path stays bit-identical.

## Motivation

A uniform grid resolves the thinnest feature it must capture, and pays for it everywhere. The motivating case is a walking character: torso, tapering legs, thin ankle stubs. At Resolution 0.25 the ankles receive zero cells and the legs two or three; giving the ankles two cells uniformly means ~0.03 everywhere, which is (0.25/0.03)³ ≈ 580x the torso's node count. The cage is only the simulation proxy - the render surface is skinned from it - so what matters is that the material is thick enough in cells to bend and contact correctly, and that thickness varies over the shape.

## Design decisions

### D1. Octree over the bounds, refinement driven by distance to surface

The root cells are the current uniform grid at Resolution (the max size). A cell refines while `size > min_size and size > k * d(centre)`, where d is the distance from the cell centre to the surface (`BVHTree.find_nearest`). One rule gives three behaviours:

- A cell touching the surface has centre depth ≈ size/2 < size, so it refines to min: a fine boundary layer.
- A thin feature is entirely inside that layer, so it fills at min: the ankles get cells.
- Deep interior has large d and stops at max: a coarse bulk.

k = 1.0, an internal constant: a thin slab of thickness t ends at size ≤ t/2 near its centre, i.e. at least two cells across.

The three-ray inside test runs only on leaves - a cell that will refine further does not need it, which halves the expensive parity casts versus testing every visited cell.

### D2. 2:1 balance

After the distance pass, a queue propagation enforces that no two face-adjacent leaves differ by more than one level. Balance is not what makes the glue correct (D4's bilinear support is defined for any point on a coarse face) - it bounds the stiffness gradient and keeps the hanging-node bookkeeping to two cases: edge midpoint and face centre.

### D3. Leaves tetrahedralized with the existing 5-tet parity split

Parity is (i+j+k) of the leaf's integer coordinates *at its own level*, so equal-size neighbours - including neighbours from different octree branches - agree on face diagonals exactly as the uniform lattice does. `lattice._SPLIT_EVEN/_SPLIT_ODD` are reused unchanged.

### D4. Hanging nodes glued by bilinear interpolation constraints

Node dedup is exact: positions are keyed by integer coordinates at the finest level (dyadic), never by float comparison.

With 2:1 balance, a fine leaf face against a 2x coarse face contributes exactly three node kinds on that face: coarse corners (conforming), coarse edge midpoints, and the coarse face centre. Axis-aligned cube faces are planar, so every triangulation of the face agrees on the linear displacement field - the diagonal mismatch that cracks adaptive grids cannot occur. Hence:

- edge midpoint: constraint x_h = 0.5(a+b), the edge's endpoints;
- face centre: constraint x_h = 0.25(c0+c1+c2+c3), the face's four corners.

Masters are always corners of the coarse leaf, hence real nodes. A master may itself hang one level up; rows are sorted by master level so a Gauss-Seidel pass sees fresh master values, and the dependency is a DAG by level - no cycles.

These are ordinary XPBD position constraints (C = x_h − Σ wᵢx_mᵢ, compliance 0), mass-weighted by inverse mass like every other projection; pinned masters or a pinned hanging node fall out of the same algebra. They do not tear.

### D5. One new GPU pass, after the elastic solve

`predict -> solve colours -> blend colours -> attach -> contacts -> integrate`. The blend constraints are part of the material's connectivity, so they run before attachment and contacts; contacts keep the last word, as they already do.

Rows are coloured with the same greedy node-disjoint colouring as tets - a row's nodes are its hanging node plus its four master slots - so a dispatch is plain read-modify-write with no atomics.

### D6. No new stiffness knob; the existing volume-scaled compliance is the normalization

SOLVE_SRC divides compliance by the tet's rest volume, the standard XPBD size normalization: a 0.05 tet and a 0.25 tet at the same mu already resist equally per unit volume. Mixed sizes therefore need no extra per-level scaling in v1. A visible softness seam at level boundaries would be answered by per-level mu scaling; noted as a ceiling.

### D7. Adaptive off is the old code path, bit-identical

`Adaptive` defaults off. Off, tetrahedralize runs the current `cell_mask_from_object` + `build_lattice` and the solver builds no blend textures - every existing test, bake and saved .blend behaves exactly as today. On, the new path runs and the cage stores its blend rows beside its tets.

### D8. UI: Resolution becomes the max; Min Size is new

The Cage box gains `Adaptive` (bool, default off) and `Min Size` (length, default 0.03, min 0.001, soft_max 0.25). Levels L = ceil(log2(max/min)); the effective min is max/2^L, so the leaf sizes are always exact halvings of Resolution.

## Data layout

One new texture, uploaded once at solver construction, allocated only when rows exist:

| Name | Format | Texels | Contents |
|---|---|---|---|
| `tex_blend` | RGBA32F | 2·R | row r: texel 2r = (h, m0, m1, m2); texel 2r+1 = (m3, w0, w1, w2); w3 recovered as 1−w0−w1−w2 |

Edge rows store masters (a, b, a, b) with weights (0.5, 0.5, 0, 0); the kernel skips zero-weight slots. Indices as floats are exact below 2^24; the largest cage measured here is 37k nodes.

Push constants: `h`, `color_begin`, `color_end` - the same shape as the solve colour loop.

Persistence: cage-mesh ID properties `marrow_blend` (flat int32, five per row: h, m0..m3) and `marrow_blend_w` (flat float32, three per row: w0..w2), mirroring how TETS_KEY rides the cage. `clear_marrow_data` removes both. Colours are not stored; the session recomputes them.

## Components

### `marrow/core/adaptive.py` (new, pure numpy)

- `refine(bounds_min, max_size, min_size, oracle) -> leaves (M,4) int64` of (level, i, j, k). The oracle exposes `distance(centre) -> float` and `inside(centre) -> bool`. The frontier is walked level by level; the BVH calls happen in a Python loop over the frontier, the same one-time CPU-cost stance as `cell_mask_from_object`.
- `balance(leaves) -> leaves` - the 2:1 face-adjacency propagation.
- `build_adaptive_lattice(bounds_min, max_size, min_size, oracle) -> (TetMesh, blend_idx (R,5) int32, blend_w (R,4) float64)` - leaf splits with per-level parity, exact dyadic node dedup, hanging detection per fine face against its coarser neighbour, rows emitted sorted by master level.

`build_lattice` and `cell_mask_from_object` are untouched.

### `marrow/core/coloring.py`

`color_sets(sets, n_nodes)` generalizes the greedy loop to variable-length rows with −1 padding; `color_tets` delegates to it and returns identical colours to today.

### `marrow/core/solver_ref.py`

The oracle gains `blend_project(positions, rows, w)`, the CPU mirror of BLEND_SRC, so the parity suite covers the new pass the way it covers solve.

### `marrow/gpu/kernels.py`: `BLEND_SRC`

Dispatched over row colours. Loads the row, skips zero-weight slots, C = x_h − Σ wᵢx_mᵢ, dλ = −C / (w_h + Σ w_mᵢ·wᵢ²), then x_h += w_h·dλ and x_mᵢ −= w_mᵢ·w·dλ. Reads and writes `p` in place - safe because one colour's rows are node-disjoint. No early return for a pinned hanging node: inv_mass 0 removes it from the denominator and from the move, which is exactly the pin semantics.

### `marrow/gpu/solver.py`

`GPUSolver.__init__` gains `blend_rows=None`, a ((R,5) int, (R,4) float) pair. When present: colour the rows, upload `tex_blend`, build the shader; `substep_constraints` dispatches the colour loop between the solve loop and `_dispatch_attach`. When absent, nothing is allocated and nothing dispatches.

### `marrow/blender/inside_bvh.py`

`cell_oracle_from_object(obj)` returns `(bounds_min, oracle)` wrapping the existing `_world_bvh`: `distance` via `bvh.find_nearest`, `inside` via the existing three-ray `is_inside`. The modifier-parking rule lives in the caller (ops), which already parks the stack around the capture, so the oracle sees the base mesh.

### `marrow/blender/ops.py`

`tetrahedralize` branches on `settings.adaptive`: oracle path into `build_adaptive_lattice`, else the current mask path. Blend rows are written to the cage via storage; the report states nodes, tets and blend rows.

### `marrow/blender/storage.py`

`write_blend(mesh, idx, w)` and `read_blend(mesh) -> (idx, w) | None`; both keys added to `clear_marrow_data`.

### `marrow/blender/session.py`

`_build_solver` reads `read_blend(cage.data)` and passes `blend_rows` through to `GPUSolver`.

### `marrow/blender/ui.py`

Cage box: `adaptive` BoolProperty("Adaptive") and `min_resolution` FloatProperty("Min Size", default 0.03, min 0.001, soft_max 0.25, unit LENGTH), the row enabled by the toggle.

## Testing

Core, no GPU (`tests/core/test_adaptive.py`):

1. Thin-slab oracle (analytic box 1×1×0.1, max 0.25, min 0.025): slab leaves at the min level, deep bulk at max; balance holds - no face-adjacent leaves differ by more than one level.
2. Blend rows: every master exists as a node; weights sum to 1; the hanging node's rest position equals Σ wᵢ·master rest positions to 1e-9; edge rows have exactly two nonzero weights of 0.5.
3. `validate()` passes; total tet volume within 5% of the analytic volume; a thin box gets ≥2 cells across its thickness.
4. Oracle forced to a single level (min = max): output identical to `build_lattice` on the same mask and zero blend rows - the D3 parity guard.
5. `color_sets`: rows within one colour are node-disjoint; −1 padding is ignored.
6. Translation exactness: translating all masters moves every hanging node by the same translation.

GPU, in the Blender suite (`tests/blender/test_adaptive.py`):

7. Kernel-vs-oracle parity: a small random adaptive cage, blend-only substep (mu = lam = 0) against `solver_ref.blend_project`, matching to float tolerance, mirroring `test_solve_vs_oracle`.
8. Rigid free-fall: an adaptive cage dropped with zero stiffness falls rigidly; hanging nodes track their masters to 1e-4 - a glue bug shows as drift.
9. Cantilever beam with a thin end, adaptive vs a uniform-at-min reference: tip deflection within 15% at ≥4x fewer nodes.
10. Operator: tetrahedralize a thick-plus-thin object with Adaptive on - the cage has blend rows, the thin region ≥2 cells across, three live frames stay finite, skin readback finite.
11. Adaptive off: the existing suite passes unchanged - the D7 regression guard.

## Ceilings

- **Python refinement loop.** BVH distance and inside calls are per-cell Python, like today's mask; adaptive visits more cells because the boundary layer sits at min. A character at 0.25/0.03 should stay a one-time cost of seconds; if not, batch the distance queries. Upgrade path, not a blocker.
- **Stiffness seam at level boundaries.** Volume-scaled compliance (D6) normalizes most of it; a visible seam would need per-level mu scaling.
- **Blend constraints do not tear.** A level boundary can never fail; tearing propagates through real tets only. Acceptable v1 behaviour.
- **The boundary layer is always at min.** Cells touching the surface refine to Min Size even on the bulk, because that is what resolves thin features; the cost is area/min² and the user trades it against Min Size. Curvature-gated refinement could trim the layer later; not in v1.
- **Hanging chains deeper than two levels** resolve through the level-sorted pass order; one Gauss-Seidel sweep per substep leaves the finest hanging nodes a sweep behind, absorbed by substeps exactly like the elastic colours.
